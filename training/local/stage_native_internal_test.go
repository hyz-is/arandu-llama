//go:build libtorch && cgo

package local

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/tayi-ai/arandu-llama/checkpoint"
	"github.com/tayi-ai/arandu-llama/training/decoder"
	"github.com/tayi-ai/arandu-llama/training/fusioncache"
	"github.com/tayi-ai/arandu-llama/training/layers"
	"github.com/tayi-ai/arandu-llama/training/optim"
	"github.com/tayi-ai/arandu-llama/training/pipeline"
	"github.com/tayi-ai/arandu-llama/training/sequence"
	"github.com/tayi-ai/arandu-llama/training/torch"
)

func sftCPUModel(t *testing.T) (*decoder.LoadedTextModel, func() error) {
	t.Helper()
	m := &decoder.LoadedTextModel{Model: &decoder.TextModel{Epsilon: 1e-6, Layers: make([]decoder.Layer, 1)}}
	var base []*torch.Tensor
	tensor := func(shape []int64, norm, leaf bool) *torch.Tensor {
		n := 1
		for _, d := range shape {
			n *= int(d)
		}
		v := make([]float32, n)
		for i := range v {
			v[i] = float32(.2 * math.Sin(float64(i+1)*.71))
			if norm {
				v[i] = 1
			}
		}
		x, e := torch.FromFloat32(v, shape, torch.CPUDevice(), leaf)
		if e != nil {
			t.Fatal(e)
		}
		if !leaf {
			base = append(base, x)
		}
		return x
	}
	m.Model.Embedding = tensor([]int64{6, 4}, false, false)
	m.Model.Head = tensor([]int64{6, 4}, false, false)
	m.Model.FinalNorm = tensor([]int64{4}, true, false)
	w := layers.DecoderWeights{InputNorm: tensor([]int64{4}, true, false), PostAttentionNorm: tensor([]int64{4}, true, false), Gate: tensor([]int64{5, 4}, false, false), Up: tensor([]int64{5, 4}, false, false), Down: tensor([]int64{4, 5}, false, false)}
	w.Full = &layers.AttentionWeights{Query: tensor([]int64{8, 4}, false, false), Key: tensor([]int64{2, 4}, false, false), Value: tensor([]int64{2, 4}, false, false), Output: tensor([]int64{4, 4}, false, false), QueryNorm: tensor([]int64{2}, true, false), KeyNorm: tensor([]int64{2}, true, false)}
	prefix := "base_model.model.model.language_model.layers.0.self_attn."
	for i, name := range []string{"q_proj.lora_A.default.weight", "q_proj.lora_B.default.weight", "v_proj.lora_A.default.weight", "v_proj.lora_B.default.weight"} {
		shape := [][]int64{{1, 4}, {8, 1}, {1, 4}, {2, 1}}[i]
		m.Parameters = append(m.Parameters, decoder.InitialParameter{Name: prefix + name, Value: tensor(shape, false, true)})
		m.Summary.AdapterElements += shape[0] * shape[1]
	}
	a := &layers.AttentionLoRA{QueryA: m.Parameters[0].Value, QueryB: m.Parameters[1].Value, ValueA: m.Parameters[2].Value, ValueB: m.Parameters[3].Value, Alpha: 2}
	m.Model.Layers[0] = decoder.Layer{Device: torch.CPUDevice(), Weights: w, Adapter: a, Config: layers.DecoderConfig{Epsilon: 1e-6, MaxInputElements: 12, Full: layers.AttentionConfig{Heads: 2, KVHeads: 1, HeadDimension: 2, RotaryDimension: 2, Epsilon: 1e-6, MaxScoreElements: 18}}}
	// ReplaceParameters transfers ownership of fixture leaves into the real owner.
	var flat []float32
	for _, p := range m.Parameters {
		v, e := p.Value.Float32Values()
		if e != nil {
			t.Fatal(e)
		}
		flat = append(flat, v...)
	}
	if _, e := m.ReplaceParameters(context.Background(), flat); e != nil {
		t.Fatal(e)
	}
	close := func() error {
		e := m.Close()
		for _, b := range base {
			e = errors.Join(e, b.Close())
		}
		return e
	}
	t.Cleanup(func() { _ = close() })
	return m, close
}
func sftState(t *testing.T, m *decoder.LoadedTextModel) optim.AdamWState {
	t.Helper()
	s := optim.AdamWState{Step: 1}
	for _, p := range m.Parameters {
		v, e := p.Value.Float32Values()
		if e != nil {
			t.Fatal(e)
		}
		s.Parameters = append(s.Parameters, v...)
	}
	s.First = make([]float32, len(s.Parameters))
	s.Second = make([]float32, len(s.Parameters))
	return s
}
func sftTensorRecords(t *testing.T, m *decoder.LoadedTextModel, s optim.AdamWState) ([]checkpoint.Float32Tensor, []checkpoint.Float32Tensor) {
	t.Helper()
	var a, b []checkpoint.Float32Tensor
	offset := 0
	for _, p := range m.Parameters {
		i, e := p.Value.Info()
		if e != nil {
			t.Fatal(e)
		}
		n := int(i.Elements)
		shape := []uint64{uint64(i.Shape[0]), uint64(i.Shape[1])}
		a = append(a, checkpoint.Float32Tensor{Name: p.Name, Shape: shape, Values: s.Parameters[offset : offset+n]})
		b = append(b, checkpoint.Float32Tensor{Name: "m." + p.Name, Shape: shape, Values: s.First[offset : offset+n]}, checkpoint.Float32Tensor{Name: "v." + p.Name, Shape: shape, Values: s.Second[offset : offset+n]})
		offset += n
	}
	return a, b
}
func sftLogical(a []checkpoint.Float32Tensor) string {
	h := sha256.New()
	var b [4]byte
	for _, p := range a {
		h.Write([]byte(p.Name))
		for _, v := range p.Values {
			binary.LittleEndian.PutUint32(b[:], math.Float32bits(v))
			h.Write(b[:])
		}
	}
	return hex.EncodeToString(h.Sum(nil))
}
func sftPin(t *testing.T, c *StageConfig) {
	t.Helper()
	var e error
	c.Protocol.RecipeSHA256, e = c.Recipe.Digest()
	if e != nil {
		t.Fatal(e)
	}
	c.Protocol.PlacementSHA256 = sessionDigest(c.Placement)
	c.ProtocolSHA256, e = c.Protocol.Digest()
	if e != nil {
		t.Fatal(e)
	}
}
func sftFixture(t *testing.T) (StageConfig, pipeline.StageContext, []example) {
	t.Helper()
	source := t.TempDir()
	artifacts := t.TempDir()
	rows := []example{{"first", []int64{1, 2}, []int64{-100, 2}, 1}, {"second", []int64{1, 2}, []int64{-100, 2}, 1}, {"third", []int64{1, 2}, []int64{-100, 2}, 1}, {"fourth", []int64{1, 2}, []int64{-100, 2}, 1}}
	var data []byte
	for _, row := range rows {
		b, _ := json.Marshal(row)
		data = append(data, append(b, '\n')...)
	}
	if e := os.WriteFile(filepath.Join(source, "tokens.jsonl"), data, 0600); e != nil {
		t.Fatal(e)
	}
	ref := func(id string) pipeline.ArtifactRef { return pipeline.ArtifactRef{ID: id, SHA256: fmtHash([]byte(id))} }
	r := pipeline.Recipe{Version: 2, Scope: "master", ID: "fixture", Method: "generational-fusion-v1", Student: ref("student"), Teachers: []pipeline.ArtifactRef{ref("teacher")}, Training: ref("training"), Recovery: ref("recovery"), Protection: ref("protection"), Heldout: ref("heldout")}
	for i, phase := range []pipeline.Phase{pipeline.PhaseSFT, pipeline.PhaseTeacherCache, pipeline.PhaseAlignment, pipeline.PhaseFusion, pipeline.PhaseRecovery, pipeline.PhaseMaster} {
		input := r.Training
		if phase == pipeline.PhaseRecovery {
			input = r.Recovery
		}
		if phase == pipeline.PhaseMaster {
			input = r.Heldout
		}
		s := pipeline.Stage{ID: fmt.Sprintf("stage-%d", i), Phase: phase, Inputs: []pipeline.ArtifactRef{input}, MaxSteps: 1, MaxTokens: 3, TimeoutSeconds: 60}
		if i > 0 {
			s.ParentStage = fmt.Sprintf("stage-%d", i-1)
		}
		if phase == pipeline.PhaseFusion || phase == pipeline.PhaseRecovery {
			s.Inputs = append(s.Inputs, r.Protection)
		}
		r.Stages = append(r.Stages, s)
	}
	r.Stages[0].MaxSteps = 3
	placement := pipeline.Placement{Backend: "fixture-mps-qualified", MemoryBytes: 64 << 20, MaxTokens: 3, RuntimeSHA256: ref("runtime").SHA256, QualificationSHA256: ref("qualification").SHA256}
	loaded, close := sftCPUModel(t)
	defer close()
	state := sftState(t, loaded)
	a, moments := sftTensorRecords(t, loaded, state)
	hash := sftLogical(a)
	pin := strings.Repeat("a", 64)
	prefix := "base_model.model.model.language_model.layers.0.self_attn."
	var one [4]byte
	binary.LittleEndian.PutUint32(one[:], math.Float32bits(1))
	recipe := Recipe{Method: "causal-sft-v1", BaseRevision: strings.Repeat("a", 40), DataSHA256: fmtHash(data), ExampleCount: 4, Identity: decoder.AssemblyIdentity{IndexSHA256: pin, ConfigSHA256: pin, ReferenceSHA256: pin, InitialAdapterSHA256: hash}, Initializer: decoder.InitialAdapterSpec{ExpectedSHA256: hash, Seed: 7, Projections: []decoder.InitialProjection{{Name: prefix + "q_proj", Input: 4, Output: 8, Rank: 1}, {Name: prefix + "v_proj", Input: 4, Output: 2, Rank: 1}}}, Rotary: decoder.RotarySpec{Theta: 10000, Dimension: 2, MaxTokens: 3, ExpectedSHA256: fmtHash(one[:])}, Optimizer: optim.AdamWConfig{LearningRate: .01, Beta1: .9, Beta2: .99, Epsilon: 1e-8, MaxGradientNorm: 1}, LossScale: 1, MaxMPSBytes: 4096, MaxCheckpointBytes: 1 << 20, Assembly: decoder.AssemblyLimits{TensorCopyBytes: 4096, Sequence: sequence.SequenceLimits{MaxTokens: 3}}}
	tables, e := decoder.TextRotary(context.Background(), 2, torch.CPUDevice(), recipe.Rotary)
	if e != nil {
		t.Fatal(e)
	}
	loaded.Model.Layers[0].Cosine = tables.Cosine
	loaded.Model.Layers[0].Sine = tables.Sine
	logits, e := forwardFingerprint(context.Background(), loaded.Model, rows[0].InputIDs, 1, 1<<20)
	if e != nil {
		t.Fatal(e)
	}
	tables.Close()
	loaded.Model.Layers[0].Cosine = nil
	loaded.Model.Layers[0].Sine = nil
	ar, e := writeAndReadback(context.Background(), filepath.Join(source, "adapter_model.safetensors"), a)
	if e != nil {
		t.Fatal(e)
	}
	mr, e := writeAndReadback(context.Background(), filepath.Join(source, "optimizer_moments.safetensors"), moments)
	if e != nil {
		t.Fatal(e)
	}
	manifest := stepManifest{RecipeSHA256: recipe.Digest(), Step: 1, ExampleID: "first", SupervisedTokens: 1, LossBefore: 2, UpdatedAdapterSHA: hash, AdapterFileSHA: ar.SHA256, OptimizerFileSHA: mr.SHA256, LogitsAfterSHA: logits, BaseRevision: recipe.BaseRevision}
	mb, _ := json.Marshal(manifest)
	if e := os.WriteFile(filepath.Join(source, "manifest.json"), mb, 0600); e != nil {
		t.Fatal(e)
	}
	student := fusioncache.ModelIdentity{Name: "student", Revision: recipe.BaseRevision, WeightsSHA256: r.Student.SHA256, TokenizerSHA256: pin, TemplateSHA256: pin, RuntimeSHA256: pin, Vocabulary: 6}
	initial := StageInitial{Step: 1, Manifest: pipeline.StageArtifact{Path: "manifest.json", SHA256: fmtHash(mb), Bytes: int64(len(mb))}, Adapter: pipeline.StageArtifact{Path: "adapter_model.safetensors", SHA256: ar.SHA256, Bytes: ar.Bytes}, Optimizer: pipeline.StageArtifact{Path: "optimizer_moments.safetensors", SHA256: mr.SHA256, Bytes: mr.Bytes}}
	protocol := StageProtocol{Version: 1, QualificationSHA256: placement.QualificationSHA256, StageID: "stage-0", Student: student, Local: recipe, Data: pipeline.ArtifactBinding{Kind: pipeline.BindingDerivedInput, Input: r.Training, Artifact: pipeline.StageArtifact{Path: "tokens.jsonl", SHA256: fmtHash(data), Bytes: int64(len(data))}}, Initial: initial, TargetStep: 4, DeliverySteps: 2,
		Limits: StageLimits{MaxProtocolBytes: 1 << 20, MaxDataBytes: 1 << 16, MaxMetadataBytes: 1 << 16, MaxManifestBytes: 1 << 16, MaxTensorFileBytes: 1 << 20, MaxTotalBytes: 10 << 20, MaxDataTokens: 256, MaxParameters: 100, WorkingBytes: 8 << 20, MaxSteps: 3, Checkpoint: checkpoint.Limits{MaxHeaderBytes: 4096, MaxTensors: 16, MaxDimensions: 2, MaxMetadataEntries: 1, MaxChunkBytes: 64}}}
	c := StageConfig{Recipe: r, Placement: placement, Protocol: protocol, BundleDirectory: source, ModelDirectory: source, DataDirectory: source, InitialDirectory: source}
	sftPin(t, &c)
	ctx := pipeline.StageContext{Execution: pipeline.Execution{RunID: "run", TenantID: "tenant", Generation: 1, Recipe: r, TargetSHA256: ref("execution-target").SHA256, Placement: placement}, Stage: r.Stages[0], ArtifactDirectory: artifacts}
	return c, ctx, rows
}

// cpuModel builds a fresh CPU model owner and its release, one per delivery.
type cpuModel func(*testing.T) (*decoder.LoadedTextModel, func() error)

func sftExecutor(t *testing.T, calls *int, closed *int) stageDelivery {
	return sftExecutorWith(t, sftCPUModel, calls, closed)
}
func sftExecutorWith(t *testing.T, model cpuModel, calls *int, closed *int) stageDelivery {
	return func(ctx context.Context, c Config, hooks *stepHooks) (err error) {
		*calls++
		m, close := model(t)
		defer func() { err = errors.Join(err, close()); *closed++ }()
		return runCurriculumWithHooks(ctx, m, c.DataPath, c.InitialCheckpoint, c.CheckpointRoot, c.MaxTokens, c.MaxSteps, c, hooks, torch.CPUDevice())
	}
}
func sftCommit(t *testing.T, h *Stage, c *pipeline.StageContext, seen *[]pipeline.StageResult) func(context.Context, pipeline.StageResult) error {
	return func(ctx context.Context, result pipeline.StageResult) error {
		identity, e := h.identity(ctx, *c)
		if e != nil {
			t.Fatal(e)
		}
		r := pipeline.StageReceipt{Version: 1, Identity: identity, StageID: c.Stage.ID, Result: result}
		r.SHA256 = sessionDigest(r)
		if e := h.Verify(ctx, *c, r); e != nil {
			t.Fatal(e)
		}
		c.Previous = &r
		*seen = append(*seen, result)
		return nil
	}
}

func TestNativeSFTStagePreservesAdamWAndThreeActualUpdates(t *testing.T) {
	c, stage, rows := sftFixture(t)
	calls, closed := 0, 0
	h, e := newStage(c, sftExecutor(t, &calls, &closed))
	if e != nil {
		t.Fatal(e)
	}
	var results []pipeline.StageResult
	if e := h.Run(context.Background(), stage, sftCommit(t, h, &stage, &results)); e != nil {
		t.Fatal(e)
	}
	if calls != 2 || closed != calls || len(results) != 3 || results[0].Steps != 1 || results[2].Steps != 3 || !results[2].Complete {
		t.Fatal("delivery or new-step accounting differs", calls, closed, results)
	}
	// Independent sequential CompletionGradient + AdamW state proves that delivery
	// boundaries restored moments rather than resetting the optimizer.
	model, close := sftCPUModel(t)
	defer close()
	state := sftState(t, model)
	var losses []float64
	for _, row := range rows[1:] {
		tables, e := decoder.TextRotary(context.Background(), len(row.InputIDs), torch.CPUDevice(), c.Protocol.Local.Rotary)
		if e != nil {
			t.Fatal(e)
		}
		model.Model.Layers[0].Cosine = tables.Cosine
		model.Model.Layers[0].Sine = tables.Sine
		g, e := decoder.CompletionGradient(context.Background(), model.Model, row.InputIDs, row.PromptTokens, decoder.Limits{MaxTokens: int64(len(row.InputIDs)), LogitRows: int64(len(row.InputIDs) - row.PromptTokens + 1), MaxCheckpointBytes: c.Protocol.Local.MaxCheckpointBytes}, c.Protocol.Local.LossScale)
		tables.Close()
		model.Model.Layers[0].Cosine = nil
		model.Model.Layers[0].Sine = nil
		if e != nil {
			t.Fatal(e)
		}
		var flat []float32
		for _, p := range g.Gradients {
			flat = append(flat, p.ValuesF32...)
		}
		state, _, e = optim.UpdateAdamW(state, flat, c.Protocol.Local.Optimizer)
		if e != nil {
			t.Fatal(e)
		}
		if _, e = model.ReplaceParameters(context.Background(), state.Parameters); e != nil {
			t.Fatal(e)
		}
		losses = append(losses, g.Loss)
	}
	root, e := stageRoot(stage.ArtifactDirectory)
	if e != nil {
		t.Fatal(e)
	}
	defer root.Close()
	var recorded []float64
	for i, result := range results {
		body, e := stageRead(context.Background(), root, result.Artifacts[4].Path, c.Protocol.Limits.MaxManifestBytes, result.Artifacts[4].SHA256)
		if e != nil {
			t.Fatal(e)
		}
		var m stepManifest
		json.Unmarshal(body, &m)
		recorded = append(recorded, m.LossBefore)
		if m.Step != uint64(i+2) {
			t.Fatal("absolute cursor differs")
		}
	}
	if !reflect.DeepEqual(recorded, losses) || losses[0] == losses[1] || losses[1] == losses[2] {
		t.Fatal("state-dependent losses differ", recorded, losses)
	}
	expected, moments := sftTensorRecords(t, model, state)
	manifest, _, e := h.checkpoint(context.Background(), root, h.stepName(4), 4, rows)
	if e != nil || manifest.UpdatedAdapterSHA != sftLogical(expected) {
		t.Fatal("final adapter differs from independent updates", e)
	}
	names := []string{}
	shapes := [][]int64{}
	for _, p := range moments {
		names = append(names, p.Name)
		shapes = append(shapes, []int64{int64(p.Shape[0]), int64(p.Shape[1])})
	}
	actual, e := readFrozenTensorFile(context.Background(), filepath.Join(stage.ArtifactDirectory, h.stepName(4), "optimizer_moments.safetensors"), manifest.OptimizerFileSHA, names, shapes)
	if e != nil {
		t.Fatal(e)
	}
	for i, p := range moments {
		if !reflect.DeepEqual(actual[i], p.Values) {
			t.Fatal("AdamW moments changed across delivery boundary")
		}
	}
}

func TestNativeSFTStageRecoversLostCallbackWithoutRepeatingStep(t *testing.T) {
	c, stage, _ := sftFixture(t)
	calls, closed := 0, 0
	h, e := newStage(c, sftExecutor(t, &calls, &closed))
	if e != nil {
		t.Fatal(e)
	}
	lost := errors.New("lost app receipt")
	if e := h.Run(context.Background(), stage, func(context.Context, pipeline.StageResult) error { return lost }); !errors.Is(e, lost) {
		t.Fatal(e)
	}
	if calls != 1 || closed != 1 {
		t.Fatal("callback failure retained model")
	}
	next, e := newStage(c, sftExecutor(t, &calls, &closed))
	if e != nil {
		t.Fatal(e)
	}
	stage.Execution.Generation++
	var results []pipeline.StageResult
	if e := next.Reconcile(context.Background(), stage, sftCommit(t, next, &stage, &results)); e != nil {
		t.Fatal(e)
	}
	if calls != 1 || len(results) != 1 || results[0].Steps != 1 {
		t.Fatal("reconciliation recalculated a step")
	}
	if e := next.Run(context.Background(), stage, sftCommit(t, next, &stage, &results)); e != nil {
		t.Fatal(e)
	}
	if len(results) != 3 || !results[2].Complete || calls != 2 || closed != calls {
		t.Fatal("resume did not preserve remaining interval")
	}
}

func TestNativeSFTStageUncertainIntentNeverReplays(t *testing.T) {
	c, stage, _ := sftFixture(t)
	calls := 0
	failure := errors.New("interrupted after intent")
	h, e := newStage(c, func(ctx context.Context, c Config, hooks *stepHooks) error {
		calls++
		if e := hooks.before(ctx, 2, "second"); e != nil {
			return e
		}
		return failure
	})
	if e != nil {
		t.Fatal(e)
	}
	if e := h.Run(context.Background(), stage, func(context.Context, pipeline.StageResult) error { t.Fatal("uncertain step committed"); return nil }); !errors.Is(e, failure) {
		t.Fatal(e)
	}
	next, e := newStage(c, func(context.Context, Config, *stepHooks) error { calls++; return nil })
	if e != nil {
		t.Fatal(e)
	}
	if e := next.Run(context.Background(), stage, func(context.Context, pipeline.StageResult) error { return nil }); !errors.Is(e, ErrAttempt) || calls != 1 {
		t.Fatal("uncertain attempt repeated", e, calls)
	}
}

func TestNativeSFTStageAdmissionOwnsSnapshotAndRefusesBounds(t *testing.T) {
	c, stage, _ := sftFixture(t)
	h, e := NewStage(c)
	if e != nil {
		t.Fatal(e)
	}
	before := sessionDigest(h.config)
	c.Protocol.Local.Initializer.Projections[0].Input++
	c.Recipe.Stages[0].Inputs[0].ID = "changed"
	if sessionDigest(h.config) != before {
		t.Fatal("handler shares caller-owned slices")
	}
	if e := h.Admit(context.Background(), h.config.Recipe, h.stage, h.config.Placement); e != nil {
		t.Fatal(e)
	}
	_ = stage
	for _, name := range []string{"copy-memory", "working", "data-budget", "zero-initial", "target", "tensor-budget", "metadata"} {
		t.Run(name, func(t *testing.T) {
			c, _, _ := sftFixture(t)
			switch name {
			case "copy-memory":
				c.Protocol.Local.Assembly.TensorCopyBytes = c.Placement.MemoryBytes
			case "working":
				c.Protocol.Limits.WorkingBytes = 1
			case "data-budget":
				c.Protocol.Limits.MaxDataBytes = 1
			case "zero-initial":
				c.Protocol.Initial.Step = 0
			case "target":
				c.Protocol.TargetStep = 5
			case "tensor-budget":
				c.Protocol.Limits.MaxTensorFileBytes = 1
			case "metadata":
				c.Protocol.Limits.MaxMetadataBytes = 0
			}
			if pin, e := c.Protocol.Digest(); e == nil {
				c.ProtocolSHA256 = pin
				if _, e = NewStage(c); e == nil {
					t.Fatal("invalid contract admitted")
				}
			}
		})
	}
}

func TestNativeSFTStageRefusesCorruptSourcesBeforeExecution(t *testing.T) {
	for _, name := range []string{"data", "adapter", "moments", "symlink", "mask"} {
		t.Run(name, func(t *testing.T) {
			c, stage, _ := sftFixture(t)
			path := filepath.Join(c.InitialDirectory, "adapter_model.safetensors")
			if name == "data" || name == "mask" {
				path = filepath.Join(c.DataDirectory, "tokens.jsonl")
			}
			if name == "moments" {
				path = filepath.Join(c.InitialDirectory, "optimizer_moments.safetensors")
			}
			if name == "symlink" {
				target := path + ".original"
				os.Rename(path, target)
				os.Symlink(target, path)
			} else if name == "mask" {
				b, _ := os.ReadFile(path)
				b = []byte(strings.ReplaceAll(string(b), "[-100,2]", "[1,2]"))
				os.WriteFile(path, b, 0600)
				c.Protocol.Local.DataSHA256 = fmtHash(b)
				c.Protocol.Data.Artifact.SHA256 = fmtHash(b)
				c.Protocol.Data.Artifact.Bytes = int64(len(b))
				sftPin(t, &c)
			} else {
				os.WriteFile(path, []byte("corrupt"), 0600)
			}
			calls := 0
			h, e := newStage(c, func(context.Context, Config, *stepHooks) error { calls++; return nil })
			if e != nil {
				t.Fatal(e)
			}
			if e := h.Run(context.Background(), stage, func(context.Context, pipeline.StageResult) error { return nil }); e == nil || calls != 0 {
				t.Fatal("invalid source reached calculation", e, calls)
			}
		})
	}
}

func TestNativeSFTStageCancellationAndMetadataBoundBeforeNativeLoad(t *testing.T) {
	c, stage, _ := sftFixture(t)
	calls := 0
	h, e := newStage(c, func(context.Context, Config, *stepHooks) error { calls++; return nil })
	if e != nil {
		t.Fatal(e)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if e := h.Run(ctx, stage, func(context.Context, pipeline.StageResult) error { return nil }); !errors.Is(e, context.Canceled) || calls != 0 {
		t.Fatal("cancelled work was admitted", e)
	}
	entries, e := os.ReadDir(stage.ArtifactDirectory)
	if e != nil || len(entries) != 0 {
		t.Fatal("cancelled invocation wrote artifacts", e)
	}
	// The production entry reads this bounded metadata before constructing a plan
	// or allocating MPS. This deliberately invalid bundle cannot reach either.
	b := make([]byte, c.Protocol.Limits.MaxMetadataBytes+1)
	if e := os.WriteFile(filepath.Join(c.BundleDirectory, "model.safetensors.index.json"), b, 0600); e != nil {
		t.Fatal(e)
	}
	h, e = NewStage(c)
	if e != nil {
		t.Fatal(e)
	}
	if e := h.Run(context.Background(), stage, func(context.Context, pipeline.StageResult) error { return nil }); !errors.Is(e, ErrStage) {
		t.Fatal("metadata bound was not enforced before hash/plan", e)
	}
}
