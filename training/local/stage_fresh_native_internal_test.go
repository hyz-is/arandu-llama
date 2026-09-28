//go:build libtorch && cgo

package local

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/tayi-ai/arandu-llama/checkpoint"
	testfixture "github.com/tayi-ai/arandu-llama/tests/Unit/Training/Fixture"
	"github.com/tayi-ai/arandu-llama/training/decoder"
	"github.com/tayi-ai/arandu-llama/training/optim"
	"github.com/tayi-ai/arandu-llama/training/pipeline"
	"github.com/tayi-ai/arandu-llama/training/torch"
)

// sftFreshFixture is the continuation fixture turned into a start from zero:
// no initial checkpoint, no source directory, and an initializer whose pin is
// computed by the independent CPU reference rather than by the code under test.
func sftFreshFixture(t *testing.T) (StageConfig, pipeline.StageContext, []example, decoder.InitialAdapterSpec) {
	t.Helper()
	c, stage, rows := sftFixture(t)
	spec := c.Protocol.Local.Initializer
	spec.ExpectedSHA256 = testfixture.Digest(t, spec)
	c.Protocol.Local.Initializer = spec
	c.Protocol.Local.Identity.InitialAdapterSHA256 = spec.ExpectedSHA256
	c.Protocol.Initial = StageInitial{}
	// Two deliveries of two and one steps: the second reloads a checkpoint the
	// first wrote, so the test crosses a real restart and not only step one.
	c.Protocol.TargetStep = 3
	c.InitialDirectory = ""
	sftPin(t, &c)
	return c, stage, rows, spec
}

// freshTrajectory is the same interval computed without the stage: the pinned
// initializer, AdamW at step zero with zero moments, and one CompletionGradient
// and update per example. Nothing here is shared with the stage beyond the
// primitives both must use.
func freshTrajectory(t *testing.T, c StageConfig, rows []example, spec decoder.InitialAdapterSpec, steps int) ([]float64, []checkpoint.Float32Tensor, []checkpoint.Float32Tensor) {
	return freshTrajectoryWith(t, sftCPUModel, c, rows, spec, steps)
}
func freshTrajectoryWith(t *testing.T, cpu cpuModel, c StageConfig, rows []example, spec decoder.InitialAdapterSpec, steps int) ([]float64, []checkpoint.Float32Tensor, []checkpoint.Float32Tensor) {
	t.Helper()
	ctx := context.Background()
	model, close := cpu(t)
	defer close()
	initial, e := decoder.InitializeAdapter(ctx, spec)
	if e != nil {
		t.Fatal(e)
	}
	values := map[string][]float32{}
	for _, p := range initial.Parameters {
		v, e := p.Value.Float32Values()
		if e != nil {
			t.Fatal(e)
		}
		values[p.Name] = v
	}
	if e := initial.Close(); e != nil {
		t.Fatal(e)
	}
	var flat []float32
	for _, p := range model.Parameters {
		flat = append(flat, values[p.Name]...)
	}
	digest, e := model.ReplaceParameters(ctx, flat)
	if e != nil || digest != spec.ExpectedSHA256 {
		t.Fatal("independent initializer does not reach its own pin", digest, e)
	}
	state := optim.AdamWState{Parameters: append([]float32(nil), flat...), First: make([]float32, len(flat)), Second: make([]float32, len(flat))}
	var losses []float64
	for _, row := range rows[:steps] {
		tables, e := decoder.TextRotary(ctx, len(row.InputIDs), torch.CPUDevice(), c.Protocol.Local.Rotary)
		if e != nil {
			t.Fatal(e)
		}
		rotary(model, tables.Cosine, tables.Sine)
		g, e := decoder.CompletionGradient(ctx, model.Model, row.InputIDs, row.PromptTokens, decoder.Limits{MaxTokens: int64(len(row.InputIDs)), LogitRows: int64(len(row.InputIDs) - row.PromptTokens + 1), MaxCheckpointBytes: c.Protocol.Local.MaxCheckpointBytes}, c.Protocol.Local.LossScale)
		tables.Close()
		rotary(model, nil, nil)
		if e != nil {
			t.Fatal(e)
		}
		var gradient []float32
		for _, p := range g.Gradients {
			gradient = append(gradient, p.ValuesF32...)
		}
		if state, _, e = optim.UpdateAdamW(state, gradient, c.Protocol.Local.Optimizer); e != nil {
			t.Fatal(e)
		}
		if _, e = model.ReplaceParameters(ctx, state.Parameters); e != nil {
			t.Fatal(e)
		}
		losses = append(losses, g.Loss)
	}
	adapter, moments := sftTensorRecords(t, model, state)
	return losses, adapter, moments
}

// sameAsTrajectory compares what the stage wrote against the independent run:
// the loss before every step, the absolute cursor, the final adapter and the
// final moments bit for bit.
func sameAsTrajectory(t *testing.T, h *Stage, c StageConfig, stage pipeline.StageContext, rows []example, results []pipeline.StageResult, spec decoder.InitialAdapterSpec) {
	sameAsTrajectoryWith(t, sftCPUModel, h, c, stage, rows, results, spec)
}
func sameAsTrajectoryWith(t *testing.T, cpu cpuModel, h *Stage, c StageConfig, stage pipeline.StageContext, rows []example, results []pipeline.StageResult, spec decoder.InitialAdapterSpec) {
	t.Helper()
	ctx := context.Background()
	steps := int(c.Protocol.TargetStep)
	losses, adapter, moments := freshTrajectoryWith(t, cpu, c, rows, spec, steps)
	root, e := stageRoot(stage.ArtifactDirectory)
	if e != nil {
		t.Fatal(e)
	}
	defer root.Close()
	recorded := make([]float64, steps)
	for step := 1; step <= steps; step++ {
		manifest, _, e := h.checkpoint(ctx, root, h.stepName(uint64(step)), uint64(step), rows)
		if e != nil {
			t.Fatal("step", step, e)
		}
		recorded[step-1] = manifest.LossBefore
		if step == 1 && manifest.UpdatedAdapterSHA == spec.ExpectedSHA256 {
			t.Fatal("the first step did not move away from the initializer")
		}
		if step == steps && manifest.UpdatedAdapterSHA != sftLogical(adapter) {
			t.Fatal("final adapter differs from the independent AdamW from zero")
		}
	}
	if !reflect.DeepEqual(recorded, losses) || losses[0] == losses[1] || losses[1] == losses[2] {
		t.Fatal("state-dependent losses differ from the independent trajectory", recorded, losses)
	}
	last := results[len(results)-1]
	if !last.Complete || last.Steps != int64(steps) {
		t.Fatal("the stage did not report its admitted interval complete", last.Steps, last.Complete)
	}
	// The manifest the result names is the step's own, found by position after
	// the seed artifacts; a hard-coded index would read another file here.
	body, e := stageRead(ctx, root, h.stepManifestArtifact(last).Path, c.Protocol.Limits.MaxManifestBytes, h.stepManifestArtifact(last).SHA256)
	if e != nil {
		t.Fatal(e)
	}
	var final stepManifest
	if e := json.Unmarshal(body, &final); e != nil || final.Step != uint64(steps) {
		t.Fatal("result names another manifest", final.Step, e)
	}
	names, shapes := []string{}, [][]int64{}
	for _, p := range moments {
		names = append(names, p.Name)
		shapes = append(shapes, []int64{int64(p.Shape[0]), int64(p.Shape[1])})
	}
	actual, e := readFrozenTensorFile(ctx, filepath.Join(stage.ArtifactDirectory, h.stepName(uint64(steps)), "optimizer_moments.safetensors"), final.OptimizerFileSHA, names, shapes)
	if e != nil {
		t.Fatal(e)
	}
	for i, p := range moments {
		if !reflect.DeepEqual(actual[i], p.Values) {
			t.Fatal("AdamW moments differ from a new optimizer's", p.Name)
		}
	}
}

func TestNativeSFTStageFreshStartMatchesAnIndependentAdamWFromZero(t *testing.T) {
	c, stage, rows, spec := sftFreshFixture(t)
	calls, closed := 0, 0
	h, e := newStage(c, sftExecutor(t, &calls, &closed))
	if e != nil {
		t.Fatal(e)
	}
	var results []pipeline.StageResult
	if e := h.Run(context.Background(), stage, sftCommit(t, h, &stage, &results)); e != nil {
		t.Fatal(e)
	}
	if calls != 2 || closed != calls || len(results) != 3 {
		t.Fatal("delivery or step accounting differs", calls, closed, len(results))
	}
	for i, r := range results {
		if r.Steps != int64(i+1) {
			t.Fatal("step count is not relative to a start at zero", i, r.Steps)
		}
	}
	// A fresh start imports its data and nothing else.
	entries, e := os.ReadDir(filepath.Join(stage.ArtifactDirectory, h.seedName()))
	if e != nil || len(entries) != 1 || entries[0].Name() != "data.jsonl" {
		t.Fatal("a fresh start imported an initial checkpoint", entries, e)
	}
	sameAsTrajectory(t, h, c, stage, rows, results, spec)
}

func TestNativeSFTStageFreshStartResumesAfterALostProcessWithoutRepeatingAStep(t *testing.T) {
	c, stage, rows, spec := sftFreshFixture(t)
	calls, closed := 0, 0
	delivery := sftExecutor(t, &calls, &closed)
	lost := errors.New("process lost after its first delivery")
	first, e := newStage(c, func(ctx context.Context, cfg Config, hooks *stepHooks) error {
		if e := delivery(ctx, cfg, hooks); e != nil {
			return e
		}
		return lost
	})
	if e != nil {
		t.Fatal(e)
	}
	var results []pipeline.StageResult
	if e := first.Run(context.Background(), stage, sftCommit(t, first, &stage, &results)); !errors.Is(e, lost) || len(results) != 2 {
		t.Fatal("first delivery did not commit its two steps before the loss", e, len(results))
	}
	next, e := newStage(c, sftExecutor(t, &calls, &closed))
	if e != nil {
		t.Fatal(e)
	}
	stage.Execution.Generation++
	if e := next.Run(context.Background(), stage, sftCommit(t, next, &stage, &results)); e != nil {
		t.Fatal(e)
	}
	// One delivery before the loss and one after: steps one and two were read
	// back, not recomputed, and the chain continued from the fresh start.
	if calls != 2 || closed != calls {
		t.Fatal("resume recomputed a committed step", calls, closed)
	}
	sameAsTrajectory(t, next, c, stage, rows, results, spec)
}

func TestNativeSFTStageFreshStartIsExplicitAndRefusesHalfDeclaredStarts(t *testing.T) {
	artifact := pipeline.StageArtifact{Path: "manifest.json", SHA256: fmtHash([]byte("m")), Bytes: 1}
	for name, mutate := range map[string]func(*StageInitial){
		"manifest beside step zero":   func(i *StageInitial) { i.Manifest = artifact },
		"adapter beside step zero":    func(i *StageInitial) { a := artifact; a.Path = "adapter_model.safetensors"; i.Adapter = a },
		"moments beside step zero":    func(i *StageInitial) { a := artifact; a.Path = "optimizer_moments.safetensors"; i.Optimizer = a },
		"historical beside step zero": func(i *StageInitial) { i.AllowHistorical = true },
	} {
		t.Run(name, func(t *testing.T) {
			c, _, _, _ := sftFreshFixture(t)
			mutate(&c.Protocol.Initial)
			if _, e := c.Protocol.Digest(); e == nil {
				t.Fatal("a continuation that forgot its step was admitted as a fresh start")
			}
		})
	}
	c, stage, _, _ := sftFreshFixture(t)
	c.InitialDirectory = stage.ArtifactDirectory
	if _, e := NewStage(c); e == nil {
		t.Fatal("a fresh start named a source to import from")
	}
	c, _, _, _ = sftFreshFixture(t)
	h, e := newStage(c, func(context.Context, Config, *stepHooks) error { return nil })
	if e != nil {
		t.Fatal(e)
	}
	fresh := h.localConfig("/admission", 0)
	if !fresh.FreshStart || fresh.InitialCheckpoint != "" {
		t.Fatal("a fresh stage did not ask its delivery to start at the initializer", fresh.FreshStart, fresh.InitialCheckpoint)
	}
	if _, e := fresh.Snapshot(); e != nil {
		t.Fatal(e)
	}
	for name, mutate := range map[string]func(*Config){
		"fresh with a checkpoint":       func(x *Config) { x.InitialCheckpoint = "/admission/elsewhere" },
		"fresh with history":            func(x *Config) { x.AdmittedCheckpoints = map[uint64]string{1: fmtHash([]byte("h"))} },
		"continuation without its path": func(x *Config) { x.FreshStart = false },
	} {
		x := fresh
		mutate(&x)
		if _, e := x.Snapshot(); e == nil {
			t.Fatal(name, "admitted")
		}
	}
	// The curriculum only stands before its first example when told to.
	data := filepath.Join(stage.ArtifactDirectory, "tokens.jsonl")
	source, e := os.ReadFile(filepath.Join(c.DataDirectory, c.Protocol.Data.Artifact.Path))
	if e != nil {
		t.Fatal(e)
	}
	if e := os.WriteFile(data, source, 0600); e != nil {
		t.Fatal(e)
	}
	continuation := fresh
	continuation.FreshStart = false
	if _, _, _, e := curriculumPosition(data, "", continuation); e == nil {
		t.Fatal("a continuation without its checkpoint became a fresh start")
	}
	if _, prior, next, e := curriculumPosition(data, "", fresh); e != nil || next != 0 || prior.Step != 0 {
		t.Fatal("a fresh start does not stand before the first example", next, prior.Step, e)
	}
}

// The execution target is the application's, not the student: the receipt
// carries whatever digest the application names, and the student stays bound
// through the recipe. A target equal to the student was the only shape the
// stage accepted before, and no application submits it.
func TestNativeSFTStageCarriesTheApplicationsTargetDigest(t *testing.T) {
	c, stage, _, _ := sftFreshFixture(t)
	h, e := newStage(c, func(context.Context, Config, *stepHooks) error { return nil })
	if e != nil {
		t.Fatal(e)
	}
	if stage.Execution.TargetSHA256 == c.Recipe.Student.SHA256 {
		t.Fatal("fixture no longer distinguishes the execution target from the student")
	}
	identity, e := h.identity(context.Background(), stage)
	if e != nil || identity.TargetSHA256 != stage.Execution.TargetSHA256 {
		t.Fatal("the application's target digest was refused or replaced", identity.TargetSHA256, e)
	}
	stage.Execution.TargetSHA256 = "not-a-digest"
	if _, e := h.identity(context.Background(), stage); e == nil {
		t.Fatal("a malformed target digest was admitted")
	}
}

// rotary sets or clears the tables of every full-attention layer.
func rotary(model *decoder.LoadedTextModel, cosine, sine *torch.Tensor) {
	for layer := range model.Model.Layers {
		if model.Model.Layers[layer].Weights.Full != nil {
			model.Model.Layers[layer].Cosine, model.Model.Layers[layer].Sine = cosine, sine
		}
	}
}
