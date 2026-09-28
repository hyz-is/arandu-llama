//go:build libtorch && cgo

package local

import (
	"bytes"
	"context"
	"errors"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/tayi-ai/arandu-llama/checkpoint"
	"github.com/tayi-ai/arandu-llama/training/decoder"
	"github.com/tayi-ai/arandu-llama/training/fusioncache"
	"github.com/tayi-ai/arandu-llama/training/optim"
	"github.com/tayi-ai/arandu-llama/training/pipeline"
	"github.com/tayi-ai/arandu-llama/training/torch"
)

// kdStageFixture is the fresh SFT stage with a distillation block: two
// teachers, one on the student's tokenizer file and one on a measured
// vocabulary, each with a cache over every curriculum row.
func kdStageFixture(t *testing.T, weights ...float64) (StageConfig, pipeline.StageContext, []example, decoder.InitialAdapterSpec) {
	t.Helper()
	c, stage, rows, spec := sftFreshFixture(t)
	d := kdDistillation(t, c.Protocol.Student, c.Recipe.Training.SHA256, weights...)
	c.CacheDirectory = t.TempDir()
	kdPin(t, c.CacheDirectory, d, rows)
	c.Protocol.Local.Distillation = d
	sftPin(t, &c)
	return c, stage, rows, spec
}

// kdRun runs a stage to its target with the CPU delivery and returns every
// step's manifest and its raw bytes.
func kdRun(t *testing.T, c StageConfig, stage pipeline.StageContext, rows []example) (*Stage, []pipeline.StageResult, []stepManifest, [][]byte) {
	t.Helper()
	calls, closed := 0, 0
	h, e := newStage(c, sftExecutor(t, &calls, &closed))
	if e != nil {
		t.Fatal(e)
	}
	var results []pipeline.StageResult
	if e := h.Run(context.Background(), stage, sftCommit(t, h, &stage, &results)); e != nil {
		t.Fatal(e)
	}
	if calls != 2 || closed != calls || len(results) != int(c.Protocol.TargetStep) {
		t.Fatal("delivery or step accounting differs", calls, closed, len(results))
	}
	root, e := stageRoot(stage.ArtifactDirectory)
	if e != nil {
		t.Fatal(e)
	}
	defer root.Close()
	var manifests []stepManifest
	var bodies [][]byte
	for step := uint64(1); step <= c.Protocol.TargetStep; step++ {
		m, artifacts, e := h.checkpoint(context.Background(), root, h.stepName(step), step, rows)
		if e != nil {
			t.Fatal("step", step, e)
		}
		body, e := stageRead(context.Background(), root, artifacts[0].Path, c.Protocol.Limits.MaxManifestBytes, artifacts[0].SHA256)
		if e != nil {
			t.Fatal(e)
		}
		manifests, bodies = append(manifests, m), append(bodies, body)
	}
	return h, results, manifests, bodies
}

func TestNativeDistillationAtAlphaZeroIsTheSFTStepBitForBit(t *testing.T) {
	plain, plainStage, rows, _ := sftFreshFixture(t)
	_, _, sft, sftBodies := kdRun(t, plain, plainStage, rows)
	distilled, distilledStage, _, _ := kdStageFixture(t, 0, 0)
	h, results, kd, kdBodies := kdRun(t, distilled, distilledStage, rows)
	if distilled.Protocol.Local.Digest() == plain.Protocol.Local.Digest() {
		t.Fatal("a declared distillation left the recipe digest unchanged")
	}
	for i := range sft {
		a, b := sft[i], kd[i]
		if a.LossBefore != b.LossBefore || a.UpdatedAdapterSHA != b.UpdatedAdapterSHA || a.AdapterFileSHA != b.AdapterFileSHA || a.OptimizerFileSHA != b.OptimizerFileSHA || a.LogitsAfterSHA != b.LogitsAfterSHA || a.SupervisedTokens != b.SupervisedTokens {
			t.Fatalf("step %d at alpha zero differs from SFT:\n%+v\n%+v", i+1, a, b)
		}
		if a.RecipeSHA256 == b.RecipeSHA256 || b.RecipeSHA256 != distilled.Protocol.Local.Digest() {
			t.Fatal("the step is not bound to its own recipe")
		}
		// Neither manifest carries a distillation reading: SFT writes the bytes
		// it always wrote, and alpha zero ran the SFT calculation.
		for _, body := range [][]byte{sftBodies[i], kdBodies[i]} {
			for _, key := range []string{"distillation_loss_before", "teacher_mass", "teacher_losses"} {
				if bytes.Contains(body, []byte(key)) {
					t.Fatal("an SFT calculation wrote a distillation field", key)
				}
			}
		}
	}
	// The imported caches are named in every result, after the data.
	seed := h.seedArtifacts()
	if len(seed) != 3 || seed[1].SHA256 != distilled.Protocol.Local.Distillation.Teachers[0].CacheSHA256 || seed[2].SHA256 != distilled.Protocol.Local.Distillation.Teachers[1].CacheSHA256 {
		t.Fatal("the caches are not seed artifacts", seed)
	}
	for _, r := range results {
		if !reflect.DeepEqual(r.Artifacts[:len(seed)], seed) {
			t.Fatal("a result does not name the imported caches")
		}
	}
}

// kdTrajectory is the distillation interval computed without the stage or its
// cache reader: the pinned initializer, AdamW from zero, and one
// FusionCompletionGradient per row over teachers built from kdTopK. At every
// step it also reads CompletionGradient's loss at the same parameters.
func kdTrajectory(t *testing.T, c StageConfig, rows []example, spec decoder.InitialAdapterSpec) (hard, blended, sft []float64, adapter, moments []checkpoint.Float32Tensor) {
	t.Helper()
	ctx := context.Background()
	model, close := sftCPUModel(t)
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
	if _, e := model.ReplaceParameters(ctx, flat); e != nil {
		t.Fatal(e)
	}
	d := c.Protocol.Local.Distillation
	state := optim.AdamWState{Parameters: append([]float32(nil), flat...), First: make([]float32, len(flat)), Second: make([]float32, len(flat))}
	for _, row := range rows[:c.Protocol.TargetStep] {
		var teachers []decoder.FusionTeacher
		for i, x := range d.Teachers {
			teacher := decoder.FusionTeacher{Name: x.Model.Name, Weight: x.Weight}
			for range len(row.InputIDs) - row.PromptTokens {
				p := decoder.FusionTeacherPosition{}
				for _, q := range kdTopK[i] {
					p.TopK = append(p.TopK, decoder.FusionTokenProbability{TokenID: q.StudentTokenID, Probability: q.Probability})
					p.RetainedMass += q.Probability
				}
				teacher.Positions = append(teacher.Positions, p)
			}
			teachers = append(teachers, teacher)
		}
		tables, e := decoder.TextRotary(ctx, len(row.InputIDs), torch.CPUDevice(), c.Protocol.Local.Rotary)
		if e != nil {
			t.Fatal(e)
		}
		rotary(model, tables.Cosine, tables.Sine)
		limits := decoder.Limits{MaxTokens: int64(len(row.InputIDs)), LogitRows: int64(len(row.InputIDs) - row.PromptTokens + 1), MaxCheckpointBytes: c.Protocol.Local.MaxCheckpointBytes}
		plain, e := decoder.CompletionGradient(ctx, model.Model, row.InputIDs, row.PromptTokens, limits, c.Protocol.Local.LossScale)
		if e != nil {
			t.Fatal(e)
		}
		fused, e := decoder.FusionCompletionGradient(ctx, model.Model, row.InputIDs, row.PromptTokens, limits, c.Protocol.Local.LossScale, teachers)
		tables.Close()
		rotary(model, nil, nil)
		if e != nil {
			t.Fatal(e)
		}
		var gradient []float32
		nonzero := 0
		for _, p := range fused.Gradients {
			for _, v := range p.ValuesF32 {
				if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
					t.Fatal("nonfinite distillation gradient")
				}
				if v != 0 {
					nonzero++
				}
			}
			gradient = append(gradient, p.ValuesF32...)
		}
		if nonzero == 0 || math.IsNaN(fused.Loss) || math.IsInf(fused.Loss, 0) || math.IsNaN(fused.HardLoss) || math.IsInf(fused.HardLoss, 0) {
			t.Fatal("distillation gradient or loss is not finite and nonzero")
		}
		if state, _, e = optim.UpdateAdamW(state, gradient, c.Protocol.Local.Optimizer); e != nil {
			t.Fatal(e)
		}
		if _, e = model.ReplaceParameters(ctx, state.Parameters); e != nil {
			t.Fatal(e)
		}
		hard, blended, sft = append(hard, fused.HardLoss), append(blended, fused.Loss), append(sft, plain.Loss)
	}
	adapter, moments = sftTensorRecords(t, model, state)
	return hard, blended, sft, adapter, moments
}

func TestNativeDistillationFollowsAnIndependentFusionAdamWTrajectory(t *testing.T) {
	c, stage, rows, spec := kdStageFixture(t, 0.375, 0.25)
	h, _, manifests, _ := kdRun(t, c, stage, rows)
	hard, blended, sft, adapter, moments := kdTrajectory(t, c, rows, spec)
	d := c.Protocol.Local.Distillation
	for i, m := range manifests {
		if m.DistillationLossBefore == nil || m.LossBefore != hard[i] || *m.DistillationLossBefore != blended[i] {
			t.Fatalf("step %d readings differ from the independent run: %v %v; want %v %v", i+1, m.LossBefore, m.DistillationLossBefore, hard[i], blended[i])
		}
		if blended[i] == hard[i] {
			t.Fatal("the distillation loss equals the hard loss", i+1)
		}
		// The hard loss is the SFT NLL accumulated in another order.
		if math.Abs(hard[i]-sft[i]) > 1e-12*math.Max(1, math.Abs(sft[i])) {
			t.Fatal("the hard loss differs from the SFT loss beyond rounding", hard[i], sft[i])
		}
		t.Logf("step %d: hard_loss=%.17g sft_loss=%.17g difference=%.3g distillation_loss=%.17g", i+1, hard[i], sft[i], hard[i]-sft[i], blended[i])
		for j, x := range d.Teachers {
			retained := 0.0
			for _, q := range kdTopK[j] {
				retained += q.Probability
			}
			if mass, ok := m.TeacherMass[x.Model.Name]; !ok || math.Abs(mass-x.Weight*retained) > 1e-15 {
				t.Fatal("effective teacher mass differs", x.Model.Name, mass)
			}
			if loss, ok := m.TeacherLosses[x.Model.Name]; !ok || !(loss > 0) {
				t.Fatal("teacher loss absent or not positive", x.Model.Name, loss)
			}
		}
	}
	final := manifests[len(manifests)-1]
	if final.UpdatedAdapterSHA != sftLogical(adapter) {
		t.Fatal("final adapter differs from the independent FusionCompletionGradient and AdamW run")
	}
	names, shapes := []string{}, [][]int64{}
	for _, p := range moments {
		names = append(names, p.Name)
		shapes = append(shapes, []int64{int64(p.Shape[0]), int64(p.Shape[1])})
	}
	actual, e := readFrozenTensorFile(context.Background(), filepath.Join(stage.ArtifactDirectory, h.stepName(c.Protocol.TargetStep), "optimizer_moments.safetensors"), final.OptimizerFileSHA, names, shapes)
	if e != nil {
		t.Fatal(e)
	}
	for i, p := range moments {
		if !reflect.DeepEqual(actual[i], p.Values) {
			t.Fatal("AdamW moments differ from the independent run", p.Name)
		}
	}
	// Distillation moved the adapter somewhere SFT does not.
	_, sftAdapter, _ := freshTrajectory(t, c, rows, spec, int(c.Protocol.TargetStep))
	if sftLogical(sftAdapter) == final.UpdatedAdapterSHA {
		t.Fatal("distillation reached the SFT adapter")
	}
}

func TestNativeDistillationStageRefusesCachesBeforeAnyStep(t *testing.T) {
	outputs := func(t *testing.T, h *Stage, stage pipeline.StageContext) []os.DirEntry {
		t.Helper()
		entries, e := os.ReadDir(filepath.Join(stage.ArtifactDirectory, h.outputName()))
		if e != nil && !errors.Is(e, os.ErrNotExist) {
			t.Fatal(e)
		}
		return entries
	}
	t.Run("altered source cache", func(t *testing.T) {
		c, stage, _, _ := kdStageFixture(t, 0.375, 0.25)
		path := filepath.Join(c.CacheDirectory, c.Protocol.Local.Distillation.Teachers[1].CacheSHA256+".json")
		body, e := os.ReadFile(path)
		if e != nil {
			t.Fatal(e)
		}
		if e := os.WriteFile(path, bytes.Replace(body, []byte(`"teacher-b"`), []byte(`"teacher-c"`), 1), 0600); e != nil {
			t.Fatal(e)
		}
		calls := 0
		h, e := newStage(c, func(context.Context, Config, *stepHooks) error { calls++; return nil })
		if e != nil {
			t.Fatal(e)
		}
		if e := h.Run(context.Background(), stage, func(context.Context, pipeline.StageResult) error { return nil }); e == nil || calls != 0 {
			t.Fatal("an altered cache reached calculation", e, calls)
		}
	})
	t.Run("cache over a generated sequence", func(t *testing.T) {
		c, stage, rows, _ := kdStageFixture(t, 0.375, 0.25)
		d := c.Protocol.Local.Distillation
		generated := append([]example(nil), rows...)
		generated[0].InputIDs = []int64{1, 5}
		d.Teachers[0].CacheSHA256, d.Teachers[0].CacheBytes = kdWrite(t, c.CacheDirectory, d, 0, kdExamples(d, generated))
		sftPin(t, &c)
		calls, closed := 0, 0
		h, e := newStage(c, sftExecutor(t, &calls, &closed))
		if e != nil {
			t.Fatal(e)
		}
		if e := h.Run(context.Background(), stage, func(context.Context, pipeline.StageResult) error { return nil }); !errors.Is(e, ErrDistillation) {
			t.Fatal("a cache over another sequence was admitted", e)
		}
		if entries := outputs(t, h, stage); len(entries) != 0 {
			t.Fatal("a refused cache left an intent or a step behind", entries)
		}
	})
	t.Run("student vocabulary", func(t *testing.T) {
		c, stage, rows, _ := sftFreshFixture(t)
		c.Protocol.Student.Vocabulary = 7
		d := kdDistillation(t, c.Protocol.Student, c.Recipe.Training.SHA256, 0.375, 0.25)
		c.CacheDirectory = t.TempDir()
		kdPin(t, c.CacheDirectory, d, rows)
		c.Protocol.Local.Distillation = d
		sftPin(t, &c)
		calls, closed := 0, 0
		h, e := newStage(c, sftExecutor(t, &calls, &closed))
		if e != nil {
			t.Fatal(e)
		}
		if e := h.Run(context.Background(), stage, func(context.Context, pipeline.StageResult) error { return nil }); !errors.Is(e, ErrDistillation) {
			t.Fatal("caches over another vocabulary reached the model", e)
		}
		if entries := outputs(t, h, stage); len(entries) != 0 {
			t.Fatal("a refused vocabulary left an intent or a step behind", entries)
		}
	})
	for name, change := range map[string]func(*StageConfig){
		"bijection": func(c *StageConfig) {
			x := &c.Protocol.Local.Distillation.Teachers[0]
			x.Mapping.Identity = false
			x.Mapping.Pairs = []fusioncache.TokenPair{{Teacher: 0, Student: 0}}
			x.MappingSHA256, _ = fusioncache.Digest(x.Mapping)
		},
		"another student":       func(c *StageConfig) { c.Protocol.Local.Distillation.Student.Name = "another" },
		"another dataset":       func(c *StageConfig) { c.Protocol.Local.Distillation.DatasetSHA256 = fmtHash([]byte("another dataset")) },
		"no cache directory":    func(c *StageConfig) { c.CacheDirectory = "" },
		"stray cache directory": func(c *StageConfig) { c.Protocol.Local.Distillation = nil },
		"cache reservation":     func(c *StageConfig) { c.Protocol.Limits.WorkingBytes = kdWorkingBytes(c.Protocol) - 1 },
	} {
		t.Run(name, func(t *testing.T) {
			c, _, _, _ := kdStageFixture(t, 0.375, 0.25)
			change(&c)
			if pin, e := c.Protocol.Digest(); e == nil {
				c.ProtocolSHA256 = pin
			}
			if _, e := NewStage(c); e == nil {
				t.Fatal("an invalid distillation stage was admitted")
			}
		})
	}
	// The reservation is exactly four times the caches over what SFT reserves.
	c, _, _, _ := kdStageFixture(t, 0.375, 0.25)
	c.Protocol.Limits.WorkingBytes = kdWorkingBytes(c.Protocol)
	sftPin(t, &c)
	if _, e := NewStage(c); e != nil {
		t.Fatal("the exact cache reservation was refused", e)
	}
}

// kdWorkingBytes is the least working reservation a distillation protocol
// admits: what the SFT stage reserves plus four times its caches.
func kdWorkingBytes(p StageProtocol) int64 {
	l := p.Limits
	n := 4*l.MaxDataBytes + 16*l.MaxMetadataBytes + 16*l.MaxDataTokens + 64*l.MaxParameters + int64(l.Checkpoint.MaxChunkBytes) + l.Checkpoint.MaxHeaderBytes
	for _, x := range p.Local.Distillation.Teachers {
		n += 4 * x.CacheBytes
	}
	return n
}
