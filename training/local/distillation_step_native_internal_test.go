//go:build libtorch && cgo

package local

import (
	"context"
	"errors"
	"math"
	"testing"

	"github.com/tayi-ai/arandu-llama/training/decoder"
	"github.com/tayi-ai/arandu-llama/training/torch"
)

// kdStepModel is the CPU fixture widened to four tokens, so a row can have
// three completion tokens and its mean is not a power-of-two scaling.
func kdStepModel(t *testing.T, rotarySpec decoder.RotarySpec, tokens int) *decoder.LoadedTextModel {
	t.Helper()
	model, _ := sftCPUModel(t)
	model.Model.Layers[0].Config.MaxInputElements = int64(4 * tokens)
	model.Model.Layers[0].Config.Full.MaxScoreElements = int64(2 * tokens * tokens)
	rotarySpec.MaxTokens = tokens
	tables, e := decoder.TextRotary(context.Background(), tokens, torch.CPUDevice(), rotarySpec)
	if e != nil {
		t.Fatal(e)
	}
	rotary(model, tables.Cosine, tables.Sine)
	t.Cleanup(func() { rotary(model, nil, nil); tables.Close() })
	return model
}

// FusionCompletionGradient with a zero-weight teacher computes the one-hot
// target with other arithmetic, and is logged here beside CompletionGradient.
// The step does not rely on the two agreeing: at alpha zero it runs
// CompletionGradient itself and reports no distillation reading.
func TestNativeCompletionStepAtAlphaZeroIsCompletionGradient(t *testing.T) {
	c, _, _, _ := sftFreshFixture(t)
	row := example{"long", []int64{1, 2, 3, 4}, []int64{-100, 2, 3, 4}, 1}
	model := kdStepModel(t, c.Protocol.Local.Rotary, len(row.InputIDs))
	recipe := c.Protocol.Local
	limits := decoder.Limits{MaxTokens: 4, LogitRows: 4, MaxCheckpointBytes: recipe.MaxCheckpointBytes}
	sft, e := decoder.CompletionGradient(context.Background(), model.Model, row.InputIDs, row.PromptTokens, limits, recipe.LossScale)
	if e != nil {
		t.Fatal(e)
	}
	teachers := func(weight float64) []decoder.FusionTeacher {
		teacher := decoder.FusionTeacher{Name: "teacher-a", Weight: weight}
		for range 3 {
			teacher.Positions = append(teacher.Positions, decoder.FusionTeacherPosition{RetainedMass: 0.75, TopK: []decoder.FusionTokenProbability{{TokenID: 1, Probability: 0.5}, {TokenID: 4, Probability: 0.25}}})
		}
		return []decoder.FusionTeacher{teacher}
	}
	fused, e := decoder.FusionCompletionGradient(context.Background(), model.Model, row.InputIDs, row.PromptTokens, limits, recipe.LossScale, teachers(0))
	if e != nil {
		t.Fatal(e)
	}
	differ := 0
	for i := range sft.Gradients {
		for j, v := range sft.Gradients[i].ValuesF32 {
			if math.Float32bits(v) != math.Float32bits(fused.Gradients[i].ValuesF32[j]) {
				differ++
			}
		}
	}
	t.Logf("fusion at alpha zero: %d gradient values differ from CompletionGradient; losses %.17g %.17g", differ, fused.Loss, sft.Loss)

	d := kdDistillation(t, kdStudent(6), fmtHash([]byte("training")), 0, 0)
	recipe.BaseRevision, recipe.Distillation = d.Student.Revision, d
	step, reading, e := completionStep(context.Background(), model.Model, row, recipe, teachers(0))
	if e != nil || reading != nil {
		t.Fatal("alpha zero did not run the SFT calculation", e, reading)
	}
	if step.Loss != sft.Loss || step.Tokens != sft.Tokens {
		t.Fatal("alpha zero loss differs from SFT", step.Loss, sft.Loss)
	}
	for i := range sft.Gradients {
		for j, v := range sft.Gradients[i].ValuesF32 {
			if math.Float32bits(v) != math.Float32bits(step.Gradients[i].ValuesF32[j]) {
				t.Fatal("alpha zero gradient differs from SFT bit for bit", sft.Gradients[i].Name, j)
			}
		}
	}
	d.Teachers[0].Weight = 0.5
	step, reading, e = completionStep(context.Background(), model.Model, row, recipe, teachers(0.5))
	if e != nil || reading == nil || reading.loss == step.Loss || math.IsNaN(reading.loss) || math.IsInf(reading.loss, 0) {
		t.Fatal("positive alpha did not distil", e, reading)
	}
	for name, call := range map[string]func() error{
		"teachers without distillation": func() error {
			plain := c.Protocol.Local
			_, _, e := completionStep(context.Background(), model.Model, row, plain, teachers(0.5))
			return e
		},
		"distillation without teachers": func() error {
			_, _, e := completionStep(context.Background(), model.Model, row, recipe, nil)
			return e
		},
		"another student vocabulary": func() error {
			d.Student.Vocabulary = 7
			defer func() { d.Student.Vocabulary = 6 }()
			_, _, e := completionStep(context.Background(), model.Model, row, recipe, teachers(0.5))
			return e
		},
	} {
		if e := call(); !errors.Is(e, ErrDistillation) {
			t.Fatal(name, "was admitted", e)
		}
	}
}
