//go:build libtorch && cgo

package local

import (
	"context"
	"errors"

	"github.com/tayi-ai/arandu-llama/training/decoder"
)

// distillationReading is what a distillation step reports beside its hard NLL.
type distillationReading struct {
	loss                       float64
	teacherMass, teacherLosses map[string]float64
}

// completionStep computes the gradient of one curriculum row. Without
// distillation, or with alpha zero, it is decoder.CompletionGradient itself,
// so the step is the SFT step bit for bit. Otherwise it is
// decoder.FusionCompletionGradient over the row's teacher positions: the
// returned Loss is the hard completion NLL and the blended objective comes
// back beside it. A recipe with distillation requires teachers and a recipe
// without it refuses them, and the student's declared vocabulary must be the
// rows of the model's output head.
func completionStep(ctx context.Context, model *decoder.TextModel, row example, recipe Recipe, teachers []decoder.FusionTeacher) (decoder.CompletionGradientResult, *distillationReading, error) {
	d := recipe.Distillation
	if (d == nil) != (teachers == nil) {
		return decoder.CompletionGradientResult{}, nil, errors.Join(ErrDistillation, errors.New("teacher signals and the recipe's distillation differ"))
	}
	if err := distillationStudent(model, d); err != nil {
		return decoder.CompletionGradientResult{}, nil, err
	}
	tokens := int64(len(row.InputIDs))
	limits := decoder.Limits{MaxTokens: tokens, LogitRows: tokens - int64(row.PromptTokens) + 1, MaxCheckpointBytes: recipe.MaxCheckpointBytes}
	if d == nil || d.alpha() == 0 {
		gradient, err := decoder.CompletionGradient(ctx, model, row.InputIDs, row.PromptTokens, limits, recipe.LossScale)
		return gradient, nil, err
	}
	fused, err := decoder.FusionCompletionGradient(ctx, model, row.InputIDs, row.PromptTokens, limits, recipe.LossScale, teachers)
	if err != nil {
		return decoder.CompletionGradientResult{}, nil, err
	}
	return decoder.CompletionGradientResult{Loss: fused.HardLoss, Tokens: fused.Tokens, Gradients: fused.Gradients},
		&distillationReading{loss: fused.Loss, teacherMass: fused.EffectiveTeacherMass, teacherLosses: fused.TeacherLosses}, nil
}

// distillationStudent refuses a model whose output head is not the student
// every cache names: the caches' token ids index the declared vocabulary.
func distillationStudent(model *decoder.TextModel, d *Distillation) error {
	if d == nil {
		return nil
	}
	head, err := model.Head.Info()
	if err != nil || len(head.Shape) != 2 || head.Shape[0] != int64(d.Student.Vocabulary) {
		return errors.Join(ErrDistillation, errors.New("the student's vocabulary differs from the model's output rows"), err)
	}
	return nil
}
