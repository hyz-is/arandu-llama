package decoder

import (
	"context"
	"errors"
	"fmt"
	"math"

	"github.com/tayi-ai/arandu-llama/training/torch"
)

// ErrCompletionStep identifies invalid autoregressive completion training data.
var ErrCompletionStep = errors.New("decoder: completion calculation rejected")

// CompletionParameterGradient owns a detached Go FP32 copy of one trainable
// adapter cotangent. Names and ordering follow the model's frozen PEFT layout.
type CompletionParameterGradient struct {
	Name      string
	Shape     []int64
	ValuesF32 []float32
}

// CompletionGradientResult contains the mean supervised-token negative log
// likelihood and the ordered LoRA gradients that produce that loss.
type CompletionGradientResult struct {
	Loss      float64
	Tokens    int
	Gradients []CompletionParameterGradient
}

// CompletionGradient computes exact causal-LM cross entropy over the final
// completion tokens only. tokenIDs contains prompt+completion. promptTokens is
// the first supervised token index. The model sees the entire sequence and the
// final target token is predicted from its preceding row, matching standard
// shifted causal-LM labels with -100 over the prompt.
//
// limits.LogitRows must equal completionTokens+1: one preceding row is needed
// to predict the first completion token, and the final row is intentionally
// unscored after the causal shift. lossScale multiplies only the cotangent;
// the reported Loss remains the unscaled mean NLL.
func CompletionGradient(ctx context.Context, model *TextModel, tokenIDs []int64, promptTokens int, limits Limits, lossScale float64) (result CompletionGradientResult, err error) {
	if ctx == nil || model == nil || len(tokenIDs) < 2 || promptTokens < 1 || promptTokens >= len(tokenIDs) {
		return result, fmt.Errorf("%w: prompt and completion geometry is invalid", ErrCompletionStep)
	}
	if math.IsNaN(lossScale) || math.IsInf(lossScale, 0) || lossScale <= 0 {
		return result, fmt.Errorf("%w: finite positive loss scale required", ErrCompletionStep)
	}
	completionTokens := len(tokenIDs) - promptTokens
	if limits.LogitRows != int64(completionTokens+1) {
		return result, fmt.Errorf("%w: logit rows must equal supervised tokens plus one", ErrCompletionStep)
	}
	if err := ctx.Err(); err != nil {
		return result, err
	}
	expected, err := candidateParameters(model)
	if err != nil {
		return result, err
	}
	snapshot, err := model.Forward(ctx, tokenIDs, limits)
	if err != nil {
		return result, err
	}
	var seed *torch.Tensor
	var nativeGradients Gradients
	defer func() {
		err = errors.Join(err, nativeGradients.Close(), seed.Close(), snapshot.Close())
		if err != nil {
			result = CompletionGradientResult{}
		}
	}()

	info, values, err := completionLogits(snapshot, completionTokens)
	if err != nil {
		return result, err
	}
	vocabulary := int(info.Shape[2])
	result.Loss, err = completionCotangent(values, vocabulary, tokenIDs[promptTokens:], lossScale)
	if err != nil {
		return result, err
	}
	result.Tokens = completionTokens
	seed, err = torch.FromFloat32(values, info.Shape, info.Device, false)
	if err != nil {
		return result, err
	}
	if err := ctx.Err(); err != nil {
		return result, err
	}
	nativeGradients, err = model.VJP(ctx, snapshot, seed)
	if err != nil {
		return result, err
	}
	if len(nativeGradients) != len(expected) {
		return result, fmt.Errorf("%w: incomplete parameter gradient set", ErrCompletionStep)
	}
	for index, gradient := range nativeGradients {
		if err := ctx.Err(); err != nil {
			return result, err
		}
		info, err := gradient.Value.Info()
		if err != nil {
			return result, err
		}
		if gradient.Name != expected[index].name || info.DType != torch.Float32 || info.RequiresGrad ||
			info.Device != expected[index].info.Device || !sameShape(info.Shape, expected[index].info.Shape) {
			return result, fmt.Errorf("%w: gradient name, geometry, precision or placement differs", ErrCompletionStep)
		}
		copied, err := gradient.Value.Float32Values()
		if err != nil {
			return result, err
		}
		for _, value := range copied {
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				return result, fmt.Errorf("%w: nonfinite parameter gradient", ErrCompletionStep)
			}
		}
		result.Gradients = append(result.Gradients, CompletionParameterGradient{
			Name: gradient.Name, Shape: info.Shape, ValuesF32: copied,
		})
	}
	return result, ctx.Err()
}

// CompletionLoss reports the Loss CompletionGradient would report for the same
// model, tokens and limits, without building a cotangent or running VJP. The
// geometry, the admitted logits and the accumulation order are the gradient's,
// so the two readings are bitwise equal; the logits are read and never written.
// Model parameters are not changed. Adapters are optional here: the loss of a
// model is defined whether or not it can be trained.
func CompletionLoss(ctx context.Context, model *TextModel, tokenIDs []int64, promptTokens int, limits Limits) (loss float64, err error) {
	if ctx == nil || model == nil || len(tokenIDs) < 2 || promptTokens < 1 || promptTokens >= len(tokenIDs) {
		return 0, fmt.Errorf("%w: prompt and completion geometry is invalid", ErrCompletionStep)
	}
	completionTokens := len(tokenIDs) - promptTokens
	if limits.LogitRows != int64(completionTokens+1) {
		return 0, fmt.Errorf("%w: logit rows must equal supervised tokens plus one", ErrCompletionStep)
	}
	if err := ctx.Err(); err != nil {
		return 0, err
	}
	snapshot, err := model.Forward(ctx, tokenIDs, limits)
	if err != nil {
		return 0, err
	}
	defer func() {
		err = errors.Join(err, snapshot.Close())
		if err != nil {
			loss = 0
		}
	}()
	info, values, err := completionLogits(snapshot, completionTokens)
	if err != nil {
		return 0, err
	}
	loss, err = completionNLL(values, int(info.Shape[2]), tokenIDs[promptTokens:], nil)
	if err != nil {
		return 0, err
	}
	return loss, ctx.Err()
}

// completionLogits admits the one logits layout both readings accept and
// returns a Go-owned FP32 copy of it.
func completionLogits(snapshot *Snapshot, completionTokens int) (torch.Info, []float32, error) {
	info, err := snapshot.Logits.Info()
	if err != nil {
		return info, nil, err
	}
	if len(info.Shape) != 3 || info.Shape[0] != 1 || info.Shape[1] != int64(completionTokens+1) ||
		info.Shape[2] <= 1 || info.DType != torch.Float32 {
		return info, nil, fmt.Errorf("%w: completion logits geometry differs", ErrCompletionStep)
	}
	values, err := snapshot.Logits.Float32Values()
	if err != nil {
		return info, nil, err
	}
	if len(values) != (completionTokens+1)*int(info.Shape[2]) {
		return info, nil, fmt.Errorf("%w: completion logit payload differs", ErrCompletionStep)
	}
	return info, values, nil
}

func completionCotangent(logits []float32, vocabulary int, targets []int64, lossScale float64) (float64, error) {
	if vocabulary < 2 || len(targets) == 0 || len(logits) != (len(targets)+1)*vocabulary ||
		math.IsNaN(lossScale) || math.IsInf(lossScale, 0) || lossScale <= 0 {
		return 0, fmt.Errorf("%w: invalid completion cotangent geometry", ErrCompletionStep)
	}
	factor := lossScale / float64(len(targets))
	loss, err := completionNLL(logits, vocabulary, targets, func(offset, target int, maximum, sum float64) error {
		for column := 0; column < vocabulary; column++ {
			probability := math.Exp(float64(logits[offset+column])-maximum) / sum
			gradient := probability * factor
			if column == target {
				gradient -= factor
			}
			rounded := float32(gradient)
			if math.IsNaN(gradient) || math.IsInf(gradient, 0) ||
				math.IsNaN(float64(rounded)) || math.IsInf(float64(rounded), 0) {
				return fmt.Errorf("%w: completion derivative is not finite FP32", ErrCompletionStep)
			}
			logits[offset+column] = rounded
		}
		return nil
	})
	if err != nil {
		return 0, err
	}
	clear(logits[len(targets)*vocabulary:])
	return loss, nil
}

// completionNLL is the one loss both the gradient and the readout report. It
// only reads logits; scored, when set, runs after each row's loss term is
// accumulated and before the next row is read, so it may overwrite that row.
// Keeping one accumulation order keeps the two losses bitwise equal.
func completionNLL(logits []float32, vocabulary int, targets []int64, scored func(offset, target int, maximum, sum float64) error) (float64, error) {
	if vocabulary < 2 || len(targets) == 0 || len(logits) != (len(targets)+1)*vocabulary {
		return 0, fmt.Errorf("%w: invalid completion loss geometry", ErrCompletionStep)
	}
	for _, target := range targets {
		if target < 0 || target >= int64(vocabulary) {
			return 0, fmt.Errorf("%w: target token outside vocabulary", ErrCompletionStep)
		}
	}
	for _, value := range logits[:len(targets)*vocabulary] {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return 0, fmt.Errorf("%w: nonfinite completion logit", ErrCompletionStep)
		}
	}

	totalLoss := 0.0
	for row, target := range targets {
		offset := row * vocabulary
		maximum := float64(logits[offset])
		for column := 1; column < vocabulary; column++ {
			maximum = math.Max(maximum, float64(logits[offset+column]))
		}
		sum := 0.0
		for column := 0; column < vocabulary; column++ {
			sum += math.Exp(float64(logits[offset+column]) - maximum)
		}
		if !(sum > 0) || math.IsNaN(sum) || math.IsInf(sum, 0) {
			return 0, fmt.Errorf("%w: invalid completion softmax normalization", ErrCompletionStep)
		}
		logSum := math.Log(sum)
		targetLogit := float64(logits[offset+int(target)])
		totalLoss += maximum + logSum - targetLogit
		if scored != nil {
			if err := scored(offset, int(target), maximum, sum); err != nil {
				return 0, err
			}
		}
	}
	loss := totalLoss / float64(len(targets))
	if math.IsNaN(loss) || math.IsInf(loss, 0) {
		return 0, fmt.Errorf("%w: nonfinite completion loss", ErrCompletionStep)
	}
	return loss, nil
}
