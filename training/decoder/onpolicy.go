package decoder

import (
	"context"
	"errors"
	"fmt"
	"math"

	"github.com/tayi-ai/arandu-llama/training/torch"
)

// OnPolicyMode names one on-policy distillation objective over a completion
// the student sampled.
type OnPolicyMode string

const (
	// OnPolicyForwardComplement is the forward KL from the teacher's top-k and
	// one complement cell to the student's mass over the same cells.
	OnPolicyForwardComplement OnPolicyMode = "fkl-complement"
	// OnPolicyJSDComplement is the generalized Jensen-Shannon divergence of
	// OnPolicyObjective.Beta over the same cells.
	OnPolicyJSDComplement OnPolicyMode = "jsd-complement"
	// OnPolicyReverseSampled is the reverse KL estimated at the realized token
	// under a truncated importance weight.
	OnPolicyReverseSampled OnPolicyMode = "rkl-sampled"
)

// onPolicyMassTolerance bounds every comparison between probability masses:
// a declared retained mass against its top-k, a residual that must vanish, and
// the teacher's probability of the realized token against its top-k. It is the
// tolerance FusionCompletionGradient applies to a retained mass.
const onPolicyMassTolerance = 1e-6

// OnPolicyObjective selects an objective and its one parameter. Beta is read
// only by jsd-complement and Clip only by rkl-sampled; a parameter set for a
// mode that does not read it is refused, never ignored.
type OnPolicyObjective struct {
	Mode OnPolicyMode
	// Beta is the teacher's share of the mixture M = Beta*P + (1-Beta)*Q, in
	// the open interval (0, 1). At either end the divergence is identically
	// zero, so both ends are refused rather than redefined.
	Beta float64
	// Clip is the ceiling of the per-token importance weight min(q/mu, Clip).
	// It is finite and at least one, so a token sampled exactly on-policy
	// keeps weight one.
	Clip float64
}

// Validate refuses an unknown mode, a parameter outside its range and a
// parameter set for a mode that does not read it.
func (o OnPolicyObjective) Validate() error {
	switch o.Mode {
	case OnPolicyForwardComplement:
		if o.Beta != 0 || o.Clip != 0 {
			return fmt.Errorf("%w: %s reads neither beta nor clip", ErrCompletionStep, o.Mode)
		}
	case OnPolicyJSDComplement:
		if !(o.Beta > 0 && o.Beta < 1) || o.Clip != 0 {
			return fmt.Errorf("%w: %s requires beta in (0, 1) and no clip", ErrCompletionStep, o.Mode)
		}
	case OnPolicyReverseSampled:
		if !(o.Clip >= 1) || math.IsInf(o.Clip, 1) || o.Beta != 0 {
			return fmt.Errorf("%w: %s requires a finite clip of at least one and no beta", ErrCompletionStep, o.Mode)
		}
	default:
		return fmt.Errorf("%w: unknown on-policy mode %q", ErrCompletionStep, o.Mode)
	}
	return nil
}

// OnPolicyTarget is the teacher's reading of a completion the student sampled.
//
// Positions holds one teacher top-k per completion token, already mapped to
// the student's vocabulary, as FusionCompletionGradient reads it: position t
// is the teacher's full-vocabulary softmax over the token at promptTokens+t,
// given every token before it, restricted to its top-k and not renormalized.
// Every mode requires Positions, with a non-empty top-k at every position.
//
// TeacherLogProbabilities holds log p_T(y_t), the teacher's full-vocabulary log
// probability of the realized token y_t, and BehaviorLogProbabilities holds
// log mu(y_t), its log probability under the policy that sampled it. Each is
// nil or has one finite, non-positive value per completion token.
// rkl-sampled requires both. The other modes do not read them, and validate
// them when present. A teacher log probability must agree with the teacher's
// own top-k: within the mass tolerance of the top-k entry when y_t is in the
// top-k, and at most the teacher's residual mass when it is not.
type OnPolicyTarget struct {
	Positions                []FusionTeacherPosition
	TeacherLogProbabilities  []float64
	BehaviorLogProbabilities []float64
}

// OnPolicyCompletionResult reports an on-policy objective and its LoRA
// gradients in the layout of FusionCompletionResult. Loss is the unscaled mean
// of the mode's per-token loss over the completion tokens, and Tokens is their
// number. StudentLogProbabilities is log q(y_t) for every completion token,
// read from the logits the gradient is built from, and is bitwise equal to
// CompletionTokenLogProbabilities over the same model, tokens and limits.
// ImportanceWeights is min(q(y_t)/mu(y_t), Clip) per token under rkl-sampled
// and nil otherwise.
type OnPolicyCompletionResult struct {
	Loss                    float64
	Tokens                  int
	StudentLogProbabilities []float64
	ImportanceWeights       []float64
	Gradients               []CompletionParameterGradient
}

// OnPolicyCompletionGradient computes an on-policy distillation objective over
// the completion of tokenIDs, a sequence the student sampled after its first
// promptTokens tokens, and the LoRA gradients of that objective. The geometry
// is CompletionGradient's: limits.LogitRows must equal completionTokens+1 and
// the final row is unscored. lossScale multiplies only the cotangent.
//
// For completion token t, q is the student's softmax over the logits z of the
// row that predicts it, K the teacher's top-k with probabilities p,
// r_T = max(0, 1 - sum_K p) the teacher's mass outside K, and
// r_Q = sum_{j not in K} q_j the student's, read in log space from the logits
// outside K rather than as 1 - sum_K q. The cells are the tokens of K and one
// complement cell holding every token outside it; P and Q are the teacher's and
// the student's masses over those cells.
//
//   - fkl-complement: l_t = -sum_K p_i log q_i - r_T log r_Q, the cross entropy
//     over the cells, which is KL(P||Q) plus the teacher's cell entropy, a
//     constant of the student. dl_t/dz_j = q_j - p_j for j in K and
//     q_j (1 - r_T/r_Q) outside K. The realized token receives no mass.
//   - jsd-complement: l_t = Beta KL(P||M) + (1-Beta) KL(Q||M) over the cells,
//     with M = Beta P + (1-Beta) Q. With g_c = (1-Beta) log(Q_c/M_c) and
//     gbar = sum_c Q_c g_c, dl_t/dz_j = q_j (g_c(j) - gbar), where c(j) is the
//     cell of token j. The realized token receives no mass.
//   - rkl-sampled: with a_t = log q(y_t) - log p_T(y_t) and the truncated weight
//     w_t = min(exp(log q(y_t) - log mu(y_t)), Clip), the cotangent row is
//     w_t a_t (onehot(y_t) - q), the gradient of w_t a_t log q(y_t) with w_t and
//     a_t held constant: the score-function estimate of the reverse KL at the
//     sampled token, with advantage -a_t and no return from later tokens.
//     l_t = w_t a_t, which is the weighted per-token estimate and not the
//     function the gradient differentiates.
//
// Loss is (1/n) sum_t l_t over the n completion tokens, summed in token order
// and divided once. The cotangent of the row predicting token t is
// (lossScale/n) dl_t/dz. Every completion token weighs the same.
//
// A cell whose teacher mass is zero contributes nothing to the loss. When K
// covers the whole vocabulary there is no complement cell and r_T must be
// within the mass tolerance of zero. When every top-k sums to exactly one and
// declares a retained mass of exactly one, fkl-complement evaluates the
// expressions of FusionCompletionGradient with one teacher of weight one, in
// the same order, so the two agree bit for bit.
func OnPolicyCompletionGradient(ctx context.Context, model *TextModel, tokenIDs []int64, promptTokens int, limits Limits, lossScale float64, objective OnPolicyObjective, target OnPolicyTarget) (result OnPolicyCompletionResult, err error) {
	if ctx == nil || model == nil || len(tokenIDs) < 2 || promptTokens < 1 || promptTokens >= len(tokenIDs) {
		return result, fmt.Errorf("%w: prompt and completion geometry is invalid", ErrCompletionStep)
	}
	if math.IsNaN(lossScale) || math.IsInf(lossScale, 0) || lossScale <= 0 {
		return result, fmt.Errorf("%w: finite positive loss scale required", ErrCompletionStep)
	}
	if err := objective.Validate(); err != nil {
		return result, err
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
			result = OnPolicyCompletionResult{}
		}
	}()

	info, values, err := completionLogits(snapshot, completionTokens)
	if err != nil {
		return result, err
	}
	stats, err := onPolicyCotangent(values, int(info.Shape[2]), tokenIDs[promptTokens:], lossScale, objective, target)
	if err != nil {
		return result, err
	}
	result.Loss = stats.loss
	result.Tokens = completionTokens
	result.StudentLogProbabilities = stats.logQ
	result.ImportanceWeights = stats.weights

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
	result.Gradients, err = copyCompletionGradients(ctx, expected, nativeGradients)
	if err != nil {
		return result, err
	}
	return result, ctx.Err()
}

// CompletionTokenLogProbabilities reads log q(y_t), the student's log-softmax
// at every completion token of tokenIDs, from one forward pass without a
// cotangent or VJP. The geometry is CompletionLoss's. The logits, the
// normalization and the expression are OnPolicyCompletionGradient's, so the
// two readings are bitwise equal. Model parameters are not changed, and
// adapters are optional.
func CompletionTokenLogProbabilities(ctx context.Context, model *TextModel, tokenIDs []int64, promptTokens int, limits Limits) (logProbabilities []float64, err error) {
	if ctx == nil || model == nil || len(tokenIDs) < 2 || promptTokens < 1 || promptTokens >= len(tokenIDs) {
		return nil, fmt.Errorf("%w: prompt and completion geometry is invalid", ErrCompletionStep)
	}
	completionTokens := len(tokenIDs) - promptTokens
	if limits.LogitRows != int64(completionTokens+1) {
		return nil, fmt.Errorf("%w: logit rows must equal supervised tokens plus one", ErrCompletionStep)
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	snapshot, err := model.Forward(ctx, tokenIDs, limits)
	if err != nil {
		return nil, err
	}
	defer func() {
		err = errors.Join(err, snapshot.Close())
		if err != nil {
			logProbabilities = nil
		}
	}()
	info, values, err := completionLogits(snapshot, completionTokens)
	if err != nil {
		return nil, err
	}
	logProbabilities, err = tokenLogProbabilities(values, int(info.Shape[2]), tokenIDs[promptTokens:])
	if err != nil {
		return nil, err
	}
	return logProbabilities, ctx.Err()
}

type onPolicyStats struct {
	loss    float64
	logQ    []float64
	weights []float64
}

// complementCell is the student's normalizer over the tokens outside K:
// maximum and sum are the shifted sum of exponentials, and logMass is
// log r_Q. present is false when K covers the vocabulary.
type complementCell struct {
	present      bool
	maximum, sum float64
	logMass      float64
}

// onPolicyCotangent validates every input before it writes a logit, then
// overwrites the supervised rows with the cotangent and zeroes the final row.
func onPolicyCotangent(logits []float32, vocabulary int, targets []int64, lossScale float64, objective OnPolicyObjective, target OnPolicyTarget) (onPolicyStats, error) {
	if math.IsNaN(lossScale) || math.IsInf(lossScale, 0) || lossScale <= 0 {
		return onPolicyStats{}, fmt.Errorf("%w: finite positive loss scale required", ErrCompletionStep)
	}
	if err := objective.Validate(); err != nil {
		return onPolicyStats{}, err
	}
	if err := admitCompletionRows(logits, vocabulary, targets); err != nil {
		return onPolicyStats{}, err
	}
	residuals, err := onPolicyResiduals(vocabulary, targets, objective, target)
	if err != nil {
		return onPolicyStats{}, err
	}

	stats := onPolicyStats{logQ: make([]float64, len(targets))}
	if objective.Mode == OnPolicyReverseSampled {
		stats.weights = make([]float64, len(targets))
	}
	var dense []float64
	var member []bool
	if objective.Mode != OnPolicyReverseSampled {
		dense = make([]float64, vocabulary)
		member = make([]bool, vocabulary)
	}
	factor := lossScale / float64(len(targets))
	for row, token := range targets {
		z := logits[row*vocabulary : (row+1)*vocabulary]
		maximum, sum, logSum, err := softmaxNormalizer(z)
		if err != nil {
			return onPolicyStats{}, err
		}
		stats.logQ[row] = float64(z[token]) - maximum - logSum
		var rowLoss float64
		switch objective.Mode {
		case OnPolicyReverseSampled:
			rowLoss, stats.weights[row], err = reverseSampledRow(z, maximum, sum, int(token), stats.logQ[row],
				target.TeacherLogProbabilities[row], target.BehaviorLogProbabilities[row], objective.Clip, factor)
		default:
			topK := target.Positions[row].TopK
			for _, item := range topK {
				member[item.TokenID] = true
			}
			complement := complementOf(z, member, maximum, logSum)
			if objective.Mode == OnPolicyForwardComplement {
				for _, item := range topK {
					dense[item.TokenID] = item.Probability
				}
				rowLoss, err = forwardComplementRow(z, maximum, sum, logSum, residuals[row], complement, factor, dense, member)
			} else {
				rowLoss, err = jsdComplementRow(z, maximum, sum, logSum, topK, residuals[row], complement, objective.Beta, factor, dense, member)
			}
			for _, item := range topK {
				member[item.TokenID], dense[item.TokenID] = false, 0
			}
		}
		if err != nil {
			return onPolicyStats{}, err
		}
		stats.loss += rowLoss
	}
	clear(logits[len(targets)*vocabulary:])
	stats.loss /= float64(len(targets))
	if math.IsNaN(stats.loss) || math.IsInf(stats.loss, 0) {
		return onPolicyStats{}, fmt.Errorf("%w: nonfinite on-policy loss", ErrCompletionStep)
	}
	return stats, nil
}

// forwardComplementRow evaluates FusionCompletionGradient's row expressions:
// dense holds p_i at the tokens of K. The complement term follows the tokens.
func forwardComplementRow(z []float32, maximum, sum, logSum, residual float64, complement complementCell, factor float64, dense []float64, member []bool) (float64, error) {
	rowLoss := 0.0
	for column, value := range z {
		probability := math.Exp(float64(value)-maximum) / sum
		var gradient float64
		if member[column] {
			mass := dense[column]
			if mass != 0 {
				rowLoss -= mass * (float64(value) - maximum - logSum)
			}
			gradient = probability - mass
		} else {
			gradient = probability - residual*math.Exp(float64(value)-complement.maximum)/complement.sum
		}
		if err := writeCotangent(z, column, gradient*factor); err != nil {
			return 0, err
		}
	}
	if residual != 0 {
		rowLoss -= residual * complement.logMass
	}
	return rowLoss, nil
}

// jsdComplementRow accumulates the cells in top-k order, then the complement.
// dense receives g_c at the tokens of K.
func jsdComplementRow(z []float32, maximum, sum, logSum float64, topK []FusionTokenProbability, residual float64, complement complementCell, beta, factor float64, dense []float64, member []bool) (float64, error) {
	logBeta, logRest := math.Log(beta), math.Log1p(-beta)
	teacherTerm, gBar := 0.0, 0.0
	cell := func(p, logQ float64) float64 {
		logM := logRest + logQ
		if p > 0 {
			logP := math.Log(p)
			logM = logAddExp(logBeta+logP, logM)
			teacherTerm += p * (logP - logM)
		}
		g := (1 - beta) * (logQ - logM)
		gBar += math.Exp(logQ) * g
		return g
	}
	for _, item := range topK {
		dense[item.TokenID] = cell(item.Probability, float64(z[item.TokenID])-maximum-logSum)
	}
	complementG := 0.0
	if complement.present {
		complementG = cell(residual, complement.logMass)
	}
	for column, value := range z {
		probability := math.Exp(float64(value)-maximum) / sum
		g := complementG
		if member[column] {
			g = dense[column]
		}
		if err := writeCotangent(z, column, probability*(g-gBar)*factor); err != nil {
			return 0, err
		}
	}
	return beta*teacherTerm + gBar, nil
}

// reverseSampledRow writes w a (onehot(token) - q) and returns w a and w.
func reverseSampledRow(z []float32, maximum, sum float64, token int, logQ, logTeacher, logBehavior, clip, factor float64) (float64, float64, error) {
	weight := math.Min(math.Exp(logQ-logBehavior), clip)
	coefficient := weight * (logQ - logTeacher)
	for column, value := range z {
		indicator := 0.0
		if column == token {
			indicator = 1
		}
		probability := math.Exp(float64(value)-maximum) / sum
		if err := writeCotangent(z, column, coefficient*(indicator-probability)*factor); err != nil {
			return 0, 0, err
		}
	}
	return coefficient, weight, nil
}

// complementOf reads the student's shifted normalizer over the tokens outside
// K from the unwritten row.
func complementOf(z []float32, member []bool, maximum, logSum float64) complementCell {
	var cell complementCell
	for column, value := range z {
		if member[column] {
			continue
		}
		if !cell.present || float64(value) > cell.maximum {
			cell.maximum = float64(value)
		}
		cell.present = true
	}
	if !cell.present {
		return cell
	}
	for column, value := range z {
		if !member[column] {
			cell.sum += math.Exp(float64(value) - cell.maximum)
		}
	}
	cell.logMass = cell.maximum - maximum + math.Log(cell.sum) - logSum
	return cell
}

// onPolicyResiduals validates the target against the completion and returns
// r_T for every position.
func onPolicyResiduals(vocabulary int, targets []int64, objective OnPolicyObjective, target OnPolicyTarget) ([]float64, error) {
	n := len(targets)
	if len(target.Positions) != n {
		return nil, fmt.Errorf("%w: one teacher position per completion token required", ErrCompletionStep)
	}
	teacherLog, behaviorLog := target.TeacherLogProbabilities, target.BehaviorLogProbabilities
	if objective.Mode == OnPolicyReverseSampled && (teacherLog == nil || behaviorLog == nil) {
		return nil, fmt.Errorf("%w: %s requires the teacher and behavior log probability of every realized token", ErrCompletionStep, objective.Mode)
	}
	for _, series := range [][]float64{teacherLog, behaviorLog} {
		if series != nil && len(series) != n {
			return nil, fmt.Errorf("%w: one sampled-token log probability per completion token required", ErrCompletionStep)
		}
		for _, value := range series {
			if !(value <= 0) || math.IsInf(value, -1) {
				return nil, fmt.Errorf("%w: sampled-token log probability must be finite and non-positive", ErrCompletionStep)
			}
		}
	}
	residuals := make([]float64, n)
	seen := make(map[int64]bool)
	for row, position := range target.Positions {
		if !(position.RetainedMass >= 0 && position.RetainedMass <= 1) {
			return nil, fmt.Errorf("%w: retained teacher mass must be within [0, 1]", ErrCompletionStep)
		}
		if len(position.TopK) == 0 {
			return nil, fmt.Errorf("%w: empty teacher top-k", ErrCompletionStep)
		}
		clear(seen)
		retained, sampled, inTopK := 0.0, 0.0, false
		for _, item := range position.TopK {
			if item.TokenID < 0 || item.TokenID >= int64(vocabulary) || seen[item.TokenID] ||
				!(item.Probability >= 0 && item.Probability <= 1) {
				return nil, fmt.Errorf("%w: invalid mapped teacher token", ErrCompletionStep)
			}
			seen[item.TokenID] = true
			retained += item.Probability
			if item.TokenID == targets[row] {
				sampled, inTopK = item.Probability, true
			}
		}
		if math.Abs(retained-position.RetainedMass) > onPolicyMassTolerance {
			return nil, fmt.Errorf("%w: teacher retained mass differs from mapped top-k", ErrCompletionStep)
		}
		residual := math.Max(0, 1-retained)
		if len(position.TopK) == vocabulary {
			if residual > onPolicyMassTolerance {
				return nil, fmt.Errorf("%w: teacher mass outside a top-k that covers the vocabulary", ErrCompletionStep)
			}
			residual = 0
		}
		residuals[row] = residual
		if teacherLog == nil {
			continue
		}
		probability := math.Exp(teacherLog[row])
		if (inTopK && math.Abs(probability-sampled) > onPolicyMassTolerance) ||
			(!inTopK && probability > residual+onPolicyMassTolerance) {
			return nil, fmt.Errorf("%w: teacher log probability of the realized token disagrees with its top-k", ErrCompletionStep)
		}
	}
	return residuals, nil
}

// tokenLogProbabilities is the reading CompletionTokenLogProbabilities and
// OnPolicyCompletionGradient share.
func tokenLogProbabilities(logits []float32, vocabulary int, targets []int64) ([]float64, error) {
	if err := admitCompletionRows(logits, vocabulary, targets); err != nil {
		return nil, err
	}
	out := make([]float64, len(targets))
	for row, token := range targets {
		z := logits[row*vocabulary : (row+1)*vocabulary]
		maximum, _, logSum, err := softmaxNormalizer(z)
		if err != nil {
			return nil, err
		}
		out[row] = float64(z[token]) - maximum - logSum
	}
	return out, nil
}

// admitCompletionRows refuses a logits payload whose geometry differs from
// the completion, a target outside the vocabulary and a nonfinite supervised
// logit.
func admitCompletionRows(logits []float32, vocabulary int, targets []int64) error {
	if vocabulary < 2 || len(targets) == 0 || len(logits) != (len(targets)+1)*vocabulary {
		return fmt.Errorf("%w: invalid completion logits geometry", ErrCompletionStep)
	}
	for _, token := range targets {
		if token < 0 || token >= int64(vocabulary) {
			return fmt.Errorf("%w: target token outside vocabulary", ErrCompletionStep)
		}
	}
	for _, value := range logits[:len(targets)*vocabulary] {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return fmt.Errorf("%w: nonfinite completion logit", ErrCompletionStep)
		}
	}
	return nil
}

// softmaxNormalizer returns the row maximum, the shifted sum of exponentials
// in column order and its logarithm, as FusionCompletionGradient builds them.
func softmaxNormalizer(z []float32) (maximum, sum, logSum float64, err error) {
	maximum = float64(z[0])
	for _, value := range z[1:] {
		maximum = math.Max(maximum, float64(value))
	}
	for _, value := range z {
		sum += math.Exp(float64(value) - maximum)
	}
	if !(sum > 0) || math.IsInf(sum, 0) {
		return 0, 0, 0, fmt.Errorf("%w: invalid softmax normalization", ErrCompletionStep)
	}
	return maximum, sum, math.Log(sum), nil
}

func writeCotangent(z []float32, column int, gradient float64) error {
	rounded := float32(gradient)
	if math.IsNaN(gradient) || math.IsInf(gradient, 0) || math.IsInf(float64(rounded), 0) {
		return fmt.Errorf("%w: on-policy derivative is not finite FP32", ErrCompletionStep)
	}
	z[column] = rounded
	return nil
}

func logAddExp(a, b float64) float64 {
	if a < b {
		a, b = b, a
	}
	return a + math.Log1p(math.Exp(b-a))
}

// copyCompletionGradients admits the VJP's gradients against the model's
// trainable parameters and returns detached Go copies in parameter order.
func copyCompletionGradients(ctx context.Context, expected []candidateParameter, native Gradients) ([]CompletionParameterGradient, error) {
	if len(native) != len(expected) {
		return nil, fmt.Errorf("%w: incomplete parameter gradient set", ErrCompletionStep)
	}
	out := make([]CompletionParameterGradient, 0, len(native))
	for index, gradient := range native {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		info, err := gradient.Value.Info()
		if err != nil {
			return nil, err
		}
		if gradient.Name != expected[index].name || info.DType != torch.Float32 || info.RequiresGrad ||
			info.Device != expected[index].info.Device || !sameShape(info.Shape, expected[index].info.Shape) {
			return nil, fmt.Errorf("%w: gradient name, geometry, precision or placement differs", ErrCompletionStep)
		}
		copied, err := gradient.Value.Float32Values()
		if err != nil {
			return nil, err
		}
		for _, value := range copied {
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				return nil, fmt.Errorf("%w: nonfinite parameter gradient", ErrCompletionStep)
			}
		}
		out = append(out, CompletionParameterGradient{Name: gradient.Name, Shape: info.Shape, ValuesF32: copied})
	}
	return out, nil
}
