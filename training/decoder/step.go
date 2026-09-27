package decoder

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"math"

	"github.com/tayi-ai/arandu-llama/training/torch"
)

// ErrCandidateStep identifies invalid candidate scoring or gradient data.
var ErrCandidateStep = errors.New("decoder: candidate calculation rejected")

// LossGradient maps four raw candidate logits to their four derivatives. The
// caller owns the loss, target, scaling and numerical policy. The callback must
// not change model weights, rotary values or token inputs during the operation.
// No softmax, loss scale, normalization or optimizer is implicit in this API.
type LossGradient func([4]float64) ([4]float64, error)

// CandidateParameterGradient owns Go copies of one named FP32 parameter
// cotangent. Shape and ValuesF32 are independent of the model/native graph.
type CandidateParameterGradient struct {
	Name      string
	Shape     []int64
	ValuesF32 []float32
}

// CandidateGradientResult contains observed raw logits and ordered parameter
// derivatives. It contains no native handles or optimizer updates. Names follow
// model layer order, canonical projection order, then A/B; registry size is
// caller-owned.
type CandidateGradientResult struct {
	Logits    [4]float64
	Gradients []CandidateParameterGradient
}

// ReadCandidateLogits returns four raw FP32 logits widened to Go float64 from
// the penultimate token position. IDs are ordered A/B/C/D; promptIDs must already
// end with candidateIDs[0]. Limits.LogitRows must be exactly two. It runs a fresh
// cache-free forward and closes its snapshot before returning, without a VJP.
func ReadCandidateLogits(ctx context.Context, model *TextModel, promptIDs []int64, candidateIDs [4]int64, limits Limits) (result [4]float64, err error) {
	if err := validateCandidateInputs(ctx, model, promptIDs, candidateIDs, limits); err != nil {
		return result, err
	}
	snapshot, err := model.Forward(ctx, promptIDs, limits)
	if err != nil {
		return result, err
	}
	defer func() {
		err = errors.Join(err, snapshot.Close())
		if err != nil {
			result = [4]float64{}
		}
	}()
	return readCandidateSnapshot(ctx, snapshot, candidateIDs)
}

// CandidateGradient runs one forward and VJP using the caller's derivative.
// Only the first of the two retained logit rows receives cotangents, and only
// the four specified vocabulary positions are nonzero. The FP64 callback values
// are rounded once to FP32; nonfinite values before/after rounding are rejected.
// All snapshots, cotangents and native gradients are closed on every return.
// Caller-owned model/rotary tensors are borrowed unchanged. The caller admits
// the full two-row FP32 seed and returned Go parameter copies separately from
// Limits.MaxCheckpointBytes; that existing limit covers activation checkpoints.
// Cancellation is checked between native operations, not inside running kernels.
func CandidateGradient(ctx context.Context, model *TextModel, promptIDs []int64, candidateIDs [4]int64, limits Limits, derivative LossGradient) (result CandidateGradientResult, err error) {
	if derivative == nil {
		return result, fmt.Errorf("%w: derivative callback is required", ErrCandidateStep)
	}
	if err := validateCandidateInputs(ctx, model, promptIDs, candidateIDs, limits); err != nil {
		return result, err
	}
	expected, err := candidateParameters(model)
	if err != nil {
		return result, err
	}
	snapshot, err := model.Forward(ctx, promptIDs, limits)
	if err != nil {
		return result, err
	}
	var seed *torch.Tensor
	var nativeGradients Gradients
	defer func() {
		err = errors.Join(err, nativeGradients.Close(), seed.Close(), snapshot.Close())
		if err != nil {
			result = CandidateGradientResult{}
		}
	}()
	result.Logits, err = readCandidateSnapshot(ctx, snapshot, candidateIDs)
	if err != nil {
		return result, err
	}
	dValues, err := derivative(result.Logits)
	if err != nil {
		return result, err
	}
	if err := ctx.Err(); err != nil {
		return result, err
	}
	var finiteSeed [4]float32
	for i, value := range dValues {
		finiteSeed[i] = float32(value)
		if math.IsNaN(value) || math.IsInf(value, 0) || math.IsInf(float64(finiteSeed[i]), 0) {
			return result, fmt.Errorf("%w: candidate derivative is not representable as finite FP32", ErrCandidateStep)
		}
	}
	info, err := snapshot.Logits.Info()
	if err != nil {
		return result, err
	}
	if info.Elements > int64(int(^uint(0)>>1))/4 {
		return result, fmt.Errorf("%w: full-logit cotangent exceeds addressable storage", ErrCandidateStep)
	}
	fullSeed := make([]float32, int(info.Elements))
	for i, id := range candidateIDs {
		fullSeed[int(id)] = finiteSeed[i]
	}
	seed, err = torch.FromFloat32(fullSeed, info.Shape, info.Device, false)
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
		return result, fmt.Errorf("%w: incomplete parameter gradient set", ErrCandidateStep)
	}
	for i, gradient := range nativeGradients {
		if err := ctx.Err(); err != nil {
			return result, err
		}
		info, err := gradient.Value.Info()
		if err != nil {
			return result, err
		}
		if gradient.Name != expected[i].name || info.DType != torch.Float32 || info.RequiresGrad || info.Device != expected[i].info.Device || !sameShape(info.Shape, expected[i].info.Shape) {
			return result, fmt.Errorf("%w: gradient name, geometry, precision or placement differs", ErrCandidateStep)
		}
		values, err := gradient.Value.Float32Values()
		if err != nil {
			return result, err
		}
		for _, value := range values {
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				return result, fmt.Errorf("%w: nonfinite parameter gradient", ErrCandidateStep)
			}
		}
		result.Gradients = append(result.Gradients, CandidateParameterGradient{Name: gradient.Name, Shape: info.Shape, ValuesF32: values})
	}
	if err := ctx.Err(); err != nil {
		return result, err
	}
	return result, nil
}

func validateCandidateInputs(ctx context.Context, model *TextModel, promptIDs []int64, candidates [4]int64, limits Limits) error {
	if ctx == nil || len(promptIDs) < 2 || limits.LogitRows != 2 || promptIDs[len(promptIDs)-1] != candidates[0] {
		return fmt.Errorf("%w: explicit prompt-plus-A and exactly two logit rows required", ErrCandidateStep)
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	for i, id := range candidates {
		if id < 0 {
			return fmt.Errorf("%w: negative candidate ID", ErrCandidateStep)
		}
		for j := 0; j < i; j++ {
			if id == candidates[j] {
				return fmt.Errorf("%w: candidate IDs must be distinct", ErrCandidateStep)
			}
		}
	}
	info, err := model.validate(ctx, promptIDs, limits)
	if err != nil {
		return err
	}
	for _, id := range candidates {
		if id >= info.Shape[0] {
			return fmt.Errorf("%w: candidate outside vocabulary", ErrCandidateStep)
		}
	}
	return nil
}

func readCandidateSnapshot(ctx context.Context, snapshot *Snapshot, candidates [4]int64) (result [4]float64, err error) {
	if err := ctx.Err(); err != nil {
		return result, err
	}
	info, err := snapshot.Logits.Info()
	if err != nil {
		return result, err
	}
	if len(info.Shape) != 3 || info.Shape[0] != 1 || info.Shape[1] != 2 || info.Shape[2] <= 0 || info.DType != torch.Float32 {
		return result, fmt.Errorf("%w: expected [1,2,vocabulary] FP32 logits", ErrCandidateStep)
	}
	var ids, row, selected *torch.Tensor
	defer func() {
		err = errors.Join(err, selected.Close(), row.Close(), ids.Close())
		if err != nil {
			result = [4]float64{}
		}
	}()
	var raw [32]byte
	for i, id := range candidates {
		binary.LittleEndian.PutUint64(raw[i*8:], uint64(id))
	}
	ids, err = torch.FromBytes(raw[:], []int64{4}, torch.Int64, info.Device, false)
	if err != nil {
		return result, err
	}
	row, err = snapshot.Logits.Select(1, 0)
	if err != nil {
		return result, err
	}
	selected, err = row.IndexSelect(1, ids)
	if err != nil {
		return result, err
	}
	values, err := selected.Float32Values()
	if err != nil {
		return result, err
	}
	if len(values) != 4 {
		return result, fmt.Errorf("%w: wrong candidate logit count", ErrCandidateStep)
	}
	for i, value := range values {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return result, fmt.Errorf("%w: nonfinite candidate logit", ErrCandidateStep)
		}
		result[i] = float64(value)
	}
	return result, ctx.Err()
}

type candidateParameter struct {
	name string
	info torch.Info
}

func candidateParameters(model *TextModel) ([]candidateParameter, error) {
	var result []candidateParameter
	for _, slot := range adapterRegistry(model) {
		info, err := slot.value.Info()
		if err != nil {
			return nil, err
		}
		if !info.RequiresGrad || info.DType != torch.Float32 || info.Elements <= 0 {
			return nil, fmt.Errorf("%w: expected trainable FP32 adapter", ErrCandidateStep)
		}
		result = append(result, candidateParameter{name: slot.name, info: info})
	}
	if len(result) == 0 {
		return nil, fmt.Errorf("%w: trainable adapters required", ErrCandidateStep)
	}
	return result, nil
}
