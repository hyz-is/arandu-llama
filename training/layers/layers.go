// Package layers implements differentiable model calculations in Go using
// native tensor primitives. It does not load models or qualify a trainer.
package layers

import (
	"errors"
	"math"
	"runtime"

	"github.com/tayi-ai/arandu-llama/training/torch"
)

// scope releases intermediate handles. Native autograd owns the saved values
// required by the returned tensor; no Go pointer is retained by these operations.
type scope struct {
	tensors []*torch.Tensor
	err     error
}

func (s *scope) run(operation func() (*torch.Tensor, error)) *torch.Tensor {
	if s.err != nil {
		return nil
	}
	value, err := operation()
	if err != nil {
		s.err = err
		return nil
	}
	s.tensors = append(s.tensors, value)
	return value
}

func (s *scope) result(value *torch.Tensor) (*torch.Tensor, error) {
	if s.err != nil {
		return nil, s.err
	}
	for index, tensor := range s.tensors {
		if tensor == value {
			s.tensors[index] = nil
			return value, nil
		}
	}
	return nil, errors.New("layers: missing result ownership")
}

func (s *scope) close() {
	for index := len(s.tensors) - 1; index >= 0; index-- {
		if s.tensors[index] != nil {
			s.tensors[index].Close()
		}
	}
}

// Linear computes x*transpose(weight) without a bias. Weight has shape
// [output,input]; the last dimension of x is input. Inputs remain caller-owned.
func Linear(x, weight *torch.Tensor) (*torch.Tensor, error) {
	if x == nil || weight == nil {
		return nil, errors.New("layers: linear requires input and weight")
	}
	xInfo, err := x.Info()
	if err != nil {
		return nil, err
	}
	wInfo, err := weight.Info()
	if err != nil {
		return nil, err
	}
	if len(xInfo.Shape) == 0 || len(wInfo.Shape) != 2 || xInfo.Shape[len(xInfo.Shape)-1] != wInfo.Shape[1] {
		return nil, errors.New("layers: linear shape mismatch")
	}
	var s scope
	defer s.close()
	transposed := s.run(func() (*torch.Tensor, error) { return weight.Transpose(0, 1) })
	result := s.run(func() (*torch.Tensor, error) { return x.MatMul(transposed) })
	return s.result(result)
}

// LoRALinear computes x*W^T + (alpha/rank)*(x*A^T)*B^T. The base weight
// must be frozen by its owner. The same input gradient traverses both branches;
// no detach or implicit precision conversion occurs here.
func LoRALinear(x, weight, a, b *torch.Tensor, alpha float64) (*torch.Tensor, error) {
	if x == nil || weight == nil || a == nil || b == nil {
		return nil, errors.New("layers: LoRA requires input, base, A and B")
	}
	if !positiveFinite(alpha) {
		return nil, errors.New("layers: LoRA alpha must be finite and positive")
	}
	aInfo, err := a.Info()
	if err != nil {
		return nil, err
	}
	bInfo, err := b.Info()
	if err != nil {
		return nil, err
	}
	wInfo, err := weight.Info()
	if err != nil {
		return nil, err
	}
	if wInfo.RequiresGrad {
		return nil, errors.New("layers: LoRA base weight must be frozen")
	}
	if len(aInfo.Shape) != 2 || len(bInfo.Shape) != 2 || len(wInfo.Shape) != 2 ||
		aInfo.Shape[0] <= 0 || bInfo.Shape[1] != aInfo.Shape[0] ||
		wInfo.Shape[1] != aInfo.Shape[1] || wInfo.Shape[0] != bInfo.Shape[0] {
		return nil, errors.New("layers: LoRA shape mismatch")
	}
	var s scope
	defer s.close()
	base := s.run(func() (*torch.Tensor, error) { return Linear(x, weight) })
	hidden := s.run(func() (*torch.Tensor, error) { return Linear(x, a) })
	delta := s.run(func() (*torch.Tensor, error) { return Linear(hidden, b) })
	delta = s.run(func() (*torch.Tensor, error) { return delta.Scale(alpha / float64(aInfo.Shape[0])) })
	result := s.run(func() (*torch.Tensor, error) { return base.Add(delta) })
	return s.result(result)
}

// RMSNorm implements the unconverted hybrid decoder checkpoint convention: learned
// weight is an offset, so the normalized value is multiplied by (1 + weight).
// Reductions use Float32 and the result returns to the input dtype.
func RMSNorm(x, weight *torch.Tensor, epsilon float64) (*torch.Tensor, error) {
	return normalize(x, weight, nil, epsilon)
}

// GatedRMSNorm uses the learned weight directly and multiplies by SiLU(gate).
// All three inputs must be Float32, matching the admitted attention precision.
// Other dtypes are refused because intermediate rounding changes the result.
func GatedRMSNorm(x, weight, gate *torch.Tensor, epsilon float64) (*torch.Tensor, error) {
	for _, input := range []*torch.Tensor{x, weight, gate} {
		if input == nil {
			return nil, errors.New("layers: gated normalization requires input, weight and gate")
		}
		info, err := input.Info()
		if err != nil {
			return nil, err
		}
		if info.DType != torch.Float32 {
			return nil, errors.New("layers: gated normalization requires Float32 attention inputs")
		}
	}
	return normalize(x, weight, gate, epsilon)
}

func normalize(x, weight, gate *torch.Tensor, epsilon float64) (*torch.Tensor, error) {
	if x == nil || weight == nil || !positiveFinite(epsilon) || !positiveFinite(float64(float32(epsilon))) {
		return nil, errors.New("layers: normalization requires tensors and positive finite epsilon")
	}
	xInfo, err := x.Info()
	if err != nil {
		return nil, err
	}
	wInfo, err := weight.Info()
	if err != nil {
		return nil, err
	}
	if len(xInfo.Shape) == 0 || len(wInfo.Shape) != 1 || xInfo.Shape[len(xInfo.Shape)-1] != wInfo.Shape[0] {
		return nil, errors.New("layers: normalization shape mismatch")
	}
	if gate != nil {
		gateInfo, gateErr := gate.Info()
		if gateErr != nil {
			return nil, gateErr
		}
		if !equalShape(xInfo.Shape, gateInfo.Shape) {
			return nil, errors.New("layers: normalization gate shape mismatch")
		}
	}
	var s scope
	defer s.close()
	value := s.run(func() (*torch.Tensor, error) { return x.To(xInfo.Device, torch.Float32) })
	square := s.run(func() (*torch.Tensor, error) { return value.Mul(value) })
	mean := s.run(func() (*torch.Tensor, error) { return square.Mean([]int64{-1}, true) })
	variance := s.run(func() (*torch.Tensor, error) { return mean.AddScalar(float64(float32(epsilon))) })
	inverse := s.run(func() (*torch.Tensor, error) { return variance.RSqrt() })
	unit := s.run(func() (*torch.Tensor, error) { return value.Mul(inverse) })
	scale := s.run(func() (*torch.Tensor, error) { return weight.To(xInfo.Device, torch.Float32) })
	if gate == nil {
		scale = s.run(func() (*torch.Tensor, error) { return scale.AddScalar(1) })
	}
	result := s.run(func() (*torch.Tensor, error) { return unit.Mul(scale) })
	if gate != nil {
		gateValue := s.run(func() (*torch.Tensor, error) { return gate.To(xInfo.Device, torch.Float32) })
		gateValue = s.run(func() (*torch.Tensor, error) { return gateValue.SiLU() })
		result = s.run(func() (*torch.Tensor, error) { return result.Mul(gateValue) })
	}
	result = s.run(func() (*torch.Tensor, error) { return result.To(xInfo.Device, xInfo.DType) })
	return s.result(result)
}

// FeedForward computes the bias-free SwiGLU block used by the text model:
// down(SiLU(gate(x)) * up(x)). Parameters and precision remain caller-owned.
func FeedForward(x, gateWeight, upWeight, downWeight *torch.Tensor) (*torch.Tensor, error) {
	var s scope
	defer s.close()
	gate := s.run(func() (*torch.Tensor, error) { return Linear(x, gateWeight) })
	gate = s.run(func() (*torch.Tensor, error) { return gate.SiLU() })
	up := s.run(func() (*torch.Tensor, error) { return Linear(x, upWeight) })
	product := s.run(func() (*torch.Tensor, error) { return gate.Mul(up) })
	result := s.run(func() (*torch.Tensor, error) { return Linear(product, downWeight) })
	return s.result(result)
}

// FeedForwardPromoted computes SwiGLU in Float32 while leaving frozen Float16
// checkpoint weights unchanged. hybrid decoder checkpoints are BFloat16; this keeps
// their exponent range on Turing devices, where native BFloat16 execution is
// unavailable, without changing the admitted Float16 residual checkpoints.
func FeedForwardPromoted(x, gateWeight, upWeight, downWeight *torch.Tensor) (*torch.Tensor, error) {
	return feedForwardPromoted(x, gateWeight, upWeight, downWeight, nil, 0)
}

// feedForwardPromoted adds the optional adapter's pairs to the promoted
// Float32 projections. A nil adapter runs exactly FeedForwardPromoted.
func feedForwardPromoted(x, gateWeight, upWeight, downWeight *torch.Tensor, adapter *FeedForwardLoRA, alpha float64) (*torch.Tensor, error) {
	if x == nil || gateWeight == nil || upWeight == nil || downWeight == nil {
		return nil, errors.New("layers: promoted feed-forward requires input and weights")
	}
	xInfo, err := x.Info()
	if err != nil {
		return nil, err
	}
	if xInfo.DType != torch.Float16 && xInfo.DType != torch.Float32 {
		return nil, errors.New("layers: promoted feed-forward requires Float16 or Float32 input")
	}
	var s scope
	defer s.close()
	promote := func(value *torch.Tensor) *torch.Tensor {
		if s.err != nil {
			return nil
		}
		info, infoErr := value.Info()
		if infoErr != nil {
			s.err = infoErr
			return nil
		}
		if info.Device != xInfo.Device || info.RequiresGrad || (info.DType != torch.Float16 && info.DType != torch.Float32) {
			s.err = errors.New("layers: promoted feed-forward weights must be frozen floating-point tensors on the input device")
			return nil
		}
		return s.run(func() (*torch.Tensor, error) { return value.To(xInfo.Device, torch.Float32) })
	}
	var pairs FeedForwardLoRA
	if adapter != nil {
		pairs = *adapter
	}
	input := s.run(func() (*torch.Tensor, error) { return x.To(xInfo.Device, torch.Float32) })
	gateBase, upBase, downBase := promote(gateWeight), promote(upWeight), promote(downWeight)
	gate := s.run(func() (*torch.Tensor, error) { return project(input, gateBase, pairs.GateA, pairs.GateB, alpha) })
	gate = s.run(func() (*torch.Tensor, error) { return gate.SiLU() })
	up := s.run(func() (*torch.Tensor, error) { return project(input, upBase, pairs.UpA, pairs.UpB, alpha) })
	product := s.run(func() (*torch.Tensor, error) { return gate.Mul(up) })
	result := s.run(func() (*torch.Tensor, error) { return project(product, downBase, pairs.DownA, pairs.DownB, alpha) })
	return s.result(result)
}

// lockGraphThread keeps a graph and its backward on one OS thread until the
// returned release runs. LibTorch numbers autograd nodes with a per-thread
// counter and accumulates gradients in that order; a goroutine that moves
// between threads across native calls would reorder the sums run to run by a
// few ULPs. Locks nest, so a caller may already hold one.
func lockGraphThread() func() {
	runtime.LockOSThread()
	return runtime.UnlockOSThread
}

func positiveFinite(value float64) bool {
	return value > 0 && !math.IsInf(value, 0) && !math.IsNaN(value)
}

func equalShape(a, b []int64) bool {
	if len(a) != len(b) {
		return false
	}
	for index := range a {
		if a[index] != b[index] {
			return false
		}
	}
	return true
}

// ValidateAttentionGeometry checks the dimensions before constructing an
// attention graph, including the interleaved query/gate projection width.
func ValidateAttentionGeometry(hidden, queryHeads, kvHeads, headDimension, queryProjection int64) error {
	if hidden <= 0 || queryHeads <= 0 || kvHeads <= 0 || headDimension <= 0 || queryHeads%kvHeads != 0 {
		return errors.New("layers: invalid attention geometry")
	}
	if queryHeads > math.MaxInt64/headDimension/2 || queryProjection != 2*queryHeads*headDimension {
		return errors.New("layers: query projection must contain interleaved query and gate heads")
	}
	return nil
}
