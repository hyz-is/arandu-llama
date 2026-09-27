package layers

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"os"

	"github.com/tayi-ai/arandu-llama/training/sequence"
	"github.com/tayi-ai/arandu-llama/training/tensor"
	"github.com/tayi-ai/arandu-llama/training/torch"
)

// LinearAttentionWeights contains unconverted, frozen hybrid decoder checkpoint
// tensors. Projections use [out,in]. Convolution is [2*Hk*K+Hv*V,1,4],
// ALog and DTBias are [Hv], and Norm is [V] (a direct multiplier, not 1+w).
type LinearAttentionWeights struct {
	QKV, Z, Beta, Alpha, Convolution *torch.Tensor
	ALog, DTBias, Norm, Output       *torch.Tensor
}

// LinearAttentionConfig defines head geometry and explicit allocation limits.
// Epsilon belongs to gated RMS normalization; Q/K L2 normalization uses the
// HF constant 1e-6. Zero Sequence uses sequence.DefaultLimits().
// MaxWorkingElements limits a conservative projection/intermediate element
// estimate, separately from Sequence's recurrence budget. Zero selects 256Mi
// elements. Neither budget measures private native allocator workspaces.
type LinearAttentionConfig struct {
	KeyHeads, ValueHeads, KeyDimension, ValueDimension int64
	Epsilon                                            float64
	MaxWorkingElements                                 int64
	Sequence                                           sequence.SequenceLimits
	// FrozenWeightsValidated records that the assembly loader already checked
	// every immutable weight for finiteness and content identity. Runtime input
	// and cotangent checks remain enabled on every call.
	FrozenWeightsValidated bool
}

type linearAttentionGeometry struct {
	batch, tokens, hidden, keys, values, channels, stateElements int64
	device                                                       torch.Device
}

// ForwardLinearAttention computes the complete causal, bias-free hybrid decoder
// linear-attention block in FP32, starting from zero convolution and GDN state.
// x is [B,T,D]; all arguments are borrowed and immutable. The caller owns the
// detached result. This block has no LoRA weights and no model-training claim.
func ForwardLinearAttention(ctx context.Context, x *torch.Tensor, weights LinearAttentionWeights, config LinearAttentionConfig) (*torch.Tensor, error) {
	return forwardLinearAttention(ctx, x, weights, linearPairs{}, config)
}

// linearPairs carries a recurrent block's optional adapter and its scale.
// The zero value is the frozen block.
type linearPairs struct {
	lora  LinearAttentionLoRA
	alpha float64
}

// projection returns the block's pair for a recurrent target.
func (p linearPairs) projection(target LoRATarget) (a, b *torch.Tensor) {
	if !target.LinearAttention() {
		return nil, nil
	}
	x, y := pairOf(target, [4][2]**torch.Tensor{}, &p.lora, &FeedForwardLoRA{})
	return *x, *y
}

func forwardLinearAttention(ctx context.Context, x *torch.Tensor, weights LinearAttentionWeights, pairs linearPairs, config LinearAttentionConfig) (*torch.Tensor, error) {
	geometry, config, err := validateLinearAttention(ctx, x, weights, pairs, nil, config)
	if err != nil {
		return nil, err
	}
	s := linearAttentionScope{ctx: ctx}
	defer s.close()
	input := s.run(func() (*torch.Tensor, error) { return x.Detach() })
	recurrence, gate := projectLinearAttention(&s, input, weights, pairs, geometry, config)
	if s.err != nil {
		return nil, s.err
	}
	output, err := sequence.Forward(ctx, recurrence, config.Sequence)
	if err != nil {
		return nil, err
	}
	defer output.Close()
	result := finishLinearAttention(&s, output.Values, gate, weights, pairs, geometry, config)
	result = s.run(func() (*torch.Tensor, error) { return result.Detach() })
	return s.result(result)
}

// LinearAttentionVJP returns the complete detached input cotangent for dy.
// It recomputes projections, invokes checkpointed recurrence VJP, and joins
// the recurrence and output-gate paths. Frozen parameters receive no gradients.
// No caller gradient flag, storage, or autograd graph is changed. Cancellation
// is checked between native operations; an already-running kernel cannot be
// preempted through this API.
func LinearAttentionVJP(ctx context.Context, x *torch.Tensor, weights LinearAttentionWeights, dy *torch.Tensor, config LinearAttentionConfig) (*torch.Tensor, error) {
	input, _, err := linearAttentionVJP(ctx, x, weights, linearPairs{}, dy, config)
	return input, err
}

// linearAttentionVJP also returns the cotangent of every adapted pair, laid
// out like the adapter. The caller owns every returned handle.
func linearAttentionVJP(ctx context.Context, x *torch.Tensor, weights LinearAttentionWeights, pairs linearPairs, dy *torch.Tensor, config LinearAttentionConfig) (*torch.Tensor, LinearAttentionLoRA, error) {
	var adapted LinearAttentionLoRA
	if dy == nil {
		return nil, adapted, errors.New("layers: linear attention VJP requires a cotangent")
	}
	geometry, config, err := validateLinearAttention(ctx, x, weights, pairs, dy, config)
	if err != nil {
		return nil, adapted, err
	}
	s := linearAttentionScope{ctx: ctx}
	defer s.close()
	input := s.run(func() (*torch.Tensor, error) { return x.Detach() })
	input = s.run(func() (*torch.Tensor, error) { return input.SetRequiresGrad(true) })
	recurrence, gate := projectLinearAttention(&s, input, weights, pairs, geometry, config)
	if s.err != nil {
		return nil, adapted, s.err
	}
	forward, err := sequence.Forward(ctx, recurrence, config.Sequence)
	if err != nil {
		return nil, adapted, err
	}
	defer forward.Close()
	outerScope := linearAttentionScope{ctx: ctx}
	defer outerScope.close()
	core := outerScope.run(func() (*torch.Tensor, error) { return forward.Values.Detach() })
	core = outerScope.run(func() (*torch.Tensor, error) { return core.SetRequiresGrad(true) })
	result := finishLinearAttention(&outerScope, core, gate, weights, pairs, geometry, config)
	// The Z/gate branch and recurrence projections share only the input leaf.
	// Release the output graph after its VJP; the projection graph remains
	// available for the recurrence cotangents below. The Z and output pairs
	// reach the result only through this graph; the others only through the
	// recurrence.
	outerTargets := pairs.present(LoRALinearOutput, LoRALinearZ)
	outer := outerScope.grad([]*torch.Tensor{result}, pairs.inputs([]*torch.Tensor{core, input}, outerTargets), []*torch.Tensor{dy}, false)
	if outerScope.err != nil {
		return nil, adapted, outerScope.err
	}
	for _, gradient := range outer {
		owned, err := outerScope.scope.result(gradient)
		if err != nil {
			return nil, adapted, err
		}
		s.tensors = append(s.tensors, owned)
	}
	// Keep only the cotangents. In particular, core aliases must close
	// before the first forward's Values storage can be released.
	outerScope.close()
	_ = forward.Close()
	backward, err := sequence.GradVJP(ctx, recurrence, outer[0], recurrence.InitialState, config.Sequence)
	if err != nil {
		return nil, adapted, err
	}
	defer backward.Close()
	// This caller consumes only adjoints, not the recomputed forward result.
	_ = backward.Output.Close()
	g := backward.Gradients
	innerTargets := pairs.present(LoRALinearQKV, LoRALinearBeta, LoRALinearAlpha)
	inner := s.grad(
		[]*torch.Tensor{recurrence.Query, recurrence.Key, recurrence.Value, recurrence.LogDecay, recurrence.Beta},
		pairs.inputs([]*torch.Tensor{input}, innerTargets), []*torch.Tensor{g.Query, g.Key, g.Value, g.LogDecay, g.Beta}, false)
	if s.err != nil {
		return nil, adapted, s.err
	}
	result = s.run(func() (*torch.Tensor, error) { return inner[0].Add(outer[1]) })
	result = s.run(func() (*torch.Tensor, error) { return result.Detach() })
	if s.err != nil {
		return nil, adapted, s.err
	}
	// Pair cotangents leave the scope first; the input result then performs
	// the final cancellation check, and a failure there releases them again.
	for _, group := range []struct {
		targets   []LoRATarget
		gradients []*torch.Tensor
	}{{outerTargets, outer[2:]}, {innerTargets, inner[1:]}} {
		for i, target := range group.targets {
			a, b := pairOf(target, [4][2]**torch.Tensor{}, &adapted, &FeedForwardLoRA{})
			*a, _ = s.scope.result(group.gradients[2*i])
			*b, _ = s.scope.result(group.gradients[2*i+1])
		}
	}
	inputGradient, err := s.result(result)
	if err != nil {
		for _, target := range LoRATargets() {
			if target.LinearAttention() {
				a, b := pairOf(target, [4][2]**torch.Tensor{}, &adapted, &FeedForwardLoRA{})
				_, _ = (*a).Close(), (*b).Close()
			}
		}
		return nil, LinearAttentionLoRA{}, err
	}
	return inputGradient, adapted, nil
}

// present lists the given targets this block adapts, in the given order.
func (p linearPairs) present(targets ...LoRATarget) []LoRATarget {
	var result []LoRATarget
	for _, target := range targets {
		if a, _ := p.projection(target); a != nil {
			result = append(result, target)
		}
	}
	return result
}

// inputs appends the A and B leaves of targets to the differentiated inputs.
func (p linearPairs) inputs(leading []*torch.Tensor, targets []LoRATarget) []*torch.Tensor {
	result := append([]*torch.Tensor(nil), leading...)
	for _, target := range targets {
		a, b := p.projection(target)
		result = append(result, a, b)
	}
	return result
}

// Every intermediate is checked, including pre-normalization squares and
// exponentials: finite output must not hide an earlier FP32 overflow.
type linearAttentionScope struct {
	scope
	ctx context.Context
}

func (s *linearAttentionScope) check() bool {
	if s.err == nil {
		s.err = s.ctx.Err()
	}
	return s.err == nil
}

func (s *linearAttentionScope) finite(value *torch.Tensor) {
	s.finiteNamed("unlabeled", value)
}

func (s *linearAttentionScope) finiteNamed(name string, value *torch.Tensor) {
	if !s.check() {
		return
	}
	finite, err := value.AllFinite()
	if err != nil {
		s.err = err
	} else if !finite {
		s.err = errors.New("layers: nonfinite linear-attention intermediate: " + name)
	}
	s.check()
}

func (s *linearAttentionScope) run(operation func() (*torch.Tensor, error)) *torch.Tensor {
	return s.runNamed("unlabeled", operation)
}

func (s *linearAttentionScope) runNamed(name string, operation func() (*torch.Tensor, error)) *torch.Tensor {
	if !s.check() {
		return nil
	}
	trace := os.Getenv("TAYI_TRACE_LINEAR") == "1"
	if trace {
		fmt.Fprintf(os.Stderr, "phase=linear_op_start op=%s\n", name)
	}
	value := s.scope.run(operation)
	if s.err == nil {
		s.finiteNamed(name, value)
	}
	if trace {
		fmt.Fprintf(os.Stderr, "phase=linear_op_end op=%s err=%v\n", name, s.err)
	}
	return value
}

func (s *linearAttentionScope) grad(outputs, inputs, seeds []*torch.Tensor, retain bool) []*torch.Tensor {
	if !s.check() {
		return nil
	}
	gradients, err := torch.Grad(outputs, inputs, seeds, retain, false)
	// Own every returned handle even if a later check fails.
	s.tensors = append(s.tensors, gradients...)
	if err != nil {
		s.err = err
	}
	for _, gradient := range gradients {
		s.finite(gradient)
	}
	return gradients
}

func (s *linearAttentionScope) result(value *torch.Tensor) (*torch.Tensor, error) {
	s.check()
	return s.scope.result(value)
}

func validateLinearAttention(ctx context.Context, x *torch.Tensor, weights LinearAttentionWeights, pairs linearPairs, dy *torch.Tensor, config LinearAttentionConfig) (linearAttentionGeometry, LinearAttentionConfig, error) {
	var geometry linearAttentionGeometry
	fail := func(message string) (linearAttentionGeometry, LinearAttentionConfig, error) {
		return geometry, config, errors.New("layers: " + message)
	}
	if ctx == nil || x == nil {
		return fail("linear attention requires context and input")
	}
	if err := ctx.Err(); err != nil {
		return geometry, config, err
	}
	info, err := x.Info()
	if err != nil {
		return geometry, config, err
	}
	if len(info.Shape) != 3 || info.DType != torch.Float32 || info.Shape[0] <= 0 || info.Shape[1] <= 0 || info.Shape[2] <= 0 {
		return fail("linear attention input must be nonempty [B,T,D] Float32")
	}
	if config.KeyHeads <= 0 || config.ValueHeads <= 0 || config.KeyDimension <= 0 || config.ValueDimension <= 0 || config.ValueHeads%config.KeyHeads != 0 ||
		!positiveFinite(config.Epsilon) || !positiveFinite(float64(float32(config.Epsilon))) {
		return fail("invalid linear-attention geometry or epsilon")
	}
	if config.MaxWorkingElements == 0 {
		config.MaxWorkingElements = 256 << 20
	}
	if config.MaxWorkingElements <= 0 {
		return fail("invalid linear-attention working budget")
	}
	if config.Sequence == (sequence.SequenceLimits{}) {
		config.Sequence = sequence.DefaultLimits()
	}
	if config.Sequence.ChunkTokens <= 0 || config.Sequence.ChunkTokens > tensor.MaxChunkTokens || config.Sequence.MaxTokens <= 0 ||
		config.Sequence.MaxOwnedElements <= 0 || info.Shape[1] > config.Sequence.MaxTokens {
		return fail("invalid or exceeded linear-attention sequence limits")
	}
	geometry.batch, geometry.tokens, geometry.hidden, geometry.device = info.Shape[0], info.Shape[1], info.Shape[2], info.Device
	geometry.keys, err = linearAttentionProduct(config.KeyHeads, config.KeyDimension)
	if err != nil {
		return geometry, config, err
	}
	geometry.values, err = linearAttentionProduct(config.ValueHeads, config.ValueDimension)
	if err != nil {
		return geometry, config, err
	}
	if geometry.keys > (math.MaxInt64-geometry.values)/2 {
		return fail("linear-attention channel count overflows")
	}
	geometry.channels = 2*geometry.keys + geometry.values
	geometry.stateElements, err = linearAttentionProduct(geometry.batch, config.ValueHeads, config.KeyDimension, config.ValueDimension)
	if err != nil {
		return geometry, config, err
	}
	// Conservative sum for all owned projection/gradient intermediates. State
	// recomputation is additionally bounded inside sequence; weights are borrowed.
	remaining := config.MaxWorkingElements
	for _, dimensions := range [][]int64{
		{32, geometry.batch, geometry.tokens, geometry.hidden},
		{32, geometry.batch, geometry.tokens, geometry.channels},
		{32, geometry.batch, geometry.tokens, geometry.values},
		{64, geometry.batch, geometry.tokens, config.ValueHeads, config.KeyDimension},
		{32, geometry.batch, geometry.tokens, config.ValueHeads},
		{4, geometry.stateElements},
	} {
		count, countErr := linearAttentionProduct(dimensions...)
		if countErr != nil || count > remaining {
			return fail("linear-attention working budget exceeded")
		}
		remaining -= count
	}
	for _, expected := range []struct {
		name   string
		value  *torch.Tensor
		shape  []int64
		frozen bool
	}{
		{"input", x, info.Shape, false},
		{"QKV", weights.QKV, []int64{geometry.channels, geometry.hidden}, true},
		{"Z", weights.Z, []int64{geometry.values, geometry.hidden}, true},
		{"Beta", weights.Beta, []int64{config.ValueHeads, geometry.hidden}, true},
		{"Alpha", weights.Alpha, []int64{config.ValueHeads, geometry.hidden}, true},
		{"Convolution", weights.Convolution, []int64{geometry.channels, 1, 4}, true},
		{"ALog", weights.ALog, []int64{config.ValueHeads}, true},
		{"DTBias", weights.DTBias, []int64{config.ValueHeads}, true},
		{"Norm", weights.Norm, []int64{config.ValueDimension}, true},
		{"Output", weights.Output, []int64{geometry.hidden, geometry.values}, true},
		{"cotangent", dy, info.Shape, false},
	} {
		if err := ctx.Err(); err != nil {
			return geometry, config, err
		}
		if expected.name == "cotangent" && expected.value == nil {
			continue
		}
		if expected.value == nil {
			return fail("missing linear-attention " + expected.name)
		}
		actual, infoErr := expected.value.Info()
		if infoErr != nil {
			return geometry, config, infoErr
		}
		if !equalShape(actual.Shape, expected.shape) || actual.DType != torch.Float32 || actual.Device != geometry.device || (expected.frozen && actual.RequiresGrad) {
			return fail("invalid linear-attention " + expected.name + " shape, dtype, device or frozen flag")
		}
		if !expected.frozen || !config.FrozenWeightsValidated {
			if os.Getenv("TAYI_TRACE_LINEAR") == "1" {
				fmt.Fprintf(os.Stderr, "phase=linear_validate_start tensor=%s elements=%d\n", expected.name, actual.Elements)
			}
			finite, finiteErr := expected.value.AllFinite()
			if finiteErr != nil {
				return geometry, config, finiteErr
			}
			if os.Getenv("TAYI_TRACE_LINEAR") == "1" {
				fmt.Fprintf(os.Stderr, "phase=linear_validate_end tensor=%s finite=%t\n", expected.name, finite)
			}
			if !finite {
				return fail("nonfinite linear-attention " + expected.name)
			}
		}
	}
	// Adapted pairs change every step, so their values are left to the
	// finite checks of every intermediate; only their geometry is admitted.
	projections := map[LoRATarget][2]int64{
		LoRALinearOutput: {geometry.hidden, geometry.values}, LoRALinearQKV: {geometry.channels, geometry.hidden},
		LoRALinearZ: {geometry.values, geometry.hidden}, LoRALinearBeta: {config.ValueHeads, geometry.hidden}, LoRALinearAlpha: {config.ValueHeads, geometry.hidden},
	}
	for _, target := range LoRATargets() {
		if !target.LinearAttention() {
			continue
		}
		a, b := pairs.projection(target)
		if a == nil && b == nil {
			continue
		}
		if a == nil || b == nil || !positiveFinite(pairs.alpha) {
			return fail("linear-attention LoRA requires both A and B and a finite positive alpha")
		}
		aInfo, err := a.Info()
		if err != nil {
			return geometry, config, err
		}
		bInfo, err := b.Info()
		if err != nil {
			return geometry, config, err
		}
		shape := projections[target]
		if len(aInfo.Shape) != 2 || len(bInfo.Shape) != 2 || aInfo.Shape[0] <= 0 || aInfo.Shape[1] != shape[1] || !equalShape(bInfo.Shape, []int64{shape[0], aInfo.Shape[0]}) ||
			aInfo.DType != torch.Float32 || bInfo.DType != torch.Float32 || aInfo.Device != geometry.device || bInfo.Device != geometry.device {
			return fail("invalid linear-attention LoRA shape, dtype or device")
		}
	}
	return geometry, config, ctx.Err()
}

func linearAttentionProduct(dimensions ...int64) (int64, error) {
	product := int64(1)
	for _, dimension := range dimensions {
		if dimension <= 0 || product > math.MaxInt64/dimension {
			return 0, errors.New("layers: linear-attention dimensions overflow")
		}
		product *= dimension
	}
	// A zero-state host allocation must also fit the platform's byte count.
	if uint64(product) > uint64(^uint(0)>>1)/4 {
		return 0, errors.New("layers: linear-attention byte count overflows")
	}
	return product, nil
}

func projectLinearAttention(s *linearAttentionScope, x *torch.Tensor, weights LinearAttentionWeights, pairs linearPairs, geometry linearAttentionGeometry, config LinearAttentionConfig) (tensor.Input, *torch.Tensor) {
	adapted := func(input, weight *torch.Tensor, target LoRATarget) func() (*torch.Tensor, error) {
		return func() (*torch.Tensor, error) {
			a, b := pairs.projection(target)
			return project(input, weight, a, b, pairs.alpha)
		}
	}
	mixed := s.runNamed("qkv_projection", adapted(x, weights.QKV, LoRALinearQKV))
	var convolution *torch.Tensor
	for tap := int64(0); tap < 4 && s.check(); tap++ {
		lag := 3 - tap
		if lag >= geometry.tokens {
			continue // The left padding is zero, with no trainable conv weights.
		}
		shifted := mixed
		if lag > 0 {
			prefix := s.runNamed("convolution_prefix_slice", func() (*torch.Tensor, error) { return mixed.Slice(1, 0, lag, 1) })
			prefix = s.runNamed("convolution_prefix_zero", func() (*torch.Tensor, error) { return prefix.Scale(0) })
			past := s.runNamed("convolution_past_slice", func() (*torch.Tensor, error) { return mixed.Slice(1, 0, geometry.tokens-lag, 1) })
			shifted = s.runNamed("convolution_shift", func() (*torch.Tensor, error) { return torch.Cat([]*torch.Tensor{prefix, past}, 1) })
		}
		kernel := s.runNamed("convolution_kernel_channel", func() (*torch.Tensor, error) { return weights.Convolution.Select(1, 0) })
		kernel = s.runNamed("convolution_kernel_tap", func() (*torch.Tensor, error) { return kernel.Select(1, tap) })
		term := s.runNamed("convolution_term", func() (*torch.Tensor, error) { return shifted.Mul(kernel) })
		if convolution == nil {
			convolution = term
		} else {
			convolution = s.runNamed("convolution_accumulate", func() (*torch.Tensor, error) { return convolution.Add(term) })
		}
	}
	convolution = s.runNamed("convolution_silu", func() (*torch.Tensor, error) { return convolution.SiLU() })
	query := s.runNamed("query_slice", func() (*torch.Tensor, error) { return convolution.Slice(2, 0, geometry.keys, 1) })
	key := s.runNamed("key_slice", func() (*torch.Tensor, error) { return convolution.Slice(2, geometry.keys, 2*geometry.keys, 1) })
	value := s.runNamed("value_slice", func() (*torch.Tensor, error) { return convolution.Slice(2, 2*geometry.keys, geometry.channels, 1) })
	query = s.runNamed("query_reshape", func() (*torch.Tensor, error) {
		return query.Reshape([]int64{geometry.batch, geometry.tokens, config.KeyHeads, config.KeyDimension})
	})
	key = s.runNamed("key_reshape", func() (*torch.Tensor, error) {
		return key.Reshape([]int64{geometry.batch, geometry.tokens, config.KeyHeads, config.KeyDimension})
	})
	value = s.runNamed("value_reshape", func() (*torch.Tensor, error) {
		return value.Reshape([]int64{geometry.batch, geometry.tokens, config.ValueHeads, config.ValueDimension})
	})
	epsilon := s.runNamed("normalization_epsilon", func() (*torch.Tensor, error) {
		return torch.FromFloat32([]float32{1e-6}, []int64{1}, geometry.device, false)
	})
	normalize := func(name string, input *torch.Tensor) *torch.Tensor {
		// Preserve x/sqrt(sum(x^2)+1e-6) while avoiding overflow when a
		// finite FP32 projection is too large to square directly. Scaling by
		// max(1,max(abs(x))) is algebraically cancelled by the denominator;
		// epsilon is scaled by the same reciprocal squared.
		magnitude := s.runNamed(name+"_abs", func() (*torch.Tensor, error) { return input.Abs() })
		magnitude = s.runNamed(name+"_amax", func() (*torch.Tensor, error) { return magnitude.AMax([]int64{-1}, true) })
		magnitude = s.runNamed(name+"_scale_floor", func() (*torch.Tensor, error) { return magnitude.ClampMin(1) })
		scale := s.runNamed(name+"_scale_reciprocal", func() (*torch.Tensor, error) { return magnitude.Reciprocal() })
		scaled := s.runNamed(name+"_scaled", func() (*torch.Tensor, error) { return input.Mul(scale) })
		squares := s.runNamed(name+"_scaled_square", func() (*torch.Tensor, error) { return scaled.Mul(scaled) })
		squares = s.runNamed(name+"_scaled_square_sum", func() (*torch.Tensor, error) { return squares.Sum([]int64{-1}, true) })
		scaledEpsilon := s.runNamed(name+"_epsilon_scale_square", func() (*torch.Tensor, error) { return scale.Mul(scale) })
		scaledEpsilon = s.runNamed(name+"_scaled_epsilon", func() (*torch.Tensor, error) { return scaledEpsilon.Mul(epsilon) })
		squares = s.runNamed(name+"_stable_norm_square", func() (*torch.Tensor, error) { return squares.Add(scaledEpsilon) })
		inverse := s.runNamed(name+"_stable_rsqrt", func() (*torch.Tensor, error) { return squares.RSqrt() })
		return s.runNamed(name+"_normalized", func() (*torch.Tensor, error) { return scaled.Mul(inverse) })
	}
	query, key = normalize("query", query), normalize("key", key)
	if config.ValueHeads != config.KeyHeads {
		repeatCount := config.ValueHeads / config.KeyHeads
		indexBytes := make([]byte, config.ValueHeads*8)
		for head := int64(0); head < config.KeyHeads; head++ {
			for count := int64(0); count < repeatCount; count++ {
				index := head*repeatCount + count
				binary.LittleEndian.PutUint64(indexBytes[index*8:], uint64(head))
			}
		}
		indices := s.runNamed("head_repeat_indices", func() (*torch.Tensor, error) {
			return torch.FromBytes(indexBytes, []int64{config.ValueHeads}, torch.Int64, geometry.device, false)
		})
		query = s.runNamed("query_head_repeat", func() (*torch.Tensor, error) { return query.IndexSelect(2, indices) })
		key = s.runNamed("key_head_repeat", func() (*torch.Tensor, error) { return key.IndexSelect(2, indices) })
	}
	query = s.runNamed("query_scale", func() (*torch.Tensor, error) { return query.Scale(1 / math.Sqrt(float64(config.KeyDimension))) })
	beta := s.runNamed("beta_projection", adapted(x, weights.Beta, LoRALinearBeta))
	beta = s.runNamed("beta_sigmoid", func() (*torch.Tensor, error) { return beta.Sigmoid() })
	decay := s.runNamed("decay_projection", adapted(x, weights.Alpha, LoRALinearAlpha))
	decay = s.runNamed("decay_bias", func() (*torch.Tensor, error) { return decay.Add(weights.DTBias) })
	decay = s.runNamed("decay_softplus", func() (*torch.Tensor, error) { return decay.Softplus() })
	negativeA := s.runNamed("negative_a_exp", func() (*torch.Tensor, error) { return weights.ALog.Exp() })
	negativeA = s.runNamed("negative_a_scale", func() (*torch.Tensor, error) { return negativeA.Scale(-1) })
	decay = s.runNamed("decay_log", func() (*torch.Tensor, error) { return decay.Mul(negativeA) })
	gate := s.runNamed("gate_projection", adapted(x, weights.Z, LoRALinearZ))
	gate = s.runNamed("gate_reshape", func() (*torch.Tensor, error) {
		return gate.Reshape([]int64{geometry.batch, geometry.tokens, config.ValueHeads, config.ValueDimension})
	})
	initial := s.runNamed("initial_state", func() (*torch.Tensor, error) {
		return torch.FromFloat32(make([]float32, geometry.stateElements), []int64{geometry.batch, config.ValueHeads, config.KeyDimension, config.ValueDimension}, geometry.device, false)
	})
	return tensor.Input{Query: query, Key: key, Value: value, LogDecay: decay, Beta: beta, InitialState: initial}, gate
}

func finishLinearAttention(s *linearAttentionScope, core, gate *torch.Tensor, weights LinearAttentionWeights, pairs linearPairs, geometry linearAttentionGeometry, config LinearAttentionConfig) *torch.Tensor {
	value := s.runNamed("gated_rms_norm", func() (*torch.Tensor, error) { return GatedRMSNorm(core, weights.Norm, gate, config.Epsilon) })
	value = s.runNamed("output_reshape", func() (*torch.Tensor, error) {
		return value.Reshape([]int64{geometry.batch, geometry.tokens, geometry.values})
	})
	return s.runNamed("output_projection", func() (*torch.Tensor, error) {
		a, b := pairs.projection(LoRALinearOutput)
		return project(value, weights.Output, a, b, pairs.alpha)
	})
}
