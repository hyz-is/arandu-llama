package layers

import (
	"errors"
	"math"

	"github.com/tayi-ai/arandu-llama/training/torch"
)

// AttentionWeights contains the unconverted checkpoint tensors for a gated
// hybrid decoder full-attention block. Every base tensor must remain frozen.
type AttentionWeights struct {
	Query, Key, Value, Output *torch.Tensor
	QueryNorm, KeyNorm        *torch.Tensor
}

// AttentionLoRA supplies every trainable pair of one decoder block: the
// full-attention projections directly, a recurrent block's projections in
// Linear and the MLP in FeedForward. Query and Value alone are the q/v
// adapter. A nil pair leaves its projection frozen, one nil tensor of a pair
// is refused, and every pair is scaled by Alpha/rank with rank from its A. A
// nil adapter means a base-only forward, not a training qualification.
type AttentionLoRA struct {
	QueryA, QueryB, ValueA, ValueB *torch.Tensor
	Alpha                          float64
	KeyA, KeyB, OutputA, OutputB   *torch.Tensor
	Linear                         LinearAttentionLoRA
	FeedForward                    FeedForwardLoRA
}

// AttentionConfig describes grouped query attention and its explicit budget.
// MaxScoreElements covers B*Heads*T*T scores, not total model memory. Rotary
// inputs must use positions and frequencies from the admitted checkpoint.
type AttentionConfig struct {
	Heads, KVHeads, HeadDimension, RotaryDimension int64
	Epsilon                                        float64
	MaxScoreElements                               int64
}

// FullAttention computes a causal, bias-free, dropout-free gated attention block
// in Go. x is [batch,tokens,hidden] Float32. Cosine and sine are
// [tokens,RotaryDimension/2] Float32 and shared by text query/key heads. The
// output is Float32; the decoder owns the conversion back to storage precision.
// All arguments remain caller-owned and all graph derivatives remain intact.
func FullAttention(x *torch.Tensor, weights AttentionWeights, adapter *AttentionLoRA, cosine, sine *torch.Tensor, config AttentionConfig) (*torch.Tensor, error) {
	if x == nil || cosine == nil || sine == nil {
		return nil, errors.New("layers: attention requires input and rotary positions")
	}
	xInfo, err := x.Info()
	if err != nil {
		return nil, err
	}
	if len(xInfo.Shape) != 3 || xInfo.DType != torch.Float32 || xInfo.Shape[0] <= 0 || xInfo.Shape[1] <= 0 {
		return nil, errors.New("layers: attention input must be nonempty [batch,tokens,hidden] Float32")
	}
	batch, tokens, hidden := xInfo.Shape[0], xInfo.Shape[1], xInfo.Shape[2]
	if config.Heads <= 0 || config.KVHeads <= 0 || config.Heads%config.KVHeads != 0 ||
		config.HeadDimension <= 0 || config.RotaryDimension <= 0 || config.RotaryDimension%2 != 0 ||
		config.RotaryDimension > config.HeadDimension || !positiveFinite(config.Epsilon) || config.MaxScoreElements <= 0 {
		return nil, errors.New("layers: invalid full-attention configuration")
	}
	if config.Heads > math.MaxInt64/config.HeadDimension/2 || config.KVHeads > math.MaxInt64/config.HeadDimension {
		return nil, errors.New("layers: full-attention dimensions overflow")
	}
	scoreElements := batch
	for _, dimension := range []int64{config.Heads, tokens, tokens} {
		if scoreElements > config.MaxScoreElements/dimension {
			return nil, errors.New("layers: attention score budget exceeded")
		}
		scoreElements *= dimension
	}
	for _, expected := range []struct {
		tensor *torch.Tensor
		shape  []int64
	}{
		{weights.Query, []int64{2 * config.Heads * config.HeadDimension, hidden}},
		{weights.Key, []int64{config.KVHeads * config.HeadDimension, hidden}},
		{weights.Value, []int64{config.KVHeads * config.HeadDimension, hidden}},
		{weights.Output, []int64{hidden, config.Heads * config.HeadDimension}},
		{weights.QueryNorm, []int64{config.HeadDimension}},
		{weights.KeyNorm, []int64{config.HeadDimension}},
		{cosine, []int64{tokens, config.RotaryDimension / 2}},
		{sine, []int64{tokens, config.RotaryDimension / 2}},
	} {
		if expected.tensor == nil {
			return nil, errors.New("layers: missing attention tensor")
		}
		info, infoErr := expected.tensor.Info()
		if infoErr != nil {
			return nil, infoErr
		}
		if !equalShape(info.Shape, expected.shape) || info.DType != torch.Float32 || info.Device != xInfo.Device || info.RequiresGrad {
			return nil, errors.New("layers: attention base and positions must have matching frozen Float32 geometry and device")
		}
	}
	var s scope
	defer s.close()
	projection := func(input, weight *torch.Tensor, target LoRATarget) *torch.Tensor {
		if adapter == nil {
			return s.run(func() (*torch.Tensor, error) { return Linear(input, weight) })
		}
		a, b := adapter.Pair(target)
		return s.run(func() (*torch.Tensor, error) { return project(input, weight, *a, *b, adapter.Alpha) })
	}
	query := projection(x, weights.Query, LoRAQuery)
	value := projection(x, weights.Value, LoRAValue)
	key := projection(x, weights.Key, LoRAKey)
	query = s.run(func() (*torch.Tensor, error) {
		return query.Reshape([]int64{batch, tokens, config.Heads, 2 * config.HeadDimension})
	})
	gate := s.run(func() (*torch.Tensor, error) { return query.Slice(-1, config.HeadDimension, 2*config.HeadDimension, 1) })
	query = s.run(func() (*torch.Tensor, error) { return query.Slice(-1, 0, config.HeadDimension, 1) })
	key = s.run(func() (*torch.Tensor, error) {
		return key.Reshape([]int64{batch, tokens, config.KVHeads, config.HeadDimension})
	})
	value = s.run(func() (*torch.Tensor, error) {
		return value.Reshape([]int64{batch, tokens, config.KVHeads, config.HeadDimension})
	})
	query = s.run(func() (*torch.Tensor, error) { return RMSNorm(query, weights.QueryNorm, config.Epsilon) })
	key = s.run(func() (*torch.Tensor, error) { return RMSNorm(key, weights.KeyNorm, config.Epsilon) })
	cosineView := s.run(func() (*torch.Tensor, error) {
		return cosine.Reshape([]int64{1, tokens, 1, config.RotaryDimension / 2})
	})
	sineView := s.run(func() (*torch.Tensor, error) { return sine.Reshape([]int64{1, tokens, 1, config.RotaryDimension / 2}) })
	query = rotary(&s, query, cosineView, sineView, config.RotaryDimension, config.HeadDimension)
	key = rotary(&s, key, cosineView, sineView, config.RotaryDimension, config.HeadDimension)
	key = repeatHeads(&s, key, config.KVHeads, config.Heads/config.KVHeads)
	value = repeatHeads(&s, value, config.KVHeads, config.Heads/config.KVHeads)
	query = s.run(func() (*torch.Tensor, error) { return query.Transpose(1, 2) })
	key = s.run(func() (*torch.Tensor, error) { return key.Transpose(1, 2) })
	key = s.run(func() (*torch.Tensor, error) { return key.Transpose(2, 3) })
	value = s.run(func() (*torch.Tensor, error) { return value.Transpose(1, 2) })
	scores := s.run(func() (*torch.Tensor, error) { return query.MatMul(key) })
	scores = s.run(func() (*torch.Tensor, error) { return scores.Scale(1 / math.Sqrt(float64(config.HeadDimension))) })
	if s.err != nil {
		return nil, s.err
	}
	maskData := make([]byte, tokens*tokens)
	for row := int64(0); row < tokens; row++ {
		for column := row + 1; column < tokens; column++ {
			maskData[row*tokens+column] = 1
		}
	}
	mask := s.run(func() (*torch.Tensor, error) {
		return torch.FromBytes(maskData, []int64{tokens, tokens}, torch.Bool, xInfo.Device, false)
	})
	scores = s.run(func() (*torch.Tensor, error) { return scores.MaskedFill(mask, math.Inf(-1)) })
	probabilities := s.run(func() (*torch.Tensor, error) { return scores.Softmax(-1) })
	result := s.run(func() (*torch.Tensor, error) { return probabilities.MatMul(value) })
	result = s.run(func() (*torch.Tensor, error) { return result.Transpose(1, 2) })
	gate = s.run(func() (*torch.Tensor, error) { return gate.Sigmoid() })
	result = s.run(func() (*torch.Tensor, error) { return result.Mul(gate) })
	result = s.run(func() (*torch.Tensor, error) {
		return result.Reshape([]int64{batch, tokens, config.Heads * config.HeadDimension})
	})
	result = projection(result, weights.Output, LoRAOutput)
	return s.result(result)
}

func rotary(s *scope, value, cosine, sine *torch.Tensor, dimension, headDimension int64) *torch.Tensor {
	first := s.run(func() (*torch.Tensor, error) { return value.Slice(-1, 0, dimension/2, 1) })
	second := s.run(func() (*torch.Tensor, error) { return value.Slice(-1, dimension/2, dimension, 1) })
	fc := s.run(func() (*torch.Tensor, error) { return first.Mul(cosine) })
	ss := s.run(func() (*torch.Tensor, error) { return second.Mul(sine) })
	sc := s.run(func() (*torch.Tensor, error) { return second.Mul(cosine) })
	fs := s.run(func() (*torch.Tensor, error) { return first.Mul(sine) })
	a := s.run(func() (*torch.Tensor, error) { return fc.Sub(ss) })
	b := s.run(func() (*torch.Tensor, error) { return sc.Add(fs) })
	parts := []*torch.Tensor{a, b}
	if dimension < headDimension {
		tail := s.run(func() (*torch.Tensor, error) { return value.Slice(-1, dimension, headDimension, 1) })
		parts = append(parts, tail)
	}
	return s.run(func() (*torch.Tensor, error) { return torch.Cat(parts, -1) })
}

func repeatHeads(s *scope, value *torch.Tensor, heads, repeats int64) *torch.Tensor {
	parts := make([]*torch.Tensor, 0, heads*repeats)
	for head := int64(0); head < heads; head++ {
		part := s.run(func() (*torch.Tensor, error) { return value.Select(2, head) })
		for repeat := int64(0); repeat < repeats; repeat++ {
			parts = append(parts, part)
		}
	}
	return s.run(func() (*torch.Tensor, error) { return torch.Stack(parts, 2) })
}
