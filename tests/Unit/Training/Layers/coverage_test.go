//go:build libtorch && cgo

package layers_test

import (
	"context"
	"math"
	"testing"

	"github.com/tayi-ai/arandu-llama/training/layers"
	"github.com/tayi-ai/arandu-llama/training/torch"
)

// Declared tolerances. Forward values are FP32 against an FP64 reference;
// gradients are FP32 autograd against FP64 central differences, step 1e-5.
const (
	coverageForwardTolerance  = 4e-6
	coverageGradientTolerance = 1e-5
)

// referencePair points at one pair's reference data, so a finite difference
// perturbs exactly what the scalar reference reads.
type referencePair struct {
	target         layers.LoRATarget
	a, b           *[]float64
	aShape, bShape []int64
}

func pattern(size int, phase float64) []float64 {
	result := make([]float64, size)
	for index := range result {
		result[index] = math.Sin(float64(index)*0.8+phase) * 0.3
	}
	return result
}

// withKeyAndOutput adds the pairs the q/v fixture never had.
func withKeyAndOutput(f attentionFixture) attentionFixture {
	f.ka, f.kb, f.oa, f.ob = pattern(8, 0.15), pattern(16, 0.35), pattern(32, 0.45), pattern(8, 0.65)
	return f
}

func attentionPairs(f *attentionFixture) map[layers.LoRATarget]referencePair {
	return map[layers.LoRATarget]referencePair{
		layers.LoRAQuery:  {layers.LoRAQuery, &f.qa, &f.qb, []int64{4, 2}, []int64{16, 4}},
		layers.LoRAKey:    {layers.LoRAKey, &f.ka, &f.kb, []int64{4, 2}, []int64{4, 4}},
		layers.LoRAValue:  {layers.LoRAValue, &f.va, &f.vb, []int64{4, 2}, []int64{4, 4}},
		layers.LoRAOutput: {layers.LoRAOutput, &f.oa, &f.ob, []int64{4, 8}, []int64{2, 4}},
	}
}

// selectPairs keeps only the chosen pairs in the reference and builds the
// matching trainable adapter with the reference's alpha/rank of 8/4.
func selectPairs(t *testing.T, all map[layers.LoRATarget]referencePair, targets []layers.LoRATarget, zeroB bool) (*layers.AttentionLoRA, []referencePair) {
	t.Helper()
	adapter := &layers.AttentionLoRA{Alpha: 8}
	var chosen []referencePair
	for target, pair := range all {
		keep := false
		for _, wanted := range targets {
			keep = keep || wanted == target
		}
		if !keep {
			*pair.a, *pair.b = nil, nil
		}
	}
	for _, target := range targets {
		pair := all[target]
		if zeroB {
			clear(*pair.b)
		}
		a, b := adapter.Pair(target)
		*a, *b = tensor32(t, *pair.a, pair.aShape, true), tensor32(t, *pair.b, pair.bShape, true)
		chosen = append(chosen, pair)
	}
	return adapter, chosen
}

// finiteDifference differentiates dy·reference with respect to data.
func finiteDifference(data []float64, dy []float64, reference func() []float64) []float64 {
	result := make([]float64, len(data))
	for index := range data {
		original := data[index]
		const step = 1e-5
		data[index] = original + step
		plus := reference()
		data[index] = original - step
		minus := reference()
		data[index] = original
		for output := range plus {
			result[index] += dy[output] * (plus[output] - minus[output]) / (2 * step)
		}
	}
	return result
}

func maxError(got, expected []float64) float64 {
	worst := 0.0
	for index := range got {
		worst = math.Max(worst, math.Abs(got[index]-expected[index]))
	}
	return worst
}

func attentionConfig() layers.AttentionConfig {
	return layers.AttentionConfig{Heads: 4, KVHeads: 2, HeadDimension: 2, RotaryDimension: 2, Epsilon: 1e-6, MaxScoreElements: 36}
}

func attentionWeights(t *testing.T, f attentionFixture) layers.AttentionWeights {
	return layers.AttentionWeights{
		Query: tensor32(t, f.q, []int64{16, 2}, false), Key: tensor32(t, f.k, []int64{4, 2}, false),
		Value: tensor32(t, f.v, []int64{4, 2}, false), Output: tensor32(t, f.out, []int64{2, 8}, false),
		QueryNorm: tensor32(t, f.qnorm, []int64{2}, false), KeyNorm: tensor32(t, f.knorm, []int64{2}, false),
	}
}

func TestFullAttentionLoRAOnEveryProjectionMatchesIndependentReference(t *testing.T) {
	for _, test := range []struct {
		name    string
		targets []layers.LoRATarget
	}{
		{"k_proj", []layers.LoRATarget{layers.LoRAKey}},
		{"o_proj", []layers.LoRATarget{layers.LoRAOutput}},
		{"q_k_v_o", []layers.LoRATarget{layers.LoRAQuery, layers.LoRAKey, layers.LoRAValue, layers.LoRAOutput}},
	} {
		t.Run(test.name, func(t *testing.T) {
			f := withKeyAndOutput(fixture())
			adapter, pairs := selectPairs(t, attentionPairs(&f), test.targets, false)
			xt := tensor32(t, f.x, []int64{1, 3, 2}, true)
			cosine, sine := tensor32(t, f.cosine, []int64{3, 1}, false), tensor32(t, f.sine, []int64{3, 1}, false)
			result, err := layers.FullAttention(xt, attentionWeights(t, f), adapter, cosine, sine, attentionConfig())
			result = own(t, result, err)
			forward := maxError(values(t, result), attentionReference(f))
			near(t, values(t, result), attentionReference(f), coverageForwardTolerance)
			dy := []float64{0.3, -0.5, 0.2, 0.7, -0.1, 0.6}
			inputs := []*torch.Tensor{xt}
			data := []*[]float64{&f.x}
			for _, pair := range pairs {
				a, b := adapter.Pair(pair.target)
				inputs = append(inputs, *a, *b)
				data = append(data, pair.a, pair.b)
			}
			gradients, err := torch.Grad([]*torch.Tensor{result}, inputs, []*torch.Tensor{tensor32(t, dy, []int64{1, 3, 2}, false)}, false, false)
			if err != nil {
				t.Fatal(err)
			}
			closeGradients(t, gradients)
			worst := 0.0
			for index, gradient := range gradients {
				expected := finiteDifference(*data[index], dy, func() []float64 { return attentionReference(f) })
				worst = math.Max(worst, maxError(values(t, gradient), expected))
				near(t, values(t, gradient), expected, coverageGradientTolerance)
			}
			t.Logf("%s: forward max error %.3g, %d gradients max error %.3g", test.name, forward, len(gradients), worst)
		})
	}
}

// mlpFixture is a bias-free SwiGLU of width 3 over hidden 2 with optional
// rank-4 pairs, applied at the reference's alpha/rank of 8/4.
type mlpFixture struct {
	gate, up, down         []float64
	ga, gb, ua, ub, da, db []float64
	inputNorm, postNorm    []float64
}

func newMLPFixture() mlpFixture {
	return mlpFixture{
		gate: pattern(6, 0.25), up: pattern(6, 0.55), down: pattern(6, 0.85),
		ga: pattern(8, 0.1), gb: pattern(12, 0.2), ua: pattern(8, 0.3), ub: pattern(12, 0.4), da: pattern(12, 0.5), db: pattern(8, 0.6),
		inputNorm: []float64{0.1, -0.2}, postNorm: []float64{-0.15, 0.25},
	}
}

func (m *mlpFixture) pairs() map[layers.LoRATarget]referencePair {
	return map[layers.LoRATarget]referencePair{
		layers.LoRAGate: {layers.LoRAGate, &m.ga, &m.gb, []int64{4, 2}, []int64{3, 4}},
		layers.LoRAUp:   {layers.LoRAUp, &m.ua, &m.ub, []int64{4, 2}, []int64{3, 4}},
		layers.LoRADown: {layers.LoRADown, &m.da, &m.db, []int64{4, 3}, []int64{2, 4}},
	}
}

// effective merges W + 2*B*A in FP64; a nil pair returns W unchanged.
func effective(weight, a, b []float64, output, input int) []float64 {
	result := append([]float64(nil), weight...)
	if a == nil {
		return result
	}
	for o := range output {
		for i := range input {
			for rank := range 4 {
				result[o*input+i] += 2 * b[o*4+rank] * a[rank*input+i]
			}
		}
	}
	return result
}

// normalizeRows applies the offset RMS normalization of the checkpoint.
func normalizeRows(x, weight []float64, width int) []float64 {
	result := make([]float64, len(x))
	for row := 0; row < len(x)/width; row++ {
		variance := 1e-6
		for column := range width {
			variance += x[row*width+column] * x[row*width+column] / float64(width)
		}
		for column := range width {
			result[row*width+column] = x[row*width+column] / math.Sqrt(variance) * (1 + weight[column])
		}
	}
	return result
}

// fullDecoderReference composes the attention reference with both
// normalizations, both residuals and the adapted SwiGLU, in scalar FP64.
func fullDecoderReference(f attentionFixture, m mlpFixture) []float64 {
	attentionInput := f
	attentionInput.x = normalizeRows(f.x, m.inputNorm, 2)
	attended := attentionReference(attentionInput)
	residual := make([]float64, len(f.x))
	for index := range residual {
		residual[index] = f.x[index] + attended[index]
	}
	normalized := normalizeRows(residual, m.postNorm, 2)
	gate, up, down := effective(m.gate, m.ga, m.gb, 3, 2), effective(m.up, m.ua, m.ub, 3, 2), effective(m.down, m.da, m.db, 2, 3)
	result := append([]float64(nil), residual...)
	for token := range 3 {
		for feature := range 3 {
			var gated, lifted float64
			for column := range 2 {
				gated += normalized[token*2+column] * gate[feature*2+column]
				lifted += normalized[token*2+column] * up[feature*2+column]
			}
			activated := gated / (1 + math.Exp(-gated)) * lifted
			for column := range 2 {
				result[token*2+column] += activated * down[column*3+feature]
			}
		}
	}
	return result
}

func fullDecoder(t *testing.T, f attentionFixture, m mlpFixture) (layers.DecoderWeights, layers.DecoderConfig, *torch.Tensor, *torch.Tensor) {
	weights := attentionWeights(t, f)
	decoder := layers.DecoderWeights{
		InputNorm: tensor32(t, m.inputNorm, []int64{2}, false), PostAttentionNorm: tensor32(t, m.postNorm, []int64{2}, false),
		Gate: tensor32(t, m.gate, []int64{3, 2}, false), Up: tensor32(t, m.up, []int64{3, 2}, false), Down: tensor32(t, m.down, []int64{2, 3}, false),
		Full: &weights,
	}
	config := layers.DecoderConfig{Epsilon: 1e-6, MaxInputElements: 6, Full: attentionConfig()}
	return decoder, config, tensor32(t, f.cosine, []int64{3, 1}, false), tensor32(t, f.sine, []int64{3, 1}, false)
}

func TestFullAttentionDecoderLoRAMatchesIndependentReference(t *testing.T) {
	all := []layers.LoRATarget{layers.LoRAQuery, layers.LoRAKey, layers.LoRAValue, layers.LoRAOutput, layers.LoRAGate, layers.LoRAUp, layers.LoRADown}
	for _, test := range []struct {
		name    string
		targets []layers.LoRATarget
	}{
		{"gate_proj", []layers.LoRATarget{layers.LoRAGate}},
		{"up_proj", []layers.LoRATarget{layers.LoRAUp}},
		{"down_proj", []layers.LoRATarget{layers.LoRADown}},
		{"attention_and_mlp", all},
	} {
		t.Run(test.name, func(t *testing.T) {
			f, m := withKeyAndOutput(fixture()), newMLPFixture()
			pairs := attentionPairs(&f)
			for target, pair := range m.pairs() {
				pairs[target] = pair
			}
			adapter, chosen := selectPairs(t, pairs, test.targets, false)
			weights, config, cosine, sine := fullDecoder(t, f, m)
			x := tensor32(t, f.x, []int64{1, 3, 2}, false)
			reference := func() []float64 { return fullDecoderReference(f, m) }
			output, err := layers.DecoderForward(context.Background(), x, weights, adapter, cosine, sine, config)
			output = own(t, output, err)
			forward := maxError(values(t, output), reference())
			near(t, values(t, output), reference(), coverageForwardTolerance)
			dy := []float64{0.4, -0.3, 0.6, 0.1, -0.7, 0.2}
			gradients, err := layers.DecoderVJP(context.Background(), x, weights, adapter, cosine, sine, tensor32(t, dy, []int64{1, 3, 2}, false), config)
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() { _ = gradients.Close() })
			expected := finiteDifference(f.x, dy, reference)
			worst := maxError(values(t, gradients.Input), expected)
			near(t, values(t, gradients.Input), expected, coverageGradientTolerance)
			for _, pair := range chosen {
				a, b := gradients.Pair(pair.target)
				for index, gradient := range []*torch.Tensor{*a, *b} {
					data := []*[]float64{pair.a, pair.b}[index]
					expected := finiteDifference(*data, dy, reference)
					worst = math.Max(worst, maxError(values(t, gradient), expected))
					near(t, values(t, gradient), expected, coverageGradientTolerance)
				}
			}
			for _, target := range layers.LoRATargets() {
				a, b := gradients.Pair(target)
				adapted, _ := adapter.Pair(target)
				if (*a != nil || *b != nil) != (*adapted != nil) {
					t.Fatalf("gradient presence differs from the adapter for target %d", target)
				}
			}
			t.Logf("%s: forward max error %.3g, input and %d pair gradients max error %.3g", test.name, forward, 2*len(chosen), worst)
		})
	}
}

// With every B at exact zero the adapted block is the frozen block: the
// output and the input cotangent equal the unadapted ones value for value.
func TestZeroBReproducesTheFrozenFullAttentionDecoder(t *testing.T) {
	f, m := withKeyAndOutput(fixture()), newMLPFixture()
	pairs := attentionPairs(&f)
	for target, pair := range m.pairs() {
		pairs[target] = pair
	}
	adapter, _ := selectPairs(t, pairs, []layers.LoRATarget{layers.LoRAQuery, layers.LoRAKey, layers.LoRAValue, layers.LoRAOutput, layers.LoRAGate, layers.LoRAUp, layers.LoRADown}, true)
	weights, config, cosine, sine := fullDecoder(t, f, m)
	x := tensor32(t, f.x, []int64{1, 3, 2}, false)
	adapted, err := layers.DecoderForward(context.Background(), x, weights, adapter, cosine, sine, config)
	adapted = own(t, adapted, err)
	frozen, err := layers.DecoderForward(context.Background(), x, weights, nil, cosine, sine, config)
	frozen = own(t, frozen, err)
	exactly(t, "forward", values(t, adapted), values(t, frozen))
	seed := tensor32(t, []float64{0.4, -0.3, 0.6, 0.1, -0.7, 0.2}, []int64{1, 3, 2}, false)
	withAdapter, err := layers.DecoderVJP(context.Background(), x, weights, adapter, cosine, sine, seed, config)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = withAdapter.Close() })
	without, err := layers.DecoderVJP(context.Background(), x, weights, nil, cosine, sine, seed, config)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = without.Close() })
	exactly(t, "input cotangent", values(t, withAdapter.Input), values(t, without.Input))
}

func exactly(t *testing.T, what string, got, expected []float64) {
	t.Helper()
	if len(got) != len(expected) {
		t.Fatalf("%s length differs", what)
	}
	for index := range got {
		if got[index] != expected[index] {
			t.Fatalf("%s element %d: %.9g != %.9g", what, index, got[index], expected[index])
		}
	}
}

func TestDecoderRefusesPairsOfTheOtherAttentionVariant(t *testing.T) {
	f, m := withKeyAndOutput(fixture()), newMLPFixture()
	weights, config, cosine, sine := fullDecoder(t, f, m)
	x := tensor32(t, f.x, []int64{1, 3, 2}, false)
	recurrent := &layers.AttentionLoRA{Alpha: 8, Linear: layers.LinearAttentionLoRA{ZA: tensor32(t, pattern(8, 0.1), []int64{4, 2}, true), ZB: tensor32(t, pattern(16, 0.2), []int64{4, 4}, true)}}
	half := &layers.AttentionLoRA{Alpha: 8, KeyA: tensor32(t, pattern(8, 0.1), []int64{4, 2}, true)}
	empty := &layers.AttentionLoRA{Alpha: 8}
	for name, adapter := range map[string]*layers.AttentionLoRA{"recurrent_pair": recurrent, "half_pair": half, "no_pair": empty} {
		if output, err := layers.DecoderForward(context.Background(), x, weights, adapter, cosine, sine, config); err == nil || output != nil {
			t.Fatalf("%s admitted on a full-attention block", name)
		}
	}
}
