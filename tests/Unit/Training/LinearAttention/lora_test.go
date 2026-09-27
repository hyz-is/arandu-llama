//go:build libtorch && cgo

package linearattention_test

import (
	"context"
	"math"
	"testing"

	"github.com/tayi-ai/arandu-llama/training/layers"
	"github.com/tayi-ai/arandu-llama/training/torch"
)

// Declared tolerances: FP32 forward against the FP64 oracle, and FP32
// cotangents against FP64 central differences with step 1e-5, as the frozen
// recurrent decoder oracle already admits.
const (
	recurrentForwardTolerance  = 4e-6
	recurrentGradientTolerance = 4e-5
	recurrentRank              = 2
	recurrentAlpha             = 4.0
)

// recurrentPair is one rank-2 pair over a projection of the recurrent block
// or its MLP. a and b are the oracle's float64 copies of the tensors' values.
type recurrentPair struct {
	target        layers.LoRATarget
	name          string
	output, input int
	a, b          []float64
}

type recurrentDecoder struct {
	f                                         fixture
	inputNorm, postNorm, gate, up, down       []float32
	intermediate                              int
	pairs                                     []*recurrentPair
	weights                                   layers.DecoderWeights
	config                                    layers.DecoderConfig
	x                                         *torch.Tensor
	adapter                                   *layers.AttentionLoRA
	attention                                 layers.LinearAttentionWeights
	channels, values, heads, hidden, features int
}

func newRecurrentDecoder(t *testing.T) *recurrentDecoder {
	d := &recurrentDecoder{f: newFixture(), intermediate: 5}
	f := d.f
	d.hidden, d.heads, d.features = f.hidden, f.vh, d.intermediate
	d.channels, d.values = 2*f.kh*f.kd+f.vh*f.vd, f.vh*f.vd
	d.inputNorm, d.postNorm = data(f.hidden, 0.2, 0.1), data(f.hidden, 0.8, 0.1)
	d.gate, d.up, d.down = data(d.intermediate*f.hidden, 0.3, 0.4), data(d.intermediate*f.hidden, 0.7, 0.4), data(f.hidden*d.intermediate, 0.5, 0.4)
	x, attention := f.tensors(t, false)
	d.x, d.attention = x, attention
	h := int64(f.hidden)
	d.weights = layers.DecoderWeights{
		InputNorm: tensor(t, d.inputNorm, []int64{h}, false), PostAttentionNorm: tensor(t, d.postNorm, []int64{h}, false),
		Gate: tensor(t, d.gate, []int64{int64(d.intermediate), h}, false), Up: tensor(t, d.up, []int64{int64(d.intermediate), h}, false),
		Down: tensor(t, d.down, []int64{h, int64(d.intermediate)}, false), Linear: &d.attention,
	}
	d.config = layers.DecoderConfig{Epsilon: 1e-4, MaxInputElements: int64(len(f.x)), Linear: f.config}
	return d
}

// every lists all eight recurrent and MLP targets with their base geometry.
func (d *recurrentDecoder) every() []*recurrentPair {
	return []*recurrentPair{
		{target: layers.LoRALinearOutput, name: "out_proj", output: d.hidden, input: d.values},
		{target: layers.LoRALinearQKV, name: "in_proj_qkv", output: d.channels, input: d.hidden},
		{target: layers.LoRALinearZ, name: "in_proj_z", output: d.values, input: d.hidden},
		{target: layers.LoRALinearBeta, name: "in_proj_b", output: d.heads, input: d.hidden},
		{target: layers.LoRALinearAlpha, name: "in_proj_a", output: d.heads, input: d.hidden},
		{target: layers.LoRAGate, name: "gate_proj", output: d.features, input: d.hidden},
		{target: layers.LoRAUp, name: "up_proj", output: d.features, input: d.hidden},
		{target: layers.LoRADown, name: "down_proj", output: d.hidden, input: d.features},
	}
}

// adapt installs trainable tensors for the chosen pairs. B is exact zero
// when zeroB is set, as a fresh initializer leaves it.
func (d *recurrentDecoder) adapt(t *testing.T, pairs []*recurrentPair, zeroB bool) {
	d.pairs, d.adapter = pairs, &layers.AttentionLoRA{Alpha: recurrentAlpha}
	for index, pair := range pairs {
		a := data(recurrentRank*pair.input, 0.35+float64(index)*0.21, 0.4)
		b := data(pair.output*recurrentRank, 0.65+float64(index)*0.17, 0.3)
		if zeroB {
			clear(b)
		}
		pair.a, pair.b = doubles(a), doubles(b)
		x, y := d.adapter.Pair(pair.target)
		*x = tensor(t, a, []int64{recurrentRank, int64(pair.input)}, true)
		*y = tensor(t, b, []int64{int64(pair.output), recurrentRank}, true)
	}
}

// merged returns W + (alpha/rank)*B*A for target in float64, or W unchanged.
func (d *recurrentDecoder) merged(target layers.LoRATarget, weight []float32) []float64 {
	result := doubles(weight)
	for _, pair := range d.pairs {
		if pair.target != target {
			continue
		}
		for o := 0; o < pair.output; o++ {
			for i := 0; i < pair.input; i++ {
				for r := 0; r < recurrentRank; r++ {
					result[o*pair.input+i] += recurrentAlpha / recurrentRank * pair.b[o*recurrentRank+r] * pair.a[r*pair.input+i]
				}
			}
		}
	}
	return result
}

// oracle is the recurrent decoder of the frozen oracle test with every
// adapted projection merged into its matrix in float64.
func (d *recurrentDecoder) oracle(input []float64) []float64 {
	f := d.f
	normalize := func(input []float64, weight []float32) []float64 {
		result := make([]float64, len(input))
		for row := 0; row < f.batch*f.tokens; row++ {
			variance := d.config.Epsilon
			for column := 0; column < f.hidden; column++ {
				value := input[row*f.hidden+column]
				variance += value * value / float64(f.hidden)
			}
			for column := 0; column < f.hidden; column++ {
				index := row*f.hidden + column
				result[index] = input[index] / math.Sqrt(variance) * (1 + float64(weight[column]))
			}
		}
		return result
	}
	attended := f.oracleWith(normalize(input, d.inputNorm), projections{
		qkv: d.merged(layers.LoRALinearQKV, f.qkv), z: d.merged(layers.LoRALinearZ, f.z),
		alpha: d.merged(layers.LoRALinearAlpha, f.alpha), beta: d.merged(layers.LoRALinearBeta, f.beta), out: d.merged(layers.LoRALinearOutput, f.out),
	})
	residual := make([]float64, len(input))
	for index := range input {
		residual[index] = input[index] + attended[index]
	}
	normalized := normalize(residual, d.postNorm)
	gate, up, down := d.merged(layers.LoRAGate, d.gate), d.merged(layers.LoRAUp, d.up), d.merged(layers.LoRADown, d.down)
	result := append([]float64(nil), residual...)
	for row := 0; row < f.batch*f.tokens; row++ {
		for feature := 0; feature < d.intermediate; feature++ {
			var gated, lifted float64
			for column := 0; column < f.hidden; column++ {
				value := normalized[row*f.hidden+column]
				gated += value * gate[feature*f.hidden+column]
				lifted += value * up[feature*f.hidden+column]
			}
			activated := gated / (1 + math.Exp(-gated)) * lifted
			for column := 0; column < f.hidden; column++ {
				result[row*f.hidden+column] += activated * down[column*d.intermediate+feature]
			}
		}
	}
	return result
}

// differences returns dy·d(oracle)/d(values) by central differences.
func (d *recurrentDecoder) differences(values []float64, dy []float32, oracle func() []float64) []float64 {
	result := make([]float64, len(values))
	for index := range values {
		const step = 1e-5
		original := values[index]
		values[index] = original + step
		plus := oracle()
		values[index] = original - step
		minus := oracle()
		values[index] = original
		for output := range plus {
			result[index] += float64(dy[output]) * (plus[output] - minus[output]) / (2 * step)
		}
	}
	return result
}

func TestRecurrentDecoderLoRAMatchesIndependentScalarOracle(t *testing.T) {
	cases := []string{"out_proj", "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "gate_proj", "up_proj", "down_proj", "all"}
	for _, name := range cases {
		t.Run(name, func(t *testing.T) {
			d := newRecurrentDecoder(t)
			var chosen []*recurrentPair
			for _, pair := range d.every() {
				if name == "all" || pair.name == name {
					chosen = append(chosen, pair)
				}
			}
			d.adapt(t, chosen, false)
			input := doubles(d.f.x)
			oracle := func() []float64 { return d.oracle(input) }
			output, err := layers.DecoderForward(context.Background(), d.x, d.weights, d.adapter, nil, nil, d.config)
			output = own(t, output, err)
			forwardError := near(t, read(t, output), oracle(), recurrentForwardTolerance)
			dy := data(len(d.f.x), 0.3, 0.7)
			seed := tensor(t, dy, []int64{int64(d.f.batch), int64(d.f.tokens), int64(d.f.hidden)}, false)
			gradients, err := layers.DecoderVJP(context.Background(), d.x, d.weights, d.adapter, nil, nil, seed, d.config)
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() { _ = gradients.Close() })
			gradientError := near(t, read(t, gradients.Input), d.differences(input, dy, oracle), recurrentGradientTolerance)
			for _, pair := range chosen {
				a, b := gradients.Pair(pair.target)
				if *a == nil || *b == nil {
					t.Fatalf("%s gradient missing", pair.name)
				}
				gradientError = math.Max(gradientError, near(t, read(t, *a), d.differences(pair.a, dy, oracle), recurrentGradientTolerance))
				gradientError = math.Max(gradientError, near(t, read(t, *b), d.differences(pair.b, dy, oracle), recurrentGradientTolerance))
			}
			for _, target := range layers.LoRATargets() {
				a, _ := gradients.Pair(target)
				adapted, _ := d.adapter.Pair(target)
				if (*a != nil) != (*adapted != nil) {
					t.Fatalf("gradient presence differs from the adapter for target %d", target)
				}
			}
			t.Logf("%s: forward max error %.3g; input and %d pair gradients max error %.3g", name, forwardError, 2*len(chosen), gradientError)
		})
	}
}

// With every B at exact zero the adapted recurrent block is the frozen one:
// output and input cotangent equal the unadapted ones value for value.
func TestZeroBReproducesTheFrozenRecurrentDecoder(t *testing.T) {
	d := newRecurrentDecoder(t)
	d.adapt(t, d.every(), true)
	adapted, err := layers.DecoderForward(context.Background(), d.x, d.weights, d.adapter, nil, nil, d.config)
	adapted = own(t, adapted, err)
	frozen, err := layers.DecoderForward(context.Background(), d.x, d.weights, nil, nil, nil, d.config)
	frozen = own(t, frozen, err)
	same(t, "forward", read(t, adapted), read(t, frozen))
	seed := tensor(t, data(len(d.f.x), 0.3, 0.7), []int64{int64(d.f.batch), int64(d.f.tokens), int64(d.f.hidden)}, false)
	withAdapter, err := layers.DecoderVJP(context.Background(), d.x, d.weights, d.adapter, nil, nil, seed, d.config)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = withAdapter.Close() })
	without, err := layers.DecoderVJP(context.Background(), d.x, d.weights, nil, nil, nil, seed, d.config)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = without.Close() })
	same(t, "input cotangent", read(t, withAdapter.Input), read(t, without.Input))
}

func same(t *testing.T, what string, got, expected []float32) {
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

// A q/v adapter still has no place on a recurrent block; only recurrent and
// MLP pairs are admitted there.
func TestRecurrentDecoderRefusesFullAttentionPairs(t *testing.T) {
	d := newRecurrentDecoder(t)
	d.adapt(t, d.every()[:1], false)
	d.adapter.QueryA = tensor(t, data(6, 0.1, 0.2), []int64{2, 3}, true)
	d.adapter.QueryB = tensor(t, data(40, 0.2, 0.2), []int64{20, 2}, true)
	if output, err := layers.DecoderForward(context.Background(), d.x, d.weights, d.adapter, nil, nil, d.config); err == nil || output != nil {
		t.Fatal("a full-attention pair was admitted on a recurrent block")
	}
}
