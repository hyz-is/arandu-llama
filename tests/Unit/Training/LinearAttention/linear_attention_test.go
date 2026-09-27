//go:build libtorch && cgo

package linearattention_test

import (
	"context"
	"errors"
	"math"
	"reflect"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/tayi-ai/arandu-llama/training/layers"
	"github.com/tayi-ai/arandu-llama/training/sequence"
	"github.com/tayi-ai/arandu-llama/training/torch"
)

type fixture struct {
	batch, tokens, hidden, kh, vh, kd, vd             int
	x, qkv, z, beta, alpha, conv, alog, dt, norm, out []float32
	config                                            layers.LinearAttentionConfig
}

func data(size int, phase, scale float64) []float32 {
	values := make([]float32, size)
	for i := range values {
		values[i] = float32(scale * math.Sin(float64(i)*0.73+phase))
	}
	return values
}

func newFixture() fixture {
	f := fixture{batch: 2, tokens: 5, hidden: 3, kh: 2, vh: 4, kd: 2, vd: 3}
	channels, values := 2*f.kh*f.kd+f.vh*f.vd, f.vh*f.vd
	f.x = data(f.batch*f.tokens*f.hidden, 0.4, 0.7)
	f.qkv, f.z = data(channels*f.hidden, 0.2, 0.6), data(values*f.hidden, 0.7, 0.6)
	f.beta, f.alpha = data(f.vh*f.hidden, 0.5, 0.7), data(f.vh*f.hidden, 0.1, 0.6)
	f.conv = data(channels*4, 0.4, 0.7)
	f.alog, f.dt = data(f.vh, 0.2, 0.3), data(f.vh, 0.8, 0.4)
	f.norm, f.out = data(f.vd, 0.4, 0.8), data(f.hidden*values, 0.9, 0.7)
	f.config = layers.LinearAttentionConfig{
		KeyHeads: int64(f.kh), ValueHeads: int64(f.vh), KeyDimension: int64(f.kd), ValueDimension: int64(f.vd),
		Epsilon: 1e-4, MaxWorkingElements: 1 << 20,
		Sequence: sequence.SequenceLimits{ChunkTokens: 2, MaxTokens: 64, MaxOwnedElements: 1 << 20},
	}
	return f
}

func tensor(t *testing.T, values []float32, shape []int64, grad bool) *torch.Tensor {
	t.Helper()
	v, err := torch.FromFloat32(values, shape, torch.CPUDevice(), grad)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = v.Close() })
	return v
}

func own(t *testing.T, value *torch.Tensor, err error) *torch.Tensor {
	t.Helper()
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = value.Close() })
	return value
}

func read(t *testing.T, value *torch.Tensor) []float32 {
	t.Helper()
	v, err := value.Float32Values()
	if err != nil {
		t.Fatal(err)
	}
	return v
}

func (f fixture) tensors(t *testing.T, inputGrad bool) (*torch.Tensor, layers.LinearAttentionWeights) {
	d, kh, vh, kd, vd := int64(f.hidden), int64(f.kh), int64(f.vh), int64(f.kd), int64(f.vd)
	c, v := 2*kh*kd+vh*vd, vh*vd
	return tensor(t, f.x, []int64{int64(f.batch), int64(f.tokens), d}, inputGrad), layers.LinearAttentionWeights{
		QKV: tensor(t, f.qkv, []int64{c, d}, false), Z: tensor(t, f.z, []int64{v, d}, false),
		Beta: tensor(t, f.beta, []int64{vh, d}, false), Alpha: tensor(t, f.alpha, []int64{vh, d}, false),
		Convolution: tensor(t, f.conv, []int64{c, 1, 4}, false), ALog: tensor(t, f.alog, []int64{vh}, false),
		DTBias: tensor(t, f.dt, []int64{vh}, false), Norm: tensor(t, f.norm, []int64{vd}, false),
		Output: tensor(t, f.out, []int64{d, v}, false),
	}
}

func doubles(values []float32) []float64 {
	v := make([]float64, len(values))
	for i, value := range values {
		v[i] = float64(value)
	}
	return v
}

// oracle uses only scalar Go float64 loops. It deliberately does not call
// layers, sequence, tensor, or the pure-Go GDN implementation under test.
func (f fixture) oracle(x []float64) []float64 {
	return f.oracleWith(x, projections{qkv: doubles(f.qkv), z: doubles(f.z), alpha: doubles(f.alpha), beta: doubles(f.beta), out: doubles(f.out)})
}

// projections holds the five projection matrices in float64, so an adapted
// oracle can merge its pairs into them without rounding.
type projections struct{ qkv, z, alpha, beta, out []float64 }

// oracleWith is the oracle over explicit projection matrices.
func (f fixture) oracleWith(x []float64, w projections) []float64 {
	channels, keys, values := 2*f.kh*f.kd+f.vh*f.vd, f.kh*f.kd, f.vh*f.vd
	project := func(weights []float64, size int) []float64 {
		result := make([]float64, f.batch*f.tokens*size)
		for row := 0; row < f.batch*f.tokens; row++ {
			for output := 0; output < size; output++ {
				for input := 0; input < f.hidden; input++ {
					result[row*size+output] += x[row*f.hidden+input] * weights[output*f.hidden+input]
				}
			}
		}
		return result
	}
	mixed, z, a, beta := project(w.qkv, channels), project(w.z, values), project(w.alpha, f.vh), project(w.beta, f.vh)
	conv := make([]float64, len(mixed))
	for b := 0; b < f.batch; b++ {
		for token := 0; token < f.tokens; token++ {
			for c := 0; c < channels; c++ {
				var sum float64
				for tap := 0; tap < 4; tap++ {
					past := token + tap - 3
					if past >= 0 {
						sum += mixed[(b*f.tokens+past)*channels+c] * float64(f.conv[c*4+tap])
					}
				}
				conv[(b*f.tokens+token)*channels+c] = sum / (1 + math.Exp(-sum))
			}
		}
	}
	core := make([]float64, f.batch*f.tokens*values)
	for b := 0; b < f.batch; b++ {
		state := make([]float64, f.vh*f.kd*f.vd)
		for token := 0; token < f.tokens; token++ {
			row := b*f.tokens + token
			for h := 0; h < f.vh; h++ {
				kh := h / (f.vh / f.kh) // repeat_interleave, not a tiled head order
				q, k := make([]float64, f.kd), make([]float64, f.kd)
				qn, kn := 1e-6, 1e-6
				for i := 0; i < f.kd; i++ {
					q[i], k[i] = conv[row*channels+kh*f.kd+i], conv[row*channels+keys+kh*f.kd+i]
					qn += q[i] * q[i]
					kn += k[i] * k[i]
				}
				for i := range q {
					q[i] /= math.Sqrt(qn) * math.Sqrt(float64(f.kd))
					k[i] /= math.Sqrt(kn)
				}
				softplusInput := a[row*f.vh+h] + float64(f.dt[h])
				softplus := softplusInput
				if softplusInput <= 20 {
					softplus = math.Log1p(math.Exp(softplusInput))
				}
				decay := math.Exp(-math.Exp(float64(f.alog[h])) * softplus)
				step := 1 / (1 + math.Exp(-beta[row*f.vh+h]))
				for i := 0; i < f.kd*f.vd; i++ {
					state[h*f.kd*f.vd+i] *= decay
				}
				for j := 0; j < f.vd; j++ {
					prediction := 0.0
					for i := 0; i < f.kd; i++ {
						prediction += k[i] * state[(h*f.kd+i)*f.vd+j]
					}
					delta := step * (conv[row*channels+2*keys+h*f.vd+j] - prediction)
					for i := 0; i < f.kd; i++ {
						state[(h*f.kd+i)*f.vd+j] += k[i] * delta
						core[row*values+h*f.vd+j] += q[i] * state[(h*f.kd+i)*f.vd+j]
					}
				}
				variance := f.config.Epsilon
				for j := 0; j < f.vd; j++ {
					value := core[row*values+h*f.vd+j]
					variance += value * value / float64(f.vd)
				}
				for j := 0; j < f.vd; j++ {
					i := row*values + h*f.vd + j
					core[i] = core[i] / math.Sqrt(variance) * float64(f.norm[j]) * z[i] / (1 + math.Exp(-z[i]))
				}
			}
		}
	}
	result := make([]float64, len(x))
	for row := 0; row < f.batch*f.tokens; row++ {
		for out := 0; out < f.hidden; out++ {
			for in := 0; in < values; in++ {
				result[row*f.hidden+out] += core[row*values+in] * w.out[out*values+in]
			}
		}
	}
	return result
}

func near(t *testing.T, got []float32, expected []float64, tolerance float64) float64 {
	t.Helper()
	if len(got) != len(expected) {
		t.Fatalf("length mismatch %d != %d", len(got), len(expected))
	}
	maxError := 0.0
	for i, value := range got {
		err := math.Abs(float64(value) - expected[i])
		maxError = math.Max(maxError, err)
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) || err > tolerance {
			t.Fatalf("element %d: got %.9g expected %.9g error %.4g tolerance %.4g", i, value, expected[i], err, tolerance)
		}
	}
	return maxError
}

func assertDetached(t *testing.T, value *torch.Tensor) {
	t.Helper()
	info, err := value.Info()
	if err != nil {
		t.Fatal(err)
	}
	if info.RequiresGrad {
		t.Fatal("result retained an autograd graph")
	}
}

func TestForwardAndCompleteInputVJPAgainstIndependentScalarOracle(t *testing.T) {
	f := newFixture()
	x, w := f.tensors(t, true)
	dy := data(len(f.x), 0.3, 0.7)
	seed := tensor(t, dy, []int64{int64(f.batch), int64(f.tokens), int64(f.hidden)}, false)
	output, err := layers.ForwardLinearAttention(context.Background(), x, w, f.config)
	output = own(t, output, err)
	gradient, err := layers.LinearAttentionVJP(context.Background(), x, w, seed, f.config)
	gradient = own(t, gradient, err)
	assertDetached(t, output)
	assertDetached(t, gradient)
	forwardError := near(t, read(t, output), f.oracle(doubles(f.x)), 2e-6)
	fd := make([]float64, len(f.x))
	for i := range fd {
		plus, minus := doubles(f.x), doubles(f.x)
		const step = 1e-5
		plus[i] += step
		minus[i] -= step
		yp, ym := f.oracle(plus), f.oracle(minus)
		for j := range yp {
			fd[i] += float64(dy[j]) * (yp[j] - ym[j]) / (2 * step)
		}
	}
	gradientError := near(t, read(t, gradient), fd, 2e-5)
	t.Logf("scalar FP64 oracle: forward max error %.9g; input VJP %d finite differences max error %.9g", forwardError, len(fd), gradientError)
	if !reflect.DeepEqual(read(t, x), f.x) || !reflect.DeepEqual(read(t, w.QKV), f.qkv) || !reflect.DeepEqual(read(t, w.Convolution), f.conv) {
		t.Fatal("borrowed input or frozen weights changed")
	}
	info, _ := x.Info()
	if !info.RequiresGrad {
		t.Fatal("caller input gradient flag changed")
	}
	for _, weight := range []*torch.Tensor{w.QKV, w.Z, w.Beta, w.Alpha, w.Convolution, w.ALog, w.DTBias, w.Norm, w.Output} {
		assertDetached(t, weight)
	}
	// A second call must not consume or attach itself to the caller's graph.
	again, err := layers.LinearAttentionVJP(context.Background(), x, w, seed, f.config)
	again = own(t, again, err)
	// Native autograd may accumulate independent FP32 branches in a different
	// order. Keep this tolerance far below the independent finite-difference
	// bound above instead of requiring bitwise-identical recomputation.
	repeatError := near(t, read(t, again), doubles(read(t, gradient)), 2e-7)
	t.Logf("repeated input VJP max error %.9g", repeatError)
}

func TestFutureCausalityAndCheckpointIntervalEquivalence(t *testing.T) {
	f := newFixture()
	x, w := f.tensors(t, false)
	dy := make([]float32, len(f.x))
	for b := 0; b < f.batch; b++ {
		for token := 0; token < f.tokens-1; token++ {
			for d := 0; d < f.hidden; d++ {
				dy[(b*f.tokens+token)*f.hidden+d] = float32(d+1) * 0.2
			}
		}
	}
	seed := tensor(t, dy, []int64{int64(f.batch), int64(f.tokens), int64(f.hidden)}, false)
	base, err := layers.ForwardLinearAttention(context.Background(), x, w, f.config)
	base = own(t, base, err)
	baseGrad, err := layers.LinearAttentionVJP(context.Background(), x, w, seed, f.config)
	baseGrad = own(t, baseGrad, err)
	future := append([]float32(nil), f.x...)
	for b := 0; b < f.batch; b++ {
		for d := 0; d < f.hidden; d++ {
			future[(b*f.tokens+f.tokens-1)*f.hidden+d] += 0.9
		}
	}
	xFuture := tensor(t, future, []int64{int64(f.batch), int64(f.tokens), int64(f.hidden)}, false)
	changed, err := layers.ForwardLinearAttention(context.Background(), xFuture, w, f.config)
	changed = own(t, changed, err)
	before, after, dx := read(t, base), read(t, changed), read(t, baseGrad)
	for b := 0; b < f.batch; b++ {
		start, end := b*f.tokens*f.hidden, (b*f.tokens+f.tokens-1)*f.hidden
		near(t, after[start:end], doubles(before[start:end]), 0)
		near(t, dx[end:end+f.hidden], make([]float64, f.hidden), 0)
	}
	for _, chunk := range []int64{1, 4, 8} {
		config := f.config
		config.Sequence.ChunkTokens = chunk
		value, err := layers.ForwardLinearAttention(context.Background(), x, w, config)
		value = own(t, value, err)
		gradient, err := layers.LinearAttentionVJP(context.Background(), x, w, seed, config)
		gradient = own(t, gradient, err)
		near(t, read(t, value), doubles(before), 1e-7)
		near(t, read(t, gradient), doubles(dx), 1e-6)
	}
}

func TestSingleTokenEqualHeadsAndZeroInput(t *testing.T) {
	f := newFixture()
	f.batch, f.tokens, f.kh = 1, 1, f.vh
	f.config.KeyHeads = int64(f.kh)
	f.x = data(f.hidden, 0.4, 0.7)
	f.qkv = data((2*f.kh*f.kd+f.vh*f.vd)*f.hidden, 0.2, 0.6)
	f.conv = data((2*f.kh*f.kd+f.vh*f.vd)*4, 0.4, 0.7)
	for _, zero := range []bool{false, true} {
		if zero {
			clear(f.x)
		}
		x, w := f.tensors(t, false)
		seed := tensor(t, []float32{0.3, -0.4, 0.2}, []int64{1, 1, 3}, false)
		value, err := layers.ForwardLinearAttention(context.Background(), x, w, f.config)
		value = own(t, value, err)
		gradient, err := layers.LinearAttentionVJP(context.Background(), x, w, seed, f.config)
		gradient = own(t, gradient, err)
		near(t, read(t, value), f.oracle(doubles(f.x)), 2e-6)
		if zero {
			near(t, read(t, gradient), make([]float64, len(f.x)), 0)
		}
	}
}

func TestLateTokenLossReachesEarliestTokenAcrossChunks(t *testing.T) {
	f := newFixture()
	f.tokens = 9 // Beyond the four-tap convolution's direct receptive field.
	f.x = data(f.batch*f.tokens*f.hidden, 0.4, 0.7)
	x, w := f.tensors(t, false)
	dy := make([]float32, len(f.x))
	for d := 0; d < f.hidden; d++ {
		dy[(f.tokens-1)*f.hidden+d] = float32(d+1) * 0.3
	}
	seed := tensor(t, dy, []int64{int64(f.batch), int64(f.tokens), int64(f.hidden)}, false)
	gradient, err := layers.LinearAttentionVJP(context.Background(), x, w, seed, f.config)
	gradient = own(t, gradient, err)
	dx := read(t, gradient)
	var earliestNorm float64
	for d := 0; d < f.hidden; d++ {
		earliestNorm += math.Abs(float64(dx[d]))
		plus, minus := doubles(f.x), doubles(f.x)
		const step = 1e-5
		plus[d] += step
		minus[d] -= step
		yp, ym := f.oracle(plus), f.oracle(minus)
		var expected float64
		for j := range yp {
			expected += float64(dy[j]) * (yp[j] - ym[j]) / (2 * step)
		}
		near(t, dx[d:d+1], []float64{expected}, 2e-6)
	}
	if earliestNorm < 1e-9 {
		t.Fatal("late loss did not cross recurrence chunk boundaries")
	}
	near(t, dx[f.tokens*f.hidden:], make([]float64, f.tokens*f.hidden), 0)
	t.Logf("nine-token recurrence: earliest-token gradient L1 %.9g", earliestNorm)
}

// Deterministic cancellation during native work, without timing assumptions.
type checkingContext struct {
	context.Context
	cancel context.CancelFunc
	calls  atomic.Int32
}

func (c *checkingContext) Err() error {
	if c.calls.Add(1) >= 80 {
		c.cancel()
	}
	return c.Context.Err()
}

func TestCancellationDuringProjectionPreservesBorrowedTensors(t *testing.T) {
	f := newFixture()
	x, w := f.tensors(t, true)
	seed := tensor(t, data(len(f.x), 0.3, 0.7), []int64{2, 5, 3}, false)
	for _, backward := range []bool{false, true} {
		ctx, cancel := context.WithCancel(context.Background())
		checking := &checkingContext{Context: ctx, cancel: cancel}
		var value *torch.Tensor
		var err error
		if backward {
			value, err = layers.LinearAttentionVJP(checking, x, w, seed, f.config)
		} else {
			value, err = layers.ForwardLinearAttention(checking, x, w, f.config)
		}
		cancel()
		if value != nil || !errors.Is(err, context.Canceled) {
			t.Fatalf("mid-projection cancellation returned %v, %v", value, err)
		}
		if !reflect.DeepEqual(read(t, x), f.x) || !reflect.DeepEqual(read(t, w.QKV), f.qkv) {
			t.Fatal("cancellation mutated borrowed tensors")
		}
	}
}

func TestFiniteProjectionOverflowNamesItsStage(t *testing.T) {
	f := newFixture()
	x, w := f.tensors(t, false)
	w.ALog = tensor(t, []float32{0, 0, 1000, 0}, []int64{4}, false)
	value, err := layers.ForwardLinearAttention(context.Background(), x, w, f.config)
	if value != nil || err == nil || !strings.Contains(err.Error(), "negative_a_exp") {
		t.Fatalf("overflow stage = value %v error %v", value, err)
	}
}

func TestLargeFiniteQueryUsesStableL2Normalization(t *testing.T) {
	f := newFixture()
	queryRows := f.kh * f.kd
	for row := 0; row < queryRows; row++ {
		for column := 0; column < f.hidden; column++ {
			f.qkv[row*f.hidden+column] *= 1e20
		}
	}
	x, w := f.tensors(t, true)
	seedValues := data(len(f.x), 0.3, 0.7)
	seed := tensor(t, seedValues, []int64{int64(f.batch), int64(f.tokens), int64(f.hidden)}, false)
	value, err := layers.ForwardLinearAttention(context.Background(), x, w, f.config)
	value = own(t, value, err)
	gradient, err := layers.LinearAttentionVJP(context.Background(), x, w, seed, f.config)
	gradient = own(t, gradient, err)
	if finite, err := value.AllFinite(); err != nil || !finite {
		t.Fatalf("large-query forward finite=%t err=%v", finite, err)
	}
	if finite, err := gradient.AllFinite(); err != nil || !finite {
		t.Fatalf("large-query VJP finite=%t err=%v", finite, err)
	}
	near(t, read(t, value), f.oracle(doubles(f.x)), 5e-5)
}

func TestCUDALargeFiniteQueryUsesStableL2Normalization(t *testing.T) {
	if !torch.CUDAEnabled() {
		t.Skip("CUDA bridge is not enabled")
	}
	count, err := torch.CUDADeviceCount()
	if err != nil {
		t.Fatal(err)
	}
	if count < 1 {
		t.Skip("one CUDA device is required")
	}
	f := newFixture()
	queryRows := f.kh * f.kd
	for row := 0; row < queryRows; row++ {
		for column := 0; column < f.hidden; column++ {
			f.qkv[row*f.hidden+column] *= 1e20
		}
	}
	cpuX, cpuWeights := f.tensors(t, true)
	device := torch.CUDADevice(0)
	move := func(value *torch.Tensor) *torch.Tensor {
		moved, err := value.To(device, torch.Float32)
		return own(t, moved, err)
	}
	x := move(cpuX)
	w := layers.LinearAttentionWeights{
		QKV: move(cpuWeights.QKV), Z: move(cpuWeights.Z), Beta: move(cpuWeights.Beta), Alpha: move(cpuWeights.Alpha),
		Convolution: move(cpuWeights.Convolution), ALog: move(cpuWeights.ALog), DTBias: move(cpuWeights.DTBias),
		Norm: move(cpuWeights.Norm), Output: move(cpuWeights.Output),
	}
	seedCPU := tensor(t, data(len(f.x), 0.3, 0.7), []int64{int64(f.batch), int64(f.tokens), int64(f.hidden)}, false)
	seed := move(seedCPU)
	value, err := layers.ForwardLinearAttention(context.Background(), x, w, f.config)
	value = own(t, value, err)
	gradient, err := layers.LinearAttentionVJP(context.Background(), x, w, seed, f.config)
	gradient = own(t, gradient, err)
	if finite, err := value.AllFinite(); err != nil || !finite {
		t.Fatalf("CUDA large-query forward finite=%t err=%v", finite, err)
	}
	if finite, err := gradient.AllFinite(); err != nil || !finite {
		t.Fatalf("CUDA large-query VJP finite=%t err=%v", finite, err)
	}
	near(t, read(t, value), f.oracle(doubles(f.x)), 2e-4)
}

func TestValidationCancellationAndOverflowFailWithoutMutatingInputs(t *testing.T) {
	f := newFixture()
	x, w := f.tensors(t, true)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if value, err := layers.ForwardLinearAttention(ctx, x, w, f.config); value != nil || !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled context: %v %v", value, err)
	}
	if value, err := layers.ForwardLinearAttention(nil, x, w, f.config); value != nil || err == nil {
		t.Fatal("nil context accepted")
	}
	if value, err := layers.LinearAttentionVJP(context.Background(), x, w, nil, f.config); value != nil || err == nil {
		t.Fatal("nil seed accepted")
	}
	for name, change := range map[string]func(*layers.LinearAttentionConfig){
		"head ratio":         func(c *layers.LinearAttentionConfig) { c.ValueHeads = 3 },
		"dimension overflow": func(c *layers.LinearAttentionConfig) { c.KeyDimension = math.MaxInt64 },
		"budget":             func(c *layers.LinearAttentionConfig) { c.MaxWorkingElements = 1 },
		"negative budget":    func(c *layers.LinearAttentionConfig) { c.MaxWorkingElements = -1 },
		"epsilon":            func(c *layers.LinearAttentionConfig) { c.Epsilon = math.NaN() },
		"epsilon underflow":  func(c *layers.LinearAttentionConfig) { c.Epsilon = math.SmallestNonzeroFloat64 },
		"sequence tokens":    func(c *layers.LinearAttentionConfig) { c.Sequence.MaxTokens = 1 },
		"sequence chunk":     func(c *layers.LinearAttentionConfig) { c.Sequence.ChunkTokens = 33 },
	} {
		t.Run(name, func(t *testing.T) {
			c := f.config
			change(&c)
			if value, err := layers.ForwardLinearAttention(context.Background(), x, w, c); value != nil || err == nil {
				t.Fatal("invalid configuration accepted")
			}
		})
	}
	for name, change := range map[string]func(*layers.LinearAttentionWeights){
		"missing":   func(v *layers.LinearAttentionWeights) { v.Norm = nil },
		"bad shape": func(v *layers.LinearAttentionWeights) { v.Norm = v.ALog },
		"trainable": func(v *layers.LinearAttentionWeights) { v.Norm = tensor(t, f.norm, []int64{int64(f.vd)}, true) },
		"nonfinite": func(v *layers.LinearAttentionWeights) {
			v.ALog = tensor(t, []float32{0, 0, float32(math.Inf(1)), 0}, []int64{4}, false)
		},
		"finite exponential overflow": func(v *layers.LinearAttentionWeights) {
			v.ALog = tensor(t, []float32{0, 0, 1000, 0}, []int64{4}, false)
		},
	} {
		t.Run(name, func(t *testing.T) {
			v := w
			change(&v)
			if value, err := layers.ForwardLinearAttention(context.Background(), x, v, f.config); value != nil || err == nil {
				t.Fatal("invalid weights accepted")
			}
		})
	}
	nonfinite := append([]float32(nil), f.x...)
	nonfinite[0] = float32(math.NaN())
	badX := tensor(t, nonfinite, []int64{2, 5, 3}, false)
	if value, err := layers.ForwardLinearAttention(context.Background(), badX, w, f.config); value != nil || err == nil {
		t.Fatal("nonfinite input accepted")
	}
	badSeed := tensor(t, []float32{1}, []int64{1}, false)
	if value, err := layers.LinearAttentionVJP(context.Background(), x, w, badSeed, f.config); value != nil || err == nil {
		t.Fatal("wrong seed shape accepted")
	}
	wrongDType, err := torch.FromFloat64(doubles(f.x), []int64{2, 5, 3}, torch.CPUDevice(), false)
	wrongDType = own(t, wrongDType, err)
	if value, err := layers.ForwardLinearAttention(context.Background(), wrongDType, w, f.config); value != nil || err == nil {
		t.Fatal("unqualified dtype accepted")
	}
	if !reflect.DeepEqual(read(t, x), f.x) || !reflect.DeepEqual(read(t, w.ALog), f.alog) {
		t.Fatal("failure changed caller values")
	}
	// Zero-valued limits select documented defaults, rather than a zero budget.
	c := f.config
	c.MaxWorkingElements = 0
	c.Sequence = sequence.SequenceLimits{}
	value, err := layers.ForwardLinearAttention(context.Background(), x, w, c)
	value = own(t, value, err)
	near(t, read(t, value), f.oracle(doubles(f.x)), 2e-6)
}
