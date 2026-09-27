//go:build libtorch && cgo

package layers_test

import (
	"math"
	"testing"

	"github.com/tayi-ai/arandu-llama/training/layers"
	"github.com/tayi-ai/arandu-llama/training/torch"
)

func tensor32(t *testing.T, data []float64, shape []int64, grad bool) *torch.Tensor {
	t.Helper()
	floats := make([]float32, len(data))
	for index := range data {
		floats[index] = float32(data[index])
	}
	value, err := torch.FromFloat32(floats, shape, torch.CPUDevice(), grad)
	return own(t, value, err)
}

type attentionFixture struct {
	x, q, k, v, out, qnorm, knorm, cosine, sine []float64
	qa, qb, va, vb                              []float64
	// Key and output pairs are absent unless a test sets them.
	ka, kb, oa, ob []float64
}

func fixture() attentionFixture {
	pattern := func(size int, phase float64) []float64 {
		result := make([]float64, size)
		for index := range result {
			result[index] = math.Sin(float64(index)*0.8+phase) * 0.3
		}
		return result
	}
	return attentionFixture{
		x: []float64{0.4, -0.2, 0.5, 0.3, -0.1, 0.8},
		q: pattern(32, 0.2), k: pattern(8, 0.4), v: pattern(8, 0.7), out: pattern(16, 0.9),
		qnorm: []float64{0.2, -0.1}, knorm: []float64{-0.3, 0.4},
		cosine: []float64{1, math.Cos(0.2), math.Cos(0.4)}, sine: []float64{0, math.Sin(0.2), math.Sin(0.4)},
		qa: pattern(8, 0.5), qb: pattern(64, 0.1), va: pattern(8, 0.3), vb: pattern(16, 0.6),
	}
}

// attentionReference evaluates a tiny 3-token, 4-query-head, 2-KV-head network
// using scalar Go loops. It does not call the native implementation.
func attentionReference(f attentionFixture) []float64 {
	project := func(weight, a, b []float64, output int) [][]float64 {
		result := make([][]float64, 3)
		for token := range 3 {
			result[token] = make([]float64, output)
			for o := range output {
				for in := range 2 {
					result[token][o] += weight[o*2+in] * f.x[token*2+in]
					for rank := range 4 {
						if a != nil {
							result[token][o] += 2 * b[o*4+rank] * a[rank*2+in] * f.x[token*2+in]
						}
					}
				}
			}
		}
		return result
	}
	qraw, keys, vals := project(f.q, f.qa, f.qb, 16), project(f.k, f.ka, f.kb, 4), project(f.v, f.va, f.vb, 4)
	queries := make([][]float64, 3)
	normalizeRotate := func(vector []float64, weight []float64, token int) {
		inverse := 1 / math.Sqrt((vector[0]*vector[0]+vector[1]*vector[1])/2+1e-6)
		a, b := vector[0]*inverse*(1+weight[0]), vector[1]*inverse*(1+weight[1])
		vector[0], vector[1] = a*f.cosine[token]-b*f.sine[token], b*f.cosine[token]+a*f.sine[token]
	}
	for token := range 3 {
		queries[token] = make([]float64, 8)
		for head := range 4 {
			copy(queries[token][head*2:head*2+2], qraw[token][head*4:head*4+2])
			normalizeRotate(queries[token][head*2:head*2+2], f.qnorm, token)
		}
		for head := range 2 {
			normalizeRotate(keys[token][head*2:head*2+2], f.knorm, token)
		}
	}
	result := make([]float64, 6)
	for token := range 3 {
		joined := make([]float64, 8)
		for head := range 4 {
			scores := make([]float64, token+1)
			maximum := math.Inf(-1)
			for previous := range token + 1 {
				for d := range 2 {
					scores[previous] += queries[token][head*2+d] * keys[previous][head/2*2+d] / math.Sqrt(2)
				}
				maximum = math.Max(maximum, scores[previous])
			}
			sum := 0.0
			for previous := range scores {
				scores[previous] = math.Exp(scores[previous] - maximum)
				sum += scores[previous]
			}
			for d := range 2 {
				for previous := range scores {
					joined[head*2+d] += scores[previous] / sum * vals[previous][head/2*2+d]
				}
				joined[head*2+d] /= 1 + math.Exp(-qraw[token][head*4+2+d])
			}
		}
		for output := range 2 {
			for input := range 8 {
				weight := f.out[output*8+input]
				if f.oa != nil {
					for rank := range 4 {
						weight += 2 * f.ob[output*4+rank] * f.oa[rank*8+input]
					}
				}
				result[token*2+output] += weight * joined[input]
			}
		}
	}
	return result
}

func TestFullAttentionMatchesIndependentForwardAndGradients(t *testing.T) {
	f := fixture()
	xt := tensor32(t, f.x, []int64{1, 3, 2}, true)
	weights := layers.AttentionWeights{
		Query: tensor32(t, f.q, []int64{16, 2}, false), Key: tensor32(t, f.k, []int64{4, 2}, false),
		Value: tensor32(t, f.v, []int64{4, 2}, false), Output: tensor32(t, f.out, []int64{2, 8}, false),
		QueryNorm: tensor32(t, f.qnorm, []int64{2}, false), KeyNorm: tensor32(t, f.knorm, []int64{2}, false),
	}
	adapter := &layers.AttentionLoRA{
		QueryA: tensor32(t, f.qa, []int64{4, 2}, true), QueryB: tensor32(t, f.qb, []int64{16, 4}, true),
		ValueA: tensor32(t, f.va, []int64{4, 2}, true), ValueB: tensor32(t, f.vb, []int64{4, 4}, true), Alpha: 8,
	}
	cosine, sine := tensor32(t, f.cosine, []int64{3, 1}, false), tensor32(t, f.sine, []int64{3, 1}, false)
	config := layers.AttentionConfig{Heads: 4, KVHeads: 2, HeadDimension: 2, RotaryDimension: 2, Epsilon: 1e-6, MaxScoreElements: 36}
	result, err := layers.FullAttention(xt, weights, adapter, cosine, sine, config)
	result = own(t, result, err)
	near(t, values(t, result), attentionReference(f), 2e-6)
	dy := []float64{0.3, -0.5, 0.2, 0.7, -0.1, 0.6}
	gradients, err := torch.Grad([]*torch.Tensor{result}, []*torch.Tensor{xt, adapter.QueryA, adapter.QueryB, adapter.ValueA, adapter.ValueB}, []*torch.Tensor{tensor32(t, dy, []int64{1, 3, 2}, false)}, false, false)
	if err != nil {
		t.Fatal(err)
	}
	closeGradients(t, gradients)
	for which, gradient := range gradients {
		var data []float64
		switch which {
		case 0:
			data = f.x
		case 1:
			data = f.qa
		case 2:
			data = f.qb
		case 3:
			data = f.va
		case 4:
			data = f.vb
		}
		finiteDifference := make([]float64, len(data))
		for index := range data {
			original := data[index]
			const step = 1e-5
			data[index] = original + step
			plus := attentionReference(f)
			data[index] = original - step
			minus := attentionReference(f)
			data[index] = original
			for output := range plus {
				finiteDifference[index] += dy[output] * (plus[output] - minus[output]) / (2 * step)
			}
		}
		near(t, values(t, gradient), finiteDifference, 3e-6)
	}
	config.MaxScoreElements = 35
	if result, err := layers.FullAttention(xt, weights, adapter, cosine, sine, config); err == nil || result != nil {
		t.Fatal("attention accepted insufficient score budget")
	}
}

func TestPartialRotaryLeavesRemainingHeadCoordinatesUnrotated(t *testing.T) {
	x := []float64{1, 0, 0.5, -0.5, 0, 1, -1, 2}
	identity := []float64{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}
	query := append(append([]float64(nil), identity...), make([]float64, 16)...)
	weights := layers.AttentionWeights{
		Query: tensor32(t, query, []int64{8, 4}, false), Key: tensor32(t, identity, []int64{4, 4}, false),
		Value: tensor32(t, identity, []int64{4, 4}, false), Output: tensor32(t, identity, []int64{4, 4}, false),
		QueryNorm: tensor32(t, make([]float64, 4), []int64{4}, false), KeyNorm: tensor32(t, make([]float64, 4), []int64{4}, false),
	}
	cosine, sine := []float64{1, math.Cos(0.5)}, []float64{0, math.Sin(0.5)}
	result, err := layers.FullAttention(tensor32(t, x, []int64{1, 2, 4}, true), weights, nil,
		tensor32(t, cosine, []int64{2, 1}, false), tensor32(t, sine, []int64{2, 1}, false),
		layers.AttentionConfig{Heads: 1, KVHeads: 1, HeadDimension: 4, RotaryDimension: 2, Epsilon: 1e-6, MaxScoreElements: 4})
	result = own(t, result, err)
	qk := append([]float64(nil), x...)
	for token := range 2 {
		variance := 1e-6
		for d := range 4 {
			variance += x[token*4+d] * x[token*4+d] / 4
		}
		for d := range 4 {
			qk[token*4+d] /= math.Sqrt(variance)
		}
		a, b := qk[token*4], qk[token*4+1]
		qk[token*4], qk[token*4+1] = a*cosine[token]-b*sine[token], b*cosine[token]+a*sine[token]
	}
	score0, score1 := 0.0, 0.0
	for d := range 4 {
		score0 += qk[4+d] * qk[d] / 2
		score1 += qk[4+d] * qk[4+d] / 2
	}
	p0 := 1 / (1 + math.Exp(score1-score0))
	expected := make([]float64, 8)
	for d := range 4 {
		expected[d] = x[d] / 2
		expected[4+d] = (p0*x[d] + (1-p0)*x[4+d]) / 2
	}
	near(t, values(t, result), expected, 2e-6)
}
