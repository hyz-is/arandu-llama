//go:build libtorch && cgo

package local

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"strings"
	"testing"

	testfixture "github.com/tayi-ai/arandu-llama/tests/Unit/Training/Fixture"
	"github.com/tayi-ai/arandu-llama/training/decoder"
	"github.com/tayi-ai/arandu-llama/training/layers"
	"github.com/tayi-ai/arandu-llama/training/pipeline"
	"github.com/tayi-ai/arandu-llama/training/sequence"
	"github.com/tayi-ai/arandu-llama/training/torch"
)

// coverageConfig is the text config of coverageCPUModel: a recurrent layer
// then a full-attention layer, hidden 4, MLP width 5 and vocabulary 6.
func coverageConfig(t *testing.T) []byte {
	t.Helper()
	body, err := json.Marshal(map[string]any{"tie_word_embeddings": false, "text_config": map[string]any{
		"hidden_size": 4, "vocab_size": 6, "intermediate_size": 5, "num_hidden_layers": 2,
		"num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 2, "layer_types": []string{"linear_attention", "full_attention"},
		"linear_num_key_heads": 1, "linear_num_value_heads": 2, "linear_key_head_dim": 2, "linear_value_head_dim": 2, "linear_conv_kernel_dim": 4,
		"rms_norm_eps": 1e-6, "hidden_act": "silu", "attention_bias": false, "attention_dropout": 0, "attn_output_gate": true, "tie_word_embeddings": false,
		"rope_parameters": map[string]any{"rope_theta": 10000, "rope_type": "default", "partial_rotary_factor": 1},
	}})
	if err != nil {
		t.Fatal(err)
	}
	return body
}

// coverageProjections is what the full declaration implies for that config
// at rank 1, in plan order.
func coverageProjections(t *testing.T) []decoder.InitialProjection {
	t.Helper()
	projections, err := decoder.CoverageProjections(coverageConfig(t), decoder.FullAdapterCoverage(), 1)
	if err != nil {
		t.Fatal(err)
	}
	return projections
}

var coverageModuleTargets = map[string]layers.LoRATarget{
	"self_attn.q_proj": layers.LoRAQuery, "self_attn.k_proj": layers.LoRAKey, "self_attn.v_proj": layers.LoRAValue, "self_attn.o_proj": layers.LoRAOutput,
	"linear_attn.out_proj": layers.LoRALinearOutput, "linear_attn.in_proj_qkv": layers.LoRALinearQKV, "linear_attn.in_proj_z": layers.LoRALinearZ,
	"linear_attn.in_proj_b": layers.LoRALinearBeta, "linear_attn.in_proj_a": layers.LoRALinearAlpha,
	"mlp.gate_proj": layers.LoRAGate, "mlp.up_proj": layers.LoRAUp, "mlp.down_proj": layers.LoRADown,
}

// coverageCPUModel is sftCPUModel with a recurrent layer before its
// full-attention layer and a rank-1, alpha-2 pair on every projection of
// both, registered in the order the full declaration implies.
func coverageCPUModel(t *testing.T) (*decoder.LoadedTextModel, func() error) {
	t.Helper()
	m := &decoder.LoadedTextModel{Model: &decoder.TextModel{Epsilon: 1e-6, Layers: make([]decoder.Layer, 2)}}
	var base []*torch.Tensor
	tensor := func(shape []int64, norm, leaf bool) *torch.Tensor {
		n := 1
		for _, d := range shape {
			n *= int(d)
		}
		v := make([]float32, n)
		for i := range v {
			v[i] = float32(.2 * math.Sin(float64(i+1)*.71))
			if norm {
				v[i] = 1
			}
		}
		x, e := torch.FromFloat32(v, shape, torch.CPUDevice(), leaf)
		if e != nil {
			t.Fatal(e)
		}
		if !leaf {
			base = append(base, x)
		}
		return x
	}
	m.Model.Embedding = tensor([]int64{6, 4}, false, false)
	m.Model.Head = tensor([]int64{6, 4}, false, false)
	m.Model.FinalNorm = tensor([]int64{4}, true, false)
	config := layers.DecoderConfig{Epsilon: 1e-6, MaxInputElements: 12,
		Full: layers.AttentionConfig{Heads: 2, KVHeads: 1, HeadDimension: 2, RotaryDimension: 2, Epsilon: 1e-6, MaxScoreElements: 18},
		Linear: layers.LinearAttentionConfig{KeyHeads: 1, ValueHeads: 2, KeyDimension: 2, ValueDimension: 2, Epsilon: 1e-6, MaxWorkingElements: 1 << 20,
			Sequence: sequence.SequenceLimits{ChunkTokens: 2, MaxTokens: 8, MaxOwnedElements: 1 << 20}}}
	for index := range m.Model.Layers {
		w := layers.DecoderWeights{InputNorm: tensor([]int64{4}, true, false), PostAttentionNorm: tensor([]int64{4}, true, false), Gate: tensor([]int64{5, 4}, false, false), Up: tensor([]int64{5, 4}, false, false), Down: tensor([]int64{4, 5}, false, false)}
		if index == 0 {
			w.Linear = &layers.LinearAttentionWeights{QKV: tensor([]int64{8, 4}, false, false), Z: tensor([]int64{4, 4}, false, false), Beta: tensor([]int64{2, 4}, false, false), Alpha: tensor([]int64{2, 4}, false, false),
				Convolution: tensor([]int64{8, 1, 4}, false, false), ALog: tensor([]int64{2}, false, false), DTBias: tensor([]int64{2}, false, false), Norm: tensor([]int64{2}, true, false), Output: tensor([]int64{4, 4}, false, false)}
		} else {
			w.Full = &layers.AttentionWeights{Query: tensor([]int64{8, 4}, false, false), Key: tensor([]int64{2, 4}, false, false), Value: tensor([]int64{2, 4}, false, false), Output: tensor([]int64{4, 4}, false, false), QueryNorm: tensor([]int64{2}, true, false), KeyNorm: tensor([]int64{2}, true, false)}
		}
		m.Model.Layers[index] = decoder.Layer{Device: torch.CPUDevice(), Weights: w, Adapter: &layers.AttentionLoRA{Alpha: 2}, Config: config}
	}
	for _, projection := range coverageProjections(t) {
		var index int
		if _, e := fmt.Sscanf(projection.Name, "base_model.model.model.language_model.layers.%d.", &index); e != nil {
			t.Fatal(e)
		}
		target, known := coverageModuleTargets[strings.TrimPrefix(projection.Name, fmt.Sprintf("base_model.model.model.language_model.layers.%d.", index))]
		if !known {
			t.Fatal("undeclared module", projection.Name)
		}
		a, b := m.Model.Layers[index].Adapter.Pair(target)
		*a, *b = tensor([]int64{projection.Rank, projection.Input}, false, true), tensor([]int64{projection.Output, projection.Rank}, false, true)
		m.Parameters = append(m.Parameters, decoder.InitialParameter{Name: projection.Name + ".lora_A.default.weight", Value: *a},
			decoder.InitialParameter{Name: projection.Name + ".lora_B.default.weight", Value: *b})
		m.Summary.AdapterElements += projection.Rank * (projection.Input + projection.Output)
	}
	// ReplaceParameters transfers ownership of fixture leaves into the real owner.
	var flat []float32
	for _, p := range m.Parameters {
		v, e := p.Value.Float32Values()
		if e != nil {
			t.Fatal(e)
		}
		flat = append(flat, v...)
	}
	if _, e := m.ReplaceParameters(context.Background(), flat); e != nil {
		t.Fatal(e)
	}
	close := func() error {
		e := m.Close()
		for _, b := range base {
			e = errors.Join(e, b.Close())
		}
		return e
	}
	t.Cleanup(func() { _ = close() })
	return m, close
}

// coverageFreshFixture is the fresh three-step stage over the coverage model:
// the recipe declares the full coverage and pins the initializer of its
// projections by the independent CPU reference.
func coverageFreshFixture(t *testing.T) (StageConfig, pipeline.StageContext, []example, decoder.InitialAdapterSpec) {
	t.Helper()
	c, stage, rows, spec := sftFreshFixture(t)
	spec.Projections, spec.ExpectedSHA256 = coverageProjections(t), ""
	spec.ExpectedSHA256 = testfixture.Digest(t, spec)
	recipe := &c.Protocol.Local
	recipe.Initializer, recipe.Identity.InitialAdapterSHA256 = spec, spec.ExpectedSHA256
	recipe.Assembly.AdapterCoverage, recipe.Assembly.AdapterRank, recipe.Assembly.AdapterAlpha = decoder.FullAdapterCoverage(), 1, 2
	c.Protocol.Limits.MaxParameters = 256
	c.Protocol.Limits.Checkpoint.MaxTensors, c.Protocol.Limits.Checkpoint.MaxHeaderBytes = 64, 64<<10
	sftPin(t, &c)
	return c, stage, rows, spec
}

// The model's registry is the declared coverage, name for name, so the pinned
// initializer, the checkpoints and the moments all cover every projection.
func TestCoverageModelRegistryIsTheDeclaredCoverage(t *testing.T) {
	m, close := coverageCPUModel(t)
	defer close()
	projections := coverageProjections(t)
	if len(m.Parameters) != 2*len(projections) || len(projections) != 15 || m.Summary.AdapterElements != 126 {
		t.Fatal("coverage registry size differs", len(m.Parameters), len(projections), m.Summary.AdapterElements)
	}
	for i, projection := range projections {
		if m.Parameters[2*i].Name != projection.Name+".lora_A.default.weight" || m.Parameters[2*i+1].Name != projection.Name+".lora_B.default.weight" {
			t.Fatal("registry order differs from the declaration", i, m.Parameters[2*i].Name)
		}
	}
	c, _, _, _ := coverageFreshFixture(t)
	legacy, _, _, _ := sftFreshFixture(t)
	if c.Protocol.Local.Digest() == legacy.Protocol.Local.Digest() || !strings.Contains(string(mustJSON(t, c.Protocol.Local)), `"AdapterCoverage":{"version":1`) {
		t.Fatal("the declared coverage is not part of the recipe identity")
	}
}

func mustJSON(t *testing.T, v any) []byte {
	t.Helper()
	body, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	return body
}

// Three fresh updates with every projection adapted, delivered as two model
// residencies: the second reloads the checkpoint the first wrote. Losses,
// final adapter and moments equal an uninterrupted independent AdamW run from
// the pinned initializer bit for bit.
func TestNativeSFTStageFreshFullCoverageMatchesAnIndependentAdamWFromZero(t *testing.T) {
	c, stage, rows, spec := coverageFreshFixture(t)
	calls, closed := 0, 0
	h, e := newStage(c, sftExecutorWith(t, coverageCPUModel, &calls, &closed))
	if e != nil {
		t.Fatal(e)
	}
	var results []pipeline.StageResult
	if e := h.Run(context.Background(), stage, sftCommit(t, h, &stage, &results)); e != nil {
		t.Fatal(e)
	}
	if calls != 2 || closed != calls || len(results) != 3 {
		t.Fatal("delivery or step accounting differs", calls, closed, len(results))
	}
	sameAsTrajectoryWith(t, coverageCPUModel, h, c, stage, rows, results, spec)
}

func TestNativeSFTStageFreshFullCoverageResumesAfterALostProcess(t *testing.T) {
	c, stage, rows, spec := coverageFreshFixture(t)
	calls, closed := 0, 0
	delivery := sftExecutorWith(t, coverageCPUModel, &calls, &closed)
	lost := errors.New("process lost after its first delivery")
	first, e := newStage(c, func(ctx context.Context, cfg Config, hooks *stepHooks) error {
		if e := delivery(ctx, cfg, hooks); e != nil {
			return e
		}
		return lost
	})
	if e != nil {
		t.Fatal(e)
	}
	var results []pipeline.StageResult
	if e := first.Run(context.Background(), stage, sftCommit(t, first, &stage, &results)); !errors.Is(e, lost) || len(results) != 2 {
		t.Fatal("first delivery did not commit its two steps before the loss", e, len(results))
	}
	next, e := newStage(c, sftExecutorWith(t, coverageCPUModel, &calls, &closed))
	if e != nil {
		t.Fatal(e)
	}
	stage.Execution.Generation++
	if e := next.Run(context.Background(), stage, sftCommit(t, next, &stage, &results)); e != nil {
		t.Fatal(e)
	}
	if calls != 2 || closed != calls {
		t.Fatal("resume recomputed a committed step", calls, closed)
	}
	sameAsTrajectoryWith(t, coverageCPUModel, next, c, stage, rows, results, spec)
}

// The readout restores each fully covered checkpoint and reproduces the
// loss_before its next step logged, bitwise, and the initializer reproduces
// step 1's.
func TestNativeLossReadoutReproducesFullCoverageLossBefore(t *testing.T) {
	c, stage, rows, spec := coverageFreshFixture(t)
	cfg, steps, rows, spec, _ := lossTrajectoryWith(t, coverageCPUModel, c, stage, rows, spec)
	var logged []stepManifest
	for _, step := range steps {
		logged = append(logged, readLossManifest(t, step))
	}
	requests := []LossRequest{{"", rows[0].ID}, {steps[0], rows[1].ID}, {steps[1], rows[2].ID}}
	results, e := readLossesWith(t, coverageCPUModel, cfg, requests)
	if e != nil {
		t.Fatal(e)
	}
	for i, r := range results {
		adapter := spec.ExpectedSHA256
		if i > 0 {
			adapter = logged[i-1].UpdatedAdapterSHA
		}
		if r.Step != uint64(i) || r.AdapterSHA256 != adapter || math.Float64bits(r.Loss) != math.Float64bits(logged[i].LossBefore) {
			t.Fatalf("reading %d: step %d loss %.17g adapter %s; logged loss_before %.17g", i, r.Step, r.Loss, r.AdapterSHA256, logged[i].LossBefore)
		}
	}
}
