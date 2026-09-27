//go:build libtorch && cgo

package decoder

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"math"
	"strings"
	"testing"

	"github.com/tayi-ai/arandu-llama/checkpoint"
	"github.com/tayi-ai/arandu-llama/training/sequence"
	"github.com/tayi-ai/arandu-llama/training/torch"
)

// tinyCoverageDocuments is a four-layer hybrid config, three recurrent layers
// then one full-attention layer, and a reference admitting its plan under
// limits. Hashes are placeholders: this plan is only wired, never loaded.
func tinyCoverageDocuments(t *testing.T, limits AssemblyLimits) (index, config, reference []byte, identity AssemblyIdentity) {
	t.Helper()
	text := map[string]any{
		"hidden_size": 4, "vocab_size": 6, "intermediate_size": 5, "num_hidden_layers": 4,
		"num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 2,
		"layer_types":          []string{"linear_attention", "linear_attention", "linear_attention", "full_attention"},
		"linear_num_key_heads": 1, "linear_num_value_heads": 2, "linear_key_head_dim": 2, "linear_value_head_dim": 2, "linear_conv_kernel_dim": 4,
		"rms_norm_eps": 1e-6, "hidden_act": "silu", "attention_bias": false, "attention_dropout": 0, "attn_output_gate": true, "tie_word_embeddings": false,
		"rope_parameters": map[string]any{"rope_theta": 10000, "rope_type": "default", "partial_rotary_factor": 1},
	}
	marshal := func(v any) []byte {
		b, err := json.Marshal(v)
		if err != nil {
			t.Fatal(err)
		}
		return b
	}
	config = marshal(map[string]any{"tie_word_embeddings": false, "text_config": text})
	geometry, err := validateAssemblyConfig(config)
	if err != nil {
		t.Fatal(err)
	}
	weights, adapters := assemblyGeometry(geometry, limits)
	weightMap := map[string]string{}
	var rows []assemblyReference
	for _, item := range append(weights, adapters...) {
		rows = append(rows, assemblyReference{Name: item.ReferenceName, Kind: "parameter", DType: assemblyDTypeName(item.DType), Device: fmt.Sprintf("cuda:%d", item.Device.Index), Shape: item.Shape, Bytes: item.Bytes, SHA256: strings.Repeat("0", 64), RequiresGrad: item.Trainable})
		if !item.Trainable {
			weightMap[item.SourceName] = "model-00001-of-00001.safetensors"
		}
	}
	index, reference = marshal(map[string]any{"weight_map": weightMap}), marshal(map[string]any{"tensors": rows})
	digest := func(b []byte) string { return fmt.Sprintf("%x", sha256.Sum256(b)) }
	return index, config, reference, AssemblyIdentity{IndexSHA256: digest(index), ConfigSHA256: digest(config), ReferenceSHA256: digest(reference), InitialAdapterSHA256: strings.Repeat("1", 64)}
}

func tinyCoverageLimits(coverage *AdapterCoverage) AssemblyLimits {
	return AssemblyLimits{HeaderLimits: checkpoint.DefaultLimits(), TensorCopyBytes: 1 << 20, PersistentBytes: []int64{1 << 20}, DeviceByLayer: make([]int, 4),
		AdapterRank: 2, AdapterAlpha: 4, HashChunkBytes: 64, MaxInputElements: 64, MaxScoreElements: 64, MaxWorkingElements: 1 << 20,
		Sequence: sequence.SequenceLimits{ChunkTokens: 2, MaxTokens: 8, MaxOwnedElements: 1 << 20}, AdapterCoverage: coverage}
}

// wiredCoverageModel wires every plan tensor as an FP32 CPU tensor, exactly as
// LoadTextAssembly wires placed tensors, then places each layer on CPU.
func wiredCoverageModel(t *testing.T, plan *AssemblyPlan, bScale float32) *TextModel {
	t.Helper()
	values := map[string]*torch.Tensor{}
	for i, item := range plan.Tensors() {
		data := make([]float32, item.Bytes/assemblyWidth(item.DType))
		for j := range data {
			data[j] = float32(0.2 * math.Sin(float64(i)*1.3+float64(j)*0.71))
			if strings.HasSuffix(item.ReferenceName, "norm.weight") {
				data[j] = 0.05 * data[j]
			}
			if strings.HasSuffix(item.ReferenceName, ".lora_B.default.weight") {
				data[j] *= bScale
			}
		}
		value, err := torch.FromFloat32(data, item.Shape, torch.CPUDevice(), item.Trainable)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { _ = value.Close() })
		values[item.ReferenceName] = value
	}
	model := &TextModel{Layers: make([]Layer, plan.geometry.Layers), Epsilon: plan.geometry.Epsilon}
	wireAssembly(model, values, plan.geometry, plan.limits, false)
	cosine, err := torch.FromFloat32([]float32{1, float32(math.Cos(0.4)), float32(math.Cos(0.8))}, []int64{3, 1}, torch.CPUDevice(), false)
	if err != nil {
		t.Fatal(err)
	}
	sine, err := torch.FromFloat32([]float32{0, float32(math.Sin(0.4)), float32(math.Sin(0.8))}, []int64{3, 1}, torch.CPUDevice(), false)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _, _ = cosine.Close(), sine.Close() })
	for i := range model.Layers {
		model.Layers[i].Device = torch.CPUDevice()
		if model.Layers[i].Weights.Full != nil {
			model.Layers[i].Cosine, model.Layers[i].Sine = cosine, sine
		}
	}
	return model
}

func assemblyWidth(dtype torch.DType) int64 {
	if dtype == torch.Float32 {
		return 4
	}
	return 2
}

// The loader installs the initializer in plan order and every update checks
// that order against the registry, so the two must be one order: layer, the
// declared attention modules, then the declared MLP modules, A before B. The
// wired model runs on CPU with every base under its coverage name.
func TestWiredCoverageRegistryIsThePlanOrder(t *testing.T) {
	for name, coverage := range map[string]*AdapterCoverage{
		"q_v":     nil,
		"full":    FullAdapterCoverage(),
		"partial": {Version: 1, FullAttention: []string{"k_proj", "o_proj"}, LinearAttention: []string{"in_proj_z", "in_proj_a"}, MLP: []string{"down_proj"}},
	} {
		t.Run(name, func(t *testing.T) {
			limits := tinyCoverageLimits(coverage)
			index, config, reference, identity := tinyCoverageDocuments(t, limits)
			plan, err := PlanTextAssembly(index, config, reference, identity, limits)
			if err != nil {
				t.Fatal(err)
			}
			model := wiredCoverageModel(t, plan, 1)
			registry := adapterRegistry(model)
			if len(registry) != len(plan.adapters) {
				t.Fatalf("registry %d tensors, plan %d", len(registry), len(plan.adapters))
			}
			for i, slot := range registry {
				if slot.name != plan.adapters[i].ReferenceName {
					t.Fatalf("tensor %d: registry %s, plan %s", i, slot.name, plan.adapters[i].ReferenceName)
				}
			}
			snapshot, err := model.Forward(context.Background(), []int64{1, 4, 2}, Limits{MaxTokens: 3, LogitRows: 2, MaxCheckpointBytes: 1 << 20})
			if err != nil {
				t.Fatal(err)
			}
			defer snapshot.Close()
			seed, err := torch.FromFloat32(make([]float32, 12), []int64{1, 2, 6}, torch.CPUDevice(), false)
			if err != nil {
				t.Fatal(err)
			}
			defer seed.Close()
			gradients, err := model.VJP(context.Background(), snapshot, seed)
			if err != nil {
				t.Fatal(err)
			}
			defer gradients.Close()
			for i, gradient := range gradients {
				if gradient.Name != plan.adapters[i].ReferenceName {
					t.Fatalf("gradient %d: %s, plan %s", i, gradient.Name, plan.adapters[i].ReferenceName)
				}
			}
		})
	}
}

// With every B zero the wired full declaration is the wired q/v model with
// its adapters removed: base_layer names reach the same frozen tensors.
func TestWiredFreshCoverageIsTheBaseModel(t *testing.T) {
	logits := map[string][]float32{}
	for name, coverage := range map[string]*AdapterCoverage{"q_v": nil, "full": FullAdapterCoverage()} {
		limits := tinyCoverageLimits(coverage)
		index, config, reference, identity := tinyCoverageDocuments(t, limits)
		plan, err := PlanTextAssembly(index, config, reference, identity, limits)
		if err != nil {
			t.Fatal(err)
		}
		model := wiredCoverageModel(t, plan, 0)
		if name == "q_v" {
			for i := range model.Layers {
				model.Layers[i].Adapter = nil
			}
		}
		snapshot, err := model.Forward(context.Background(), []int64{1, 4, 2}, Limits{MaxTokens: 3, LogitRows: 2, MaxCheckpointBytes: 1 << 20})
		if err != nil {
			t.Fatal(err)
		}
		logits[name], err = snapshot.Logits.Float32Values()
		if closeErr := snapshot.Close(); err != nil || closeErr != nil {
			t.Fatal(err, closeErr)
		}
	}
	for i := range logits["q_v"] {
		if logits["q_v"][i] != logits["full"][i] {
			t.Fatalf("logit %d: fresh full coverage %.9g, base %.9g", i, logits["full"][i], logits["q_v"][i])
		}
	}
}
