//go:build libtorch && cgo

package decoder_test

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/tayi-ai/arandu-llama/checkpoint"
	"github.com/tayi-ai/arandu-llama/training/decoder"
	"github.com/tayi-ai/arandu-llama/training/layers"
	"github.com/tayi-ai/arandu-llama/training/torch"
)

// coverageModules names every adapted module of a layer kind in canonical
// order, written out here rather than read from the code under test.
var coverageModules = map[bool][]string{
	true:  {"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"},
	false: {"linear_attn.out_proj", "linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.in_proj_b", "linear_attn.in_proj_a", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"},
}

// coverageTargets pairs each module above with its layer target.
var coverageTargets = map[string]layers.LoRATarget{
	"self_attn.q_proj": layers.LoRAQuery, "self_attn.k_proj": layers.LoRAKey, "self_attn.v_proj": layers.LoRAValue, "self_attn.o_proj": layers.LoRAOutput,
	"linear_attn.out_proj": layers.LoRALinearOutput, "linear_attn.in_proj_qkv": layers.LoRALinearQKV, "linear_attn.in_proj_z": layers.LoRALinearZ,
	"linear_attn.in_proj_b": layers.LoRALinearBeta, "linear_attn.in_proj_a": layers.LoRALinearAlpha,
	"mlp.gate_proj": layers.LoRAGate, "mlp.up_proj": layers.LoRAUp, "mlp.down_proj": layers.LoRADown,
}

// coverageShapes is each base projection of the tiny fixture as [out,in].
var coverageShapes = map[string][2]int64{
	"self_attn.q_proj": {8, 4}, "self_attn.k_proj": {2, 4}, "self_attn.v_proj": {2, 4}, "self_attn.o_proj": {4, 4},
	"linear_attn.out_proj": {4, 4}, "linear_attn.in_proj_qkv": {8, 4}, "linear_attn.in_proj_z": {4, 4},
	"linear_attn.in_proj_b": {2, 4}, "linear_attn.in_proj_a": {2, 4},
	"mlp.gate_proj": {5, 4}, "mlp.up_proj": {5, 4}, "mlp.down_proj": {4, 5},
}

type coverageParameter struct {
	name   string
	layer  int
	target layers.LoRATarget
	b      bool
	shape  []int64
}

// coverageLayout is every adapted tensor of the coverage fixture in the
// canonical order: layer, module, then A before B.
func coverageLayout(f *fixture) []coverageParameter {
	var result []coverageParameter
	for index, layer := range f.model.Layers {
		for _, module := range coverageModules[layer.Weights.Full != nil] {
			shape := coverageShapes[module]
			for _, b := range []bool{false, true} {
				letter, dims := "A", []int64{2, shape[1]}
				if b {
					letter, dims = "B", []int64{shape[0], 2}
				}
				result = append(result, coverageParameter{
					name:  fmt.Sprintf("base_model.model.model.language_model.layers.%d.%s.lora_%s.default.weight", index, module, letter),
					layer: index, target: coverageTargets[module], b: b, shape: dims,
				})
			}
		}
	}
	return result
}

func (p coverageParameter) slot(f *fixture) **torch.Tensor {
	a, b := f.model.Layers[p.layer].Adapter.Pair(p.target)
	if p.b {
		return b
	}
	return a
}

// newCoverageFixture keeps the first eight layers of the 32-layer fixture,
// two full-attention blocks among six recurrent ones, and replaces their q/v
// adapters with rank-2 pairs on every projection of every layer. B is scaled
// by bScale, so zero reproduces a fresh initializer.
func newCoverageFixture(t *testing.T, bScale float64) *fixture {
	f := newFixture(t, torch.Float32)
	f.model.Layers = f.model.Layers[:8]
	for index := range f.model.Layers {
		f.model.Layers[index].Adapter = &layers.AttentionLoRA{Alpha: 4}
	}
	for i, p := range coverageLayout(f) {
		scale := 0.1
		if p.b {
			scale = bScale
		}
		*p.slot(f) = tensor(t, values(int(p.shape[0]*p.shape[1]), float64(i)*0.37, scale), p.shape, true)
	}
	return f
}

func TestFullCoverageVJPReturnsEveryPairInCanonicalOrder(t *testing.T) {
	f := newCoverageFixture(t, 0.1)
	before := baseHash(t, f)
	layout := coverageLayout(f)
	gradients := backward(t, f, forward(t, f))
	if len(layout) != 2*(2*7+6*8) || len(gradients) != len(layout) {
		t.Fatalf("got %d gradients for %d adapted tensors", len(gradients), len(layout))
	}
	for i, gradient := range gradients {
		if gradient.Name != layout[i].name {
			t.Fatalf("parameter %d: got %s, expected %s", i, gradient.Name, layout[i].name)
		}
		assertDetachedFinite(t, gradient.Value, torch.Float32)
		info, err := gradient.Value.Info()
		if err != nil || fmt.Sprint(info.Shape) != fmt.Sprint(layout[i].shape) {
			t.Fatalf("parameter %d gradient geometry differs: %v", i, err)
		}
	}
	if baseHash(t, f) != before {
		t.Fatal("frozen base bytes changed")
	}
}

// A fresh adapter (every B exactly zero) is the base model: the logits equal
// the unadapted model's value for value, every A cotangent is exactly zero,
// and every B cotangent reaches the loss. The last proves each of the twelve
// kinds of pair is wired into the graph at every layer.
func TestFreshFullCoverageStartsAtTheBaseModel(t *testing.T) {
	f := newCoverageFixture(t, 0)
	adapted := read(t, forward(t, f).Logits)
	gradients := backward(t, f, forward(t, f))
	layout := coverageLayout(f)
	for i, gradient := range gradients {
		data := read(t, gradient.Value)
		nonzero := false
		for _, value := range data {
			nonzero = nonzero || value != 0
		}
		if layout[i].b != nonzero {
			t.Fatalf("%s: nonzero cotangent %t, expected %t", layout[i].name, nonzero, layout[i].b)
		}
	}
	adapters := make([]*layers.AttentionLoRA, len(f.model.Layers))
	for index := range f.model.Layers {
		adapters[index], f.model.Layers[index].Adapter = f.model.Layers[index].Adapter, nil
	}
	frozen := read(t, forward(t, f).Logits)
	for index := range f.model.Layers {
		f.model.Layers[index].Adapter = adapters[index]
	}
	for i := range frozen {
		if adapted[i] != frozen[i] {
			t.Fatalf("logit %d: fresh adapter %.9g, base %.9g", i, adapted[i], frozen[i])
		}
	}
}

// Central differences of the model-level forward check one coordinate of
// every A and B of the first recurrent layer and of the first full-attention
// layer's attention pairs, through every later decoder and the head.
func TestFullCoverageCotangentsMatchModelCentralDifferences(t *testing.T) {
	f := newCoverageFixture(t, 0.1)
	layout := coverageLayout(f)
	gradients := backward(t, f, forward(t, f))
	checked, worst := 0, 0.0
	for i, p := range layout {
		if p.layer != 0 && (p.layer != 3 || !p.target.FullAttention()) {
			continue
		}
		analytic := read(t, gradients[i].Value)
		coordinate := 0
		for j := range analytic {
			if math.Abs(float64(analytic[j])) > math.Abs(float64(analytic[coordinate])) {
				coordinate = j
			}
		}
		if math.Abs(float64(analytic[coordinate])) < 1e-5 {
			t.Fatalf("%s: weak derivative %.9g", p.name, analytic[coordinate])
		}
		pointer := p.slot(f)
		original := *pointer
		baseline := read(t, original)
		const step = 0.005
		var losses [2]float64
		var coordinates [2]float32
		for direction, sign := range []float32{1, -1} {
			perturbed := append([]float32(nil), baseline...)
			perturbed[coordinate] += sign * step
			coordinates[direction] = perturbed[coordinate]
			*pointer = tensor(t, perturbed, p.shape, true)
			losses[direction] = scalarLoss(t, f)
			*pointer = original
		}
		central := (losses[0] - losses[1]) / float64(coordinates[0]-coordinates[1])
		absolute := math.Abs(float64(analytic[coordinate]) - central)
		// The same FP32 allowance as the q/v model-level check.
		if absolute > 2e-5+0.002*math.Abs(float64(analytic[coordinate])) {
			t.Fatalf("%s coordinate %d: analytic %.9g central %.9g", p.name, coordinate, analytic[coordinate], central)
		}
		worst = math.Max(worst, absolute)
		checked++
	}
	if checked != 2*(8+4) {
		t.Fatalf("checked %d tensors", checked)
	}
	t.Logf("%d model-level central differences over layers 0 and 3: maximum absolute error %.3g", checked, worst)
}

func coverageDigest(names []string, values [][]float32) string {
	h := sha256.New()
	var scalar [4]byte
	for i, name := range names {
		_, _ = h.Write([]byte(name))
		for _, value := range values[i] {
			binary.LittleEndian.PutUint32(scalar[:], math.Float32bits(value))
			_, _ = h.Write(scalar[:])
		}
	}
	return hex.EncodeToString(h.Sum(nil))
}

// Replacement and restoration address every pair by its registry position:
// each tensor lands in its own slot, the digest is the independent name/value
// hash, and a restored checkpoint reproduces the replaced model's logits.
func TestFullCoverageReplaceAndRestoreInstallEveryPair(t *testing.T) {
	f := newCoverageFixture(t, 0.1)
	layout := coverageLayout(f)
	loaded := &decoder.LoadedTextModel{Model: f.model}
	names := make([]string, len(layout))
	for i, p := range layout {
		names[i] = p.name
		loaded.Parameters = append(loaded.Parameters, decoder.InitialParameter{Name: p.name, Value: *p.slot(f)})
	}
	t.Cleanup(func() { _ = loaded.Close() })
	next := make([][]float32, len(layout))
	var flat []float32
	for i, p := range layout {
		next[i] = values(int(p.shape[0]*p.shape[1]), float64(i)*0.53+0.1, 0.05)
		flat = append(flat, next[i]...)
	}
	digest, err := loaded.ReplaceParameters(context.Background(), flat)
	if err != nil || digest != coverageDigest(names, next) {
		t.Fatalf("replacement digest %s: %v", digest, err)
	}
	for i, p := range layout {
		if loaded.Parameters[i].Value != *p.slot(f) {
			t.Fatalf("%s: registry and layer hold different handles", p.name)
		}
		got := read(t, *p.slot(f))
		for j := range got {
			if math.Float32bits(got[j]) != math.Float32bits(next[i][j]) {
				t.Fatalf("%s element %d landed elsewhere", p.name, j)
			}
		}
	}
	replaced := read(t, forward(t, f).Logits)
	tensors := make([]checkpoint.Float32Tensor, len(layout))
	for i, p := range layout {
		tensors[i] = checkpoint.Float32Tensor{Name: p.name, Shape: []uint64{uint64(p.shape[0]), uint64(p.shape[1])}, Values: next[i]}
	}
	var body bytes.Buffer
	limits := checkpoint.Limits{MaxHeaderBytes: 1 << 20, MaxTensors: 1024, MaxDimensions: 2, MaxMetadataEntries: 1, MaxChunkBytes: 4096}
	receipt, err := checkpoint.WriteFloat32(context.Background(), &body, tensors, limits)
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "adapter_model.safetensors")
	if err := os.WriteFile(path, body.Bytes(), 0600); err != nil {
		t.Fatal(err)
	}
	if _, err := loaded.ReplaceParameters(context.Background(), make([]float32, len(flat))); err != nil {
		t.Fatal(err)
	}
	restored, err := loaded.RestoreAdapter(context.Background(), decoder.AdapterCheckpoint{Path: path, FileSHA256: receipt.SHA256,
		ParametersSHA256: digest, MaxBytes: receipt.Bytes, MaxWorkingBytes: 64 << 20, Limits: limits})
	if err != nil || restored != digest {
		t.Fatalf("restoration digest %s: %v", restored, err)
	}
	again := read(t, forward(t, f).Logits)
	for i := range replaced {
		if math.Float32bits(replaced[i]) != math.Float32bits(again[i]) {
			t.Fatal("restored adapter changed the logits")
		}
	}
}
