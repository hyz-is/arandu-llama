package unit_test

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/tayi-ai/arandu-llama/checkpoint"
	services "github.com/tayi-ai/arandu-llama/training/adapter"
	"github.com/tayi-ai/arandu-llama/training/adapter/tests/Fixtures"
	"github.com/tayi-ai/arandu-llama/training/decoder"
)

// The fixture is a four-layer decoder, three DeltaNet layers and one full
// attention layer, with 2 key heads and 6 value heads of width 3. Three value
// heads per key head make the tiled order differ from its own inverse, which
// two per key head would not.
var fixtureLayout = vHeadLayout{keyHeads: 2, valueHeads: 6, keyHeadDim: 3, valueHeadDim: 3}

const (
	fixtureRank   = 4
	fixtureAlpha  = 8
	fixtureLayers = 4
)

func fixtureConfig(t *testing.T) []byte {
	t.Helper()
	body, err := json.Marshal(map[string]any{"tie_word_embeddings": false, "text_config": map[string]any{
		"hidden_size": 8, "vocab_size": 16, "intermediate_size": 10, "num_hidden_layers": fixtureLayers,
		"num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 4,
		"layer_types":          []string{"linear_attention", "linear_attention", "linear_attention", "full_attention"},
		"linear_num_key_heads": fixtureLayout.keyHeads, "linear_num_value_heads": fixtureLayout.valueHeads,
		"linear_key_head_dim": fixtureLayout.keyHeadDim, "linear_value_head_dim": fixtureLayout.valueHeadDim, "linear_conv_kernel_dim": 4,
		"rms_norm_eps": 1e-6, "hidden_act": "silu", "attention_bias": false, "attention_dropout": 0, "attn_output_gate": true, "tie_word_embeddings": false,
		"rope_parameters": map[string]any{"rope_theta": 10000, "rope_type": "default", "partial_rotary_factor": 0.5},
	}})
	if err != nil {
		t.Fatal(err)
	}
	return body
}

// exportFixture is a checkpoint in the trainer's namespace and a base whose
// header agrees with it. Tests change one part and export.
type exportFixture struct {
	dir         string
	projections []decoder.InitialProjection
	factors     []fixtures.RawTensor
	keys        []fixtures.GGUFKey
	base        []fixtures.BaseTensor
	request     services.Qwen35LoRAExport
}

func splitModule(t *testing.T, name string) (int, string) {
	t.Helper()
	rest, found := strings.CutPrefix(name, "base_model.model.model.language_model.layers.")
	layer, module, split := strings.Cut(rest, ".")
	var index int
	if _, err := fmt.Sscanf(layer, "%d", &index); !found || !split || err != nil {
		t.Fatalf("projection %s is outside the trainer's namespace", name)
	}
	return index, module
}

func newExportFixture(t *testing.T) *exportFixture {
	t.Helper()
	projections, err := decoder.CoverageProjections(fixtureConfig(t), decoder.FullAdapterCoverage(), fixtureRank)
	if err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	fixture := &exportFixture{dir: dir, projections: projections, keys: []fixtures.GGUFKey{
		{Key: "general.architecture", Value: "qwen35"},
		{Key: "general.type", Value: "model"},
		{Key: "qwen35.block_count", Value: uint32(fixtureLayers + 1)},
		{Key: "qwen35.nextn_predict_layers", Value: uint32(1)},
		{Key: "qwen35.attention.layer_norm_rms_epsilon", Value: float32(1e-6)},
		{Key: "qwen35.ssm.group_count", Value: uint32(fixtureLayout.keyHeads)},
		{Key: "qwen35.ssm.time_step_rank", Value: uint32(fixtureLayout.valueHeads)},
		{Key: "qwen35.ssm.state_size", Value: uint32(fixtureLayout.keyHeadDim)},
		{Key: "qwen35.ssm.inner_size", Value: uint32(fixtureLayout.valueHeads * fixtureLayout.valueHeadDim)},
	}}
	fixture.base = append(fixture.base,
		fixtures.BaseTensor{Name: "token_embd.weight", Dims: []uint64{8, 16}},
		fixtures.BaseTensor{Name: "blk.0.ssm_a", Dims: []uint64{uint64(fixtureLayout.valueHeads)}},
		// The MTP block: present in the base, outside the text stack.
		fixtures.BaseTensor{Name: fmt.Sprintf("blk.%d.attn_q.weight", fixtureLayers), Dims: []uint64{8, 16}},
	)
	stream := rand.New(rand.NewPCG(7, 11))
	for _, projection := range projections {
		layer, module := splitModule(t, projection.Name)
		fixture.base = append(fixture.base, fixtures.BaseTensor{
			Name: fmt.Sprintf("blk.%d.%s.weight", layer, ggufTensorOf[module]), Dims: []uint64{uint64(projection.Input), uint64(projection.Output)}})
		a := make([]float32, projection.Rank*projection.Input)
		b := make([]float32, projection.Output*projection.Rank)
		for i := range a {
			a[i] = float32(stream.NormFloat64())
		}
		for i := range b {
			b[i] = float32(stream.NormFloat64())
		}
		fixture.factors = append(fixture.factors,
			fixtures.RawTensor{Name: projection.Name + ".lora_A.default.weight", DType: "F32", Shape: []uint64{uint64(projection.Rank), uint64(projection.Input)}, Data: fixtures.Float32Bytes(a)},
			fixtures.RawTensor{Name: projection.Name + ".lora_B.default.weight", DType: "F32", Shape: []uint64{uint64(projection.Output), uint64(projection.Rank)}, Data: fixtures.Float32Bytes(b)})
	}
	fixture.request = services.Qwen35LoRAExport{
		Checkpoint: filepath.Join(dir, "adapter_model.safetensors"), Base: filepath.Join(dir, "base.gguf"),
		Out: filepath.Join(dir, "out", "adapter.gguf"), Rank: fixtureRank, Alpha: fixtureAlpha,
	}
	return fixture
}

// writeWithTrainer writes the checkpoint with the writer training/local uses.
func (f *exportFixture) writeWithTrainer(t *testing.T) {
	t.Helper()
	tensors := make([]checkpoint.Float32Tensor, 0, len(f.factors))
	for _, factor := range f.factors {
		tensors = append(tensors, checkpoint.Float32Tensor{Name: factor.Name, Shape: factor.Shape, Values: decodeF32(factor.Data)})
	}
	file, err := os.Create(f.request.Checkpoint)
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	if _, err := checkpoint.WriteFloat32(context.Background(), file, tensors, checkpoint.DefaultLimits()); err != nil {
		t.Fatal(err)
	}
	fixtures.WriteBaseGGUF(t, f.request.Base, f.keys, f.base)
}

// writeRaw writes the checkpoint as given, including what the trainer's
// writer refuses to produce.
func (f *exportFixture) writeRaw(t *testing.T) {
	t.Helper()
	fixtures.WriteRawSafetensors(t, f.request.Checkpoint, f.factors)
	fixtures.WriteBaseGGUF(t, f.request.Base, f.keys, f.base)
}

func decodeF32(data []byte) []float32 {
	values := make([]float32, len(data)/4)
	for i := range values {
		values[i] = math.Float32frombits(uint32(data[4*i]) | uint32(data[4*i+1])<<8 | uint32(data[4*i+2])<<16 | uint32(data[4*i+3])<<24)
	}
	return values
}

func (f *exportFixture) factor(t *testing.T, name string) *fixtures.RawTensor {
	t.Helper()
	for i := range f.factors {
		if f.factors[i].Name == name {
			return &f.factors[i]
		}
	}
	t.Fatalf("fixture has no factor %s", name)
	return nil
}

func (f *exportFixture) baseTensor(t *testing.T, name string) *fixtures.BaseTensor {
	t.Helper()
	for i := range f.base {
		if f.base[i].Name == name {
			return &f.base[i]
		}
	}
	t.Fatalf("fixture base has no tensor %s", name)
	return nil
}

const fixturePrefix = "base_model.model.model.language_model.layers."

func TestExportedLoRAIsTheConvertedDeltaOfTheCheckpoint(t *testing.T) {
	fixture := newExportFixture(t)
	fixture.writeWithTrainer(t)
	receipt, err := services.ExportQwen35LoRA(fixture.request)
	if err != nil {
		t.Fatal(err)
	}
	body, err := os.ReadFile(fixture.request.Out)
	if err != nil {
		t.Fatal(err)
	}
	for key, want := range map[string]string{"general.architecture": "qwen35", "general.type": "adapter", "adapter.type": "lora"} {
		if got, ok := ggufString(body, key); !ok || got != want {
			t.Fatalf("%s = %q, present %t; want %q", key, got, ok, want)
		}
	}
	if alpha, ok := services.AlphaOf(body); !ok || alpha != fixtureAlpha {
		t.Fatalf("adapter.lora.alpha = %v, present %t", alpha, ok)
	}
	container, err := services.ReadGGUF(fixture.request.Out)
	if err != nil {
		t.Fatal(err)
	}
	if len(receipt.Targets) != len(fixture.projections) || len(container.Tensors) != 2*len(fixture.projections) {
		t.Fatalf("%d targets and %d tensors for %d projections", len(receipt.Targets), len(container.Tensors), len(fixture.projections))
	}

	// The projections are the trainer's plan order, which is the order the
	// adapter has to be written in.
	permuted := map[string]bool{"linear_attn.in_proj_qkv": true, "linear_attn.in_proj_z": true, "linear_attn.in_proj_b": true, "linear_attn.in_proj_a": true, "linear_attn.out_proj": true}
	for i, projection := range fixture.projections {
		layer, module := splitModule(t, projection.Name)
		name := fmt.Sprintf("blk.%d.%s.weight", layer, ggufTensorOf[module])
		target := receipt.Targets[i]
		if target.Module != projection.Name || target.Tensor != name || target.Reordered != permuted[module] {
			t.Fatalf("target %d is %+v; want %s as %s, reordered %t", i, target, projection.Name, name, permuted[module])
		}
		a, b := container.Tensors[2*i], container.Tensors[2*i+1]
		rows, cols, rank := int(projection.Output), int(projection.Input), int(projection.Rank)
		if a.Name != name+".lora_a" || b.Name != name+".lora_b" || a.Type != 0 || b.Type != 0 ||
			!slices.Equal(a.Dims, []uint64{uint64(cols), uint64(rank)}) || !slices.Equal(b.Dims, []uint64{uint64(rank), uint64(rows)}) {
			t.Fatalf("tensors %d are %s %v type %d and %s %v type %d", i, a.Name, a.Dims, a.Type, b.Name, b.Dims, b.Type)
		}
		gotA, err := services.ReadTensorValues(fixture.request.Out, a.Name)
		if err != nil {
			t.Fatal(err)
		}
		gotB, err := services.ReadTensorValues(fixture.request.Out, b.Name)
		if err != nil {
			t.Fatal(err)
		}
		trainedA := decodeF32(fixture.factor(t, projection.Name+".lora_A.default.weight").Data)
		trainedB := decodeF32(fixture.factor(t, projection.Name+".lora_B.default.weight").Data)
		want := convertLikeConverter(module, product(trainedB, rows, rank, trainedA, cols), rows, cols, fixtureLayout)
		got := product(gotB, rows, rank, gotA, cols)
		for j := range want {
			if got[j] != want[j] {
				t.Fatalf("%s: delta[%d, %d] = %v; the converted checkpoint delta is %v", name, j/cols, j%cols, got[j], want[j])
			}
		}
		if permuted[module] && slices.Equal(got, product(trainedB, rows, rank, trainedA, cols)) {
			t.Fatalf("%s: the fixture does not distinguish a reordered delta from the trained one", name)
		}
	}
}

func TestExportIsDeterministic(t *testing.T) {
	fixture := newExportFixture(t)
	fixture.writeWithTrainer(t)
	first, err := services.ExportQwen35LoRA(fixture.request)
	if err != nil {
		t.Fatal(err)
	}
	again := fixture.request
	again.Out = filepath.Join(fixture.dir, "again", "adapter.gguf")
	second, err := services.ExportQwen35LoRA(again)
	if err != nil {
		t.Fatal(err)
	}
	left, _ := os.ReadFile(fixture.request.Out)
	right, _ := os.ReadFile(again.Out)
	sum := sha256.Sum256(left)
	if !bytes.Equal(left, right) || first.Digest != second.Digest || first.Digest != hex.EncodeToString(sum[:]) {
		t.Fatalf("the same checkpoint exported as %s and %s (file %x)", first.Digest, second.Digest, sum)
	}
}

// An adapter holding one MLP pair, which no value-head rule touches, has to be
// byte for byte the file WriteInitialLoRA writes for the same factors: the
// two share one container, and an exporter that drifted from it would be a
// second opinion about the format.
func TestExportOfAnUnpermutedPairIsTheInitialEncoding(t *testing.T) {
	fixture := newExportFixture(t)
	const module = fixturePrefix + "0.mlp.gate_proj"
	initial := filepath.Join(fixture.dir, "initial.gguf")
	weight := fixture.baseTensor(t, "blk.0.ffn_gate.weight")
	if _, err := services.WriteInitialLoRA(initial, "qwen35", fixtureRank, fixtureAlpha, 59, []services.LoRATarget{{Name: weight.Name, Input: weight.Dims[0], Output: weight.Dims[1]}}); err != nil {
		t.Fatal(err)
	}
	a, err := services.ReadTensorValues(initial, weight.Name+".lora_a")
	if err != nil {
		t.Fatal(err)
	}
	fixture.factors = []fixtures.RawTensor{
		{Name: module + ".lora_A.default.weight", DType: "F32", Shape: []uint64{fixtureRank, weight.Dims[0]}, Data: fixtures.Float32Bytes(a)},
		{Name: module + ".lora_B.default.weight", DType: "F32", Shape: []uint64{weight.Dims[1], fixtureRank}, Data: make([]byte, 4*fixtureRank*weight.Dims[1])},
	}
	fixture.writeWithTrainer(t)
	if _, err := services.ExportQwen35LoRA(fixture.request); err != nil {
		t.Fatal(err)
	}
	exported, _ := os.ReadFile(fixture.request.Out)
	written, _ := os.ReadFile(initial)
	if !bytes.Equal(exported, written) {
		t.Fatalf("the exported pair (%d bytes) differs from the initial encoding (%d bytes)", len(exported), len(written))
	}
}

func TestExportRefuses(t *testing.T) {
	const linear = fixturePrefix + "0.linear_attn."
	cases := []struct {
		name   string
		change func(t *testing.T, f *exportFixture)
		reason string
	}{
		{"an unknown module", func(t *testing.T, f *exportFixture) {
			f.factor(t, linear+"in_proj_a.lora_A.default.weight").Name = linear + "conv1d.lora_A.default.weight"
			f.factor(t, linear+"in_proj_a.lora_B.default.weight").Name = linear + "conv1d.lora_B.default.weight"
		}, `module "linear_attn.conv1d"`},
		{"a factor outside the layer namespace", func(t *testing.T, f *exportFixture) {
			f.factor(t, linear+"in_proj_a.lora_A.default.weight").Name = "base_model.model.lm_head.lora_A.default.weight"
		}, "is outside"},
		{"a factor without the default adapter name", func(t *testing.T, f *exportFixture) {
			f.factor(t, linear+"in_proj_a.lora_A.default.weight").Name = linear + "in_proj_a.lora_A.weight"
		}, "not a lora_A or lora_B factor"},
		{"an optimizer moment", func(t *testing.T, f *exportFixture) {
			factor := f.factor(t, linear+"in_proj_a.lora_A.default.weight")
			factor.Name = "m." + factor.Name
		}, "is outside"},
		{"the MTP layer", func(t *testing.T, f *exportFixture) {
			f.factors = append(f.factors,
				fixtures.RawTensor{Name: fmt.Sprintf("%s%d.self_attn.q_proj.lora_A.default.weight", fixturePrefix, fixtureLayers), DType: "F32", Shape: []uint64{fixtureRank, 8}, Data: make([]byte, 4*fixtureRank*8)},
				fixtures.RawTensor{Name: fmt.Sprintf("%s%d.self_attn.q_proj.lora_B.default.weight", fixturePrefix, fixtureLayers), DType: "F32", Shape: []uint64{16, fixtureRank}, Data: make([]byte, 4*fixtureRank*16)})
		}, "text stack has 4 layers"},
		{"a non-canonical layer index", func(t *testing.T, f *exportFixture) {
			f.factor(t, linear+"in_proj_a.lora_A.default.weight").Name = fixturePrefix + "00.linear_attn.in_proj_a.lora_A.default.weight"
		}, "no canonical layer index"},
		{"a missing half of a pair", func(t *testing.T, f *exportFixture) {
			f.factors = slices.DeleteFunc(f.factors, func(r fixtures.RawTensor) bool { return r.Name == linear+"in_proj_z.lora_B.default.weight" })
		}, "missing one of lora_A and lora_B"},
		{"an input width the base does not have", func(t *testing.T, f *exportFixture) {
			factor := f.factor(t, linear+"in_proj_z.lora_A.default.weight")
			factor.Shape = []uint64{fixtureRank, 9}
			factor.Data = make([]byte, 4*fixtureRank*9)
		}, "in ggml order"},
		{"an output width the base does not have", func(t *testing.T, f *exportFixture) {
			factor := f.factor(t, linear+"in_proj_z.lora_B.default.weight")
			factor.Shape = []uint64{17, fixtureRank}
			factor.Data = make([]byte, 4*fixtureRank*17)
		}, "in ggml order"},
		{"a rank other than the recipe's", func(t *testing.T, f *exportFixture) {
			f.request.Rank = 2
		}, "the recipe rank is 2"},
		{"a rank above the limit", func(t *testing.T, f *exportFixture) {
			f.request.Rank = 257
		}, "rank 257 is outside 1..256"},
		{"a pair whose factors disagree on the rank", func(t *testing.T, f *exportFixture) {
			factor := f.factor(t, linear+"in_proj_z.lora_B.default.weight")
			factor.Shape = []uint64{18, 3}
			factor.Data = make([]byte, 4*18*3)
		}, "the recipe rank is 4"},
		{"a half-precision factor", func(t *testing.T, f *exportFixture) {
			factor := f.factor(t, linear+"in_proj_z.lora_A.default.weight")
			factor.DType = "F16"
			factor.Data = factor.Data[:len(factor.Data)/2]
		}, "not a two-dimensional F32 factor"},
		{"a three-dimensional factor", func(t *testing.T, f *exportFixture) {
			factor := f.factor(t, linear+"in_proj_z.lora_A.default.weight")
			factor.Shape = []uint64{1, fixtureRank, 8}
		}, "not a two-dimensional F32 factor"},
		{"a non-finite value", func(t *testing.T, f *exportFixture) {
			factor := f.factor(t, linear+"in_proj_z.lora_A.default.weight")
			factor.Data = append(fixtures.Float32Bytes([]float32{float32(math.NaN())}), factor.Data[4:]...)
		}, "non-finite value at 0"},
		{"an empty checkpoint", func(t *testing.T, f *exportFixture) {
			f.factors = nil
		}, "holds 0 tensors"},
		{"a base of another architecture", func(t *testing.T, f *exportFixture) {
			f.keys[0].Value = "qwen3next"
		}, `declares architecture "qwen3next"`},
		{"a target the base does not have", func(t *testing.T, f *exportFixture) {
			f.factors = append(f.factors,
				fixtures.RawTensor{Name: fixturePrefix + "0.self_attn.q_proj.lora_A.default.weight", DType: "F32", Shape: []uint64{fixtureRank, 8}, Data: make([]byte, 4*fixtureRank*8)},
				fixtures.RawTensor{Name: fixturePrefix + "0.self_attn.q_proj.lora_B.default.weight", DType: "F32", Shape: []uint64{16, fixtureRank}, Data: make([]byte, 4*fixtureRank*16)})
		}, "blk.0.attn_q.weight, which the base does not have"},
		{"a base tensor of another geometry", func(t *testing.T, f *exportFixture) {
			weight := f.baseTensor(t, "blk.0.ffn_up.weight")
			weight.Dims = []uint64{weight.Dims[1], weight.Dims[0]}
		}, "base blk.0.ffn_up.weight is [10, 8]"},
		{"a base whose value channels disagree with its heads", func(t *testing.T, f *exportFixture) {
			f.keys[8].Value = uint32(17)
		}, "inconsistent geometry"},
		{"a base whose head count is a float", func(t *testing.T, f *exportFixture) {
			f.keys[5].Value = float32(2)
		}, "no usable qwen35.ssm.group_count"},
		{"a base whose heads do not fit its projections", func(t *testing.T, f *exportFixture) {
			// Three key heads of width 3 still divide six value heads and
			// eighteen value channels, but 2*9+18 is not the 30 rows of qkv.
			f.keys[5].Value = uint32(3)
		}, "not 2*9 query-key channels plus 18 value channels"},
		{"a relative output path", func(t *testing.T, f *exportFixture) {
			f.request.Out = "adapter.gguf"
		}, "has to be absolute"},
		{"an output over the checkpoint", func(t *testing.T, f *exportFixture) {
			f.request.Out = f.request.Checkpoint
		}, "would overwrite an input"},
		{"a zero alpha", func(t *testing.T, f *exportFixture) {
			f.request.Alpha = 0
		}, "not a positive finite number"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			fixture := newExportFixture(t)
			c.change(t, fixture)
			fixture.writeRaw(t)
			before, _ := os.ReadFile(fixture.request.Checkpoint)
			_, err := services.ExportQwen35LoRA(fixture.request)
			if !errors.Is(err, services.ErrLoRAExport) || !strings.Contains(err.Error(), c.reason) {
				t.Fatalf("export returned %v; want a refusal naming %q", err, c.reason)
			}
			if _, err := os.Stat(fixture.request.Out); !errors.Is(err, os.ErrNotExist) && fixture.request.Out != fixture.request.Checkpoint {
				t.Fatalf("a refused export left %s behind: %v", fixture.request.Out, err)
			}
			if after, _ := os.ReadFile(fixture.request.Checkpoint); !bytes.Equal(before, after) {
				t.Fatal("a refused export changed the checkpoint")
			}
		})
	}
}

// The written fixture is itself admissible: every refusal above comes from
// its one change, not from a fixture the exporter would refuse anyway.
func TestExportAdmitsTheRawFixture(t *testing.T) {
	fixture := newExportFixture(t)
	fixture.writeRaw(t)
	if _, err := services.ExportQwen35LoRA(fixture.request); err != nil {
		t.Fatal(err)
	}
}
