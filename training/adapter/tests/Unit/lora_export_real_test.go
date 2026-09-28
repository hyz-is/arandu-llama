package unit_test

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"testing"

	"github.com/tayi-ai/arandu-llama/checkpoint"
	services "github.com/tayi-ai/arandu-llama/training/adapter"
)

// These tests read real files and are skipped unless their paths are given:
//
//	TAYI_LORA_EXPORT_BASE_GGUF   the qwen35 GGUF the adapter is applied to
//	TAYI_LORA_EXPORT_HF_DIR      the Hugging Face safetensors shards it came from
//	TAYI_LORA_EXPORT_CHECKPOINT  an adapter_model.safetensors from training/local
//	TAYI_LORA_EXPORT_RANK        the recipe's LoRA rank
//	TAYI_LORA_EXPORT_ALPHA       the recipe's LoRA alpha
//
// They read headers and small tensors only; no model is loaded.

// baseFile is a GGUF read far enough to fetch tensors and integer keys.
type baseFile struct {
	path      string
	header    []byte
	container services.GGUF
	dims      map[string]services.GGUFTensor
}

func openBase(t *testing.T, path string) baseFile {
	t.Helper()
	container, err := services.ReadGGUF(path)
	if err != nil {
		t.Fatal(err)
	}
	file, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	header := make([]byte, container.DataOffset)
	if _, err := io.ReadFull(file, header); err != nil {
		t.Fatal(err)
	}
	dims := make(map[string]services.GGUFTensor, len(container.Tensors))
	for _, tensor := range container.Tensors {
		dims[tensor.Name] = tensor
	}
	return baseFile{path: path, header: header, container: container, dims: dims}
}

// uint32Key finds a UINT32 metadata value by its key's own bytes.
func (b baseFile) uint32Key(t *testing.T, key string) int {
	t.Helper()
	var prefix []byte
	prefix = binary.LittleEndian.AppendUint64(prefix, uint64(len(key)))
	prefix = append(prefix, key...)
	at := bytes.Index(b.header, prefix)
	if at < 0 || bytes.Index(b.header[at+len(prefix):], prefix) >= 0 {
		t.Fatalf("%s does not hold %s exactly once", filepath.Base(b.path), key)
	}
	at += len(prefix)
	if binary.LittleEndian.Uint32(b.header[at:]) != 4 {
		t.Fatalf("%s is not UINT32", key)
	}
	return int(binary.LittleEndian.Uint32(b.header[at+4:]))
}

// f32 reads one F32 tensor without reading the rest of the file.
func (b baseFile) f32(t *testing.T, name string) []float32 {
	t.Helper()
	tensor, found := b.dims[name]
	if !found || tensor.Type != 0 {
		t.Fatalf("%s holds no F32 tensor %s", filepath.Base(b.path), name)
	}
	file, err := os.Open(b.path)
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	raw := make([]byte, tensor.Elements*4)
	if _, err := file.ReadAt(raw, b.container.DataOffset+tensor.Offset); err != nil {
		t.Fatal(err)
	}
	values := make([]float32, tensor.Elements)
	for i := range values {
		values[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[4*i:]))
	}
	return values
}

func (b baseFile) layout(t *testing.T) vHeadLayout {
	valueHeads := b.uint32Key(t, "qwen35.ssm.time_step_rank")
	return vHeadLayout{
		keyHeads: b.uint32Key(t, "qwen35.ssm.group_count"), valueHeads: valueHeads,
		keyHeadDim: b.uint32Key(t, "qwen35.ssm.state_size"), valueHeadDim: b.uint32Key(t, "qwen35.ssm.inner_size") / valueHeads,
	}
}

// shards indexes every tensor of a directory of safetensors shards.
type shards struct {
	files  []*os.File
	byName map[string]*checkpoint.Safetensors
}

func openShards(t *testing.T, dir string) shards {
	t.Helper()
	paths, err := filepath.Glob(filepath.Join(dir, "*.safetensors"))
	if err != nil || len(paths) == 0 {
		t.Fatalf("no safetensors under %s: %v", dir, err)
	}
	result := shards{byName: map[string]*checkpoint.Safetensors{}}
	for _, path := range paths {
		file, err := os.Open(path)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { file.Close() })
		info, err := file.Stat()
		if err != nil {
			t.Fatal(err)
		}
		index, err := checkpoint.OpenSafetensors(file, info.Size(), checkpoint.DefaultLimits())
		if err != nil {
			t.Fatal(err)
		}
		for _, tensor := range index.Tensors() {
			result.byName[tensor.Name] = index
		}
	}
	return result
}

// f32 decodes a BF16 or F32 tensor exactly into float32.
func (s shards) f32(t *testing.T, name string) []float32 {
	t.Helper()
	index, found := s.byName[name]
	if !found {
		t.Fatalf("no shard holds %s", name)
	}
	tensor, _ := index.Tensor(name)
	reader, err := index.TensorReader(name)
	if err != nil {
		t.Fatal(err)
	}
	raw, err := io.ReadAll(reader)
	if err != nil {
		t.Fatal(err)
	}
	switch tensor.DType {
	case "BF16":
		values := make([]float32, len(raw)/2)
		for i := range values {
			values[i] = math.Float32frombits(uint32(binary.LittleEndian.Uint16(raw[2*i:])) << 16)
		}
		return values
	case "F32":
		values := make([]float32, len(raw)/4)
		for i := range values {
			values[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[4*i:]))
		}
		return values
	}
	t.Fatalf("%s is %s", name, tensor.DType)
	return nil
}

// exactMatches counts the positions where got equals want bit for bit.
func exactMatches(got []float32, want []float64) int {
	count := 0
	for i := range got {
		if math.Float32bits(got[i]) == math.Float32bits(float32(want[i])) {
			count++
		}
	}
	return count
}

func maxRelative(got []float32, want []float64) float64 {
	worst := 0.0
	for i := range got {
		worst = math.Max(worst, math.Abs(float64(got[i])-want[i])/math.Abs(want[i]))
	}
	return worst
}

// TestOfficialBaseStoresValueHeadsTiled measures the order in which the base
// GGUF stores the linear-attention value heads, by comparing its F32 tensors
// with the Hugging Face tensors they were converted from, under three
// hypotheses: grouped (the Hugging Face order, no reorder), tiled (what
// conversion/qwen.py writes and the exporter permutes to) and the inverse of
// the tiled permutation. The exporter is right only if tiled matches exactly.
func TestOfficialBaseStoresValueHeadsTiled(t *testing.T) {
	basePath, hfDir := os.Getenv("TAYI_LORA_EXPORT_BASE_GGUF"), os.Getenv("TAYI_LORA_EXPORT_HF_DIR")
	if basePath == "" || hfDir == "" {
		t.Skip("set TAYI_LORA_EXPORT_BASE_GGUF and TAYI_LORA_EXPORT_HF_DIR to measure the official base")
	}
	base := openBase(t, basePath)
	hf := openShards(t, hfDir)
	layout := base.layout(t)
	perKey := layout.valueHeads / layout.keyHeads
	t.Logf("base %s: key heads %d, value heads %d, key head dim %d, value head dim %d", filepath.Base(basePath), layout.keyHeads, layout.valueHeads, layout.keyHeadDim, layout.valueHeadDim)
	if layout.keyHeads == layout.valueHeads {
		t.Fatal("the base has as many key heads as value heads; there is no order to measure")
	}

	hypotheses := []struct {
		name  string
		apply func(values []float64, rows, cols, headDim int) []float64
	}{
		{"grouped", func(values []float64, _, _, _ int) []float64 { return values }},
		{"tiled", func(values []float64, rows, cols, headDim int) []float64 {
			return reorderLikeConverter(values, rows, cols, 0, layout.keyHeads, perKey, headDim)
		}},
		{"inverse", func(values []float64, rows, cols, headDim int) []float64 {
			return reorderLikeConverter(values, rows, cols, 0, perKey, layout.keyHeads, headDim)
		}},
	}

	prefix := "model.language_model.layers."
	layers := 0
	for layer := 0; ; layer++ {
		if _, found := base.dims[fmt.Sprintf("blk.%d.attn_norm.weight", layer)]; !found {
			break
		}
		if _, found := base.dims[fmt.Sprintf("blk.%d.ssm_a", layer)]; !found {
			continue
		}
		layers++
		ggufA := base.f32(t, fmt.Sprintf("blk.%d.ssm_a", layer))
		ggufDT := base.f32(t, fmt.Sprintf("blk.%d.ssm_dt.bias", layer))
		ggufConv := base.f32(t, fmt.Sprintf("blk.%d.ssm_conv1d.weight", layer))
		ggufNorm := base.f32(t, fmt.Sprintf("blk.%d.ssm_norm.weight", layer))
		logA := hf.f32(t, fmt.Sprintf("%s%d.linear_attn.A_log", prefix, layer))
		dt := hf.f32(t, fmt.Sprintf("%s%d.linear_attn.dt_bias", prefix, layer))
		conv := hf.f32(t, fmt.Sprintf("%s%d.linear_attn.conv1d.weight", prefix, layer))
		norm := hf.f32(t, fmt.Sprintf("%s%d.linear_attn.norm.weight", prefix, layer))

		negExpA := make([]float64, len(logA))
		for i, value := range logA {
			negExpA[i] = -math.Exp(float64(value))
		}
		kernel := len(conv) / (2*layout.keyHeads*layout.keyHeadDim + layout.valueHeads*layout.valueHeadDim)
		qk := 2 * layout.keyHeads * layout.keyHeadDim * kernel
		if len(ggufA) != layout.valueHeads || len(ggufDT) != layout.valueHeads || len(logA) != layout.valueHeads || len(dt) != layout.valueHeads || len(ggufConv) != len(conv) {
			t.Fatalf("layer %d: shapes differ: ssm_a %d, dt %d, A_log %d, dt_bias %d, conv %d/%d", layer, len(ggufA), len(ggufDT), len(logA), len(dt), len(ggufConv), len(conv))
		}
		if matches := exactMatches(ggufNorm, widen(norm)); matches != len(norm) {
			t.Errorf("layer %d: ssm_norm matches linear_attn.norm at %d of %d positions", layer, matches, len(norm))
		}

		line := fmt.Sprintf("layer %2d", layer)
		for _, hypothesis := range hypotheses {
			wantA := hypothesis.apply(negExpA, layout.valueHeads, 1, 1)
			wantDT := hypothesis.apply(widen(dt), layout.valueHeads, 1, 1)
			wantConv := append(widen(conv[:qk]), hypothesis.apply(widen(conv[qk:]), layout.valueHeads*layout.valueHeadDim, kernel, layout.valueHeadDim)...)
			dtMatches, convMatches, relative := exactMatches(ggufDT, wantDT), exactMatches(ggufConv, wantConv), maxRelative(ggufA, wantA)
			line += fmt.Sprintf(" | %s: dt %2d/%d conv %5d/%d ssm_a max-rel %.3g", hypothesis.name, dtMatches, len(dt), convMatches, len(conv), relative)
			if hypothesis.name == "tiled" {
				if dtMatches != len(dt) || convMatches != len(conv) || relative > 1.0/256 {
					t.Errorf("layer %d: the tiled order does not reproduce the base", layer)
				}
			} else if dtMatches == len(dt) && convMatches == len(conv) {
				t.Errorf("layer %d: the %s order also reproduces the base, so the measurement cannot tell the orders apart", layer, hypothesis.name)
			}
		}
		t.Log(line)
		if layer == 0 {
			t.Logf("layer 0 ssm_dt.bias (GGUF) first 8: %v", ggufDT[:8])
			t.Logf("layer 0 dt_bias (HF, grouped) first 8: %v", dt[:8])
			t.Logf("layer 0 dt_bias (HF, tiled by the converter rule) first 8: %v", hypotheses[1].apply(widen(dt), layout.valueHeads, 1, 1)[:8])
			t.Logf("layer 0 ssm_a (GGUF) first 8: %v", ggufA[:8])
			t.Logf("layer 0 -exp(A_log) (HF, tiled) first 8: %v", hypotheses[1].apply(negExpA, layout.valueHeads, 1, 1)[:8])
		}
	}
	if layers == 0 {
		t.Fatal("the base has no linear-attention layer")
	}
	t.Logf("%d linear-attention layers compared", layers)
}

// TestExportedCheckpointMatchesTheOfficialBase exports a real checkpoint and
// checks every tensor against the base header and the checkpoint's factors.
func TestExportedCheckpointMatchesTheOfficialBase(t *testing.T) {
	basePath, checkpointPath := os.Getenv("TAYI_LORA_EXPORT_BASE_GGUF"), os.Getenv("TAYI_LORA_EXPORT_CHECKPOINT")
	rankText, alphaText := os.Getenv("TAYI_LORA_EXPORT_RANK"), os.Getenv("TAYI_LORA_EXPORT_ALPHA")
	if basePath == "" || checkpointPath == "" || rankText == "" || alphaText == "" {
		t.Skip("set TAYI_LORA_EXPORT_BASE_GGUF, TAYI_LORA_EXPORT_CHECKPOINT, TAYI_LORA_EXPORT_RANK and TAYI_LORA_EXPORT_ALPHA to export a real checkpoint")
	}
	rank, err := strconv.ParseUint(rankText, 10, 64)
	if err != nil {
		t.Fatal(err)
	}
	alpha, err := strconv.ParseFloat(alphaText, 32)
	if err != nil {
		t.Fatal(err)
	}
	checkpointBody, err := os.ReadFile(checkpointPath)
	if err != nil {
		t.Fatal(err)
	}
	checkpointSum := sha256.Sum256(checkpointBody)
	t.Logf("checkpoint %s sha256 %s", checkpointPath, hex.EncodeToString(checkpointSum[:]))

	out := filepath.Join(t.TempDir(), "adapter.gguf")
	receipt, err := services.ExportQwen35LoRA(services.Qwen35LoRAExport{Checkpoint: checkpointPath, Base: basePath, Out: out, Rank: rank, Alpha: float32(alpha)})
	if err != nil {
		t.Fatal(err)
	}
	again, err := services.ExportQwen35LoRA(services.Qwen35LoRAExport{Checkpoint: checkpointPath, Base: basePath, Out: out + ".again", Rank: rank, Alpha: float32(alpha)})
	if err != nil {
		t.Fatal(err)
	}
	body, err := os.ReadFile(out)
	if err != nil {
		t.Fatal(err)
	}
	againBody, err := os.ReadFile(out + ".again")
	if err != nil {
		t.Fatal(err)
	}
	sum := sha256.Sum256(body)
	if receipt.Digest != hex.EncodeToString(sum[:]) || again.Digest != receipt.Digest || !bytes.Equal(body, againBody) {
		t.Fatalf("export is not reproducible: %s, %s, file %x", receipt.Digest, again.Digest, sum)
	}
	t.Logf("adapter %d bytes sha256 %s, %d targets", len(body), receipt.Digest, len(receipt.Targets))

	if architecture, _ := ggufString(body, "general.architecture"); architecture != "qwen35" {
		t.Fatalf("general.architecture = %q", architecture)
	}
	if kind, _ := ggufString(body, "general.type"); kind != "adapter" {
		t.Fatalf("general.type = %q", kind)
	}
	if kind, _ := ggufString(body, "adapter.type"); kind != "lora" {
		t.Fatalf("adapter.type = %q", kind)
	}
	if stored, ok := services.AlphaOf(body); !ok || stored != float32(alpha) {
		t.Fatalf("adapter.lora.alpha = %v, present %t", stored, ok)
	}

	base := openBase(t, basePath)
	layout := base.layout(t)
	adapter, err := services.ReadGGUF(out)
	if err != nil {
		t.Fatal(err)
	}
	checkpointFile, err := os.Open(checkpointPath)
	if err != nil {
		t.Fatal(err)
	}
	defer checkpointFile.Close()
	factors, err := checkpoint.OpenSafetensors(checkpointFile, int64(len(checkpointBody)), checkpoint.DefaultLimits())
	if err != nil {
		t.Fatal(err)
	}
	if want := len(factors.Tensors()); len(adapter.Tensors) != want || len(receipt.Targets)*2 != want {
		t.Fatalf("adapter holds %d tensors and %d targets; the checkpoint holds %d factors", len(adapter.Tensors), len(receipt.Targets), want)
	}

	read := func(values []byte, offset int64, count uint64) []float32 {
		out := make([]float32, count)
		for i := range out {
			out[i] = math.Float32frombits(binary.LittleEndian.Uint32(values[offset+int64(4*i):]))
		}
		return out
	}
	kinds := map[string][]string{}
	reordered := 0
	for i, target := range receipt.Targets {
		a, b := adapter.Tensors[2*i], adapter.Tensors[2*i+1]
		if a.Name != target.Tensor+".lora_a" || b.Name != target.Tensor+".lora_b" || a.Type != 0 || b.Type != 0 {
			t.Fatalf("target %d is %s/%s types %d/%d, receipt says %s", i, a.Name, b.Name, a.Type, b.Type, target.Tensor)
		}
		weight, found := base.dims[target.Tensor]
		if !found || len(weight.Dims) != 2 {
			t.Fatalf("%s is not a 2-D base tensor", target.Tensor)
		}
		if !slices.Equal(a.Dims, []uint64{weight.Dims[0], rank}) || !slices.Equal(b.Dims, []uint64{rank, weight.Dims[1]}) {
			t.Fatalf("%s: lora_a %v lora_b %v against base %v", target.Tensor, a.Dims, b.Dims, weight.Dims)
		}
		module := target.Module[strings.Index(target.Module, ".layers.")+len(".layers."):]
		layer, relative, _ := strings.Cut(module, ".")
		if want := "blk." + layer + "." + ggufTensorOf[relative] + ".weight"; want != target.Tensor {
			t.Fatalf("%s was written as %s, the converter names it %s", target.Module, target.Tensor, want)
		}

		input, output := int(weight.Dims[0]), int(weight.Dims[1])
		wantA := widen(readCheckpointFactor(t, factors, target.Module+".lora_A.default.weight"))
		wantB := widen(readCheckpointFactor(t, factors, target.Module+".lora_B.default.weight"))
		if relative == "linear_attn.out_proj" {
			wantA = convertLikeConverter(relative, wantA, int(rank), input, layout)
		} else {
			wantB = convertLikeConverter(relative, wantB, output, int(rank), layout)
		}
		gotA := read(body, adapter.DataOffset+a.Offset, uint64(a.Elements))
		gotB := read(body, adapter.DataOffset+b.Offset, uint64(b.Elements))
		if exactMatches(gotA, wantA) != len(wantA) || exactMatches(gotB, wantB) != len(wantB) {
			t.Fatalf("%s: factors differ from the checkpoint under the converter's layout", target.Tensor)
		}
		if target.Reordered {
			reordered++
		}
		kind := target.Tensor[strings.Index(target.Tensor[len("blk."):], ".")+len("blk.")+1:]
		kinds[kind] = append(kinds[kind], layer)
		if len(kinds[kind]) == 1 {
			t.Logf("%-20s lora_a %v lora_b %v base %v reordered=%t", kind, a.Dims, b.Dims, weight.Dims, target.Reordered)
		}
	}
	names := make([]string, 0, len(kinds))
	for kind := range kinds {
		names = append(names, kind)
	}
	slices.Sort(names)
	for _, kind := range names {
		t.Logf("%-20s %2d layers: %s", kind, len(kinds[kind]), strings.Join(kinds[kind], ","))
	}
	t.Logf("%d targets, %d reordered, alpha %v, rank %d", len(receipt.Targets), reordered, alpha, rank)
}

func readCheckpointFactor(t *testing.T, index *checkpoint.Safetensors, name string) []float32 {
	t.Helper()
	reader, err := index.TensorReader(name)
	if err != nil {
		t.Fatal(err)
	}
	raw, err := io.ReadAll(reader)
	if err != nil {
		t.Fatal(err)
	}
	values := make([]float32, len(raw)/4)
	for i := range values {
		values[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[4*i:]))
	}
	return values
}
