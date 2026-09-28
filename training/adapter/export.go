package adapter

import (
	"bufio"
	"cmp"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"

	"github.com/tayi-ai/arandu-llama/checkpoint"
)

// The exporter: a checkpoint written by training/local, re-encoded as the
// LoRA the pinned llama.cpp loads over a qwen35 base.
//
// The checkpoint stores PEFT factors in F32: A as [rank, in] and B as
// [out, rank], row-major. llama-adapter.cpp wants lora_a with ne = [in, rank]
// and lora_b with ne = [rank, out], which are the same bytes, so no factor is
// transposed. What does move is the order of the linear-attention value heads.
// The converter that produced the base stores V heads tiled
// ([v0: K0..Kn-1, v1: K0..Kn-1, ...]) while the checkpoint trained against
// the Hugging Face layout, which groups them by key head
// ([K0: v0..vr-1, K1: ...]). The converter permutes the V rows of
// in_proj_qkv, the rows of in_proj_z, in_proj_b and in_proj_a, and the
// columns of out_proj; the adapter has to follow, or it loads without an error
// and adds its update to the wrong heads. A row permutation of W = B*A is a row
// permutation of B, and a column permutation is a column permutation of A.

// qwen35Architecture is the general.architecture the pinned llama.cpp reads
// as LLM_ARCH_QWEN35.
const qwen35Architecture = "qwen35"

// Bounds of an export. They are the exporter's own and do not widen
// WriteInitialLoRA, which keeps its 64 targets and rank 256.
const (
	maxExportRank     = 256
	maxExportTargets  = 4096
	maxExportDim      = 1 << 20
	maxExportElements = 1 << 28
)

// ErrLoRAExport marks every refusal of ExportQwen35LoRA.
var ErrLoRAExport = errors.New("cluster: LoRA export refused")

// Qwen35LoRAExport names the files ExportQwen35LoRA reads and writes and the
// recipe the checkpoint was trained under.
type Qwen35LoRAExport struct {
	// Checkpoint is adapter_model.safetensors as training/local writes it:
	// F32 factors named base_model.model.model.language_model.layers.<N>.
	// <module>.lora_A.default.weight and .lora_B.default.weight.
	Checkpoint string
	// Base is the qwen35 GGUF the adapter will be applied to. Only its header
	// is read: every target is checked against a base tensor of that geometry,
	// and the value-head layout is taken from its metadata.
	Base string
	// Out is the absolute path of the adapter, written atomically.
	Out string
	// Rank is the recipe's LoRA rank. Every pair must have exactly this rank.
	Rank uint64
	// Alpha is the recipe's LoRA alpha, stored as adapter.lora.alpha so that
	// llama.cpp applies scale*alpha/rank, the factor PEFT applied in training.
	Alpha float32
}

// ExportedLoRATarget is one adapted projection of an exported adapter.
type ExportedLoRATarget struct {
	// Module is the checkpoint module, without the .lora_A/.lora_B suffix.
	Module string
	// Tensor is the base GGUF tensor the pair adapts.
	Tensor string
	// Input and Output are in ggml order, as the base tensor declares them.
	Input, Output uint64
	// Reordered reports whether the value heads of this pair were permuted.
	Reordered bool
}

// Qwen35LoRAReceipt describes a written adapter.
type Qwen35LoRAReceipt struct {
	// Digest is the SHA-256 of the file's bytes.
	Digest string
	// Targets are in file order: layer, then the checkpoint's canonical
	// module order, each written as lora_a then lora_b.
	Targets []ExportedLoRATarget
}

// qwen35LoRAModule maps one checkpoint module to the GGUF tensor the
// converter writes it as, and to the permutation the converter applied.
type qwen35LoRAModule struct {
	module  string
	tensor  string
	reorder vHeadReorder
}

// vHeadReorder is where the converter permutes value heads in a projection.
type vHeadReorder int

const (
	reorderNone vHeadReorder = iota
	// reorderQKVRows permutes the V rows that follow the Q and K rows.
	reorderQKVRows
	// reorderValueRows permutes rows of head_v_dim channels per V head.
	reorderValueRows
	// reorderHeadRows permutes rows of one channel per V head.
	reorderHeadRows
	// reorderValueColumns permutes columns of head_v_dim channels per V head.
	reorderValueColumns
)

// qwen35LoRAModules is the checkpoint's canonical target order, with the
// names from gguf-py tensor_mapping and the permutations from
// conversion/qwen.py at the pinned llama.cpp.
var qwen35LoRAModules = [...]qwen35LoRAModule{
	{"self_attn.q_proj", "attn_q", reorderNone},
	{"self_attn.k_proj", "attn_k", reorderNone},
	{"self_attn.v_proj", "attn_v", reorderNone},
	{"self_attn.o_proj", "attn_output", reorderNone},
	{"linear_attn.out_proj", "ssm_out", reorderValueColumns},
	{"linear_attn.in_proj_qkv", "attn_qkv", reorderQKVRows},
	{"linear_attn.in_proj_z", "attn_gate", reorderValueRows},
	{"linear_attn.in_proj_b", "ssm_beta", reorderHeadRows},
	{"linear_attn.in_proj_a", "ssm_alpha", reorderHeadRows},
	{"mlp.gate_proj", "ffn_gate", reorderNone},
	{"mlp.up_proj", "ffn_up", reorderNone},
	{"mlp.down_proj", "ffn_down", reorderNone},
}

const (
	checkpointLayerPrefix = "base_model.model.model.language_model.layers."
	checkpointASuffix     = ".lora_A.default.weight"
	checkpointBSuffix     = ".lora_B.default.weight"
)

// qwen35Geometry is what the base header says about the layers and the
// linear-attention heads.
type qwen35Geometry struct {
	textLayers   int
	keyHeads     uint64
	valueHeads   uint64
	keyHeadDim   uint64
	valueHeadDim uint64
}

// ExportQwen35LoRA writes the checkpoint as a GGUF LoRA for the qwen35 base
// and returns the digest of the bytes written.
//
// It refuses rather than guesses: a tensor outside the checkpoint namespace,
// an unknown module, a layer outside the base's text stack (the MTP block
// included), a missing half of a pair, a factor that is not two-dimensional
// F32, a rank different from the recipe's or above 256, a non-finite value,
// a base that is not qwen35, and a target whose base tensor is missing or has
// a different geometry. The same checkpoint and base always produce the same
// bytes.
//
// The base is taken to be a conversion by llama.cpp, which tiles the value
// heads of every qwen35 model whose key and value head counts differ. The
// header does not record that, so it is not checked here: a base converted
// some other way loads this adapter onto the wrong heads.
func ExportQwen35LoRA(request Qwen35LoRAExport) (Qwen35LoRAReceipt, error) {
	if request.Out == "" || !filepath.IsAbs(request.Out) {
		return Qwen35LoRAReceipt{}, fmt.Errorf("%w: the output path has to be absolute", ErrLoRAExport)
	}
	if request.Out == request.Checkpoint || request.Out == request.Base {
		return Qwen35LoRAReceipt{}, fmt.Errorf("%w: the output path would overwrite an input", ErrLoRAExport)
	}
	if request.Rank == 0 || request.Rank > maxExportRank {
		return Qwen35LoRAReceipt{}, fmt.Errorf("%w: rank %d is outside 1..%d", ErrLoRAExport, request.Rank, maxExportRank)
	}
	alpha := float64(request.Alpha)
	if alpha <= 0 || math.IsNaN(alpha) || math.IsInf(alpha, 0) {
		return Qwen35LoRAReceipt{}, fmt.Errorf("%w: alpha %v is not a positive finite number", ErrLoRAExport, request.Alpha)
	}

	base, err := readQwen35Base(request.Base)
	if err != nil {
		return Qwen35LoRAReceipt{}, err
	}
	pairs, err := readCheckpointPairs(request.Checkpoint, base.geometry.textLayers)
	if err != nil {
		return Qwen35LoRAReceipt{}, err
	}

	receipt := Qwen35LoRAReceipt{Targets: make([]ExportedLoRATarget, 0, len(pairs))}
	tensors := make([]initialLoRATensor, 0, 2*len(pairs))
	elements := uint64(0)
	for _, pair := range pairs {
		module := qwen35LoRAModules[pair.module]
		name := fmt.Sprintf("blk.%d.%s.weight", pair.layer, module.tensor)
		dims, found := base.tensors[name]
		if !found {
			return Qwen35LoRAReceipt{}, fmt.Errorf("%w: %s adapts %s, which the base does not have", ErrLoRAExport, pair.name, name)
		}
		if len(dims) != 2 {
			return Qwen35LoRAReceipt{}, fmt.Errorf("%w: base tensor %s has %d dimensions, not 2", ErrLoRAExport, name, len(dims))
		}
		input, output := dims[0], dims[1]
		rank := pair.aShape[0]
		if rank != request.Rank || pair.bShape[1] != rank {
			return Qwen35LoRAReceipt{}, fmt.Errorf("%w: %s has A %v and B %v; the recipe rank is %d", ErrLoRAExport, pair.name, pair.aShape, pair.bShape, request.Rank)
		}
		if pair.aShape[1] != input || pair.bShape[0] != output {
			return Qwen35LoRAReceipt{}, fmt.Errorf("%w: %s has A %v and B %v; base %s is [%d, %d] in ggml order", ErrLoRAExport, pair.name, pair.aShape, pair.bShape, name, input, output)
		}
		elements += (input + output) * rank
		if elements > maxExportElements {
			return Qwen35LoRAReceipt{}, fmt.Errorf("%w: the adapter exceeds %d factor elements", ErrLoRAExport, maxExportElements)
		}
		reordered, err := reorderValueHeads(module.reorder, base.geometry, pair, input, output)
		if err != nil {
			return Qwen35LoRAReceipt{}, fmt.Errorf("%w: %s: %v", ErrLoRAExport, pair.name, err)
		}
		tensors = append(tensors,
			initialLoRATensor{Name: name + ".lora_a", Dims: []uint64{input, rank}, Values: pair.a},
			initialLoRATensor{Name: name + ".lora_b", Dims: []uint64{rank, output}, Values: pair.b},
		)
		receipt.Targets = append(receipt.Targets, ExportedLoRATarget{Module: pair.name, Tensor: name, Input: input, Output: output, Reordered: reordered})
	}

	body, err := encodeInitialLoRA(qwen35Architecture, request.Alpha, tensors)
	if err != nil {
		return Qwen35LoRAReceipt{}, err
	}
	if err := os.MkdirAll(filepath.Dir(request.Out), 0o755); err != nil {
		return Qwen35LoRAReceipt{}, err
	}
	if err := writeAdapterAtomically(body, request.Out, ".export-lora-"); err != nil {
		return Qwen35LoRAReceipt{}, err
	}
	receipt.Digest = Digest(body)
	return receipt, nil
}

// reorderValueHeads permutes the factors of one pair in place, as the
// converter permuted the base weight, and reports whether it did.
func reorderValueHeads(kind vHeadReorder, geometry qwen35Geometry, pair *checkpointPair, input, output uint64) (bool, error) {
	if kind == reorderNone || geometry.keyHeads == geometry.valueHeads {
		return false, nil
	}
	valueChannels := geometry.valueHeads * geometry.valueHeadDim
	rank := int(pair.aShape[0])
	switch kind {
	case reorderQKVRows:
		keyChannels := geometry.keyHeads * geometry.keyHeadDim
		if output != 2*keyChannels+valueChannels {
			return false, fmt.Errorf("output %d is not 2*%d query-key channels plus %d value channels", output, keyChannels, valueChannels)
		}
		permuteRows(pair.b, rank, int(2*keyChannels), tiledValueHeadSource(geometry, geometry.valueHeadDim))
	case reorderValueRows:
		if output != valueChannels {
			return false, fmt.Errorf("output %d is not the %d value channels", output, valueChannels)
		}
		permuteRows(pair.b, rank, 0, tiledValueHeadSource(geometry, geometry.valueHeadDim))
	case reorderHeadRows:
		if output != geometry.valueHeads {
			return false, fmt.Errorf("output %d is not the %d value heads", output, geometry.valueHeads)
		}
		permuteRows(pair.b, rank, 0, tiledValueHeadSource(geometry, 1))
	case reorderValueColumns:
		if input != valueChannels {
			return false, fmt.Errorf("input %d is not the %d value channels", input, valueChannels)
		}
		permuteColumns(pair.a, int(input), tiledValueHeadSource(geometry, geometry.valueHeadDim))
	default:
		return false, fmt.Errorf("value-head reorder %d is unknown", kind)
	}
	return true, nil
}

// tiledValueHeadSource maps each position of the tiled layout to the position
// it comes from in the grouped layout: tiled index (v, k, d) holds grouped
// index (k, v, d), for v per key head, key head k and channel d of a head.
func tiledValueHeadSource(geometry qwen35Geometry, headDim uint64) []int {
	keyHeads := int(geometry.keyHeads)
	perKey := int(geometry.valueHeads / geometry.keyHeads)
	width := int(headDim)
	source := make([]int, keyHeads*perKey*width)
	for v := 0; v < perKey; v++ {
		for k := 0; k < keyHeads; k++ {
			for d := 0; d < width; d++ {
				source[(v*keyHeads+k)*width+d] = (k*perKey+v)*width + d
			}
		}
	}
	return source
}

// permuteRows reorders the rows of a row-major [rows, width] matrix from
// offset on: row offset+i takes the old row offset+source[i].
func permuteRows(values []float32, width, offset int, source []int) {
	block := make([]float32, len(source)*width)
	copy(block, values[offset*width:(offset+len(source))*width])
	for i, from := range source {
		copy(values[(offset+i)*width:(offset+i+1)*width], block[from*width:(from+1)*width])
	}
}

// permuteColumns reorders every row of a row-major [rows, width] matrix:
// column i takes the old column source[i].
func permuteColumns(values []float32, width int, source []int) {
	row := make([]float32, width)
	for start := 0; start < len(values); start += width {
		copy(row, values[start:start+width])
		for i, from := range source {
			values[start+i] = row[from]
		}
	}
}

// qwen35Base is what the exporter reads from the base header.
type qwen35Base struct {
	tensors  map[string][]uint64
	geometry qwen35Geometry
}

// readQwen35Base reads the base header and derives the geometry the
// converter permuted with.
func readQwen35Base(path string) (qwen35Base, error) {
	metadata, tensors, err := readGGUFHeader(path)
	if err != nil {
		return qwen35Base{}, err
	}
	if architecture, _ := metadata["general.architecture"].(string); architecture != qwen35Architecture {
		return qwen35Base{}, fmt.Errorf("%w: base %s declares architecture %q, not %q", ErrLoRAExport, filepath.Base(path), architecture, qwen35Architecture)
	}
	count := func(key string, required bool) (uint64, error) {
		value, found := metadata[qwen35Architecture+"."+key]
		if !found && !required {
			return 0, nil
		}
		number, ok := value.(uint64)
		if !ok || number > maxExportDim {
			return 0, fmt.Errorf("%w: base %s has no usable %s.%s", ErrLoRAExport, filepath.Base(path), qwen35Architecture, key)
		}
		return number, nil
	}
	blocks, err := count("block_count", true)
	if err != nil {
		return qwen35Base{}, err
	}
	nextn, err := count("nextn_predict_layers", false)
	if err != nil {
		return qwen35Base{}, err
	}
	keyHeads, err := count("ssm.group_count", true)
	if err != nil {
		return qwen35Base{}, err
	}
	valueHeads, err := count("ssm.time_step_rank", true)
	if err != nil {
		return qwen35Base{}, err
	}
	stateSize, err := count("ssm.state_size", true)
	if err != nil {
		return qwen35Base{}, err
	}
	innerSize, err := count("ssm.inner_size", true)
	if err != nil {
		return qwen35Base{}, err
	}
	// The engine reads both head widths from ssm.state_size and the value
	// channels from ssm.inner_size; a base where they disagree is one whose
	// head layout this exporter cannot know.
	if blocks == 0 || nextn >= blocks || keyHeads == 0 || valueHeads == 0 || valueHeads%keyHeads != 0 || stateSize == 0 || innerSize != valueHeads*stateSize {
		return qwen35Base{}, fmt.Errorf("%w: base %s declares an inconsistent geometry: %d blocks, %d MTP, %d key heads, %d value heads, state %d, inner %d", ErrLoRAExport, filepath.Base(path), blocks, nextn, keyHeads, valueHeads, stateSize, innerSize)
	}
	return qwen35Base{tensors: tensors, geometry: qwen35Geometry{
		textLayers: int(blocks - nextn), keyHeads: keyHeads, valueHeads: valueHeads,
		keyHeadDim: stateSize, valueHeadDim: innerSize / valueHeads,
	}}, nil
}

// readGGUFHeader reads the metadata and tensor shapes of a GGUF v3 header.
//
// Unlike ReadGGUF it keeps the scalar and string metadata, because the
// exporter's decisions come from those keys. Integers are returned as uint64
// and strings as string; floats, bools and arrays are kept as nil.
func readGGUFHeader(path string) (map[string]any, map[string][]uint64, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, nil, err
	}
	defer file.Close()
	reader := &ggufReader{r: bufio.NewReaderSize(file, 1<<20)}
	if magic := reader.u32(); magic != ggufMagic {
		return nil, nil, fmt.Errorf("%w: %s does not start with the GGUF magic", ErrLoRAExport, filepath.Base(path))
	}
	if version := reader.u32(); version != 3 {
		return nil, nil, fmt.Errorf("%w: %s is GGUF version %d, not 3", ErrLoRAExport, filepath.Base(path), version)
	}
	tensorCount := reader.u64()
	kvCount := reader.u64()
	if reader.err != nil {
		return nil, nil, fmt.Errorf("%w: reading the header of %s: %v", ErrLoRAExport, filepath.Base(path), reader.err)
	}
	if tensorCount > 1<<20 || kvCount > 1<<20 {
		return nil, nil, fmt.Errorf("%w: %s declares %d tensors and %d keys", ErrLoRAExport, filepath.Base(path), tensorCount, kvCount)
	}
	metadata := make(map[string]any, kvCount)
	for i := uint64(0); i < kvCount; i++ {
		key := reader.str()
		kind := reader.u32()
		value := reader.valueOf(kind)
		if reader.err != nil {
			return nil, nil, fmt.Errorf("%w: reading metadata key %d of %s: %v", ErrLoRAExport, i, filepath.Base(path), reader.err)
		}
		if _, repeated := metadata[key]; repeated {
			return nil, nil, fmt.Errorf("%w: %s repeats metadata key %s", ErrLoRAExport, filepath.Base(path), key)
		}
		switch kind {
		case 6, 7, 12:
			// A float or a bool is not a count; keeping its bits as a number
			// would let a reader take it for one.
			metadata[key] = nil
		default:
			metadata[key] = value
		}
	}
	tensors := make(map[string][]uint64, tensorCount)
	for i := uint64(0); i < tensorCount; i++ {
		name := reader.str()
		count := reader.u32()
		if reader.err != nil || count > 4 {
			return nil, nil, fmt.Errorf("%w: tensor %d of %s declares %d dimensions: %v", ErrLoRAExport, i, filepath.Base(path), count, reader.err)
		}
		dims := make([]uint64, count)
		for d := range dims {
			dims[d] = reader.u64()
		}
		reader.u32()
		reader.u64()
		if reader.err != nil {
			return nil, nil, fmt.Errorf("%w: reading tensor %d of %s: %v", ErrLoRAExport, i, filepath.Base(path), reader.err)
		}
		if _, repeated := tensors[name]; repeated {
			return nil, nil, fmt.Errorf("%w: %s repeats tensor %s", ErrLoRAExport, filepath.Base(path), name)
		}
		tensors[name] = dims
	}
	return metadata, tensors, nil
}

// checkpointPair is one module's factors as the checkpoint stores them.
type checkpointPair struct {
	name   string
	layer  int
	module int
	aShape [2]uint64
	bShape [2]uint64
	a, b   []float32
}

// readCheckpointPairs reads and pairs every factor of the checkpoint, in
// layer then canonical module order.
func readCheckpointPairs(path string, textLayers int) ([]*checkpointPair, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	info, err := file.Stat()
	if err != nil {
		return nil, err
	}
	container, err := checkpoint.OpenSafetensors(file, info.Size(), checkpoint.DefaultLimits())
	if err != nil {
		return nil, fmt.Errorf("%w: %s: %v", ErrLoRAExport, filepath.Base(path), err)
	}
	descriptors := container.Tensors()
	if len(descriptors) == 0 || len(descriptors) > 2*maxExportTargets {
		return nil, fmt.Errorf("%w: %s holds %d tensors", ErrLoRAExport, filepath.Base(path), len(descriptors))
	}

	byName := make(map[string]*checkpointPair, len(descriptors)/2)
	for _, descriptor := range descriptors {
		module, factorB, err := parseCheckpointName(descriptor.Name)
		if err != nil {
			return nil, err
		}
		layer, index, err := parseCheckpointModule(module, textLayers)
		if err != nil {
			return nil, err
		}
		if descriptor.DType != "F32" || len(descriptor.Shape) != 2 {
			return nil, fmt.Errorf("%w: %s is %s %v, not a two-dimensional F32 factor", ErrLoRAExport, descriptor.Name, descriptor.DType, descriptor.Shape)
		}
		rows, columns := descriptor.Shape[0], descriptor.Shape[1]
		if rows == 0 || columns == 0 || rows > maxExportDim || columns > maxExportDim || rows*columns > maxExportElements {
			return nil, fmt.Errorf("%w: %s has shape %v", ErrLoRAExport, descriptor.Name, descriptor.Shape)
		}
		values, err := readFloat32Tensor(container, descriptor, rows*columns)
		if err != nil {
			return nil, err
		}
		pair := byName[module]
		if pair == nil {
			pair = &checkpointPair{name: module, layer: layer, module: index}
			byName[module] = pair
		}
		if factorB {
			pair.bShape, pair.b = [2]uint64{rows, columns}, values
		} else {
			pair.aShape, pair.a = [2]uint64{rows, columns}, values
		}
	}

	ordered := make([]*checkpointPair, 0, len(byName))
	for _, pair := range byName {
		if pair.a == nil || pair.b == nil {
			return nil, fmt.Errorf("%w: %s is missing one of lora_A and lora_B", ErrLoRAExport, pair.name)
		}
		ordered = append(ordered, pair)
	}
	slices.SortFunc(ordered, func(left, right *checkpointPair) int {
		if left.layer != right.layer {
			return cmp.Compare(left.layer, right.layer)
		}
		return cmp.Compare(left.module, right.module)
	})
	if len(ordered) > maxExportTargets {
		return nil, fmt.Errorf("%w: %s adapts %d targets; the limit is %d", ErrLoRAExport, filepath.Base(path), len(ordered), maxExportTargets)
	}
	return ordered, nil
}

// parseCheckpointName splits a factor name into its module and letter.
func parseCheckpointName(name string) (module string, factorB bool, err error) {
	switch {
	case strings.HasSuffix(name, checkpointASuffix):
		return strings.TrimSuffix(name, checkpointASuffix), false, nil
	case strings.HasSuffix(name, checkpointBSuffix):
		return strings.TrimSuffix(name, checkpointBSuffix), true, nil
	default:
		return "", false, fmt.Errorf("%w: %s is not a lora_A or lora_B factor of the checkpoint namespace", ErrLoRAExport, name)
	}
}

// parseCheckpointModule resolves a module name to its layer and canonical
// module index, refusing anything outside the text stack.
func parseCheckpointModule(module string, textLayers int) (int, int, error) {
	rest, found := strings.CutPrefix(module, checkpointLayerPrefix)
	digits, relative, split := strings.Cut(rest, ".")
	if !found || !split {
		return 0, 0, fmt.Errorf("%w: %s is outside %s<N>", ErrLoRAExport, module, checkpointLayerPrefix)
	}
	layer, err := strconv.Atoi(digits)
	if err != nil || layer < 0 || strconv.Itoa(layer) != digits {
		return 0, 0, fmt.Errorf("%w: %s has no canonical layer index", ErrLoRAExport, module)
	}
	if layer >= textLayers {
		return 0, 0, fmt.Errorf("%w: %s is layer %d; the base text stack has %d layers", ErrLoRAExport, module, layer, textLayers)
	}
	for index, known := range qwen35LoRAModules {
		if known.module == relative {
			return layer, index, nil
		}
	}
	return 0, 0, fmt.Errorf("%w: %s adapts module %q, which has no qwen35 GGUF tensor", ErrLoRAExport, module, relative)
}

// readFloat32Tensor decodes a little-endian F32 tensor and refuses a
// non-finite value: NaN or infinity in an adapter poisons every token.
func readFloat32Tensor(container *checkpoint.Safetensors, descriptor checkpoint.Tensor, elements uint64) ([]float32, error) {
	if descriptor.Size() != int64(elements*4) {
		return nil, fmt.Errorf("%w: %s stores %d bytes for %d F32 values", ErrLoRAExport, descriptor.Name, descriptor.Size(), elements)
	}
	reader, err := container.TensorReader(descriptor.Name)
	if err != nil {
		return nil, err
	}
	raw := make([]byte, descriptor.Size())
	if _, err := io.ReadFull(reader, raw); err != nil {
		return nil, fmt.Errorf("%w: reading %s: %v", ErrLoRAExport, descriptor.Name, err)
	}
	values := make([]float32, elements)
	for i := range values {
		value := math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return nil, fmt.Errorf("%w: %s holds a non-finite value at %d", ErrLoRAExport, descriptor.Name, i)
		}
		values[i] = value
	}
	return values, nil
}
