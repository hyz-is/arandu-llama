package unit_test

import (
	"bytes"
	"encoding/binary"
	"slices"
)

// The rules below are written from llama.cpp at the pinned commit, without
// calling the exporter, so that a test compares two readings of the same
// source rather than the exporter with itself.

// ggufTensorOf is gguf-py/gguf/tensor_mapping.py for the qwen3.5 modules the
// checkpoint can adapt: ATTN_QKV (in_proj_qkv), ATTN_GATE (in_proj_z),
// SSM_OUT (out_proj), SSM_ALPHA (in_proj_a), SSM_BETA (in_proj_b), and the
// attention and MLP projections every Qwen maps the same way.
var ggufTensorOf = map[string]string{
	"self_attn.q_proj":        "attn_q",
	"self_attn.k_proj":        "attn_k",
	"self_attn.v_proj":        "attn_v",
	"self_attn.o_proj":        "attn_output",
	"linear_attn.in_proj_qkv": "attn_qkv",
	"linear_attn.in_proj_z":   "attn_gate",
	"linear_attn.in_proj_b":   "ssm_beta",
	"linear_attn.in_proj_a":   "ssm_alpha",
	"linear_attn.out_proj":    "ssm_out",
	"mlp.gate_proj":           "ffn_gate",
	"mlp.up_proj":             "ffn_up",
	"mlp.down_proj":           "ffn_down",
}

// transposeContiguous is torch's tensor.permute(order).contiguous() on a
// row-major tensor of the given shape, returned flat.
func transposeContiguous(data []float64, shape, order []int) []float64 {
	strides := make([]int, len(shape))
	stride := 1
	for axis := len(shape) - 1; axis >= 0; axis-- {
		strides[axis] = stride
		stride *= shape[axis]
	}
	permuted := make([]int, len(order))
	for i, axis := range order {
		permuted[i] = shape[axis]
	}
	out := make([]float64, len(data))
	index := make([]int, len(order))
	for flat := range out {
		rest := flat
		for i := len(permuted) - 1; i >= 0; i-- {
			index[i] = rest % permuted[i]
			rest /= permuted[i]
		}
		source := 0
		for i, axis := range order {
			source += index[i] * strides[axis]
		}
		out[flat] = data[source]
	}
	return out
}

// reorderLikeConverter is _reorder_v_heads of conversion/qwen.py on a
// row-major [rows, cols] matrix: axis dim is viewed as [keyHeads, perKey,
// headDim], its first two sub-axes are swapped and it is flattened back.
func reorderLikeConverter(data []float64, rows, cols, dim, keyHeads, perKey, headDim int) []float64 {
	if dim == 0 {
		return transposeContiguous(data, []int{keyHeads, perKey, headDim, cols}, []int{1, 0, 2, 3})
	}
	return transposeContiguous(data, []int{rows, keyHeads, perKey, headDim}, []int{0, 2, 1, 3})
}

// vHeadLayout is the linear-attention head geometry the converter reads
// from the Hugging Face config.
type vHeadLayout struct {
	keyHeads, valueHeads, keyHeadDim, valueHeadDim int
}

// convertLikeConverter applies to a [rows, cols] projection of module what
// _LinearAttentionVReorderBase.modify_tensors applies to the base weight.
func convertLikeConverter(module string, w []float64, rows, cols int, layout vHeadLayout) []float64 {
	if layout.keyHeads == layout.valueHeads {
		return w
	}
	perKey := layout.valueHeads / layout.keyHeads
	switch module {
	case "linear_attn.in_proj_qkv":
		qk := 2 * layout.keyHeads * layout.keyHeadDim
		v := reorderLikeConverter(w[qk*cols:], rows-qk, cols, 0, layout.keyHeads, perKey, layout.valueHeadDim)
		return append(slices.Clone(w[:qk*cols]), v...)
	case "linear_attn.in_proj_z":
		return reorderLikeConverter(w, rows, cols, 0, layout.keyHeads, perKey, layout.valueHeadDim)
	case "linear_attn.in_proj_b", "linear_attn.in_proj_a":
		return reorderLikeConverter(w, rows, cols, 0, layout.keyHeads, perKey, 1)
	case "linear_attn.out_proj":
		return reorderLikeConverter(w, rows, cols, 1, layout.keyHeads, perKey, layout.valueHeadDim)
	}
	return w
}

func widen(values []float32) []float64 {
	out := make([]float64, len(values))
	for i, value := range values {
		out[i] = float64(value)
	}
	return out
}

// product is B*A for B row-major [rows, rank] and A row-major [rank, cols],
// summed over the rank in index order.
func product(b []float32, rows, rank int, a []float32, cols int) []float64 {
	out := make([]float64, rows*cols)
	for i := 0; i < rows; i++ {
		for c := 0; c < cols; c++ {
			sum := 0.0
			for j := 0; j < rank; j++ {
				sum += float64(b[i*rank+j]) * float64(a[j*cols+c])
			}
			out[i*cols+c] = sum
		}
	}
	return out
}

// ggufString finds a string metadata value by its key's own bytes.
func ggufString(body []byte, key string) (string, bool) {
	var prefix []byte
	prefix = binary.LittleEndian.AppendUint64(prefix, uint64(len(key)))
	prefix = append(prefix, key...)
	at := bytes.Index(body, prefix)
	if at < 0 || bytes.Index(body[at+len(prefix):], prefix) >= 0 {
		return "", false
	}
	at += len(prefix)
	if at+12 > len(body) || binary.LittleEndian.Uint32(body[at:]) != 8 {
		return "", false
	}
	length := int(binary.LittleEndian.Uint64(body[at+4:]))
	start := at + 12
	if start+length > len(body) {
		return "", false
	}
	return string(body[start : start+length]), true
}
