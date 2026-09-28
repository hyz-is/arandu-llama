package fixtures

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"testing"
)

// GGUFKey is one metadata key of a synthetic base GGUF. Value is a string,
// a uint32 or a float32, written with GGUF types 8, 4 and 6.
type GGUFKey struct {
	Key   string
	Value any
}

// BaseTensor is one weight of a synthetic base GGUF. Dims are in ggml order.
type BaseTensor struct {
	Name string
	Dims []uint64
}

// WriteBaseGGUF writes a GGUF version 3 model with the given metadata and
// zero-filled F32 tensors. It stands for a converted base in tests that need
// only its header: names, shapes and the keys a reader derives geometry from.
func WriteBaseGGUF(t *testing.T, path string, keys []GGUFKey, tensors []BaseTensor) {
	t.Helper()
	const alignment = 32
	var head bytes.Buffer
	u32 := func(v uint32) { _ = binary.Write(&head, binary.LittleEndian, v) }
	u64 := func(v uint64) { _ = binary.Write(&head, binary.LittleEndian, v) }
	str := func(s string) { u64(uint64(len(s))); head.WriteString(s) }

	u32(0x46554747) // "GGUF"
	u32(3)
	u64(uint64(len(tensors)))
	u64(uint64(len(keys)))
	for _, key := range keys {
		str(key.Key)
		switch value := key.Value.(type) {
		case string:
			u32(8)
			str(value)
		case uint32:
			u32(4)
			u32(value)
		case float32:
			u32(6)
			u32(math.Float32bits(value))
		default:
			t.Fatalf("fixture key %s has unsupported value %T", key.Key, key.Value)
		}
	}
	offset := uint64(0)
	for _, tensor := range tensors {
		str(tensor.Name)
		u32(uint32(len(tensor.Dims)))
		elements := uint64(1)
		for _, d := range tensor.Dims {
			u64(d)
			elements *= d
		}
		u32(0)
		u64(offset)
		offset += (elements*4 + alignment - 1) / alignment * alignment
	}
	for head.Len()%alignment != 0 {
		head.WriteByte(0)
	}
	head.Write(make([]byte, offset))
	if err := os.WriteFile(path, head.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
}

// RawTensor is one tensor of a hand-built safetensors file: any dtype string
// and any payload, so a test can hand a reader what the checkpoint writer
// refuses to produce.
type RawTensor struct {
	Name  string
	DType string
	Shape []uint64
	Data  []byte
}

// WriteRawSafetensors writes the tensors in the given order, back to back.
func WriteRawSafetensors(t *testing.T, path string, tensors []RawTensor) {
	t.Helper()
	header := make(map[string]any, len(tensors))
	var payload bytes.Buffer
	for _, tensor := range tensors {
		start := payload.Len()
		payload.Write(tensor.Data)
		header[tensor.Name] = map[string]any{"dtype": tensor.DType, "shape": tensor.Shape, "data_offsets": []int{start, payload.Len()}}
	}
	body, err := json.Marshal(header)
	if err != nil {
		t.Fatal(err)
	}
	for len(body)%8 != 0 {
		body = append(body, ' ')
	}
	var file bytes.Buffer
	_ = binary.Write(&file, binary.LittleEndian, uint64(len(body)))
	file.Write(body)
	file.Write(payload.Bytes())
	if err := os.WriteFile(path, file.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
}

// Float32Bytes encodes values little endian, bit for bit.
func Float32Bytes(values []float32) []byte {
	out := make([]byte, 4*len(values))
	for i, value := range values {
		binary.LittleEndian.PutUint32(out[4*i:], math.Float32bits(value))
	}
	return out
}
