package bridge

import (
	"bufio"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"os"
)

// GGUF metadata value types, as the container numbers them.
const (
	ggufUint8   = 0
	ggufInt8    = 1
	ggufUint16  = 2
	ggufInt16   = 3
	ggufUint32  = 4
	ggufInt32   = 5
	ggufFloat32 = 6
	ggufBool    = 7
	ggufString  = 8
	ggufArray   = 9
	ggufUint64  = 10
	ggufInt64   = 11
	ggufFloat64 = 12
)

// Bounds on what the metadata reader allocates for one value.
const (
	maxGGUFKeys        = 1 << 16
	maxGGUFKeyBytes    = 1 << 16
	maxGGUFStringBytes = 1 << 26
	maxGGUFArray       = 1 << 24
)

// llama.cpp token types, as tokenizer.ggml.token_type stores them.
const (
	ggufTokenNormal      = 1
	ggufTokenUnknown     = 2
	ggufTokenControl     = 3
	ggufTokenUserDefined = 4
	ggufTokenUnused      = 5
	ggufTokenByte        = 6
)

// LoadGGUF reads the tokenizer metadata of the GGUF file at path. Only the
// header and the metadata are read; tensor data is never touched, so a file
// cut after its metadata is enough.
func LoadGGUF(path string) (*Vocabulary, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	return ReadGGUF(file)
}

// ReadGGUF reads tokenizer.ggml.model, tokenizer.ggml.tokens and
// tokenizer.ggml.token_type from a GGUF stream and builds its Vocabulary.
// Model "gpt2" is byte-level; "llama" and "gemma4" are SentencePiece-style,
// which is how llama.cpp decodes them. Token types follow llama.cpp: a
// missing type array makes every token normal, and an undefined type has no
// bytes.
func ReadGGUF(source io.Reader) (*Vocabulary, error) {
	r := &ggufReader{r: bufio.NewReaderSize(source, 1<<20)}
	var magic [4]byte
	r.read(&magic)
	if r.err == nil && string(magic[:]) != "GGUF" {
		return nil, fmt.Errorf("%w: not a GGUF stream", ErrVocabulary)
	}
	version := r.u32()
	if r.err == nil && version != 2 && version != 3 {
		return nil, fmt.Errorf("%w: GGUF version %d is not 2 or 3", ErrVocabulary, version)
	}
	r.u64() // tensor count; tensors are never read
	keys := r.u64()
	if r.err == nil && keys > maxGGUFKeys {
		return nil, fmt.Errorf("%w: %d metadata keys", ErrVocabulary, keys)
	}
	var (
		model  string
		tokens []string
		types  []int32
		found  = map[string]bool{}
	)
	for i := uint64(0); i < keys && r.err == nil; i++ {
		key := r.str(maxGGUFKeyBytes)
		kind := r.u32()
		if r.err != nil {
			break
		}
		if found[key] {
			return nil, fmt.Errorf("%w: metadata key %q repeated", ErrVocabulary, key)
		}
		found[key] = true
		switch key {
		case "tokenizer.ggml.model":
			if kind != ggufString {
				return nil, fmt.Errorf("%w: %s has type %d", ErrVocabulary, key, kind)
			}
			model = r.str(maxGGUFStringBytes)
		case "tokenizer.ggml.tokens":
			tokens = r.stringArray(kind)
		case "tokenizer.ggml.token_type":
			types = r.int32Array(kind)
		default:
			r.skip(kind, 0)
		}
	}
	if r.err != nil {
		return nil, fmt.Errorf("%w: metadata: %v", ErrVocabulary, r.err)
	}
	var scheme Scheme
	switch model {
	case "gpt2":
		scheme = SchemeByteLevel
	case "llama", "gemma4":
		scheme = SchemeSentencePiece
	default:
		return nil, fmt.Errorf("%w: tokenizer model %q is not supported", ErrVocabulary, model)
	}
	if tokens == nil {
		return nil, fmt.Errorf("%w: no tokenizer.ggml.tokens", ErrVocabulary)
	}
	if types != nil && len(types) != len(tokens) {
		return nil, fmt.Errorf("%w: %d token types for %d tokens", ErrVocabulary, len(types), len(tokens))
	}
	specs := make([]TokenSpec, len(tokens))
	for id, text := range tokens {
		kind := KindNormal
		if types != nil {
			kind = ggufKind(types[id])
		}
		specs[id] = TokenSpec{Text: text, Kind: kind}
	}
	return NewVocabulary(scheme, specs)
}

func ggufKind(t int32) Kind {
	switch t {
	case ggufTokenNormal:
		return KindNormal
	case ggufTokenControl:
		return KindControl
	case ggufTokenUserDefined:
		return KindUserDefined
	case ggufTokenUnused:
		return KindUnused
	case ggufTokenByte:
		return KindByte
	}
	return KindUnknown
}

// ggufReader reads little-endian GGUF values and keeps the first error.
type ggufReader struct {
	r   *bufio.Reader
	err error
}

func (g *ggufReader) read(v any) {
	if g.err == nil {
		g.err = binary.Read(g.r, binary.LittleEndian, v)
	}
}

func (g *ggufReader) u32() uint32 {
	var v uint32
	g.read(&v)
	return v
}

func (g *ggufReader) u64() uint64 {
	var v uint64
	g.read(&v)
	return v
}

func (g *ggufReader) str(limit uint64) string {
	n := g.u64()
	if g.err != nil {
		return ""
	}
	if n > limit {
		g.err = fmt.Errorf("string of %d bytes exceeds %d", n, limit)
		return ""
	}
	b := make([]byte, n)
	_, g.err = io.ReadFull(g.r, b)
	return string(b)
}

func (g *ggufReader) arrayHeader(kind uint32, element uint32) uint64 {
	if g.err != nil {
		return 0
	}
	if kind != ggufArray {
		g.err = fmt.Errorf("want an array, got type %d", kind)
		return 0
	}
	got := g.u32()
	n := g.u64()
	if g.err != nil {
		return 0
	}
	if got != element {
		g.err = fmt.Errorf("want array elements of type %d, got %d", element, got)
		return 0
	}
	if n > maxGGUFArray {
		g.err = fmt.Errorf("array of %d elements exceeds %d", n, maxGGUFArray)
		return 0
	}
	return n
}

func (g *ggufReader) stringArray(kind uint32) []string {
	n := g.arrayHeader(kind, ggufString)
	if g.err != nil {
		return nil
	}
	if n > maxTokens {
		g.err = fmt.Errorf("%d tokens exceed %d", n, maxTokens)
		return nil
	}
	out := make([]string, 0, n)
	for i := uint64(0); i < n && g.err == nil; i++ {
		out = append(out, g.str(maxGGUFStringBytes))
	}
	return out
}

func (g *ggufReader) int32Array(kind uint32) []int32 {
	n := g.arrayHeader(kind, ggufInt32)
	if g.err != nil {
		return nil
	}
	if n > maxTokens {
		g.err = fmt.Errorf("%d token types exceed %d", n, maxTokens)
		return nil
	}
	out := make([]int32, n)
	g.read(out)
	return out
}

// skip advances past one value of the given type without keeping it.
func (g *ggufReader) skip(kind uint32, depth int) {
	if g.err != nil {
		return
	}
	switch {
	case kind == ggufString:
		n := g.u64()
		if g.err == nil && n > maxGGUFStringBytes {
			g.err = fmt.Errorf("string of %d bytes exceeds %d", n, maxGGUFStringBytes)
		}
		g.discard(n)
	case kind == ggufArray:
		if depth > 0 {
			g.err = errors.New("nested GGUF arrays are not supported")
			return
		}
		element := g.u32()
		n := g.u64()
		if g.err == nil && n > maxGGUFArray {
			g.err = fmt.Errorf("array of %d elements exceeds %d", n, maxGGUFArray)
		}
		if w, fixed := ggufWidth(element); fixed {
			g.discard(n * uint64(w))
			return
		}
		for i := uint64(0); i < n && g.err == nil; i++ {
			g.skip(element, depth+1)
		}
	default:
		w, fixed := ggufWidth(kind)
		if !fixed {
			g.err = fmt.Errorf("unknown GGUF metadata type %d", kind)
			return
		}
		g.discard(uint64(w))
	}
}

// ggufWidth returns the size in bytes of a fixed-width metadata type.
func ggufWidth(kind uint32) (int, bool) {
	switch kind {
	case ggufUint8, ggufInt8, ggufBool:
		return 1, true
	case ggufUint16, ggufInt16:
		return 2, true
	case ggufUint32, ggufInt32, ggufFloat32:
		return 4, true
	case ggufUint64, ggufInt64, ggufFloat64:
		return 8, true
	}
	return 0, false
}

func (g *ggufReader) discard(n uint64) {
	if g.err != nil {
		return
	}
	if _, err := io.CopyN(io.Discard, g.r, int64(n)); err != nil {
		g.err = err
	}
}
