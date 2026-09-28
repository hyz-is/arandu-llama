package tokenizer_test

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"fmt"
	"math"
	"strings"
	"testing"

	"github.com/tayi-ai/arandu-llama/training/tokenizer"
)

// gguf writes GGUF metadata by hand, so the reader is checked against the
// specification's layout rather than against another reader.
type gguf struct {
	entries uint64
	body    bytes.Buffer
}

func (g *gguf) u32(w *bytes.Buffer, v uint32) { binary.Write(w, binary.LittleEndian, v) }
func (g *gguf) u64(w *bytes.Buffer, v uint64) { binary.Write(w, binary.LittleEndian, v) }
func (g *gguf) text(w *bytes.Buffer, s string) {
	g.u64(w, uint64(len(s)))
	w.WriteString(s)
}
func (g *gguf) key(name string, kind uint32) {
	g.entries++
	g.text(&g.body, name)
	g.u32(&g.body, kind)
}
func (g *gguf) texts(name string, values []string) {
	g.key(name, 9)
	g.u32(&g.body, 8)
	g.u64(&g.body, uint64(len(values)))
	for _, v := range values {
		g.text(&g.body, v)
	}
}
func (g *gguf) int32s(name string, values []int32) {
	g.key(name, 9)
	g.u32(&g.body, 5)
	g.u64(&g.body, uint64(len(values)))
	for _, v := range values {
		binary.Write(&g.body, binary.LittleEndian, v)
	}
}
func (g *gguf) scalar(name string, kind uint32, value any) {
	g.key(name, kind)
	binary.Write(&g.body, binary.LittleEndian, value)
}
func (g *gguf) str(name, value string) {
	g.key(name, 8)
	g.text(&g.body, value)
}

// metadata returns the header and key/value section, which is what the
// reader hashes, and nothing after it.
func (g *gguf) metadata(version uint32) []byte {
	var out bytes.Buffer
	out.WriteString("GGUF")
	g.u32(&out, version)
	g.u64(&out, 3)
	g.u64(&out, g.entries)
	out.Write(g.body.Bytes())
	return out.Bytes()
}

var (
	vocabularyTokens = []string{"a", "b", "ab", "c", "abc", "<|end|>"}
	vocabularyMerges = []string{"a b", "ab c"}
)

func vocabularyJSON(merges string) string {
	return `{"version":"1.0","added_tokens":[{"id":5,"content":"<|end|>","special":true}],"model":{"type":"BPE","vocab":{"a":0,"b":1,"ab":2,"c":3,"abc":4},"merges":` + merges + `}}`
}

// paddedGGUF declares the same six tokens followed by the [PAD…] tail a GGUF
// uses to reach the model's embedding rows, among unrelated metadata of every
// scalar type, a nested array and a string.
func paddedGGUF(tokens []string, merges []string) *gguf {
	g := &gguf{}
	g.str("general.architecture", "qwen35")
	g.scalar("general.alignment", 4, uint32(32))
	g.scalar("qwen35.rope.freq_base", 6, float32(1e7))
	g.scalar("tokenizer.ggml.add_bos_token", 7, uint8(0))
	g.scalar("qwen35.context_length", 10, uint64(4096))
	g.scalar("general.signed", 11, int64(-1))
	g.scalar("general.ratio", 12, float64(0.5))
	g.scalar("general.small", 0, uint8(1))
	g.scalar("general.short", 3, int16(-2))
	g.str("tokenizer.ggml.model", "gpt2")
	g.str("tokenizer.ggml.pre", "qwen35")
	g.texts("tokenizer.ggml.tokens", append(append([]string(nil), tokens...), "[PAD6]", "[PAD7]"))
	g.int32s("tokenizer.ggml.token_type", []int32{1, 1, 1, 1, 1, 3, 4, 4})
	g.texts("tokenizer.ggml.merges", merges)
	g.key("general.nested", 9)
	g.u32(&g.body, 9)
	g.u64(&g.body, 2)
	for _, n := range []int{1, 2} {
		g.u32(&g.body, 4)
		g.u64(&g.body, uint64(n))
		for range n {
			g.u32(&g.body, 7)
		}
	}
	g.str("tokenizer.chat_template", strings.Repeat("{{ x }}", 64))
	return g
}

func readJSON(t *testing.T, body string) tokenizer.Vocabulary {
	t.Helper()
	v, err := tokenizer.ReadVocabularyJSON(context.Background(), strings.NewReader(body), tokenizer.DefaultVocabularyLimits())
	if err != nil {
		t.Fatal(err)
	}
	return v
}

func readGGUF(t *testing.T, body []byte) tokenizer.Vocabulary {
	t.Helper()
	v, err := tokenizer.ReadVocabularyGGUF(context.Background(), bytes.NewReader(body), tokenizer.DefaultVocabularyLimits())
	if err != nil {
		t.Fatal(err)
	}
	return v
}

func sum(b []byte) string { s := sha256.Sum256(b); return hex.EncodeToString(s[:]) }

func TestAJSONAndAPaddedGGUFShareTheirCommonVocabularyDigest(t *testing.T) {
	lines := vocabularyJSON(`["a b","ab c"]`)
	pairs := vocabularyJSON(`[["a","b"],["ab","c"]]`)
	fromLines, fromPairs := readJSON(t, lines), readJSON(t, pairs)
	if fromLines.Format != tokenizer.VocabularyJSON || fromLines.SourceSHA256 != sum([]byte(lines)) || len(fromLines.Tokens) != 6 || fromLines.Tokens[5] != "<|end|>" {
		t.Fatalf("tokenizer.json measurement differs: %+v", fromLines)
	}
	metadata := paddedGGUF(vocabularyTokens, vocabularyMerges).metadata(3)
	// Tensor descriptors and data follow the metadata; neither is read.
	file := append(append([]byte(nil), metadata...), bytes.Repeat([]byte{0xff}, 4096)...)
	fromGGUF := readGGUF(t, file)
	if fromGGUF.Format != tokenizer.VocabularyGGUF || fromGGUF.SourceSHA256 != sum(metadata) || len(fromGGUF.Tokens) != 8 {
		t.Fatalf("GGUF measurement differs: %+v", fromGGUF)
	}
	if cut := readGGUF(t, metadata); cut.SourceSHA256 != fromGGUF.SourceSHA256 {
		t.Fatal("a GGUF cut after its metadata measured differently")
	}
	want, err := fromLines.Digest(6)
	if err != nil {
		t.Fatal(err)
	}
	for name, other := range map[string]tokenizer.Vocabulary{"pairs": fromPairs, "gguf": fromGGUF} {
		got, err := tokenizer.SameVocabulary(fromLines, other, 6)
		if err != nil || got != want {
			t.Fatalf("%s: shared digest %q, %v; want %q", name, got, err, want)
		}
	}
	// The padded tail is outside the common range and never compared.
	if _, err := tokenizer.SameVocabulary(fromLines, fromGGUF, 7); !errors.Is(err, tokenizer.ErrVocabulary) {
		t.Fatal("a common range past the shorter vocabulary was admitted", err)
	}
	if full, err := fromGGUF.Digest(8); err != nil || full == want {
		t.Fatal("the padded tail did not enter a digest that covers it", err)
	}
}

func TestSameVocabularyRefusesEveryDifferenceInsideTheCommonRange(t *testing.T) {
	base := readJSON(t, vocabularyJSON(`["a b","ab c"]`))
	for name, change := range map[string]func(*tokenizer.Vocabulary){
		"token":        func(v *tokenizer.Vocabulary) { v.Tokens[3] = "d" },
		"added token":  func(v *tokenizer.Vocabulary) { v.Tokens[5] = "<|eot|>" },
		"merge order":  func(v *tokenizer.Vocabulary) { v.Merges[0], v.Merges[1] = v.Merges[1], v.Merges[0] },
		"merge count":  func(v *tokenizer.Vocabulary) { v.Merges = v.Merges[:1] },
		"merge symbol": func(v *tokenizer.Vocabulary) { v.Merges[1].Right = "b" },
	} {
		t.Run(name, func(t *testing.T) {
			other := base
			other.Tokens = append([]string(nil), base.Tokens...)
			other.Merges = append([]tokenizer.MergeRule(nil), base.Merges...)
			change(&other)
			if _, err := tokenizer.SameVocabulary(base, other, 6); !errors.Is(err, tokenizer.ErrVocabulary) {
				t.Fatal("a different vocabulary was admitted", err)
			}
			a, _ := base.Digest(6)
			b, _ := other.Digest(6)
			if a == b {
				t.Fatal("two different vocabularies share a digest")
			}
		})
	}
	// A difference above the common range is outside what is compared.
	tail := base
	tail.Tokens = append(append([]string(nil), base.Tokens[:5]...), "<|audio_start|>")
	if _, err := tokenizer.SameVocabulary(base, tail, 5); err != nil {
		t.Fatal("a difference above the common range was compared", err)
	}
}

func TestVocabularyDigestKeepsTokenAndMergeBoundaries(t *testing.T) {
	one := tokenizer.Vocabulary{Tokens: []string{"ab", "c"}, Merges: []tokenizer.MergeRule{{Left: "a", Right: "bc"}}}
	two := tokenizer.Vocabulary{Tokens: []string{"a", "bc"}, Merges: []tokenizer.MergeRule{{Left: "ab", Right: "c"}}}
	three := tokenizer.Vocabulary{Tokens: []string{"ab", "c"}, Merges: []tokenizer.MergeRule{{Left: "ab", Right: "c"}}}
	seen := map[string]bool{}
	for _, v := range []tokenizer.Vocabulary{one, two, three} {
		d, err := v.Digest(2)
		if err != nil {
			t.Fatal(err)
		}
		if seen[d] {
			t.Fatal("moving a boundary kept the digest")
		}
		seen[d] = true
	}
	if _, err := one.Digest(0); !errors.Is(err, tokenizer.ErrVocabulary) {
		t.Fatal("an empty common range was admitted")
	}
}

func TestReadVocabularyJSONRefusesAmbiguousOrUndefinedIds(t *testing.T) {
	for name, body := range map[string]string{
		"gap":              `{"added_tokens":[],"model":{"type":"BPE","vocab":{"a":0,"b":1,"ab":3},"merges":["a b"]}}`,
		"repeated id":      `{"added_tokens":[],"model":{"type":"BPE","vocab":{"a":0,"b":0},"merges":["a b"]}}`,
		"added collision":  `{"added_tokens":[{"id":1,"content":"<|end|>"}],"model":{"type":"BPE","vocab":{"a":0,"b":1},"merges":["a b"]}}`,
		"added without id": `{"added_tokens":[{"content":"<|end|>"}],"model":{"type":"BPE","vocab":{"a":0,"b":1},"merges":["a b"]}}`,
		"null id":          `{"added_tokens":[],"model":{"type":"BPE","vocab":{"a":0,"b":null},"merges":["a b"]}}`,
		"negative id":      `{"added_tokens":[],"model":{"type":"BPE","vocab":{"a":-1,"b":0},"merges":["a b"]}}`,
		"merge line":       `{"added_tokens":[],"model":{"type":"BPE","vocab":{"a":0,"b":1},"merges":["a b c"]}}`,
		"merge pair":       `{"added_tokens":[],"model":{"type":"BPE","vocab":{"a":0,"b":1},"merges":[["a","b","c"]]}}`,
		"no merges":        `{"added_tokens":[],"model":{"type":"BPE","vocab":{"a":0,"b":1}}}`,
		"unigram":          `{"added_tokens":[],"model":{"type":"Unigram","vocab":{"a":0,"b":1},"merges":["a b"]}}`,
		"duplicate key":    `{"added_tokens":[],"model":{"type":"BPE","vocab":{"a":0,"a":1},"merges":["a b"]}}`,
	} {
		t.Run(name, func(t *testing.T) {
			if _, err := tokenizer.ReadVocabularyJSON(context.Background(), strings.NewReader(body), tokenizer.DefaultVocabularyLimits()); err == nil {
				t.Fatal("an ambiguous tokenizer.json was measured")
			}
		})
	}
	// An added token may restate the entry the model vocabulary already has.
	restated := `{"added_tokens":[{"id":1,"content":"b"}],"model":{"type":"BPE","vocab":{"a":0,"b":1},"merges":["a b"]}}`
	if v := readJSON(t, restated); len(v.Tokens) != 2 || v.Tokens[1] != "b" {
		t.Fatal("a restated added token changed the vocabulary", v.Tokens)
	}
	limits := tokenizer.DefaultVocabularyLimits()
	limits.MaxSourceBytes = 16
	if _, err := tokenizer.ReadVocabularyJSON(context.Background(), strings.NewReader(restated), limits); !errors.Is(err, tokenizer.ErrBudget) {
		t.Fatal("an oversized source was read", err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := tokenizer.ReadVocabularyJSON(ctx, strings.NewReader(restated), tokenizer.DefaultVocabularyLimits()); !errors.Is(err, context.Canceled) {
		t.Fatal("a cancelled measurement ran", err)
	}
}

func TestReadVocabularyGGUFRefusesMalformedMetadata(t *testing.T) {
	good := paddedGGUF(vocabularyTokens, vocabularyMerges).metadata(3)
	repeated := paddedGGUF(vocabularyTokens, vocabularyMerges)
	repeated.str("tokenizer.ggml.pre", "gpt-4o")
	numeric := &gguf{}
	numeric.int32s("tokenizer.ggml.tokens", []int32{1, 2})
	unknown := &gguf{}
	unknown.key("general.future", 13)
	merge := paddedGGUF(vocabularyTokens, []string{"a b", "ab  c"})
	none := &gguf{}
	none.str("tokenizer.ggml.model", "gpt2")
	for name, body := range map[string][]byte{
		"magic":         append([]byte("GGML"), good[4:]...),
		"version one":   paddedGGUF(vocabularyTokens, vocabularyMerges).metadata(1),
		"truncated":     good[:len(good)-40],
		"repeated key":  repeated.metadata(3),
		"numeric token": numeric.metadata(3),
		"unknown type":  unknown.metadata(3),
		"merge line":    merge.metadata(3),
		"no tokens":     none.metadata(3),
	} {
		t.Run(name, func(t *testing.T) {
			if _, err := tokenizer.ReadVocabularyGGUF(context.Background(), bytes.NewReader(body), tokenizer.DefaultVocabularyLimits()); err == nil {
				t.Fatal("malformed GGUF metadata was measured")
			}
		})
	}
	for name, change := range map[string]func(*tokenizer.VocabularyLimits){
		"source":  func(l *tokenizer.VocabularyLimits) { l.MaxSourceBytes = int64(len(good) - 1) },
		"tokens":  func(l *tokenizer.VocabularyLimits) { l.MaxTokens = 7 },
		"merges":  func(l *tokenizer.VocabularyLimits) { l.MaxMerges = 1 },
		"string":  func(l *tokenizer.VocabularyLimits) { l.MaxStringBytes = 64 },
		"entries": func(l *tokenizer.VocabularyLimits) { l.MaxMetadataEntries = 4 },
		"invalid": func(l *tokenizer.VocabularyLimits) { l.MaxSourceBytes = math.MaxInt64 },
	} {
		t.Run("limit "+name, func(t *testing.T) {
			limits := tokenizer.DefaultVocabularyLimits()
			change(&limits)
			if _, err := tokenizer.ReadVocabularyGGUF(context.Background(), bytes.NewReader(good), limits); err == nil {
				t.Fatal("a GGUF beyond its limits was measured")
			}
		})
	}
	limits := tokenizer.DefaultVocabularyLimits()
	limits.MaxSourceBytes = int64(len(good))
	if _, err := tokenizer.ReadVocabularyGGUF(context.Background(), bytes.NewReader(good), limits); err != nil {
		t.Fatal(fmt.Errorf("metadata exactly at the byte bound was refused: %w", err))
	}
}
