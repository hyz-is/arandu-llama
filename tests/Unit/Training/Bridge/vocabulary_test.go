package bridge_test

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"errors"
	"testing"

	"github.com/tayi-ai/arandu-llama/training/bridge"
)

func TestByteLevelTokensDecodeThroughTheGPT2Table(t *testing.T) {
	v, err := bridge.NewVocabulary(bridge.SchemeByteLevel, []bridge.TokenSpec{
		{Text: "Ġthe", Kind: bridge.KindNormal},
		{Text: "ĊĊ", Kind: bridge.KindNormal},
		{Text: "Ã©", Kind: bridge.KindNormal},
		{Text: "ĉ", Kind: bridge.KindNormal},
		{Text: "<think>", Kind: bridge.KindUserDefined},
		{Text: "<|im_end|>", Kind: bridge.KindControl},
		{Text: "€", Kind: bridge.KindNormal},
	})
	if err != nil {
		t.Fatal(err)
	}
	for id, want := range []string{" the", "\n\n", "é", "\t", "<think>"} {
		got, ok := v.Bytes(int64(id))
		if !ok || string(got) != want {
			t.Errorf("token %d decodes to %q (%v), want %q", id, got, ok, want)
		}
	}
	if _, ok := v.Bytes(5); ok {
		t.Error("a control token has bytes")
	}
	if _, ok := v.Bytes(6); ok {
		t.Error("a rune outside the bytes-to-unicode table decoded")
	}
	stats := v.Stats()
	if stats.WithBytes != 5 || stats.Unrepresentable != 1 || stats.Control != 1 || stats.UserDefined != 1 || stats.Normal != 5 {
		t.Errorf("stats %+v", stats)
	}
	if got := v.Unrepresentable(); len(got) != 1 || got[0] != 6 {
		t.Errorf("unrepresentable %v, want [6]", got)
	}
	if !v.WhitespaceOnly(3) || v.WhitespaceOnly(1) || v.WhitespaceOnly(0) {
		t.Error("whitespace-only is spaces and tabs, not newlines or words")
	}
}

func TestSentencePieceTokensTurnTheMarkerIntoASpaceAndBytesIntoBytes(t *testing.T) {
	v, err := bridge.NewVocabulary(bridge.SchemeSentencePiece, []bridge.TokenSpec{
		{Text: "▁the", Kind: bridge.KindNormal},
		{Text: "<0x0A>", Kind: bridge.KindByte},
		{Text: "<0xE2>", Kind: bridge.KindByte},
		{Text: "▁", Kind: bridge.KindNormal},
		{Text: "<0x20>", Kind: bridge.KindByte},
		{Text: "▁▁", Kind: bridge.KindNormal},
		{Text: "<unused0>", Kind: bridge.KindUnused},
	})
	if err != nil {
		t.Fatal(err)
	}
	for id, want := range []string{" the", "\n", "\xe2", " ", " ", "  "} {
		got, ok := v.Bytes(int64(id))
		if !ok || string(got) != want {
			t.Errorf("token %d decodes to %q (%v), want %q", id, got, ok, want)
		}
	}
	if got, _ := v.Find([]byte(" ")); got != 3 {
		t.Errorf("Find(space) = %d, want the normal piece 3 over the byte token 4", got)
	}
	if stats := v.Stats(); stats.SharedBytes != 1 || stats.Byte != 3 || stats.Unused != 1 {
		t.Errorf("stats %+v", stats)
	}
	if _, err := bridge.NewVocabulary(bridge.SchemeSentencePiece, []bridge.TokenSpec{{Text: "<0xZZ>", Kind: bridge.KindByte}}); !errors.Is(err, bridge.ErrVocabulary) {
		t.Errorf("malformed byte token accepted: %v", err)
	}
}

func TestTheVocabularyDigestNamesTheTokenList(t *testing.T) {
	a := vocab(t, "a", "b")
	b := vocab(t, "a", "b")
	c := vocab(t, "a", "!control:b")
	if a.Digest() != b.Digest() || a.Digest() == c.Digest() || len(a.Digest()) != 64 {
		t.Errorf("digests %s %s %s", a.Digest(), b.Digest(), c.Digest())
	}
}

// gguf writes a GGUF v3 header with the given metadata and no tensors.
type gguf struct{ bytes.Buffer }

func (g *gguf) u32(v uint32) { binary.Write(&g.Buffer, binary.LittleEndian, v) }
func (g *gguf) u64(v uint64) { binary.Write(&g.Buffer, binary.LittleEndian, v) }
func (g *gguf) str(s string) { g.u64(uint64(len(s))); g.WriteString(s) }

func newGGUF(version uint32, keys uint64) *gguf {
	g := &gguf{}
	g.WriteString("GGUF")
	g.u32(version)
	g.u64(0)
	g.u64(keys)
	return g
}

func (g *gguf) stringKey(key, value string) { g.str(key); g.u32(8); g.str(value) }
func (g *gguf) stringArray(key string, values ...string) {
	g.str(key)
	g.u32(9)
	g.u32(8)
	g.u64(uint64(len(values)))
	for _, v := range values {
		g.str(v)
	}
}
func (g *gguf) int32Array(key string, values ...int32) {
	g.str(key)
	g.u32(9)
	g.u32(5)
	g.u64(uint64(len(values)))
	for _, v := range values {
		binary.Write(&g.Buffer, binary.LittleEndian, v)
	}
}

func TestReadGGUFTakesTheTokensAndTheirTypesAndSkipsTheRest(t *testing.T) {
	g := newGGUF(3, 7)
	g.stringKey("general.architecture", "gemma4")
	g.str("general.alignment")
	g.u32(4)
	g.u32(32)
	g.str("general.scores")
	g.u32(9)
	g.u32(6)
	g.u64(3)
	g.Write(make([]byte, 12))
	g.stringKey("tokenizer.ggml.model", "gemma4")
	g.stringArray("tokenizer.ggml.tokens", "<eos>", "▁the", "<0x0A>", "<|channel>")
	g.int32Array("tokenizer.ggml.token_type", 3, 1, 6, 4)
	g.stringArray("tokenizer.ggml.merges", "▁ t", "h e")
	g.WriteString("tensor infos follow and are never read")

	v, err := bridge.ReadGGUF(bytes.NewReader(g.Bytes()))
	if err != nil {
		t.Fatal(err)
	}
	if v.Scheme() != bridge.SchemeSentencePiece || v.Size() != 4 {
		t.Fatalf("scheme %s size %d", v.Scheme(), v.Size())
	}
	wantKinds := []bridge.Kind{bridge.KindControl, bridge.KindNormal, bridge.KindByte, bridge.KindUserDefined}
	wantBytes := []string{"", " the", "\n", "<|channel>"}
	for i := range wantKinds {
		got, _ := v.Bytes(int64(i))
		if v.Kind(int64(i)) != wantKinds[i] || string(got) != wantBytes[i] {
			t.Errorf("token %d: %s %q, want %s %q", i, v.Kind(int64(i)), got, wantKinds[i], wantBytes[i])
		}
	}
}

func TestReadGGUFRefusesWhatItCannotDecode(t *testing.T) {
	cases := map[string]func() []byte{
		"magic": func() []byte { return []byte("GGML\x03\x00\x00\x00") },
		"version": func() []byte {
			g := newGGUF(1, 0)
			return g.Bytes()
		},
		"model": func() []byte {
			g := newGGUF(3, 2)
			g.stringKey("tokenizer.ggml.model", "rwkv")
			g.stringArray("tokenizer.ggml.tokens", "a")
			return g.Bytes()
		},
		"types": func() []byte {
			g := newGGUF(3, 3)
			g.stringKey("tokenizer.ggml.model", "gpt2")
			g.stringArray("tokenizer.ggml.tokens", "a", "b")
			g.int32Array("tokenizer.ggml.token_type", 1)
			return g.Bytes()
		},
		"truncated": func() []byte {
			g := newGGUF(3, 2)
			g.stringKey("tokenizer.ggml.model", "gpt2")
			return g.Bytes()
		},
		"no tokens": func() []byte {
			g := newGGUF(3, 1)
			g.stringKey("tokenizer.ggml.model", "gpt2")
			return g.Bytes()
		},
	}
	for name, build := range cases {
		if _, err := bridge.ReadGGUF(bytes.NewReader(build())); !errors.Is(err, bridge.ErrVocabulary) {
			t.Errorf("%s: err %v, want ErrVocabulary", name, err)
		}
	}
}

func TestReadTokenizerJSONPlacesVocabularyAndAddedTokens(t *testing.T) {
	doc := map[string]any{
		"added_tokens": []map[string]any{
			{"id": 4, "content": "<|im_end|>", "special": true},
			{"id": 5, "content": "<think>", "special": false},
		},
		"decoder": map[string]any{"type": "ByteLevel"},
		"model": map[string]any{
			"type":  "BPE",
			"vocab": map[string]int{"Ġthe": 0, "ĊĊ": 1, "1": 2},
		},
	}
	body, _ := json.Marshal(doc)
	v, err := bridge.ReadTokenizerJSON(bytes.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	wantKinds := []bridge.Kind{bridge.KindNormal, bridge.KindNormal, bridge.KindNormal, bridge.KindUnused, bridge.KindControl, bridge.KindUserDefined}
	wantBytes := []string{" the", "\n\n", "1", "", "", "<think>"}
	if v.Size() != len(wantKinds) || v.Scheme() != bridge.SchemeByteLevel {
		t.Fatalf("size %d scheme %s", v.Size(), v.Scheme())
	}
	for i := range wantKinds {
		got, _ := v.Bytes(int64(i))
		if v.Kind(int64(i)) != wantKinds[i] || string(got) != wantBytes[i] {
			t.Errorf("token %d: %s %q, want %s %q", i, v.Kind(int64(i)), got, wantKinds[i], wantBytes[i])
		}
	}

	doc["decoder"] = map[string]any{"type": "Sequence"}
	doc["model"] = map[string]any{"type": "BPE", "byte_fallback": true, "vocab": map[string]int{"▁a": 0, "<0x41>": 1}}
	doc["added_tokens"] = []map[string]any{}
	body, _ = json.Marshal(doc)
	v, err = bridge.ReadTokenizerJSON(bytes.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	if got, _ := v.Bytes(1); v.Scheme() != bridge.SchemeSentencePiece || v.Kind(1) != bridge.KindByte || string(got) != "A" {
		t.Errorf("byte-fallback vocabulary: %s %s %q", v.Scheme(), v.Kind(1), got)
	}

	doc["model"] = map[string]any{"type": "BPE", "vocab": map[string]int{"a": 0}}
	body, _ = json.Marshal(doc)
	if _, err := bridge.ReadTokenizerJSON(bytes.NewReader(body)); !errors.Is(err, bridge.ErrVocabulary) {
		t.Errorf("a vocabulary with no byte scheme was accepted: %v", err)
	}
}
