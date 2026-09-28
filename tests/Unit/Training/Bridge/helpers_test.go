package bridge_test

import (
	"strings"
	"testing"

	"github.com/tayi-ai/arandu-llama/training/bridge"
)

// byteLevel writes raw bytes in the GPT-2 bytes-to-unicode encoding, the
// form a byte-level vocabulary stores them in.
func byteLevel(raw string) string {
	var alphabet [256]rune
	extra := rune(256)
	for b := range 256 {
		alphabet[b] = rune(b)
		if b < 33 || b > 126 && b < 161 || b == 173 {
			alphabet[b] = extra
			extra++
		}
	}
	var out strings.Builder
	for i := 0; i < len(raw); i++ {
		out.WriteRune(alphabet[raw[i]])
	}
	return out.String()
}

// vocab builds a byte-level vocabulary from raw byte strings; entries of the
// form "!control:<text>" become control tokens and "!user:<text>"
// user-defined tokens.
func vocab(t *testing.T, raw ...string) *bridge.Vocabulary {
	t.Helper()
	specs := make([]bridge.TokenSpec, len(raw))
	for i, r := range raw {
		switch {
		case strings.HasPrefix(r, "!control:"):
			specs[i] = bridge.TokenSpec{Text: strings.TrimPrefix(r, "!control:"), Kind: bridge.KindControl}
		case strings.HasPrefix(r, "!user:"):
			specs[i] = bridge.TokenSpec{Text: strings.TrimPrefix(r, "!user:"), Kind: bridge.KindUserDefined}
		default:
			specs[i] = bridge.TokenSpec{Text: byteLevel(r), Kind: bridge.KindNormal}
		}
	}
	v, err := bridge.NewVocabulary(bridge.SchemeByteLevel, specs)
	if err != nil {
		t.Fatalf("NewVocabulary: %v", err)
	}
	return v
}

// id finds the token whose bytes are raw.
func id(t *testing.T, v *bridge.Vocabulary, raw string) int64 {
	t.Helper()
	found, ok := v.Find([]byte(raw))
	if !ok {
		t.Fatalf("no token for %q", raw)
	}
	return found
}
