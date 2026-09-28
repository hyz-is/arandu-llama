package bridge

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"fmt"
	"strings"
	"unicode/utf8"
)

// ErrVocabulary reports a vocabulary source or token list outside the
// supported schema.
var ErrVocabulary = errors.New("bridge: vocabulary rejected")

// maxTokens bounds a vocabulary so that a corrupt source cannot demand an
// unbounded allocation.
const maxTokens = 1 << 22

// Scheme names how a tokenizer writes the bytes of a token as text.
type Scheme uint8

const (
	// SchemeByteLevel is GPT-2 byte-level BPE: every byte of a normal token is
	// written as one rune of the bytes-to-unicode table. Qwen, Ornith and the
	// o200k vocabulary of gpt-oss use it.
	SchemeByteLevel Scheme = iota + 1
	// SchemeSentencePiece writes a space as U+2581 and a byte that no piece
	// covers as a <0xNN> byte-fallback token. Gemma uses it.
	SchemeSentencePiece
)

// String names the scheme.
func (s Scheme) String() string {
	switch s {
	case SchemeByteLevel:
		return "byte-level"
	case SchemeSentencePiece:
		return "sentencepiece"
	}
	return fmt.Sprintf("scheme(%d)", uint8(s))
}

// Kind classifies a token by what it contributes to the byte stream.
type Kind uint8

const (
	// KindNormal is text written in the scheme's encoding.
	KindNormal Kind = iota + 1
	// KindByte is a <0xNN> byte-fallback token; it stands for one byte.
	KindByte
	// KindUserDefined is an added token matched literally; its text is its
	// bytes, as the llama.cpp detokenizer copies it.
	KindUserDefined
	// KindControl is a special or control token. It has no bytes.
	KindControl
	// KindUnknown is the unknown token or a token of undefined type. It has
	// no bytes.
	KindUnknown
	// KindUnused is a reserved or padding slot. It has no bytes.
	KindUnused
)

// String names the kind.
func (k Kind) String() string {
	switch k {
	case KindNormal:
		return "normal"
	case KindByte:
		return "byte"
	case KindUserDefined:
		return "user-defined"
	case KindControl:
		return "control"
	case KindUnknown:
		return "unknown"
	case KindUnused:
		return "unused"
	}
	return fmt.Sprintf("kind(%d)", uint8(k))
}

// TokenSpec is one vocabulary entry as its source declares it: the token's
// text in the source's own encoding and its kind.
type TokenSpec struct {
	Text string
	Kind Kind
}

// Stats counts the tokens of a Vocabulary by kind and by representation.
type Stats struct {
	Tokens      int
	Normal      int
	Byte        int
	UserDefined int
	Control     int
	Unknown     int
	Unused      int
	// WithBytes counts the tokens that take part in marginalization.
	WithBytes int
	// Unrepresentable counts normal and user-defined tokens whose text does
	// not decode to bytes under the scheme. They are treated as byte-less.
	Unrepresentable int
	// SharedBytes counts tokens whose bytes equal those of a lower id.
	SharedBytes int
}

// Vocabulary holds the exact bytes every token of one tokenizer decodes to.
// It is immutable after construction and safe for concurrent use.
type Vocabulary struct {
	scheme Scheme
	texts  []string
	kinds  []Kind
	// bytes is the decoded representation; empty when the token has none.
	bytes           []string
	stats           Stats
	unrepresentable []int64
	digest          string
}

// NewVocabulary decodes tokens, given in id order, under scheme. Control,
// unknown and unused tokens carry no bytes. A normal or user-defined token
// whose text cannot be decoded is kept, counted as unrepresentable, and
// carries no bytes; a byte token whose text is not <0xNN> is an error.
func NewVocabulary(scheme Scheme, tokens []TokenSpec) (*Vocabulary, error) {
	if scheme != SchemeByteLevel && scheme != SchemeSentencePiece {
		return nil, fmt.Errorf("%w: unsupported scheme %d", ErrVocabulary, scheme)
	}
	if len(tokens) == 0 || len(tokens) > maxTokens {
		return nil, fmt.Errorf("%w: %d tokens is outside 1..%d", ErrVocabulary, len(tokens), maxTokens)
	}
	v := &Vocabulary{
		scheme: scheme,
		texts:  make([]string, len(tokens)),
		kinds:  make([]Kind, len(tokens)),
		bytes:  make([]string, len(tokens)),
	}
	v.stats.Tokens = len(tokens)
	seen := make(map[string]struct{}, len(tokens))
	for id, token := range tokens {
		v.texts[id], v.kinds[id] = token.Text, token.Kind
		decoded, ok, err := decodeToken(scheme, token)
		if err != nil {
			return nil, fmt.Errorf("%w: token %d: %v", ErrVocabulary, id, err)
		}
		switch token.Kind {
		case KindNormal:
			v.stats.Normal++
		case KindByte:
			v.stats.Byte++
		case KindUserDefined:
			v.stats.UserDefined++
		case KindControl:
			v.stats.Control++
		case KindUnknown:
			v.stats.Unknown++
		case KindUnused:
			v.stats.Unused++
		}
		if !ok {
			if token.Kind == KindNormal || token.Kind == KindUserDefined {
				v.stats.Unrepresentable++
				v.unrepresentable = append(v.unrepresentable, int64(id))
			}
			continue
		}
		v.bytes[id] = decoded
		v.stats.WithBytes++
		if _, shared := seen[decoded]; shared {
			v.stats.SharedBytes++
		} else {
			seen[decoded] = struct{}{}
		}
	}
	v.digest = vocabularyDigest(scheme, tokens)
	return v, nil
}

// decodeToken returns the bytes of one token and whether it has any.
func decodeToken(scheme Scheme, token TokenSpec) (string, bool, error) {
	switch token.Kind {
	case KindControl, KindUnknown, KindUnused:
		return "", false, nil
	case KindByte:
		b, ok := byteFallback(token.Text)
		if !ok {
			return "", false, fmt.Errorf("byte token %q is not <0xNN>", token.Text)
		}
		return string([]byte{b}), true, nil
	case KindUserDefined:
		return token.Text, token.Text != "", nil
	case KindNormal:
		if token.Text == "" {
			return "", false, nil
		}
		if scheme == SchemeSentencePiece {
			return strings.ReplaceAll(token.Text, "▁", " "), true, nil
		}
		return byteLevelDecode(token.Text)
	}
	return "", false, fmt.Errorf("unknown kind %d", token.Kind)
}

// byteFallback parses a SentencePiece byte token of the form <0xNN>.
func byteFallback(text string) (byte, bool) {
	if len(text) != 6 || !strings.HasPrefix(text, "<0x") || text[5] != '>' {
		return 0, false
	}
	decoded, err := hex.DecodeString(text[3:5])
	if err != nil {
		return 0, false
	}
	return decoded[0], true
}

// byteLevelRunes maps a rune of the GPT-2 bytes-to-unicode table back to the
// byte it stands for; entries not in the table hold -1.
var byteLevelRunes = func() [512]int16 {
	var table [512]int16
	for i := range table {
		table[i] = -1
	}
	next := 256
	for b := range 256 {
		if b >= 33 && b <= 126 || b >= 161 && b <= 172 || b >= 174 {
			table[b] = int16(b)
			continue
		}
		table[next] = int16(b)
		next++
	}
	return table
}()

// byteLevelDecode inverts the bytes-to-unicode table. A rune outside it makes
// the token unrepresentable, where llama.cpp would print a placeholder.
func byteLevelDecode(text string) (string, bool, error) {
	if !utf8.ValidString(text) {
		return "", false, nil
	}
	out := make([]byte, 0, len(text))
	for _, r := range text {
		if r < 0 || int(r) >= len(byteLevelRunes) || byteLevelRunes[r] < 0 {
			return "", false, nil
		}
		out = append(out, byte(byteLevelRunes[r]))
	}
	return string(out), true, nil
}

// vocabularyDigest hashes the scheme and every token's kind and text, each
// length-prefixed, in id order.
func vocabularyDigest(scheme Scheme, tokens []TokenSpec) string {
	h := sha256.New()
	h.Write([]byte("bridge-vocabulary-v1\x00"))
	var word [8]byte
	h.Write([]byte{byte(scheme)})
	binary.LittleEndian.PutUint64(word[:], uint64(len(tokens)))
	h.Write(word[:])
	for _, token := range tokens {
		h.Write([]byte{byte(token.Kind)})
		binary.LittleEndian.PutUint64(word[:], uint64(len(token.Text)))
		h.Write(word[:])
		h.Write([]byte(token.Text))
	}
	return hex.EncodeToString(h.Sum(nil))
}

// Scheme returns the scheme the vocabulary was decoded under.
func (v *Vocabulary) Scheme() Scheme { return v.scheme }

// Size returns the number of token ids, including byte-less ones.
func (v *Vocabulary) Size() int { return len(v.kinds) }

// Stats returns the counts computed at construction.
func (v *Vocabulary) Stats() Stats { return v.stats }

// Digest returns the SHA-256 of the scheme and the token list in id order.
func (v *Vocabulary) Digest() string { return v.digest }

// Unrepresentable returns, in ascending order, the ids of normal and
// user-defined tokens whose text did not decode to bytes.
func (v *Vocabulary) Unrepresentable() []int64 {
	return append([]int64(nil), v.unrepresentable...)
}

// Kind returns the kind of token id, or zero when id is out of range.
func (v *Vocabulary) Kind(id int64) Kind {
	if id < 0 || id >= int64(len(v.kinds)) {
		return 0
	}
	return v.kinds[id]
}

// Text returns the source text of token id, or an empty string when id is
// out of range.
func (v *Vocabulary) Text(id int64) string {
	if id < 0 || id >= int64(len(v.texts)) {
		return ""
	}
	return v.texts[id]
}

// Bytes returns a copy of the bytes token id decodes to, and false when the
// token has none or id is out of range.
func (v *Vocabulary) Bytes(id int64) ([]byte, bool) {
	s, ok := v.representation(id)
	if !ok {
		return nil, false
	}
	return []byte(s), true
}

func (v *Vocabulary) representation(id int64) (string, bool) {
	if id < 0 || id >= int64(len(v.bytes)) || v.bytes[id] == "" {
		return "", false
	}
	return v.bytes[id], true
}

// Find returns the token whose bytes are exactly b, preferring a
// user-defined token, then a normal token, then a byte token, then the lowest
// id -- the token a tokenizer emits for those bytes. It scans the vocabulary.
func (v *Vocabulary) Find(b []byte) (int64, bool) {
	best := int64(-1)
	for id, s := range v.bytes {
		if s == "" || s != string(b) {
			continue
		}
		if best < 0 || kindPriority(v.kinds[id]) > kindPriority(v.kinds[best]) {
			best = int64(id)
		}
	}
	return best, best >= 0
}

// WhitespaceOnly reports whether token id decodes to a nonempty run of spaces
// and tabs only: the rows BPM withholds from the loss.
func (v *Vocabulary) WhitespaceOnly(id int64) bool {
	s, ok := v.representation(id)
	if !ok {
		return false
	}
	for i := 0; i < len(s); i++ {
		if s[i] != ' ' && s[i] != '\t' {
			return false
		}
	}
	return true
}

// kindPriority orders tokens that share bytes by which one a tokenizer emits:
// added tokens are matched before the model runs, and byte fallback is used
// only when no piece covers the byte.
func kindPriority(k Kind) int {
	switch k {
	case KindUserDefined:
		return 3
	case KindNormal:
		return 2
	case KindByte:
		return 1
	}
	return 0
}
