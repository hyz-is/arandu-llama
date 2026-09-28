package bridge

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"strconv"
)

// maxTokenizerJSONBytes bounds the tokenizer.json a reader accepts.
const maxTokenizerJSONBytes = 64 << 20

// LoadTokenizerJSON reads the Hugging Face tokenizer.json at path.
func LoadTokenizerJSON(path string) (*Vocabulary, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	return ReadTokenizerJSON(file)
}

// ReadTokenizerJSON builds a Vocabulary from a Hugging Face tokenizer.json:
// the model vocabulary placed at its ids, then the added tokens, special ones
// as control tokens and the others as user-defined. A ByteLevel decoder makes
// the scheme byte-level; a model with byte_fallback makes it SentencePiece,
// where <0xNN> entries are byte tokens. An id no entry claims is unused.
func ReadTokenizerJSON(source io.Reader) (*Vocabulary, error) {
	body, err := io.ReadAll(io.LimitReader(source, maxTokenizerJSONBytes+1))
	if err != nil {
		return nil, err
	}
	if len(body) > maxTokenizerJSONBytes {
		return nil, fmt.Errorf("%w: tokenizer.json exceeds %d bytes", ErrVocabulary, maxTokenizerJSONBytes)
	}
	var doc struct {
		AddedTokens []struct {
			ID      *int64 `json:"id"`
			Content string `json:"content"`
			Special bool   `json:"special"`
		} `json:"added_tokens"`
		Decoder *struct {
			Type string `json:"type"`
		} `json:"decoder"`
		Model struct {
			Type         string            `json:"type"`
			ByteFallback bool              `json:"byte_fallback"`
			Vocab        map[string]jsonID `json:"vocab"`
		} `json:"model"`
	}
	if err := json.Unmarshal(body, &doc); err != nil {
		return nil, fmt.Errorf("%w: tokenizer.json: %v", ErrVocabulary, err)
	}
	var scheme Scheme
	switch {
	case doc.Model.Type != "BPE":
		return nil, fmt.Errorf("%w: model type %q is not BPE", ErrVocabulary, doc.Model.Type)
	case doc.Decoder != nil && doc.Decoder.Type == "ByteLevel":
		scheme = SchemeByteLevel
	case doc.Model.ByteFallback:
		scheme = SchemeSentencePiece
	default:
		return nil, fmt.Errorf("%w: neither a ByteLevel decoder nor byte fallback", ErrVocabulary)
	}
	if len(doc.Model.Vocab) == 0 {
		return nil, fmt.Errorf("%w: empty model vocabulary", ErrVocabulary)
	}
	size := int64(0)
	for _, id := range doc.Model.Vocab {
		size = max(size, int64(id)+1)
	}
	for _, added := range doc.AddedTokens {
		if added.ID == nil {
			return nil, fmt.Errorf("%w: added token %q without id", ErrVocabulary, added.Content)
		}
		size = max(size, *added.ID+1)
	}
	if size > maxTokens {
		return nil, fmt.Errorf("%w: %d ids exceed %d", ErrVocabulary, size, maxTokens)
	}
	specs := make([]TokenSpec, size)
	for text, sourceID := range doc.Model.Vocab {
		id := int64(sourceID)
		if id < 0 {
			return nil, fmt.Errorf("%w: negative id for %q", ErrVocabulary, text)
		}
		if specs[id].Kind != 0 {
			return nil, fmt.Errorf("%w: id %d assigned twice", ErrVocabulary, id)
		}
		kind := KindNormal
		if _, isByte := byteFallback(text); isByte && scheme == SchemeSentencePiece {
			kind = KindByte
		}
		specs[id] = TokenSpec{Text: text, Kind: kind}
	}
	for _, added := range doc.AddedTokens {
		id := *added.ID
		if id < 0 {
			return nil, fmt.Errorf("%w: negative id for added token %q", ErrVocabulary, added.Content)
		}
		if specs[id].Kind != 0 && specs[id].Text != added.Content {
			return nil, fmt.Errorf("%w: id %d is %q in the vocabulary and %q as an added token", ErrVocabulary, id, specs[id].Text, added.Content)
		}
		kind := KindUserDefined
		if added.Special {
			kind = KindControl
		}
		specs[id] = TokenSpec{Text: added.Content, Kind: kind}
	}
	for id := range specs {
		if specs[id].Kind == 0 {
			specs[id].Kind = KindUnused
		}
	}
	return NewVocabulary(scheme, specs)
}

// jsonID refuses a null or fractional id, which encoding/json would otherwise
// turn into zero or truncate.
type jsonID int64

// UnmarshalJSON parses a decimal integer id.
func (id *jsonID) UnmarshalJSON(body []byte) error {
	value, err := strconv.ParseInt(string(body), 10, 64)
	if err != nil {
		return err
	}
	*id = jsonID(value)
	return nil
}
