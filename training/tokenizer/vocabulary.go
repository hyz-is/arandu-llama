package tokenizer

import (
	"bufio"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"hash"
	"io"
	"math"
	"strings"
)

// VocabularyAlgorithm names the encoding Vocabulary.Digest hashes. A digest is
// comparable only with another digest of the same algorithm.
const VocabularyAlgorithm = "tayi-vocabulary-v1"

// ErrVocabulary reports a vocabulary source that is refused, or two
// vocabularies that differ where they were required to agree.
var ErrVocabulary = errors.New("tokenizer: vocabulary refused")

// Vocabulary formats a measurement reads.
const (
	VocabularyJSON = "tokenizer.json"
	VocabularyGGUF = "gguf"
)

// MergeRule is one BPE merge: the two symbols it joins, in rank order.
type MergeRule struct {
	Left, Right string
}

// Vocabulary is what one tokenizer artifact declares: the token string of
// every id, in id order, and the merges, in rank order. Format is
// VocabularyJSON or VocabularyGGUF. SourceSHA256 is the SHA-256 of the bytes
// the measurement read: the whole tokenizer.json, or a GGUF's header and
// metadata up to its first tensor descriptor, which is where a GGUF keeps its
// tokenizer. Tensor data is never read.
type Vocabulary struct {
	Format       string
	SourceSHA256 string
	Tokens       []string
	Merges       []MergeRule
}

// VocabularyLimits bounds one measured source. MaxSourceBytes applies to the
// bytes read, which for a GGUF is its metadata only. MaxStringBytes applies to
// any single GGUF string, key or value. These are payload bounds, not a
// measured process peak.
type VocabularyLimits struct {
	MaxSourceBytes     int64
	MaxTokens          int
	MaxMerges          int
	MaxStringBytes     int
	MaxMetadataEntries int
}

// DefaultVocabularyLimits admits the tokenizers of the model families this
// package measures: about 250 thousand tokens and 500 thousand merges, whose
// GGUF metadata stays well below 64 MiB.
func DefaultVocabularyLimits() VocabularyLimits {
	return VocabularyLimits{MaxSourceBytes: 64 << 20, MaxTokens: 1 << 20, MaxMerges: 1 << 21, MaxStringBytes: 1 << 20, MaxMetadataEntries: 1 << 16}
}

func (l VocabularyLimits) valid() bool {
	return l.MaxSourceBytes > 0 && l.MaxSourceBytes != math.MaxInt64 && l.MaxTokens > 0 && l.MaxMerges > 0 && l.MaxStringBytes > 0 && l.MaxMetadataEntries > 0
}

// Digest hashes ids [0, commonTokens) and every merge under
// VocabularyAlgorithm. Each string carries its byte length and each list its
// count, so no two different vocabularies share an encoding by moving a
// separator, and a merge is a pair rather than a line that could be split
// differently. Ids at and above commonTokens are not part of the digest.
func (v Vocabulary) Digest(commonTokens int) (string, error) {
	if commonTokens < 1 || commonTokens > len(v.Tokens) {
		return "", fmt.Errorf("%w: common range of %d ids exceeds a vocabulary of %d", ErrVocabulary, commonTokens, len(v.Tokens))
	}
	h := sha256.New()
	h.Write([]byte(VocabularyAlgorithm + "\x00"))
	writeLength(h, commonTokens)
	for _, token := range v.Tokens[:commonTokens] {
		writeString(h, token)
	}
	writeLength(h, len(v.Merges))
	for _, merge := range v.Merges {
		writeString(h, merge.Left)
		writeString(h, merge.Right)
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

func writeLength(h hash.Hash, n int) {
	var word [8]byte
	binary.LittleEndian.PutUint64(word[:], uint64(n))
	h.Write(word[:])
}

func writeString(h hash.Hash, s string) {
	writeLength(h, len(s))
	h.Write([]byte(s))
}

// SameVocabulary returns the digest two vocabularies share over ids
// [0, commonTokens) and all their merges, or refuses with the first id or
// merge rank where they differ. Vocabularies of different lengths are compared
// on the declared common range only: a GGUF pads its list up to the model's
// embedding rows, and one tokenizer.json may declare tokens another lacks.
// Whoever uses the result must refuse any id at or above commonTokens.
func SameVocabulary(a, b Vocabulary, commonTokens int) (string, error) {
	if commonTokens < 1 || commonTokens > len(a.Tokens) || commonTokens > len(b.Tokens) {
		return "", fmt.Errorf("%w: common range of %d ids exceeds vocabularies of %d and %d", ErrVocabulary, commonTokens, len(a.Tokens), len(b.Tokens))
	}
	for id := range commonTokens {
		if a.Tokens[id] != b.Tokens[id] {
			return "", fmt.Errorf("%w: token %d is %q and %q", ErrVocabulary, id, a.Tokens[id], b.Tokens[id])
		}
	}
	for rank := range min(len(a.Merges), len(b.Merges)) {
		if a.Merges[rank] != b.Merges[rank] {
			return "", fmt.Errorf("%w: merge %d differs", ErrVocabulary, rank)
		}
	}
	if len(a.Merges) != len(b.Merges) {
		return "", fmt.Errorf("%w: %d and %d merges", ErrVocabulary, len(a.Merges), len(b.Merges))
	}
	left, err := a.Digest(commonTokens)
	if err != nil {
		return "", err
	}
	right, err := b.Digest(commonTokens)
	if err != nil {
		return "", err
	}
	if left != right {
		return "", fmt.Errorf("%w: digests differ", ErrVocabulary)
	}
	return left, nil
}

// splitMerge reads the "left right" form both tokenizer.json and GGUF use. A
// line with any other number of spaces is ambiguous and refused.
func splitMerge(line string) (MergeRule, bool) {
	left, right, found := strings.Cut(line, " ")
	if !found || left == "" || right == "" || strings.Contains(right, " ") {
		return MergeRule{}, false
	}
	return MergeRule{Left: left, Right: right}, true
}

// ReadVocabularyJSON measures the model vocabulary, added tokens and merges of
// a BPE tokenizer.json. Added tokens take their declared ids; an id that the
// model vocabulary and an added token name differently, an id named twice, or
// an id left undefined below the highest one is refused. Merges may be "a b"
// strings or [a, b] pairs. Duplicate JSON keys are refused, as in Load. The
// source is not opened here; a blocked Reader cannot be cancelled.
func ReadVocabularyJSON(ctx context.Context, source io.Reader, limits VocabularyLimits) (Vocabulary, error) {
	if ctx == nil || source == nil || !limits.valid() {
		return Vocabulary{}, fmt.Errorf("%w: context, source and positive limits required", ErrVocabulary)
	}
	body, err := readBounded(ctx, source, limits.MaxSourceBytes)
	if err != nil {
		return Vocabulary{}, err
	}
	sum := sha256.Sum256(body)
	if err := uniqueKeys(ctx, body); err != nil {
		return Vocabulary{}, err
	}
	var document struct {
		Added []struct {
			ID      *int64  `json:"id"`
			Content *string `json:"content"`
		} `json:"added_tokens"`
		Model struct {
			Type       string             `json:"type"`
			Vocabulary map[string]tokenID `json:"vocab"`
			Merges     json.RawMessage    `json:"merges"`
		} `json:"model"`
	}
	if err := json.Unmarshal(body, &document); err != nil {
		return Vocabulary{}, fmt.Errorf("%w: invalid tokenizer.json: %v", ErrVocabulary, err)
	}
	if document.Model.Type != "BPE" || len(document.Model.Vocabulary) == 0 || len(document.Model.Merges) == 0 {
		return Vocabulary{}, fmt.Errorf("%w: a BPE model with a vocabulary and merges is required", ErrVocabulary)
	}
	if len(document.Model.Vocabulary)+len(document.Added) > limits.MaxTokens {
		return Vocabulary{}, ErrBudget
	}
	names := make(map[int64]string, len(document.Model.Vocabulary)+len(document.Added))
	size := int64(0)
	name := func(id int64, token string, added bool) error {
		if id < 0 || id >= int64(limits.MaxTokens) {
			return fmt.Errorf("%w: token id %d outside the admitted range", ErrVocabulary, id)
		}
		if prior, exists := names[id]; exists && (!added || prior != token) {
			return fmt.Errorf("%w: id %d is named %q and %q", ErrVocabulary, id, prior, token)
		}
		names[id] = token
		size = max(size, id+1)
		return nil
	}
	for token, id := range document.Model.Vocabulary {
		if err := name(int64(id), token, false); err != nil {
			return Vocabulary{}, err
		}
	}
	for _, added := range document.Added {
		if added.ID == nil || added.Content == nil {
			return Vocabulary{}, fmt.Errorf("%w: added token without id or content", ErrVocabulary)
		}
		if err := name(*added.ID, *added.Content, true); err != nil {
			return Vocabulary{}, err
		}
	}
	if int64(len(names)) != size {
		return Vocabulary{}, fmt.Errorf("%w: %d ids are undefined below id %d", ErrVocabulary, size-int64(len(names)), size)
	}
	v := Vocabulary{Format: VocabularyJSON, SourceSHA256: hex.EncodeToString(sum[:]), Tokens: make([]string, size)}
	for id, token := range names {
		v.Tokens[id] = token
	}
	var lines []string
	if err := json.Unmarshal(document.Model.Merges, &lines); err == nil {
		if len(lines) > limits.MaxMerges {
			return Vocabulary{}, ErrBudget
		}
		v.Merges = make([]MergeRule, 0, len(lines))
		for rank, line := range lines {
			merge, ok := splitMerge(line)
			if !ok {
				return Vocabulary{}, fmt.Errorf("%w: merge %d is not one pair", ErrVocabulary, rank)
			}
			v.Merges = append(v.Merges, merge)
		}
	} else {
		var pairs [][]string
		if err := json.Unmarshal(document.Model.Merges, &pairs); err != nil {
			return Vocabulary{}, fmt.Errorf("%w: merges are neither lines nor pairs", ErrVocabulary)
		}
		if len(pairs) > limits.MaxMerges {
			return Vocabulary{}, ErrBudget
		}
		v.Merges = make([]MergeRule, 0, len(pairs))
		for rank, pair := range pairs {
			if len(pair) != 2 || pair[0] == "" || pair[1] == "" {
				return Vocabulary{}, fmt.Errorf("%w: merge %d is not one pair", ErrVocabulary, rank)
			}
			v.Merges = append(v.Merges, MergeRule{Left: pair[0], Right: pair[1]})
		}
	}
	return v, ctx.Err()
}

// GGUF metadata value types, as the GGUF specification numbers them.
const (
	ggufUint8 uint32 = iota
	ggufInt8
	ggufUint16
	ggufInt16
	ggufUint32
	ggufInt32
	ggufFloat32
	ggufBool
	ggufString
	ggufArray
	ggufUint64
	ggufInt64
	ggufFloat64
)

// ggufWidths gives the byte width of every fixed-size GGUF metadata type.
var ggufWidths = map[uint32]int64{ggufUint8: 1, ggufInt8: 1, ggufUint16: 2, ggufInt16: 2, ggufUint32: 4, ggufInt32: 4, ggufFloat32: 4, ggufBool: 1, ggufUint64: 8, ggufInt64: 8, ggufFloat64: 8}

// ReadVocabularyGGUF measures tokenizer.ggml.tokens and tokenizer.ggml.merges
// from a GGUF's metadata, reading only the header and the key/value section. A
// file cut anywhere after its metadata is enough. Versions 2 and 3 are
// admitted; a repeated key, a tokens or merges value that is not an array of
// strings, or a GGUF without tokens is refused. A GGUF without merges declares
// none. The source is not opened here; a blocked Reader cannot be cancelled.
func ReadVocabularyGGUF(ctx context.Context, source io.Reader, limits VocabularyLimits) (Vocabulary, error) {
	if ctx == nil || source == nil || !limits.valid() {
		return Vocabulary{}, fmt.Errorf("%w: context, source and positive limits required", ErrVocabulary)
	}
	g := &ggufReader{ctx: ctx, source: bufio.NewReaderSize(source, 1<<20), hash: sha256.New(), limits: limits}
	if magic := g.fixed(4); g.err == nil && string(magic) != "GGUF" {
		return Vocabulary{}, fmt.Errorf("%w: not a GGUF", ErrVocabulary)
	}
	version := g.uint32()
	g.uint64() // tensor count: tensors are never read
	entries := g.uint64()
	if g.err != nil {
		return Vocabulary{}, g.err
	}
	if version != 2 && version != 3 {
		return Vocabulary{}, fmt.Errorf("%w: GGUF version %d", ErrVocabulary, version)
	}
	if entries > uint64(limits.MaxMetadataEntries) {
		return Vocabulary{}, ErrBudget
	}
	v := Vocabulary{Format: VocabularyGGUF}
	seen := make(map[string]bool, entries)
	for range entries {
		key := g.text()
		kind := g.uint32()
		if g.err != nil {
			return Vocabulary{}, g.err
		}
		if seen[key] {
			return Vocabulary{}, fmt.Errorf("%w: metadata key %q repeated", ErrVocabulary, key)
		}
		seen[key] = true
		switch key {
		case "tokenizer.ggml.tokens":
			v.Tokens = g.textArray(kind, limits.MaxTokens)
		case "tokenizer.ggml.merges":
			lines := g.textArray(kind, limits.MaxMerges)
			v.Merges = make([]MergeRule, 0, len(lines))
			for rank, line := range lines {
				merge, ok := splitMerge(line)
				if !ok {
					return Vocabulary{}, fmt.Errorf("%w: merge %d is not one pair", ErrVocabulary, rank)
				}
				v.Merges = append(v.Merges, merge)
			}
		default:
			g.skip(kind, 0)
		}
		if g.err != nil {
			return Vocabulary{}, g.err
		}
	}
	if len(v.Tokens) == 0 {
		return Vocabulary{}, fmt.Errorf("%w: GGUF declares no tokenizer.ggml.tokens", ErrVocabulary)
	}
	v.SourceSHA256 = hex.EncodeToString(g.hash.Sum(nil))
	return v, ctx.Err()
}

// ggufReader hashes exactly the bytes it consumes, so read-ahead buffering
// never enters SourceSHA256, and refuses to consume beyond MaxSourceBytes.
type ggufReader struct {
	ctx      context.Context
	source   *bufio.Reader
	hash     hash.Hash
	limits   VocabularyLimits
	consumed int64
	texts    int
	err      error
}

func (g *ggufReader) fail(err error) {
	if g.err == nil {
		g.err = err
	}
}

func (g *ggufReader) take(n int64) bool {
	if g.err != nil {
		return false
	}
	if n < 0 || n > g.limits.MaxSourceBytes-g.consumed {
		g.fail(ErrBudget)
		return false
	}
	g.consumed += n
	return true
}

func (g *ggufReader) fixed(n int) []byte {
	if !g.take(int64(n)) {
		return make([]byte, n)
	}
	b := make([]byte, n)
	if _, err := io.ReadFull(g.source, b); err != nil {
		g.fail(fmt.Errorf("%w: GGUF metadata truncated: %v", ErrVocabulary, err))
		return b
	}
	g.hash.Write(b)
	return b
}

func (g *ggufReader) uint32() uint32 { return binary.LittleEndian.Uint32(g.fixed(4)) }
func (g *ggufReader) uint64() uint64 { return binary.LittleEndian.Uint64(g.fixed(8)) }

func (g *ggufReader) text() string {
	n := g.uint64()
	if g.err != nil {
		return ""
	}
	if n > uint64(g.limits.MaxStringBytes) {
		g.fail(ErrBudget)
		return ""
	}
	g.texts++
	if g.texts&1023 == 0 {
		if err := g.ctx.Err(); err != nil {
			g.fail(err)
			return ""
		}
	}
	return string(g.fixed(int(n)))
}

// textArray reads one array of strings whose length is admitted before any
// element is allocated.
func (g *ggufReader) textArray(kind uint32, maximum int) []string {
	if kind != ggufArray {
		g.fail(fmt.Errorf("%w: GGUF value type %d where an array of strings is required", ErrVocabulary, kind))
		return nil
	}
	element := g.uint32()
	count := g.uint64()
	if g.err != nil {
		return nil
	}
	if element != ggufString {
		g.fail(fmt.Errorf("%w: GGUF array of type %d where strings are required", ErrVocabulary, element))
		return nil
	}
	if count > uint64(maximum) {
		g.fail(ErrBudget)
		return nil
	}
	out := make([]string, 0, count)
	for range count {
		s := g.text()
		if g.err != nil {
			return nil
		}
		out = append(out, s)
	}
	return out
}

// skip consumes one value of any type without keeping it. Nested arrays are
// admitted to a small depth; anything deeper is refused rather than recursed.
func (g *ggufReader) skip(kind uint32, depth int) {
	if width, fixed := ggufWidths[kind]; fixed {
		g.discard(width)
		return
	}
	switch kind {
	case ggufString:
		g.text()
	case ggufArray:
		element := g.uint32()
		count := g.uint64()
		if g.err != nil {
			return
		}
		if depth >= 4 {
			g.fail(fmt.Errorf("%w: GGUF arrays nested too deep", ErrVocabulary))
			return
		}
		if width, fixed := ggufWidths[element]; fixed {
			if count > uint64(g.limits.MaxSourceBytes)/uint64(width) {
				g.fail(ErrBudget)
				return
			}
			g.discard(int64(count) * width)
			return
		}
		if count > uint64(g.limits.MaxSourceBytes) {
			g.fail(ErrBudget)
			return
		}
		for range count {
			g.skip(element, depth+1)
			if g.err != nil {
				return
			}
		}
	default:
		g.fail(fmt.Errorf("%w: unknown GGUF metadata type %d", ErrVocabulary, kind))
	}
}

func (g *ggufReader) discard(n int64) {
	if !g.take(n) {
		return
	}
	written, err := io.CopyN(g.hash, g.source, n)
	if err != nil || written != n {
		g.fail(fmt.Errorf("%w: GGUF metadata truncated: %v", ErrVocabulary, err))
		return
	}
	if err := g.ctx.Err(); err != nil {
		g.fail(err)
	}
}
