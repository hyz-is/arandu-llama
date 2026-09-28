// Package fusioncache binds teacher-forced signals to immutable causal prefixes.
// A valid contract does not attest that a backend actually executed its model.
package fusioncache

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"reflect"
	"slices"
	"strings"

	"github.com/tayi-ai/arandu-llama/training/tokenizer"
)

// ErrContract identifies a cache or projection that violates its declared contract.
var ErrContract = errors.New("fusioncache: contract rejected")

// ModelIdentity pins the artifacts and numerical implementation of a model.
type ModelIdentity struct {
	Name            string `json:"name"`
	Revision        string `json:"revision"`
	WeightsSHA256   string `json:"weights_sha256"`
	TokenizerSHA256 string `json:"tokenizer_sha256"`
	TemplateSHA256  string `json:"template_sha256"`
	RuntimeSHA256   string `json:"runtime_sha256"`
	Vocabulary      int    `json:"vocabulary"`
}

// TokenPair is a prequalified one-to-one token correspondence.
type TokenPair struct {
	Teacher int64 `json:"teacher"`
	Student int64 `json:"student"`
}

// TokenMapping permits identical tokenizers or a pinned bijection of the tokens
// used by the cache. EvidenceSHA256 identifies external token-byte/template
// equivalence qualification. Different position counts are refused.
//
// An identity mapping without Vocabulary requires one tokenizer file on both
// sides and equal vocabulary sizes. With Vocabulary it admits two tokenizer
// artifacts whose files differ but whose measured vocabularies agree, and
// every token it maps must lie below Vocabulary.CommonTokens. An absent
// Vocabulary is not encoded, so every earlier mapping keeps its digest.
type TokenMapping struct {
	TeacherTokenizerSHA256 string              `json:"teacher_tokenizer_sha256"`
	StudentTokenizerSHA256 string              `json:"student_tokenizer_sha256"`
	Identity               bool                `json:"identity"`
	Pairs                  []TokenPair         `json:"pairs,omitempty"`
	EvidenceSHA256         string              `json:"evidence_sha256"`
	Vocabulary             *VocabularyIdentity `json:"vocabulary,omitempty"`
}

// VocabularyIdentity records that the teacher's and the student's tokenizer
// artifacts were measured to declare the same token at every id below
// CommonTokens and the same merges in the same order. SHA256 is that shared
// digest under Algorithm, which is tokenizer.VocabularyAlgorithm.
//
// The common range is how two vocabularies of different lengths are compared.
// A GGUF pads its token list up to the model's embedding rows with [PAD…]
// tokens, and one tokenizer.json may declare tokens another lacks; neither
// tail is compared. So no id at or above CommonTokens is admitted anywhere: a
// gold token or a teacher top-k token there refuses the cache, whose mass is
// never dropped or renormalized.
type VocabularyIdentity struct {
	Algorithm    string `json:"algorithm"`
	CommonTokens int    `json:"common_tokens"`
	SHA256       string `json:"sha256"`
}

// VocabularyMapping measures the identity mapping between two tokenizer
// artifacts, refusing unless their vocabularies agree on ids
// [0, commonTokens) and on every merge. The mapping names each side's
// tokenizer by the SHA-256 of the bytes its measurement read, so the teacher's
// and student's ModelIdentity.TokenizerSHA256 must be those same digests for
// the mapping to be admitted. For a GGUF that is the digest of its metadata.
func VocabularyMapping(teacher, student tokenizer.Vocabulary, commonTokens int, evidenceSHA256 string) (TokenMapping, error) {
	if !validSHA(teacher.SourceSHA256) || !validSHA(student.SourceSHA256) || !validSHA(evidenceSHA256) || commonTokens < 2 {
		return TokenMapping{}, fmt.Errorf("%w: measured sources, evidence and a common range of two ids or more required", ErrContract)
	}
	digest, err := tokenizer.SameVocabulary(teacher, student, commonTokens)
	if err != nil {
		return TokenMapping{}, fmt.Errorf("%w: %w", ErrContract, err)
	}
	return TokenMapping{
		TeacherTokenizerSHA256: teacher.SourceSHA256, StudentTokenizerSHA256: student.SourceSHA256, Identity: true, EvidenceSHA256: evidenceSHA256,
		Vocabulary: &VocabularyIdentity{Algorithm: tokenizer.VocabularyAlgorithm, CommonTokens: commonTokens, SHA256: digest},
	}, nil
}

// Example contains a prompt followed by its admitted gold completion in both
// token spaces. PromptTokens is the first target index, not a logit-row index.
type Example struct {
	DatasetID     string  `json:"dataset_id"`
	DatasetSHA256 string  `json:"dataset_sha256"`
	ID            string  `json:"id"`
	Role          string  `json:"role"`
	TeacherTokens []int64 `json:"teacher_tokens"`
	StudentTokens []int64 `json:"student_tokens"`
	PromptTokens  int     `json:"prompt_tokens"`
}

// FeatureSpec identifies one tensor at the prefix's final input position.
type FeatureSpec struct {
	Layer     int    `json:"layer"`
	Tensor    string `json:"tensor"`
	DType     string `json:"dtype"`
	Dimension int    `json:"dimension"`
}

// Expectation is the independently supplied student-side admission contract.
type Expectation struct {
	Teacher       ModelIdentity `json:"teacher"`
	Student       ModelIdentity `json:"student"`
	Mapping       TokenMapping  `json:"mapping"`
	MappingSHA256 string        `json:"mapping_sha256"`
	Examples      []Example     `json:"examples"`
	Features      []FeatureSpec `json:"features,omitempty"`
}

// Limits bounds persisted JSON and admitted in-memory collection sizes.
type Limits struct {
	MaxBytes         int64
	MaxExamples      int
	MaxTokens        int
	MaxPositions     int
	MaxTopK          int
	MaxFeatureValues int
	MaxMappingPairs  int
}

// Probability preserves both token IDs and the unrenormalized retained mass.
type Probability struct {
	TeacherTokenID int64   `json:"teacher_token_id"`
	StudentTokenID int64   `json:"student_token_id"`
	Probability    float64 `json:"probability"`
}

// Feature is one finite, unpooled prefix feature vector.
type Feature struct {
	Spec   FeatureSpec `json:"spec"`
	Values []float64   `json:"values"`
}

// Position binds signals to the prefix ending immediately before TargetIndex.
type Position struct {
	TargetIndex         int           `json:"target_index"`
	TeacherPrefixSHA256 string        `json:"teacher_prefix_sha256"`
	StudentPrefixSHA256 string        `json:"student_prefix_sha256"`
	RetainedMass        float64       `json:"retained_mass"`
	Probabilities       []Probability `json:"probabilities"`
	Features            []Feature     `json:"features,omitempty"`
}

// Record contains every supervised position of one exact example, in order.
type Record struct {
	Example   Example    `json:"example"`
	Positions []Position `json:"positions"`
}

// Cache is a teacher-forced artifact, validated against an external Expectation.
type Cache struct {
	Schema        int           `json:"schema"`
	Mode          string        `json:"mode"`
	Teacher       ModelIdentity `json:"teacher"`
	Student       ModelIdentity `json:"student"`
	Mapping       TokenMapping  `json:"mapping"`
	MappingSHA256 string        `json:"mapping_sha256"`
	Features      []FeatureSpec `json:"features,omitempty"`
	Records       []Record      `json:"records"`
}

// Digest returns the SHA-256 of deterministic Go JSON, refusing nonfinite data.
func Digest(value any) (string, error) {
	encoded, err := json.Marshal(value)
	if err != nil {
		return "", fmt.Errorf("%w: encode digest: %v", ErrContract, err)
	}
	sum := sha256.Sum256(encoded)
	return hex.EncodeToString(sum[:]), nil
}

// PrefixDigest hashes token count and ordered int64 IDs without string ambiguity.
func PrefixDigest(tokens []int64) string {
	h := sha256.New()
	h.Write([]byte("fusioncache-prefix-v1\x00"))
	var word [8]byte
	binary.LittleEndian.PutUint64(word[:], uint64(len(tokens)))
	h.Write(word[:])
	for _, token := range tokens {
		binary.LittleEndian.PutUint64(word[:], uint64(token))
		h.Write(word[:])
	}
	return hex.EncodeToString(h.Sum(nil))
}

func validSHA(s string) bool {
	if len(s) != 64 || s != strings.ToLower(s) {
		return false
	}
	_, err := hex.DecodeString(s)
	return err == nil
}

func validModel(m ModelIdentity) bool {
	if strings.TrimSpace(m.Name) == "" || (len(m.Revision) != 40 && len(m.Revision) != 64) || m.Revision != strings.ToLower(m.Revision) || m.Vocabulary < 2 {
		return false
	}
	if _, err := hex.DecodeString(m.Revision); err != nil {
		return false
	}
	return validSHA(m.WeightsSHA256) && validSHA(m.TokenizerSHA256) && validSHA(m.TemplateSHA256) && validSHA(m.RuntimeSHA256)
}

func finite(v float64) bool        { return !math.IsNaN(v) && !math.IsInf(v, 0) }
func allowedRole(role string) bool { return role == "train" || role == "calibration" }
func exampleKey(e Example) string  { return e.DatasetSHA256 + "/" + e.ID }

func validateLimits(l Limits) error {
	if l.MaxBytes <= 0 || l.MaxBytes == math.MaxInt64 || l.MaxExamples <= 0 || l.MaxTokens < 2 || l.MaxPositions <= 0 || l.MaxTopK <= 0 || l.MaxFeatureValues < 0 || l.MaxMappingPairs < 0 {
		return fmt.Errorf("%w: explicit positive cache bounds required", ErrContract)
	}
	return nil
}

// tokenSpace is an admitted mapping: the identity below limit, or a table.
type tokenSpace struct {
	identity bool
	limit    int64
	table    map[int64]int64
}

func mappingTable(e Expectation, l Limits) (tokenSpace, error) {
	m := e.Mapping
	if len(m.Pairs) > l.MaxMappingPairs || (m.Identity && len(m.Pairs) != 0) {
		return tokenSpace{}, fmt.Errorf("%w: mapping size exceeds bound", ErrContract)
	}
	digest, err := Digest(m)
	if err != nil || !validSHA(e.MappingSHA256) || digest != e.MappingSHA256 || !validSHA(m.EvidenceSHA256) || m.TeacherTokenizerSHA256 != e.Teacher.TokenizerSHA256 || m.StudentTokenizerSHA256 != e.Student.TokenizerSHA256 {
		return tokenSpace{}, fmt.Errorf("%w: mapping identity differs", ErrContract)
	}
	if v := m.Vocabulary; v != nil {
		if !m.Identity || v.Algorithm != tokenizer.VocabularyAlgorithm || !validSHA(v.SHA256) || v.CommonTokens < 2 || v.CommonTokens > e.Teacher.Vocabulary || v.CommonTokens > e.Student.Vocabulary {
			return tokenSpace{}, fmt.Errorf("%w: measured vocabulary identity differs", ErrContract)
		}
		return tokenSpace{identity: true, limit: int64(v.CommonTokens)}, nil
	}
	if m.Identity {
		if m.TeacherTokenizerSHA256 != m.StudentTokenizerSHA256 || e.Teacher.Vocabulary != e.Student.Vocabulary || len(m.Pairs) != 0 {
			return tokenSpace{}, fmt.Errorf("%w: identity mapping differs", ErrContract)
		}
		return tokenSpace{identity: true, limit: int64(e.Teacher.Vocabulary)}, nil
	}
	if len(m.Pairs) == 0 || len(m.Pairs) > l.MaxMappingPairs {
		return tokenSpace{}, fmt.Errorf("%w: mapping size exceeds bound", ErrContract)
	}
	table := make(map[int64]int64, len(m.Pairs))
	students := make(map[int64]bool, len(m.Pairs))
	for index, pair := range m.Pairs {
		if pair.Teacher < 0 || pair.Teacher >= int64(e.Teacher.Vocabulary) || pair.Student < 0 || pair.Student >= int64(e.Student.Vocabulary) || students[pair.Student] || (index > 0 && pair.Teacher <= m.Pairs[index-1].Teacher) {
			return tokenSpace{}, fmt.Errorf("%w: mapping must be ordered and one-to-one", ErrContract)
		}
		table[pair.Teacher], students[pair.Student] = pair.Student, true
	}
	return tokenSpace{table: table}, nil
}

// mappedToken refuses an identity token outside the admitted range. Without a
// measured vocabulary that range is the teacher's whole vocabulary, which the
// callers already bound, so plain identity mappings behave as before.
func mappedToken(space tokenSpace, token int64) (int64, bool) {
	if space.identity {
		return token, token >= 0 && token < space.limit
	}
	v, ok := space.table[token]
	return v, ok
}

func sameMapping(a, b TokenMapping) bool {
	return a.Identity == b.Identity && a.TeacherTokenizerSHA256 == b.TeacherTokenizerSHA256 &&
		a.StudentTokenizerSHA256 == b.StudentTokenizerSHA256 && a.EvidenceSHA256 == b.EvidenceSHA256 && slices.Equal(a.Pairs, b.Pairs) &&
		(a.Vocabulary == nil) == (b.Vocabulary == nil) && (a.Vocabulary == nil || *a.Vocabulary == *b.Vocabulary)
}

// ValidateMapping checks two model identities and the mapping between them
// without any example, so a caller that names a teacher can be refused before
// its cache is read. ValidateExpectation applies the same checks.
func ValidateMapping(teacher, student ModelIdentity, mapping TokenMapping, mappingSHA256 string, l Limits) error {
	if err := validateLimits(l); err != nil {
		return err
	}
	if !validModel(teacher) || !validModel(student) {
		return fmt.Errorf("%w: invalid models", ErrContract)
	}
	_, err := mappingTable(Expectation{Teacher: teacher, Student: student, Mapping: mapping, MappingSHA256: mappingSHA256}, l)
	return err
}

// ValidateExpectation refuses ambiguous alignment and nontraining data before a
// teacher is opened. Matching hashes attest identity, not semantic equivalence.
func ValidateExpectation(e Expectation, l Limits) error {
	if err := validateLimits(l); err != nil {
		return err
	}
	if !validModel(e.Teacher) || !validModel(e.Student) || len(e.Examples) == 0 || len(e.Examples) > l.MaxExamples {
		return fmt.Errorf("%w: invalid models or examples", ErrContract)
	}
	table, err := mappingTable(e, l)
	if err != nil {
		return err
	}
	featureValues := 0
	seenFeatures := map[string]bool{}
	for _, f := range e.Features {
		key := fmt.Sprintf("%d/%s", f.Layer, f.Tensor)
		if f.Layer < 0 || f.Tensor == "" || (f.DType != "float32" && f.DType != "float64") || f.Dimension <= 0 || f.Dimension > l.MaxFeatureValues-featureValues || seenFeatures[key] {
			return fmt.Errorf("%w: invalid feature specification", ErrContract)
		}
		featureValues += f.Dimension
		seenFeatures[key] = true
	}
	seen := map[string]bool{}
	positions := 0
	for _, x := range e.Examples {
		count := len(x.StudentTokens)
		if x.DatasetID == "" || !validSHA(x.DatasetSHA256) || x.ID == "" || !allowedRole(x.Role) || seen[exampleKey(x)] || count > l.MaxTokens || len(x.TeacherTokens) != count || x.PromptTokens < 1 || x.PromptTokens >= count {
			return fmt.Errorf("%w: invalid example identity, role or geometry", ErrContract)
		}
		if count-x.PromptTokens > l.MaxPositions-positions {
			return fmt.Errorf("%w: position limit exceeded", ErrContract)
		}
		positions += count - x.PromptTokens
		seen[exampleKey(x)] = true
		for i, teacher := range x.TeacherTokens {
			student, ok := mappedToken(table, teacher)
			if teacher < 0 || teacher >= int64(e.Teacher.Vocabulary) || !ok || student != x.StudentTokens[i] || student < 0 || student >= int64(e.Student.Vocabulary) {
				return fmt.Errorf("%w: gold prefix token mapping differs", ErrContract)
			}
		}
	}
	if featureValues != 0 && positions > l.MaxFeatureValues/featureValues {
		return fmt.Errorf("%w: total feature values exceed bound", ErrContract)
	}
	return nil
}

// ValidateAgainstStudent checks every causal row against independently admitted
// student tokens. A valid free-generation cache cannot pass with divergent gold.
func ValidateAgainstStudent(c Cache, e Expectation, l Limits) error {
	if err := ValidateExpectation(e, l); err != nil {
		return err
	}
	if c.Schema != 1 || c.Mode != "teacher_forced" || c.Teacher != e.Teacher || c.Student != e.Student || c.MappingSHA256 != e.MappingSHA256 || !sameMapping(c.Mapping, e.Mapping) || !slices.Equal(c.Features, e.Features) || len(c.Records) != len(e.Examples) {
		return fmt.Errorf("%w: cache identity differs from student expectation", ErrContract)
	}
	table, err := mappingTable(e, l)
	if err != nil {
		return err
	}
	for index, record := range c.Records {
		x := e.Examples[index]
		if !reflect.DeepEqual(record.Example, x) || len(record.Positions) != len(x.StudentTokens)-x.PromptTokens {
			return fmt.Errorf("%w: cached example or order differs", ErrContract)
		}
		for row, p := range record.Positions {
			position := x.PromptTokens + row
			if p.TargetIndex != position || p.TeacherPrefixSHA256 != PrefixDigest(x.TeacherTokens[:position]) || p.StudentPrefixSHA256 != PrefixDigest(x.StudentTokens[:position]) {
				return fmt.Errorf("%w: causal prefix or position differs", ErrContract)
			}
			if !finite(p.RetainedMass) || p.RetainedMass < 0 || p.RetainedMass > 1 || len(p.Probabilities) == 0 || len(p.Probabilities) > l.MaxTopK || len(p.Features) != len(e.Features) {
				return fmt.Errorf("%w: invalid distribution or features", ErrContract)
			}
			sum := 0.0
			for j, probability := range p.Probabilities {
				student, ok := mappedToken(table, probability.TeacherTokenID)
				if !ok || student != probability.StudentTokenID || probability.TeacherTokenID < 0 || probability.TeacherTokenID >= int64(e.Teacher.Vocabulary) || student < 0 || student >= int64(e.Student.Vocabulary) || !finite(probability.Probability) || probability.Probability < 0 || probability.Probability > 1 || (j > 0 && probability.TeacherTokenID <= p.Probabilities[j-1].TeacherTokenID) {
					return fmt.Errorf("%w: mapped probability invalid or unordered", ErrContract)
				}
				sum += probability.Probability
			}
			if math.Abs(sum-p.RetainedMass) > 1e-10 {
				return fmt.Errorf("%w: retained mass differs from probability sum", ErrContract)
			}
			for j, feature := range p.Features {
				if feature.Spec != e.Features[j] || len(feature.Values) != feature.Spec.Dimension {
					return fmt.Errorf("%w: feature tensor identity differs", ErrContract)
				}
				for _, value := range feature.Values {
					if !finite(value) {
						return fmt.Errorf("%w: nonfinite feature", ErrContract)
					}
				}
			}
		}
	}
	return nil
}
