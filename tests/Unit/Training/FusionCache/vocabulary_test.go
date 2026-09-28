package fusioncache_test

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"github.com/tayi-ai/arandu-llama/training/fusioncache"
	"github.com/tayi-ai/arandu-llama/training/tokenizer"
)

// The mapping the fixture in cache_test.go declares, pinned by its digest.
// Its JSON has no vocabulary key, so adding the measured mode left it as it was.
const identityMappingSHA256 = "329d5b052bed1b785469f02b63d0097df7eed2740ee82b54b6ab2f8b16a558e7"

func TestAnIdentityMappingWithoutVocabularyKeepsItsDigest(t *testing.T) {
	e := expectation(t)
	encoded, err := fusioncache.Digest(e.Mapping)
	if err != nil {
		t.Fatal(err)
	}
	if encoded != identityMappingSHA256 {
		t.Fatalf("an identity mapping changed its digest: %s", encoded)
	}
}

// measured returns a six-token student vocabulary from a tokenizer.json and a
// teacher vocabulary from a GGUF that pads the same list to eight ids.
func measured() (teacher, student tokenizer.Vocabulary) {
	tokens := []string{"a", "b", "ab", "c", "abc", "<|end|>"}
	merges := []tokenizer.MergeRule{{Left: "a", Right: "b"}, {Left: "ab", Right: "c"}}
	student = tokenizer.Vocabulary{Format: tokenizer.VocabularyJSON, SourceSHA256: strings.Repeat("5", 64), Tokens: tokens, Merges: merges}
	teacher = tokenizer.Vocabulary{Format: tokenizer.VocabularyGGUF, SourceSHA256: strings.Repeat("7", 64), Tokens: append(append([]string(nil), tokens...), "[PAD6]", "[PAD7]"), Merges: merges}
	return teacher, student
}

// vocabularyFixture is the cache fixture with a teacher whose tokenizer file
// differs from the student's and whose vocabulary was measured to agree on
// the six ids the student's tokenizer declares. Both models have output rows
// past those ids, the student seven and the teacher eight, as Ornith's
// 248320 rows lie past the 248070 ids of its tokenizer.json, so only the
// common range bounds a token there.
func vocabularyFixture(t *testing.T) (fusioncache.Cache, fusioncache.Expectation) {
	t.Helper()
	teacher, student := measured()
	mapping, err := fusioncache.VocabularyMapping(teacher, student, 6, strings.Repeat("f", 64))
	if err != nil {
		t.Fatal(err)
	}
	c, e := fixture(t)
	e.Teacher.TokenizerSHA256, e.Student.TokenizerSHA256 = teacher.SourceSHA256, student.SourceSHA256
	e.Student.Vocabulary = 7
	e.Mapping = mapping
	e.MappingSHA256 = digest(t, mapping)
	c.Teacher, c.Student, c.Mapping, c.MappingSHA256 = e.Teacher, e.Student, e.Mapping, e.MappingSHA256
	if err := fusioncache.ValidateAgainstStudent(c, e, limits()); err != nil {
		t.Fatal(err)
	}
	return c, e
}

func TestAMeasuredVocabularyAdmitsDifferentTokenizerFiles(t *testing.T) {
	c, e := vocabularyFixture(t)
	if e.Teacher.TokenizerSHA256 == e.Student.TokenizerSHA256 || e.Teacher.Vocabulary == e.Student.Vocabulary {
		t.Fatal("fixture no longer exercises different files and sizes")
	}
	path := filepath.Join(t.TempDir(), "cache.json")
	receipt, err := fusioncache.WriteAtomic(path, c, e, limits())
	if err != nil {
		t.Fatal(err)
	}
	if _, err := fusioncache.Read(path, receipt.SHA256, e, limits()); err != nil {
		t.Fatal(err)
	}
	// Without the measurement the same pair is what it always was: refused.
	plain := clone(t, e)
	plain.Mapping.Vocabulary = nil
	plain.MappingSHA256 = digest(t, plain.Mapping)
	if err := fusioncache.ValidateExpectation(plain, limits()); !errors.Is(err, fusioncache.ErrContract) {
		t.Fatal("different tokenizer files were admitted without a measured vocabulary", err)
	}
	if err := fusioncache.ValidateMapping(e.Teacher, e.Student, e.Mapping, e.MappingSHA256, limits()); err != nil {
		t.Fatal(err)
	}
}

func TestAMeasuredVocabularyRefusesMassOutsideItsCommonRange(t *testing.T) {
	c, e := vocabularyFixture(t)
	// Id 6 is a [PAD…] token of the teacher, inside both models' output rows
	// and outside the range the two tokenizers were measured on.
	outside := clone(t, c)
	p := &outside.Records[0].Positions[0]
	p.Probabilities = append(p.Probabilities, fusioncache.Probability{TeacherTokenID: 6, StudentTokenID: 6, Probability: 0.125})
	p.RetainedMass += 0.125
	if err := fusioncache.ValidateAgainstStudent(outside, e, limits()); !errors.Is(err, fusioncache.ErrContract) {
		t.Fatal("teacher mass outside the common range was admitted", err)
	}
	gold := clone(t, e)
	gold.Examples[0].TeacherTokens[3], gold.Examples[0].StudentTokens[3] = 6, 6
	if err := fusioncache.ValidateExpectation(gold, limits()); !errors.Is(err, fusioncache.ErrContract) {
		t.Fatal("a gold token outside the common range was admitted", err)
	}
	factory := &fakeFactory{}
	sink := fakeSink{factory: factory, sink: fusioncache.DirectorySink{Directory: t.TempDir()}}
	if _, err := fusioncache.ProduceSequential(context.Background(), []fusioncache.Expectation{e}, outsideFactory{factory}, sink, limits()); !errors.Is(err, fusioncache.ErrContract) {
		t.Fatal("a teacher that put mass outside the common range produced a cache", err)
	}
}

// outsideFactory's teacher answers like fakeFactory's and adds mass on id 6.
type outsideFactory struct{ *fakeFactory }
type outsideProducer struct{ fusioncache.Producer }

func (f outsideFactory) Open(ctx context.Context, identity fusioncache.ModelIdentity) (fusioncache.Producer, error) {
	p, err := f.fakeFactory.Open(ctx, identity)
	return outsideProducer{p}, err
}
func (p outsideProducer) TeacherForce(ctx context.Context, request fusioncache.PrefixRequest) (fusioncache.Signal, error) {
	signal, err := p.Producer.TeacherForce(ctx, request)
	signal.Probabilities = append(signal.Probabilities, fusioncache.TeacherProbability{TokenID: 6, Probability: 0.125})
	signal.RetainedMass += 0.125
	return signal, err
}

func TestAMeasuredVocabularyIsRefusedWhereItCannotHold(t *testing.T) {
	teacher, student := measured()
	evidence := strings.Repeat("f", 64)
	different := teacher
	different.Tokens = append([]string(nil), teacher.Tokens...)
	different.Tokens[4] = "abd"
	if _, err := fusioncache.VocabularyMapping(different, student, 6, evidence); !errors.Is(err, fusioncache.ErrContract) || !errors.Is(err, tokenizer.ErrVocabulary) {
		t.Fatal("a different vocabulary was mapped as identical", err)
	}
	if _, err := fusioncache.VocabularyMapping(teacher, student, 7, evidence); err == nil {
		t.Fatal("a common range past the student vocabulary was mapped")
	}
	unmeasured := teacher
	unmeasured.SourceSHA256 = ""
	if _, err := fusioncache.VocabularyMapping(unmeasured, student, 6, evidence); err == nil {
		t.Fatal("a vocabulary without a measured source was mapped")
	}
	_, e := vocabularyFixture(t)
	for name, change := range map[string]func(*fusioncache.Expectation){
		"bijection":      func(e *fusioncache.Expectation) { e.Mapping.Identity = false },
		"algorithm":      func(e *fusioncache.Expectation) { e.Mapping.Vocabulary.Algorithm = "tayi-vocabulary-v0" },
		"digest":         func(e *fusioncache.Expectation) { e.Mapping.Vocabulary.SHA256 = "not-a-digest" },
		"student range":  func(e *fusioncache.Expectation) { e.Mapping.Vocabulary.CommonTokens = 8 },
		"teacher range":  func(e *fusioncache.Expectation) { e.Teacher.Vocabulary = 5 },
		"single token":   func(e *fusioncache.Expectation) { e.Mapping.Vocabulary.CommonTokens = 1 },
		"tokenizer file": func(e *fusioncache.Expectation) { e.Teacher.TokenizerSHA256 = strings.Repeat("6", 64) },
	} {
		t.Run(name, func(t *testing.T) {
			bad := clone(t, e)
			change(&bad)
			bad.MappingSHA256 = digest(t, bad.Mapping)
			if err := fusioncache.ValidateExpectation(bad, limits()); !errors.Is(err, fusioncache.ErrContract) {
				t.Fatal("an invalid measured vocabulary was admitted", err)
			}
		})
	}
	// A cache made under another measurement is another cache.
	c, e := vocabularyFixture(t)
	other := clone(t, c)
	other.Mapping.Vocabulary.SHA256 = strings.Repeat("9", 64)
	if err := fusioncache.ValidateAgainstStudent(other, e, limits()); !errors.Is(err, fusioncache.ErrContract) {
		t.Fatal("a cache under a different measured vocabulary was admitted", err)
	}
}

// TestMeasuredVocabulariesOfRealArtifacts reads real tokenizer artifacts named
// by the environment: TRAINING_VOCABULARY_STUDENT, a tokenizer.json or GGUF;
// TRAINING_VOCABULARY_SAME and TRAINING_VOCABULARY_DIFFERENT, lists of such
// paths separated by the OS path-list separator; TRAINING_VOCABULARY_COMMON,
// the common range. Only a GGUF's metadata is read.
func TestMeasuredVocabulariesOfRealArtifacts(t *testing.T) {
	studentPath := os.Getenv("TRAINING_VOCABULARY_STUDENT")
	if studentPath == "" {
		t.Skip("set TRAINING_VOCABULARY_STUDENT, TRAINING_VOCABULARY_SAME, TRAINING_VOCABULARY_DIFFERENT and TRAINING_VOCABULARY_COMMON")
	}
	common, err := strconv.Atoi(os.Getenv("TRAINING_VOCABULARY_COMMON"))
	if err != nil {
		t.Fatal(err)
	}
	evidence := strings.Repeat("e", 64)
	student := readArtifact(t, studentPath)
	t.Logf("student %s: format=%s source_sha256=%s tokens=%d merges=%d", filepath.Base(studentPath), student.Format, student.SourceSHA256, len(student.Tokens), len(student.Merges))
	studentModel := realModel("student", student, common)
	same := filepath.SplitList(os.Getenv("TRAINING_VOCABULARY_SAME"))
	different := filepath.SplitList(os.Getenv("TRAINING_VOCABULARY_DIFFERENT"))
	if len(same) == 0 || len(different) == 0 {
		t.Fatal("both an admitted and a refused teacher are required")
	}
	for _, path := range same {
		teacher := readArtifact(t, path)
		mapping, err := fusioncache.VocabularyMapping(teacher, student, common, evidence)
		if err != nil {
			t.Fatalf("%s: %v", filepath.Base(path), err)
		}
		mappingSHA256, err := fusioncache.Digest(mapping)
		if err != nil {
			t.Fatal(err)
		}
		limits := limits()
		if err := fusioncache.ValidateMapping(realModel(filepath.Base(path), teacher, common), studentModel, mapping, mappingSHA256, limits); err != nil {
			t.Fatalf("%s: measured mapping refused: %v", filepath.Base(path), err)
		}
		t.Logf("admitted %s: format=%s source_sha256=%s tokens=%d merges=%d common=%d vocabulary_sha256=%s", filepath.Base(path), teacher.Format, teacher.SourceSHA256, len(teacher.Tokens), len(teacher.Merges), common, mapping.Vocabulary.SHA256)
	}
	for _, path := range different {
		teacher := readArtifact(t, path)
		_, declared := fusioncache.VocabularyMapping(teacher, student, common, evidence)
		if declared == nil {
			t.Fatalf("%s: a different vocabulary was admitted at the declared range", filepath.Base(path))
		}
		// Refused at the declared range could mean only that one list is
		// shorter; the shorter range shows the tokens themselves differ.
		shorter := min(common, len(teacher.Tokens))
		_, compared := fusioncache.VocabularyMapping(teacher, student, shorter, evidence)
		if !errors.Is(compared, tokenizer.ErrVocabulary) {
			t.Fatalf("%s: a different vocabulary was admitted over %d ids", filepath.Base(path), shorter)
		}
		t.Logf("refused %s: format=%s source_sha256=%s tokens=%d merges=%d; at %d ids: %v; at %d ids: %v", filepath.Base(path), teacher.Format, teacher.SourceSHA256, len(teacher.Tokens), len(teacher.Merges), common, declared, shorter, compared)
	}
}

func readArtifact(t *testing.T, path string) tokenizer.Vocabulary {
	t.Helper()
	file, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	read := tokenizer.ReadVocabularyJSON
	if strings.HasSuffix(path, ".gguf") {
		read = tokenizer.ReadVocabularyGGUF
	}
	v, err := read(context.Background(), file, tokenizer.DefaultVocabularyLimits())
	if err != nil {
		t.Fatalf("%s: %v", filepath.Base(path), err)
	}
	return v
}

// realModel names a measured artifact; its vocabulary is at least the common
// range, as a model's embedding rows are at least its tokenizer's ids.
func realModel(name string, v tokenizer.Vocabulary, common int) fusioncache.ModelIdentity {
	m := model(name)
	m.TokenizerSHA256, m.Vocabulary = v.SourceSHA256, max(len(v.Tokens), common)
	return m
}
