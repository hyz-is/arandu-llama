package local

import (
	"context"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/tayi-ai/arandu-llama/training/fusioncache"
	"github.com/tayi-ai/arandu-llama/training/tokenizer"
)

// kdStudent is the student every distillation fixture names.
func kdStudent(vocabulary int) fusioncache.ModelIdentity {
	pin := strings.Repeat("a", 64)
	return fusioncache.ModelIdentity{Name: "student", Revision: strings.Repeat("a", 40), WeightsSHA256: fmtHash([]byte("student")), TokenizerSHA256: pin, TemplateSHA256: pin, RuntimeSHA256: pin, Vocabulary: vocabulary}
}

// kdDistillation declares two teachers of the student: teacher-a on the
// student's own tokenizer file, and teacher-b on another file whose
// vocabulary was measured to agree on every id the student has, padded as a
// GGUF pads it. Caches are pinned later, by kdWrite.
func kdDistillation(t *testing.T, student fusioncache.ModelIdentity, datasetSHA256 string, weights ...float64) *Distillation {
	t.Helper()
	if len(weights) != 2 {
		t.Fatal("two teacher weights required")
	}
	pin := strings.Repeat("a", 64)
	limits := fusioncache.Limits{MaxBytes: 1 << 20, MaxExamples: 16, MaxTokens: 8, MaxPositions: 64, MaxTopK: 8}
	d := &Distillation{Method: DistillationMethod, DatasetID: "fixture-training", DatasetSHA256: datasetSHA256, Student: student, CacheLimits: limits}
	own := fusioncache.ModelIdentity{Name: "teacher-a", Revision: strings.Repeat("b", 40), WeightsSHA256: fmtHash([]byte("teacher-a")), TokenizerSHA256: student.TokenizerSHA256, TemplateSHA256: pin, RuntimeSHA256: pin, Vocabulary: student.Vocabulary}
	ownMapping := fusioncache.TokenMapping{TeacherTokenizerSHA256: own.TokenizerSHA256, StudentTokenizerSHA256: student.TokenizerSHA256, Identity: true, EvidenceSHA256: fmtHash([]byte("evidence"))}
	// The measured teacher: the same token list, two [PAD…] ids past it.
	studentTokens := make([]string, student.Vocabulary)
	for i := range studentTokens {
		studentTokens[i] = fmt.Sprintf("t%d", i)
	}
	merges := []tokenizer.MergeRule{{Left: "t0", Right: "t1"}}
	measuredStudent := tokenizer.Vocabulary{Format: tokenizer.VocabularyJSON, SourceSHA256: student.TokenizerSHA256, Tokens: studentTokens, Merges: merges}
	measuredTeacher := tokenizer.Vocabulary{Format: tokenizer.VocabularyGGUF, SourceSHA256: fmtHash([]byte("teacher-b metadata")), Tokens: append(append([]string(nil), studentTokens...), "[PAD0]", "[PAD1]"), Merges: merges}
	measuredMapping, err := fusioncache.VocabularyMapping(measuredTeacher, measuredStudent, student.Vocabulary, fmtHash([]byte("evidence")))
	if err != nil {
		t.Fatal(err)
	}
	measured := fusioncache.ModelIdentity{Name: "teacher-b", Revision: strings.Repeat("c", 40), WeightsSHA256: fmtHash([]byte("teacher-b")), TokenizerSHA256: measuredTeacher.SourceSHA256, TemplateSHA256: pin, RuntimeSHA256: pin, Vocabulary: len(measuredTeacher.Tokens)}
	for i, pair := range []struct {
		model   fusioncache.ModelIdentity
		mapping fusioncache.TokenMapping
	}{{own, ownMapping}, {measured, measuredMapping}} {
		sha, err := fusioncache.Digest(pair.mapping)
		if err != nil {
			t.Fatal(err)
		}
		d.Teachers = append(d.Teachers, DistillationTeacher{Model: pair.model, Mapping: pair.mapping, MappingSHA256: sha, Weight: weights[i], CacheSHA256: strings.Repeat("0", 64), CacheBytes: 1})
	}
	return d
}

// kdExamples are the examples a cache over rows must hold, built here from
// the contract rather than by the code under test.
func kdExamples(d *Distillation, rows []example) []fusioncache.Example {
	out := make([]fusioncache.Example, len(rows))
	for i, row := range rows {
		tokens := append([]int64(nil), row.InputIDs...)
		out[i] = fusioncache.Example{DatasetID: d.DatasetID, DatasetSHA256: d.DatasetSHA256, ID: row.ID, Role: "train", TeacherTokens: tokens, StudentTokens: append([]int64(nil), tokens...), PromptTokens: row.PromptTokens}
	}
	return out
}

// kdTopK is each teacher's retained top-k at every position, ascending by id.
var kdTopK = [][]fusioncache.Probability{
	{{TeacherTokenID: 1, StudentTokenID: 1, Probability: 0.5}, {TeacherTokenID: 4, StudentTokenID: 4, Probability: 0.25}},
	{{TeacherTokenID: 0, StudentTokenID: 0, Probability: 0.125}, {TeacherTokenID: 2, StudentTokenID: 2, Probability: 0.25}, {TeacherTokenID: 5, StudentTokenID: 5, Probability: 0.5}},
}

// kdCache is teacher's cache over examples, with kdTopK at every position.
func kdCache(d *Distillation, teacher int, examples []fusioncache.Example) (fusioncache.Cache, fusioncache.Expectation) {
	x := d.Teachers[teacher]
	e := fusioncache.Expectation{Teacher: x.Model, Student: d.Student, Mapping: x.Mapping, MappingSHA256: x.MappingSHA256, Examples: examples}
	c := fusioncache.Cache{Schema: 1, Mode: "teacher_forced", Teacher: x.Model, Student: d.Student, Mapping: x.Mapping, MappingSHA256: x.MappingSHA256}
	for _, example := range examples {
		r := fusioncache.Record{Example: example}
		for position := example.PromptTokens; position < len(example.TeacherTokens); position++ {
			p := fusioncache.Position{TargetIndex: position, TeacherPrefixSHA256: fusioncache.PrefixDigest(example.TeacherTokens[:position]), StudentPrefixSHA256: fusioncache.PrefixDigest(example.StudentTokens[:position]), Probabilities: append([]fusioncache.Probability(nil), kdTopK[teacher]...)}
			for _, q := range p.Probabilities {
				p.RetainedMass += q.Probability
			}
			r.Positions = append(r.Positions, p)
		}
		c.Records = append(c.Records, r)
	}
	return c, e
}

// kdWrite stores teacher's cache over examples in dir, under its digest, and
// returns the digest and size the recipe must pin.
func kdWrite(t *testing.T, dir string, d *Distillation, teacher int, examples []fusioncache.Example) (string, int64) {
	t.Helper()
	c, e := kdCache(d, teacher, examples)
	receipt, err := fusioncache.DirectorySink{Directory: dir}.Store(context.Background(), c, e, d.CacheLimits)
	if err != nil {
		t.Fatal(err)
	}
	return receipt.SHA256, receipt.Bytes
}

// kdPin writes every teacher's cache over rows and pins it in d.
func kdPin(t *testing.T, dir string, d *Distillation, rows []example) {
	t.Helper()
	for i := range d.Teachers {
		d.Teachers[i].CacheSHA256, d.Teachers[i].CacheBytes = kdWrite(t, dir, d, i, kdExamples(d, rows))
	}
}

var kdRows = []example{{"first", []int64{1, 2}, []int64{-100, 2}, 1}, {"second", []int64{1, 2, 3}, []int64{-100, 2, 3}, 1}, {"third", []int64{4, 1, 2}, []int64{-100, -100, 2}, 2}}

func kdConfig(t *testing.T, weights ...float64) (Config, *Distillation) {
	t.Helper()
	d := kdDistillation(t, kdStudent(6), fmtHash([]byte("training")), weights...)
	dir := t.TempDir()
	kdPin(t, dir, d, kdRows)
	return Config{Recipe: Recipe{ExampleCount: len(kdRows), BaseRevision: d.Student.Revision, Distillation: d}, CacheDir: dir}, d
}

func TestDistillationBlockIsRefusedBeforeAnyCacheIsRead(t *testing.T) {
	_, d := kdConfig(t, 0.25, 0.5)
	if err := d.validate(d.Student.Revision); err != nil {
		t.Fatal(err)
	}
	for name, change := range map[string]func(*Distillation){
		"method":           func(d *Distillation) { d.Method = "logit-kd-v0" },
		"dataset id":       func(d *Distillation) { d.DatasetID = "" },
		"dataset digest":   func(d *Distillation) { d.DatasetSHA256 = "training" },
		"student revision": func(d *Distillation) { d.Student.Revision = strings.Repeat("f", 40) },
		"no teacher":       func(d *Distillation) { d.Teachers = nil },
		"nine teachers": func(d *Distillation) {
			d.Teachers = append(d.Teachers, d.Teachers[0], d.Teachers[0], d.Teachers[0], d.Teachers[0], d.Teachers[0], d.Teachers[0], d.Teachers[0])
		},
		"repeated teacher":  func(d *Distillation) { d.Teachers[1] = d.Teachers[0] },
		"negative weight":   func(d *Distillation) { d.Teachers[0].Weight = -0.25 },
		"NaN weight":        func(d *Distillation) { d.Teachers[0].Weight = math.NaN() },
		"infinite weight":   func(d *Distillation) { d.Teachers[0].Weight = math.Inf(1) },
		"alpha above one":   func(d *Distillation) { d.Teachers[0].Weight = 0.625 },
		"cache digest":      func(d *Distillation) { d.Teachers[0].CacheSHA256 = "cache" },
		"empty cache":       func(d *Distillation) { d.Teachers[0].CacheBytes = 0 },
		"cache above bound": func(d *Distillation) { d.Teachers[0].CacheBytes = d.CacheLimits.MaxBytes + 1 },
		"cache bounds":      func(d *Distillation) { d.CacheLimits.MaxTopK = 0 },
		"mapping digest":    func(d *Distillation) { d.Teachers[1].MappingSHA256 = d.Teachers[0].MappingSHA256 },
		"bijection": func(d *Distillation) {
			d.Teachers[0].Mapping.Identity = false
			d.Teachers[0].Mapping.Pairs = []fusioncache.TokenPair{{Teacher: 0, Student: 1}, {Teacher: 1, Student: 0}}
			d.Teachers[0].MappingSHA256, _ = fusioncache.Digest(d.Teachers[0].Mapping)
		},
		"range past the student": func(d *Distillation) {
			d.Teachers[1].Mapping.Vocabulary.CommonTokens = 7
			d.Teachers[1].MappingSHA256, _ = fusioncache.Digest(d.Teachers[1].Mapping)
		},
		"unmeasured files": func(d *Distillation) {
			d.Teachers[1].Mapping.Vocabulary = nil
			d.Teachers[1].MappingSHA256, _ = fusioncache.Digest(d.Teachers[1].Mapping)
		},
	} {
		t.Run(name, func(t *testing.T) {
			_, bad := kdConfig(t, 0.25, 0.5)
			change(bad)
			if err := bad.validate(kdStudent(6).Revision); !errors.Is(err, ErrDistillation) {
				t.Fatal("an invalid distillation block was admitted", err)
			}
		})
	}
	// Alpha one is admitted, as FusionCompletionGradient admits it.
	_, full := kdConfig(t, 0.5, 0.5)
	if err := full.validate(full.Student.Revision); err != nil {
		t.Fatal(err)
	}
}

func TestOnlyADistillationRecipeNamesACacheDirectory(t *testing.T) {
	c := checkpointFixture(t)
	if _, err := c.Snapshot(); err != nil {
		t.Fatal(err)
	}
	stray := c
	stray.CacheDir = t.TempDir()
	if _, err := stray.Snapshot(); !errors.Is(err, ErrDistillation) {
		t.Fatal("a cache directory was admitted without distillation", err)
	}
	_, d := kdConfig(t, 0.25, 0.5)
	distilled := c
	distilled.Recipe.BaseRevision, distilled.Recipe.Distillation = d.Student.Revision, d
	for name, dir := range map[string]string{"absent": "", "relative": "caches", "unclean": "/caches/../caches"} {
		distilled.CacheDir = dir
		if _, err := distilled.Snapshot(); !errors.Is(err, ErrDistillation) {
			t.Fatal(name, "cache directory was admitted", err)
		}
	}
	distilled.CacheDir = t.TempDir()
	owned, err := distilled.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(owned.Recipe.Distillation, d) || owned.Recipe.Distillation == d {
		t.Fatal("the snapshot does not own an equal distillation block")
	}
}

func TestLoadDistillationKeepsEachTeachersMappedTopKForEveryRow(t *testing.T) {
	c, d := kdConfig(t, 0.25, 0.5)
	signals, err := loadDistillation(context.Background(), c, kdRows)
	if err != nil {
		t.Fatal(err)
	}
	for index, row := range kdRows {
		teachers, err := signals.teachersFor(index, row)
		if err != nil {
			t.Fatal(err)
		}
		if len(teachers) != 2 {
			t.Fatal("teacher count differs")
		}
		for i, teacher := range teachers {
			if teacher.Name != d.Teachers[i].Model.Name || teacher.Weight != d.Teachers[i].Weight || len(teacher.Positions) != len(row.InputIDs)-row.PromptTokens {
				t.Fatal("teacher identity, weight or positions differ", teacher.Name, len(teacher.Positions))
			}
			for _, p := range teacher.Positions {
				if len(p.TopK) != len(kdTopK[i]) {
					t.Fatal("top-k differs")
				}
				mass := 0.0
				for k, q := range p.TopK {
					if q.TokenID != kdTopK[i][k].StudentTokenID || q.Probability != kdTopK[i][k].Probability {
						t.Fatal("top-k entry differs", q)
					}
					mass += q.Probability
				}
				if p.RetainedMass != mass {
					t.Fatal("retained mass differs")
				}
			}
		}
	}
	// Rows first and third both have one completion token, so only the id
	// tells their signals apart.
	if _, err := signals.teachersFor(2, kdRows[0]); !errors.Is(err, ErrDistillation) {
		t.Fatal("another row's signals were handed out", err)
	}
	if _, err := signals.teachersFor(len(kdRows), kdRows[0]); !errors.Is(err, ErrDistillation) {
		t.Fatal("signals past the curriculum were handed out", err)
	}
	short := kdRows[1]
	short.InputIDs = short.InputIDs[:2]
	if _, err := signals.teachersFor(1, short); !errors.Is(err, ErrDistillation) {
		t.Fatal("signals for another completion length were handed out", err)
	}
	if none, err := (*distillationSignals)(nil).teachersFor(0, kdRows[0]); none != nil || err != nil {
		t.Fatal("a recipe without distillation received teachers")
	}
	plain := c
	plain.Recipe.Distillation = nil
	if none, err := loadDistillation(context.Background(), plain, kdRows); none != nil || err != nil {
		t.Fatal("a recipe without distillation read caches", err)
	}
}

func TestLoadDistillationRefusesCachesThatAreNotTheCurriculums(t *testing.T) {
	// teacher-b's cache is replaced by one that is valid on its own terms and
	// pinned by its own digest, so only its relation to the curriculum differs.
	examples := func(change func([]example) []example) func(*testing.T, Config, *Distillation) {
		return func(t *testing.T, c Config, d *Distillation) {
			rows := change(append([]example(nil), kdRows...))
			d.Teachers[1].CacheSHA256, d.Teachers[1].CacheBytes = kdWrite(t, c.CacheDir, d, 1, kdExamples(d, rows))
		}
	}
	path := func(c Config, d *Distillation) string {
		return filepath.Join(c.CacheDir, d.Teachers[1].CacheSHA256+".json")
	}
	for name, change := range map[string]func(*testing.T, Config, *Distillation){
		"generated sequence": examples(func(rows []example) []example {
			rows[1].InputIDs = []int64{1, 2, 5}
			return rows
		}),
		"prompt boundary": examples(func(rows []example) []example {
			rows[1].PromptTokens = 2
			return rows
		}),
		"row order":   examples(func(rows []example) []example { rows[0], rows[1] = rows[1], rows[0]; return rows }),
		"missing row": examples(func(rows []example) []example { return rows[:2] }),
		"extra row": examples(func(rows []example) []example {
			return append(rows, example{"fourth", []int64{1, 2}, []int64{-100, 2}, 1})
		}),
		"dataset": func(t *testing.T, c Config, d *Distillation) {
			other := *d
			other.DatasetSHA256 = fmtHash([]byte("another dataset"))
			d.Teachers[1].CacheSHA256, d.Teachers[1].CacheBytes = kdWrite(t, c.CacheDir, &other, 1, kdExamples(&other, kdRows))
		},
		"other measured vocabulary": func(t *testing.T, c Config, d *Distillation) {
			other := *d
			other.Teachers = append([]DistillationTeacher(nil), d.Teachers...)
			vocabulary := *d.Teachers[1].Mapping.Vocabulary
			vocabulary.SHA256 = fmtHash([]byte("another measurement"))
			other.Teachers[1].Mapping.Vocabulary = &vocabulary
			other.Teachers[1].MappingSHA256, _ = fusioncache.Digest(other.Teachers[1].Mapping)
			d.Teachers[1].CacheSHA256, d.Teachers[1].CacheBytes = kdWrite(t, c.CacheDir, &other, 1, kdExamples(&other, kdRows))
		},
		"other teacher": func(t *testing.T, c Config, d *Distillation) {
			d.Teachers[1].CacheSHA256, d.Teachers[1].CacheBytes = d.Teachers[0].CacheSHA256, d.Teachers[0].CacheBytes
		},
		"altered bytes": func(t *testing.T, c Config, d *Distillation) {
			body, err := os.ReadFile(path(c, d))
			if err != nil {
				t.Fatal(err)
			}
			altered := strings.Replace(string(body), `"teacher-b"`, `"teacher-c"`, 1)
			if altered == string(body) {
				t.Fatal("fixture has nothing to alter")
			}
			if err := os.WriteFile(path(c, d), []byte(altered), 0600); err != nil {
				t.Fatal(err)
			}
		},
		"declared size": func(t *testing.T, c Config, d *Distillation) { d.Teachers[1].CacheBytes++ },
		"missing file": func(t *testing.T, c Config, d *Distillation) {
			if err := os.Remove(path(c, d)); err != nil {
				t.Fatal(err)
			}
		},
		"symlink": func(t *testing.T, c Config, d *Distillation) {
			target := path(c, d) + ".target"
			if err := os.Rename(path(c, d), target); err != nil {
				t.Fatal(err)
			}
			if err := os.Symlink(target, path(c, d)); err != nil {
				t.Fatal(err)
			}
		},
	} {
		t.Run(name, func(t *testing.T) {
			c, d := kdConfig(t, 0.25, 0.5)
			change(t, c, d)
			if _, err := loadDistillation(context.Background(), c, kdRows); !errors.Is(err, ErrDistillation) {
				t.Fatal("a cache that is not the curriculum's was admitted", err)
			}
		})
	}
	c, _ := kdConfig(t, 0.25, 0.5)
	if _, err := loadDistillation(context.Background(), c, kdRows[:2]); !errors.Is(err, ErrDistillation) {
		t.Fatal("a curriculum of another length was admitted")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := loadDistillation(ctx, c, kdRows); !errors.Is(err, context.Canceled) {
		t.Fatal("a cancelled read ran", err)
	}
}
