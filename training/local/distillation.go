package local

import (
	"context"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"

	"github.com/tayi-ai/arandu-llama/training/decoder"
	"github.com/tayi-ai/arandu-llama/training/fusioncache"
)

// ErrDistillation refuses a distillation block, cache or step whose identity,
// alignment or bounds differ from what the recipe declares.
var ErrDistillation = errors.New("local training: distillation refused")

// DistillationMethod is the one distillation this package runs: teacher-forced
// top-k logit distillation over the student's own vocabulary.
const DistillationMethod = "logit-kd-v1"

// maxDistillationTeachers is the teacher bound every fusion recipe shares.
const maxDistillationTeachers = 8

// Distillation declares teacher-forced logit distillation for every step of
// the curriculum. Each teacher names one fusioncache cache by the SHA-256 of
// its bytes. A cache holds one record for every curriculum row, in curriculum
// order, taken over exactly that row's tokens: the examples it is validated
// against are built from the admitted curriculum itself, with DatasetID,
// DatasetSHA256, role "train" and the row's token ids on both sides. So a
// cache taken over a sequence the teacher generated, over another prompt
// boundary or over any other rows is refused, never realigned.
//
// Weights are the teachers' shares of every position's target and their sum
// is alpha, at most one. The remaining mass, and each teacher's mass outside
// its retained top-k, goes to the gold token, as
// decoder.FusionCompletionGradient defines it. With alpha zero the target is
// the gold token alone and the step runs the SFT calculation itself, so it is
// the SFT step bit for bit; the caches are still read and validated.
//
// CacheLimits bounds every cache read. Student is the student every cache
// names; its Vocabulary must equal the rows of the model's output head.
type Distillation struct {
	Method        string
	DatasetID     string
	DatasetSHA256 string
	Student       fusioncache.ModelIdentity
	Teachers      []DistillationTeacher
	CacheLimits   fusioncache.Limits
}

// DistillationTeacher is one teacher, its mapping to the student, its weight
// and its cache. The mapping must be an identity mapping, over one tokenizer
// file or a measured vocabulary (fusioncache.VocabularyMapping); a bijection
// between different vocabularies is refused here. CacheBytes is the exact size
// of the cache, reserved before it is read.
type DistillationTeacher struct {
	Model         fusioncache.ModelIdentity
	Mapping       fusioncache.TokenMapping
	MappingSHA256 string
	Weight        float64
	CacheSHA256   string
	CacheBytes    int64
}

// alpha is the teachers' total share of the target.
func (d *Distillation) alpha() float64 {
	alpha := 0.0
	for _, t := range d.Teachers {
		alpha += t.Weight
	}
	return alpha
}

// validate holds the block to what can be checked without reading a cache.
func (d *Distillation) validate(baseRevision string) error {
	if d.Method != DistillationMethod || d.DatasetID == "" || !validHash(d.DatasetSHA256) || d.Student.Revision != baseRevision || len(d.Teachers) < 1 || len(d.Teachers) > maxDistillationTeachers {
		return fmt.Errorf("%w: method, dataset, student revision or teacher count differs", ErrDistillation)
	}
	names := make(map[string]bool, len(d.Teachers))
	for _, t := range d.Teachers {
		if names[t.Model.Name] || !t.Mapping.Identity || !(t.Weight >= 0) || math.IsInf(t.Weight, 0) || !validHash(t.CacheSHA256) || t.CacheBytes < 1 || t.CacheBytes > d.CacheLimits.MaxBytes {
			return fmt.Errorf("%w: teacher %q repeated, not on the student's vocabulary, or its weight or cache invalid", ErrDistillation, t.Model.Name)
		}
		if err := fusioncache.ValidateMapping(t.Model, d.Student, t.Mapping, t.MappingSHA256, d.CacheLimits); err != nil {
			return fmt.Errorf("%w: teacher %q: %w", ErrDistillation, t.Model.Name, err)
		}
		names[t.Model.Name] = true
	}
	// The same tolerance FusionCompletionGradient applies to a sum of weights.
	if alpha := d.alpha(); alpha > 1+1e-12 {
		return fmt.Errorf("%w: alpha %g exceeds one", ErrDistillation, alpha)
	}
	return nil
}

// distillationSignals holds, for every curriculum row, each teacher's
// positions over that row's completion, in the student's token ids.
type distillationSignals struct {
	ids      []string
	teachers []distilledTeacher
}

type distilledTeacher struct {
	name   string
	weight float64
	rows   [][]decoder.FusionTeacherPosition
}

// loadDistillation reads and validates every cache the recipe names against
// the admitted curriculum, one teacher at a time, keeping only the mapped
// top-k of each position. A recipe without distillation loads nothing.
func loadDistillation(ctx context.Context, c Config, rows []example) (*distillationSignals, error) {
	d := c.Recipe.Distillation
	if d == nil {
		return nil, nil
	}
	if len(rows) != c.Recipe.ExampleCount || !filepath.IsAbs(c.CacheDir) {
		return nil, ErrDistillation
	}
	signals := &distillationSignals{ids: make([]string, len(rows))}
	examples := make([]fusioncache.Example, len(rows))
	for i, row := range rows {
		signals.ids[i] = row.ID
		examples[i] = fusioncache.Example{DatasetID: d.DatasetID, DatasetSHA256: d.DatasetSHA256, ID: row.ID, Role: "train", TeacherTokens: row.InputIDs, StudentTokens: row.InputIDs, PromptTokens: row.PromptTokens}
	}
	for _, t := range d.Teachers {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		path := filepath.Join(c.CacheDir, t.CacheSHA256+".json")
		info, err := os.Lstat(path)
		if err != nil {
			return nil, errors.Join(ErrDistillation, err)
		}
		if !info.Mode().IsRegular() || info.Size() != t.CacheBytes {
			return nil, fmt.Errorf("%w: cache of %s is not a regular file of its declared size", ErrDistillation, t.Model.Name)
		}
		expected := fusioncache.Expectation{Teacher: t.Model, Student: d.Student, Mapping: t.Mapping, MappingSHA256: t.MappingSHA256, Examples: examples}
		cache, err := fusioncache.Read(path, t.CacheSHA256, expected, d.CacheLimits)
		if err != nil {
			return nil, fmt.Errorf("%w: cache of %s: %w", ErrDistillation, t.Model.Name, err)
		}
		teacher := distilledTeacher{name: t.Model.Name, weight: t.Weight, rows: make([][]decoder.FusionTeacherPosition, len(cache.Records))}
		for i, record := range cache.Records {
			positions := make([]decoder.FusionTeacherPosition, len(record.Positions))
			for j, p := range record.Positions {
				top := make([]decoder.FusionTokenProbability, len(p.Probabilities))
				for k, q := range p.Probabilities {
					top[k] = decoder.FusionTokenProbability{TokenID: q.StudentTokenID, Probability: q.Probability}
				}
				positions[j] = decoder.FusionTeacherPosition{RetainedMass: p.RetainedMass, TopK: top}
			}
			teacher.rows[i] = positions
		}
		signals.teachers = append(signals.teachers, teacher)
	}
	return signals, ctx.Err()
}

// teachersFor returns the teachers' signals for curriculum row index, refusing
// unless the cached row is this row and covers each of its completion tokens.
func (s *distillationSignals) teachersFor(index int, row example) ([]decoder.FusionTeacher, error) {
	if s == nil {
		return nil, nil
	}
	if index < 0 || index >= len(s.ids) || s.ids[index] != row.ID {
		return nil, fmt.Errorf("%w: cached row differs from curriculum row %q", ErrDistillation, row.ID)
	}
	completion := len(row.InputIDs) - row.PromptTokens
	teachers := make([]decoder.FusionTeacher, len(s.teachers))
	for i, t := range s.teachers {
		if len(t.rows[index]) != completion {
			return nil, fmt.Errorf("%w: %s holds %d positions for the %d completion tokens of %q", ErrDistillation, t.name, len(t.rows[index]), completion, row.ID)
		}
		teachers[i] = decoder.FusionTeacher{Name: t.name, Weight: t.weight, Positions: t.rows[index]}
	}
	return teachers, nil
}
