package router

import (
	"cmp"
	"fmt"
	"math"
	"slices"
	"strings"
)

// Level names what a decision chose teachers for.
type Level string

const (
	// LevelCapability is the offline shortlist of one capability.
	LevelCapability Level = "capability"
	// LevelExample is the choice for one example, made before generation.
	LevelExample Level = "example"
)

// DecisionSchema identifies the encoding of a Decision.
const DecisionSchema = "tayi.tcap.router.decision.v1"

const inputsSchema = "tayi.tcap.router.inputs.v1"

// Score is an admitted teacher's factors and its score before redundancy:
// Base is S_t with L_t = 0. Competence is the configured statistic of the
// cell the decision scored on, and Independence is I_t over the admitted
// pool.
type Score struct {
	Teacher         string  `json:"teacher"`
	Bias            float64 `json:"bias"`
	Reliability     float64 `json:"reliability"`
	Competence      float64 `json:"competence"`
	Reproducibility float64 `json:"reproducibility"`
	Independence    float64 `json:"independence"`
	Cost            float64 `json:"cost"`
	Base            float64 `json:"base"`
}

// Standing is a candidate's redundancy L_t and score S_t in one round.
type Standing struct {
	Teacher    string  `json:"teacher"`
	Redundancy float64 `json:"redundancy"`
	Score      float64 `json:"score"`
}

// Round lists every candidate still unchosen when the teacher of Rank was
// picked, best first. Its first standing is the teacher chosen.
type Round struct {
	Rank      int        `json:"rank"`
	Standings []Standing `json:"standings"`
}

// Selection is a chosen teacher: its rank in the router's order, its
// redundancy and score when it was picked, and its softmax weight among the
// chosen teachers.
type Selection struct {
	Teacher    string  `json:"teacher"`
	Rank       int     `json:"rank"`
	Redundancy float64 `json:"redundancy"`
	Score      float64 `json:"score"`
	Weight     float64 `json:"weight"`
}

// Exclusion is a teacher refused by its gates, with every reason that
// applied, in a fixed order.
type Exclusion struct {
	Teacher string   `json:"teacher"`
	Reasons []Reason `json:"reasons"`
}

// Decision is the router's serializable record. Selected is in the router's
// order, which is the order a caller consults teachers in; Scores and
// Excluded are sorted by teacher id and together name every teacher of the
// input exactly once. An empty Selected means no teacher was admitted, and
// the caller falls back to the ground truth.
//
// ConfigSHA256 is the configuration's Digest, TeachersSHA256 the digest of
// the canonical teacher table and InputsSHA256 the digest of the teacher
// table together with the level and the routed identity.
type Decision struct {
	Schema         string      `json:"schema"`
	Level          Level       `json:"level"`
	ExampleID      string      `json:"example_id,omitempty"`
	Subcapability  string      `json:"subcapability"`
	ConfigSHA256   string      `json:"config_sha256"`
	TeachersSHA256 string      `json:"teachers_sha256"`
	InputsSHA256   string      `json:"inputs_sha256"`
	K              int         `json:"k"`
	Selected       []Selection `json:"selected"`
	Rounds         []Round     `json:"rounds"`
	Scores         []Score     `json:"scores"`
	Excluded       []Exclusion `json:"excluded"`
}

// Route chooses at most config.K teachers for one example, before any of them
// generates. Teachers failing a gate, or without a competence cell for the
// example's subcapability, are excluded with their reasons. The rest are
// chosen one at a time: each round scores every unchosen candidate with its
// redundancy against the teachers already chosen and picks the highest
// score, the lower id on an exact tie.
func Route(config Config, teachers []Teacher, example Example) (Decision, error) {
	if !validID(example.ID) || !validID(example.Subcapability) {
		return Decision{}, fmt.Errorf("%w: example id %q or subcapability %q is empty", ErrInput, example.ID, example.Subcapability)
	}
	return decide(config, teachers, LevelExample, example.ID, example.Subcapability, config.K)
}

// Shortlist is the offline level of the selection: it keeps at most
// config.Shortlist teachers for a capability, scored on each teacher's cell
// for the capability itself, with the same gates, factors and redundancy as
// Route. A caller routes examples among the teachers it keeps.
func Shortlist(config Config, teachers []Teacher, capability string) (Decision, error) {
	if !validID(capability) {
		return Decision{}, fmt.Errorf("%w: capability %q is empty", ErrInput, capability)
	}
	return decide(config, teachers, LevelCapability, "", capability, config.Shortlist)
}

type candidate struct {
	teacher Teacher
	score   Score
}

func decide(config Config, teachers []Teacher, level Level, exampleID, key string, k int) (Decision, error) {
	configSHA256, err := config.Digest()
	if err != nil {
		return Decision{}, err
	}
	table, err := canonical(teachers)
	if err != nil {
		return Decision{}, err
	}
	teachersSHA256, err := digest(struct {
		Schema   string    `json:"schema"`
		Teachers []Teacher `json:"teachers"`
	}{teachersSchema, table})
	if err != nil {
		return Decision{}, err
	}
	inputsSHA256, err := digest(struct {
		Schema         string `json:"schema"`
		Level          Level  `json:"level"`
		TeachersSHA256 string `json:"teachers_sha256"`
		ExampleID      string `json:"example_id"`
		Subcapability  string `json:"subcapability"`
	}{inputsSchema, level, teachersSHA256, exampleID, key})
	if err != nil {
		return Decision{}, err
	}
	d := Decision{
		Schema:         DecisionSchema,
		Level:          level,
		ExampleID:      exampleID,
		Subcapability:  key,
		ConfigSHA256:   configSHA256,
		TeachersSHA256: teachersSHA256,
		InputsSHA256:   inputsSHA256,
		K:              k,
		Selected:       []Selection{},
		Rounds:         []Round{},
		Scores:         []Score{},
		Excluded:       []Exclusion{},
	}

	var admitted []candidate
	for _, t := range table {
		reasons, cell := gates(t, key)
		if len(reasons) > 0 {
			d.Excluded = append(d.Excluded, Exclusion{Teacher: t.ID, Reasons: reasons})
			continue
		}
		competence := cell.Accuracy
		if config.Competence == StatisticLowerBound {
			competence = cell.CILower
		}
		admitted = append(admitted, candidate{teacher: t, score: Score{
			Teacher:         t.ID,
			Bias:            config.Biases[t.ID],
			Reliability:     t.Reliability,
			Competence:      competence,
			Reproducibility: t.Reproducibility,
			Cost:            t.Cost,
		}})
	}
	for i := range admitted {
		relatives := 0
		for _, other := range admitted {
			if sharedRoots(admitted[i].teacher.Roots, other.teacher.Roots) > 0 {
				relatives++
			}
		}
		s := &admitted[i].score
		s.Independence = 1 / float64(relatives)
		s.Base = base(config, *s)
		if !finite(s.Base) {
			return Decision{}, fmt.Errorf("%w: teacher %s base score is %v", ErrScore, s.Teacher, s.Base)
		}
		d.Scores = append(d.Scores, *s)
	}

	remaining := admitted
	var chosen []candidate
	for rank := 1; rank <= k && len(remaining) > 0; rank++ {
		standings := make([]Standing, len(remaining))
		for i, c := range remaining {
			shared := 0
			for _, picked := range chosen {
				shared += sharedRoots(c.teacher.Roots, picked.teacher.Roots)
			}
			redundancy := float64(shared)
			score := float64(c.score.Base - float64(config.Coefficients.Redundancy*redundancy))
			if !finite(score) {
				return Decision{}, fmt.Errorf("%w: teacher %s score is %v", ErrScore, c.teacher.ID, score)
			}
			standings[i] = Standing{Teacher: c.teacher.ID, Redundancy: redundancy, Score: score}
		}
		slices.SortFunc(standings, func(a, b Standing) int {
			if a.Score != b.Score {
				return cmp.Compare(b.Score, a.Score)
			}
			return strings.Compare(a.Teacher, b.Teacher)
		})
		best := standings[0]
		d.Rounds = append(d.Rounds, Round{Rank: rank, Standings: standings})
		d.Selected = append(d.Selected, Selection{Teacher: best.Teacher, Rank: rank, Redundancy: best.Redundancy, Score: best.Score})
		index := slices.IndexFunc(remaining, func(c candidate) bool { return c.teacher.ID == best.Teacher })
		chosen = append(chosen, remaining[index])
		remaining = slices.Delete(slices.Clone(remaining), index, index+1)
	}
	if err := weigh(d.Selected, config.Tau); err != nil {
		return Decision{}, err
	}
	return d, nil
}

// base is S_t without the redundancy term, evaluated in a fixed order. Every
// product is converted explicitly so the compiler cannot fuse it with the
// following addition, which would change the last bits on some architectures.
func base(config Config, s Score) float64 {
	a := config.Coefficients
	eps := config.Epsilon
	term := func(coefficient, factor float64) float64 {
		return float64(coefficient * math.Log(float64(eps+factor)))
	}
	total := s.Bias
	total = float64(total + term(a.Reliability, s.Reliability))
	total = float64(total + term(a.Competence, s.Competence))
	total = float64(total + term(a.Reproducibility, s.Reproducibility))
	total = float64(total + term(a.Independence, s.Independence))
	return float64(total - float64(a.Cost*s.Cost))
}

// weigh sets the softmax of score/tau over the chosen teachers, summed in the
// router's order. A weight that is not positive and finite is refused.
func weigh(selected []Selection, tau float64) error {
	if len(selected) == 0 {
		return nil
	}
	top := selected[0].Score
	for _, s := range selected {
		top = math.Max(top, s.Score)
	}
	exponents := make([]float64, len(selected))
	sum := 0.0
	for i, s := range selected {
		exponents[i] = math.Exp(float64(s.Score-top) / tau)
		sum += exponents[i]
	}
	for i := range selected {
		weight := exponents[i] / sum
		if !finite(weight) || weight <= 0 {
			return fmt.Errorf("%w: weight of %s is %v; tau %v silences it", ErrScore, selected[i].Teacher, weight, tau)
		}
		selected[i].Weight = weight
	}
	return nil
}
