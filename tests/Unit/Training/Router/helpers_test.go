package router_test

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"math"
	"math/rand/v2"
	"slices"
	"testing"

	"github.com/tayi-ai/arandu-llama/training/router"
)

const (
	capability    = "reasoning.math"
	subcapability = "reasoning.math.gsm8k"
	items         = 300
)

// fixture is a test hypothesis, not a registered one: the package ships no
// coefficients, and these exist only to exercise the arithmetic.
func fixture() router.Config {
	return router.Config{
		Coefficients: router.Coefficients{Reliability: 1, Competence: 4, Reproducibility: 1, Independence: 0, Redundancy: 0.25, Cost: 0},
		Epsilon:      0.01,
		Tau:          1,
		K:            2,
		Shortlist:    3,
		Competence:   router.StatisticAccuracy,
		Independence: router.IndependenceInverseRelatives,
		Redundancy:   router.RedundancySharedRoots,
	}
}

func pin(id string) string {
	sum := sha256.Sum256([]byte(id))
	return hex.EncodeToString(sum[:])
}

// cell builds a consistent competence cell with a normal-approximation
// interval; the router checks the interval's shape, not how it was drawn.
func cell(sub string, correct, n int) router.Competence {
	accuracy := float64(correct) / float64(n)
	half := 1.96 * math.Sqrt(accuracy*(1-accuracy)/float64(n))
	return router.Competence{
		Subcapability: sub,
		Items:         n,
		Correct:       correct,
		Accuracy:      accuracy,
		CILower:       math.Max(0, accuracy-half),
		CIUpper:       math.Min(1, accuracy+half),
	}
}

// teacher is admitted on every gate, perfectly reliable and reproducible, at
// no cost, with the same result on the capability and the subcapability.
func teacher(id string, roots []string, correct int) router.Teacher {
	return router.Teacher{
		ID:              id,
		Roots:           roots,
		Gates:           router.Gates{LicenseVerified: true, ArtifactSHA256: pin(id), Qualified: true},
		Reliability:     1,
		Reproducibility: 1,
		Competence:      []router.Competence{cell(subcapability, correct, items), cell(capability, correct, items)},
	}
}

func example() router.Example {
	return router.Example{ID: "gsm8k-train-0042", Subcapability: subcapability}
}

func route(t *testing.T, config router.Config, teachers []router.Teacher) router.Decision {
	t.Helper()
	d, err := router.Route(config, teachers, example())
	if err != nil {
		t.Fatalf("route: %v", err)
	}
	return d
}

func selected(d router.Decision) []string {
	ids := make([]string, len(d.Selected))
	for i, s := range d.Selected {
		ids[i] = s.Teacher
	}
	return ids
}

func score(d router.Decision, id string) router.Score {
	for _, s := range d.Scores {
		if s.Teacher == id {
			return s
		}
	}
	return router.Score{Teacher: "absent", Base: math.NaN()}
}

func encode(t *testing.T, d router.Decision) string {
	t.Helper()
	body, err := json.Marshal(d)
	if err != nil {
		t.Fatalf("encode decision: %v", err)
	}
	return string(body)
}

// shuffled returns a deep copy with teachers, roots and competence cells in a
// random order drawn from source.
func shuffled(teachers []router.Teacher, source *rand.Rand) []router.Teacher {
	out := make([]router.Teacher, len(teachers))
	for i, t := range teachers {
		t.Roots = slices.Clone(t.Roots)
		source.Shuffle(len(t.Roots), func(a, b int) { t.Roots[a], t.Roots[b] = t.Roots[b], t.Roots[a] })
		t.Competence = slices.Clone(t.Competence)
		source.Shuffle(len(t.Competence), func(a, b int) { t.Competence[a], t.Competence[b] = t.Competence[b], t.Competence[a] })
		out[i] = t
	}
	source.Shuffle(len(out), func(a, b int) { out[a], out[b] = out[b], out[a] })
	return out
}
