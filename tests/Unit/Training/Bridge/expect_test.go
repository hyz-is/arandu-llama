package bridge_test

import (
	"math"
	"testing"

	"github.com/tayi-ai/arandu-llama/training/bridge"
)

// tolerance is the rounding a hand-computed probability may differ by.
const tolerance = 1e-12

// dist builds a distribution from raw teacher byte strings and probabilities,
// with retained mass their sum.
func dist(t *testing.T, v *bridge.Vocabulary, pairs map[string]float64) bridge.Distribution {
	t.Helper()
	var d bridge.Distribution
	for raw, p := range pairs {
		d.Probabilities = append(d.Probabilities, bridge.Probability{Token: id(t, v, raw), Probability: p})
		d.RetainedMass += p
	}
	return d
}

func newBridge(t *testing.T, teacher, student *bridge.Vocabulary, stops ...int64) *bridge.Bridge {
	t.Helper()
	b, err := bridge.New(teacher, student, stops)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return b
}

// expectCells requires the target's cells to be exactly want, keyed by the
// student bytes, within tolerance, and in strictly ascending student id.
func expectCells(t *testing.T, student *bridge.Vocabulary, got bridge.Target, want map[string]float64) {
	t.Helper()
	if len(got.Cells) != len(want) {
		t.Errorf("cells %v, want %d cells %v", describe(student, got), len(want), want)
	}
	for raw, p := range want {
		if q := got.Probability(id(t, student, raw)); math.Abs(q-p) > tolerance {
			t.Errorf("cell %q = %.15f, want %.15f (target %v)", raw, q, p, describe(student, got))
		}
	}
	for i := 1; i < len(got.Cells); i++ {
		if got.Cells[i].Token <= got.Cells[i-1].Token {
			t.Errorf("cells not in strictly ascending student id: %v", got.Cells)
		}
	}
}

func expectNear(t *testing.T, name string, got, want float64) {
	t.Helper()
	if math.Abs(got-want) > tolerance {
		t.Errorf("%s = %.15f, want %.15f", name, got, want)
	}
}

// expectPartition requires the target to account for all of the teacher's
// mass with no negative part.
func expectPartition(t *testing.T, got bridge.Target) {
	t.Helper()
	if math.Abs(got.Mass()-1) > tolerance {
		t.Errorf("mass %.17f, want 1", got.Mass())
	}
	for _, part := range []float64{got.Residual, got.Stop, got.Special, got.Uncovered, got.Chain} {
		if part < 0 {
			t.Errorf("negative part in %+v", got)
		}
	}
	for _, c := range got.Cells {
		if c.Probability < 0 {
			t.Errorf("negative cell %+v", c)
		}
	}
}

func describe(v *bridge.Vocabulary, target bridge.Target) map[string]float64 {
	out := map[string]float64{}
	for _, c := range target.Cells {
		b, _ := v.Bytes(c.Token)
		out[string(b)] = c.Probability
	}
	return out
}
