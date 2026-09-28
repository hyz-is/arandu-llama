package bridge

import (
	"cmp"
	"errors"
	"fmt"
	"slices"
	"strings"
)

// ErrNoSupport reports a realized prefix to which the teacher distribution
// gives no mass at all, retained or uncovered: the conditional target of
// Eq. 5 is undefined there, and the row is excluded from the loss.
var ErrNoSupport = errors.New("bridge: realized prefix has no teacher mass")

// Bound says what a target guarantees about the byte-prefix marginal.
type Bound uint8

const (
	// BoundExact is the byte-prefix marginal itself: the student token does
	// not cross a teacher boundary (Proposition C.3 and Eq. 5).
	BoundExact Bound = iota + 1
	// BoundLowerBound carries the realized-path chain value of Eq. 6 on the
	// spanning cells, a lower bound on their marginal.
	BoundLowerBound
)

// String names the bound.
func (b Bound) String() string {
	switch b {
	case BoundExact:
		return "exact"
	case BoundLowerBound:
		return "lower-bound"
	}
	return fmt.Sprintf("bound(%d)", uint8(b))
}

// Query is one student position against one teacher distribution.
type Query struct {
	// Prefix holds the bytes of the current teacher token already emitted
	// before this student position: empty at a teacher boundary.
	Prefix []byte
	// Limit caps, in bytes, how much of a teacher token's continuation one
	// student token may take; zero leaves it uncapped. The paper caps the
	// within-token target at the chunk boundary.
	Limit int
	// Teacher is the teacher's distribution for the token that holds, or
	// begins at, this position.
	Teacher Distribution
}

// Target is the teacher distribution re-expressed over the student
// vocabulary. Cells, Residual, Stop, Special and Uncovered partition the
// teacher mass and sum to one up to rounding.
type Target struct {
	// Cells are the student tokens the routing map, and any spanning
	// correction, represents explicitly, in ascending student id. A cell may
	// hold zero.
	Cells []Probability
	// Residual is the paper's empty cell: mass of byte-bearing teacher tokens
	// that no student token prefixes, including, inside a token, the tokens
	// whose bytes end exactly at the realized prefix.
	Residual float64
	// Stop is the mass of the declared teacher stop tokens. The paper routes
	// it to the student's realized stop token; at the row that emits that
	// token it is the stopping bridge b(s*) of Eq. 8.
	Stop float64
	// Special is the mass of byte-less teacher tokens that are not stops:
	// control, unknown, unused and unrepresentable tokens.
	Special float64
	// Uncovered is the mass outside the teacher's retained top-k.
	Uncovered float64
	// Chain is the realized-path mass moved to spanning cells (Eq. 6).
	Chain float64
	// Support is the teacher mass the target is conditioned on: one at a
	// teacher boundary; inside a token, the retained mass of the tokens that
	// continue the prefix plus the uncovered mass.
	Support float64
	// Bound says whether the target is the exact marginal or a lower bound.
	Bound Bound
}

// Mass returns the total of the partition, one up to rounding.
func (t Target) Mass() float64 {
	sum := t.Residual + t.Stop + t.Special + t.Uncovered
	for _, c := range t.Cells {
		sum += c.Probability
	}
	return sum
}

// Complement returns the mass no student cell holds, stops excluded: the
// student side's complement coordinate trains against it.
func (t Target) Complement() float64 { return t.Residual + t.Special + t.Uncovered }

// Probability returns the mass of student token id, zero when it has no cell.
func (t Target) Probability(id int64) float64 {
	if i, ok := slices.BinarySearchFunc(t.Cells, id, func(c Probability, id int64) int { return cmp.Compare(c.Token, id) }); ok {
		return t.Cells[i].Probability
	}
	return 0
}

// cell is one routed teacher entry on its way to a student cell.
type cell struct {
	student int64
	p       float64
}

// Marginalize computes the BPM target at one student position. With an empty
// prefix it is the scatter of Eq. 4: each teacher token's mass goes to the
// longest student token prefixing its bytes. With a prefix it is Eq. 5:
// the teacher tokens that continue the prefix, scattered by the student token
// that prefixes their remaining bytes and normalized by their mass plus the
// uncovered mass, which keeps every cell a lower bound when the distribution
// is truncated. Both are exact while the student token stays inside the
// teacher token; a spanning token needs Rows.
func (b *Bridge) Marginalize(q Query) (Target, error) {
	if q.Limit < 0 {
		return Target{}, fmt.Errorf("%w: negative limit %d", ErrQuery, q.Limit)
	}
	entries, uncovered, err := b.validate(q.Teacher)
	if err != nil {
		return Target{}, err
	}
	return b.marginalize(string(q.Prefix), q.Limit, entries, uncovered)
}

func (b *Bridge) marginalize(prefix string, limit int, entries []entry, uncovered float64) (Target, error) {
	t := Target{Bound: BoundExact}
	routed := make([]cell, 0, len(entries))
	if prefix == "" {
		for _, e := range entries {
			s, ok := b.teacher.representation(e.token)
			switch {
			case b.stops[e.token]:
				t.Stop += e.p
			case !ok:
				t.Special += e.p
			default:
				u := b.phi[e.token]
				if limit > 0 && u != NoToken && len(b.student.bytes[u]) > limit {
					u = b.prefix(s, limit)
				}
				if u == NoToken {
					t.Residual += e.p
				} else {
					routed = append(routed, cell{u, e.p})
				}
			}
		}
		t.Uncovered, t.Support = uncovered, 1
		t.Cells = gather(routed, 1)
		return t, nil
	}
	support := 0.0
	for _, e := range entries {
		s, ok := b.teacher.representation(e.token)
		if !ok || b.stops[e.token] || !strings.HasPrefix(s, prefix) {
			continue
		}
		support += e.p
		u := NoToken
		if rest := s[len(prefix):]; rest != "" {
			u = b.prefix(rest, limit)
		}
		if u == NoToken {
			t.Residual += e.p
		} else {
			routed = append(routed, cell{u, e.p})
		}
	}
	denominator := support + uncovered
	if denominator <= 0 {
		return Target{}, ErrNoSupport
	}
	t.Residual /= denominator
	t.Uncovered = uncovered / denominator
	t.Support = denominator
	t.Cells = gather(routed, denominator)
	return t, nil
}

// gather sums routed entries per student token, in teacher-id order within a
// token, and divides each sum by denominator.
func gather(routed []cell, denominator float64) []Probability {
	slices.SortStableFunc(routed, func(x, y cell) int { return cmp.Compare(x.student, y.student) })
	out := make([]Probability, 0, len(routed))
	for _, r := range routed {
		if n := len(out); n > 0 && out[n-1].Token == r.student {
			out[n-1].Probability += r.p
			continue
		}
		out = append(out, Probability{r.student, r.p})
	}
	if denominator != 1 {
		for i := range out {
			out[i].Probability /= denominator
		}
	}
	return out
}
