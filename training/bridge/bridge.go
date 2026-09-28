package bridge

import (
	"cmp"
	"errors"
	"fmt"
	"math"
	"slices"
)

// ErrQuery reports a distribution, query or alignment the bridge refuses.
var ErrQuery = errors.New("bridge: query rejected")

// NoToken is the student id of the paper's bottom: no student token prefixes
// the teacher bytes.
const NoToken int64 = -1

// Bridge routes teacher tokens to student tokens through their bytes. It is
// compiled once per tokenizer pair, is immutable, and is safe for concurrent
// use.
type Bridge struct {
	teacher, student *Vocabulary
	// index maps student bytes to the token a tokenizer emits for them.
	index map[string]int64
	// longest is the byte length of the longest student token.
	longest int
	// phi is the longest-prefix map over teacher ids (Eq. 3), NoToken for
	// byte-less teacher tokens and for those no student token prefixes.
	phi []int64
	// stops marks the declared teacher stop tokens.
	stops []bool
}

// New compiles the bridge from teacher to student. teacherStops are the
// teacher's declared stop tokens -- the tokenizer's end of sequence, the
// generation configuration's and the chat template's turn end. Their mass is
// reported apart from the byte stream, as Target.Stop, for the caller to give
// to the student's realized stop token.
func New(teacher, student *Vocabulary, teacherStops []int64) (*Bridge, error) {
	if teacher == nil || student == nil {
		return nil, fmt.Errorf("%w: teacher and student vocabularies required", ErrQuery)
	}
	b := &Bridge{
		teacher: teacher,
		student: student,
		index:   make(map[string]int64, student.stats.WithBytes),
		phi:     make([]int64, teacher.Size()),
		stops:   make([]bool, teacher.Size()),
	}
	for id, s := range student.bytes {
		if s == "" {
			continue
		}
		prior, exists := b.index[s]
		if !exists || kindPriority(student.kinds[id]) > kindPriority(student.kinds[prior]) {
			b.index[s] = int64(id)
		}
		b.longest = max(b.longest, len(s))
	}
	for _, stop := range teacherStops {
		if stop < 0 || stop >= int64(teacher.Size()) {
			return nil, fmt.Errorf("%w: teacher stop %d outside the vocabulary", ErrQuery, stop)
		}
		b.stops[stop] = true
	}
	for id, s := range teacher.bytes {
		b.phi[id] = NoToken
		if s != "" && !b.stops[id] {
			b.phi[id] = b.prefix(s, 0)
		}
	}
	return b, nil
}

// Teacher returns the teacher vocabulary.
func (b *Bridge) Teacher() *Vocabulary { return b.teacher }

// Student returns the student vocabulary.
func (b *Bridge) Student() *Vocabulary { return b.student }

// Phi returns the longest student token whose bytes prefix the bytes of
// teacher token id (the paper's Eq. 3), or NoToken when none does, when the
// token has no bytes, or when it is a declared stop.
func (b *Bridge) Phi(id int64) int64 {
	if id < 0 || id >= int64(len(b.phi)) {
		return NoToken
	}
	return b.phi[id]
}

// Stop reports whether teacher token id is a declared stop token.
func (b *Bridge) Stop(id int64) bool {
	return id >= 0 && id < int64(len(b.stops)) && b.stops[id]
}

// prefix returns the longest student token whose bytes prefix s within limit
// bytes (no limit when limit is zero), or NoToken.
func (b *Bridge) prefix(s string, limit int) int64 {
	n := min(len(s), b.longest)
	if limit > 0 {
		n = min(n, limit)
	}
	for ; n > 0; n-- {
		if id, ok := b.index[s[:n]]; ok {
			return id
		}
	}
	return NoToken
}

// Probability is one token's probability. Token is a teacher id in a
// Distribution and a student id in a Target.
type Probability struct {
	Token       int64
	Probability float64
}

// Distribution is the teacher's next-token distribution as a cache keeps it:
// the retained top-k entries, in any order, and their unrenormalized sum.
// The mass outside the entries is 1 - RetainedMass.
type Distribution struct {
	Probabilities []Probability
	RetainedMass  float64
}

// massTolerance is how far RetainedMass may sit from the sum of the entries,
// and the sum above one, before a distribution is refused.
const massTolerance = 1e-9

// entry is a validated teacher entry.
type entry struct {
	token int64
	p     float64
}

// validated is a distribution after validate: entries in teacher-id order and
// the mass outside them.
type validated struct {
	entries   []entry
	uncovered float64
}

// validate returns the entries sorted by teacher id, their sum accumulated in
// that order, and the uncovered mass.
func (b *Bridge) validate(d Distribution) ([]entry, float64, error) {
	entries := make([]entry, len(d.Probabilities))
	for i, p := range d.Probabilities {
		if p.Token < 0 || p.Token >= int64(b.teacher.Size()) {
			return nil, 0, fmt.Errorf("%w: teacher token %d outside the vocabulary", ErrQuery, p.Token)
		}
		if !finite(p.Probability) || p.Probability < 0 || p.Probability > 1 {
			return nil, 0, fmt.Errorf("%w: probability %v of teacher token %d", ErrQuery, p.Probability, p.Token)
		}
		entries[i] = entry{p.Token, p.Probability}
	}
	slices.SortFunc(entries, func(x, y entry) int { return cmp.Compare(x.token, y.token) })
	sum := 0.0
	for i, e := range entries {
		if i > 0 && e.token == entries[i-1].token {
			return nil, 0, fmt.Errorf("%w: teacher token %d repeated", ErrQuery, e.token)
		}
		sum += e.p
	}
	if !finite(d.RetainedMass) || sum > 1+massTolerance || math.Abs(sum-d.RetainedMass) > massTolerance {
		return nil, 0, fmt.Errorf("%w: retained mass %v differs from the entries' sum %v", ErrQuery, d.RetainedMass, sum)
	}
	return entries, max(0, 1-sum), nil
}

func finite(v float64) bool { return !math.IsNaN(v) && !math.IsInf(v, 0) }
