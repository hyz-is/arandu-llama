package bridge_test

import (
	"errors"
	"math/rand/v2"
	"reflect"
	"strings"
	"testing"

	"github.com/tayi-ai/arandu-llama/training/bridge"
)

// tokens maps raw byte strings to ids of v.
func tokens(t *testing.T, v *bridge.Vocabulary, raw ...string) []int64 {
	t.Helper()
	out := make([]int64, len(raw))
	for i, r := range raw {
		out[i] = id(t, v, r)
	}
	return out
}

func rows(t *testing.T, b *bridge.Bridge, student, teacher []int64, dists []bridge.Distribution, options bridge.RowOptions) []bridge.Row {
	t.Helper()
	a, err := b.Align(student, teacher)
	if err != nil {
		t.Fatal(err)
	}
	out, err := b.Rows(a, dists, options)
	if err != nil {
		t.Fatal(err)
	}
	return out
}

func TestIdenticalSegmentationsAreOneToOneAndExact(t *testing.T) {
	v := vocab(t, "The", " answer", " is", " 12", ".")
	b := newBridge(t, v, v)
	seq := tokens(t, v, "The", " answer", " is", " 12", ".")
	dists := make([]bridge.Distribution, len(seq))
	for j, token := range seq {
		dists[j] = bridge.Distribution{Probabilities: []bridge.Probability{{Token: token, Probability: 0.7}, {Token: (token + 1) % 5, Probability: 0.2}}, RetainedMass: 0.9}
	}
	for i, row := range rows(t, b, seq, seq, dists, bridge.RowOptions{}) {
		if row.Relation != bridge.RelationOneToOne || row.Target == nil || row.Target.Bound != bridge.BoundExact {
			t.Fatalf("row %d: %+v", i, row)
		}
		expectNear(t, "realized", row.Target.Probability(seq[i]), 0.7)
		expectNear(t, "uncovered", row.Target.Uncovered, 0.1)
		expectPartition(t, *row.Target)
	}
}

// TestOneStudentTokenSpanningThreeTeacherTokens is the paper's </think> case:
// the teacher reads "</" "think" ">\n\n" and the student "</think>" "\n\n".
// The scatter alone gives the spanning token nothing; the chain gives it
// q("</") * q("think") * prefix-mass(">") = 0.7 * 0.9 * 0.95, taken from the
// head cell.
func TestOneStudentTokenSpanningThreeTeacherTokens(t *testing.T) {
	student := vocab(t, "<", "</", "</think>", "think", ">", "\n", "\n\n", "a", "x")
	teacher := vocab(t, "</", "<", "a", "think", "thinking", "x", ">\n\n", ">", ">\n", "b")
	b := newBridge(t, teacher, student)
	dists := []bridge.Distribution{
		dist(t, teacher, map[string]float64{"</": 0.7, "<": 0.2, "a": 0.1}),
		dist(t, teacher, map[string]float64{"think": 0.9, "thinking": 0.05, "x": 0.05}),
		dist(t, teacher, map[string]float64{">\n\n": 0.6, ">": 0.2, ">\n": 0.15, "b": 0.05}),
	}
	got := rows(t, b, tokens(t, student, "</think>", "\n\n"), tokens(t, teacher, "</", "think", ">\n\n"), dists, bridge.RowOptions{})

	spanning := got[0]
	if spanning.Relation != bridge.RelationSpanning || spanning.Target.Bound != bridge.BoundLowerBound {
		t.Fatalf("row 0: %+v", spanning)
	}
	chain := 0.7 * 0.9 * 0.95
	expectCells(t, student, *spanning.Target, map[string]float64{"<": 0.2, "</": 0.7 - chain, "</think>": chain, "a": 0.1})
	expectNear(t, "chain", spanning.Target.Chain, chain)
	expectPartition(t, *spanning.Target)

	scatter, err := b.Marginalize(bridge.Query{Teacher: dists[0]})
	if err != nil {
		t.Fatal(err)
	}
	if scatter.Probability(id(t, student, "</think>")) != 0 {
		t.Error("the first-token scatter alone was expected to miss the spanning token")
	}

	interior := got[1]
	if interior.Relation != bridge.RelationInterior || interior.Prefix != 1 || interior.Target.Bound != bridge.BoundExact {
		t.Fatalf("row 1: %+v", interior)
	}
	expectCells(t, student, *interior.Target, map[string]float64{"\n\n": 0.6 / 0.95, "\n": 0.15 / 0.95})
	expectNear(t, "residual", interior.Target.Residual, 0.2/0.95)
	expectPartition(t, *interior.Target)
}

// TestNestedSpanningCandidatesTelescope adds "</think" to the student: the
// realized path now supports two nested spanning tokens, and each keeps the
// chain value of its bytes minus that of the next longer one.
func TestNestedSpanningCandidatesTelescope(t *testing.T) {
	student := vocab(t, "<", "</", "</think", "</think>", ">", "\n\n", "a")
	teacher := vocab(t, "</", "<", "a", "think", "thinking", "x", ">\n\n", ">", "b")
	b := newBridge(t, teacher, student)
	dists := []bridge.Distribution{
		dist(t, teacher, map[string]float64{"</": 0.7, "<": 0.2, "a": 0.1}),
		dist(t, teacher, map[string]float64{"think": 0.9, "thinking": 0.05, "x": 0.05}),
		dist(t, teacher, map[string]float64{">\n\n": 0.6, ">": 0.35, "b": 0.05}),
	}
	got := rows(t, b, tokens(t, student, "</think>", "\n\n"), tokens(t, teacher, "</", "think", ">\n\n"), dists, bridge.RowOptions{})
	shorter := 0.7 * (0.9 + 0.05)
	longer := 0.7 * 0.9 * 0.95
	expectCells(t, student, *got[0].Target, map[string]float64{
		"<": 0.2, "</": 0.7 - shorter, "</think": shorter - longer, "</think>": longer, "a": 0.1,
	})
	expectNear(t, "chain", got[0].Target.Chain, shorter)
	expectPartition(t, *got[0].Target)
}

// TestTwoTeacherTokensIntoThreeStudentTokens covers M:N: the teacher reads
// "ab" "cd", the student "a" "bc" "d". The middle row starts inside "ab"
// after "a" and ends inside "cd".
func TestTwoTeacherTokensIntoThreeStudentTokens(t *testing.T) {
	student := vocab(t, "a", "b", "c", "d", "x", "bc")
	teacher := vocab(t, "ab", "ax", "a", "b", "cd", "c", "e")
	b := newBridge(t, teacher, student)
	dists := []bridge.Distribution{
		dist(t, teacher, map[string]float64{"ab": 0.5, "ax": 0.2, "a": 0.1, "b": 0.2}),
		dist(t, teacher, map[string]float64{"cd": 0.7, "c": 0.1, "e": 0.2}),
	}
	got := rows(t, b, tokens(t, student, "a", "bc", "d"), tokens(t, teacher, "ab", "cd"), dists, bridge.RowOptions{})

	if got[0].Relation != bridge.RelationHead || got[0].Target.Bound != bridge.BoundExact {
		t.Fatalf("row 0: %+v", got[0])
	}
	expectCells(t, student, *got[0].Target, map[string]float64{"a": 0.8, "b": 0.2})

	middle := got[1]
	if middle.Relation != bridge.RelationMidSpanning || middle.Prefix != 1 || middle.Target.Bound != bridge.BoundLowerBound {
		t.Fatalf("row 1: %+v", middle)
	}
	// Conditioned on "a": support 0.8. The chain of "bc" is q0("ab") times
	// the mass of teacher tokens starting with "c", 0.5 * 0.8, over 0.8.
	expectCells(t, student, *middle.Target, map[string]float64{"b": 0.5/0.8 - 0.5, "bc": 0.5, "x": 0.2 / 0.8})
	expectNear(t, "residual", middle.Target.Residual, 0.1/0.8)
	expectPartition(t, *middle.Target)

	last := got[2]
	if last.Relation != bridge.RelationInterior || last.Target.Bound != bridge.BoundExact {
		t.Fatalf("row 2: %+v", last)
	}
	expectCells(t, student, *last.Target, map[string]float64{"d": 0.7 / 0.8})
	expectNear(t, "residual", last.Target.Residual, 0.1/0.8)
}

// TestARowThatReachesNoSyncPointIsExcluded: teacher "abc" "de", student "a"
// "b" "cd" "e". Row "b" starts inside "abc", whose end falls inside the
// student's "cd".
func TestARowThatReachesNoSyncPointIsExcluded(t *testing.T) {
	student := vocab(t, "a", "b", "cd", "e", "c", "d")
	teacher := vocab(t, "abc", "de", "ab")
	b := newBridge(t, teacher, student)
	dists := []bridge.Distribution{
		dist(t, teacher, map[string]float64{"abc": 0.9, "ab": 0.1}),
		dist(t, teacher, map[string]float64{"de": 1}),
	}
	got := rows(t, b, tokens(t, student, "a", "b", "cd", "e"), tokens(t, teacher, "abc", "de"), dists, bridge.RowOptions{})
	want := []bridge.Relation{bridge.RelationHead, bridge.RelationExcluded, bridge.RelationMidSpanning, bridge.RelationInterior}
	for i, row := range got {
		if row.Relation != want[i] || (row.Target == nil) != (want[i] == bridge.RelationExcluded) {
			t.Errorf("row %d: %s target %v, want %s", i, row.Relation, row.Target, want[i])
		}
		if row.Target != nil {
			expectPartition(t, *row.Target)
		}
	}
}

func TestWhitespaceOnlyRowsAreWithheldUnlessKept(t *testing.T) {
	v := vocab(t, "if", "\n", "    ", "\t", "x", "  ")
	b := newBridge(t, v, v)
	seq := tokens(t, v, "if", "\n", "    ", "x")
	dists := make([]bridge.Distribution, len(seq))
	for j, token := range seq {
		dists[j] = bridge.Distribution{Probabilities: []bridge.Probability{{Token: token, Probability: 1}}, RetainedMass: 1}
	}
	masked := rows(t, b, seq, seq, dists, bridge.RowOptions{})
	if !masked[2].Masked || masked[2].Target != nil || masked[1].Masked || masked[1].Target == nil {
		t.Errorf("rows %+v", masked)
	}
	kept := rows(t, b, seq, seq, dists, bridge.RowOptions{KeepWhitespace: true})
	if kept[2].Masked || kept[2].Target == nil {
		t.Errorf("kept whitespace row %+v", kept[2])
	}
}

func TestAlignRefusesByteLessTokensAndDifferentBytes(t *testing.T) {
	v := vocab(t, "a", "b", "!control:<|im_end|>", "!user:<think>")
	returnID := int64(3)
	b := newBridge(t, v, v, returnID)
	if _, err := b.Align([]int64{0}, []int64{1}); !errors.Is(err, bridge.ErrQuery) {
		t.Errorf("different bytes aligned: %v", err)
	}
	if _, err := b.Align([]int64{0, 2}, []int64{0}); !errors.Is(err, bridge.ErrQuery) {
		t.Errorf("a control token in the byte stream was aligned: %v", err)
	}
	if _, err := b.Align([]int64{0, 3}, []int64{0, 3}); !errors.Is(err, bridge.ErrQuery) {
		t.Errorf("a declared stop with bytes in the teacher stream was aligned: %v", err)
	}
	a, err := b.Align([]int64{0, 1}, []int64{0, 1})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := b.Rows(a, nil, bridge.RowOptions{}); !errors.Is(err, bridge.ErrQuery) {
		t.Errorf("rows without teacher distributions: %v", err)
	}
}

// TestRandomSegmentationsPreserveMassOnEveryRow segments random text two ways
// and requires every admitted row to be a partition of the teacher mass, with
// the bound its relation implies, and the same rows on a second run.
func TestRandomSegmentationsPreserveMassOnEveryRow(t *testing.T) {
	var teacherRaw, studentRaw []string
	for _, a := range "abc" {
		teacherRaw = append(teacherRaw, string(a))
		studentRaw = append(studentRaw, string(a))
		for _, b := range "abc" {
			teacherRaw = append(teacherRaw, string(a)+string(b))
			for _, c := range "abc" {
				teacherRaw = append(teacherRaw, string(a)+string(b)+string(c))
			}
		}
	}
	studentRaw = append(studentRaw, "ab", "ca", "bca", "abca", "cabc", "bb")
	teacher, student := vocab(t, teacherRaw...), vocab(t, studentRaw...)
	b := newBridge(t, teacher, student)
	rng := rand.New(rand.NewPCG(3, 5))
	segment := func(text string, raw []string) []string {
		var out []string
		for len(text) > 0 {
			var fits []string
			for _, r := range raw {
				if strings.HasPrefix(text, r) {
					fits = append(fits, r)
				}
			}
			pick := fits[rng.IntN(len(fits))]
			out = append(out, pick)
			text = text[len(pick):]
		}
		return out
	}
	seen := map[bridge.Relation]int{}
	for trial := 0; trial < 300; trial++ {
		var text strings.Builder
		for range 4 + rng.IntN(12) {
			text.WriteByte("abc"[rng.IntN(3)])
		}
		teacherSeq := segment(text.String(), teacherRaw)
		studentSeq := segment(text.String(), studentRaw)
		dists := make([]bridge.Distribution, len(teacherSeq))
		for j, realized := range teacherSeq {
			var d bridge.Distribution
			for token := range teacherRaw {
				if teacherRaw[token] == realized || rng.IntN(4) == 0 {
					p := rng.Float64() / float64(len(teacherRaw))
					d.Probabilities = append(d.Probabilities, bridge.Probability{Token: int64(token), Probability: p})
					d.RetainedMass += p
				}
			}
			dists[j] = d
		}
		studentIDs, teacherIDs := tokens(t, student, studentSeq...), tokens(t, teacher, teacherSeq...)
		first := rows(t, b, studentIDs, teacherIDs, dists, bridge.RowOptions{})
		for _, row := range first {
			seen[row.Relation]++
			if row.Target == nil {
				continue
			}
			expectPartition(t, *row.Target)
			lower := row.Relation == bridge.RelationSpanning || row.Relation == bridge.RelationMidSpanning
			if (row.Target.Bound == bridge.BoundLowerBound) != lower {
				t.Errorf("trial %d row %d: %s with bound %s", trial, row.Position, row.Relation, row.Target.Bound)
			}
		}
		if second := rows(t, b, studentIDs, teacherIDs, dists, bridge.RowOptions{}); !reflect.DeepEqual(first, second) {
			t.Fatalf("trial %d: two runs differ", trial)
		}
	}
	for _, relation := range []bridge.Relation{bridge.RelationOneToOne, bridge.RelationHead, bridge.RelationInterior, bridge.RelationSpanning, bridge.RelationMidSpanning, bridge.RelationExcluded} {
		if seen[relation] == 0 {
			t.Errorf("no %s row was generated; the property did not cover it (seen %v)", relation, seen)
		}
	}
}
