package bridge_test

import (
	"errors"
	"math"
	"math/rand/v2"
	"reflect"
	"testing"

	"github.com/tayi-ai/arandu-llama/training/bridge"
)

func TestIdenticalVocabulariesReturnTheDistributionUnchanged(t *testing.T) {
	v := vocab(t, "a", "at", "ate", "bit", " the", "\n\n", "!control:<|end|>")
	b := newBridge(t, v, v)
	for token := int64(0); token < 6; token++ {
		if b.Phi(token) != token {
			t.Errorf("phi(%d) = %d, want itself", token, b.Phi(token))
		}
	}
	d := bridge.Distribution{Probabilities: []bridge.Probability{
		{Token: 4, Probability: 0.4}, {Token: 0, Probability: 0.25}, {Token: 5, Probability: 0.2}, {Token: 2, Probability: 0.05},
	}, RetainedMass: 0.9}
	got, err := b.Marginalize(bridge.Query{Teacher: d})
	if err != nil {
		t.Fatal(err)
	}
	want := []bridge.Probability{{Token: 0, Probability: 0.25}, {Token: 2, Probability: 0.05}, {Token: 4, Probability: 0.4}, {Token: 5, Probability: 0.2}}
	if !reflect.DeepEqual(got.Cells, want) {
		t.Errorf("cells %v, want the distribution itself %v", got.Cells, want)
	}
	expectNear(t, "uncovered", got.Uncovered, 0.1)
	if got.Residual != 0 || got.Special != 0 || got.Stop != 0 || got.Bound != bridge.BoundExact || got.Support != 1 {
		t.Errorf("target %+v", got)
	}
	expectPartition(t, got)
}

// TestThePaperWorkedExample reproduces the byte-prefix marginal of the
// paper's worked example and its routing figure: teacher tokens absent from
// the student go to their longest student-token prefix, and bytes no student
// token prefixes go to the residual.
func TestThePaperWorkedExample(t *testing.T) {
	student := vocab(t, "a", "at", "ate", "bit", "the", "an")
	teacher := vocab(t, "at", "ate", "a", "bit", "the", "then", "they", "an", "zq")
	b := newBridge(t, teacher, student)

	got, err := b.Marginalize(bridge.Query{Teacher: dist(t, teacher, map[string]float64{"at": 0.5, "ate": 0.2, "a": 0.2, "bit": 0.1})})
	if err != nil {
		t.Fatal(err)
	}
	expectCells(t, student, got, map[string]float64{"a": 0.2, "at": 0.5, "ate": 0.2, "bit": 0.1})
	expectPartition(t, got)

	got, err = b.Marginalize(bridge.Query{Teacher: dist(t, teacher, map[string]float64{
		"the": 0.38, "then": 0.22, "they": 0.12, "a": 0.10, "an": 0.08, "zq": 0.10,
	})})
	if err != nil {
		t.Fatal(err)
	}
	expectCells(t, student, got, map[string]float64{"the": 0.72, "a": 0.10, "an": 0.08})
	expectNear(t, "residual", got.Residual, 0.10)
	expectPartition(t, got)
}

func TestMassIsPreservedWithATruncatedTopK(t *testing.T) {
	student := vocab(t, "a", "b", "ab")
	teacher := vocab(t, "a", "ab", "abc", "c", "!control:<|channel|>", "!control:<|return|>")
	returnID, channelID := int64(5), int64(4)
	b := newBridge(t, teacher, student, returnID)
	d := bridge.Distribution{Probabilities: []bridge.Probability{
		{Token: 0, Probability: 0.1}, {Token: 1, Probability: 0.2}, {Token: 2, Probability: 0.15},
		{Token: 3, Probability: 0.05}, {Token: channelID, Probability: 0.03}, {Token: returnID, Probability: 0.07},
	}, RetainedMass: 0.6}
	got, err := b.Marginalize(bridge.Query{Teacher: d})
	if err != nil {
		t.Fatal(err)
	}
	expectCells(t, student, got, map[string]float64{"a": 0.1, "ab": 0.35})
	expectNear(t, "residual", got.Residual, 0.05)
	expectNear(t, "stop", got.Stop, 0.07)
	expectNear(t, "special", got.Special, 0.03)
	expectNear(t, "uncovered", got.Uncovered, 0.4)
	expectNear(t, "complement", got.Complement(), 0.48)
	expectPartition(t, got)
}

// TestOneTeacherTokenSplitIntoTwoStudentTokens covers the 1:N relation: at
// the head the scatter routes "12" to "1"; inside the token the target is
// conditioned on the "1" already emitted.
func TestOneTeacherTokenSplitIntoTwoStudentTokens(t *testing.T) {
	student := vocab(t, "1", "2", "3", "7")
	teacher := vocab(t, "12", "13", "1", "7")
	b := newBridge(t, teacher, student)
	full := dist(t, teacher, map[string]float64{"12": 0.6, "13": 0.1, "1": 0.1, "7": 0.2})

	head, err := b.Marginalize(bridge.Query{Teacher: full})
	if err != nil {
		t.Fatal(err)
	}
	expectCells(t, student, head, map[string]float64{"1": 0.8, "7": 0.2})
	expectPartition(t, head)

	interior, err := b.Marginalize(bridge.Query{Prefix: []byte("1"), Limit: 1, Teacher: full})
	if err != nil {
		t.Fatal(err)
	}
	expectCells(t, student, interior, map[string]float64{"2": 0.6 / 0.8, "3": 0.1 / 0.8})
	expectNear(t, "residual", interior.Residual, 0.1/0.8)
	expectNear(t, "support", interior.Support, 0.8)
	expectPartition(t, interior)

	// With 10% of the teacher mass outside the top-k, the uncovered mass may
	// all continue the prefix, so it joins the denominator and every cell
	// stays a lower bound of the full-distribution target.
	truncated := dist(t, teacher, map[string]float64{"12": 0.6, "13": 0.1, "1": 0.1, "7": 0.1})
	interior, err = b.Marginalize(bridge.Query{Prefix: []byte("1"), Limit: 1, Teacher: truncated})
	if err != nil {
		t.Fatal(err)
	}
	expectCells(t, student, interior, map[string]float64{"2": 0.6 / 0.9, "3": 0.1 / 0.9})
	expectNear(t, "residual", interior.Residual, 0.1/0.9)
	expectNear(t, "uncovered", interior.Uncovered, 0.1/0.9)
	expectPartition(t, interior)

	if _, err := b.Marginalize(bridge.Query{Prefix: []byte("9"), Teacher: full}); !errors.Is(err, bridge.ErrNoSupport) {
		t.Errorf("a prefix the teacher never continues: %v, want ErrNoSupport", err)
	}
}

func TestTheLimitCapsTheStudentTokenAtTheChunkBoundary(t *testing.T) {
	student := vocab(t, "b", "bc", "bcd", "x")
	teacher := vocab(t, "abcd", "abc", "bcd")
	b := newBridge(t, teacher, student)
	d := dist(t, teacher, map[string]float64{"abcd": 0.5, "abc": 0.25, "bcd": 0.25})
	uncapped, err := b.Marginalize(bridge.Query{Prefix: []byte("a"), Teacher: d})
	if err != nil {
		t.Fatal(err)
	}
	expectCells(t, student, uncapped, map[string]float64{"bcd": 0.5 / 0.75, "bc": 0.25 / 0.75})
	capped, err := b.Marginalize(bridge.Query{Prefix: []byte("a"), Limit: 2, Teacher: d})
	if err != nil {
		t.Fatal(err)
	}
	expectCells(t, student, capped, map[string]float64{"bc": 1})
	boundary, err := b.Marginalize(bridge.Query{Limit: 2, Teacher: d})
	if err != nil {
		t.Fatal(err)
	}
	expectCells(t, student, boundary, map[string]float64{"bc": 0.25})
	expectNear(t, "residual", boundary.Residual, 0.75)
}

// TestSpaceAndByteFallback routes a SentencePiece teacher, whose spaces are
// U+2581 and whose unknown bytes are <0xNN>, onto a byte-level student.
func TestSpaceAndByteFallback(t *testing.T) {
	teacher, err := bridge.NewVocabulary(bridge.SchemeSentencePiece, []bridge.TokenSpec{
		{Text: "▁the", Kind: bridge.KindNormal},
		{Text: "▁", Kind: bridge.KindNormal},
		{Text: "▁▁▁▁", Kind: bridge.KindNormal},
		{Text: "<0xC3>", Kind: bridge.KindByte},
		{Text: "<0xA9>", Kind: bridge.KindByte},
		{Text: "é", Kind: bridge.KindNormal},
		{Text: "<0x0A>", Kind: bridge.KindByte},
	})
	if err != nil {
		t.Fatal(err)
	}
	student := vocab(t, " ", "  ", " the", "\xc3", "\xa9", "é", "\n")
	b := newBridge(t, teacher, student)
	want := map[int64]string{0: " the", 1: " ", 2: "  ", 3: "\xc3", 4: "\xa9", 5: "é", 6: "\n"}
	for token, raw := range want {
		if got := b.Phi(token); got != id(t, student, raw) {
			t.Errorf("phi(%q) = %d, want the student token for %q", teacher.Text(token), got, raw)
		}
	}
	d := bridge.Distribution{Probabilities: []bridge.Probability{
		{Token: 0, Probability: 0.5}, {Token: 2, Probability: 0.2}, {Token: 3, Probability: 0.2}, {Token: 5, Probability: 0.1},
	}, RetainedMass: 1}
	got, err := b.Marginalize(bridge.Query{Teacher: d})
	if err != nil {
		t.Fatal(err)
	}
	expectCells(t, student, got, map[string]float64{" the": 0.5, "  ": 0.2, "\xc3": 0.2, "é": 0.1})
	expectPartition(t, got)

	// Inside a byte-fallback character: after <0xC3>, only the teacher tokens
	// beginning with that byte count.
	inside, err := b.Marginalize(bridge.Query{Prefix: []byte{0xc3}, Teacher: d})
	if err != nil {
		t.Fatal(err)
	}
	expectCells(t, student, inside, map[string]float64{"\xa9": 0.1 / 0.3})
	expectNear(t, "residual", inside.Residual, 0.2/0.3)
	expectPartition(t, inside)
}

func TestSpecialTokensStayOutOfTheByteStream(t *testing.T) {
	student := vocab(t, "a", "<", "!control:<|im_end|>")
	teacher := vocab(t, "a", "!control:<|return|>", "!control:<|channel|>", "!user:<|channel>")
	returnID := id(t, teacher, "a") + 1
	b := newBridge(t, teacher, student, returnID)
	if b.Phi(returnID) != bridge.NoToken || b.Phi(2) != bridge.NoToken {
		t.Error("a byte-less teacher token was routed to a student token")
	}
	if b.Phi(3) != id(t, student, "<") {
		t.Errorf("a user-defined token is routed by its literal bytes: phi = %d", b.Phi(3))
	}
	got, err := b.Marginalize(bridge.Query{Teacher: bridge.Distribution{Probabilities: []bridge.Probability{
		{Token: 0, Probability: 0.5}, {Token: returnID, Probability: 0.3}, {Token: 2, Probability: 0.15}, {Token: 3, Probability: 0.05},
	}, RetainedMass: 1}})
	if err != nil {
		t.Fatal(err)
	}
	expectCells(t, student, got, map[string]float64{"a": 0.5, "<": 0.05})
	expectNear(t, "stop", got.Stop, 0.3)
	expectNear(t, "special", got.Special, 0.15)
	expectPartition(t, got)

	if _, err := bridge.New(teacher, student, []int64{99}); !errors.Is(err, bridge.ErrQuery) {
		t.Errorf("a stop outside the vocabulary was accepted: %v", err)
	}
}

func TestInvalidDistributionsAreRefused(t *testing.T) {
	v := vocab(t, "a", "b")
	b := newBridge(t, v, v)
	cases := map[string]bridge.Distribution{
		"out of range": {Probabilities: []bridge.Probability{{Token: 2, Probability: 0.5}}, RetainedMass: 0.5},
		"negative":     {Probabilities: []bridge.Probability{{Token: 0, Probability: -0.1}}, RetainedMass: -0.1},
		"nan":          {Probabilities: []bridge.Probability{{Token: 0, Probability: math.NaN()}}, RetainedMass: 0},
		"repeated":     {Probabilities: []bridge.Probability{{Token: 0, Probability: 0.2}, {Token: 0, Probability: 0.2}}, RetainedMass: 0.4},
		"above one":    {Probabilities: []bridge.Probability{{Token: 0, Probability: 0.7}, {Token: 1, Probability: 0.7}}, RetainedMass: 1.4},
		"mass differs": {Probabilities: []bridge.Probability{{Token: 0, Probability: 0.7}}, RetainedMass: 0.8},
	}
	for name, d := range cases {
		if _, err := b.Marginalize(bridge.Query{Teacher: d}); !errors.Is(err, bridge.ErrQuery) {
			t.Errorf("%s: %v, want ErrQuery", name, err)
		}
	}
	if _, err := b.Marginalize(bridge.Query{Limit: -1}); !errors.Is(err, bridge.ErrQuery) {
		t.Errorf("negative limit: %v", err)
	}
}

func TestTheTargetIsDeterministicWhateverTheInputOrder(t *testing.T) {
	raw := []string{"a", "ab", "abc", "b", "bc", "c", "ca", "cab", "bca", "abca"}
	teacher := vocab(t, raw...)
	student := vocab(t, "a", "b", "c", "ab", "ca")
	b := newBridge(t, teacher, student)
	rng := rand.New(rand.NewPCG(7, 11))
	for trial := 0; trial < 200; trial++ {
		var d bridge.Distribution
		for token := range raw {
			if rng.IntN(3) == 0 {
				continue
			}
			p := rng.Float64() / float64(len(raw))
			d.Probabilities = append(d.Probabilities, bridge.Probability{Token: int64(token), Probability: p})
		}
		for _, p := range d.Probabilities {
			d.RetainedMass += p.Probability
		}
		prefix := []byte(nil)
		if trial%2 == 1 {
			prefix = []byte("ab")[:1+rng.IntN(2)]
		}
		first, err := b.Marginalize(bridge.Query{Prefix: prefix, Teacher: d})
		if errors.Is(err, bridge.ErrNoSupport) {
			continue
		}
		if err != nil {
			t.Fatal(err)
		}
		expectPartition(t, first)
		shuffled := d
		shuffled.Probabilities = append([]bridge.Probability(nil), d.Probabilities...)
		rng.Shuffle(len(shuffled.Probabilities), func(i, j int) {
			shuffled.Probabilities[i], shuffled.Probabilities[j] = shuffled.Probabilities[j], shuffled.Probabilities[i]
		})
		second, err := b.Marginalize(bridge.Query{Prefix: prefix, Teacher: shuffled})
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(first, second) {
			t.Fatalf("trial %d: the same distribution in another order gave\n%+v\nand\n%+v", trial, first, second)
		}
	}
}
