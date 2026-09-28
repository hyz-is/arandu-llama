package rollout_test

import (
	"math"
	"sort"
	"testing"

	llama "github.com/tayi-ai/arandu-llama"
)

// The arithmetic of one rollout position, on logit rows whose answer is known.
// DrawFromLogits runs the native code SampleRollout runs; no model is loaded.

// tolerance bounds the difference between the native log mu and the one
// computed here. Both subtract the same float64 values and differ only in the
// last bit of exp and log between Go's math and the C library, which is about
// 1e-16; 1e-12 leaves room without letting a wrong formula through, since the
// smallest effect the cases below distinguish is above 1e-2.
const tolerance = 1e-12

func sampling(temperature float32, topK int, topP, minP float32) llama.RolloutSampling {
	return llama.RolloutSampling{
		Temperature: llama.Float32(temperature),
		TopK:        llama.Int(topK),
		TopP:        llama.Float32(topP),
		MinP:        llama.Float32(minP),
	}
}

// logits returns float32 logits whose softmax is probabilities.
func logits(probabilities ...float64) []float32 {
	out := make([]float32, len(probabilities))
	for i, p := range probabilities {
		out[i] = float32(math.Log(p))
	}
	return out
}

// synthetic is a deterministic row with no dominant token, so a nucleus spans
// many tokens and the native partial sort has to extend its prefix repeatedly.
func synthetic(n int) []float32 {
	out := make([]float32, n)
	for i := range out {
		x := float64(i)
		out[i] = float32(6*math.Sin(0.731*x+0.3) + 2.5*math.Cos(1.917*x) + 1e-4*x)
	}
	return out
}

// reference is the definition documented on RolloutSampling, written again
// here from the documentation with a full sort instead of the native partial
// one.
func reference(row []float32, temperature float64, topK int, topP, minP float64) []float64 {
	n := len(row)
	z := make([]float64, n)
	best := math.Inf(-1)
	for i, l := range row {
		z[i] = float64(l) / temperature
		best = math.Max(best, z[i])
	}
	w := make([]float64, n)
	for i := range z {
		w[i] = math.Exp(z[i] - best)
	}
	order := make([]int, n)
	for i := range order {
		order[i] = i
	}
	sort.Slice(order, func(a, b int) bool {
		return z[order[a]] > z[order[b]] || z[order[a]] == z[order[b]] && order[a] < order[b]
	})
	keep := make([]bool, n)
	for i := range keep {
		keep[i] = true
	}
	limit := n
	prefix := func(length int) {
		limit = length
		for i := range keep {
			keep[i] = false
		}
		for _, id := range order[:length] {
			keep[id] = true
		}
	}
	if topK > 0 && topK < n {
		prefix(topK)
	}
	if topP < 1 {
		total := 0.0
		for i := range w {
			if keep[i] {
				total += w[i]
			}
		}
		reached, taken := 0.0, 0
		for taken < limit {
			reached += w[order[taken]]
			taken++
			if reached >= topP*total {
				break
			}
		}
		prefix(taken)
	}
	if minP > 0 {
		for i := range keep {
			keep[i] = keep[i] && w[i] >= minP
		}
	}
	total := 0.0
	for i := range w {
		if keep[i] {
			total += w[i]
		}
	}
	out := make([]float64, n)
	for i := range out {
		out[i] = math.Inf(-1)
		if keep[i] {
			out[i] = z[i] - best - math.Log(total)
		}
	}
	return out
}

func draw(t *testing.T, row []float32, s llama.RolloutSampling, u float64) *llama.LogitDraw {
	t.Helper()
	result, err := llama.DrawFromLogits(row, s, u)
	if err != nil {
		t.Fatalf("DrawFromLogits: %v", err)
	}
	if len(result.LogProbabilities) != len(row) {
		t.Fatalf("%d log probabilities for %d logits", len(result.LogProbabilities), len(row))
	}
	if result.LogMu != result.LogProbabilities[result.Token] {
		t.Fatalf("LogMu %v is not the log probability %v of the drawn token %d",
			result.LogMu, result.LogProbabilities[result.Token], result.Token)
	}
	return result
}

func kept(values []float64) []int {
	var ids []int
	for i, value := range values {
		if !math.IsInf(value, -1) {
			ids = append(ids, i)
		}
	}
	return ids
}

func sameIDs(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func expectClose(t *testing.T, label string, got, want []float64) {
	t.Helper()
	if g, w := kept(got), kept(want); !sameIDs(g, w) {
		first := 0
		for first < len(g) && first < len(w) && g[first] == w[first] {
			first++
		}
		if len(g) <= 8 && len(w) <= 8 {
			t.Fatalf("%s: kept %v, want %v", label, g, w)
		}
		t.Fatalf("%s: kept %d tokens, want %d; the sets part at the %d-th kept id", label, len(g), len(w), first)
	}
	for i := range got {
		if math.IsInf(want[i], -1) {
			continue
		}
		if math.Abs(got[i]-want[i]) > tolerance {
			t.Fatalf("%s: log mu of token %d is %.17g, want %.17g (|diff| %.3g > %g)",
				label, i, got[i], want[i], math.Abs(got[i]-want[i]), tolerance)
		}
	}
}

func expectMassOne(t *testing.T, label string, values []float64) {
	t.Helper()
	total := 0.0
	for _, value := range values {
		total += math.Exp(value)
	}
	if math.Abs(total-1) > 1e-12 {
		t.Fatalf("%s: exp(log mu) sums to %.17g, not one", label, total)
	}
}

// At temperature one with every filter off, log mu is the log-softmax of the
// logits in float64, over the whole vocabulary -- here at the size of the
// Ornith vocabulary.
func TestLogMuWithoutFiltersIsTheLogSoftmaxOfTheLogits(t *testing.T) {
	t.Parallel()

	row := synthetic(248320)
	row[1000] = 30 // a head, so the tail sits far below it
	result := draw(t, row, sampling(1, 0, 1, 0), 0.5)

	best := math.Inf(-1)
	for _, l := range row {
		best = math.Max(best, float64(l))
	}
	sum := 0.0
	for _, l := range row {
		sum += math.Exp(float64(l) - best)
	}
	want := make([]float64, len(row))
	for i, l := range row {
		want[i] = float64(l) - best - math.Log(sum)
	}
	expectClose(t, "log-softmax", result.LogProbabilities, want)
	expectMassOne(t, "log-softmax", result.LogProbabilities)
}

// Each filter, and the order they compose in, on rows whose kept sets are
// computed by hand. The comments give the arithmetic.
func TestEachFilterKeepsWhatItsDefinitionSays(t *testing.T) {
	t.Parallel()

	row := logits(0.5, 0.3, 0.15, 0.05)
	logOf := func(ids []int, z func(int) float64) []float64 {
		out := []float64{math.Inf(-1), math.Inf(-1), math.Inf(-1), math.Inf(-1)}
		sum := 0.0
		for _, id := range ids {
			sum += math.Exp(z(id))
		}
		for _, id := range ids {
			out[id] = z(id) - math.Log(sum)
		}
		return out
	}
	raw := func(id int) float64 { return float64(row[id]) }
	tempered := func(id int) float64 { return float64(row[id]) / 3 }

	for _, c := range []struct {
		name     string
		sampling llama.RolloutSampling
		keep     []int
		z        func(int) float64
	}{
		{"no filter", sampling(1, 0, 1, 0), []int{0, 1, 2, 3}, raw},
		{"top-k 2", sampling(1, 2, 1, 0), []int{0, 1}, raw},
		{"top-k at the vocabulary is off", sampling(1, 4, 1, 0), []int{0, 1, 2, 3}, raw},
		{"top-k above the vocabulary is off", sampling(1, 10, 1, 0), []int{0, 1, 2, 3}, raw},
		// 0.5, then 0.8 >= 0.75
		{"top-p 0.75", sampling(1, 0, 0.75, 0), []int{0, 1}, raw},
		// 0.8 < 0.85, then 0.95
		{"top-p 0.85", sampling(1, 0, 0.85, 0), []int{0, 1, 2}, raw},
		// ratios to the largest: 1, 0.6, 0.3, 0.1
		{"min-p 0.5", sampling(1, 0, 1, 0.5), []int{0, 1}, raw},
		{"min-p 0.25", sampling(1, 0, 1, 0.25), []int{0, 1, 2}, raw},
		// top-k 2 renormalised is 0.625, 0.375, and 0.625 >= 0.6 already. Over
		// the whole vocabulary 0.5 < 0.6 would have kept two.
		{"top-p reads the renormalised top-k", sampling(1, 2, 0.6, 0), []int{0}, raw},
		// At T = 3 the probabilities are p^(1/3) normalised: 0.336, 0.283,
		// 0.225, 0.156, cumulative 0.336, 0.619, 0.844. Filtering the raw
		// logits first, as llama.cpp's chain does, would keep two.
		{"temperature before top-p", sampling(3, 0, 0.75, 0), []int{0, 1, 2}, tempered},
		// Tempered ratios 1, 0.843, 0.669, 0.464; raw ones would keep two.
		{"temperature before min-p", sampling(3, 0, 1, 0.5), []int{0, 1, 2}, tempered},
		// The shortest prefix wins: top-p would keep three, min-p two.
		{"top-p and min-p", sampling(1, 0, 0.85, 0.5), []int{0, 1}, raw},
		{"top-k, top-p and min-p", sampling(1, 3, 0.99, 0.25), []int{0, 1, 2}, raw},
	} {
		result := draw(t, row, c.sampling, 0.5)
		want := logOf(c.keep, c.z)
		expectClose(t, c.name, result.LogProbabilities, want)
		expectMassOne(t, c.name, result.LogProbabilities)
		temperature, topK, topP, minP := float64(*c.sampling.Temperature), *c.sampling.TopK, float64(*c.sampling.TopP), float64(*c.sampling.MinP)
		expectClose(t, c.name+" against the reference", result.LogProbabilities, reference(row, temperature, topK, topP, minP))
	}
}

// Equal logits are ordered by ascending token id, so the filters cut ties the
// same way on every platform.
func TestTiesAreCutByTokenID(t *testing.T) {
	t.Parallel()

	for _, c := range []struct {
		name     string
		row      []float32
		sampling llama.RolloutSampling
		keep     []int
	}{
		{"top-k over a tie", []float32{1, 1, 1, 0}, sampling(1, 2, 1, 0), []int{0, 1}},
		{"top-k over a tie after a lower id", []float32{0, 1, 1, 1}, sampling(1, 2, 1, 0), []int{1, 2}},
		// each tied token holds e/(3e+1) = 0.306: two reach 0.5
		{"top-p over a tie", []float32{1, 1, 1, 0}, sampling(1, 0, 0.5, 0), []int{0, 1}},
	} {
		result := draw(t, c.row, c.sampling, 0.5)
		if got := kept(result.LogProbabilities); !sameIDs(got, c.keep) {
			t.Errorf("%s: kept %v, want %v", c.name, got, c.keep)
		}
	}
}

// A nucleus of thousands of tokens, which the native side reaches by
// extending a sorted prefix several times, against the full sort of the
// reference; and a large top-k beside it.
func TestLargeFiltersMatchTheReference(t *testing.T) {
	t.Parallel()

	row := synthetic(20000)
	for _, c := range []struct {
		name                string
		temperature         float32
		topK                int
		topP, minP          float32
		smallestKept, below int
	}{
		// More than 64 kept: the native prefix starts at 64 and has to grow.
		{"nucleus 0.9", 1, 0, 0.9, 0, 65, 20000},
		{"nucleus 0.5 at T 0.7", 0.7, 0, 0.5, 0, 65, 20000},
		{"top-k 5000 then nucleus 0.95", 1.3, 5000, 0.95, 0, 65, 5000},
		{"min-p 0.01", 1, 0, 1, 0.01, 65, 20000},
	} {
		result := draw(t, row, sampling(c.temperature, c.topK, c.topP, c.minP), 0.25)
		n := len(kept(result.LogProbabilities))
		if n < c.smallestKept || n > c.below {
			t.Fatalf("%s: kept %d tokens, outside [%d, %d]; the case no longer exercises what it names",
				c.name, n, c.smallestKept, c.below)
		}
		expectClose(t, c.name, result.LogProbabilities,
			reference(row, float64(c.temperature), c.topK, float64(c.topP), float64(c.minP)))
		expectMassOne(t, c.name, result.LogProbabilities)
	}
}

// The draw is the inverse CDF over the kept tokens in ascending id.
func TestTheDrawIsTheInverseCDFInTokenOrder(t *testing.T) {
	t.Parallel()

	row := logits(0.25, 0.25, 0.5)
	below1 := math.Nextafter(1, 0)
	for _, c := range []struct {
		u     float64
		token int32
	}{
		{0, 0}, {0.2, 0}, {0.3, 1}, {0.49, 1}, {0.51, 2}, {0.9, 2}, {below1, 2},
	} {
		if got := draw(t, row, sampling(1, 0, 1, 0), c.u).Token; got != c.token {
			t.Errorf("u = %v drew %d, want %d", c.u, got, c.token)
		}
	}
	// Outside the kept set nothing is drawn, whatever u is.
	for _, u := range []float64{0, 0.3, below1} {
		result := draw(t, row, sampling(1, 1, 1, 0), u)
		if result.Token != 2 || result.LogMu != 0 {
			t.Errorf("top-k 1 at u = %v drew %d with log mu %v, want 2 with 0", u, result.Token, result.LogMu)
		}
	}
	// A token whose weight underflows to zero is never drawn, even at the top
	// of the uniform's range.
	if got := draw(t, []float32{0, -2000}, sampling(1, 0, 1, 0), below1).Token; got != 0 {
		t.Errorf("a zero-weight token was drawn: %d", got)
	}
}

// Drawn with the seeded uniforms many times, each token comes up about as
// often as exp(log mu) says. This ties log mu to the draw: a draw over a
// different distribution than the one log mu describes fails here.
func TestDrawFrequenciesFollowLogMu(t *testing.T) {
	t.Parallel()

	const draws = 200000
	uniforms, err := llama.RolloutUniforms(11, draws)
	if err != nil {
		t.Fatal(err)
	}
	row := logits(0.5, 0.3, 0.15, 0.05, 0.0001)
	for _, s := range []llama.RolloutSampling{sampling(1, 0, 1, 0), sampling(2, 0, 0.9, 0), sampling(0.8, 3, 1, 0)} {
		counts := make([]int, len(row))
		var probabilities []float64
		for _, u := range uniforms {
			result := draw(t, row, s, u)
			counts[result.Token]++
			probabilities = result.LogProbabilities
		}
		for id, count := range counts {
			p := math.Exp(probabilities[id])
			sigma := math.Sqrt(draws * p * (1 - p))
			if math.Abs(float64(count)-draws*p) > 5*sigma+1 {
				t.Errorf("T %v top-k %d top-p %v: token %d drawn %d times in %d, expected %.0f (sigma %.1f)",
					*s.Temperature, *s.TopK, *s.TopP, id, count, draws, draws*p, sigma)
			}
		}
	}
}

func TestTheDrawIsDeterministic(t *testing.T) {
	t.Parallel()

	row := synthetic(50000)
	s := sampling(0.9, 2000, 0.8, 0.001)
	first := draw(t, row, s, 0.37)
	second := draw(t, row, s, 0.37)
	if first.Token != second.Token || math.Float64bits(first.LogMu) != math.Float64bits(second.LogMu) {
		t.Fatalf("same inputs, different draws: %d %v and %d %v", first.Token, first.LogMu, second.Token, second.LogMu)
	}
	for i := range first.LogProbabilities {
		if math.Float64bits(first.LogProbabilities[i]) != math.Float64bits(second.LogProbabilities[i]) {
			t.Fatalf("same inputs, log mu of token %d differs: %v and %v", i, first.LogProbabilities[i], second.LogProbabilities[i])
		}
	}
	// The uniform picks the token, not the distribution.
	other := draw(t, row, s, 0.91)
	for i := range first.LogProbabilities {
		if math.Float64bits(first.LogProbabilities[i]) != math.Float64bits(other.LogProbabilities[i]) {
			t.Fatalf("the uniform moved log mu of token %d", i)
		}
	}
}

func TestRowsWithoutASoftmaxAreRefused(t *testing.T) {
	t.Parallel()

	for _, value := range []float32{float32(math.NaN()), float32(math.Inf(1)), float32(math.Inf(-1))} {
		if _, err := llama.DrawFromLogits([]float32{0, value, 1}, sampling(1, 0, 1, 0), 0.5); err == nil {
			t.Errorf("a row holding %v was accepted", value)
		}
	}
	if _, err := llama.DrawFromLogits(nil, sampling(1, 0, 1, 0), 0.5); err == nil {
		t.Error("an empty row was accepted")
	}
	for _, u := range []float64{-0.1, 1, math.NaN()} {
		if _, err := llama.DrawFromLogits([]float32{0, 1}, sampling(1, 0, 1, 0), u); err == nil {
			t.Errorf("the uniform %v was accepted", u)
		}
	}
}

// The uniform stream is mt19937_64 as the C++ standard specifies it, so a seed
// names the same rollout on the Mac and on a node. The standard fixes the
// 10000th output of the default seed, 5489, at 9981545732273789042.
func TestTheUniformsAreTheStandardGenerator(t *testing.T) {
	t.Parallel()

	uniforms, err := llama.RolloutUniforms(5489, 10000)
	if err != nil {
		t.Fatal(err)
	}
	want := float64(uint64(9981545732273789042)>>11) * 0x1p-53
	if math.Float64bits(uniforms[9999]) != math.Float64bits(want) {
		t.Fatalf("the 10000th uniform of seed 5489 is %.17g, want %.17g", uniforms[9999], want)
	}
	again, err := llama.RolloutUniforms(5489, 10000)
	if err != nil {
		t.Fatal(err)
	}
	for i := range uniforms {
		if math.Float64bits(uniforms[i]) != math.Float64bits(again[i]) {
			t.Fatalf("seed 5489 gave a different uniform at %d", i)
		}
		if !(uniforms[i] >= 0 && uniforms[i] < 1) {
			t.Fatalf("uniform %d is %v, outside [0, 1)", i, uniforms[i])
		}
	}
	other, err := llama.RolloutUniforms(5490, 3)
	if err != nil {
		t.Fatal(err)
	}
	if other[0] == uniforms[0] && other[1] == uniforms[1] {
		t.Fatal("two seeds gave the same stream")
	}
	if _, err := llama.RolloutUniforms(-1, 1); err == nil {
		t.Fatal("a negative seed was accepted")
	}
}
