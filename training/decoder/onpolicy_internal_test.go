package decoder

import (
	"errors"
	"math"
	"testing"
)

const onPolicyVocabulary = 5

// onPolicyFixture holds two supervised rows and the unscored causal row. The
// realized tokens are 3, outside the first top-k, and 1, inside the second.
// Under Clip 2 the first token's importance weight is truncated (q/mu is about
// 8.8) and the second's is not (about 1.16).
func onPolicyFixture() ([]float32, []int64, OnPolicyTarget) {
	logits := []float32{
		0.4, -0.3, 1.1, 0.2, -0.8,
		-0.7, 0.9, 0.1, 0.5, 0.3,
		9, 8, 7, 6, 5,
	}
	targets := []int64{3, 1}
	target := OnPolicyTarget{
		Positions: []FusionTeacherPosition{
			{RetainedMass: 0.85, TopK: []FusionTokenProbability{{2, 0.60}, {1, 0.25}}},
			{RetainedMass: 0.90, TopK: []FusionTokenProbability{{1, 0.50}, {4, 0.30}, {0, 0.10}}},
		},
		TeacherLogProbabilities:  []float64{math.Log(0.10), math.Log(0.50)},
		BehaviorLogProbabilities: []float64{math.Log(0.02), math.Log(0.30)},
	}
	return logits, targets, target
}

func onPolicyObjectives() map[string]OnPolicyObjective {
	return map[string]OnPolicyObjective{
		"fkl-complement":     {Mode: OnPolicyForwardComplement},
		"jsd-complement 0.1": {Mode: OnPolicyJSDComplement, Beta: 0.1},
		"jsd-complement 0.5": {Mode: OnPolicyJSDComplement, Beta: 0.5},
		"jsd-complement 0.9": {Mode: OnPolicyJSDComplement, Beta: 0.9},
		"rkl-sampled":        {Mode: OnPolicyReverseSampled, Clip: 2},
	}
}

func widen(values []float32) []float64 {
	out := make([]float64, len(values))
	for i, value := range values {
		out[i] = float64(value)
	}
	return out
}

func softmax64(row []float64) []float64 {
	maximum := row[0]
	for _, value := range row[1:] {
		maximum = math.Max(maximum, value)
	}
	out := make([]float64, len(row))
	sum := 0.0
	for i, value := range row {
		out[i] = math.Exp(value - maximum)
		sum += out[i]
	}
	for i := range out {
		out[i] /= sum
	}
	return out
}

// manualCells builds the teacher and student cell masses of one row the plain
// way: the student's complement is 1 - sum_K q.
func manualCells(logits []float64, row int, position FusionTeacherPosition) (teacher, student []float64) {
	q := softmax64(logits[row*onPolicyVocabulary : (row+1)*onPolicyVocabulary])
	retained, covered := 0.0, 0.0
	for _, item := range position.TopK {
		teacher = append(teacher, item.Probability)
		student = append(student, q[item.TokenID])
		retained += item.Probability
		covered += q[item.TokenID]
	}
	if len(position.TopK) < onPolicyVocabulary {
		teacher = append(teacher, 1-retained)
		student = append(student, 1-covered)
	}
	return teacher, student
}

func manualForwardComplement(logits []float64, target OnPolicyTarget) float64 {
	total := 0.0
	for row, position := range target.Positions {
		teacher, student := manualCells(logits, row, position)
		for c := range teacher {
			if teacher[c] != 0 {
				total -= teacher[c] * math.Log(student[c])
			}
		}
	}
	return total / float64(len(target.Positions))
}

func manualJSD(logits []float64, target OnPolicyTarget, beta float64) float64 {
	total := 0.0
	for row, position := range target.Positions {
		teacher, student := manualCells(logits, row, position)
		for c := range teacher {
			mixture := beta*teacher[c] + (1-beta)*student[c]
			if teacher[c] != 0 {
				total += beta * teacher[c] * math.Log(teacher[c]/mixture)
			}
			total += (1 - beta) * student[c] * math.Log(student[c]/mixture)
		}
	}
	return total / float64(len(target.Positions))
}

// manualCellKL is KL(first||second) over the cells, summed over rows and
// divided by their number; forward selects KL(P||Q), otherwise KL(Q||P).
func manualCellKL(logits []float64, target OnPolicyTarget, forward bool) float64 {
	total := 0.0
	for row, position := range target.Positions {
		teacher, student := manualCells(logits, row, position)
		first, second := student, teacher
		if forward {
			first, second = teacher, student
		}
		for c := range first {
			if first[c] != 0 {
				total += first[c] * math.Log(first[c]/second[c])
			}
		}
	}
	return total / float64(len(target.Positions))
}

// manualSurrogate is (1/n) sum_t coefficient_t log q(y_t), the function whose
// gradient rkl-sampled writes when w_t a_t is held at coefficient_t.
func manualSurrogate(logits []float64, targets []int64, coefficients []float64) float64 {
	total := 0.0
	for row, token := range targets {
		q := softmax64(logits[row*onPolicyVocabulary : (row+1)*onPolicyVocabulary])
		total += coefficients[row] * math.Log(q[token])
	}
	return total / float64(len(targets))
}

// manualReverseCoefficients returns w_t a_t and w_t from a plain log-softmax.
func manualReverseCoefficients(logits []float64, targets []int64, target OnPolicyTarget, clip float64) (coefficients, weights []float64) {
	for row, token := range targets {
		q := softmax64(logits[row*onPolicyVocabulary : (row+1)*onPolicyVocabulary])
		logQ := math.Log(q[token])
		weight := math.Min(math.Exp(logQ-target.BehaviorLogProbabilities[row]), clip)
		coefficients = append(coefficients, weight*(logQ-target.TeacherLogProbabilities[row]))
		weights = append(weights, weight)
	}
	return coefficients, weights
}

func TestOnPolicyCotangentMatchesIndependentLossesAndFiniteDifferences(t *testing.T) {
	for name, objective := range onPolicyObjectives() {
		t.Run(name, func(t *testing.T) {
			original, targets, target := onPolicyFixture()
			base := widen(original)
			got := append([]float32(nil), original...)
			stats, err := onPolicyCotangent(got, onPolicyVocabulary, targets, 1, objective, target)
			if err != nil {
				t.Fatal(err)
			}
			var loss func([]float64) float64
			var wantLoss float64
			switch objective.Mode {
			case OnPolicyForwardComplement:
				loss = func(z []float64) float64 { return manualForwardComplement(z, target) }
				wantLoss = loss(base)
			case OnPolicyJSDComplement:
				loss = func(z []float64) float64 { return manualJSD(z, target, objective.Beta) }
				wantLoss = loss(base)
			default:
				coefficients, weights := manualReverseCoefficients(base, targets, target, objective.Clip)
				loss = func(z []float64) float64 { return manualSurrogate(z, targets, coefficients) }
				wantLoss = (coefficients[0] + coefficients[1]) / 2
				for row, weight := range weights {
					if math.Abs(stats.weights[row]-weight) > 1e-12 {
						t.Fatalf("row %d weight %.17g != %.17g", row, stats.weights[row], weight)
					}
				}
			}
			if math.Abs(stats.loss-wantLoss) > 1e-12 {
				t.Fatalf("loss %.17g != independent %.17g", stats.loss, wantLoss)
			}
			for _, value := range got[len(targets)*onPolicyVocabulary:] {
				if value != 0 {
					t.Fatal("unscored final row is not zero")
				}
			}
			const step = 1e-5
			for coordinate := 0; coordinate < len(targets)*onPolicyVocabulary; coordinate++ {
				plus := append([]float64(nil), base...)
				minus := append([]float64(nil), base...)
				plus[coordinate] += step
				minus[coordinate] -= step
				central := (loss(plus) - loss(minus)) / (2 * step)
				analytic := float64(got[coordinate])
				if math.Abs(analytic-central) > 1e-8+1e-6*math.Abs(central) {
					t.Fatalf("coordinate %d analytic %.12g central %.12g", coordinate, analytic, central)
				}
			}
		})
	}
}

func TestOnPolicyForwardComplementGradientIsTheCellFormula(t *testing.T) {
	original, targets, target := onPolicyFixture()
	got := append([]float32(nil), original...)
	if _, err := onPolicyCotangent(got, onPolicyVocabulary, targets, 1, OnPolicyObjective{Mode: OnPolicyForwardComplement}, target); err != nil {
		t.Fatal(err)
	}
	base := widen(original)
	for row, position := range target.Positions {
		q := softmax64(base[row*onPolicyVocabulary : (row+1)*onPolicyVocabulary])
		teacher := make(map[int64]float64)
		retained, covered := 0.0, 0.0
		for _, item := range position.TopK {
			teacher[item.TokenID] = item.Probability
			retained += item.Probability
			covered += q[item.TokenID]
		}
		residual, complement := 1-retained, 1-covered
		for j := 0; j < onPolicyVocabulary; j++ {
			want := q[j] * (1 - residual/complement)
			if p, ok := teacher[int64(j)]; ok {
				want = q[j] - p
			}
			want /= float64(len(targets))
			if math.Abs(float64(got[row*onPolicyVocabulary+j])-want) > 1e-8+1e-7*math.Abs(want) {
				t.Fatalf("row %d token %d gradient %.9g != %.9g", row, j, got[row*onPolicyVocabulary+j], want)
			}
		}
	}
}

func TestOnPolicyCellModesPutNoMassOnTheRealizedToken(t *testing.T) {
	original, targets, target := onPolicyFixture()
	other := []int64{0, 4}
	for name, objective := range onPolicyObjectives() {
		t.Run(name, func(t *testing.T) {
			first, second := target, target
			if objective.Mode == OnPolicyReverseSampled {
				second.TeacherLogProbabilities = []float64{math.Log(0.10), math.Log(0.30)}
			} else {
				first.TeacherLogProbabilities, first.BehaviorLogProbabilities = nil, nil
				second = first
			}
			a := append([]float32(nil), original...)
			b := append([]float32(nil), original...)
			x, err := onPolicyCotangent(a, onPolicyVocabulary, targets, 1, objective, first)
			if err != nil {
				t.Fatal(err)
			}
			y, err := onPolicyCotangent(b, onPolicyVocabulary, other, 1, objective, second)
			if err != nil {
				t.Fatal(err)
			}
			same := math.Float64bits(x.loss) == math.Float64bits(y.loss)
			for i := range a {
				same = same && math.Float32bits(a[i]) == math.Float32bits(b[i])
			}
			if objective.Mode == OnPolicyReverseSampled {
				if same {
					t.Fatal("rkl-sampled did not read the realized tokens")
				}
			} else if !same {
				t.Fatal("the realized tokens changed a cell objective")
			}
		})
	}
}

func TestOnPolicyJSDApproachesForwardAndReverseKLAtItsEnds(t *testing.T) {
	original, targets, target := onPolicyFixture()
	base := widen(original)
	forward := append([]float32(nil), original...)
	if _, err := onPolicyCotangent(forward, onPolicyVocabulary, targets, 1, OnPolicyObjective{Mode: OnPolicyForwardComplement}, target); err != nil {
		t.Fatal(err)
	}
	near := func(beta float64) ([]float32, float64) {
		got := append([]float32(nil), original...)
		stats, err := onPolicyCotangent(got, onPolicyVocabulary, targets, 1, OnPolicyObjective{Mode: OnPolicyJSDComplement, Beta: beta}, target)
		if err != nil {
			t.Fatal(err)
		}
		return got, stats.loss
	}

	const small = 1e-6
	low, lowLoss := near(small)
	if kl := manualCellKL(base, target, true); math.Abs(lowLoss/small-kl) > 1e-5 {
		t.Fatalf("JSD/beta %.12g does not approach KL(P||Q) %.12g", lowLoss/small, kl)
	}
	for i := range low[:len(targets)*onPolicyVocabulary] {
		if math.Abs(float64(low[i])/small-float64(forward[i])) > 1e-5 {
			t.Fatalf("coordinate %d JSD gradient/beta %.9g != fkl %.9g", i, float64(low[i])/small, forward[i])
		}
	}

	beta := 1 - small
	rest := 1 - beta
	high, highLoss := near(beta)
	reverse := func(z []float64) float64 { return manualCellKL(z, target, false) }
	if kl := reverse(base); math.Abs(highLoss/rest-kl) > 1e-5 {
		t.Fatalf("JSD/(1-beta) %.12g does not approach KL(Q||P) %.12g", highLoss/rest, kl)
	}
	const step = 1e-5
	for i := range high[:len(targets)*onPolicyVocabulary] {
		plus := append([]float64(nil), base...)
		minus := append([]float64(nil), base...)
		plus[i] += step
		minus[i] -= step
		central := (reverse(plus) - reverse(minus)) / (2 * step)
		if math.Abs(float64(high[i])/rest-central) > 1e-5 {
			t.Fatalf("coordinate %d JSD gradient/(1-beta) %.9g != reverse KL %.9g", i, float64(high[i])/rest, central)
		}
	}
}

func TestOnPolicyReverseSampledTruncatesTheImportanceWeight(t *testing.T) {
	original, targets, target := onPolicyFixture()
	clipped := append([]float32(nil), original...)
	loose := append([]float32(nil), original...)
	a, err := onPolicyCotangent(clipped, onPolicyVocabulary, targets, 1, OnPolicyObjective{Mode: OnPolicyReverseSampled, Clip: 2}, target)
	if err != nil {
		t.Fatal(err)
	}
	b, err := onPolicyCotangent(loose, onPolicyVocabulary, targets, 1, OnPolicyObjective{Mode: OnPolicyReverseSampled, Clip: 100}, target)
	if err != nil {
		t.Fatal(err)
	}
	ratios := []float64{
		math.Exp(a.logQ[0] - target.BehaviorLogProbabilities[0]),
		math.Exp(a.logQ[1] - target.BehaviorLogProbabilities[1]),
	}
	if !(ratios[0] > 2 && ratios[0] < 100 && ratios[1] < 2) {
		t.Fatalf("fixture ratios %v do not straddle the clip", ratios)
	}
	if a.weights[0] != 2 || a.weights[1] != ratios[1] || b.weights[0] != ratios[0] || b.weights[1] != ratios[1] {
		t.Fatalf("weights %v and %v are not min(q/mu, clip) of %v", a.weights, b.weights, ratios)
	}
	for column := 0; column < onPolicyVocabulary; column++ {
		first, second := column, onPolicyVocabulary+column
		if clipped[first] == loose[first] {
			t.Fatalf("clip did not change the truncated row at %d", column)
		}
		if math.Float32bits(clipped[second]) != math.Float32bits(loose[second]) {
			t.Fatalf("clip changed the untruncated row at %d", column)
		}
	}
}

func TestOnPolicyCotangentRefusesInvalidInputsBeforeWriting(t *testing.T) {
	forward := OnPolicyObjective{Mode: OnPolicyForwardComplement}
	jsd := OnPolicyObjective{Mode: OnPolicyJSDComplement, Beta: 0.5}
	reverse := OnPolicyObjective{Mode: OnPolicyReverseSampled, Clip: 2}
	every := []OnPolicyObjective{forward, jsd, reverse}
	position := func(retained float64, items ...FusionTokenProbability) FusionTeacherPosition {
		return FusionTeacherPosition{RetainedMass: retained, TopK: items}
	}
	type input struct {
		logits    []float32
		targets   []int64
		target    OnPolicyTarget
		lossScale float64
	}
	cases := []struct {
		name       string
		objectives []OnPolicyObjective
		mutate     func(*input)
	}{
		{"retained mass above one", every, func(in *input) {
			in.target.Positions[0] = position(1.2, FusionTokenProbability{2, 0.7}, FusionTokenProbability{1, 0.5})
		}},
		{"top-k mass above one", every, func(in *input) {
			in.target.Positions[0] = position(1, FusionTokenProbability{2, 0.7}, FusionTokenProbability{1, 0.5})
		}},
		{"probability above one", every, func(in *input) { in.target.Positions[0] = position(1, FusionTokenProbability{2, 1.5}) }},
		{"empty top-k", every, func(in *input) { in.target.Positions[0] = position(0) }},
		{"duplicate token", every, func(in *input) {
			in.target.Positions[0] = position(0.85, FusionTokenProbability{2, 0.6}, FusionTokenProbability{2, 0.25})
		}},
		{"token outside vocabulary", every, func(in *input) { in.target.Positions[0].TopK[0].TokenID = onPolicyVocabulary }},
		{"negative token", every, func(in *input) { in.target.Positions[0].TopK[0].TokenID = -1 }},
		{"negative probability", every, func(in *input) {
			in.target.Positions[0] = position(0.85, FusionTokenProbability{2, 0.9}, FusionTokenProbability{1, -0.05})
		}},
		{"NaN probability", every, func(in *input) { in.target.Positions[0].TopK[0].Probability = math.NaN() }},
		{"NaN retained mass", every, func(in *input) { in.target.Positions[0].RetainedMass = math.NaN() }},
		{"retained mass differs from top-k", every, func(in *input) { in.target.Positions[0].RetainedMass = 0.86 }},
		{"missing position", every, func(in *input) { in.target.Positions = in.target.Positions[:1] }},
		{"residual beyond a full top-k", every, func(in *input) {
			in.target.Positions[1] = position(0.9, FusionTokenProbability{0, 0.1}, FusionTokenProbability{1, 0.5},
				FusionTokenProbability{2, 0.1}, FusionTokenProbability{3, 0.1}, FusionTokenProbability{4, 0.1})
		}},
		{"missing behavior log probabilities", []OnPolicyObjective{reverse}, func(in *input) { in.target.BehaviorLogProbabilities = nil }},
		{"missing teacher log probabilities", []OnPolicyObjective{reverse}, func(in *input) { in.target.TeacherLogProbabilities = nil }},
		{"short behavior series", every, func(in *input) { in.target.BehaviorLogProbabilities = in.target.BehaviorLogProbabilities[:1] }},
		{"NaN behavior log probability", every, func(in *input) { in.target.BehaviorLogProbabilities[1] = math.NaN() }},
		{"NaN teacher log probability", every, func(in *input) { in.target.TeacherLogProbabilities[1] = math.NaN() }},
		{"positive log probability", every, func(in *input) { in.target.BehaviorLogProbabilities[0] = 1e-9 }},
		{"negative infinite log probability", every, func(in *input) { in.target.TeacherLogProbabilities[0] = math.Inf(-1) }},
		{"teacher log probability differs from its top-k", every, func(in *input) { in.target.TeacherLogProbabilities[1] = math.Log(0.4) }},
		{"teacher log probability exceeds the residual", every, func(in *input) { in.target.TeacherLogProbabilities[0] = math.Log(0.3) }},
		{"NaN logit", every, func(in *input) { in.logits[3] = float32(math.NaN()) }},
		{"infinite logit", every, func(in *input) { in.logits[6] = float32(math.Inf(1)) }},
		{"target outside vocabulary", every, func(in *input) { in.targets[0] = onPolicyVocabulary }},
		{"short logits", every, func(in *input) { in.logits = in.logits[:10] }},
		{"zero loss scale", every, func(in *input) { in.lossScale = 0 }},
		{"NaN loss scale", every, func(in *input) { in.lossScale = math.NaN() }},
	}
	for _, c := range cases {
		for _, objective := range c.objectives {
			t.Run(c.name+"/"+string(objective.Mode), func(t *testing.T) {
				logits, targets, target := onPolicyFixture()
				in := input{logits: logits, targets: targets, target: target, lossScale: 1}
				c.mutate(&in)
				before := append([]float32(nil), in.logits...)
				if _, err := onPolicyCotangent(in.logits, onPolicyVocabulary, in.targets, in.lossScale, objective, in.target); !errors.Is(err, ErrCompletionStep) {
					t.Fatalf("invalid on-policy input accepted: %v", err)
				}
				for i := range before {
					if math.Float32bits(before[i]) != math.Float32bits(in.logits[i]) {
						t.Fatal("a refused input wrote the logits")
					}
				}
			})
		}
	}

	objectives := []struct {
		name      string
		objective OnPolicyObjective
	}{
		{"unknown mode", OnPolicyObjective{Mode: "kl"}},
		{"fkl with beta", OnPolicyObjective{Mode: OnPolicyForwardComplement, Beta: 0.5}},
		{"fkl with clip", OnPolicyObjective{Mode: OnPolicyForwardComplement, Clip: 2}},
		{"jsd beta zero", OnPolicyObjective{Mode: OnPolicyJSDComplement}},
		{"jsd beta one", OnPolicyObjective{Mode: OnPolicyJSDComplement, Beta: 1}},
		{"jsd beta negative", OnPolicyObjective{Mode: OnPolicyJSDComplement, Beta: -0.1}},
		{"jsd beta NaN", OnPolicyObjective{Mode: OnPolicyJSDComplement, Beta: math.NaN()}},
		{"jsd with clip", OnPolicyObjective{Mode: OnPolicyJSDComplement, Beta: 0.5, Clip: 2}},
		{"rkl clip below one", OnPolicyObjective{Mode: OnPolicyReverseSampled, Clip: 0.5}},
		{"rkl clip infinite", OnPolicyObjective{Mode: OnPolicyReverseSampled, Clip: math.Inf(1)}},
		{"rkl clip NaN", OnPolicyObjective{Mode: OnPolicyReverseSampled, Clip: math.NaN()}},
		{"rkl with beta", OnPolicyObjective{Mode: OnPolicyReverseSampled, Beta: 0.5, Clip: 2}},
	}
	for _, c := range objectives {
		t.Run(c.name, func(t *testing.T) {
			if err := c.objective.Validate(); !errors.Is(err, ErrCompletionStep) {
				t.Fatalf("invalid objective accepted: %v", err)
			}
			logits, targets, target := onPolicyFixture()
			if _, err := onPolicyCotangent(logits, onPolicyVocabulary, targets, 1, c.objective, target); !errors.Is(err, ErrCompletionStep) {
				t.Fatalf("invalid objective reached the cotangent: %v", err)
			}
		})
	}

	// The cell modes do not require the sampled-token series.
	for _, objective := range []OnPolicyObjective{forward, jsd} {
		logits, targets, target := onPolicyFixture()
		target.TeacherLogProbabilities, target.BehaviorLogProbabilities = nil, nil
		if _, err := onPolicyCotangent(logits, onPolicyVocabulary, targets, 1, objective, target); err != nil {
			t.Fatalf("%s refused a target without sampled-token series: %v", objective.Mode, err)
		}
	}
}

func TestOnPolicyCotangentIsDeterministicAndLossScaleOnlyScalesTheGradient(t *testing.T) {
	for name, objective := range onPolicyObjectives() {
		t.Run(name, func(t *testing.T) {
			original, targets, target := onPolicyFixture()
			first := append([]float32(nil), original...)
			again := append([]float32(nil), original...)
			scaled := append([]float32(nil), original...)
			a, err := onPolicyCotangent(first, onPolicyVocabulary, targets, 1, objective, target)
			if err != nil {
				t.Fatal(err)
			}
			b, err := onPolicyCotangent(again, onPolicyVocabulary, targets, 1, objective, target)
			if err != nil {
				t.Fatal(err)
			}
			c, err := onPolicyCotangent(scaled, onPolicyVocabulary, targets, 2, objective, target)
			if err != nil {
				t.Fatal(err)
			}
			if math.Float64bits(a.loss) != math.Float64bits(b.loss) || math.Float64bits(a.loss) != math.Float64bits(c.loss) {
				t.Fatalf("loss %.17g, repeated %.17g, scaled %.17g", a.loss, b.loss, c.loss)
			}
			for row := range targets {
				if math.Float64bits(a.logQ[row]) != math.Float64bits(b.logQ[row]) {
					t.Fatal("repeated log q differs")
				}
				if a.weights != nil && math.Float64bits(a.weights[row]) != math.Float64bits(b.weights[row]) {
					t.Fatal("repeated importance weight differs")
				}
			}
			for i := range first {
				if math.Float32bits(first[i]) != math.Float32bits(again[i]) {
					t.Fatalf("repeated cotangent %d differs", i)
				}
				if scaled[i] != 2*first[i] {
					t.Fatalf("scaled cotangent %d %.9g != 2 * %.9g", i, scaled[i], first[i])
				}
			}
		})
	}
}

// With every top-k summing to exactly one and a retained mass of one, the
// complement carries nothing and fkl-complement is fusionCotangent with one
// teacher of weight one, over a full and over a partial top-k.
func TestOnPolicyForwardComplementIsFusionAtFullRetainedMass(t *testing.T) {
	cases := map[string][]FusionTeacherPosition{
		"top-k covers the vocabulary": {
			{RetainedMass: 1, TopK: []FusionTokenProbability{{2, 0.5}, {1, 0.25}, {3, 0.125}, {0, 0.0625}, {4, 0.0625}}},
			{RetainedMass: 1, TopK: []FusionTokenProbability{{1, 0.5}, {4, 0.25}, {0, 0.125}, {2, 0.0625}, {3, 0.0625}}},
		},
		"top-k holds all the mass": {
			{RetainedMass: 1, TopK: []FusionTokenProbability{{2, 0.75}, {1, 0.25}}},
			{RetainedMass: 1, TopK: []FusionTokenProbability{{4, 0.5}, {1, 0.5}}},
		},
	}
	for name, positions := range cases {
		t.Run(name, func(t *testing.T) {
			original, targets, _ := onPolicyFixture()
			onPolicy := append([]float32(nil), original...)
			fused := append([]float32(nil), original...)
			for _, scale := range []float64{1, 3} {
				copy(onPolicy, original)
				copy(fused, original)
				got, err := onPolicyCotangent(onPolicy, onPolicyVocabulary, targets, scale, OnPolicyObjective{Mode: OnPolicyForwardComplement}, OnPolicyTarget{Positions: positions})
				if err != nil {
					t.Fatal(err)
				}
				want, err := fusionCotangent(fused, onPolicyVocabulary, targets, scale, []FusionTeacher{{Name: "teacher", Weight: 1, Positions: positions}})
				if err != nil {
					t.Fatal(err)
				}
				if math.Float64bits(got.loss) != math.Float64bits(want.loss) {
					t.Fatalf("scale %g loss %.17g != fusion %.17g", scale, got.loss, want.loss)
				}
				for i := range fused {
					if math.Float32bits(onPolicy[i]) != math.Float32bits(fused[i]) {
						t.Fatalf("scale %g cotangent %d %.9g != fusion %.9g", scale, i, onPolicy[i], fused[i])
					}
				}
			}
		})
	}
}

func TestTokenLogProbabilitiesAreTheCotangentsReading(t *testing.T) {
	original, targets, target := onPolicyFixture()
	reading, err := tokenLogProbabilities(original, onPolicyVocabulary, targets)
	if err != nil {
		t.Fatal(err)
	}
	for name, objective := range onPolicyObjectives() {
		stats, err := onPolicyCotangent(append([]float32(nil), original...), onPolicyVocabulary, targets, 1, objective, target)
		if err != nil {
			t.Fatal(err)
		}
		for row := range targets {
			if math.Float64bits(stats.logQ[row]) != math.Float64bits(reading[row]) {
				t.Fatalf("%s row %d log q %.17g != reading %.17g", name, row, stats.logQ[row], reading[row])
			}
		}
	}
	nll, err := completionNLL(original, onPolicyVocabulary, targets, nil)
	if err != nil {
		t.Fatal(err)
	}
	if mean := -(reading[0] + reading[1]) / 2; math.Abs(mean-nll) > 1e-15 {
		t.Fatalf("mean negative log q %.17g != completion NLL %.17g", mean, nll)
	}
	for name, call := range map[string]func() error{
		"short logits":  func() error { _, err := tokenLogProbabilities(original[:10], onPolicyVocabulary, targets); return err },
		"no targets":    func() error { _, err := tokenLogProbabilities(original, onPolicyVocabulary, nil); return err },
		"target beyond": func() error { _, err := tokenLogProbabilities(original, onPolicyVocabulary, []int64{0, 5}); return err },
	} {
		if !errors.Is(call(), ErrCompletionStep) {
			t.Fatalf("%s accepted", name)
		}
	}
}
