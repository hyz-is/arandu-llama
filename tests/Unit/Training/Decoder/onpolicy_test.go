//go:build libtorch && cgo

package decoder_test

import (
	"context"
	"errors"
	"math"
	"reflect"
	"testing"

	"github.com/tayi-ai/arandu-llama/training/decoder"
	"github.com/tayi-ai/arandu-llama/training/torch"
)

// onPolicyTarget reads the fixture's completion, tokens 4 and 2 after a
// one-token prompt: 4 is inside the first top-k, 2 outside the second.
func onPolicyTarget() decoder.OnPolicyTarget {
	return decoder.OnPolicyTarget{
		Positions: []decoder.FusionTeacherPosition{
			{RetainedMass: 0.75, TopK: []decoder.FusionTokenProbability{{TokenID: 4, Probability: 0.5}, {TokenID: 1, Probability: 0.25}}},
			{RetainedMass: 0.8, TopK: []decoder.FusionTokenProbability{{TokenID: 0, Probability: 0.4}, {TokenID: 5, Probability: 0.25}, {TokenID: 3, Probability: 0.15}}},
		},
		TeacherLogProbabilities:  []float64{math.Log(0.5), math.Log(0.15)},
		BehaviorLogProbabilities: []float64{math.Log(0.02), math.Log(0.2)},
	}
}

func onPolicyModes() map[string]decoder.OnPolicyObjective {
	return map[string]decoder.OnPolicyObjective{
		"fkl-complement": {Mode: decoder.OnPolicyForwardComplement},
		"jsd-complement": {Mode: decoder.OnPolicyJSDComplement, Beta: 0.3},
		"rkl-sampled":    {Mode: decoder.OnPolicyReverseSampled, Clip: 2},
	}
}

// completionRows returns the supervised rows' plain softmax from a fresh
// forward over the fixture's current parameters.
func completionRows(t *testing.T, f *fixture, limits decoder.Limits, rows int) [][]float64 {
	t.Helper()
	snapshot, err := f.model.Forward(context.Background(), f.tokens, limits)
	if err != nil {
		t.Fatal(err)
	}
	defer snapshot.Close()
	info, err := snapshot.Logits.Info()
	if err != nil {
		t.Fatal(err)
	}
	logits := read(t, snapshot.Logits)
	vocabulary := int(info.Shape[2])
	out := make([][]float64, rows)
	for row := range out {
		maximum := float64(logits[row*vocabulary])
		for _, value := range logits[row*vocabulary : (row+1)*vocabulary] {
			maximum = math.Max(maximum, float64(value))
		}
		sum := 0.0
		out[row] = make([]float64, vocabulary)
		for column := range out[row] {
			out[row][column] = math.Exp(float64(logits[row*vocabulary+column]) - maximum)
			sum += out[row][column]
		}
		for column := range out[row] {
			out[row][column] /= sum
		}
	}
	return out
}

// manualOnPolicyLoss is the mode's mean loss written the plain way, with the
// student's complement read as 1 - sum_K q. Under rkl-sampled it is the
// surrogate (1/n) sum_t coefficient_t log q(y_t), whose gradient the mode
// writes when w_t a_t is held at coefficient_t.
func manualOnPolicyLoss(t *testing.T, f *fixture, limits decoder.Limits, objective decoder.OnPolicyObjective, target decoder.OnPolicyTarget, coefficients []float64) float64 {
	t.Helper()
	q := completionRows(t, f, limits, len(target.Positions))
	total := 0.0
	for row, position := range target.Positions {
		if objective.Mode == decoder.OnPolicyReverseSampled {
			total += coefficients[row] * math.Log(q[row][f.tokens[1+row]])
			continue
		}
		var teacher, student []float64
		retained, covered := 0.0, 0.0
		for _, item := range position.TopK {
			teacher = append(teacher, item.Probability)
			student = append(student, q[row][item.TokenID])
			retained += item.Probability
			covered += q[row][item.TokenID]
		}
		teacher = append(teacher, 1-retained)
		student = append(student, 1-covered)
		for c := range teacher {
			if objective.Mode == decoder.OnPolicyForwardComplement {
				total -= teacher[c] * math.Log(student[c])
				continue
			}
			mixture := objective.Beta*teacher[c] + (1-objective.Beta)*student[c]
			total += objective.Beta*teacher[c]*math.Log(teacher[c]/mixture) + (1-objective.Beta)*student[c]*math.Log(student[c]/mixture)
		}
	}
	return total / float64(len(target.Positions))
}

func TestOnPolicyCompletionGradientMatchesFiniteDifferencesInEveryMode(t *testing.T) {
	for name, objective := range onPolicyModes() {
		t.Run(name, func(t *testing.T) {
			f := newFixture(t, torch.Float32)
			before := baseHash(t, f)
			limits := f.limits
			limits.LogitRows = 3
			target := onPolicyTarget()
			result, err := decoder.OnPolicyCompletionGradient(context.Background(), f.model, f.tokens, 1, limits, 1, objective, target)
			if err != nil {
				t.Fatal(err)
			}
			if result.Tokens != 2 || len(result.Gradients) != 32 || len(result.StudentLogProbabilities) != 2 {
				t.Fatalf("on-policy result tokens=%d gradients=%d log q=%d", result.Tokens, len(result.Gradients), len(result.StudentLogProbabilities))
			}
			var coefficients []float64
			want := 0.0
			if objective.Mode == decoder.OnPolicyReverseSampled {
				q := completionRows(t, f, limits, 2)
				for row := range q {
					logQ := math.Log(q[row][f.tokens[1+row]])
					ratio := math.Exp(logQ - target.BehaviorLogProbabilities[row])
					weight := math.Min(ratio, objective.Clip)
					if (row == 0) != (ratio > objective.Clip) {
						t.Fatalf("fixture row %d ratio %.9g does not sit on its side of the clip", row, ratio)
					}
					if math.Abs(result.ImportanceWeights[row]-weight) > 1e-12 || math.Abs(result.StudentLogProbabilities[row]-logQ) > 1e-12 {
						t.Fatalf("row %d weight %.17g log q %.17g, want %.17g %.17g", row, result.ImportanceWeights[row], result.StudentLogProbabilities[row], weight, logQ)
					}
					coefficients = append(coefficients, weight*(logQ-target.TeacherLogProbabilities[row]))
				}
				want = (coefficients[0] + coefficients[1]) / 2
			} else {
				if result.ImportanceWeights != nil {
					t.Fatal("a cell mode reported importance weights")
				}
				want = manualOnPolicyLoss(t, f, limits, objective, target, nil)
			}
			if math.Abs(result.Loss-want) > 1e-10 {
				t.Fatalf("loss %.12g != manual %.12g", result.Loss, want)
			}
			for _, index := range []int{0, 31} {
				gradient := result.Gradients[index].ValuesF32
				coordinate := 0
				for i := range gradient {
					if math.Abs(float64(gradient[i])) > math.Abs(float64(gradient[coordinate])) {
						coordinate = i
					}
				}
				pointer := parameter(f.model.Layers[index/4*4+3].Adapter, index%4)
				original := *pointer
				info, err := original.Info()
				if err != nil {
					t.Fatal(err)
				}
				baseline := read(t, original)
				losses, positions := [2]float64{}, [2]float32{}
				for direction, sign := range []float32{1, -1} {
					perturbed := append([]float32(nil), baseline...)
					perturbed[coordinate] += sign * 0.005
					positions[direction] = perturbed[coordinate]
					replacement, err := torch.FromFloat32(perturbed, info.Shape, torch.CPUDevice(), true)
					if err != nil {
						t.Fatal(err)
					}
					*pointer = replacement
					losses[direction] = manualOnPolicyLoss(t, f, limits, objective, target, coefficients)
					*pointer = original
					if err := replacement.Close(); err != nil {
						t.Fatal(err)
					}
				}
				central := (losses[0] - losses[1]) / float64(positions[0]-positions[1])
				analytic := float64(gradient[coordinate])
				tolerance := 1e-6 + 0.003*math.Abs(analytic)
				if analytic == 0 || math.Abs(analytic-central) > tolerance {
					t.Fatalf("%s[%d] analytic=%.9g central=%.9g tolerance=%.9g", result.Gradients[index].Name, coordinate, analytic, central, tolerance)
				}
				t.Logf("%s[%d] analytic=%.9g central=%.9g", result.Gradients[index].Name, coordinate, analytic, central)
			}
			if baseHash(t, f) != before {
				t.Fatal("on-policy gradient changed the frozen base")
			}
		})
	}
}

// With every top-k summing to exactly one and a retained mass of one,
// fkl-complement is FusionCompletionGradient with one teacher of weight one,
// alpha one, bit for bit in the loss and in every gradient.
func TestOnPolicyForwardComplementIsFusionAtFullRetainedMassAndAlphaOne(t *testing.T) {
	cases := map[string][]decoder.FusionTeacherPosition{
		"top-k covers the vocabulary": {
			{RetainedMass: 1, TopK: []decoder.FusionTokenProbability{{TokenID: 4, Probability: 0.5}, {TokenID: 1, Probability: 0.25}, {TokenID: 0, Probability: 0.125}, {TokenID: 2, Probability: 0.0625}, {TokenID: 3, Probability: 0.03125}, {TokenID: 5, Probability: 0.03125}}},
			{RetainedMass: 1, TopK: []decoder.FusionTokenProbability{{TokenID: 2, Probability: 0.25}, {TokenID: 5, Probability: 0.25}, {TokenID: 0, Probability: 0.25}, {TokenID: 1, Probability: 0.125}, {TokenID: 3, Probability: 0.0625}, {TokenID: 4, Probability: 0.0625}}},
		},
		"top-k holds all the mass": {
			{RetainedMass: 1, TopK: []decoder.FusionTokenProbability{{TokenID: 4, Probability: 0.75}, {TokenID: 1, Probability: 0.25}}},
			{RetainedMass: 1, TopK: []decoder.FusionTokenProbability{{TokenID: 3, Probability: 0.5}, {TokenID: 0, Probability: 0.5}}},
		},
	}
	for name, positions := range cases {
		t.Run(name, func(t *testing.T) {
			f := newFixture(t, torch.Float32)
			limits := f.limits
			limits.LogitRows = 3
			for _, scale := range []float64{1, 2} {
				got, err := decoder.OnPolicyCompletionGradient(context.Background(), f.model, f.tokens, 1, limits, scale,
					decoder.OnPolicyObjective{Mode: decoder.OnPolicyForwardComplement}, decoder.OnPolicyTarget{Positions: positions})
				if err != nil {
					t.Fatal(err)
				}
				want, err := decoder.FusionCompletionGradient(context.Background(), f.model, f.tokens, 1, limits, scale,
					[]decoder.FusionTeacher{{Name: "teacher", Weight: 1, Positions: positions}})
				if err != nil {
					t.Fatal(err)
				}
				if math.Float64bits(got.Loss) != math.Float64bits(want.Loss) || got.Tokens != want.Tokens {
					t.Fatalf("scale %g loss %.17g tokens %d != fusion %.17g tokens %d", scale, got.Loss, got.Tokens, want.Loss, want.Tokens)
				}
				if len(got.Gradients) != len(want.Gradients) {
					t.Fatalf("scale %g gradient count %d != fusion %d", scale, len(got.Gradients), len(want.Gradients))
				}
				for i := range want.Gradients {
					if got.Gradients[i].Name != want.Gradients[i].Name || !reflect.DeepEqual(got.Gradients[i].Shape, want.Gradients[i].Shape) {
						t.Fatalf("scale %g gradient %d identity differs", scale, i)
					}
					for j, value := range want.Gradients[i].ValuesF32 {
						if math.Float32bits(got.Gradients[i].ValuesF32[j]) != math.Float32bits(value) {
							t.Fatalf("scale %g %s[%d] %.9g != fusion %.9g", scale, want.Gradients[i].Name, j, got.Gradients[i].ValuesF32[j], value)
						}
					}
				}
			}
		})
	}
}

func TestCompletionTokenLogProbabilitiesAreTheGradientsReadingAndChangeNothing(t *testing.T) {
	f := newFixture(t, torch.Float32)
	ctx := context.Background()
	limits := f.limits
	limits.LogitRows = 3
	before := baseHash(t, f)
	adapters := adapterValues(t, f)
	reading, err := decoder.CompletionTokenLogProbabilities(ctx, f.model, f.tokens, 1, limits)
	if err != nil {
		t.Fatal(err)
	}
	if len(reading) != 2 {
		t.Fatalf("reading has %d rows", len(reading))
	}
	for name, objective := range onPolicyModes() {
		result, err := decoder.OnPolicyCompletionGradient(ctx, f.model, f.tokens, 1, limits, 1, objective, onPolicyTarget())
		if err != nil {
			t.Fatal(err)
		}
		for row := range reading {
			if math.Float64bits(result.StudentLogProbabilities[row]) != math.Float64bits(reading[row]) {
				t.Fatalf("%s row %d log q %.17g != reading %.17g", name, row, result.StudentLogProbabilities[row], reading[row])
			}
		}
	}
	loss, err := decoder.CompletionLoss(ctx, f.model, f.tokens, 1, limits)
	if err != nil {
		t.Fatal(err)
	}
	if mean := -(reading[0] + reading[1]) / 2; math.Abs(mean-loss) > 1e-12 {
		t.Fatalf("mean negative log q %.17g != completion loss %.17g", mean, loss)
	}
	if baseHash(t, f) != before {
		t.Fatal("the reading changed the frozen base")
	}
	after := adapterValues(t, f)
	for i := range adapters {
		for j := range adapters[i] {
			if math.Float32bits(after[i][j]) != math.Float32bits(adapters[i][j]) {
				t.Fatal("the reading changed an adapter parameter")
			}
		}
	}
	bare := newFixture(t, torch.Float32)
	for i := range bare.model.Layers {
		bare.model.Layers[i].Adapter = nil
	}
	if _, err := decoder.CompletionTokenLogProbabilities(ctx, bare.model, bare.tokens, 1, limits); err != nil {
		t.Fatalf("a model without adapters has no reading: %v", err)
	}
}

func TestOnPolicyCompletionGradientIsDeterministic(t *testing.T) {
	f := newFixture(t, torch.Float32)
	limits := f.limits
	limits.LogitRows = 3
	for name, objective := range onPolicyModes() {
		first, err := decoder.OnPolicyCompletionGradient(context.Background(), f.model, f.tokens, 1, limits, 1, objective, onPolicyTarget())
		if err != nil {
			t.Fatal(err)
		}
		second, err := decoder.OnPolicyCompletionGradient(context.Background(), f.model, f.tokens, 1, limits, 1, objective, onPolicyTarget())
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(first, second) {
			t.Fatalf("%s: two calls over the same inputs differ", name)
		}
	}
}

func TestOnPolicyCompletionGradientRefusesInvalidInputsAndCancellation(t *testing.T) {
	f := newFixture(t, torch.Float32)
	limits := f.limits
	limits.LogitRows = 3
	forward := decoder.OnPolicyObjective{Mode: decoder.OnPolicyForwardComplement}
	reverse := decoder.OnPolicyObjective{Mode: decoder.OnPolicyReverseSampled, Clip: 2}
	call := func(model *decoder.TextModel, prompt int, limits decoder.Limits, scale float64, objective decoder.OnPolicyObjective, mutate func(*decoder.OnPolicyTarget)) error {
		target := onPolicyTarget()
		if mutate != nil {
			mutate(&target)
		}
		_, err := decoder.OnPolicyCompletionGradient(context.Background(), model, f.tokens, prompt, limits, scale, objective, target)
		return err
	}
	wrongRows := limits
	wrongRows.LogitRows = 2
	aboveOne := func(x *decoder.OnPolicyTarget) {
		x.Positions[0].RetainedMass, x.Positions[0].TopK[1].Probability = 1.25, 0.75
	}
	for name, err := range map[string]error{
		"nil model":            call(nil, 1, limits, 1, forward, nil),
		"empty prompt":         call(f.model, 0, limits, 1, forward, nil),
		"wrong logit rows":     call(f.model, 1, wrongRows, 1, forward, nil),
		"zero loss scale":      call(f.model, 1, limits, 0, forward, nil),
		"jsd beta one":         call(f.model, 1, limits, 1, decoder.OnPolicyObjective{Mode: decoder.OnPolicyJSDComplement, Beta: 1}, nil),
		"rkl clip below one":   call(f.model, 1, limits, 1, decoder.OnPolicyObjective{Mode: decoder.OnPolicyReverseSampled, Clip: 0.5}, nil),
		"missing behavior":     call(f.model, 1, limits, 1, reverse, func(x *decoder.OnPolicyTarget) { x.BehaviorLogProbabilities = nil }),
		"missing teacher":      call(f.model, 1, limits, 1, reverse, func(x *decoder.OnPolicyTarget) { x.TeacherLogProbabilities = nil }),
		"NaN behavior":         call(f.model, 1, limits, 1, reverse, func(x *decoder.OnPolicyTarget) { x.BehaviorLogProbabilities[0] = math.NaN() }),
		"empty top-k":          call(f.model, 1, limits, 1, forward, func(x *decoder.OnPolicyTarget) { x.Positions[0] = decoder.FusionTeacherPosition{} }),
		"mass above one":       call(f.model, 1, limits, 1, forward, aboveOne),
		"NaN teacher mass":     call(f.model, 1, limits, 1, forward, func(x *decoder.OnPolicyTarget) { x.Positions[1].TopK[0].Probability = math.NaN() }),
		"missing position":     call(f.model, 1, limits, 1, forward, func(x *decoder.OnPolicyTarget) { x.Positions = x.Positions[:1] }),
		"teacher log disagree": call(f.model, 1, limits, 1, reverse, func(x *decoder.OnPolicyTarget) { x.TeacherLogProbabilities[0] = math.Log(0.4) }),
	} {
		t.Run(name, func(t *testing.T) {
			if !errors.Is(err, decoder.ErrCompletionStep) {
				t.Fatalf("invalid on-policy request accepted: %v", err)
			}
		})
	}
	for name, err := range map[string]error{
		"reading nil model": func() error {
			_, err := decoder.CompletionTokenLogProbabilities(context.Background(), nil, f.tokens, 1, limits)
			return err
		}(),
		"reading wrong logit rows": func() error {
			_, err := decoder.CompletionTokenLogProbabilities(context.Background(), f.model, f.tokens, 1, wrongRows)
			return err
		}(),
		"reading empty completion": func() error {
			_, err := decoder.CompletionTokenLogProbabilities(context.Background(), f.model, f.tokens, len(f.tokens), limits)
			return err
		}(),
	} {
		t.Run(name, func(t *testing.T) {
			if !errors.Is(err, decoder.ErrCompletionStep) {
				t.Fatalf("invalid reading accepted: %v", err)
			}
		})
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	result, err := decoder.OnPolicyCompletionGradient(ctx, f.model, f.tokens, 1, limits, 1, forward, onPolicyTarget())
	if !errors.Is(err, context.Canceled) || !reflect.DeepEqual(result, decoder.OnPolicyCompletionResult{}) {
		t.Fatalf("canceled on-policy result=%+v err=%v", result, err)
	}
	reading, err := decoder.CompletionTokenLogProbabilities(ctx, f.model, f.tokens, 1, limits)
	if !errors.Is(err, context.Canceled) || reading != nil {
		t.Fatalf("canceled reading=%v err=%v", reading, err)
	}
}
