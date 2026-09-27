package decoder

import (
	"errors"
	"fmt"
	"math"
	"testing"
)

func completionReferenceLoss(logits []float32, vocabulary int, targets []int64) float64 {
	total := 0.0
	for row, target := range targets {
		offset := row * vocabulary
		maximum := float64(logits[offset])
		for column := 1; column < vocabulary; column++ {
			maximum = math.Max(maximum, float64(logits[offset+column]))
		}
		sum := 0.0
		for column := 0; column < vocabulary; column++ {
			sum += math.Exp(float64(logits[offset+column]) - maximum)
		}
		total += maximum + math.Log(sum) - float64(logits[offset+int(target)])
	}
	return total / float64(len(targets))
}

func TestCompletionCotangentMatchesReferenceAndFiniteDifference(t *testing.T) {
	original := []float32{
		0.4, -0.3, 1.1, 0.2,
		-0.7, 0.9, 0.1, 0.5,
		9, 8, 7, 6,
	}
	targets := []int64{2, 1}
	got := append([]float32(nil), original...)
	loss, err := completionCotangent(got, 4, targets, 1)
	if err != nil {
		t.Fatal(err)
	}
	wantLoss := completionReferenceLoss(original, 4, targets)
	if math.Abs(loss-wantLoss) > 1e-12 {
		t.Fatalf("loss %.12g != reference %.12g", loss, wantLoss)
	}
	for row := range targets {
		sum := 0.0
		for column := 0; column < 4; column++ {
			sum += float64(got[row*4+column])
		}
		if math.Abs(sum) > 1e-7 {
			t.Fatalf("row %d cotangent sum %.9g", row, sum)
		}
	}
	for _, value := range got[8:] {
		if value != 0 {
			t.Fatal("unscored final row has a cotangent")
		}
	}

	epsilon := float32(1e-3)
	for coordinate := 0; coordinate < 8; coordinate++ {
		plus := append([]float32(nil), original...)
		minus := append([]float32(nil), original...)
		plus[coordinate] += epsilon
		minus[coordinate] -= epsilon
		numerical := (completionReferenceLoss(plus, 4, targets) - completionReferenceLoss(minus, 4, targets)) /
			float64(2*epsilon)
		analytic := float64(got[coordinate])
		if math.Abs(analytic-numerical) > 2e-5 {
			t.Fatalf("coordinate %d analytic %.9g numerical %.9g", coordinate, analytic, numerical)
		}
	}
}

func TestCompletionCotangentScalesOnlyDerivative(t *testing.T) {
	original := []float32{
		0.4, -0.3, 1.1, 0.2,
		-0.7, 0.9, 0.1, 0.5,
		9, 8, 7, 6,
	}
	targets := []int64{2, 1}
	one := append([]float32(nil), original...)
	two := append([]float32(nil), original...)
	lossOne, err := completionCotangent(one, 4, targets, 1)
	if err != nil {
		t.Fatal(err)
	}
	lossTwo, err := completionCotangent(two, 4, targets, 2)
	if err != nil {
		t.Fatal(err)
	}
	if lossOne != lossTwo {
		t.Fatalf("loss scale changed loss: %.12g != %.12g", lossOne, lossTwo)
	}
	for index := range one {
		want := 2 * float64(one[index])
		if math.Abs(float64(two[index])-want) > 1e-7 {
			t.Fatalf("scaled cotangent %d %.9g != %.9g", index, two[index], want)
		}
	}
}

// preSplitCompletionCotangent is completionCotangent as it stood before the loss
// moved into completionNLL, kept verbatim. A tolerance cannot tell a moved
// rounding from none, so the split is held to these bits.
func preSplitCompletionCotangent(logits []float32, vocabulary int, targets []int64, lossScale float64) (float64, error) {
	if vocabulary < 2 || len(targets) == 0 || len(logits) != (len(targets)+1)*vocabulary ||
		math.IsNaN(lossScale) || math.IsInf(lossScale, 0) || lossScale <= 0 {
		return 0, fmt.Errorf("%w: invalid completion cotangent geometry", ErrCompletionStep)
	}
	for _, target := range targets {
		if target < 0 || target >= int64(vocabulary) {
			return 0, fmt.Errorf("%w: target token outside vocabulary", ErrCompletionStep)
		}
	}
	for _, value := range logits[:len(targets)*vocabulary] {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return 0, fmt.Errorf("%w: nonfinite completion logit", ErrCompletionStep)
		}
	}

	factor := lossScale / float64(len(targets))
	totalLoss := 0.0
	for row, target := range targets {
		offset := row * vocabulary
		maximum := float64(logits[offset])
		for column := 1; column < vocabulary; column++ {
			maximum = math.Max(maximum, float64(logits[offset+column]))
		}
		sum := 0.0
		for column := 0; column < vocabulary; column++ {
			sum += math.Exp(float64(logits[offset+column]) - maximum)
		}
		if !(sum > 0) || math.IsNaN(sum) || math.IsInf(sum, 0) {
			return 0, fmt.Errorf("%w: invalid completion softmax normalization", ErrCompletionStep)
		}
		logSum := math.Log(sum)
		targetLogit := float64(logits[offset+int(target)])
		totalLoss += maximum + logSum - targetLogit
		for column := 0; column < vocabulary; column++ {
			probability := math.Exp(float64(logits[offset+column])-maximum) / sum
			gradient := probability * factor
			if column == int(target) {
				gradient -= factor
			}
			rounded := float32(gradient)
			if math.IsNaN(gradient) || math.IsInf(gradient, 0) ||
				math.IsNaN(float64(rounded)) || math.IsInf(float64(rounded), 0) {
				return 0, fmt.Errorf("%w: completion derivative is not finite FP32", ErrCompletionStep)
			}
			logits[offset+column] = rounded
		}
	}
	clear(logits[len(targets)*vocabulary:])
	loss := totalLoss / float64(len(targets))
	if math.IsNaN(loss) || math.IsInf(loss, 0) {
		return 0, fmt.Errorf("%w: nonfinite completion loss", ErrCompletionStep)
	}
	return loss, nil
}

// completionCase spreads logits over a wide range so that the exponentials,
// the running maximum and the float32 rounding of every derivative all matter.
func completionCase(rows, vocabulary int, phase float64) ([]float32, []int64) {
	logits := make([]float32, (rows+1)*vocabulary)
	for i := range logits {
		logits[i] = float32(9 * math.Sin(phase+float64(i)*0.37) * math.Cos(float64(i)*0.011))
	}
	targets := make([]int64, rows)
	for row := range targets {
		targets[row] = int64((row*7 + int(phase*10)) % vocabulary)
	}
	return logits, targets
}

func TestCompletionCotangentKeepsThePreSplitBits(t *testing.T) {
	for _, shape := range []struct {
		rows, vocabulary int
		phase, scale     float64
	}{{1, 2, 0.1, 1}, {2, 4, 0.4, 1}, {5, 257, 0.7, 1}, {7, 1031, 1.3, 1}, {7, 1031, 1.3, 3}, {3, 4099, 2.2, 1024}} {
		logits, targets := completionCase(shape.rows, shape.vocabulary, shape.phase)
		got := append([]float32(nil), logits...)
		want := append([]float32(nil), logits...)
		gotLoss, gotErr := completionCotangent(got, shape.vocabulary, targets, shape.scale)
		wantLoss, wantErr := preSplitCompletionCotangent(want, shape.vocabulary, targets, shape.scale)
		if gotErr != nil || wantErr != nil {
			t.Fatal(gotErr, wantErr)
		}
		if math.Float64bits(gotLoss) != math.Float64bits(wantLoss) {
			t.Fatalf("%+v loss %x != pre-split %x", shape, math.Float64bits(gotLoss), math.Float64bits(wantLoss))
		}
		for i := range got {
			if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
				t.Fatalf("%+v cotangent %d %x != pre-split %x", shape, i, math.Float32bits(got[i]), math.Float32bits(want[i]))
			}
		}
	}
}

func TestCompletionNLLIsTheCotangentLossAndWritesNothing(t *testing.T) {
	for _, shape := range []struct {
		rows, vocabulary int
		phase            float64
	}{{1, 2, 0.1}, {5, 257, 0.7}, {7, 1031, 1.3}} {
		logits, targets := completionCase(shape.rows, shape.vocabulary, shape.phase)
		read := append([]float32(nil), logits...)
		loss, err := completionNLL(read, shape.vocabulary, targets, nil)
		if err != nil {
			t.Fatal(err)
		}
		for i := range read {
			if math.Float32bits(read[i]) != math.Float32bits(logits[i]) {
				t.Fatalf("%+v loss readout wrote logit %d", shape, i)
			}
		}
		cotangent, err := completionCotangent(append([]float32(nil), logits...), shape.vocabulary, targets, 2)
		if err != nil {
			t.Fatal(err)
		}
		if math.Float64bits(loss) != math.Float64bits(cotangent) {
			t.Fatalf("%+v readout %.17g != gradient loss %.17g", shape, loss, cotangent)
		}
	}
	valid := []float32{0, 1, 2, 0, 0, 0}
	for name, call := range map[string]func() error{
		"empty targets":             func() error { _, err := completionNLL(valid, 2, nil, nil); return err },
		"wrong geometry":            func() error { _, err := completionNLL([]float32{0, 1}, 2, []int64{1}, nil); return err },
		"one-token vocabulary":      func() error { _, err := completionNLL([]float32{0, 1, 2}, 1, []int64{0, 0}, nil); return err },
		"target outside vocabulary": func() error { _, err := completionNLL(valid, 2, []int64{2, 0}, nil); return err },
		"nonfinite logit": func() error {
			bad := append([]float32(nil), valid...)
			bad[1] = float32(math.Inf(1))
			_, err := completionNLL(bad, 2, []int64{1, 0}, nil)
			return err
		},
	} {
		if err := call(); !errors.Is(err, ErrCompletionStep) {
			t.Fatalf("%s: invalid completion loss accepted: %v", name, err)
		}
	}
}

func TestCompletionCotangentRejectsInvalidInputs(t *testing.T) {
	valid := []float32{0, 1, 2, 0, 0, 0}
	for name, run := range map[string]func() error{
		"empty targets": func() error {
			_, err := completionCotangent(append([]float32(nil), valid...), 2, nil, 1)
			return err
		},
		"wrong geometry": func() error {
			_, err := completionCotangent([]float32{0, 1}, 2, []int64{1}, 1)
			return err
		},
		"target outside vocabulary": func() error {
			_, err := completionCotangent(append([]float32(nil), valid...), 2, []int64{2, 0}, 1)
			return err
		},
		"zero scale": func() error {
			_, err := completionCotangent(append([]float32(nil), valid...), 2, []int64{1, 0}, 0)
			return err
		},
		"nonfinite logit": func() error {
			bad := append([]float32(nil), valid...)
			bad[0] = float32(math.NaN())
			_, err := completionCotangent(bad, 2, []int64{1, 0}, 1)
			return err
		},
	} {
		t.Run(name, func(t *testing.T) {
			if err := run(); err == nil {
				t.Fatal("invalid completion cotangent accepted")
			}
		})
	}
}
