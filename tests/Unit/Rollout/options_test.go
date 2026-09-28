package rollout_test

import (
	"errors"
	"math"
	"strings"
	"testing"

	llama "github.com/tayi-ai/arandu-llama"
)

// Nothing here loads a model: the options are validated before a context is
// touched, and the refusals are what these tests read.

func pure() llama.RolloutSampling {
	return llama.RolloutSampling{
		Temperature: llama.Float32(1),
		TopK:        llama.Int(0),
		TopP:        llama.Float32(1),
		MinP:        llama.Float32(0),
	}
}

func options() llama.RolloutOptions {
	return llama.RolloutOptions{Sampling: pure(), Seed: llama.Int(7), MaxTokens: llama.Int(16)}
}

func TestTheZeroOptionsAreRefusedRatherThanDefaulted(t *testing.T) {
	t.Parallel()

	err := llama.RolloutOptions{}.Validate()
	if err == nil || !strings.Contains(err.Error(), "Temperature is required") {
		t.Fatalf("zero options: got %v, want the missing temperature named", err)
	}
	if err := (llama.RolloutSampling{}).Validate(); err == nil {
		t.Fatal("a zero sampling was accepted: the defaults of Generate would have been applied silently")
	}
	if _, err := llama.DrawFromLogits([]float32{0, 1}, llama.RolloutSampling{}, 0.5); err == nil {
		t.Fatal("DrawFromLogits accepted a zero sampling")
	}
	if err := options().Validate(); err != nil {
		t.Fatalf("complete options were refused: %v", err)
	}
}

func TestEveryOptionIsRequired(t *testing.T) {
	t.Parallel()

	for _, field := range []struct {
		name  string
		clear func(*llama.RolloutOptions)
	}{
		{"Temperature", func(o *llama.RolloutOptions) { o.Sampling.Temperature = nil }},
		{"TopK", func(o *llama.RolloutOptions) { o.Sampling.TopK = nil }},
		{"TopP", func(o *llama.RolloutOptions) { o.Sampling.TopP = nil }},
		{"MinP", func(o *llama.RolloutOptions) { o.Sampling.MinP = nil }},
		{"Seed", func(o *llama.RolloutOptions) { o.Seed = nil }},
		{"MaxTokens", func(o *llama.RolloutOptions) { o.MaxTokens = nil }},
	} {
		o := options()
		field.clear(&o)
		err := o.Validate()
		if err == nil {
			t.Errorf("%s left nil was accepted", field.name)
			continue
		}
		if !strings.Contains(err.Error(), field.name+" is required") {
			t.Errorf("%s left nil: the error %q does not name it", field.name, err)
		}
	}
}

func TestOptionsOutsideTheirRangeAreRefused(t *testing.T) {
	t.Parallel()

	nan := float32(math.NaN())
	infinity := float32(math.Inf(1))
	for _, invalid := range []struct {
		name string
		set  func(*llama.RolloutOptions)
	}{
		{"temperature zero is greedy", func(o *llama.RolloutOptions) { o.Sampling.Temperature = llama.Float32(0) }},
		{"negative temperature", func(o *llama.RolloutOptions) { o.Sampling.Temperature = llama.Float32(-1) }},
		{"NaN temperature", func(o *llama.RolloutOptions) { o.Sampling.Temperature = &nan }},
		{"infinite temperature", func(o *llama.RolloutOptions) { o.Sampling.Temperature = &infinity }},
		{"negative top-k", func(o *llama.RolloutOptions) { o.Sampling.TopK = llama.Int(-1) }},
		{"top-p zero keeps nothing", func(o *llama.RolloutOptions) { o.Sampling.TopP = llama.Float32(0) }},
		{"top-p above one", func(o *llama.RolloutOptions) { o.Sampling.TopP = llama.Float32(1.5) }},
		{"NaN top-p", func(o *llama.RolloutOptions) { o.Sampling.TopP = &nan }},
		{"min-p one is greedy", func(o *llama.RolloutOptions) { o.Sampling.MinP = llama.Float32(1) }},
		{"negative min-p", func(o *llama.RolloutOptions) { o.Sampling.MinP = llama.Float32(-0.1) }},
		{"NaN min-p", func(o *llama.RolloutOptions) { o.Sampling.MinP = &nan }},
		{"negative seed means random", func(o *llama.RolloutOptions) { o.Seed = llama.Int(-1) }},
		{"zero tokens", func(o *llama.RolloutOptions) { o.MaxTokens = llama.Int(0) }},
	} {
		o := options()
		invalid.set(&o)
		if err := o.Validate(); err == nil {
			t.Errorf("%s: accepted", invalid.name)
		}
	}
}

func TestTheValuesThatDisableAFilterAreAccepted(t *testing.T) {
	t.Parallel()

	o := options()
	o.Sampling.TopK = llama.Int(1 << 30) // at or above any vocabulary: disabled
	o.Seed = llama.Int(0)
	o.MaxTokens = llama.Int(1)
	if err := o.Validate(); err != nil {
		t.Fatalf("boundary values refused: %v", err)
	}
}

// SampleRollout validates before it touches the context, so a nil context
// answers invalid options with the validation error and valid ones with
// ErrNoContext. That order is what makes the refusal testable without a model,
// and it is the order a caller hits it in.
func TestSampleRolloutValidatesBeforeTheContext(t *testing.T) {
	t.Parallel()

	var context *llama.Context
	if _, err := context.SampleRollout([]int32{1, 2}, llama.RolloutOptions{}); err == nil || errors.Is(err, llama.ErrNoContext) {
		t.Fatalf("zero options on a nil context: got %v, want the validation error", err)
	}
	if _, err := context.SampleRollout([]int32{1, 2}, options()); !errors.Is(err, llama.ErrNoContext) {
		t.Fatalf("valid options on a nil context: got %v, want ErrNoContext", err)
	}
	if _, err := context.IsEndOfGeneration(0); !errors.Is(err, llama.ErrNoContext) {
		t.Fatalf("IsEndOfGeneration on a nil context: got %v, want ErrNoContext", err)
	}
}

func TestTheStopReasonsHaveNames(t *testing.T) {
	t.Parallel()

	if llama.RolloutStopEndOfGeneration.String() != "end_of_generation" || llama.RolloutStopLength.String() != "length" {
		t.Fatalf("stop names: %q, %q", llama.RolloutStopEndOfGeneration, llama.RolloutStopLength)
	}
	if llama.RolloutStop(0).String() != "unknown" {
		t.Fatalf("the zero stop reason is named %q", llama.RolloutStop(0))
	}
}
