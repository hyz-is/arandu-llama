package rollout_test

import (
	"math"
	"os"
	"strconv"
	"testing"

	llama "github.com/tayi-ai/arandu-llama"
)

// Rollouts against a real model. Opt-in, because they load one:
//
//	TEST_ROLLOUT_MODEL       path to a GGUF (required; the test skips without it)
//	TEST_ROLLOUT_GPU_LAYERS  layers to offload; 0, the CPU, when unset
//	TEST_ROLLOUT_ADAPTER     a LoRA GGUF for the model (optional)
//
// scoreTolerance is declared before any run, from the arithmetic, not fitted
// to one. Against a context with a batch of one, Score and SampleRollout
// decode the same tokens through the same one-token graphs, so their logits
// are the same numbers; what separates the two log probabilities is that
// Score subtracts the maximum logit in float32 before widening, and
// SampleRollout in float64. That rounding is 2^-24 of |logit - max|, about
// 3e-6 at a gap of 50, on the target term and on each term of the
// denominator, which holds that term's weight in the sum. 1e-4 per token
// leaves a factor of thirty.
const scoreTolerance = 1e-4

func TestRolloutAgainstAModel(t *testing.T) {
	path := os.Getenv("TEST_ROLLOUT_MODEL")
	if path == "" {
		t.Skip("TEST_ROLLOUT_MODEL not set - skipping the rollout test that loads a model")
	}
	layers := 0
	if value := os.Getenv("TEST_ROLLOUT_GPU_LAYERS"); value != "" {
		parsed, err := strconv.Atoi(value)
		if err != nil {
			t.Fatalf("TEST_ROLLOUT_GPU_LAYERS=%q: %v", value, err)
		}
		layers = parsed
	}
	model, err := llama.LoadModel(path, llama.WithGPULayers(layers))
	if err != nil {
		t.Fatalf("loading %s: %v", path, err)
	}
	defer model.Close()
	quantisation, err := model.Describe()
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("model %s (%s), gpu layers %d", path, quantisation, layers)

	// Batch one: every decode is one token, in the rollout and in Score alike.
	ctx, err := model.NewContext(llama.WithContext(1024), llama.WithBatch(1))
	if err != nil {
		t.Fatal(err)
	}
	defer ctx.Close()
	prompt, err := ctx.Tokenize("The history of the printing press begins")
	if err != nil {
		t.Fatal(err)
	}

	t.Run("same seed and prompt give the same rollout, whatever ran before", func(t *testing.T) {
		for _, s := range []llama.RolloutSampling{sampling(1, 0, 1, 0), sampling(0.7, 40, 0.9, 0.05)} {
			first := rollout(t, ctx, prompt, s, 7, 24)
			// Another rollout, and a prefix-cached generation, in between.
			rollout(t, ctx, prompt, s, 8, 24)
			if _, err := ctx.Generate("An unrelated prompt that fills the cache", llama.WithMaxTokens(4), llama.WithSeed(1)); err != nil {
				t.Fatal(err)
			}
			again := rollout(t, ctx, prompt, s, 7, 24)
			if len(first.Tokens) != len(again.Tokens) {
				t.Fatalf("seed 7 gave %d then %d tokens", len(first.Tokens), len(again.Tokens))
			}
			for i := range first.Tokens {
				if first.Tokens[i] != again.Tokens[i] || math.Float64bits(first.LogMu[i]) != math.Float64bits(again.LogMu[i]) {
					t.Fatalf("seed 7, position %d: token %d log mu %.17g, then token %d log mu %.17g",
						i, first.Tokens[i], first.LogMu[i], again.Tokens[i], again.LogMu[i])
				}
			}
			if first.Stop != again.Stop {
				t.Fatalf("seed 7 stopped on %v then on %v", first.Stop, again.Stop)
			}
		}
	})

	t.Run("at temperature one without filters log mu is the teacher-forced log probability", func(t *testing.T) {
		r := rollout(t, ctx, prompt, sampling(1, 0, 1, 0), 3, 16)
		worst := 0.0
		for i := range r.Tokens {
			sequence := append(append([]int32(nil), prompt...), r.Tokens[:i+1]...)
			score, err := ctx.Score(sequence, len(prompt)+i)
			if err != nil {
				t.Fatalf("scoring position %d: %v", i, err)
			}
			if score.Tokens != 1 {
				t.Fatalf("scoring position %d read %d positions, want 1", i, score.Tokens)
			}
			difference := math.Abs(-score.SumNLL - r.LogMu[i])
			worst = math.Max(worst, difference)
			if difference > scoreTolerance {
				t.Fatalf("position %d, token %d: log mu %.12g, teacher-forced %.12g, |diff| %.3g > %g",
					i, r.Tokens[i], r.LogMu[i], -score.SumNLL, difference, scoreTolerance)
			}
		}
		whole, err := ctx.Score(append(append([]int32(nil), prompt...), r.Tokens...), len(prompt))
		if err != nil {
			t.Fatal(err)
		}
		if whole.Tokens != len(r.Tokens) || math.Abs(-whole.SumNLL-r.SumLogMu()) > scoreTolerance*float64(len(r.Tokens)) {
			t.Fatalf("whole sequence: %d positions summing %.12g, rollout %d tokens summing %.12g",
				whole.Tokens, -whole.SumNLL, len(r.Tokens), r.SumLogMu())
		}
		t.Logf("batch one: %d tokens, largest |log mu - teacher-forced| %.3g", len(r.Tokens), worst)

		// Measured, not asserted: the same comparison against a context that
		// scores in windows of 512, which is how a trainer reads log pi. This
		// is the part of the gap between mu and pi that the decode shape
		// alone contributes.
		wide, err := model.NewContext(llama.WithContext(1024))
		if err != nil {
			t.Fatal(err)
		}
		defer wide.Close()
		widest := 0.0
		for i := range r.Tokens {
			sequence := append(append([]int32(nil), prompt...), r.Tokens[:i+1]...)
			score, err := wide.Score(sequence, len(prompt)+i)
			if err != nil {
				t.Fatal(err)
			}
			widest = math.Max(widest, math.Abs(-score.SumNLL-r.LogMu[i]))
		}
		t.Logf("batch 512: largest |log mu - teacher-forced| %.3g", widest)
	})

	t.Run("the end-of-generation token is kept and named, and the ceiling is named", func(t *testing.T) {
		chat, err := model.FormatChatPrompt([]llama.ChatMessage{{Role: "user", Content: "Reply with the single word yes."}},
			llama.ChatOptions{EnableThinking: llama.Bool(false)})
		if err != nil {
			t.Fatal(err)
		}
		answered, err := ctx.Tokenize(chat + "yes")
		if err != nil {
			t.Fatal(err)
		}
		ended := 0
		for seed := 1; seed <= 32; seed++ {
			r := rollout(t, ctx, answered, sampling(1, 0, 1, 0), seed, 8)
			checkStop(t, ctx, r, 8)
			if r.Stop == llama.RolloutStopEndOfGeneration {
				ended++
				// The end-of-generation token's log mu is the model's own
				// probability of ending here, like every other token's.
				whole, err := ctx.Score(append(append([]int32(nil), answered...), r.Tokens...), len(answered))
				if err != nil {
					t.Fatal(err)
				}
				if math.Abs(-whole.SumNLL-r.SumLogMu()) > scoreTolerance*float64(len(r.Tokens)) {
					t.Fatalf("seed %d ended with %v: teacher-forced %.12g, rollout %.12g", seed, r.Tokens, -whole.SumNLL, r.SumLogMu())
				}
			}
		}
		if ended == 0 {
			t.Fatal("no rollout of 32 ended with an end-of-generation token after a finished answer; the end-of-generation path was not exercised on this model")
		}
		t.Logf("%d of 32 rollouts ended with an end-of-generation token", ended)

		lengths := 0
		for _, ceiling := range []int{1, 3} {
			for seed := 1; seed <= 4; seed++ {
				r := rollout(t, ctx, prompt, sampling(1, 0, 1, 0), seed, ceiling)
				checkStop(t, ctx, r, ceiling)
				if r.Stop == llama.RolloutStopLength {
					lengths++
				}
			}
		}
		if lengths == 0 {
			t.Fatal("no rollout stopped at the ceiling")
		}
	})

	t.Run("a prompt and ceiling longer than the context are refused", func(t *testing.T) {
		o := options()
		o.MaxTokens = llama.Int(1024)
		if _, err := ctx.SampleRollout(prompt, o); err == nil {
			t.Fatal("a rollout that cannot fit the context was accepted")
		}
	})

	adapterPath := os.Getenv("TEST_ROLLOUT_ADAPTER")
	t.Run("a rollout is labelled with the adapter it was sampled under", func(t *testing.T) {
		if adapterPath == "" {
			t.Skip("TEST_ROLLOUT_ADAPTER not set")
		}
		adapter, err := model.LoadAdapter(adapterPath)
		if err != nil {
			t.Fatal(err)
		}
		defer adapter.Close()
		if err := ctx.SetAdapters([]*llama.Adapter{adapter}, []float32{1}); err != nil {
			t.Fatal(err)
		}
		defer ctx.ClearAdapters()
		first := rollout(t, ctx, prompt, sampling(1, 0, 1, 0), 5, 16)
		again := rollout(t, ctx, prompt, sampling(1, 0, 1, 0), 5, 16)
		if first.Policy != adapter.Digest() || len(first.Scales) != 1 || first.Scales[0] != 1 {
			t.Fatalf("labelled %q %v, want %q [1]", first.Policy, first.Scales, adapter.Digest())
		}
		for i := range first.Tokens {
			if i >= len(again.Tokens) || first.Tokens[i] != again.Tokens[i] || math.Float64bits(first.LogMu[i]) != math.Float64bits(again.LogMu[i]) {
				t.Fatalf("under the adapter, seed 5 differs at position %d", i)
			}
		}
		if err := ctx.ClearAdapters(); err != nil {
			t.Fatal(err)
		}
		base := rollout(t, ctx, prompt, sampling(1, 0, 1, 0), 5, 16)
		if base.Policy != "base" || base.Scales != nil {
			t.Fatalf("after clearing: labelled %q %v", base.Policy, base.Scales)
		}
		moved := len(base.Tokens) != len(first.Tokens)
		for i := 0; !moved && i < len(base.Tokens); i++ {
			moved = base.Tokens[i] != first.Tokens[i] || base.LogMu[i] != first.LogMu[i]
		}
		t.Logf("adapter at scale 1 moved the rollout: %v", moved)
	})
}

func rollout(t *testing.T, ctx *llama.Context, prompt []int32, s llama.RolloutSampling, seed, ceiling int) *llama.Rollout {
	t.Helper()
	r, err := ctx.SampleRollout(prompt, llama.RolloutOptions{Sampling: s, Seed: llama.Int(seed), MaxTokens: llama.Int(ceiling)})
	if err != nil {
		t.Fatalf("seed %d: %v", seed, err)
	}
	if r.PromptTokens != len(prompt) || r.Version != llama.RolloutVersion || r.ModelSHA256 != "" || r.Quantisation == "" {
		t.Fatalf("seed %d: prompt %d of %d, version %q, model digest %q, quantisation %q",
			seed, r.PromptTokens, len(prompt), r.Version, r.ModelSHA256, r.Quantisation)
	}
	if r.Temperature != *s.Temperature || r.TopK != *s.TopK || r.TopP != *s.TopP || r.MinP != *s.MinP || r.Seed != seed || r.MaxTokens != ceiling {
		t.Fatalf("seed %d: the rollout does not carry the options it ran with: %+v", seed, r)
	}
	if len(r.Tokens) != len(r.LogMu) || len(r.Tokens) < 1 || len(r.Tokens) > ceiling {
		t.Fatalf("seed %d: %d tokens and %d log mu under a ceiling of %d", seed, len(r.Tokens), len(r.LogMu), ceiling)
	}
	for i, value := range r.LogMu {
		if math.IsNaN(value) || math.IsInf(value, 0) || value > 0 {
			t.Fatalf("seed %d: log mu %v at %d", seed, value, i)
		}
	}
	return r
}

// checkStop holds the stop reason to the tokens: the end-of-generation token
// is the last one exactly when that is the reason, never earlier, and a
// rollout that stopped at the ceiling filled it.
func checkStop(t *testing.T, ctx *llama.Context, r *llama.Rollout, ceiling int) {
	t.Helper()
	for i, token := range r.Tokens {
		eog, err := ctx.IsEndOfGeneration(token)
		if err != nil {
			t.Fatal(err)
		}
		last := i == len(r.Tokens)-1
		switch {
		case eog && !last:
			t.Fatalf("an end-of-generation token at %d of %d did not end the rollout", i, len(r.Tokens))
		case last && eog != (r.Stop == llama.RolloutStopEndOfGeneration):
			t.Fatalf("the last token is end-of-generation: %v; the rollout says it stopped on %v", eog, r.Stop)
		}
	}
	if r.Stop == llama.RolloutStopLength && len(r.Tokens) != ceiling {
		t.Fatalf("stopped at the ceiling of %d after %d tokens", ceiling, len(r.Tokens))
	}
}
