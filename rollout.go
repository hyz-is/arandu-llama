package llama

/*
#include "wrapper.h"
*/
import "C"

import (
	"errors"
	"fmt"
	"math"
	"runtime"
	"unsafe"
)

// Sampling a rollout for on-policy distillation.
//
// The student writes the sequences it is then corrected on, and it writes them
// here, as a quantised model with an adapter, while the trainer computes its
// gradient under its own forward pass. The two are not the same policy, and
// the gap is measured, not assumed: every sampled token carries log mu, the log
// probability of that token under the distribution it was actually drawn from,
// and the trainer compares it with its own log pi of the same token.
//
// That comparison is worth something only if log mu is exact. So the sampler
// is not llama.cpp's chain and not this package's Generate defaults: the same
// code decides which tokens survive the filters, computes log mu over them and
// draws, and every parameter of the distribution is spelled out by the caller.

// RolloutVersion names the sampling procedure and the engine revision that ran
// it. The procedure is the definition documented on RolloutSampling; the
// engine is the pinned llama.cpp commit, because a pin bump changes the logits
// and therefore every log mu, with nothing else in a rollout saying so.
const RolloutVersion = "arandu-llama.rollout.v1/llama.cpp.90c26fcd"

// RolloutSampling is the distribution a rollout samples every position from.
//
// Every field is required. A nil field is refused rather than defaulted: the
// defaults Generate applies (temperature 0.8, top-k 40, top-p 0.95, min-p
// 0.05) truncate the distribution, and a truncated distribution taken for the
// student's own is the one mistake on-policy distillation cannot see. A filter
// is disabled by saying so, with the value named on its field.
//
// The distribution of one position, with l the float32 logits llama.cpp
// returns for it and every quantity below computed in float64:
//
//	z_i   = l_i / Temperature
//	order = token ids by descending z, ascending id among equal z
//	w_i   = exp(z_i - max z)
//	K     = the first TopK ids of order when 0 < TopK < vocabulary, else all
//	P     = when TopP < 1, the shortest prefix of order within K whose summed
//	        w reaches TopP * (sum of w over K), else K
//	S     = when MinP > 0, the ids of P with w_i >= MinP, else P
//	log mu(i) = (z_i - max z) - log(sum of w over S) for i in S, -Inf otherwise
//
// Sums over a set run in ascending token id; the TopP prefix sum runs in
// order. The draw is the inverse CDF of S in ascending token id: the first id
// whose running sum of w exceeds u * (sum of w over S), with u the position's
// uniform (see RolloutUniforms). A token is therefore drawn with probability
// exp(log mu) up to the rounding of that running sum and the 2^-53 grid of u;
// a token whose w underflows to zero, log mu below about -745, is never drawn.
//
// The temperature is applied before the filters, so TopP and MinP read
// the tempered distribution. That is the opposite of llama.cpp's own sampler
// chain, which filters the raw logits and tempers what is left; the two keep
// different sets at any temperature other than one.
//
// What log mu is, filter by filter:
//
//   - none (Temperature 1, TopK 0, TopP 1, MinP 0): the log-softmax of the
//     logits over the whole vocabulary.
//   - Temperature T alone: the log-softmax of logits / T.
//   - TopK: the log-softmax of z restricted to the TopK largest z,
//     renormalised over them.
//   - TopP: renormalised over the nucleus of the tempered distribution; with
//     TopK, the nucleus of the TopK set renormalised first, not of the whole
//     vocabulary.
//   - MinP: renormalised over the tokens whose tempered probability is at
//     least MinP times the largest one. It is a ratio, so it does not depend on
//     the set it is applied after.
//   - any combination: renormalised over S, which is the shortest of the
//     prefixes each filter keeps.
//
// Every combination of these four is exact, because the kept set is not
// reconstructed after the fact: it is the set the draw is made over. What
// cannot be made exact is not offered. Temperature 0 is greedy decoding and
// has no softmax to take the log of; MinP 1 is the same thing under another
// name; TopP 0 keeps nothing. Repetition penalties, DRY, XTC, typical, top-n
// sigma, dynamic temperature, Mirostat, grammars and logit biases are absent:
// each makes the distribution depend on history, on state or on a second coin,
// and none of that is in log mu here. A logit that is not finite refuses the
// rollout.
type RolloutSampling struct {
	// Temperature divides the logits. Finite and above zero.
	Temperature *float32
	// TopK keeps the K most probable tokens. 0 disables it, and so does any K
	// at or above the vocabulary; negative is refused.
	TopK *int
	// TopP keeps the nucleus of probability TopP. In (0, 1]; 1 disables it.
	TopP *float32
	// MinP keeps the tokens at least MinP times as probable as the most
	// probable one. In [0, 1); 0 disables it.
	MinP *float32
}

// RolloutOptions is a rollout: the distribution, the seed and the ceiling.
// Every field is required, for the reason given on RolloutSampling.
type RolloutOptions struct {
	// Sampling is the distribution every position is drawn from.
	Sampling RolloutSampling
	// Seed seeds mt19937_64; position t draws its t-th uniform. Non-negative.
	// There is no random seed: a rollout that cannot be repeated cannot be
	// checked.
	Seed *int
	// MaxTokens is the ceiling on sampled tokens, the end-of-generation token
	// included. At least one; the prompt and the ceiling together must fit the
	// context.
	MaxTokens *int
}

// Validate reports the first field that is missing or outside its range.
func (s RolloutSampling) Validate() error {
	_, err := s.native()
	return err
}

// Validate reports the first field that is missing or outside its range.
func (o RolloutOptions) Validate() error {
	_, _, _, err := o.resolve()
	return err
}

func (s RolloutSampling) native() (C.llama_wrapper_rollout_sampling, error) {
	var native C.llama_wrapper_rollout_sampling
	switch {
	case s.Temperature == nil:
		return native, errors.New("rollout: Temperature is required; there is no default")
	case s.TopK == nil:
		return native, errors.New("rollout: TopK is required; 0 disables it, there is no default")
	case s.TopP == nil:
		return native, errors.New("rollout: TopP is required; 1 disables it, there is no default")
	case s.MinP == nil:
		return native, errors.New("rollout: MinP is required; 0 disables it, there is no default")
	}
	temperature, topK, topP, minP := *s.Temperature, *s.TopK, *s.TopP, *s.MinP
	// Written so NaN fails every range: a comparison with NaN is false.
	switch {
	case !(temperature > 0) || math.IsInf(float64(temperature), 0):
		return native, fmt.Errorf("rollout: Temperature %v must be finite and above zero; greedy decoding has no sampling distribution", temperature)
	case topK < 0 || topK > math.MaxInt32:
		return native, fmt.Errorf("rollout: TopK %d must be 0 (disabled) or positive", topK)
	case !(topP > 0 && topP <= 1):
		return native, fmt.Errorf("rollout: TopP %v must be in (0, 1]; 1 disables it", topP)
	case !(minP >= 0 && minP < 1):
		return native, fmt.Errorf("rollout: MinP %v must be in [0, 1); 0 disables it", minP)
	}
	native.temperature = C.float(temperature)
	native.top_k = C.int(topK)
	native.top_p = C.float(topP)
	native.min_p = C.float(minP)
	return native, nil
}

func (o RolloutOptions) resolve() (C.llama_wrapper_rollout_sampling, uint64, int, error) {
	native, err := o.Sampling.native()
	if err != nil {
		return native, 0, 0, err
	}
	switch {
	case o.Seed == nil:
		return native, 0, 0, errors.New("rollout: Seed is required; there is no random seed")
	case o.MaxTokens == nil:
		return native, 0, 0, errors.New("rollout: MaxTokens is required; there is no default ceiling")
	case *o.Seed < 0:
		return native, 0, 0, fmt.Errorf("rollout: Seed %d must be non-negative", *o.Seed)
	case *o.MaxTokens < 1 || *o.MaxTokens > math.MaxInt32:
		return native, 0, 0, fmt.Errorf("rollout: MaxTokens %d must be at least one", *o.MaxTokens)
	}
	return native, uint64(*o.Seed), *o.MaxTokens, nil
}

// RolloutStop is why a rollout ended. A failed decode is not one of them: it
// is an error, because a rollout it cut short is not a sample of anything.
type RolloutStop int

const (
	// RolloutStopEndOfGeneration: the model sampled an end-of-generation
	// token, which is the last entry of Rollout.Tokens.
	RolloutStopEndOfGeneration RolloutStop = iota + 1
	// RolloutStopLength: MaxTokens tokens were sampled and none of them ended
	// generation.
	RolloutStopLength
)

// String names the reason.
func (s RolloutStop) String() string {
	switch s {
	case RolloutStopEndOfGeneration:
		return "end_of_generation"
	case RolloutStopLength:
		return "length"
	default:
		return "unknown"
	}
}

// Rollout is one sampled continuation and what produced it.
//
// Tokens and LogMu run in parallel: LogMu[t] is log mu of Tokens[t] under the
// distribution position t was drawn from, as defined on RolloutSampling. When
// Stop is RolloutStopEndOfGeneration the last token is the end-of-generation
// token, with its log mu, because the trainer has to learn to emit it.
//
// The rest labels the policy. ModelSHA256 is left empty: a Context does not
// know the digest of the file its model was loaded from, and a label this
// package cannot verify is not one it writes. The caller that verified the
// file sets it.
type Rollout struct {
	Tokens       []int32   // sampled ids, the end-of-generation token included
	LogMu        []float64 // log mu of each sampled id
	Stop         RolloutStop
	PromptTokens int // prompt tokens decoded from an empty cache

	Temperature float32 // the distribution, as resolved from the options
	TopK        int
	TopP        float32
	MinP        float32
	Seed        int
	MaxTokens   int

	Policy         string    // adapter digests in applied order, or "base"
	Scales         []float32 // scale of each applied adapter, same order; nil for "base"
	Quantisation   string    // Model.Describe, read from the model
	KVCacheType    string    // the context's KV cache type, which the logits depend on
	FlashAttention string    // the context's flash attention mode
	ModelSHA256    string    // empty; set by the caller that verified the model file
	Version        string    // RolloutVersion
}

// SumLogMu sums LogMu, the log probability the sampling distribution gave
// the whole rollout.
func (r *Rollout) SumLogMu() float64 {
	sum := 0.0
	for _, value := range r.LogMu {
		sum += value
	}
	return sum
}

// SampleRollout samples a continuation of prompt, which the caller has already
// tokenised, and records log mu for every sampled token.
//
// The KV cache is cleared first and the whole prompt is decoded from position
// zero, so a rollout does not depend on which rollout ran before it on this
// context, and a cache filled under another adapter is never read. The
// consequence is the one Score has: prefix-cached generation on this context
// re-decodes its prompt afterwards.
//
// The rollout runs under the adapters applied to this context and is labelled
// with them. An adapter whose digest is unknown is refused, as it is for a
// reading: a rollout that cannot name its policy cannot be compared with the
// trainer's.
//
// Refused before crossing into C: invalid options (see RolloutSampling and
// RolloutOptions), a nil or closed context or model, an empty prompt, a
// negative token, a prompt plus MaxTokens longer than the context. Refused in
// C: a token outside the vocabulary, a logit that is not finite. A decode that
// fails is an error, never a shorter rollout.
//
// Thread safety: holds the context's write lock for the whole rollout, so the
// label names the adapters the tokens were sampled under.
func (c *Context) SampleRollout(prompt []int32, options RolloutOptions) (*Rollout, error) {
	sampling, seed, maxTokens, err := options.resolve()
	if err != nil {
		return nil, err
	}
	if c == nil {
		return nil, ErrNoContext
	}
	if len(prompt) == 0 {
		return nil, errors.New("rollout: at least one prompt token is required to condition on")
	}
	if len(prompt) > math.MaxInt32-maxTokens {
		return nil, fmt.Errorf("rollout: a prompt of %d tokens is too long", len(prompt))
	}
	for i, token := range prompt {
		if token < 0 {
			return nil, fmt.Errorf("rollout: invalid token %d at index %d", token, i)
		}
	}

	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closed {
		return nil, errors.New("context is closed")
	}
	if c.model == nil {
		return nil, errors.New("model is closed")
	}
	c.model.mu.RLock()
	modelClosed := c.model.closed
	c.model.mu.RUnlock()
	if modelClosed {
		return nil, errors.New("model is closed")
	}
	// Checked here as well as in C so the output buffers are never sized by a
	// ceiling the context cannot hold. The requested size is the lower bound
	// of the one llama.cpp allocated.
	if len(prompt)+maxTokens > c.config.contextSize {
		return nil, fmt.Errorf("rollout: a prompt of %d tokens and %d sampled tokens exceed the context of %d",
			len(prompt), maxTokens, c.config.contextSize)
	}

	policy, err := c.appliedPolicyLocked()
	if err != nil {
		return nil, err
	}
	quantisation, err := c.model.Describe()
	if err != nil {
		return nil, err
	}

	tokens := make([]int32, maxTokens)
	logMu := make([]float64, maxTokens)
	var generated, stop, decoded C.int
	result := C.llama_wrapper_rollout(c.contextPtr,
		(*C.int)(unsafe.Pointer(&prompt[0])), C.int(len(prompt)),
		sampling, C.ulonglong(seed), C.int(maxTokens),
		(*C.int)(unsafe.Pointer(&tokens[0])), (*C.double)(unsafe.Pointer(&logMu[0])), C.int(maxTokens),
		&generated, &stop, &decoded)
	// The C side reads the prompt and writes both outputs for the length of the
	// call and retains none of them.
	runtime.KeepAlive(prompt)
	runtime.KeepAlive(tokens)
	runtime.KeepAlive(logMu)

	if result != 0 {
		if stop == C.LLAMA_WRAPPER_STOP_DECODE_FAILED {
			return nil, fmt.Errorf("rollout: decode failed after %d sampled tokens: %s",
				int(generated), C.GoString(C.llama_wrapper_last_error()))
		}
		return nil, fmt.Errorf("rollout: %s", C.GoString(C.llama_wrapper_last_error()))
	}

	var reason RolloutStop
	switch stop {
	case C.LLAMA_WRAPPER_STOP_EOG:
		reason = RolloutStopEndOfGeneration
	case C.LLAMA_WRAPPER_STOP_LENGTH:
		reason = RolloutStopLength
	default:
		return nil, fmt.Errorf("rollout: the sampler reported stop reason %d", int(stop))
	}
	count := int(generated)
	if count < 1 || count > maxTokens || int(decoded) != len(prompt) {
		return nil, fmt.Errorf("rollout: the sampler reported %d tokens after decoding %d of %d prompt tokens",
			count, int(decoded), len(prompt))
	}
	for t, value := range logMu[:count] {
		if math.IsNaN(value) || math.IsInf(value, 0) || value > 0 {
			return nil, fmt.Errorf("rollout: log mu %v at position %d is not a log probability", value, t)
		}
	}

	var scales []float32
	if len(c.appliedScales) > 0 {
		scales = append([]float32(nil), c.appliedScales...)
	}
	return &Rollout{
		Tokens:         tokens[:count:count],
		LogMu:          logMu[:count:count],
		Stop:           reason,
		PromptTokens:   int(decoded),
		Temperature:    *options.Sampling.Temperature,
		TopK:           *options.Sampling.TopK,
		TopP:           *options.Sampling.TopP,
		MinP:           *options.Sampling.MinP,
		Seed:           *options.Seed,
		MaxTokens:      maxTokens,
		Policy:         policy,
		Scales:         scales,
		Quantisation:   quantisation,
		KVCacheType:    c.config.kvCacheType,
		FlashAttention: c.config.flashAttn,
		Version:        RolloutVersion,
	}, nil
}

// IsEndOfGeneration reports whether token ends generation for this context's
// model: the test SampleRollout stops on.
func (c *Context) IsEndOfGeneration(token int32) (bool, error) {
	if c == nil {
		return false, ErrNoContext
	}
	c.mu.RLock()
	defer c.mu.RUnlock()
	if c.closed {
		return false, errors.New("context is closed")
	}
	// The vocabulary is read through the model, which a closed model has freed.
	if c.model == nil {
		return false, errors.New("model is closed")
	}
	c.model.mu.RLock()
	modelClosed := c.model.closed
	c.model.mu.RUnlock()
	if modelClosed {
		return false, errors.New("model is closed")
	}
	result := C.llama_wrapper_rollout_is_eog(c.contextPtr, C.int(token))
	if result < 0 {
		return false, fmt.Errorf("rollout: %s", C.GoString(C.llama_wrapper_last_error()))
	}
	return result == 1, nil
}

// LogitDraw is one position of a rollout computed on a supplied logit row.
type LogitDraw struct {
	Token int32   // the id drawn
	LogMu float64 // log mu of Token
	// LogProbabilities is log mu over the whole vocabulary, -Inf for every
	// token the filters removed.
	LogProbabilities []float64
}

// DrawFromLogits runs the arithmetic of one rollout position on logits, with
// the uniform u in [0, 1) given instead of drawn. It is the same native code
// SampleRollout runs, with no model involved: the definition on
// RolloutSampling can be checked against a row whose answer is known.
func DrawFromLogits(logits []float32, sampling RolloutSampling, u float64) (*LogitDraw, error) {
	native, err := sampling.native()
	if err != nil {
		return nil, err
	}
	if len(logits) == 0 || len(logits) > math.MaxInt32 {
		return nil, fmt.Errorf("rollout: a logit row of %d entries", len(logits))
	}
	if !(u >= 0 && u < 1) {
		return nil, fmt.Errorf("rollout: the uniform %v must be in [0, 1)", u)
	}
	all := make([]float64, len(logits))
	var token C.int
	var logMu C.double
	result := C.llama_wrapper_rollout_draw((*C.float)(unsafe.Pointer(&logits[0])), C.int(len(logits)),
		native, C.double(u), &token, &logMu, (*C.double)(unsafe.Pointer(&all[0])))
	runtime.KeepAlive(logits)
	runtime.KeepAlive(all)
	if result != 0 {
		return nil, fmt.Errorf("rollout: %s", C.GoString(C.llama_wrapper_last_error()))
	}
	return &LogitDraw{Token: int32(token), LogMu: float64(logMu), LogProbabilities: all}, nil
}

// RolloutUniforms returns the first n uniforms a rollout seeded with seed
// draws, one per position: (x >> 11) * 2^-53 for each output x of
// mt19937_64(seed). The generator and the conversion are both specified bit
// for bit, so the stream is the same on every platform and standard library.
func RolloutUniforms(seed, n int) ([]float64, error) {
	if seed < 0 {
		return nil, fmt.Errorf("rollout: Seed %d must be non-negative", seed)
	}
	if n < 1 || n > math.MaxInt32 {
		return nil, fmt.Errorf("rollout: %d uniforms requested", n)
	}
	out := make([]float64, n)
	if C.llama_wrapper_rollout_uniforms(C.ulonglong(uint64(seed)), C.int(n), (*C.double)(unsafe.Pointer(&out[0]))) != 0 {
		return nil, fmt.Errorf("rollout: %s", C.GoString(C.llama_wrapper_last_error()))
	}
	runtime.KeepAlive(out)
	return out, nil
}
