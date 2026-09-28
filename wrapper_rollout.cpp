// Sampling a rollout and recording, token by token, the log probability of the
// token under the distribution it was actually drawn from.
//
// On-policy distillation trains on sequences the student wrote, and the
// sequences are written here, by a quantised model with an adapter, while the
// gradient is computed elsewhere, by the trainer's own forward pass. The two
// policies are not the same one: quantisation and adapter staleness separate
// them. What makes the gap measurable is log mu(y_t) recorded at sampling time
// and compared later with the trainer's log pi(y_t). That comparison is only
// worth anything when log mu is the log probability of the distribution the
// token really came from, filters included, and not the model's unfiltered
// softmax or a float approximation of it.
//
// That is why this file does not use llama.cpp's sampler chain. The chain
// applies its filters to the raw logits and the temperature afterwards,
// computes the final softmax in float, and hands back a token. Reconstructing
// the distribution it drew from means repeating its arithmetic exactly, and a
// reconstruction that differs in one rounding differs in the kept set at the
// boundary. Here the same code decides the kept set, computes log mu and
// draws, so there is nothing to reconstruct.
//
// It lives in a file of its own for the reason wrapper_adapter.cpp does: this
// fork's diff against upstream stays out of the middle of wrapper.cpp.

#include "wrapper.h"

#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <vector>

// The handle layout wrapper.cpp allocates, repeated as wrapper_adapter.cpp
// repeats it and for the same reason. Keep it in sync with wrapper.cpp.
typedef struct {
    llama_context* ctx;
    llama_model* model;
    std::vector<int> cached_tokens;
} llama_wrapper_context_t;

// The error slot wrapper.cpp owns.
extern std::string g_last_error;

namespace {

// validate_rollout_sampling refuses what has no exact log mu or is not a
// sampling distribution at all. The Go side checks the same bounds; both stay,
// because the C function is callable from anywhere.
bool validate_rollout_sampling(const llama_wrapper_rollout_sampling& s, std::string& failure) {
    // A temperature of zero is greedy decoding: there is no softmax to take the
    // log of, and a point mass has no importance weight worth recording.
    if (!(std::isfinite(s.temperature) && s.temperature > 0.0f)) {
        failure = "Temperature must be finite and above zero; greedy decoding has no sampling distribution";
        return false;
    }
    if (s.top_k < 0) {
        failure = "top_k must be zero (disabled) or positive";
        return false;
    }
    // Zero keeps nothing. One disables the filter.
    if (!(s.top_p > 0.0f && s.top_p <= 1.0f)) {
        failure = "top_p must be in (0, 1]; 1 disables it";
        return false;
    }
    // One keeps only the tokens tied with the maximum, which is greedy decoding
    // under another name. Zero disables the filter.
    if (!(s.min_p >= 0.0f && s.min_p < 1.0f)) {
        failure = "min_p must be in [0, 1); 0 disables it";
        return false;
    }
    return true;
}

// rollout_scratch holds the per-position buffers so a rollout allocates them
// once rather than once per token. Each is n_vocab long.
struct rollout_scratch {
    std::vector<double> z;              // logit / temperature
    std::vector<double> w;              // exp(z - max z), in [0, 1]
    std::vector<unsigned char> keep;    // 1 when the token survives every filter
    std::vector<int> order;             // token ids, a sorted prefix of them
};

// rollout_uniform is the next uniform in [0, 1) from the generator: the top 53
// bits of one 64-bit output, scaled by 2^-53.
//
// Not std::uniform_real_distribution. Its algorithm is left to the standard
// library, and libc++ on the Mac and libstdc++ on the nodes are allowed to
// produce different numbers from the same generator. mt19937_64 itself is
// specified bit for bit by the standard, and so is this conversion.
double rollout_uniform(std::mt19937_64& generator) {
    return static_cast<double>(generator() >> 11) * 0x1.0p-53;
}

// rollout_step is the whole sampling arithmetic of one position. The
// definition it implements is the one documented on llama_wrapper_rollout in
// wrapper.h, and every step below is named after a line of it.
//
// log_mu_all may be null. When it is not, it receives log mu for every token,
// -infinity for those the filters removed.
bool rollout_step(const float* logits, int n_vocab, const llama_wrapper_rollout_sampling& s,
                  double u, rollout_scratch& scratch, int& token, double& log_mu,
                  double* log_mu_all, std::string& failure) {
    const size_t vocabulary = static_cast<size_t>(n_vocab);
    std::vector<double>& z = scratch.z;
    std::vector<double>& w = scratch.w;
    std::vector<unsigned char>& keep = scratch.keep;
    std::vector<int>& order = scratch.order;
    z.resize(vocabulary);
    w.resize(vocabulary);
    keep.assign(vocabulary, 1);

    // z = logit / T, in double. A logit that is not finite has no softmax,
    // and a -infinity is refused with the rest rather than read as a zero:
    // nothing in this procedure puts one there, so one arriving is a fault.
    const double temperature = static_cast<double>(s.temperature);
    double z_max = -std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < vocabulary; i++) {
        if (!std::isfinite(logits[i])) {
            failure = "The logit of token " + std::to_string(i) + " is not finite";
            return false;
        }
        z[i] = static_cast<double>(logits[i]) / temperature;
        if (z[i] > z_max) z_max = z[i];
    }
    if (!std::isfinite(z_max)) {
        failure = "The scaled logits are not finite at this temperature";
        return false;
    }
    // w = exp(z - max): the unnormalised probability, one at the maximum.
    for (size_t i = 0; i < vocabulary; i++) {
        w[i] = std::exp(z[i] - z_max);
    }

    // The order the filters read: descending z, ascending token id among
    // equal z. It is a strict total order, so the sorted prefix is unique and
    // does not depend on which standard library sorted it.
    auto before = [&z](int a, int b) {
        return z[static_cast<size_t>(a)] > z[static_cast<size_t>(b)] ||
               (z[static_cast<size_t>(a)] == z[static_cast<size_t>(b)] && a < b);
    };
    size_t sorted = 0;
    bool ordered = false;
    // ensure_sorted makes order[0, n) the first n tokens of the order. Only
    // the prefix the filters read is sorted: at 248320 tokens a full sort per
    // position costs about what the forward pass does.
    auto ensure_sorted = [&](size_t n) {
        if (!ordered) {
            order.resize(vocabulary);
            std::iota(order.begin(), order.end(), 0);
            ordered = true;
        }
        n = std::min(n, vocabulary);
        if (n <= sorted) return;
        // Every element of [0, sorted) precedes every element after it, so
        // sorting the head of the remainder extends the prefix.
        std::partial_sort(order.begin() + static_cast<std::ptrdiff_t>(sorted),
                          order.begin() + static_cast<std::ptrdiff_t>(n), order.end(), before);
        sorted = n;
    };

    // Every filter leaves a prefix of the order, so the kept set is always
    // order[0, limit) once a filter has run, and every token otherwise.
    size_t limit = vocabulary;

    // top-k: the first k tokens of the order. k >= n_vocab keeps them all.
    if (s.top_k > 0 && static_cast<size_t>(s.top_k) < vocabulary) {
        limit = static_cast<size_t>(s.top_k);
        ensure_sorted(limit);
        std::fill(keep.begin(), keep.end(), 0);
        for (size_t j = 0; j < limit; j++) keep[static_cast<size_t>(order[j])] = 1;
    }

    // top-p: the shortest prefix of what top-k kept whose weight reaches p of
    // the weight top-k kept, which is top-p over the renormalised top-k
    // distribution. The total is summed in ascending token id; the prefix in
    // the order.
    if (s.top_p < 1.0f) {
        double total = 0.0;
        for (size_t i = 0; i < vocabulary; i++) {
            if (keep[i]) total += w[i];
        }
        const double target = static_cast<double>(s.top_p) * total;
        double reached = 0.0;
        size_t taken = 0;
        while (taken < limit) {
            if (taken == sorted) ensure_sorted(std::max<size_t>(64, 2 * sorted));
            reached += w[static_cast<size_t>(order[taken])];
            taken++;
            if (reached >= target) break;
        }
        limit = taken;
        std::fill(keep.begin(), keep.end(), 0);
        for (size_t j = 0; j < limit; j++) keep[static_cast<size_t>(order[j])] = 1;
    }

    // min-p: the tokens whose probability is at least min_p times the largest,
    // which is w >= min_p because the largest has w = 1. Normalisation cancels
    // in the ratio, so it does not matter over which set it is taken.
    if (s.min_p > 0.0f) {
        const double ratio = static_cast<double>(s.min_p);
        for (size_t i = 0; i < vocabulary; i++) {
            if (keep[i] && !(w[i] >= ratio)) keep[i] = 0;
        }
    }

    // The normaliser over what survived, in ascending token id. The maximum
    // survives every filter -- it heads the order and has w = 1 >= min_p --
    // so the total is at least one and its log is finite.
    double total = 0.0;
    for (size_t i = 0; i < vocabulary; i++) {
        if (keep[i]) total += w[i];
    }
    if (!(total >= 1.0) || !std::isfinite(total)) {
        failure = "The kept probability mass is " + std::to_string(total) + ", not a distribution";
        return false;
    }
    const double log_total = std::log(total);

    // The draw: inverse CDF over the kept tokens in ascending token id, with
    // the same running sum that produced the total. u * total is below the
    // final sum unless rounding lifted it onto it; then the last kept token of
    // non-zero weight is taken, which is where the inverse CDF ends. A token of
    // zero weight is never taken: it adds nothing to the running sum.
    const double threshold = u * total;
    double running = 0.0;
    int chosen = -1;
    int last_positive = -1;
    for (size_t i = 0; i < vocabulary; i++) {
        if (!keep[i]) continue;
        if (w[i] > 0.0) last_positive = static_cast<int>(i);
        running += w[i];
        if (threshold < running) {
            chosen = static_cast<int>(i);
            break;
        }
    }
    if (chosen < 0) chosen = last_positive;
    if (chosen < 0) {
        failure = "No token with non-zero probability survived the filters";
        return false;
    }

    token = chosen;
    log_mu = (z[static_cast<size_t>(chosen)] - z_max) - log_total;
    if (log_mu_all) {
        for (size_t i = 0; i < vocabulary; i++) {
            log_mu_all[i] = keep[i] ? (z[i] - z_max) - log_total
                                    : -std::numeric_limits<double>::infinity();
        }
    }
    return true;
}

// rollout_batch owns one llama_batch for the length of a rollout, so every
// exit frees it.
struct rollout_batch {
    llama_batch batch;
    explicit rollout_batch(int capacity) : batch(llama_batch_init(capacity, 0, 1)) {}
    ~rollout_batch() { llama_batch_free(batch); }
    rollout_batch(const rollout_batch&) = delete;
    rollout_batch& operator=(const rollout_batch&) = delete;
};

}  // namespace

extern "C" {

int llama_wrapper_rollout_draw(const float* logits, int n_vocab, llama_wrapper_rollout_sampling sampling,
                               double u, int* out_token, double* out_log_mu, double* out_log_mu_all) {
    if (!logits || !out_token || !out_log_mu) {
        g_last_error = "Logits, a token output and a log mu output are required";
        return -1;
    }
    *out_token = -1;
    *out_log_mu = 0.0;
    if (n_vocab < 1) {
        g_last_error = "The logit row is empty";
        return -1;
    }
    if (!(u >= 0.0 && u < 1.0)) {
        g_last_error = "The uniform must be in [0, 1)";
        return -1;
    }
    try {
        std::string failure;
        if (!validate_rollout_sampling(sampling, failure)) {
            g_last_error = failure;
            return -1;
        }
        rollout_scratch scratch;
        int token = -1;
        double log_mu = 0.0;
        if (!rollout_step(logits, n_vocab, sampling, u, scratch, token, log_mu, out_log_mu_all, failure)) {
            g_last_error = failure;
            return -1;
        }
        *out_token = token;
        *out_log_mu = log_mu;
        return 0;
    } catch (const std::exception& e) {
        g_last_error = "Exception drawing from logits: " + std::string(e.what());
        return -1;
    }
}

int llama_wrapper_rollout_uniforms(unsigned long long seed, int n, double* out) {
    if (!out || n < 1) {
        g_last_error = "An output buffer of at least one uniform is required";
        return -1;
    }
    std::mt19937_64 generator(static_cast<std::uint_fast64_t>(seed));
    for (int i = 0; i < n; i++) out[i] = rollout_uniform(generator);
    return 0;
}

int llama_wrapper_rollout_is_eog(void* ctx, int token) {
    if (!ctx) {
        g_last_error = "Context cannot be null";
        return -1;
    }
    auto wrapper = static_cast<llama_wrapper_context_t*>(ctx);
    if (!wrapper->ctx || !wrapper->model) {
        g_last_error = "Context has been freed";
        return -1;
    }
    const llama_vocab* vocab = llama_model_get_vocab(wrapper->model);
    if (token < 0 || token >= llama_vocab_n_tokens(vocab)) {
        g_last_error = "Token " + std::to_string(token) + " is outside the vocabulary";
        return -1;
    }
    return llama_vocab_is_eog(vocab, token) ? 1 : 0;
}

int llama_wrapper_rollout(void* ctx, const int* prompt, int n_prompt,
                          llama_wrapper_rollout_sampling sampling, unsigned long long seed, int max_tokens,
                          int* out_tokens, double* out_log_mu, int capacity,
                          int* out_generated, int* out_stop_reason, int* out_prompt_decoded) {
    if (!out_generated || !out_stop_reason || !out_prompt_decoded) {
        g_last_error = "The count, stop reason and prompt outputs are required";
        return -1;
    }
    // Written first, so a caller never reads what an earlier call left.
    *out_generated = 0;
    *out_stop_reason = LLAMA_WRAPPER_STOP_UNKNOWN;
    *out_prompt_decoded = 0;
    if (!ctx || !prompt || !out_tokens || !out_log_mu) {
        g_last_error = "Context, prompt and both per-token outputs are required";
        return -1;
    }
    if (n_prompt < 1) {
        g_last_error = "A rollout needs at least one prompt token to condition on";
        return -1;
    }
    if (max_tokens < 1) {
        g_last_error = "max_tokens must be at least one";
        return -1;
    }
    if (capacity < max_tokens) {
        g_last_error = "The output buffers hold " + std::to_string(capacity) +
                       " tokens; the rollout may write " + std::to_string(max_tokens);
        return -1;
    }
    try {
        std::string failure;
        if (!validate_rollout_sampling(sampling, failure)) {
            g_last_error = failure;
            return -1;
        }
        auto wrapper = static_cast<llama_wrapper_context_t*>(ctx);
        if (!wrapper->ctx || !wrapper->model) {
            g_last_error = "Context has been freed";
            return -1;
        }
        // The whole rollout has to fit, decided before anything is decoded: a
        // rollout cut by the context would end for a reason that is neither
        // the model's nor the caller's.
        const long long n_ctx = static_cast<long long>(llama_n_ctx(wrapper->ctx));
        if (static_cast<long long>(n_prompt) + max_tokens > n_ctx) {
            g_last_error = "A prompt of " + std::to_string(n_prompt) + " tokens and " +
                           std::to_string(max_tokens) + " sampled tokens exceed the context of " +
                           std::to_string(n_ctx);
            return -1;
        }
        const llama_vocab* vocab = llama_model_get_vocab(wrapper->model);
        const int n_vocab = static_cast<int>(llama_vocab_n_tokens(vocab));
        if (n_vocab <= 0) {
            g_last_error = "The model reports an empty vocabulary";
            return -1;
        }
        for (int i = 0; i < n_prompt; i++) {
            if (prompt[i] < 0 || prompt[i] >= n_vocab) {
                g_last_error = "Prompt token " + std::to_string(prompt[i]) + " at position " +
                               std::to_string(i) + " is outside the vocabulary";
                return -1;
            }
        }

        // An empty cache for every rollout. A prefix left by another rollout
        // would make this one depend on which ran before it, and one written
        // under another adapter would sample from a mixture of two policies.
        llama_memory_clear(llama_get_memory(wrapper->ctx), true);
        // The prefix bookkeeping goes with the cache it describes, so a later
        // generate call does not skip tokens that are no longer there.
        wrapper->cached_tokens.clear();

        // The prompt, in windows of n_batch, with logits on its last token
        // only: one position of logits is n_vocab floats, and n_outputs_max is
        // an assertion rather than an error.
        const int n_batch = std::max(1, static_cast<int>(llama_n_batch(wrapper->ctx)));
        {
            rollout_batch window(std::min(n_batch, n_prompt));
            for (int start = 0; start < n_prompt; start += n_batch) {
                const int count = std::min(n_batch, n_prompt - start);
                for (int i = 0; i < count; i++) {
                    const int position = start + i;
                    window.batch.token[i] = prompt[position];
                    window.batch.pos[i] = position;
                    window.batch.n_seq_id[i] = 1;
                    window.batch.seq_id[i][0] = 0;
                    window.batch.logits[i] = position == n_prompt - 1 ? 1 : 0;
                }
                window.batch.n_tokens = count;
                const int32_t decoded = llama_decode(wrapper->ctx, window.batch);
                if (decoded != 0) {
                    g_last_error = "llama_decode failed with " + std::to_string(decoded) +
                                   " on the prompt window starting at " + std::to_string(start);
                    return -1;
                }
            }
        }
        *out_prompt_decoded = n_prompt;

        std::mt19937_64 generator(static_cast<std::uint_fast64_t>(seed));
        rollout_scratch scratch;
        rollout_batch next(1);
        for (int t = 0; t < max_tokens; t++) {
            const float* logits = llama_get_logits_ith(wrapper->ctx, -1);
            if (!logits) {
                g_last_error = "No logits after decoding position " + std::to_string(n_prompt + t - 1) +
                               "; a context created for embeddings does not produce them";
                return -1;
            }
            // One uniform per position, drawn before anything can end the
            // loop, so position t always reads the t-th uniform of the seed.
            const double u = rollout_uniform(generator);
            int token = -1;
            double log_mu = 0.0;
            if (!rollout_step(logits, n_vocab, sampling, u, scratch, token, log_mu, nullptr, failure)) {
                g_last_error = "Position " + std::to_string(n_prompt + t) + ": " + failure;
                return -1;
            }
            out_tokens[t] = token;
            out_log_mu[t] = log_mu;
            *out_generated = t + 1;

            // The end-of-generation token is kept in the rollout: it was
            // sampled, it has a log mu, and the trainer has to learn to emit it.
            if (llama_vocab_is_eog(vocab, token)) {
                *out_stop_reason = LLAMA_WRAPPER_STOP_EOG;
                return 0;
            }
            // The last token is never decoded: nothing is sampled after it.
            if (t + 1 == max_tokens) {
                *out_stop_reason = LLAMA_WRAPPER_STOP_LENGTH;
                return 0;
            }
            next.batch.token[0] = token;
            next.batch.pos[0] = n_prompt + t;
            next.batch.n_seq_id[0] = 1;
            next.batch.seq_id[0][0] = 0;
            next.batch.logits[0] = 1;
            next.batch.n_tokens = 1;
            const int32_t decoded = llama_decode(wrapper->ctx, next.batch);
            if (decoded != 0) {
                *out_stop_reason = LLAMA_WRAPPER_STOP_DECODE_FAILED;
                g_last_error = "llama_decode failed with " + std::to_string(decoded) +
                               " on sampled token " + std::to_string(t) + " at position " +
                               std::to_string(n_prompt + t);
                return -1;
            }
        }
        // Unreachable: the loop returns at max_tokens.
        *out_stop_reason = LLAMA_WRAPPER_STOP_LENGTH;
        return 0;
    } catch (const std::exception& e) {
        g_last_error = "Exception sampling a rollout: " + std::string(e.what());
        return -1;
    }
}

}  // extern "C"
