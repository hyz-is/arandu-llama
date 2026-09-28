#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stdint.h>

// Progress callback type (matches llama.cpp signature)
typedef bool (*llama_progress_callback_wrapper)(float progress, void* user_data);

// Model parameters for loading
typedef struct {
    int n_ctx;              // Context size
    int n_batch;            // Batch size (logical)
    int n_ubatch;           // Micro-batch size (physical); 0 = match n_batch
    int n_gpu_layers;       // Number of GPU layers
    int n_threads;          // Number of threads for generation (per token)
    int n_threads_batch;    // Number of threads for batch processing (prompt)
    int n_parallel;         // Number of parallel sequences (for batch embeddings)
    bool f16_memory;        // Use F16 for memory
    bool mlock;            // Memory lock
    bool mmap;             // Memory mapping
    bool embeddings;       // Enable embeddings
    const char* main_gpu;   // Main GPU
    const char* tensor_split; // Tensor split
    const char* kv_cache_type; // KV cache quantization: "f16", "q8_0", "q4_0"
    const char* flash_attn;    // Flash Attention: "auto", "enabled", "disabled"
    bool disable_progress_callback;           // For silent loading
    llama_progress_callback_wrapper progress_callback;  // Custom callback
    void* progress_callback_user_data;        // User data for callback
} llama_wrapper_model_params;

// Generation parameters
typedef struct {
    const char* prompt;
    int max_tokens;
    int seed;
    const char** stop_words;
    int stop_words_count;
    int n_draft;           // For speculative sampling
    bool debug;
    uintptr_t callback_handle; // Handle to Go callback
    bool enable_prefix_caching; // Enable KV cache reuse for matching prefixes

    // Basic sampling parameters
    float temperature;
    int top_k;
    float top_p;
    float min_p;
    float typ_p;
    float top_n_sigma;
    int min_keep;

    // Repetition penalties
    int penalty_last_n;
    float penalty_repeat;
    float penalty_freq;
    float penalty_present;

    // DRY sampling
    float dry_multiplier;
    float dry_base;
    int dry_allowed_length;
    int dry_penalty_last_n;
    const char** dry_sequence_breakers;
    int dry_sequence_breakers_count;

    // Dynamic temperature
    float dynatemp_range;
    float dynatemp_exponent;

    // XTC sampling
    float xtc_probability;
    float xtc_threshold;

    // Mirostat sampling
    int mirostat;
    float mirostat_tau;
    float mirostat_eta;

    // Other parameters
    int n_prev;
    int n_probs;
    bool ignore_eos;

    // Optional outputs, written when not NULL and ignored otherwise; see
    // llama_wrapper_stop_reason. They are written on every return, a failed
    // one included, so a caller never reads what an earlier call left.
    int* n_generated_out;  // tokens whose pieces make up the returned text
    int* stop_reason_out;  // a llama_wrapper_stop_reason
} llama_wrapper_generate_params;

// Why a generation loop returned. The returned text alone cannot say: an
// answer cut at the token ceiling, one that ended with the model's own
// end-of-generation token and one abandoned after a failed decode are all a
// string.
//
// The plain loop (llama_wrapper_generate, llama_wrapper_generate_with_tokens)
// reports every reason exactly. The speculative loop reports EOG, STOP_WORD,
// CALLBACK and DECODE_FAILED exactly and UNKNOWN otherwise: its ceiling is
// compared against the length of the text in bytes rather than a token count,
// so reaching it is not "max_tokens reached", and an empty verification is not
// any of the others. Its n_generated_out is exact.
typedef enum {
    LLAMA_WRAPPER_STOP_UNKNOWN = 0,
    LLAMA_WRAPPER_STOP_EOG = 1,            // the model sampled an end-of-generation token
    LLAMA_WRAPPER_STOP_WORD = 2,           // the text contains one of stop_words
    LLAMA_WRAPPER_STOP_CALLBACK = 3,       // the token callback returned false
    LLAMA_WRAPPER_STOP_LENGTH = 4,         // max_tokens tokens were generated (128 when it is 0)
    LLAMA_WRAPPER_STOP_DECODE_FAILED = 5   // llama_decode failed on a sampled token
} llama_wrapper_stop_reason;

// Callback for streaming tokens
typedef bool (*llama_wrapper_token_callback)(const char* token);

// Logging initialization
void llama_wrapper_init_logging();

// Model management
void* llama_wrapper_model_load(const char* model_path, llama_wrapper_model_params params);
void llama_wrapper_model_free(void* model);

// Context management (kept for API compatibility)
void* llama_wrapper_context_create(void* model, llama_wrapper_model_params params);
void llama_wrapper_context_free(void* ctx);

// Text generation
char* llama_wrapper_generate(void* ctx, llama_wrapper_generate_params params);
char* llama_wrapper_generate_with_tokens(void* ctx, const int* tokens, int n_tokens, int prefix_len, llama_wrapper_generate_params params);

// Speculative generation with draft model
char* llama_wrapper_generate_draft(void* ctx_target, void* ctx_draft, llama_wrapper_generate_params params);
char* llama_wrapper_generate_draft_with_tokens(void* ctx_target, void* ctx_draft, const int* tokens, int n_tokens, int target_prefix_len, int draft_prefix_len, llama_wrapper_generate_params params);

// Tokenization
int llama_wrapper_tokenize(void* ctx, const char* text, int* tokens, int max_tokens);

// Tokenise with dynamic allocation (C manages memory)
// Allocates exact size needed for tokens - caller must free with llama_wrapper_free_tokens
// tokens: output parameter for allocated token array pointer
// count: output parameter for number of tokens (or -1 on error)
void llama_wrapper_tokenize_alloc(void* ctx, const char* text, int** tokens, int* count);

// Free tokens allocated by llama_wrapper_tokenize_alloc
void llama_wrapper_free_tokens(int* tokens);

// Embeddings
int llama_wrapper_embeddings(void* ctx, const char* text, float* embeddings, int max_embeddings);

// Batch embeddings - process multiple texts efficiently
// texts: array of text strings to embed
// n_texts: number of texts in the array
// embeddings: output buffer (must have space for n_texts * n_embd floats)
// n_embd: embedding dimension from model (llama_model_n_embd)
// Returns number of embeddings generated (should equal n_texts), or -1 on error
int llama_wrapper_embeddings_batch(void* ctx, const char** texts, int n_texts, float* embeddings, int n_embd);

// Utility functions
void llama_wrapper_free_result(char* result);
const char* llama_wrapper_last_error();
int llama_wrapper_get_cached_token_count(void* ctx);

// Get model's native maximum context length
int llama_wrapper_get_model_context_length(void* model);

// Get model's embedding dimension
int llama_wrapper_model_n_embd(void* model);

// Chat template support
const char* llama_wrapper_get_chat_template(void* model);

// Reasoning content parsing
typedef enum {
    REASONING_FORMAT_NONE = 0,
    REASONING_FORMAT_AUTO = 1,
    REASONING_FORMAT_DEEPSEEK_LEGACY = 2,
    REASONING_FORMAT_DEEPSEEK = 3
} llama_wrapper_reasoning_format;

typedef struct {
    const char* content;
    const char* reasoning_content;  // NULL if empty
} llama_wrapper_parsed_message;

// Parse model output into content and reasoning with the parser the render
// returned. parser is the serialized common_peg_arena and generation_prompt
// the text the render appended to open the answer, both from
// llama_wrapper_chat_templates_render for the same messages and options. At
// the pinned llama.cpp every parse is a PEG parse: with no parser the output is
// read as content, reasoning included, whatever format says, which is why an
// empty parser also drops generation_prompt -- the pure-content parser would
// put it in the content. is_partial is true for output that stopped before the
// model ended it (streaming, or the token ceiling), false for a complete one.
// Returns NULL on error. Free result with llama_wrapper_free_parsed_message()
llama_wrapper_parsed_message* llama_wrapper_parse_reasoning(
    const char* text,
    bool is_partial,
    llama_wrapper_reasoning_format format,
    int chat_format,
    const char* generation_prompt,
    const char* parser
);

void llama_wrapper_free_parsed_message(llama_wrapper_parsed_message* msg);

// Chat templates rendered by llama.cpp's Jinja engine (common/chat).
// init takes a model, a template override, or both; with no model the override
// is parsed alone and bos_token/eos_token are empty. With a model and no
// override a model without a template is given ChatML, so check
// llama_wrapper_get_chat_template first. Returns NULL on error.
void* llama_wrapper_chat_templates_init(void* model, const char* template_override);
void llama_wrapper_chat_templates_free(void* templates);
// Renders the messages and returns the prompt (free with
// llama_wrapper_free_result), writing the detected common_chat_format to
// format_out when it is not NULL. reasoning_contents is NULL or holds one
// entry per message, NULL or empty for a message without reasoning, and each
// other entry becomes that message's reasoning_content, which a template that
// writes thinking into an assistant turn reads. With add_generation_prompt
// false the render ends with the last message, as a recorded conversation
// does. enable_thinking is -1 for llama-server's
// default, 0 or 1 to set it; kwarg values are JSON text. reasoning_format
// decides the parser llama.cpp builds: with NONE its parser leaves reasoning in
// the content. When not NULL, generation_prompt_out and parser_out receive
// what llama_wrapper_parse_reasoning needs for output generated from this
// prompt, each a string to free with llama_wrapper_free_result. Returns NULL on
// error with llama_wrapper_last_error set and both left NULL; there is no
// fallback formatter.
char* llama_wrapper_chat_templates_render(
    void* templates,
    const char** roles,
    const char** contents,
    const char** reasoning_contents,
    int n_messages,
    bool add_generation_prompt,
    int enable_thinking,
    llama_wrapper_reasoning_format reasoning_format,
    const char** kwarg_names,
    const char** kwarg_values,
    int n_kwargs,
    int* format_out,
    char** generation_prompt_out,
    char** parser_out
);

// Model metadata access
const char* llama_wrapper_model_meta_string(void* model, const char* key);
int llama_wrapper_model_meta_count(void* model);

// GPU information
typedef struct {
    int device_id;
    char device_name[256];
    int free_memory_mb;
    int total_memory_mb;
} llama_wrapper_gpu_info;

int llama_wrapper_get_gpu_count();
bool llama_wrapper_get_gpu_info(int device_id, llama_wrapper_gpu_info* info);

// Model runtime information
typedef struct {
    int n_ctx;           // Context size
    int n_batch;         // Batch size
    int kv_cache_size_mb; // Estimated KV cache memory usage
    int gpu_layers;      // GPU layers loaded
    int total_layers;    // Total layers in model
} llama_wrapper_runtime_info;

void llama_wrapper_get_runtime_info(void* model, void* ctx, const char* kv_cache_type, llama_wrapper_runtime_info* info);

// LoRA adapters, and scoring a sequence the model did not write.
//
// Implemented in wrapper_adapter.cpp. They exist for training rather than
// inference: an adapter that can be moved between two forward passes, and a loss
// of the same quantity a backpropagating trainer reports, so the two can be
// compared.

// Load an adapter against the model it will be applied to. Returns NULL on
// failure; free with llama_wrapper_adapter_free.
void* llama_wrapper_adapter_load(void* model, const char* path);
void llama_wrapper_adapter_free(void* adapter);

// Read one GGUF metadata value from a loaded adapter, into buffer.
// Returns the length written, or -1 when the key is absent.
// "adapter.lora.alpha" is the one that matters: llama.cpp scales by
// adapter_scale * alpha / rank when it is non-zero and by adapter_scale alone
// when it is not, so an adapter that lost the key is applied at a different
// magnitude than the trainer applied it, silently.
int llama_wrapper_adapter_meta(void* adapter, const char* key, char* buffer, int buffer_size);

// Set the adapters applied to a context, with one scale each. Passing zero
// adapters removes all of them. Returns the number applied, or -1 on failure.
int llama_wrapper_adapters_set(void* ctx, void** adapters, float* scales, int n_adapters);

// Score a token sequence by teacher forcing.
// n_skip drops that many leading tokens from the average, so the loss can be
// over a completion rather than the prompt that led to it.
// The sequence is decoded in windows bounded by n_batch and by a ceiling on
// logit memory: one flagged position costs n_vocab floats, which is 993280
// bytes for this model, and n_outputs_max is an assertion rather than an error.
// Writes the summed negative log likelihood and the number of positions scored.
// Returns 0 on success, -1 on failure.
int llama_wrapper_score(void* ctx, const int* tokens, int n_tokens, int n_skip,
                        double* out_sum, int* out_count);

// Describe the loaded model in the terms it was quantised in, e.g.
// "qwen35 27B Q4_K - Medium". Returns the length written, or -1 on failure.
// The representation is read from the model rather than named by the caller,
// because a caller that can name the bit-width can name it wrongly, and a
// measurement labelled with the wrong representation is worse than an
// unlabelled one: it enters a comparison and moves the answer.
int llama_wrapper_model_describe(void* model, char* buffer, int buffer_size);

// Width of one captured row: llama_model_n_embd_out, which is n_embd unless the
// architecture declares a separate output width (2048 for SmolLM3-3B). The
// embeddings buffer llama.cpp fills is sized by this and not by n_embd, so the
// caller sizes its buffer by this too. Returns -1 on a null model.
int llama_wrapper_model_n_embd_out(void* model);

// Capture the per-token final representation of a token sequence: the hidden
// state after the final RMS normalisation and before the output projection,
// which at the pinned llama.cpp commit is the graph tensor named "result_norm"
// (res->t_embd). It is read through llama.cpp's own copy of that tensor: the
// embeddings flag is switched on for the decode and restored to
// embeddings_after on every exit, exceptions included. Measured bitwise equal
// to the tensor observed with the scheduler callback, 14/14 rows.
//
// Clears the KV cache and the prefix bookkeeping first, as llama_wrapper_score
// does: a cache filled under another adapter would capture a mixture of two
// models. Decodes in the same windows as scoring, because logits are reserved
// per output row in embeddings mode too. Every row of a window is an output
// row: embeddings mode overrides a partial selection anyway, and a subset
// decode differs from the full-window rows by up to 1.9e-6, so the rule is part
// of the capture version.
//
// positions must be strictly increasing, each in [0, n_tokens). out receives
// n_positions rows of n_embd_out floats, row k for positions[k]; out_floats
// must equal n_positions * n_embd_out exactly. A non-finite value anywhere in a
// row is a failure, not a result. Refused when the context pools
// (pooling_type != NONE): the copied tensor would be a sequence embedding.
// Returns 0 on success, -1 on failure with llama_wrapper_last_error set; on
// failure out is unspecified.
int llama_wrapper_capture_final(void* ctx, const int* tokens, int n_tokens,
                                const int* positions, int n_positions,
                                bool embeddings_after,
                                float* out, long long out_floats);

// The six tensors of one projection's LoRA arithmetic, in the order the graph
// computes them. Every buffer is owned by the caller and its length is checked
// exactly, never "at least": a buffer sized for another width is a buffer for
// another capture.
//
//   x       the projection's input,     n_in  x n_tokens
//   h       A * x,                      rank  x n_tokens
//   u_pre   B * h, before the scale,    n_out x n_tokens
//   u       the scaled contribution,    n_out x n_tokens
//   y_base  W * x, without the adapter, n_out x n_tokens
//   y       y_base + u,                 n_out x n_tokens
//
// u divided by u_pre is the scale llama.cpp actually applied, read off the graph
// rather than inferred, which is what a comparison of two adapters against each
// other cannot give.
typedef struct llama_wrapper_lora_capture {
    float* x;
    long long x_floats;
    float* h;
    long long h_floats;
    float* u_pre;
    long long u_pre_floats;
    float* u;
    long long u_floats;
    float* y_base;
    long long y_base_floats;
    float* y;
    long long y_floats;
} llama_wrapper_lora_capture;

// Capture the LoRA arithmetic of one projection, on a context of its own.
//
// The nodes are found the way tools/imatrix finds its tensors: by the operation
// and by the name of the WEIGHT operand, src[0]->name, which is the GGUF tensor
// name and is stable. The graph node's own label is not usable -- llama.cpp
// names build_lora_mm's intermediates "node_<N>", an ordinal that moves with the
// graph shape, and on SmolLM3 the label "ffn_out-<il>" is even emitted for two
// different tensors of the same layer.
//
// The context is created here and destroyed before returning, because cb_eval
// can only be installed through llama_context_params at llama_init_from_model
// and llama.h offers no setter afterwards. That is also why installing it costs
// nothing that matters: the synchronisation it forces on every split lasts as
// long as this one decode, not the life of a context that serves inference.
//
// module is the projection's weight name without a suffix, e.g.
// "blk.35.ffn_down.weight"; the factors are looked for as module + ".lora_a"
// and module + ".lora_b". adapter may be null, in which case only y_base and x
// are filled and the four LoRA buffers must be null with zero lengths.
//
// A non-finite value anywhere is a failure, not a result. Returns 0 on success,
// -1 on failure with llama_wrapper_last_error set.
int llama_wrapper_capture_lora(void* model, const char* module,
                               void* adapter, float adapter_scale,
                               const int* tokens, int n_tokens, int n_ctx,
                               llama_wrapper_lora_capture* out);

// Load owned teacher weights without mmap. Zero GPU layers additionally clears
// the GPU device list, so CPU execution never initializes a GPU backend.
// Released through llama_wrapper_model_free. Tensor splitting is unsupported.
void* llama_wrapper_teacher_model_load(const char* path, llama_wrapper_model_params options);

// Inspect the exact feature tensor supported by the pinned teacher graph.
// result_norm is final normalized hidden state; l_out-N is a decoder output
// only for the explicitly admitted llama/qwen35 builders. No aliasing of names.
int llama_wrapper_teacher_geometry(void* model, const char* tensor, int* vocabulary, int* width);

// Capture supplied gold-prefix rows in a fresh, adapter-free causal context.
// Position p predicts tokens[p+1]; positions must strictly increase in [0,n-2].
// Every decode-window row is an output row. Features are read from the exact
// named graph tensor through the native callback; no token is sampled.
// Probabilities use full-vocabulary softmax at temperature 1, then top-k ordered
// by descending probability and ascending token ID on ties. Retained mass is
// the sum of these unrenormalized probabilities. Exact output lengths required.
// context_tokens must be a positive multiple of 256, the pinned KV alignment.
// max_window_bytes bounds window logits, features and top-k scratch, excluding weights, KV
// and decoder workspaces. The caller admits those separately. Returns 0 or -1;
// errors set llama_wrapper_last_error and invalidate every output buffer.
int llama_wrapper_capture_teacher(void* model, const char* tensor,
                                  const int* tokens, int n_tokens, const int* positions, int n_positions,
                                  int top_k, int context_tokens, int window_tokens, int threads, bool cpu_only,
                                  long long max_window_bytes, int* out_ids, double* out_probabilities,
                                  long long top_elements, double* out_mass, float* out_features, long long feature_elements);

#ifdef __cplusplus
}
#endif
