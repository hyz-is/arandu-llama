# Changelog

Everything worth knowing about a release of Arandu Llama is recorded here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
the versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

A published module version is immutable: Go serves it from the proxy forever, so
a release is corrected by another release and never by moving a tag.

## [Unreleased]

### Added

- `RenderChatTemplate` renders a conversation with a Jinja chat template without
  loading a model, and `ChatTemplateEngine` names the formatter a prompt came
  from.
- `decoder.CompletionLoss` and `local.MeasureLosses` read the completion loss
  of existing checkpoints without computing a gradient.
- `decoder.AdapterCoverage` declares which projections carry a LoRA pair, by
  layer kind: full attention, the linear-attention (DeltaNet) projections and
  the MLP of every layer. `FullAdapterCoverage` declares all of them, and
  `DeriveInitialReference` derives the initial adapter reference for a declared
  coverage. A nil coverage keeps `q_proj` and `v_proj` of the full-attention
  layers with every earlier digest unchanged.
- `ParseChatOutput` reads a recorded chat output back into `Content` and
  `ReasoningContent` without a model, the way `Chat` does. `ChatResponse` gains
  `Output`, `FinishReason` (`FinishReasonStop`, `FinishReasonLength`) and
  `GeneratedTokens`, counted in the generation loop. `ChatOptions` gains `MinP`.
- `ChatMessage.ReasoningContent` hands an assistant turn's thinking to the
  template as `reasoning_content`; an empty value renders the message as
  before. `ChatOptions.AddGenerationPrompt` set to false renders a conversation
  that ends with its last message, as a training row needs; `Chat`,
  `ChatStream` and `ParseChatOutput` refuse it.
- `training/trajectory` turns a verified teacher trajectory (question,
  reasoning, answer) into the row `training/local` reads. The student's own
  template renders the prompt with thinking on and the completion through the
  end of turn, and the student's tokenizer encodes both, with -100 labels over
  the prompt. `Report.JointAgrees` says whether encoding the whole text agrees
  with the prompt-then-completion boundary.
- `training/bridge` re-expresses a teacher's next-token distribution over a
  student vocabulary with another tokenizer, by Byte-Prefix Marginalization
  (arXiv 2607.22334). It is pure Go and decodes the exact bytes of byte-level
  BPE and SentencePiece-convention vocabularies from a GGUF or a
  `tokenizer.json`. It aligns the realized segmentations and labels each row
  exact or lower bound. Mass is conserved: the cells plus the residual, stop,
  special and uncovered mass sum to one. It is not wired to `fusioncache` or
  to training.
- `training/tokenizer` measures a vocabulary from a `tokenizer.json` or from
  GGUF metadata without reading tensors (`ReadVocabularyJSON`,
  `ReadVocabularyGGUF`, `Vocabulary.Digest`, `SameVocabulary`).
  `training/fusioncache` admits an identity mapping over a measured identical
  vocabulary with a declared common id range (`TokenMapping.Vocabulary`,
  `VocabularyMapping`, `ValidateMapping`). A gold or teacher token outside the
  range refuses the cache.
- `training/local` takes an optional `Recipe.Distillation`: teacher-forced
  logit distillation through `FusionCompletionGradient`, under the same AdamW,
  checkpoints and stage, with `Config.CacheDir` and
  `StageConfig.CacheDirectory`. When alpha is above zero, step manifests gain
  `distillation_loss_before`, `teacher_mass` and `teacher_losses`. Recipes,
  mappings and manifests without these fields keep their digests and bytes.
- `adapter.ExportQwen35LoRA` writes an F32 LoRA checkpoint from `training/local`
  as the GGUF adapter the pinned llama.cpp loads over a qwen35 base: pairs
  `blk.N.<tensor>.weight.lora_a` and `.lora_b` with `adapter.lora.alpha`, each
  target checked against a base tensor of the same geometry. The
  linear-attention value heads are reordered to the tiled layout the converter
  writes, in the V rows of `in_proj_qkv`, the rows of `in_proj_z`, `in_proj_b`
  and `in_proj_a`, and the columns of `out_proj`. Unknown modules, the MTP
  block, a rank other than the recipe's and non-finite factors are refused.
  `WriteInitialLoRA` is unchanged, byte for byte.
- `decoder.OnPolicyCompletionGradient` distils on a completion the student
  sampled, with three objectives over the teacher's top-k and one complement
  cell:
  - `fkl-complement` puts no mass on the realized token. It equals
    `FusionCompletionGradient` with one teacher of weight one when every
    retained mass is exactly one.
  - `jsd-complement` is the generalized JSD, with beta in (0, 1).
  - `rkl-sampled` is the reverse KL at the realized token, with a truncated
    per-token importance weight. It requires the teacher's and the sampler's
    log probability of every token.

  The loss is the mean per completion token; `lossScale` scales only the
  cotangent. `decoder.CompletionTokenLogProbabilities` reads the student's
  log-softmax at every completion token.
- `Context.SampleRollout` samples a continuation of a tokenised prompt from an
  empty KV cache. For every token it returns the id and log mu: the float64
  log probability under the distribution the token was actually drawn from,
  filters included.
  - `RolloutOptions` requires temperature, top-k, top-p, min-p, seed and a
    token ceiling. Nothing is defaulted.
  - The temperature is applied before the filters, and top-p reads the
    renormalised top-k set.
  - The end-of-generation token is kept, `Rollout.Stop` names why the rollout
    ended, and a failed decode is an error.
  - A rollout is labelled with the adapter digests and scales, the
    quantisation, the KV cache type and `RolloutVersion`.

  `DrawFromLogits` and `RolloutUniforms` run the same arithmetic on a supplied
  logit row.
- `training/router` is the TCAP teacher router. Both entry points work from a
  `Config` pinned by its canonical digest:
  - `Route` chooses at most k teachers for an example before any of them
    generates.
  - `Shortlist` keeps a per-capability shortlist offline.

  License, pinned artifact, qualification and a measured competence cell are
  gates: a teacher that fails one is excluded with the reasons recorded.
  Selection is sequential, with lineage redundancy conditional on the teachers
  already chosen; ties go to the lower id, and a weight that underflows is
  refused. The serializable `Decision` carries the order, each round's
  standings, the factors, the softmax weights and the configuration and input
  digests. No entry point accepts a teacher output.
- `fusioncache.CapabilityPackage` (schema 1) is the immutable provenance of one
  admitted training unit. It records:
  - the capability, the example and the digest of its rendered context;
  - the versioned verifier and its verdict;
  - the consulted teachers, each with its pinned model, license, score, weight
    and verified outcome;
  - the admitted signal and its position mask;
  - a lineage snapshot and the Protection cohort;
  - an optional router decision;
  - an index of every cited artifact.

  `EncodePackage` gives a canonical encoding whose SHA-256 is the package
  digest. `ValidatePackage` and `DecodePackage` refuse incomplete, ambiguous or
  incoherent packages, including duplicate, case-folded and unknown JSON keys.
  `Supersede` makes the next revision.
- `fusioncache.PackageStore` freezes packages under their digest with an
  atomic install that never replaces a file. A correction requires its
  predecessor in the store, and a package can be corrected only once.
  `History`, `Successor`, `FreezeManifest` and `ReadManifest` read the chain
  back.
- `trajectory.Example.AnswerOnly` declares a row of the answer alone, with no
  trajectory.
  - Its prompt is the one a trajectory row for the same messages gets, with
    thinking on.
  - Its completion is what the template writes for an assistant turn with empty
    reasoning and the answer as content. Through the Ornith's template that is
    `"\n</think>\n\n#### 42<|im_end|>"`.
  - `Render` refuses any other text in the completion, including reasoning the
    template extracts from the content.
  - Reasoning in an answer-only example is refused, and empty reasoning is still
    refused in an example that does not declare it.

  With the Ornith's tokenizer, an answer-only row reports `JointAgrees` false:
  the prompt's last newline and the completion's first one merge when the whole
  text is tokenized.

### Fixed

- Each training backward runs on one OS thread, so a local SFT step on the CPU
  repeats bit for bit.
- With a `ReasoningFormat` other than none, `Chat` and `ChatStream` parse the
  output with the parser llama.cpp builds for the template, and with its
  generation prompt. gpt-oss's analysis channel, Qwen3.8's thinking block and
  Gemma 4's thought channel now land in `ReasoningContent` instead of
  `Content`.

### Changed

- Chat prompts are rendered with the model's own GGUF template through
  llama.cpp's Jinja engine for every family. The heuristic legacy formatter is
  gone and nothing falls back to it: a template that does not render is an
  error. `EnableThinking` and `ChatTemplateKwargs` now reach the template, and
  thinking follows the llama-server default when unset.
- `Chat` returns an error where it used to return text: when the output does
  not parse under a `ReasoningFormat` other than none, when `llama_decode`
  fails mid-generation, and `ctx.Err()` when the context is cancelled. It no
  longer returns the raw output as `Content` after a failed parse. A
  `ReasoningFormat` outside the four constants is an error.

- Training uses architecture and role contracts, with installation-owned model
  identities and recipes supplied explicitly. Existing calculations and artifact
  integrity checks remain part of the implementation.
- MX model admission uses an immutable caller-owned catalog. The library no
  longer selects a production model or publishes operational model pins.
- Operational identifiers have been removed from this document's current text;
  previously published versions remain historical releases.
- **Breaking.** `Llama` embeds the non-generic `model.Model`, and its table is
  declared once beside it with `model.NewTable`. `Llamas` takes a `model.DB`
  and returns the generated `*LlamaQuery`; `Get` returns `LlamaCollection` and
  `New` replaces `NewInstance(nil, false)`. The fields and methods
  `model.Model[Llama]` promoted onto `Llama` are gone, and `Exists` is a
  method. `UPGRADE.md` names every symbol.
- Requires Hesape `v0.48.0` and Framework `v0.50.2`; `arandu.mod.toml` declares
  `framework = ">= 0.50"`. The store route reads its fields from the body of
  the `POST` only, as Hesape now reads every `POST`. Routes, migrations,
  actions, policy decisions and tenant scoping are unchanged.

## [0.5.0] - 2026-09-15

### Added

- A pinned seven-shard Q2_K runtime candidate with exact source revision and
  model manifest, while retaining the then-configured Q4 default.
- `MXModelRecipeForDigest` recovers an admitted model recipe from its persisted
  model digest and rejects empty, unknown, or ambiguous identities.

### Fixed

- CI now builds the pinned CPU llama.cpp archives before tests that link the
  in-process cgo binding. The release verifier does the same outside the tagged
  archive, then copies only generated archives into the verification tree.
- Release verification no longer assumes a package-local `configure.go` exists.

## [0.4.0] - 2026-09-13

### Added

- A versioned `backends/mx` subprocess package beside the existing in-process
  llama.cpp package. It admits only the qualified SM75 build at revision
  `a245214d`, verifies all three executable digests and the model manifest, and
  returns typed process specifications without accepting shell text or paths
  from a request.
- Bounded RPC and generation specifications for the measured Tayi topology:
  twenty unique private or CGNAT RPC endpoints, two CUDA devices per RPC host,
  64 scheduler backends, fixed context and deterministic sampling, all under a
  process deadline.
- A checked-in backend manifest and reproducible Makefile for the qualified
  `llama-cli`, `llama-server` and `ggml-rpc-server` build.
- `Model.CaptureLoRA` records the input, low-rank projection, unscaled and
  scaled contribution, base projection and final projection for each token.
  `LoRACapture.AppliedScale` exposes the factor the graph actually applied.

## [0.3.1] - 2026-09-10

### Fixed

- Scoring refuses a mean equal to `log(n_vocab)`, the uniform-distribution
  result observed intermittently when CUDA split the measured model over two
  devices, instead of returning it as a plausible loss.

## [0.3.0] - 2026-09-10

### Added

- `Context.CaptureFinal` returns requested per-token `result_norm` rows as
  finite `f32` values, labelled by token, quantisation, policy and adapter
  digests so captures from different conditions cannot share an identity.

## [0.2.1] - 2026-09-10

### Fixed

- Worker construction failures return an error instead of aborting the process.

## [0.2.0] - 2026-09-10

The first release of this repository. It starts at `0.2.0` rather than `0.1.0`
because the proxy already holds `v0.1.0`, `v0.1.1` and `v0.1.2` of this module
path from a previous repository, with checksums that do not match this history.
Reusing those numbers would hand a consumer a checksum mismatch, which reads as
an attack rather than as a version change. **Those three versions are not this
code and must never be tagged again.**

### The package

- An Arandu module: one entity, one policy that denies everything until an
  action is opened, one service that owns the database handle, and the routes
  behind it.
- Five actions, all closed by default: `LlamaView`, `LlamaList`,
  `LlamaCreate`, `LlamaUpdate` and `LlamaDelete`. Opening one is the
  installer's decision, taken in their own policy registration.
- One migration, `20260823_0001_create_llamas`, which creates the `llamas`
  table. `aru migrate` is a pipeline step before the rollout and never runs at
  boot: with N replicas starting together, N migrations race.
- An inference engine in the same package: llama.cpp linked through cgo, with
  the submodule pinned at `90c26fcd` — build b10675.

### What it does that an inference binding does not

- **Adapters over a quantised base, without merging.** Loaded once, applied per
  context, scale changed between forward passes in 0.4 ms.
- **Teacher-forced scoring.** The loss of a completion the model did not write,
  of the same quantity a backpropagating trainer reports. Windowed and bounded
  by `n_batch`, because flagging every position of a 248320-token vocabulary
  asks for 2.46 GB of logits and exceeding `n_outputs_max` is a `GGML_ASSERT`
  that aborts the process.
- **A reading that keeps.** Sum, count, policy digest, quantisation and
  duration as a row, so two measurements can be put side by side.

### Refusals worth knowing about

- A reading whose adapter could not be hashed is **refused**, not stored. Two
  adapters labelled `unknown` compare equal, which is the one confusion the
  digest exists to prevent.
- `Score` returns a sum and a count and never a mean, so a caller combining
  examples weights each by its length.
- Scoring clears the KV cache before decoding, because the cache was filled
  under whatever adapter was applied when it was written.

### Documentation

- [docs/building.md](docs/building.md) — the archives, the linkage modes, the
  linker flag, upgrading llama.cpp.
- [docs/scoring.md](docs/scoring.md) — adapters and scoring, and every place a
  plausible number can come out wrong.
- `.agents/skills/` — six procedures, one per situation.

[Unreleased]: https://github.com/tayi-ai/arandu-llama/compare/v0.5.0...HEAD
[0.5.0]: https://github.com/tayi-ai/arandu-llama/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/tayi-ai/arandu-llama/compare/v0.3.1...v0.4.0
[0.3.1]: https://github.com/tayi-ai/arandu-llama/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/tayi-ai/arandu-llama/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/tayi-ai/arandu-llama/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/tayi-ai/arandu-llama/releases/tag/v0.2.0
