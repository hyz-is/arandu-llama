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

### Changed

- Chat prompts are rendered with the model's own GGUF template through
  llama.cpp's Jinja engine for every family. The heuristic legacy formatter is
  gone and nothing falls back to it: a template that does not render is an
  error. `EnableThinking` and `ChatTemplateKwargs` now reach the template, and
  thinking follows the llama-server default when unset.

- Training uses architecture and role contracts, with installation-owned model
  identities and recipes supplied explicitly. Existing calculations and artifact
  integrity checks remain part of the implementation.
- MX model admission uses an immutable caller-owned catalog. The library no
  longer selects a production model or publishes operational model pins.
- Operational identifiers have been removed from this document's current text;
  previously published versions remain historical releases.

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
