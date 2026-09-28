# Upgrade Guide

## Unreleased

Chat prompts change for templates the legacy formatter read as a known format:
they are now the model's own Jinja template, which may add a reasoning system
turn or a `<think>` block. Anything that stores prompt boundaries or token
counts produced through `FormatChatPrompt` must be regenerated after upgrading,
and should record `ChatTemplateEngine` beside the template digest. Pass
`EnableThinking` explicitly when the thinking default matters.

Callers that scored `ChatResponse.Content` from a reasoning model with a
`ReasoningFormat` other than none were scoring the model's reasoning, because
the parse had no parser and returned everything as content. Read recorded
outputs again with `ParseChatOutput`. `Chat` can now fail where it used to
return text: a parse failure, a decode failure or a cancelled context is an
error. Check `FinishReason` before treating `Content` as an answer; with
`FinishReasonLength` it may be empty. `ReasoningFormatNone` is unchanged.

The internal C signatures of the chat render, the reasoning parse and the
generation parameters changed under unchanged symbol names. cgo compiles
`wrapper.cpp`, `wrapper_adapter.cpp` and `wrapper_rollout.cpp` with the package,
so those symbols always come from the sources being built. The copies in
`libbinding.a` are not what the Go link resolves them to. Rebuild `libbinding.a`
with `make` so it carries the new object, as the Makefile does, and so the
llama.cpp libraries beside it match the pinned commit.

Existing recipes, mappings and caches do not change. To distil:
1. Measure both tokenizers.
2. Build the mapping with `fusioncache.VocabularyMapping`, and set each
   `ModelIdentity.TokenizerSHA256` to the measured `SourceSHA256`. For a GGUF
   that is the metadata digest.
3. Produce one cache per teacher over every curriculum row, in order: role
   `train`, no features, and `TeacherTokens == StudentTokens == input_ids`.
4. Declare `Recipe.Distillation` and pass the cache directory.

A distillation recipe cannot continue a checkpoint written under another
recipe; start fresh.

Model identities now belong to private installation configuration. Construct
`NewMXCatalog` from typed `MXModelIdentity` entries and pass the selected identity
in `MXConfig.Model`. Repository, revision, manifest, quantisation and shard
identity remain mandatory; externalizing configuration does not waive lineage,
artifact integrity or license obligations.

Replace `backends/mx.AdmittedMXManifest` and `backends/mx.AdmittedMXManifestFor` with
`MXCatalog.ManifestFor`. Replace `MXModelRecipeForDigest` with
`MXCatalog.RecipeForDigest`; its full removed name is
`backends/mx.MXModelRecipeForDigest`. Replace `backends/mx.MXModelMXFP4`,
`backends/mx.MXModelQ2K` and `backends/mx.MXModelTayiQ2Progressive` with
installation-owned `MXModelRecipe` selectors. `backends/mx.MXConfig.ModelRecipe`
becomes `MXConfig.Model`. No model is selected by omission.
Runtime binary identities and the qualified SM75/64-backend build are unchanged.

Training backends use `training/decoder`; geometry, initial projections, rotary
parameters and local recipes must be supplied explicitly. Checkpoint inspection
and resumption retain artifact checks. A configuration's admission is not proof
that a new architecture or training recipe has been scientifically qualified.

## v0.5.0

This historical release retained its Q4 MX default. Q2 callers selected an
explicit recipe and materialized the exact seven-shard manifest admitted by
that recipe. Persist the recipe digest with the execution;
`MXModelRecipeForDigest` can recover only a uniquely admitted digest and rejects
unknown or ambiguous values.

No change is required for applications that continue to use the Q4 recipe or
the in-process llama.cpp binding.

## v0.4.0

The existing in-process llama.cpp engine remains the default and its setup does
not change. The `github.com/tayi-ai/arandu-llama/backends/mx` package is opt-in:
construct its backend with absolute
runtime and model cache roots, install the three binaries whose hashes appear
in `backends/mx-llama.cpp/manifest.json`, and preserve the model `SHA256SUMS`
file beside its shards. The backend refuses a different build or model before
creating a process specification.

`Model.CaptureLoRA` is additive. It creates a temporary context for one measured
projection and can force device synchronization while it observes the graph;
keep it in qualification and debugging paths rather than latency-sensitive
inference.

## v0.3.1

`Score` can now return an error when the reported mean equals
`log(n_vocab)`. Treat that refusal as an invalid forward pass and rerun on the
qualified single-device path; do not store the uniform value as a model loss.

## v0.3.0

`Context.CaptureFinal` is additive. Stored captures should use both
`TokenDigest` and `SnapshotDigest` in their identity so readings made under
different token positions, quantisations or adapter policies cannot collide.

## v0.2.1

No API migration is required. Worker creation failures are returned to the
caller instead of terminating the process.

## v0.2.0

Nothing to upgrade from. This is the first release of this repository.

### Do not resolve v0.1.x of this module path

`github.com/tayi-ai/arandu-llama` has `v0.1.0`, `v0.1.1` and `v0.1.2` in the Go
proxy from a previous repository, and their checksums do not match this
history. A `go.mod` still requiring one of those gets a checksum mismatch,
which the go command reports as a security error rather than as a version
problem:

```text
SECURITY ERROR
This download does NOT match the one reported by the checksum server.
```

Move to `v0.2.0`:

```bash
go get github.com/tayi-ai/arandu-llama@v0.2.0
```

Nothing else changes: the import path, the package name and the exported API
are the same.

### Installing for the first time

Three things beyond `go get`, in this order:

1. **Build the static archives.** The module ships no compiled code and will
   compile but not link without them — `ld: library 'binding' not found` is the
   expected first failure. See [docs/building.md](docs/building.md).
2. **Register the module** in `bootstrap/app.go`, and pass a `Tenant`. It is
   required and has no default.
3. **Run `aru migrate`** before the application serves. The package owns the
   `llamas` table.

The policy denies every action until you open one. A 404 from a route you
believe exists is usually the tenant check, not routing.

### Published views move out of `vendor/`

The view a package publishes lands in `resources/views/modules/<slug>/` and
compiles to `storage/framework/views/modules/<slug>`. It used to be `vendor/` in
both, and that address could not work: the go command reserves the name twice,
and a tree of published views hit both rules.

A file under a directory named `vendor` is left out of the module zip at any
depth. The file stays in the package's repository and is missing for everyone
who downloads it, so the `go:embed` that names its directory matches nothing and
the person building the project reads

```
pattern resources/views: no matching files found
```

— an error about the package, raised in their project. And a package whose
import path carries the element cannot be imported at all:

```
bootstrap/app.go:98:2: use of vendored package not allowed
```

which is exactly the import `(*Module).Boot` asks for. A published view is
compiled into a Go package the application has to import for its `init()` to
register anything, so the second rule refused the last step of the install.

Both were reproduced before this changed: `zip.CheckDir` reports the view as
`file is in vendor directory`, and a package under
`storage/framework/views/vendor/<slug>` is refused at import.

To move a package already released:

1. `git mv resources/views/vendor resources/views/modules`.
2. Rename the `vendorDir` constant in `views.go` to `moduleDir`, with the value
   `modules`.
3. Release the package, and tell the projects that installed it to publish
   again. The old files are theirs now, so `aru vendor:publish --apply` writes
   the new tree beside the old one and the old one is deleted by hand, along
   with its lines in `vendor-publish.lock` and its import in `bootstrap/app.go`.

Framework `v0.46.4` and Hesape `v0.37.0` refuse a publication that carries the
reserved name, so a package that has not moved fails its own tests with a
message naming both rules, rather than failing in the first project that
installs it.
