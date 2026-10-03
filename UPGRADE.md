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

### The llama is a concrete type over the non-generic model

Hesape `v0.47.0` removes the generic model layer, and this release moves to it
with Hesape `v0.48.0` and Framework `v0.50.2`. `Llama` embeds the non-generic
`model.Model`, its table is declared once beside it with `model.NewTable`, and
the query that starts from it is generated beside it by `aru model:build`, in
`LlamaQuery.go`. No route, migration, action, policy decision or tenant rule
changed.

**`Llamas` returns the generated query.** It takes a `model.DB` -- a `*data.DB`
passes unchanged -- and returns `*llama.LlamaQuery` instead of
`*model.Model[llama.Llama]`. A chain that started from it keeps its text, minus
the calls that no longer exist:

| before | now |
|---|---|
| `llama.Llamas(db).NewQuery().Where(…)` | `llama.Llamas(db).Where(…)` |
| `llama.Llamas(db).NewInstance(nil, false)` and `.Entity` | `llama.Llamas(db).New()`, which returns `*Llama` |
| `Get` → `model.Collection[llama.Llama]` | `Get` → `llama.LlamaCollection` (`[]*Llama`) |
| `func(q *model.Builder[llama.Llama])` in a grouped `Where` | `func(q *llama.LlamaQuery)` |
| `llama.Llamas(db).GetTable()`, `.KeyType`, `.TenantColumn` | nothing: the table is unexported, and its settings are not read off the query |

`First`, `Find` and the other row terminals still return `*Llama`, and still
take the Grant. Every `LlamaService` method keeps its signature: `Create` and
`Find` still return `*Llama`, and `List` still returns `[]*Llama`.

**The entity no longer carries the model's configuration.** `Llama` embeds
`model.Model`, so the fields and methods `model.Model[Llama]` promoted onto it
are gone: the configuration fields (`PrimaryKey`, `KeyType`, `Incrementing`,
`Timestamps`, `TenantColumn`, `Table` and the rest) live in the table, which this
package keeps unexported, and `Exists` and `WasRecentlyCreated` are methods,
`row.Exists()`. A copied row still reads its fields but refuses every write with
`model.ErrUnwired`, so keep the pointers the queries return.

**Upgrade the floor.** The module requires Hesape `v0.48.0` and Framework
`v0.50.2`, and `arandu.mod.toml` declares `framework = ">= 0.50"`. An
application that pins a Hesape below `v0.47.0` cannot compile this release:
every generic model type it would need is gone from Hesape itself. The
published views are unchanged and need no republish.

**What changes without a compiler error.** The store route, `POST` on the
prefix, reads `name`, `subject`, `policy`, `quantisation`, `loss`, `tokens` and
`milliseconds` through `Context.Input`, and since Hesape `v0.44.0` the input of
a `POST` is its body alone: a url-encoded or multipart form, or a JSON object. A
field sent only in the query string of the `POST` is no longer read. The listing
and the record routes are `GET` and read the query string as before.

<details>
<summary>Every incompatible symbol <code>apidiff</code> reports for the model change</summary>

Most of these are the methods and fields `model.Model[Llama]` promoted onto
`Llama`, which left with the generic type.

```text
Llama.ConnectionName
Llama.CreatedAtColumn
Llama.DeletedAtColumn
Llama.Entity
Llama.Exists
Llama.Grammar
Llama.Incrementing
Llama.KeyType
Llama.NamedScopes
Llama.PerPage
Llama.PrimaryKey
Llama.Processor
Llama.RelationResolvers
Llama.SoftDeletes
Llama.Table
Llama.TenantColumn
Llama.Timestamps
Llama.UpdatedAtColumn
Llama.WasRecentlyCreated
Llamas
github.com/arandu-io/hesape/database/model.(*Model[github.com/tayi-ai/arandu-llama.Llama]).AddGlobalScope, method set of *Llama
github.com/arandu-io/hesape/database/model.(*Model[github.com/tayi-ai/arandu-llama.Llama]).All, method set of *Llama
github.com/arandu-io/hesape/database/model.(*Model[github.com/tayi-ai/arandu-llama.Llama]).Append, method set of *Llama
github.com/arandu-io/hesape/database/model.(*Model[github.com/tayi-ai/arandu-llama.Llama]).AttributesToArray, method set of *Llama
github.com/arandu-io/hesape/database/model.(*Model[github.com/tayi-ai/arandu-llama.Llama]).CallNamedScope, method set of *Llama
github.com/arandu-io/hesape/database/model.(*Model[github.com/tayi-ai/arandu-llama.Llama]).Create, method set of *Llama
github.com/arandu-io/hesape/database/model.(*Model[github.com/tayi-ai/arandu-llama.Llama]).Destroy, method set of *Llama
github.com/arandu-io/hesape/database/model.(*Model[github.com/tayi-ai/arandu-llama.Llama]).DiscardChanges, method set of *Llama
github.com/arandu-io/hesape/database/model.(*Model[github.com/tayi-ai/arandu-llama.Llama]).Except, method set of *Llama
github.com/arandu-io/hesape/database/model.(*Model[github.com/tayi-ai/arandu-llama.Llama]).Find, method set of *Llama
github.com/arandu-io/hesape/database/model.(*Model[github.com/tayi-ai/arandu-llama.Llama]).FindMany, method set of *Llama
github.com/arandu-io/hesape/database/model.(*Model[github.com/tayi-ai/arandu-llama.Llama]).FindOrFail, method set of *Llama
github.com/arandu-io/hesape/database/model.(*Model[github.com/tayi-ai/arandu-llama.Llama]).FindOrNew, method set of *Llama
github.com/arandu-io/hesape/database/model.(*Model[github.com/tayi-ai/arandu-llama.Llama]).First, method set of *Llama
github.com/arandu-io/hesape/database/model.(*Model[github.com/tayi-ai/arandu-llama.Llama]).FirstOrCreate, method set of *Llama
github.com/arandu-io/hesape/database/model.(*Model[github.com/tayi-ai/arandu-llama.Llama]).FirstOrNew, method set of *Llama
github.com/arandu-io/hesape/database/model.(*Model[github.com/tayi-ai/arandu-llama.Llama]).ForceCreate, method set of *Llama
github.com/arandu-io/hesape/database/model.(*Model[github.com/tayi-ai/arandu-llama.Llama]).ForceDeleteQuietly, method set of *Llama
github.com/arandu-io/hesape/database/model.(*Model[github.com/tayi-ai/arandu-llama.Llama]).ForceDeleted, method set of *Llama
github.com/arandu-io/hesape/database/model.(*Model[github.com/tayi-ai/arandu-llama.Llama]).ForceDeleting, method set of *Llama
github.com/arandu-io/hesape/database/model.(*Model[github.com/tayi-ai/arandu-llama.Llama]).ForceDestroy, method set of *Llama
github.com/arandu-io/hesape/database/model.(*Model[github.com/tayi-ai/arandu-llama.Llama]).FreshTimestamp, method set of *Llama
github.com/arandu-io/hesape/database/model.(*Model[github.com/tayi-ai/arandu-llama.Llama]).GetAppends, method set of *Llama
github.com/arandu-io/hesape/database/model.(*Model[github.com/tayi-ai/arandu-llama.Llama]).GetConnectionName, method set of *Llama
github.com/arandu-io/hesape/database/model.(*Model[github.com/tayi-ai/arandu-llama.Llama]).GetCreatedAtColumn, method set of *Llama
github.com/arandu-io/hesape/database/model.(*Model[github.com/tayi-ai/arandu-llama.Llama]).GetDeletedAtColumn, method set of *Llama
github.com/arandu-io/hesape/database/model.(*Model[github.com/tayi-ai/arandu-llama.Llama]).GetForeignKey, method set of *Llama
github.com/arandu-io/hesape/database/model.(*Model[github.com/tayi-ai/arandu-llama.Llama]).GetGlobalScopes, method set of *Llama
github.com/arandu-io/hesape/database/model.(*Model[github.com/tayi-ai/arandu-llama.Llama]).GetHidden, method set of *Llama
github.com/arandu-io/hesape/database/model.(*Model[github.com/tayi-ai/arandu-llama.Llama]).GetIncrementing, method set of *Llama
github.com/arandu-io/hesape/database/model.(*Model[github.com/tayi-ai/arandu-llama.Llama]).GetKeyName, method set of *Llama
github.com/arandu-io/hesape/database/model.(*Model[github.com/tayi-ai/arandu-llama.Llama]).GetKeyType, method set of *Llama
github.com/arandu-io/hesape/database/model.(*Model[github.com/tayi-ai/arandu-llama.Llama]).GetMorphClass, method set of *Llama
github.com/arandu-io/hesape/database/model.(*Model[github.com/tayi-ai/arandu-llama.Llama]).GetPerPage, method set of *Llama
github.com/arandu-io/hesape/database/model.(*Model[github.com/tayi-ai/arandu-llama.Llama]).GetPrevious, method set of *Llama
github.com/arandu-io/hesape/database/model.(*Model[github.com/tayi-ai/arandu-llama.Llama]).GetQualifiedCreatedAtColumn, method set of *Llama
github.com/arandu-io/hesape/database/model.(*Model[github.com/tayi-ai/arandu-llama.Llama]).GetQualifiedDeletedAtColumn, method set of *Llama
github.com/arandu-io/hesape/database/model.(*Model[github.com/tayi-ai/arandu-llama.Llama]).GetQualifiedKeyName, method set of *Llama
github.com/arandu-io/hesape/database/model.(*Model[github.com/tayi-ai/arandu-llama.Llama]).GetQualifiedUpdatedAtColumn, method set of *Llama
github.com/arandu-io/hesape/database/model.(*Model[github.com/tayi-ai/arandu-llama.Llama]).GetQueueableConnection, method set of *Llama
github.com/arandu-io/hesape/database/model.(*Model[github.com/tayi-ai/arandu-llama.Llama]).GetQueueableID, method set of *Llama
github.com/arandu-io/hesape/database/model.(*Model[github.com/tayi-ai/arandu-llama.Llama]).GetQueueableRelations, method set of *Llama
github.com/arandu-io/hesape/database/model.(*Model[github.com/tayi-ai/arandu-llama.Llama]).GetRawOriginal, method set of *Llama
github.com/arandu-io/hesape/database/model.(*Model[github.com/tayi-ai/arandu-llama.Llama]).GetRelation, method set of *Llama
github.com/arandu-io/hesape/database/model.(*Model[github.com/tayi-ai/arandu-llama.Llama]).GetRelations, method set of *Llama
github.com/arandu-io/hesape/database/model.(*Model[github.com/tayi-ai/arandu-llama.Llama]).GetRouteKey, method set of *Llama
github.com/arandu-io/hesape/database/model.(*Model[github.com/tayi-ai/arandu-llama.Llama]).GetRouteKeyName, method set of *Llama
github.com/arandu-io/hesape/database/model.(*Model[github.com/tayi-ai/arandu-llama.Llama]).GetTable, method set of *Llama
github.com/arandu-io/hesape/database/model.(*Model[github.com/tayi-ai/arandu-llama.Llama]).GetTouchedRelations, method set of *Llama
github.com/arandu-io/hesape/database/model.(*Model[github.com/tayi-ai/arandu-llama.Llama]).GetUpdatedAtColumn, method set of *Llama
github.com/arandu-io/hesape/database/model.(*Model[github.com/tayi-ai/arandu-llama.Llama]).GetVisible, method set of *Llama
github.com/arandu-io/hesape/database/model.(*Model[github.com/tayi-ai/arandu-llama.Llama]).HasAppended, method set of *Llama
github.com/arandu-io/hesape/database/model.(*Model[github.com/tayi-ai/arandu-llama.Llama]).HasGlobalScope, method set of *Llama
github.com/arandu-io/hesape/database/model.(*Model[github.com/tayi-ai/arandu-llama.Llama]).HasNamedScope, method set of *Llama
github.com/arandu-io/hesape/database/model.(*Model[github.com/tayi-ai/arandu-llama.Llama]).Is
github.com/arandu-io/hesape/database/model.(*Model[github.com/tayi-ai/arandu-llama.Llama]).IsForceDeleting, method set of *Llama
github.com/arandu-io/hesape/database/model.(*Model[github.com/tayi-ai/arandu-llama.Llama]).IsIgnoringTouch, method set of *Llama
github.com/arandu-io/hesape/database/model.(*Model[github.com/tayi-ai/arandu-llama.Llama]).IsNot, method set of *Llama
github.com/arandu-io/hesape/database/model.(*Model[github.com/tayi-ai/arandu-llama.Llama]).IsRelation, method set of *Llama
github.com/arandu-io/hesape/database/model.(*Model[github.com/tayi-ai/arandu-llama.Llama]).IsSoftDeletable, method set of *Llama
github.com/arandu-io/hesape/database/model.(*Model[github.com/tayi-ai/arandu-llama.Llama]).LoadAggregate, method set of *Llama
github.com/arandu-io/hesape/database/model.(*Model[github.com/tayi-ai/arandu-llama.Llama]).LoadMorph, method set of *Llama
github.com/arandu-io/hesape/database/model.(*Model[github.com/tayi-ai/arandu-llama.Llama]).LoadMorphAggregate, method set of *Llama
github.com/arandu-io/hesape/database/model.(*Model[github.com/tayi-ai/arandu-llama.Llama]).LoadMorphAvg, method set of *Llama
github.com/arandu-io/hesape/database/model.(*Model[github.com/tayi-ai/arandu-llama.Llama]).LoadMorphCount, method set of *Llama
github.com/arandu-io/hesape/database/model.(*Model[github.com/tayi-ai/arandu-llama.Llama]).LoadMorphMax, method set of *Llama
github.com/arandu-io/hesape/database/model.(*Model[github.com/tayi-ai/arandu-llama.Llama]).LoadMorphMin, method set of *Llama
github.com/arandu-io/hesape/database/model.(*Model[github.com/tayi-ai/arandu-llama.Llama]).LoadMorphSum, method set of *Llama
github.com/arandu-io/hesape/database/model.(*Model[github.com/tayi-ai/arandu-llama.Llama]).MakeHidden
github.com/arandu-io/hesape/database/model.(*Model[github.com/tayi-ai/arandu-llama.Llama]).MakeVisible
github.com/arandu-io/hesape/database/model.(*Model[github.com/tayi-ai/arandu-llama.Llama]).NewBaseQueryBuilder, method set of *Llama
github.com/arandu-io/hesape/database/model.(*Model[github.com/tayi-ai/arandu-llama.Llama]).NewCollection, method set of *Llama
github.com/arandu-io/hesape/database/model.(*Model[github.com/tayi-ai/arandu-llama.Llama]).NewFromBuilder, method set of *Llama
github.com/arandu-io/hesape/database/model.(*Model[github.com/tayi-ai/arandu-llama.Llama]).NewInstance, method set of *Llama
github.com/arandu-io/hesape/database/model.(*Model[github.com/tayi-ai/arandu-llama.Llama]).NewModelQuery, method set of *Llama
github.com/arandu-io/hesape/database/model.(*Model[github.com/tayi-ai/arandu-llama.Llama]).NewQuery, method set of *Llama
github.com/arandu-io/hesape/database/model.(*Model[github.com/tayi-ai/arandu-llama.Llama]).NewQueryForRestoration, method set of *Llama
github.com/arandu-io/hesape/database/model.(*Model[github.com/tayi-ai/arandu-llama.Llama]).NewQueryWithoutRelationships, method set of *Llama
github.com/arandu-io/hesape/database/model.(*Model[github.com/tayi-ai/arandu-llama.Llama]).NewQueryWithoutScope, method set of *Llama
github.com/arandu-io/hesape/database/model.(*Model[github.com/tayi-ai/arandu-llama.Llama]).NewQueryWithoutScopes, method set of *Llama
github.com/arandu-io/hesape/database/model.(*Model[github.com/tayi-ai/arandu-llama.Llama]).NewTypedBuilder, method set of *Llama
github.com/arandu-io/hesape/database/model.(*Model[github.com/tayi-ai/arandu-llama.Llama]).On, method set of *Llama
github.com/arandu-io/hesape/database/model.(*Model[github.com/tayi-ai/arandu-llama.Llama]).OnWriteConnection, method set of *Llama
github.com/arandu-io/hesape/database/model.(*Model[github.com/tayi-ai/arandu-llama.Llama]).Only, method set of *Llama
github.com/arandu-io/hesape/database/model.(*Model[github.com/tayi-ai/arandu-llama.Llama]).OnlyTrashed, method set of *Llama
github.com/arandu-io/hesape/database/model.(*Model[github.com/tayi-ai/arandu-llama.Llama]).OriginalIsEquivalent, method set of *Llama
github.com/arandu-io/hesape/database/model.(*Model[github.com/tayi-ai/arandu-llama.Llama]).PushQuietly, method set of *Llama
github.com/arandu-io/hesape/database/model.(*Model[github.com/tayi-ai/arandu-llama.Llama]).QualifyColumn, method set of *Llama
github.com/arandu-io/hesape/database/model.(*Model[github.com/tayi-ai/arandu-llama.Llama]).QualifyColumns, method set of *Llama
github.com/arandu-io/hesape/database/model.(*Model[github.com/tayi-ai/arandu-llama.Llama]).Query, method set of *Llama
github.com/arandu-io/hesape/database/model.(*Model[github.com/tayi-ai/arandu-llama.Llama]).Ref, method set of *Llama
github.com/arandu-io/hesape/database/model.(*Model[github.com/tayi-ai/arandu-llama.Llama]).RegisterGlobalScopes, method set of *Llama
github.com/arandu-io/hesape/database/model.(*Model[github.com/tayi-ai/arandu-llama.Llama]).RegisterModelEvent, method set of *Llama
github.com/arandu-io/hesape/database/model.(*Model[github.com/tayi-ai/arandu-llama.Llama]).ReplicateQuietly, method set of *Llama
github.com/arandu-io/hesape/database/model.(*Model[github.com/tayi-ai/arandu-llama.Llama]).ResolveRouteBinding, method set of *Llama
github.com/arandu-io/hesape/database/model.(*Model[github.com/tayi-ai/arandu-llama.Llama]).ResolveRouteBindingQuery, method set of *Llama
github.com/arandu-io/hesape/database/model.(*Model[github.com/tayi-ai/arandu-llama.Llama]).ResolveSoftDeletableRouteBinding, method set of *Llama
github.com/arandu-io/hesape/database/model.(*Model[github.com/tayi-ai/arandu-llama.Llama]).RestoreQuietly, method set of *Llama
github.com/arandu-io/hesape/database/model.(*Model[github.com/tayi-ai/arandu-llama.Llama]).Restored, method set of *Llama
github.com/arandu-io/hesape/database/model.(*Model[github.com/tayi-ai/arandu-llama.Llama]).Restoring, method set of *Llama
github.com/arandu-io/hesape/database/model.(*Model[github.com/tayi-ai/arandu-llama.Llama]).SetAppends, method set of *Llama
github.com/arandu-io/hesape/database/model.(*Model[github.com/tayi-ai/arandu-llama.Llama]).SetConnection, method set of *Llama
github.com/arandu-io/hesape/database/model.(*Model[github.com/tayi-ai/arandu-llama.Llama]).SetHidden, method set of *Llama
github.com/arandu-io/hesape/database/model.(*Model[github.com/tayi-ai/arandu-llama.Llama]).SetIncrementing, method set of *Llama
github.com/arandu-io/hesape/database/model.(*Model[github.com/tayi-ai/arandu-llama.Llama]).SetKeyName, method set of *Llama
github.com/arandu-io/hesape/database/model.(*Model[github.com/tayi-ai/arandu-llama.Llama]).SetKeyType, method set of *Llama
github.com/arandu-io/hesape/database/model.(*Model[github.com/tayi-ai/arandu-llama.Llama]).SetPerPage, method set of *Llama
github.com/arandu-io/hesape/database/model.(*Model[github.com/tayi-ai/arandu-llama.Llama]).SetRelation
github.com/arandu-io/hesape/database/model.(*Model[github.com/tayi-ai/arandu-llama.Llama]).SetRelations, method set of *Llama
github.com/arandu-io/hesape/database/model.(*Model[github.com/tayi-ai/arandu-llama.Llama]).SetTable, method set of *Llama
github.com/arandu-io/hesape/database/model.(*Model[github.com/tayi-ai/arandu-llama.Llama]).SetTouchedRelations, method set of *Llama
github.com/arandu-io/hesape/database/model.(*Model[github.com/tayi-ai/arandu-llama.Llama]).SetVisible, method set of *Llama
github.com/arandu-io/hesape/database/model.(*Model[github.com/tayi-ai/arandu-llama.Llama]).SoftDeleted, method set of *Llama
github.com/arandu-io/hesape/database/model.(*Model[github.com/tayi-ai/arandu-llama.Llama]).SyncChanges, method set of *Llama
github.com/arandu-io/hesape/database/model.(*Model[github.com/tayi-ai/arandu-llama.Llama]).SyncOriginal
github.com/arandu-io/hesape/database/model.(*Model[github.com/tayi-ai/arandu-llama.Llama]).SyncOriginalAttribute, method set of *Llama
github.com/arandu-io/hesape/database/model.(*Model[github.com/tayi-ai/arandu-llama.Llama]).SyncOriginalAttributes, method set of *Llama
github.com/arandu-io/hesape/database/model.(*Model[github.com/tayi-ai/arandu-llama.Llama]).ToPrettyJSON, method set of *Llama
github.com/arandu-io/hesape/database/model.(*Model[github.com/tayi-ai/arandu-llama.Llama]).Touches, method set of *Llama
github.com/arandu-io/hesape/database/model.(*Model[github.com/tayi-ai/arandu-llama.Llama]).UnsetAttribute, method set of *Llama
github.com/arandu-io/hesape/database/model.(*Model[github.com/tayi-ai/arandu-llama.Llama]).UnsetRelation, method set of *Llama
github.com/arandu-io/hesape/database/model.(*Model[github.com/tayi-ai/arandu-llama.Llama]).UnsetRelations, method set of *Llama
github.com/arandu-io/hesape/database/model.(*Model[github.com/tayi-ai/arandu-llama.Llama]).UpdateOrCreate, method set of *Llama
github.com/arandu-io/hesape/database/model.(*Model[github.com/tayi-ai/arandu-llama.Llama]).UpdateOrFail, method set of *Llama
github.com/arandu-io/hesape/database/model.(*Model[github.com/tayi-ai/arandu-llama.Llama]).UpdateQuietly, method set of *Llama
github.com/arandu-io/hesape/database/model.(*Model[github.com/tayi-ai/arandu-llama.Llama]).UpdateTimestamps, method set of *Llama
github.com/arandu-io/hesape/database/model.(*Model[github.com/tayi-ai/arandu-llama.Llama]).UsesTimestamps, method set of *Llama
github.com/arandu-io/hesape/database/model.(*Model[github.com/tayi-ai/arandu-llama.Llama]).Where, method set of *Llama
github.com/arandu-io/hesape/database/model.(*Model[github.com/tayi-ai/arandu-llama.Llama]).WhereKey, method set of *Llama
github.com/arandu-io/hesape/database/model.(*Model[github.com/tayi-ai/arandu-llama.Llama]).With, method set of *Llama
github.com/arandu-io/hesape/database/model.(*Model[github.com/tayi-ai/arandu-llama.Llama]).WithTrashed, method set of *Llama
github.com/arandu-io/hesape/database/model.(*Model[github.com/tayi-ai/arandu-llama.Llama]).WithoutRelations, method set of *Llama
github.com/arandu-io/hesape/database/model.(*Model[github.com/tayi-ai/arandu-llama.Llama]).WithoutTimestamps, method set of *Llama
```

</details>

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
