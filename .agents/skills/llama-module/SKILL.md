---
name: llama-module
description: Change what this Arandu package registers — its routes, handlers, configuration, response shape or schema. Use when the request is to "add a route", "add an endpoint", "add a handler", "add a config option", "change the prefix", "add a field to the response", "add a column", "write a migration", "return the tenant id too", or when a change touches module.go, config.go, model.go or views.go.
license: MIT
---

# The module contract

`foundation.Module` is `Name()` and `Routes()`, and nothing else. That pair is
the whole public contract between a package and the framework: an application
calls `New`, gets something that satisfies it, and registers it by hand.

```go
var (
	_ foundation.Module     = (*Module)(nil)
	_ foundation.Migratable = (*Module)(nil)
)
```

Compile-time proof, at the top of `module.go`. Add one for every contract the
module takes on, so the failure lands here rather than at the registration in
somebody else's repository.

Confirm the set for the version in `go.mod` with:

```sh
export GOWORK=off
go doc github.com/arandu-io/framework/foundation
```

Two of those interfaces change what `arandu.mod.toml` has to say. A
`Background` loop that calls out needs `network = true`; anything that writes a
file needs `filesystem = true`. **The manifest is capability, not intent**, and
`aru doctor` compares it against what the code imports — a package declaring no
outbound calls whilst holding an HTTP client is rejected. That check has already
caught this package once: `filesystem` said false whilst the engine loaded
gigabytes of weights.

## The data path is the configured Model

`Llama` embeds `model.Model`, and its table is declared once beside it in
`model.go`:

```go
var llamaTable = model.NewTable(model.TableSpec{
	Name:      "llamas",
	New:       func() model.Entity { return new(Llama) },
	ManualKey: true,
})
```

The primary key is application-generated text, so the table says `ManualKey`
-- a DEFAULT that produces a uuid is spelled differently in every engine. Do not
set `Global`: it is the only setting that drops the tenant filter, and this
package owns tenant data.

`func Llamas(db model.DB) *LlamaQuery` is the one entry point for that table,
and every read and write in the package goes through it. It is generated, with
`LlamaQuery` and `LlamaCollection`, into `LlamaQuery.go` by `aru model:build`;
never edit that file. After changing the entity or its table, run
`aru model:build` and commit the regenerated file with the change, and
`aru model:build --check` exits non-zero while it is stale.

Keep rows as pointers after `New`, `First`, `Find`, or `Get`. A copy of the row
still reads its fields, but every write promoted from its embedded Model refuses
with `model.ErrUnwired`: the copy is not the row the table built. Never assign
over a row that `New` returned (`*row = other`): that replaces its embedded
Model with the other value's, and the row can no longer save.

Never add a Repository beside it. A type wrapping the Model becomes a second
data path, and a second data path is one the policy does not guard.

A service method reaches it only after `security.Authorize` has returned a
Grant, and writes the tenant from that Grant rather than from the request. The
order is `validate -> security.Authorize -> Grant -> Model terminal`, and the
structural test reads every exported method to confirm it.

## Handlers stay thin

A handler reads the request, calls one service method, and renders. It does not
authorise, does not touch the Model, and does not decide anything.

The reason is that the service is what the structural test reads. Logic in a
handler is logic outside the proof.

## Responses go through Resource

Never encode the entity. An encoder handed `Llama` answers with whatever fields
it happens to have, **including the ones somebody adds later without opening
the handler** — and `TenantID` is exactly such a field: it names another
customer's identifier and belongs in no response.

`Resource` is a declared list with unexported fields, so adding a column to the
entity cannot leak it. Adding a field to the response means editing
`newResource` and `ToArray`, deliberately, in one place.

`ToArray` computes `mean` rather than storing it. The sum and the count stay
the record and the average stays a convenience — see `llama-measure` for why
those two are columns and the mean is not.

`With()` carries what describes the answer rather than the things answered
with: the cursor goes there, beside the items, never inside them.

## Migrations

Named by what they do, with no date prefix. The framework tracks what has been
applied by name.

`aru migrate` is a **pipeline step, before the rollout, never at boot**. With N
replicas starting together, N migrations race.

A column added to the table means: the migration, the `db` tag on the entity,
and a decision about whether it belongs in `Resource`. The default answer to
the third is no.

## Configuration

Three fields today, each with a default in `config.go`. A new one needs the
default and a line in the README table.

`MaxPerPage` is a ceiling and not a suggestion: a caller asking for more gets
the ceiling, not an error and not the number they asked for. An unbounded page
size is a denial-of-service with extra steps.

## Before you finish

```bash
gofmt -l .
go vet ./...
go test -race ./...
```

A new route means a new test in `tests/Feature`. A new service method means a
denial test and a structural entry — see `llama-policy`.
