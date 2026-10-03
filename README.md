# Arandu Llama

An Arandu package that is also an inference engine. It registers its own routes,
owns its own table, decides for itself who may reach either — and links
llama.cpp through cgo, so the model runs in the process rather than behind an
HTTP call.

It exists for one question: **what does a lower bit-width cost?** Answering it
needs three things an inference binding has no reason to carry.

- **A LoRA adapter applied over a quantised base, without merging.** The
  weights stay on the card in Q4_K_M; the adapter is loaded once and its scale
  is changed between forward passes in 0.4 ms.
- **Teacher-forced scoring.** The loss of a completion somebody else wrote, of
  the same quantity a backpropagating trainer reports, so the two can be
  compared. Windowed, because flagging every position of a 248320-token
  vocabulary asks for 2.46 GB of logits.
- **A reading that keeps.** Sum, count, policy digest and quantisation stored
  as a row. A run that keeps its numbers in a variable has nothing to put side
  by side, and a twenty-step run once reported the same loss at every step
  without anybody noticing.

## Install

```bash
go get github.com/tayi-ai/arandu-llama
```

The Go side compiles anywhere. Linking needs the static archives, which are
built once — see [docs/building.md](docs/building.md).

## Build the archives

The module ships no compiled code. `libbinding.a` and the ggml archives beside
it are produced from the pinned llama.cpp submodule:

```bash
git submodule update --init llama.cpp
make BUILD_TYPE=cublas BUILD_LINKAGE=static CUDA_ARCHITECTURES=75 libbinding.a
```

`BUILD_TYPE=metal` on a Mac, `cublas` on a CUDA host, empty for CPU only. `75`
is compute 7.5.

Then point the linker at them, from the application:

```bash
CGO_ENABLED=1 CGO_LDFLAGS="-L/path/to/archives" go build -tags llama ./...
```

The second `-L` is the whole trick. The package declares `-L./`, which for a
dependency resolves to the module cache — read-only, shared by every build on
the machine, verified against `go.sum`. Dropping archives in there means making
it writable, which corrupts something that is not yours. The linker collects
every search path and looks in all of them for each `-l`, so naming a second
directory is enough.

## Wire it

An Arandu application registers a module explicitly. There is no service
provider, no container and no discovery, so these are the lines to paste into
`bootstrap/app.go` and there are no others.

The import, with the other module imports:

```go
import (
	llama "github.com/tayi-ai/arandu-llama"
)
```

The construction, in `Build`, after the session store exists and before
`k.Register`:

```go
	llamaModule, err := llama.New(llama.Config{
		Tenant: cfg.Auth.Tenant,
		Prefix: "/readings",
	}, db, sessions)
	if err != nil {
		return App{}, err
	}
```

And the registration, inside the `k.Register(...)` call already there:

```go
		llamaModule,
```

Then, once, before the application serves:

```bash
aru migrate
```

This package owns the `llamas` table, which is why the migration step is not
optional and why `arandu.mod.toml` says `migrations = true`.

## Configuration

| field | required | what it decides |
| --- | --- | --- |
| `Tenant` | yes | the customer a visitor with no session is read as |
| `Prefix` | no | where the routes are mounted; `/llama` when empty |

`Tenant` has no default on purpose. A visitor with no session has to be read as
some customer, and it cannot be one the request names — a tenant taken from a
URL or a header is a tenant the caller chooses.

## Routes

| method | path | action |
| --- | --- | --- |
| `GET` | `/llama` | `LlamaList` |
| `GET` | `/llama/{id}` | `LlamaView` |
| `POST` | `/llama` | `LlamaCreate` |
| `PUT` | `/llama/{id}` | `LlamaUpdate` |
| `DELETE` | `/llama/{id}` | `LlamaDelete` |

Every one of them goes through the policy first, and **the policy denies
everything until an action is opened**. A row belonging to another tenant
answers 404 rather than 403: the two are deliberately indistinguishable,
because 403 confirms the row exists.

## Take a reading

Loading a model, applying an adapter and scoring an example:

```go
model, err := llama.LoadModel("model.gguf", llama.WithGPULayers(-1))
defer model.Close()

ctx, err := model.NewContext(llama.WithContextSize(8192))
defer ctx.Close()

adapter, err := model.LoadAdapter("rank4.gguf")
defer adapter.Close()

// The positive half of a zeroth-order pair.
err = ctx.SetAdapters([]*llama.Adapter{adapter}, []float32{1.0})

reading, err := llama.Take(request, service, actor, ctx,
	"step-14 positive", "example-3", tokens, len(promptTokens))
```

`Take` stores before it returns, and that ordering is the point.

Scoring on its own, without storing:

```go
score, err := ctx.ScoreText("2 + 2 =", " 4")
fmt.Printf("%f over %d positions\n", score.Mean(), score.Tokens)
```

The sum and the count travel apart and never as a mean. A caller combining
several examples has to weight each by its length; averaging the means
over-weights short completions silently, and the number then drifts away from
what a trainer reports for the same data.

## Layout

| file | what lives there |
| --- | --- |
| `module.go` | the module contract, the routes, the migration |
| `config.go` | the three fields above, and their defaults |
| `model.go` | the `Llama` entity and the response shapes |
| `policy.go` | who may read, write and delete |
| `service.go` | the authorised use cases |
| `engine.go` | `Take`, and the policy label a reading is stored under |
| `adapter.go` | the Go handle for a loaded adapter, applying and clearing |
| `score.go` | the scoring surface, over tokens and over text |
| `weights.go`, `context.go` | the model and the context |
| `wrapper_adapter.cpp` | the C surface: adapters, metadata, scoring |
| `cgo_headers/` | llama.cpp headers, mirrored byte-for-byte |
| `llama.cpp/` | the pinned submodule, `90c26fcd` = build b10675 |

## Why the headers are mirrored

The Go module proxy does not package git submodules — it strips gitlinks from
the zip. So a consumer fetching this module through the proxy gets no
`llama.cpp/` and could not resolve the wrapper's includes. `cgo_headers/`
mirrors what the wrapper needs, byte-for-byte, and CI checks it for drift. The
submodule is needed to *build the archives*, not to consume the module.

## Model-first data path

`Llama` embeds `model.Model`, its table is declared once beside it in
`model.go`, and `Llamas(db)` is the one entry point every read and write goes
through. The query it returns, `LlamaQuery`, is generated in `LlamaQuery.go` by
`aru model:build` and is never edited by hand; `aru model:build --check` fails
when the committed file is not the one the entity produces. There is no
Repository beside it: a type wrapping the Model becomes a second data path, and
a second data path is one the policy does not guard.

A service method follows `validate -> security.Authorize -> Grant -> Model
terminal`, in that order and no other. Handlers stay thin — they read the
request, call one method and render — because the service is what the
structural test reads, and logic in a handler is logic outside the proof.

Every Model terminal requires the Grant that `security.Authorize` produced, and
the tenant is written from that Grant rather than from the request.

## Tests

```bash
go test ./...
go test -race ./...
```

Two halves, both of which matter. The denial tests run with a **nil database**,
so any path reaching the Model before authorisation panics rather than passing
quietly. The structural twin reads every exported service method and checks the
same order on the allowed path.

## Licence

MIT. `LICENSE` carries the notice the code arrived under and is kept unchanged
— the MIT terms require it to travel with every copy, and taking over
maintenance is not an exception.

Licence is per artifact, and three meet in this directory. `llama.cpp/` is a
submodule of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp), MIT,
with its own notice. `cgo_headers/` mirrors headers llama.cpp bundles from
elsewhere — xxHash under BSD 2-Clause, nlohmann and cpp-httplib under MIT,
miniaudio and the stb-style headers under public-domain dedications — and every
one of those notices travels inside its own file. That is the second reason the
drift check is byte-for-byte rather than semantic: a formatter rewriting those
files would be rewriting notices this repository is required to redistribute
intact.

None of it says anything about the licence on model weights or on a dataset
that passes through.
