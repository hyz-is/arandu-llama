---
name: llama-release
description: Cut a version of this package — run the gates, update the manifest or the changelog, bump a dependency, tag, or recover from a bad release. Use when the request is to "release", "tag it", "cut v0.3.0", "bump the framework", "update the changelog", "publish", "the tag is broken", or when a change touches go.mod, arandu.mod.toml or CHANGELOG.md.
license: MIT
---

# A published version is immutable, and that decides the whole procedure

The Go module proxy resolves a version once and serves it forever, and the
checksum database records its hash. Re-tagging hands every consumer a checksum
mismatch, which reads as an attack rather than as a mistake.

**A release is corrected by another release.** Never by moving a tag.

This has already cost this project once: `arandu-attempt v0.1.1` was tagged on a
commit carrying half a change, did not compile, and by the time it was noticed
the proxy had already cached it. It is retracted in `go.mod` and superseded, and
it will be in the version list forever.

## The gates, in order

Run all of them. A red gate blocks; it does not become a finding.

```bash
gofmt -l .                    # excluding llama.cpp/ and cgo_headers/
go vet ./...
go vet -tags llama ./...
go test -race ./...
```

Then the one that matters most and is the easiest to skip:

```bash
git worktree add --detach /tmp/check HEAD
go -C /tmp/check vet ./...
go -C /tmp/check test ./...
```

**Verify a clean checkout before tagging, not your working tree.** A warm
working tree hides an uncommitted file, a stale `go.sum` and a missing gitlink.
All three have shipped from here: a rename that edited a file the commit did not
include, a `go.sum` pinning a version `go.mod` did not ask for, and a submodule
that never entered the first commit at all.

For this package the clean checkout will fail to **link** — no archives — and
that is expected. `go vet ./...` and `go vet -tags llama ./...` are the two that
must pass there; point `CGO_LDFLAGS` at built archives to run the tests.

## The tag

Annotated, never lightweight, and the message says what changed and why
somebody would take it:

```bash
git tag -a v0.3.0 -m "arandu-llama v0.3.0

<what changed, and what it costs to not take it>"
git push origin main
git push origin v0.3.0
```

Pre-1.0 means the API moves without a major bump, and saying so in the tag is
honest rather than apologetic.

## Versions this path may not reuse

`v0.1.0`, `v0.1.1` and `v0.1.2` of `github.com/tayi-ai/arandu-llama` are in the
proxy from a previous repository. Their checksums do not match this history.
**Never tag those numbers again** — a consumer resolving one gets a mismatch and
a security error, and there is no way to withdraw it.

## Retracting

When a released version must not be used:

```go
// v0.1.1 does not compile: <the reason, in one sentence>.
retract v0.1.1
```

The comment is what `go list -m -retracted` prints, so it is the whole
explanation a consumer gets. Write it for somebody who has the failure in front
of them and not the history.

## The manifest

`arandu.mod.toml` is checked by `aru doctor` against what the code actually
imports. Update it in the same commit as the capability, never afterwards:

- an outbound call anywhere means `network = true`
- reading or writing a file means `filesystem = true`
- owning a table means `migrations = true`, which is how an installer learns
  they have to run `aru migrate`

The framework floor is checked against `go.mod` by a test, so the two cannot
drift. It reads:

```toml
framework = ">= 0.50"
```

It moves only when something in the code needs the newer version, and it moves
in `go.mod`, in the manifest and in this file together. Widening it because a
bump happened to work is a claim nobody measured.

## The shape a release must not lose

Two properties are load-bearing and a release that quietly drops either is a
release that has to be followed by another one.

**Model-first.** `Llamas(db)` is the one configured copy of the data path, and
every service method reaches it only after `security.Authorize`. A CRUD
Repository reintroduced beside it is a second path the policy does not guard.

**The configured copy is the only copy.** A package cloned and renamed from
this one carries the same tests; if a configured copy ever starts differing
from what those tests read, the difference is the defect.

## The changelog

One entry per release, newest first, in the same words a reader would search
for. What changed, what breaks, what to do about it.

An entry that says "various fixes" is worse than no entry: it costs a reader
the time to open it.

## After tagging

Bump the consumer and verify it there too:

```bash
go get github.com/tayi-ai/arandu-llama@v0.3.0
go vet ./... && go vet -tags llama ./... && go test ./...
```

If the consumer builds an image that pins the version separately from `go.mod`,
move both. The archives and the cgo shim coming from different versions is not
a link error — the symbol names survive a release, so the build succeeds and the
wrong code runs.
