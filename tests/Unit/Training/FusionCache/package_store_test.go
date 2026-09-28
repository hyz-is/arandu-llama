package fusioncache_test

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"io/fs"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strings"
	"testing"

	"github.com/tayi-ai/arandu-llama/training/fusioncache"
)

// noContext is the nil context a caller could pass by mistake.
var noContext context.Context

func refusedWith(t *testing.T, err error, want string) {
	t.Helper()
	if !errors.Is(err, fusioncache.ErrContract) || !strings.Contains(err.Error(), want) {
		t.Fatalf("want a refusal containing %q, got %v", want, err)
	}
}

func storeNames(t *testing.T, directory string) []string {
	t.Helper()
	entries, err := os.ReadDir(directory)
	if err != nil {
		t.Fatal(err)
	}
	var names []string
	for _, entry := range entries {
		names = append(names, entry.Name())
	}
	return names
}

func sha256Hex(data []byte) string {
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:])
}

// correction returns p with its admitted signal replaced, as a fix of a
// trajectory that was not normalized to the student template would.
func correction(t *testing.T, p fusioncache.CapabilityPackage, label string) fusioncache.CapabilityPackage {
	t.Helper()
	c := clone(t, p)
	c.Signal.RefSHA256 = hashOf(label)
	c.Artifacts = artifactIndex(c)
	return c
}

func TestAFrozenPackageIsNeverRewritten(t *testing.T) {
	ctx := context.Background()
	store := fusioncache.PackageStore{Directory: t.TempDir()}
	p := routedPackage(t)
	want := packageDigest(t, p)
	canonical := encodePackage(t, p)
	path := filepath.Join(store.Directory, want+".json")

	receipt, err := store.Freeze(ctx, p, packageLimits())
	if err != nil {
		t.Fatal(err)
	}
	if receipt != (fusioncache.Receipt{Location: path, SHA256: want, Bytes: int64(len(canonical))}) {
		t.Fatalf("unexpected receipt %+v", receipt)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(data, canonical) || sha256Hex(data) != want {
		t.Fatal("the frozen file is not the canonical encoding named by its digest")
	}
	installed, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	if installed.Mode().Perm() != 0o444 {
		t.Fatalf("a frozen package is writable: %v", installed.Mode())
	}
	read, err := store.Read(want, packageLimits())
	if err != nil || !reflect.DeepEqual(read, p) {
		t.Fatalf("read back %v", err)
	}

	again, err := store.Freeze(ctx, p, packageLimits())
	if err != nil || again != receipt {
		t.Fatalf("an identical freeze was not idempotent: %+v, %v", again, err)
	}
	after, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	if !os.SameFile(installed, after) {
		t.Fatal("an identical freeze replaced the frozen file")
	}
	if names := storeNames(t, store.Directory); !slices.Equal(names, []string{want + ".json"}) {
		t.Fatalf("the store holds more than the frozen package: %v", names)
	}

	// Another package's name, already holding different bytes, is never taken over.
	ground := groundTruthPackage(t)
	groundDigest := packageDigest(t, ground)
	planted := filepath.Join(store.Directory, groundDigest+".json")
	if err := os.WriteFile(planted, canonical, 0o644); err != nil {
		t.Fatal(err)
	}
	_, err = store.Freeze(ctx, ground, packageLimits())
	refusedWith(t, err, "never replaced")
	if kept, _ := os.ReadFile(planted); !bytes.Equal(kept, canonical) {
		t.Fatal("a refused freeze rewrote the file in its way")
	}
	_, err = store.Read(groundDigest, packageLimits())
	refusedWith(t, err, "differ from digest")

	// A frozen file edited behind the store is refused, and not repaired.
	if err := os.Chmod(path, 0o644); err != nil {
		t.Fatal(err)
	}
	tampered := bytes.Replace(canonical, []byte(`"tick":17`), []byte(`"tick":18`), 1)
	if err := os.WriteFile(path, tampered, 0o644); err != nil {
		t.Fatal(err)
	}
	_, err = store.Read(want, packageLimits())
	refusedWith(t, err, "differ from digest")
	_, err = store.Freeze(ctx, p, packageLimits())
	refusedWith(t, err, "never replaced")
	if kept, _ := os.ReadFile(path); !bytes.Equal(kept, tampered) {
		t.Fatal("the store rewrote a frozen file")
	}
}

func TestTheStoreReadsOnlyCanonicalRegularFiles(t *testing.T) {
	store := fusioncache.PackageStore{Directory: t.TempDir()}
	canonical := encodePackage(t, routedPackage(t))
	put := func(data []byte) string {
		digest := sha256Hex(data)
		if err := os.WriteFile(filepath.Join(store.Directory, digest+".json"), data, 0o444); err != nil {
			t.Fatal(err)
		}
		return digest
	}

	unknown := put(bytes.Replace(canonical, []byte(`{"schema"`), []byte(`{"extra":1,"schema"`), 1))
	_, err := store.Read(unknown, packageLimits())
	refusedWith(t, err, "unknown field")

	var indented bytes.Buffer
	if err := json.Indent(&indented, canonical, "", "  "); err != nil {
		t.Fatal(err)
	}
	_, err = store.Read(put(indented.Bytes()), packageLimits())
	refusedWith(t, err, "not in canonical form")

	outside := filepath.Join(t.TempDir(), "package.json")
	if err := os.WriteFile(outside, canonical, 0o444); err != nil {
		t.Fatal(err)
	}
	link := sha256Hex(canonical)
	if err := os.Symlink(outside, filepath.Join(store.Directory, link+".json")); err != nil {
		t.Fatal(err)
	}
	_, err = store.Read(link, packageLimits())
	refusedWith(t, err, "not a regular file")
	_, err = store.Freeze(context.Background(), routedPackage(t), packageLimits())
	refusedWith(t, err, "never replaced")

	_, err = store.Read("not-a-digest", packageLimits())
	refusedWith(t, err, "package digest required")
	_, err = store.Read(hashOf("absent"), packageLimits())
	if !errors.Is(err, fs.ErrNotExist) {
		t.Fatalf("an absent package read as %v", err)
	}
}

func TestFreezeRefusesBeforeWriting(t *testing.T) {
	store := fusioncache.PackageStore{Directory: t.TempDir()}
	_, err := store.Freeze(noContext, routedPackage(t), packageLimits())
	refusedWith(t, err, "context required")
	cancelled, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := store.Freeze(cancelled, routedPackage(t), packageLimits()); !errors.Is(err, context.Canceled) {
		t.Fatalf("a cancelled freeze returned %v", err)
	}
	invalid := clone(t, routedPackage(t))
	invalid.Verifier.Correct = false
	_, err = store.Freeze(context.Background(), invalid, packageLimits())
	refusedWith(t, err, "correct verdict")
	if names := storeNames(t, store.Directory); len(names) != 0 {
		t.Fatalf("a refused freeze wrote %v", names)
	}
	_, err = fusioncache.PackageStore{}.Freeze(context.Background(), routedPackage(t), packageLimits())
	refusedWith(t, err, "store directory required")
}

func TestACorrectionChainsToItsPredecessorAndNeverForks(t *testing.T) {
	ctx := context.Background()
	l := packageLimits()
	store := fusioncache.PackageStore{Directory: t.TempDir()}
	first := routedPackage(t)
	firstDigest := packageDigest(t, first)
	if _, err := store.Freeze(ctx, first, l); err != nil {
		t.Fatal(err)
	}
	second, err := fusioncache.Supersede(first, correction(t, first, "sft-line normalized"), l)
	if err != nil {
		t.Fatal(err)
	}
	secondDigest := packageDigest(t, second)

	elsewhere := fusioncache.PackageStore{Directory: t.TempDir()}
	_, err = elsewhere.Freeze(ctx, second, l)
	refusedWith(t, err, "is not frozen in this store")

	if _, err := store.Freeze(ctx, second, l); err != nil {
		t.Fatal(err)
	}
	if kept, _ := os.ReadFile(filepath.Join(store.Directory, firstDigest+".json")); !bytes.Equal(kept, encodePackage(t, first)) {
		t.Fatal("a correction changed the package it corrects")
	}
	next, nextDigest, ok, err := store.Successor(firstDigest, l)
	if err != nil || !ok || nextDigest != secondDigest || !reflect.DeepEqual(next, second) {
		t.Fatalf("the successor of the first revision is %s, %v, %v", nextDigest, ok, err)
	}
	if _, _, ok, err := store.Successor(secondDigest, l); ok || err != nil {
		t.Fatalf("the latest revision reports a successor: %v, %v", ok, err)
	}
	if _, err := store.Freeze(ctx, second, l); err != nil {
		t.Fatalf("refreezing the recorded successor: %v", err)
	}

	forked, err := fusioncache.Supersede(first, correction(t, first, "a different fix"), l)
	if err != nil {
		t.Fatal(err)
	}
	forkedDigest := packageDigest(t, forked)
	_, err = store.Freeze(ctx, forked, l)
	refusedWith(t, err, "already has another successor")
	if _, err := os.Stat(filepath.Join(store.Directory, forkedDigest+".json")); !errors.Is(err, fs.ErrNotExist) {
		t.Fatalf("a refused fork was installed: %v", err)
	}

	// The second revision has no successor yet, so only the revision and unit
	// check stands between it and these.
	skipped := correction(t, second, "sft-line skipping a revision")
	skipped.Revision, skipped.Supersedes = 4, secondDigest
	_, err = store.Freeze(ctx, skipped, l)
	refusedWith(t, err, "must follow revision")
	stranger := groundTruthPackage(t)
	stranger.Revision, stranger.Supersedes = 3, secondDigest
	_, err = store.Freeze(ctx, stranger, l)
	refusedWith(t, err, "must follow revision")
	if _, _, ok, err := store.Successor(secondDigest, l); ok || err != nil {
		t.Fatalf("a refused revision claimed the successor slot: %v, %v", ok, err)
	}

	fixed := clone(t, second)
	fixed.Verifier.ReceiptSHA256 = hashOf("verdict v2")
	fixed.Artifacts = artifactIndex(fixed)
	third, err := fusioncache.Supersede(second, fixed, l)
	if err != nil {
		t.Fatal(err)
	}
	thirdDigest := packageDigest(t, third)
	if _, err := store.Freeze(ctx, third, l); err != nil {
		t.Fatal(err)
	}
	history, err := store.History(thirdDigest, l)
	if err != nil {
		t.Fatal(err)
	}
	if len(history) != 3 || !reflect.DeepEqual(history[0], first) || !reflect.DeepEqual(history[1], second) || !reflect.DeepEqual(history[2], third) {
		t.Fatalf("history is not first, second, third: %d revisions", len(history))
	}
	if history[1].Supersedes != firstDigest || history[2].Supersedes != secondDigest {
		t.Fatal("a revision does not name its predecessor")
	}

	// A fork written behind the store's back is found by walking its history.
	if err := os.WriteFile(filepath.Join(store.Directory, forkedDigest+".json"), encodePackage(t, forked), 0o444); err != nil {
		t.Fatal(err)
	}
	_, err = store.History(forkedDigest, l)
	refusedWith(t, err, "records another successor")
}

func TestAManifestNamesCurrentPackagesOfDistinctUnits(t *testing.T) {
	ctx := context.Background()
	l := packageLimits()
	store := fusioncache.PackageStore{Directory: t.TempDir()}
	routed, ground := routedPackage(t), groundTruthPackage(t)
	routedDigest, groundDigest := packageDigest(t, routed), packageDigest(t, ground)
	for _, p := range []fusioncache.CapabilityPackage{routed, ground} {
		if _, err := store.Freeze(ctx, p, l); err != nil {
			t.Fatal(err)
		}
	}

	manifest, receipt, err := store.FreezeManifest(ctx, []string{routedDigest, groundDigest}, l)
	if err != nil {
		t.Fatal(err)
	}
	if want := slices.Sorted(slices.Values([]string{routedDigest, groundDigest})); !slices.Equal(manifest.Packages, want) || manifest.Schema != fusioncache.PackageManifestSchema {
		t.Fatalf("manifest %+v", manifest)
	}
	data, err := os.ReadFile(receipt.Location)
	if err != nil {
		t.Fatal(err)
	}
	if receipt.Location != filepath.Join(store.Directory, receipt.SHA256+".manifest.json") || sha256Hex(data) != receipt.SHA256 || receipt.Bytes != int64(len(data)) {
		t.Fatalf("the manifest receipt does not name its bytes: %+v", receipt)
	}
	if want := `{"schema":1,"packages":["` + manifest.Packages[0] + `","` + manifest.Packages[1] + `"]}`; string(data) != want {
		t.Fatalf("manifest bytes %s", data)
	}
	_, reordered, err := store.FreezeManifest(ctx, []string{groundDigest, routedDigest}, l)
	if err != nil || reordered != receipt {
		t.Fatalf("argument order moved the manifest: %+v, %v", reordered, err)
	}
	read, err := store.ReadManifest(receipt.SHA256, l)
	if err != nil || !reflect.DeepEqual(read, manifest) {
		t.Fatalf("read back %+v, %v", read, err)
	}

	armC := clone(t, routed)
	armC.Router = nil
	armC.Teachers[0].Weight, armC.Teachers[1].Weight = 0.5, 0.5
	armC.Artifacts = artifactIndex(armC)
	if _, err := store.Freeze(ctx, armC, l); err != nil {
		t.Fatal(err)
	}
	refusals := map[string]struct {
		digests []string
		limits  fusioncache.PackageLimits
		want    string
	}{
		"empty":        {nil, l, "names 1 to"},
		"twice":        {[]string{routedDigest, routedDigest}, l, "unique and ascending"},
		"not_a_digest": {[]string{"routed"}, l, "unique and ascending"},
		"too_many":     {[]string{routedDigest, groundDigest}, fusioncache.PackageLimits{MaxBytes: l.MaxBytes, MaxTeachers: 4, MaxMaskSpans: 8, MaxArtifacts: 64, MaxPackages: 1}, "names 1 to 1"},
		"same_unit":    {[]string{routedDigest, packageDigest(t, armC)}, l, "same capability on the same example"},
	}
	for name, c := range refusals {
		t.Run(name, func(t *testing.T) {
			_, _, err := store.FreezeManifest(ctx, c.digests, c.limits)
			refusedWith(t, err, c.want)
		})
	}
	if _, _, err := store.FreezeManifest(ctx, []string{hashOf("never frozen")}, l); !errors.Is(err, fs.ErrNotExist) {
		t.Fatalf("a manifest named a package that is not frozen: %v", err)
	}
	_, _, err = store.FreezeManifest(noContext, []string{routedDigest}, l)
	refusedWith(t, err, "context required")

	corrected, err := fusioncache.Supersede(ground, correction(t, ground, "reference-line v2"), l)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Freeze(ctx, corrected, l); err != nil {
		t.Fatal(err)
	}
	_, _, err = store.FreezeManifest(ctx, []string{routedDigest, groundDigest}, l)
	refusedWith(t, err, "is superseded by")
	if _, err := store.ReadManifest(receipt.SHA256, l); err != nil {
		t.Fatalf("a manifest frozen before a correction stopped reading: %v", err)
	}
	if _, _, err := store.FreezeManifest(ctx, []string{routedDigest, packageDigest(t, corrected)}, l); err != nil {
		t.Fatalf("a manifest of current revisions was refused: %v", err)
	}

	put := func(data []byte) string {
		digest := sha256Hex(data)
		if err := os.WriteFile(filepath.Join(store.Directory, digest+".manifest.json"), data, 0o444); err != nil {
			t.Fatal(err)
		}
		return digest
	}
	var indented bytes.Buffer
	if err := json.Indent(&indented, data, "", "  "); err != nil {
		t.Fatal(err)
	}
	_, err = store.ReadManifest(put(indented.Bytes()), l)
	refusedWith(t, err, "not in canonical form")
	_, err = store.ReadManifest(put([]byte(`{"schema":1,"packages":["`+routedDigest+`"],"extra":true}`)), l)
	refusedWith(t, err, "unknown field")
	if _, err := store.ReadManifest(put([]byte(`{"schema":1,"packages":["`+hashOf("never frozen")+`"]}`)), l); !errors.Is(err, fs.ErrNotExist) {
		t.Fatalf("a manifest of a missing package read as %v", err)
	}
	if err := os.Chmod(receipt.Location, 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(receipt.Location, bytes.Replace(data, []byte(`"schema":1`), []byte(`"schema":2`), 1), 0o644); err != nil {
		t.Fatal(err)
	}
	_, err = store.ReadManifest(receipt.SHA256, l)
	refusedWith(t, err, "differ from digest")
}
