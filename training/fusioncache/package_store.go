package fusioncache

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"slices"
)

// PackageManifestSchema is the only package manifest shape this package reads
// or writes.
const PackageManifestSchema = 1

const (
	packageSuffix   = ".json"
	successorSuffix = ".successor.json"
	manifestSuffix  = ".manifest.json"
)

// PackageManifest names a set of frozen capability packages by digest, in
// ascending order. Its own digest is the SHA-256 of its canonical JSON.
type PackageManifest struct {
	Schema   int      `json:"schema"`
	Packages []string `json:"packages"`
}

// PackageStore freezes capability packages in one existing directory. A
// package is installed as <digest>.json, the claim a correction lays on its
// predecessor as <predecessor>.successor.json, and a manifest as
// <digest>.manifest.json; each file's SHA-256 is the digest in its name.
// A file the store installed is never replaced: writing the same bytes again
// is a no-op, and writing anything else is refused. The caller owns directory
// creation and authorization for this filesystem path.
type PackageStore struct{ Directory string }

// Freeze validates p and installs its canonical JSON under its digest. A
// revision after the first requires its predecessor frozen in this store, one
// revision earlier and for the same capability, dataset and example, and
// claims the predecessor's single successor slot before installing itself, so
// two different corrections of one package are never both frozen. Repeating
// an identical Freeze is idempotent.
func (s PackageStore) Freeze(ctx context.Context, p CapabilityPackage, l PackageLimits) (Receipt, error) {
	if ctx == nil {
		return Receipt{}, refuse("context required")
	}
	if err := ctx.Err(); err != nil {
		return Receipt{}, err
	}
	encoded, digest, err := EncodePackage(p, l)
	if err != nil {
		return Receipt{}, err
	}
	if p.Revision > 1 {
		previous, err := s.Read(p.Supersedes, l)
		if err != nil {
			return Receipt{}, fmt.Errorf("%w: predecessor %s is not frozen in this store: %v", ErrContract, p.Supersedes, err)
		}
		if previous.Revision != p.Revision-1 || unitKey(previous) != unitKey(p) {
			return Receipt{}, refuse("revision %d must follow revision %d of the same capability and example", p.Revision, previous.Revision)
		}
		if err := ctx.Err(); err != nil {
			return Receipt{}, err
		}
		if err := s.install(p.Supersedes+successorSuffix, encoded); err != nil {
			return Receipt{}, fmt.Errorf("%w: package %s already has another successor: %v", ErrContract, p.Supersedes, err)
		}
	}
	if err := ctx.Err(); err != nil {
		return Receipt{}, err
	}
	if err := s.install(digest+packageSuffix, encoded); err != nil {
		return Receipt{}, err
	}
	return Receipt{Location: filepath.Join(s.Directory, digest+packageSuffix), SHA256: digest, Bytes: int64(len(encoded))}, nil
}

// Read returns the package frozen under digest, after checking that the file
// is a regular file within the byte bound, that its bytes hash to digest and
// that they are exactly the canonical encoding of a valid package.
func (s PackageStore) Read(digest string, l PackageLimits) (CapabilityPackage, error) {
	if err := validatePackageLimits(l); err != nil {
		return CapabilityPackage{}, err
	}
	if !validSHA(digest) {
		return CapabilityPackage{}, refuse("package digest required")
	}
	data, err := readFrozen(filepath.Join(s.Directory, digest+packageSuffix), l.MaxBytes)
	if err != nil {
		return CapabilityPackage{}, err
	}
	return decodeFrozen(data, digest, l)
}

func decodeFrozen(data []byte, digest string, l PackageLimits) (CapabilityPackage, error) {
	if byteDigest(data) != digest {
		return CapabilityPackage{}, refuse("frozen package bytes differ from digest %s", digest)
	}
	p, canonical, err := DecodePackage(data, l)
	if err != nil {
		return CapabilityPackage{}, err
	}
	if canonical != digest {
		return CapabilityPackage{}, refuse("frozen package %s is not in canonical form", digest)
	}
	return p, nil
}

// Successor returns the correction frozen for the package digest names, with
// the correction's own digest, and false when the package was never
// superseded in this store.
func (s PackageStore) Successor(digest string, l PackageLimits) (CapabilityPackage, string, bool, error) {
	if err := validatePackageLimits(l); err != nil {
		return CapabilityPackage{}, "", false, err
	}
	if !validSHA(digest) {
		return CapabilityPackage{}, "", false, refuse("package digest required")
	}
	data, err := readFrozen(filepath.Join(s.Directory, digest+successorSuffix), l.MaxBytes)
	if errors.Is(err, fs.ErrNotExist) {
		return CapabilityPackage{}, "", false, nil
	}
	if err != nil {
		return CapabilityPackage{}, "", false, err
	}
	next := byteDigest(data)
	p, err := decodeFrozen(data, next, l)
	if err != nil {
		return CapabilityPackage{}, "", false, err
	}
	if p.Supersedes != digest {
		return CapabilityPackage{}, "", false, refuse("successor of %s supersedes %s", digest, p.Supersedes)
	}
	return p, next, true, nil
}

// History returns every revision that leads to the package digest names,
// first revision first. Each link is checked: a later revision names its
// predecessor's digest, comes exactly one revision after it, keeps its
// capability, dataset and example, and is the successor the store recorded
// for it.
func (s PackageStore) History(digest string, l PackageLimits) ([]CapabilityPackage, error) {
	current, err := s.Read(digest, l)
	if err != nil {
		return nil, err
	}
	chain := []CapabilityPackage{current}
	for current.Revision > 1 {
		previous, err := s.Read(current.Supersedes, l)
		if err != nil {
			return nil, err
		}
		if previous.Revision != current.Revision-1 || unitKey(previous) != unitKey(current) {
			return nil, refuse("revision %d does not follow revision %d of the same unit", current.Revision, previous.Revision)
		}
		_, next, ok, err := s.Successor(current.Supersedes, l)
		if err != nil {
			return nil, err
		}
		if !ok || next != digest {
			return nil, refuse("the store records another successor for %s", current.Supersedes)
		}
		chain = append(chain, previous)
		current, digest = previous, current.Supersedes
	}
	slices.Reverse(chain)
	return chain, nil
}

// FreezeManifest names a set of packages frozen in this store and freezes that
// manifest under its digest. Every package must read back valid, no two may
// share a capability, dataset and example, and none may already be superseded:
// a new set is built from current revisions. The digests are sorted; one
// named twice is refused.
func (s PackageStore) FreezeManifest(ctx context.Context, digests []string, l PackageLimits) (PackageManifest, Receipt, error) {
	if ctx == nil {
		return PackageManifest{}, Receipt{}, refuse("context required")
	}
	if err := ctx.Err(); err != nil {
		return PackageManifest{}, Receipt{}, err
	}
	m := PackageManifest{Schema: PackageManifestSchema, Packages: slices.Sorted(slices.Values(digests))}
	if err := s.checkManifest(m, l, true); err != nil {
		return PackageManifest{}, Receipt{}, err
	}
	encoded, err := json.Marshal(m)
	if err != nil {
		return PackageManifest{}, Receipt{}, refuse("encode manifest: %v", err)
	}
	if int64(len(encoded)) > l.MaxBytes {
		return PackageManifest{}, Receipt{}, refuse("manifest JSON exceeds its byte limit")
	}
	if err := ctx.Err(); err != nil {
		return PackageManifest{}, Receipt{}, err
	}
	digest := byteDigest(encoded)
	if err := s.install(digest+manifestSuffix, encoded); err != nil {
		return PackageManifest{}, Receipt{}, err
	}
	return m, Receipt{Location: filepath.Join(s.Directory, digest+manifestSuffix), SHA256: digest, Bytes: int64(len(encoded))}, nil
}

// ReadManifest returns the manifest frozen under digest after checking its
// bytes, its canonical form and every package it names. A package superseded
// after the manifest was frozen does not invalidate it: a frozen manifest
// records the set as it was.
func (s PackageStore) ReadManifest(digest string, l PackageLimits) (PackageManifest, error) {
	if err := validatePackageLimits(l); err != nil {
		return PackageManifest{}, err
	}
	if !validSHA(digest) {
		return PackageManifest{}, refuse("manifest digest required")
	}
	data, err := readFrozen(filepath.Join(s.Directory, digest+manifestSuffix), l.MaxBytes)
	if err != nil {
		return PackageManifest{}, err
	}
	if byteDigest(data) != digest {
		return PackageManifest{}, refuse("frozen manifest bytes differ from digest %s", digest)
	}
	var m PackageManifest
	if err := decodeStrict(data, &m); err != nil {
		return PackageManifest{}, err
	}
	canonical, err := json.Marshal(m)
	if err != nil || !bytes.Equal(canonical, data) {
		return PackageManifest{}, refuse("frozen manifest %s is not in canonical form", digest)
	}
	if err := s.checkManifest(m, l, false); err != nil {
		return PackageManifest{}, err
	}
	return m, nil
}

func (s PackageStore) checkManifest(m PackageManifest, l PackageLimits, current bool) error {
	if err := validatePackageLimits(l); err != nil {
		return err
	}
	if m.Schema != PackageManifestSchema || len(m.Packages) == 0 || len(m.Packages) > l.MaxPackages {
		return refuse("a manifest names 1 to %d packages", l.MaxPackages)
	}
	units := make(map[string]string, len(m.Packages))
	for i, digest := range m.Packages {
		if !validSHA(digest) || (i > 0 && digest <= m.Packages[i-1]) {
			return refuse("manifest digests must be valid, unique and ascending")
		}
		p, err := s.Read(digest, l)
		if err != nil {
			return err
		}
		if other, ok := units[unitKey(p)]; ok {
			return refuse("packages %s and %s teach the same capability on the same example", other, digest)
		}
		units[unitKey(p)] = digest
		if !current {
			continue
		}
		_, next, superseded, err := s.Successor(digest, l)
		if err != nil {
			return err
		}
		if superseded {
			return refuse("package %s is superseded by %s", digest, next)
		}
	}
	return nil
}

// install writes data to a new file in the store directory, syncs it, makes it
// read-only and links it to name. A hard link installs a complete file
// atomically and never replaces a path, so an existing name is compared
// instead: the same bytes are the same freeze, and different bytes are
// refused and left as they are.
func (s PackageStore) install(name string, data []byte) error {
	if s.Directory == "" {
		return refuse("store directory required")
	}
	target := filepath.Join(s.Directory, name)
	temporary, err := os.CreateTemp(s.Directory, ".capability-package-*")
	if err != nil {
		return err
	}
	temporaryName := temporary.Name()
	defer os.Remove(temporaryName)
	if _, err := temporary.Write(data); err != nil {
		return errors.Join(err, temporary.Close())
	}
	if err := temporary.Sync(); err != nil {
		return errors.Join(err, temporary.Close())
	}
	if err := temporary.Chmod(0o444); err != nil {
		return errors.Join(err, temporary.Close())
	}
	if err := temporary.Close(); err != nil {
		return err
	}
	if err := os.Link(temporaryName, target); err != nil {
		if !errors.Is(err, fs.ErrExist) {
			return err
		}
		existing, err := readFrozen(target, int64(len(data)))
		if err != nil {
			return fmt.Errorf("%w: frozen %s differs and is never replaced: %v", ErrContract, name, err)
		}
		if !bytes.Equal(existing, data) {
			return refuse("frozen %s differs and is never replaced", name)
		}
	}
	directory, err := os.Open(s.Directory)
	if err != nil {
		return err
	}
	return errors.Join(directory.Sync(), directory.Close())
}

// readFrozen reads a regular file of at most maximum bytes. A symbolic link
// is refused: the store installs hard links only.
func readFrozen(path string, maximum int64) ([]byte, error) {
	info, err := os.Lstat(path)
	if err != nil {
		return nil, err
	}
	if !info.Mode().IsRegular() || info.Size() > maximum {
		return nil, refuse("%s is not a regular file within %d bytes", filepath.Base(path), maximum)
	}
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	data, readErr := io.ReadAll(io.LimitReader(file, maximum+1))
	if err := errors.Join(readErr, file.Close()); err != nil {
		return nil, err
	}
	if int64(len(data)) > maximum {
		return nil, refuse("%s exceeds %d bytes", filepath.Base(path), maximum)
	}
	return data, nil
}
