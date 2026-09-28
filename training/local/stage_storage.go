package local

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"slices"
	"sort"
	"strings"

	"github.com/tayi-ai/arandu-llama/checkpoint"
	"github.com/tayi-ai/arandu-llama/training/pipeline"
)

func syncLocalDirectory(path string) error {
	f, e := os.Open(path)
	if e != nil {
		return e
	}
	return errors.Join(f.Sync(), f.Close())
}
func stageRoot(path string) (*os.Root, error) {
	if !filepath.IsAbs(path) || filepath.Clean(path) != path {
		return nil, ErrStage
	}
	info, e := os.Lstat(path)
	if e != nil || !info.IsDir() {
		return nil, errors.Join(ErrStage, e)
	}
	return os.OpenRoot(path)
}
func stageOpen(root *os.Root, name string) (*os.File, error) {
	if !filepath.IsLocal(name) || name == "." || filepath.Clean(name) != name {
		return nil, ErrStage
	}
	parts := strings.Split(filepath.ToSlash(name), "/")
	path := ""
	for i, part := range parts {
		path = filepath.Join(path, part)
		s, e := root.Lstat(path)
		if e != nil {
			return nil, e
		}
		if i < len(parts)-1 {
			if !s.IsDir() {
				return nil, ErrStage
			}
		} else if !s.Mode().IsRegular() {
			return nil, ErrStage
		}
	}
	before, e := root.Lstat(name)
	if e != nil {
		return nil, e
	}
	f, e := root.Open(name)
	if e != nil {
		return nil, e
	}
	after, e := f.Stat()
	if e != nil || !os.SameFile(before, after) {
		f.Close()
		return nil, ErrStage
	}
	return f, nil
}

type stageContextReader struct {
	ctx context.Context
	r   io.Reader
}

func (r stageContextReader) Read(p []byte) (int, error) {
	if e := r.ctx.Err(); e != nil {
		return 0, e
	}
	return r.r.Read(p)
}
func stageRead(ctx context.Context, root *os.Root, name string, max int64, sha string) ([]byte, error) {
	f, e := stageOpen(root, name)
	if e != nil {
		return nil, e
	}
	defer f.Close()
	s, e := f.Stat()
	if e != nil || s.Size() < 1 || s.Size() > max {
		return nil, ErrStage
	}
	b, e := io.ReadAll(io.LimitReader(stageContextReader{ctx, f}, s.Size()+1))
	if e != nil || int64(len(b)) != s.Size() || (sha != "" && fmtHash(b) != sha) {
		return nil, errors.Join(ErrStage, e)
	}
	return b, ctx.Err()
}
func stageCopy(ctx context.Context, source *os.Root, a pipeline.StageArtifact, target string) error {
	f, e := stageOpen(source, a.Path)
	if e != nil {
		return e
	}
	defer f.Close()
	s, e := f.Stat()
	if e != nil || s.Size() != a.Bytes {
		return ErrStage
	}
	out, e := os.OpenFile(target, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
	if e != nil {
		return e
	}
	hash := sha256.New()
	n, copyErr := io.CopyBuffer(io.MultiWriter(out, hash), io.LimitReader(stageContextReader{ctx, f}, a.Bytes+1), make([]byte, 64<<10))
	closeErr := errors.Join(out.Sync(), out.Close())
	if copyErr != nil || closeErr != nil || n != a.Bytes || hex.EncodeToString(hash.Sum(nil)) != a.SHA256 {
		return errors.Join(ErrStage, copyErr, closeErr)
	}
	return ctx.Err()
}
func stageHashFile(ctx context.Context, root *os.Root, name string, max int64) (pipeline.StageArtifact, error) {
	f, e := stageOpen(root, name)
	if e != nil {
		return pipeline.StageArtifact{}, e
	}
	defer f.Close()
	s, e := f.Stat()
	if e != nil || s.Size() < 1 || s.Size() > max {
		return pipeline.StageArtifact{}, ErrStage
	}
	h := sha256.New()
	n, e := io.Copy(h, io.LimitReader(stageContextReader{ctx, f}, s.Size()+1))
	if e != nil || n != s.Size() {
		return pipeline.StageArtifact{}, errors.Join(ErrStage, e)
	}
	return pipeline.StageArtifact{Path: name, SHA256: hex.EncodeToString(h.Sum(nil)), Bytes: n}, nil
}
func (h *Stage) prepare(ctx context.Context, c pipeline.StageContext) ([]example, error) {
	if _, err := h.identity(ctx, c); err != nil {
		return nil, fmt.Errorf("local SFT: execution admission: %w", err)
	}
	root, err := stageRoot(c.ArtifactDirectory)
	if err != nil {
		return nil, fmt.Errorf("local SFT: artifact directory: %w", err)
	}
	defer root.Close()
	seed := h.seedName()
	_, err = root.Lstat(seed)
	if errors.Is(err, os.ErrNotExist) {
		// Pin and copy the underlying derived data before any numerical work.
		data, err := pipeline.ResolveArtifact(ctx, c, h.config.DataDirectory, h.config.Protocol.Data, h.config.Protocol.Limits.MaxDataBytes)
		if err != nil {
			return nil, err
		}
		dataRoot, err := stageRoot(h.config.DataDirectory)
		if err != nil {
			return nil, err
		}
		defer dataRoot.Close()
		temp, err := os.MkdirTemp(c.ArtifactDirectory, ".sft-import-")
		if err != nil {
			return nil, err
		}
		defer os.RemoveAll(temp)
		if err := stageCopy(ctx, dataRoot, data.Binding.Artifact, filepath.Join(temp, "data.jsonl")); err != nil {
			return nil, err
		}
		// A fresh start imports only its data: the initial adapter is computed
		// from the recipe's initializer, not copied from anywhere.
		if !h.config.Protocol.fresh() {
			initialRoot, err := stageRoot(h.config.InitialDirectory)
			if err != nil {
				return nil, err
			}
			defer initialRoot.Close()
			p := h.config.Protocol.Initial
			for _, a := range []pipeline.StageArtifact{p.Manifest, p.Adapter, p.Optimizer} {
				if err := stageCopy(ctx, initialRoot, a, filepath.Join(temp, a.Path)); err != nil {
					return nil, err
				}
			}
		}
		// Distillation caches are pinned beside the data by their digests, so
		// a source changed after admission cannot reach a step.
		if caches := h.config.Protocol.cacheArtifacts(); len(caches) > 0 {
			cacheRoot, err := stageRoot(h.config.CacheDirectory)
			if err != nil {
				return nil, err
			}
			defer cacheRoot.Close()
			if err := os.Mkdir(filepath.Join(temp, "caches"), 0700); err != nil {
				return nil, err
			}
			for _, a := range caches {
				if err := stageCopy(ctx, cacheRoot, a, filepath.Join(temp, "caches", a.Path)); err != nil {
					return nil, fmt.Errorf("local SFT: distillation cache: %w", err)
				}
			}
			if err := syncLocalDirectory(filepath.Join(temp, "caches")); err != nil {
				return nil, err
			}
		}
		if err := syncLocalDirectory(temp); err != nil {
			return nil, err
		}
		if err := os.Rename(temp, filepath.Join(c.ArtifactDirectory, seed)); err != nil {
			return nil, err
		}
		if err := syncLocalDirectory(c.ArtifactDirectory); err != nil {
			return nil, err
		}
	} else if err != nil {
		return nil, err
	}
	info, err := root.Lstat(seed)
	if err != nil || !info.IsDir() {
		return nil, ErrStage
	}
	rows, err := h.dataset(ctx, root)
	if err != nil {
		return nil, fmt.Errorf("local SFT: tokenized dataset: %w", err)
	}
	if _, err := h.initialState(ctx, root, rows); err != nil {
		return nil, fmt.Errorf("local SFT: initial checkpoint: %w", err)
	}
	if err := root.Mkdir(h.outputName(), 0700); err != nil && !errors.Is(err, os.ErrExist) {
		return nil, err
	}
	info, err = root.Lstat(h.outputName())
	if err != nil || !info.IsDir() {
		return nil, ErrStage
	}
	return rows, syncLocalDirectory(c.ArtifactDirectory)
}
func (h *Stage) dataset(ctx context.Context, root *os.Root) ([]example, error) {
	p := h.config.Protocol
	b, e := stageRead(ctx, root, filepath.Join(h.seedName(), "data.jsonl"), p.Limits.MaxDataBytes, p.Local.DataSHA256)
	if e != nil {
		return nil, e
	}
	d := json.NewDecoder(bytes.NewReader(b))
	d.DisallowUnknownFields()
	var rows []example
	seen := map[string]bool{}
	var tokens int64
	last := 0
	for {
		var row example
		e := d.Decode(&row)
		if errors.Is(e, io.EOF) {
			break
		}
		if e != nil {
			return nil, e
		}
		if len(rows) >= p.Local.ExampleCount || !stageID(row.ID) || seen[row.ID] || len(row.InputIDs) < 2 || len(row.InputIDs) != len(row.Labels) || row.PromptTokens < 1 || row.PromptTokens >= len(row.InputIDs) || int64(len(row.InputIDs)) > p.Limits.MaxDataTokens-tokens || p.Local.RequireLengthOrder && len(row.InputIDs) < last {
			return nil, ErrStage
		}
		for i, v := range row.InputIDs {
			if v < 0 || v >= int64(p.Student.Vocabulary) || i < row.PromptTokens && row.Labels[i] != -100 || i >= row.PromptTokens && row.Labels[i] != v {
				return nil, ErrStage
			}
		}
		if uint64(len(rows)) >= p.Initial.Step && uint64(len(rows)) < p.TargetStep && len(row.InputIDs) > h.stage.MaxTokens {
			return nil, ErrStage
		}
		tokens += int64(len(row.InputIDs))
		last = len(row.InputIDs)
		seen[row.ID] = true
		rows = append(rows, row)
	}
	if len(rows) != p.Local.ExampleCount {
		return nil, ErrStage
	}
	return rows, ctx.Err()
}
func (h *Stage) checkpoint(ctx context.Context, root *os.Root, path string, step uint64, rows []example) (stepManifest, []pipeline.StageArtifact, error) {
	p := h.config.Protocol
	var m stepManifest
	b, e := stageRead(ctx, root, filepath.Join(path, "manifest.json"), p.Limits.MaxManifestBytes, "")
	if e != nil {
		return m, nil, e
	}
	if e := json.Unmarshal(b, &m); e != nil {
		return m, nil, e
	}
	if m.Step != step || step < 1 || step > uint64(len(rows)) || m.ExampleID != rows[step-1].ID || m.BaseRevision != p.Local.BaseRevision || m.SupervisedTokens != len(rows[step-1].InputIDs)-rows[step-1].PromptTokens || !finiteStage(m.LossBefore) || m.LossBefore < 0 || !validHash(m.UpdatedAdapterSHA) || !validHash(m.LogitsAfterSHA) {
		return m, nil, ErrStage
	}
	if m.RecipeSHA256 != p.Local.Digest() && !(step == p.Initial.Step && p.Initial.AllowHistorical && m.RecipeSHA256 == "" && fmtHash(b) == p.Initial.Manifest.SHA256) {
		return m, nil, ErrStage
	}
	artifacts := []pipeline.StageArtifact{{Path: filepath.Join(path, "manifest.json"), SHA256: fmtHash(b), Bytes: int64(len(b))}}
	if step == p.Initial.Step && artifacts[0].SHA256 != p.Initial.Manifest.SHA256 {
		return m, nil, ErrStage
	}
	for i, name := range []string{"adapter_model.safetensors", "optimizer_moments.safetensors"} {
		a, e := stageHashFile(ctx, root, filepath.Join(path, name), p.Limits.MaxTensorFileBytes)
		if e != nil {
			return m, nil, e
		}
		expected := m.AdapterFileSHA
		if i == 1 {
			expected = m.OptimizerFileSHA
		}
		if a.SHA256 != expected {
			return m, nil, ErrStage
		}
		if step == p.Initial.Step {
			pin := p.Initial.Adapter
			if i == 1 {
				pin = p.Initial.Optimizer
			}
			if a.SHA256 != pin.SHA256 || a.Bytes != pin.Bytes {
				return m, nil, ErrStage
			}
		}
		logical, e := h.tensorFile(ctx, root, a, i == 1)
		if e != nil || i == 0 && logical != m.UpdatedAdapterSHA {
			return m, nil, errors.Join(ErrStage, e)
		}
		artifacts = append(artifacts, a)
	}
	return m, artifacts, nil
}
func (h *Stage) tensorFile(ctx context.Context, root *os.Root, a pipeline.StageArtifact, moments bool) (string, error) {
	f, e := stageOpen(root, a.Path)
	if e != nil {
		return "", e
	}
	defer f.Close()
	index, e := checkpoint.OpenSafetensors(f, a.Bytes, h.config.Protocol.Limits.Checkpoint)
	if e != nil {
		return "", e
	}
	type tensor struct {
		name   string
		shape  []uint64
		second bool
	}
	var expected []tensor
	for _, p := range h.config.Protocol.Local.Initializer.Projections {
		for _, q := range []tensor{{p.Name + ".lora_A.default.weight", []uint64{uint64(p.Rank), uint64(p.Input)}, false}, {p.Name + ".lora_B.default.weight", []uint64{uint64(p.Output), uint64(p.Rank)}, false}} {
			if moments {
				expected = append(expected, tensor{"m." + q.name, q.shape, false}, tensor{"v." + q.name, q.shape, true})
			} else {
				expected = append(expected, q)
			}
		}
	}
	if len(index.Tensors()) != len(expected) {
		return "", ErrStage
	}
	hash := sha256.New()
	buffer := make([]byte, h.config.Protocol.Limits.Checkpoint.MaxChunkBytes/4*4)
	for _, q := range expected {
		item, ok := index.Tensor(q.name)
		if !ok || item.DType != "F32" || !slices.Equal(item.Shape, q.shape) {
			return "", ErrStage
		}
		reader, e := index.TensorReader(q.name)
		if e != nil {
			return "", e
		}
		hash.Write([]byte(q.name))
		remaining := item.Size()
		for remaining > 0 {
			if e := ctx.Err(); e != nil {
				return "", e
			}
			n := min(remaining, int64(len(buffer)))
			if _, e := io.ReadFull(reader, buffer[:n]); e != nil {
				return "", e
			}
			for i := int64(0); i < n; i += 4 {
				v := math.Float32frombits(binary.LittleEndian.Uint32(buffer[i:]))
				if !finiteStage(float64(v)) || q.second && v < 0 {
					return "", ErrStage
				}
			}
			hash.Write(buffer[:n])
			remaining -= n
		}
	}
	return hex.EncodeToString(hash.Sum(nil)), ctx.Err()
}
func (h *Stage) identity(ctx context.Context, c pipeline.StageContext) (pipeline.ReceiptIdentity, error) {
	x := c.Execution
	if err := h.Admit(ctx, x.Recipe, c.Stage, x.Placement); err != nil {
		return pipeline.ReceiptIdentity{}, err
	}
	// TargetSHA256 names the application's execution target, and the rest of
	// the pipeline treats it as an opaque digest carried into the receipt. It
	// was compared here with the student's digest, which no application sets:
	// the student is already bound by the recipe digest and by the protocol's
	// WeightsSHA256, both checked in Admit above, so the comparison only
	// refused every real submission while fixtures that set both alike passed.
	if !stageID(x.RunID) || !stageID(x.TenantID) || x.Generation == 0 || !validHash(x.TargetSHA256) || c.Parent != nil {
		return pipeline.ReceiptIdentity{}, ErrStage
	}
	return pipeline.ReceiptIdentity{RunID: x.RunID, TenantID: x.TenantID, RecipeSHA256: h.config.Protocol.RecipeSHA256, TargetSHA256: x.TargetSHA256, PlacementSHA256: h.config.Protocol.PlacementSHA256, Generation: x.Generation}, nil
}

type stageIntent struct {
	Version                int                      `json:"version"`
	Identity               pipeline.ReceiptIdentity `json:"identity"`
	ProtocolSHA256         string                   `json:"protocol_sha256"`
	Step                   uint64                   `json:"step"`
	ExampleID              string                   `json:"example_id"`
	PreviousManifestSHA256 string                   `json:"previous_manifest_sha256"`
	Data                   pipeline.ArtifactBinding `json:"data"`
}

func sameStageIdentity(a, b pipeline.ReceiptIdentity) bool {
	a.Generation, b.Generation = 0, 0
	return a == b
}
func (h *Stage) writeIntent(ctx context.Context, c pipeline.StageContext, step uint64, id, previousSHA string) error {
	identity, e := h.identity(ctx, c)
	if e != nil {
		return e
	}
	intent := stageIntent{Version: 1, Identity: identity, ProtocolSHA256: h.config.ProtocolSHA256, Step: step, ExampleID: id, PreviousManifestSHA256: previousSHA, Data: h.config.Protocol.Data}
	b, e := stageJSON(intent, h.config.Protocol.Limits.MaxManifestBytes)
	if e != nil {
		return e
	}
	path := filepath.Join(c.ArtifactDirectory, h.intentName(step))
	f, e := os.OpenFile(path, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
	if e != nil {
		return errors.Join(ErrAttempt, e)
	}
	_, writeErr := f.Write(b)
	e = errors.Join(writeErr, f.Sync(), f.Close())
	if e != nil {
		return e
	}
	return syncLocalDirectory(filepath.Dir(path))
}
func (h *Stage) readIntent(ctx context.Context, root *os.Root, c pipeline.StageContext, step uint64, id, previousSHA string) (pipeline.StageArtifact, error) {
	name := h.intentName(step)
	b, e := stageRead(ctx, root, name, h.config.Protocol.Limits.MaxManifestBytes, "")
	if e != nil {
		return pipeline.StageArtifact{}, e
	}
	var intent stageIntent
	d := json.NewDecoder(bytes.NewReader(b))
	d.DisallowUnknownFields()
	e = d.Decode(&intent)
	want, ie := h.identity(ctx, c)
	canonical, ce := stageJSON(intent, h.config.Protocol.Limits.MaxManifestBytes)
	if e != nil || ie != nil || ce != nil || !bytes.Equal(canonical, b) || intent.Version != 1 || intent.ProtocolSHA256 != h.config.ProtocolSHA256 || intent.Step != step || intent.ExampleID != id || intent.PreviousManifestSHA256 != previousSHA || intent.Data != h.config.Protocol.Data || !sameStageIdentity(intent.Identity, want) || intent.Identity.Generation == 0 || intent.Identity.Generation > want.Generation {
		return pipeline.StageArtifact{}, ErrStage
	}
	return pipeline.StageArtifact{Path: name, SHA256: fmtHash(b), Bytes: int64(len(b))}, nil
}
func (h *Stage) syncResult(ctx context.Context, c pipeline.StageContext, result pipeline.StageResult) error {
	root, e := stageRoot(c.ArtifactDirectory)
	if e != nil {
		return e
	}
	defer root.Close()
	dirs := map[string]bool{c.ArtifactDirectory: true, filepath.Dir(c.ArtifactDirectory): true, filepath.Join(c.ArtifactDirectory, h.seedName()): true, filepath.Join(c.ArtifactDirectory, h.outputName()): true}
	for _, a := range result.Artifacts {
		if e := ctx.Err(); e != nil {
			return e
		}
		f, e := stageOpen(root, a.Path)
		if e != nil {
			return e
		}
		if e := errors.Join(f.Sync(), f.Close()); e != nil {
			return e
		}
		dirs[filepath.Dir(filepath.Join(c.ArtifactDirectory, a.Path))] = true
	}
	ordered := make([]string, 0, len(dirs))
	for dir := range dirs {
		ordered = append(ordered, dir)
	}
	sort.Slice(ordered, func(i, j int) bool { return len(ordered[i]) > len(ordered[j]) })
	for _, dir := range ordered {
		if e := syncLocalDirectory(dir); e != nil {
			return e
		}
	}
	return ctx.Err()
}
func (h *Stage) seedArtifacts() []pipeline.StageArtifact {
	p := h.config.Protocol
	var out []pipeline.StageArtifact
	if !p.fresh() {
		for _, a := range []pipeline.StageArtifact{p.Initial.Manifest, p.Initial.Adapter, p.Initial.Optimizer} {
			a.Path = filepath.Join(h.seedName(), a.Path)
			out = append(out, a)
		}
	}
	a := p.Data.Artifact
	a.Path = filepath.Join(h.seedName(), "data.jsonl")
	out = append(out, a)
	for _, c := range p.cacheArtifacts() {
		c.Path = filepath.Join(h.cacheName(), c.Path)
		out = append(out, c)
	}
	return out
}

// stepManifest returns a result's own manifest, which follows the seed
// artifacts. Its position depends on whether the stage started fresh and on
// the caches it declares, so it is computed rather than written as a constant
// index.
func (h *Stage) stepManifestArtifact(r pipeline.StageResult) pipeline.StageArtifact {
	return r.Artifacts[len(h.seedArtifacts())]
}

// initialState is what step Initial.Step+1 continues from: the admitted
// checkpoint, or for a fresh start the pinned initializer, which has no
// manifest and whose adapter digest is the initializer's pin. Every later step
// must change the adapter away from it.
func (h *Stage) initialState(ctx context.Context, root *os.Root, rows []example) (stepManifest, error) {
	p := h.config.Protocol
	if p.fresh() {
		return stepManifest{BaseRevision: p.Local.BaseRevision, UpdatedAdapterSHA: p.Local.Initializer.ExpectedSHA256}, nil
	}
	m, _, e := h.checkpoint(ctx, root, h.seedName(), p.Initial.Step, rows)
	return m, e
}
func (h *Stage) result(step uint64, artifacts []pipeline.StageArtifact, intent pipeline.StageArtifact) pipeline.StageResult {
	return pipeline.StageResult{Steps: int64(step - h.config.Protocol.Initial.Step), Complete: step == h.config.Protocol.TargetStep, Artifacts: append(append(h.seedArtifacts(), artifacts...), intent)}
}
func (h *Stage) scan(ctx context.Context, c pipeline.StageContext, rows []example, target uint64) (uint64, []pipeline.StageResult, error) {
	root, e := stageRoot(c.ArtifactDirectory)
	if e != nil {
		return 0, nil, e
	}
	defer root.Close()
	p := h.config.Protocol
	prior, e := h.initialState(ctx, root, rows)
	if e != nil {
		return 0, nil, e
	}
	last := p.Initial.Step
	previousSHA := p.Initial.Manifest.SHA256
	var results []pipeline.StageResult
	gap := false
	for step := p.Initial.Step + 1; step <= target; step++ {
		info, e := root.Lstat(h.stepName(step))
		if errors.Is(e, os.ErrNotExist) {
			gap = true
			if _, ie := root.Lstat(h.intentName(step)); !errors.Is(ie, os.ErrNotExist) {
				return last, results, errors.Join(ErrAttempt, ie)
			}
			continue
		}
		if e != nil || !info.IsDir() || gap {
			return last, results, errors.Join(ErrStage, e)
		}
		intent, e := h.readIntent(ctx, root, c, step, rows[step-1].ID, previousSHA)
		if e != nil {
			return last, results, e
		}
		next, artifacts, e := h.checkpoint(ctx, root, h.stepName(step), step, rows)
		if e != nil || next.UpdatedAdapterSHA == prior.UpdatedAdapterSHA {
			return last, results, errors.Join(ErrStage, e)
		}
		results = append(results, h.result(step, artifacts, intent))
		prior = next
		previousSHA = artifacts[0].SHA256
		last = step
	}
	return last, results, nil
}

type stageBoundWriter struct {
	out       io.Writer
	remaining int64
}

func (w *stageBoundWriter) Write(b []byte) (int, error) {
	if int64(len(b)) > w.remaining {
		return 0, ErrStage
	}
	n, e := w.out.Write(b)
	w.remaining -= int64(n)
	return n, e
}
