package local

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"path/filepath"
	"reflect"

	"github.com/tayi-ai/arandu-llama/checkpoint"
	"github.com/tayi-ai/arandu-llama/training/fusioncache"
	"github.com/tayi-ai/arandu-llama/training/pipeline"
)

// ErrStage rejects an unbound or incomplete causal SFT continuation.
var ErrStage = errors.New("local SFT: stage refused")

// ErrAttempt refuses to repeat an uncertain numerical update.
var ErrAttempt = errors.New("local SFT: attempt exists without a complete checkpoint")

// StageInitial pins all state required to continue AdamW, not only the adapter.
// AllowHistorical admits exactly these manifest bytes when RecipeSHA256 is absent.
type StageInitial struct {
	Step            uint64                 `json:"step"`
	Manifest        pipeline.StageArtifact `json:"manifest"`
	Adapter         pipeline.StageArtifact `json:"adapter"`
	Optimizer       pipeline.StageArtifact `json:"optimizer"`
	AllowHistorical bool                   `json:"allow_historical"`
}

// StageLimits reserves serialized input, checkpoint and working payloads. This
// is not a measured native peak; the exact placement still needs qualification.
type StageLimits struct {
	MaxProtocolBytes   int64             `json:"max_protocol_bytes"`
	MaxDataBytes       int64             `json:"max_data_bytes"`
	MaxMetadataBytes   int64             `json:"max_metadata_bytes"`
	MaxManifestBytes   int64             `json:"max_manifest_bytes"`
	MaxTensorFileBytes int64             `json:"max_tensor_file_bytes"`
	MaxTotalBytes      int64             `json:"max_total_bytes"`
	MaxDataTokens      int64             `json:"max_data_tokens"`
	MaxParameters      int64             `json:"max_parameters"`
	WorkingBytes       int64             `json:"working_bytes"`
	MaxSteps           int               `json:"max_steps"`
	Checkpoint         checkpoint.Limits `json:"checkpoint"`
}

// StageProtocol freezes a continuation window and its unchanged causal recipe.
// Data independently binds tokenized bytes to the recipe's training dataset.
// DeliverySteps limits each model residency; it does not reset AdamW moments.
type StageProtocol struct {
	Version             int                       `json:"version"`
	RecipeSHA256        string                    `json:"recipe_sha256"`
	PlacementSHA256     string                    `json:"placement_sha256"`
	QualificationSHA256 string                    `json:"qualification_sha256"`
	StageID             string                    `json:"stage_id"`
	Student             fusioncache.ModelIdentity `json:"student"`
	Local               Recipe                    `json:"local"`
	Data                pipeline.ArtifactBinding  `json:"data"`
	Initial             StageInitial              `json:"initial"`
	TargetStep          uint64                    `json:"target_step"`
	DeliverySteps       int                       `json:"delivery_steps"`
	Limits              StageLimits               `json:"limits"`
}

// Digest hashes the validated canonical protocol without discovering files.
func (p StageProtocol) Digest() (string, error) {
	if err := p.validate(); err != nil {
		return "", err
	}
	b, err := stageJSON(p, p.Limits.MaxProtocolBytes)
	if err != nil {
		return "", err
	}
	return fmtHash(b), nil
}

// StageConfig installs the concrete local MPS continuation backend. Only source
// directories are supplied; outputs always belong to StageContext.ArtifactDirectory.
// CacheDirectory holds every teacher cache a distillation protocol names, each
// as <sha256>.json; it is required with distillation and refused without it.
type StageConfig struct {
	Recipe           pipeline.Recipe
	Placement        pipeline.Placement
	Protocol         StageProtocol
	ProtocolSHA256   string
	BundleDirectory  string
	ModelDirectory   string
	DataDirectory    string
	InitialDirectory string
	CacheDirectory   string
}

// Stage implements causal SFT continuation, with teacher-forced logit
// distillation when the protocol's recipe declares it, never constrained
// fusion or quantized recovery. Declared caches are imported beside the data
// before any calculation and are named in every result as seed artifacts.
// DurableRuntime must own the capacity/process lock; direct users must provide
// the same exclusive ownership and protect source/output files from mutation.
type Stage struct {
	config  StageConfig
	stage   pipeline.Stage
	gate    chan struct{}
	execute stageDelivery
}
type stepHooks struct {
	before  func(context.Context, uint64, string) error
	after   func(context.Context, Progress) error
	storage *stepStorage
}
type stepStorage struct {
	checkpoint                                checkpoint.Limits
	tensorBytes, manifestBytes, metadataBytes int64
}
type stageDelivery func(context.Context, Config, *stepHooks) error

var _ pipeline.StageHandler = (*Stage)(nil)

// NewStage snapshots exact configuration without allocating native model state.
// A continuation names an admitted adapter and its moments. A fresh start is an
// empty StageInitial at step 0 and begins at the recipe's pinned initializer.
func NewStage(c StageConfig) (*Stage, error) { return newStage(c, runStageDelivery) }
func newStage(c StageConfig, execute stageDelivery) (*Stage, error) {
	if execute == nil {
		return nil, ErrStage
	}
	sha, err := c.Protocol.Digest()
	if err != nil || sha != c.ProtocolSHA256 {
		return nil, errors.Join(ErrStage, err)
	}
	sources := []string{c.BundleDirectory, c.ModelDirectory, c.DataDirectory}
	if c.Protocol.fresh() {
		// Nothing is imported for a fresh start, so no source is named for it.
		if c.InitialDirectory != "" {
			return nil, ErrStage
		}
	} else {
		sources = append(sources, c.InitialDirectory)
	}
	if c.Protocol.Local.Distillation != nil {
		sources = append(sources, c.CacheDirectory)
	} else if c.CacheDirectory != "" {
		return nil, ErrStage
	}
	for _, p := range sources {
		if !filepath.IsAbs(p) || filepath.Clean(p) != p {
			return nil, ErrStage
		}
	}
	h := &Stage{config: c, gate: make(chan struct{}, 1), execute: execute}
	for _, s := range c.Recipe.Stages {
		if s.ID == c.Protocol.StageID {
			h.stage = s
		}
	}
	if err := h.Admit(context.Background(), c.Recipe, h.stage, c.Placement); err != nil {
		return nil, err
	}
	b, err := stageJSON(c, c.Protocol.Limits.MaxProtocolBytes+4<<20)
	if err != nil {
		return nil, err
	}
	var snapshot StageConfig
	if err := json.Unmarshal(b, &snapshot); err != nil {
		return nil, err
	}
	h.config = snapshot
	for _, s := range h.config.Recipe.Stages {
		if s.ID == c.Protocol.StageID {
			h.stage = s
		}
	}
	return h, nil
}
func (p StageProtocol) validate() error {
	l := p.Limits
	if p.Version != 1 || !validHash(p.RecipeSHA256) || !validHash(p.PlacementSHA256) || !validHash(p.QualificationSHA256) || !stageID(p.StageID) ||
		p.TargetStep <= p.Initial.Step || p.TargetStep > uint64(p.Local.ExampleCount) || p.TargetStep-p.Initial.Step > uint64(l.MaxSteps) ||
		p.DeliverySteps < 1 || p.DeliverySteps > 20 || l.MaxSteps < 1 || l.MaxSteps > 1_000_000 || l.MaxProtocolBytes < 1 || l.MaxProtocolBytes > 64<<20 ||
		l.MaxDataBytes < 1 || l.MaxDataBytes > 256<<20 || l.MaxMetadataBytes < 1 || l.MaxMetadataBytes > 16<<20 || l.MaxManifestBytes < 1 || l.MaxManifestBytes > 1<<20 || l.MaxTensorFileBytes < 1 || l.MaxTensorFileBytes > 1<<40 ||
		l.MaxDataTokens < 1 || l.MaxDataTokens > 1<<30 || l.MaxParameters < 1 || l.MaxParameters > 1<<30 || l.WorkingBytes < 1 ||
		p.Data.Kind != pipeline.BindingDerivedInput || p.Data.ValidateBounds(l.MaxDataBytes) != nil || p.Data.Artifact.SHA256 != p.Local.DataSHA256 ||
		p.Local.BaseRevision != p.Student.Revision || !validHash(p.Student.WeightsSHA256) || !validHash(p.Student.TokenizerSHA256) || !validHash(p.Student.TemplateSHA256) || !validHash(p.Student.RuntimeSHA256) || p.Student.Vocabulary < 2 {
		return ErrStage
	}
	// Step 0 is a fresh start only when nothing else is declared: a manifest,
	// adapter, moments or historical admission beside it would be a
	// continuation that forgot its step, and is refused rather than guessed.
	if p.fresh() {
		if p.Initial != (StageInitial{}) {
			return ErrStage
		}
	} else if p.Initial.Manifest.Path != "manifest.json" || p.Initial.Adapter.Path != "adapter_model.safetensors" || p.Initial.Optimizer.Path != "optimizer_moments.safetensors" ||
		!stageArtifact(p.Initial.Manifest, l.MaxManifestBytes) || !stageArtifact(p.Initial.Adapter, l.MaxTensorFileBytes) || !stageArtifact(p.Initial.Optimizer, l.MaxTensorFileBytes) {
		return ErrStage
	}
	// Distillation caches are this stage's student's, taken over this stage's
	// training input; their bytes are reserved on disk and, four times over
	// while one is decoded and kept, in working memory. That reservation is
	// not a measured peak.
	caches := p.cacheArtifacts()
	var cacheBytes int64
	if d := p.Local.Distillation; d != nil {
		if d.Student != p.Student || d.DatasetSHA256 != p.Data.Input.SHA256 {
			return ErrStage
		}
		for _, a := range caches {
			if !stageArtifact(a, d.CacheLimits.MaxBytes) || a.Bytes > l.MaxTotalBytes-cacheBytes {
				return ErrStage
			}
			cacheBytes += a.Bytes
		}
	}
	ck := l.Checkpoint
	if ck.MaxHeaderBytes < 1 || ck.MaxHeaderBytes > l.MaxTensorFileBytes || ck.MaxTensors < 1 || ck.MaxTensors > 65536 || ck.MaxDimensions < 2 || ck.MaxDimensions > 32 || ck.MaxMetadataEntries < 1 || ck.MaxChunkBytes < 4 || ck.MaxChunkBytes > 4<<20 {
		return ErrStage
	}
	remaining := l.WorkingBytes
	working := []int64{4 * l.MaxDataBytes, 16 * l.MaxMetadataBytes, 16 * l.MaxDataTokens, 64 * l.MaxParameters, int64(ck.MaxChunkBytes), ck.MaxHeaderBytes}
	if cacheBytes > 0 {
		if cacheBytes > math.MaxInt64/4 {
			return ErrStage
		}
		working = append(working, 4*cacheBytes)
	}
	for _, n := range working {
		if n < 1 || n > remaining {
			return ErrStage
		}
		remaining -= n
	}
	// Reserve every source plus all declared durable step files, including intent.
	remaining = l.MaxTotalBytes
	for _, a := range append([]pipeline.StageArtifact{p.Data.Artifact, p.Initial.Manifest, p.Initial.Adapter, p.Initial.Optimizer}, caches...) {
		if a.Bytes > remaining {
			return ErrStage
		}
		remaining -= a.Bytes
	}
	per := 2*l.MaxTensorFileBytes + 2*l.MaxManifestBytes
	if per < 1 || int64(p.TargetStep-p.Initial.Step) > remaining/per {
		return ErrStage
	}
	if len(p.Local.Initializer.Projections) < 1 || len(p.Local.Initializer.Projections) > ck.MaxTensors/4 {
		return ErrStage
	}
	var parameters int64
	for _, q := range p.Local.Initializer.Projections {
		if q.Name == "" || len(q.Name) > 1024 || q.Input < 1 || q.Output < 1 || q.Rank < 1 || q.Input > l.MaxParameters || q.Output > l.MaxParameters || q.Rank > l.MaxParameters/(q.Input+q.Output) {
			return ErrStage
		}
		n := q.Rank * (q.Input + q.Output)
		if n > l.MaxParameters-parameters {
			return ErrStage
		}
		parameters += n
	}
	return nil
}

// Admit verifies phase, data role, student, recipe and resource reservations.
func (h *Stage) Admit(ctx context.Context, r pipeline.Recipe, s pipeline.Stage, p pipeline.Placement) error {
	if h == nil || ctx == nil {
		return ErrStage
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	c := h.config
	x := c.Protocol
	sha, err := r.Digest()
	if err != nil || sha != x.RecipeSHA256 || sessionDigest(p) != x.PlacementSHA256 || p.QualificationSHA256 != x.QualificationSHA256 ||
		!reflect.DeepEqual(r, c.Recipe) || !reflect.DeepEqual(s, h.stage) || !reflect.DeepEqual(p, c.Placement) || s.Phase != pipeline.PhaseSFT || s.Format != "" || s.ParentStage != "" ||
		s.MaxSteps != int(x.TargetStep-x.Initial.Step) || s.MaxTokens < 2 || s.MaxTokens > 4096 || s.MaxTokens > p.MaxTokens || x.Student.WeightsSHA256 != r.Student.SHA256 || x.Data.Input != r.Training ||
		x.Data.Validate(r, s, x.Limits.MaxDataBytes) != nil {
		return ErrStage
	}
	remaining := p.MemoryBytes
	for _, n := range []int64{x.Local.MaxMPSBytes, x.Local.Assembly.TensorCopyBytes, x.Local.MaxCheckpointBytes, x.Limits.WorkingBytes, x.Limits.MaxProtocolBytes} {
		if n < 1 || n > remaining {
			return ErrStage
		}
		remaining -= n
	}
	_, err = h.localConfig("/admission", x.Initial.Step).Snapshot()
	return errors.Join(err, ctx.Err())
}
func (h *Stage) localConfig(root string, step uint64) Config {
	p := h.config.Protocol
	c := Config{BundleDir: h.config.BundleDirectory, ModelDir: h.config.ModelDirectory, DataPath: filepath.Join(root, h.seedName(), "data.jsonl"), CheckpointRoot: filepath.Join(root, h.outputName()), InitialCheckpoint: filepath.Join(root, h.seedName()), MaxTokens: h.stage.MaxTokens, MaxSteps: min(p.DeliverySteps, int(p.TargetStep-step)), Recipe: p.Local}
	if step > p.Initial.Step {
		c.InitialCheckpoint = filepath.Join(c.CheckpointRoot, fmt.Sprintf("step-%03d", step))
	} else if p.fresh() {
		// Nothing has been written yet: the delivery starts at the initializer.
		c.InitialCheckpoint, c.FreshStart = "", true
	}
	if p.Initial.AllowHistorical {
		c.AdmittedCheckpoints = map[uint64]string{p.Initial.Step: p.Initial.Manifest.SHA256}
	}
	if p.Local.Distillation != nil {
		c.CacheDir = filepath.Join(root, h.cacheName())
	}
	return c
}

// fresh reports a start at the recipe's initializer, before any checkpoint.
func (p StageProtocol) fresh() bool { return p.Initial.Step == 0 }

// cacheArtifacts are the distillation caches the recipe declares, by the name
// each has in the cache directory: its SHA-256, as fusioncache stores it.
func (p StageProtocol) cacheArtifacts() []pipeline.StageArtifact {
	if p.Local.Distillation == nil {
		return nil
	}
	caches := make([]pipeline.StageArtifact, 0, len(p.Local.Distillation.Teachers))
	for _, t := range p.Local.Distillation.Teachers {
		caches = append(caches, pipeline.StageArtifact{Path: t.CacheSHA256 + ".json", SHA256: t.CacheSHA256, Bytes: t.CacheBytes})
	}
	return caches
}

func (h *Stage) seedName() string   { return "sft-" + h.stage.ID + "-initial" }
func (h *Stage) cacheName() string  { return filepath.Join(h.seedName(), "caches") }
func (h *Stage) outputName() string { return "sft-" + h.stage.ID }
func (h *Stage) stepName(n uint64) string {
	return filepath.Join(h.outputName(), fmt.Sprintf("step-%03d", n))
}
func (h *Stage) intentName(n uint64) string {
	return filepath.Join(h.outputName(), fmt.Sprintf("step-%03d-intent.json", n))
}
func sessionDigest(v any) string {
	b, err := json.Marshal(v)
	if err != nil {
		return ""
	}
	return fmtHash(b)
}
func stageID(s string) bool {
	if len(s) < 1 || len(s) > 128 || s == "." || s == ".." {
		return false
	}
	for _, r := range s {
		if !(r >= 'a' && r <= 'z' || r >= 'A' && r <= 'Z' || r >= '0' && r <= '9' || r == '-' || r == '_' || r == '.') {
			return false
		}
	}
	return true
}
func stageArtifact(a pipeline.StageArtifact, max int64) bool {
	return filepath.IsLocal(a.Path) && a.Path != "." && filepath.Clean(a.Path) == a.Path && len(a.Path) <= 1024 && validHash(a.SHA256) && a.Bytes > 0 && a.Bytes <= max
}
func finiteStage(x float64) bool { return !math.IsNaN(x) && !math.IsInf(x, 0) }

type stageBuffer struct {
	bytes.Buffer
	max int64
}

func (b *stageBuffer) Write(v []byte) (int, error) {
	if int64(len(v)) > b.max-int64(b.Len()) {
		return 0, ErrStage
	}
	return b.Buffer.Write(v)
}
func stageJSON(v any, max int64) ([]byte, error) {
	b := stageBuffer{max: max}
	err := json.NewEncoder(&b).Encode(v)
	return b.Bytes(), err
}
