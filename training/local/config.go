// Package local executes bounded causal SFT updates with caller-owned recipes.
// It does not claim multi-teacher fusion or scientific qualification.
package local

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"math"
	"path/filepath"
	"strings"

	"github.com/tayi-ai/arandu-llama/training/decoder"
	"github.com/tayi-ai/arandu-llama/training/optim"
)

// ErrUnavailable means the local native backend cannot run on this host.
var ErrUnavailable = errors.New("local training: LibTorch backend unavailable")

// Recipe supplies every scientific identity, numerical setting and memory bound.
// The only supported method here is causal-sft-v1; fusion has a separate pipeline.
type Recipe struct {
	Method             string
	BaseRevision       string
	DataSHA256         string
	ExampleCount       int
	RequireLengthOrder bool
	Identity           decoder.AssemblyIdentity
	Assembly           decoder.AssemblyLimits
	MaxMPSBytes        int64
	Initializer        decoder.InitialAdapterSpec
	Rotary             decoder.RotarySpec
	Optimizer          optim.AdamWConfig
	LossScale          float64
	MaxCheckpointBytes int64
}

// Config fixes paths, work bounds and an explicit immutable recipe.
// AdmittedCheckpoints pins exact historical manifest bytes that predate recipe
// digests. Without this opt-in, an unbound historical checkpoint is rejected.
type Config struct {
	BundleDir, ModelDir, DataPath, CheckpointRoot, InitialCheckpoint string
	MaxTokens, MaxSteps                                              int
	Recipe                                                           Recipe
	AdmittedCheckpoints                                              map[uint64]string
	// FreshStart begins at the recipe's pinned initializer with zero AdamW
	// moments instead of loading InitialCheckpoint, which must then be empty.
	// It is a field rather than an empty path because an empty path has always
	// been refused here, and a missing path must not quietly become a new run.
	FreshStart bool
}

// Progress identifies the latest complete checkpoint of this delivery.
type Progress struct {
	Step                  uint64
	ExampleID, Checkpoint string
}

// Digest identifies the entire numerical recipe, excluding runtime paths.
func (r Recipe) Digest() string {
	b, err := json.Marshal(r)
	if err != nil {
		return ""
	}
	return fmtHash(b)
}
func fmtHash(b []byte) string { sum := sha256.Sum256(b); return hex.EncodeToString(sum[:]) }
func validHash(s string) bool {
	b, err := hex.DecodeString(s)
	return err == nil && len(b) == 32 && strings.ToLower(s) == s
}

func (c Config) validate() error {
	if !filepath.IsAbs(c.BundleDir) || !filepath.IsAbs(c.ModelDir) || !filepath.IsAbs(c.DataPath) || !filepath.IsAbs(c.CheckpointRoot) || c.MaxTokens < 2 || c.MaxTokens > 4096 || c.MaxSteps < 1 || c.MaxSteps > 20 {
		return errors.New("local training: absolute paths and bounded work required")
	}
	// A fresh start has no checkpoint to load and no history to admit; any
	// other start names the checkpoint it continues from.
	if c.FreshStart && (c.InitialCheckpoint != "" || len(c.AdmittedCheckpoints) > 0) || !c.FreshStart && !filepath.IsAbs(c.InitialCheckpoint) {
		return errors.New("local training: a fresh start loads no checkpoint; any other start names one")
	}
	r := c.Recipe
	if r.Method != "causal-sft-v1" || r.BaseRevision == "" || !validHash(r.DataSHA256) || r.ExampleCount < 1 || r.ExampleCount > 1<<24 || r.MaxMPSBytes < 1 || r.MaxCheckpointBytes < 1 || r.LossScale <= 0 || math.IsNaN(r.LossScale) || math.IsInf(r.LossScale, 0) || r.Digest() == "" || r.Initializer.ExpectedSHA256 != r.Identity.InitialAdapterSHA256 || r.Rotary.MaxTokens < c.MaxTokens {
		return errors.New("local training: explicit causal SFT recipe required")
	}
	for _, hash := range []string{r.Identity.IndexSHA256, r.Identity.ConfigSHA256, r.Identity.ReferenceSHA256, r.Identity.InitialAdapterSHA256, r.Rotary.ExpectedSHA256} {
		if !validHash(hash) {
			return errors.New("local training: recipe identity missing")
		}
	}
	if err := optim.ValidateAdamWConfig(r.Optimizer); err != nil {
		return err
	}
	for step, hash := range c.AdmittedCheckpoints {
		if step == 0 || !validHash(hash) {
			return errors.New("local training: historical checkpoint identity invalid")
		}
	}
	return nil
}

// snapshot owns all slices and maps before validation and execution. Callers
// must not mutate the input while handing it over; later changes are isolated.
func (c Config) snapshot() (Config, error) {
	body, err := json.Marshal(c)
	if err != nil {
		return Config{}, errors.New("local training: configuration cannot be snapshotted")
	}
	var owned Config
	if err := json.Unmarshal(body, &owned); err != nil {
		return Config{}, err
	}
	return owned, owned.validate()
}

// Snapshot returns an owned, validated configuration for queue admission.
// It performs CPU-only structural validation; it does not read artifacts,
// initialize a native backend, or qualify a model. Callers must not mutate
// the input during this call; subsequent changes cannot alter the snapshot.
func (c Config) Snapshot() (Config, error) {
	return c.snapshot()
}
