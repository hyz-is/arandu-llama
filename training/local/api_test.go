package local

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tayi-ai/arandu-llama/training/decoder"
	"github.com/tayi-ai/arandu-llama/training/optim"
)

func checkpointFixture(t *testing.T) Config {
	t.Helper()
	root := t.TempDir()
	data := []byte("{\"id\":\"first\",\"input_ids\":[1,2],\"labels\":[-100,2],\"prompt_tokens\":1}\n{\"id\":\"second\",\"input_ids\":[1,3],\"labels\":[-100,3],\"prompt_tokens\":1}\n")
	path := filepath.Join(root, "data.jsonl")
	if err := os.WriteFile(path, data, 0600); err != nil {
		t.Fatal(err)
	}
	pin := strings.Repeat("1", 64)
	c := Config{BundleDir: root, ModelDir: root, DataPath: path, CheckpointRoot: root, InitialCheckpoint: filepath.Join(root, "step-001"), MaxTokens: 2, MaxSteps: 1, Recipe: Recipe{Method: "causal-sft-v1", BaseRevision: "synthetic-revision", DataSHA256: fmtHash(data), ExampleCount: 2, Identity: decoder.AssemblyIdentity{IndexSHA256: pin, ConfigSHA256: pin, ReferenceSHA256: pin, InitialAdapterSHA256: pin}, Initializer: decoder.InitialAdapterSpec{ExpectedSHA256: pin}, Rotary: decoder.RotarySpec{ExpectedSHA256: pin, MaxTokens: 2}, MaxMPSBytes: 1024, MaxCheckpointBytes: 1024, LossScale: 1, Optimizer: optim.AdamWConfig{LearningRate: .01, Beta1: .9, Beta2: .99, Epsilon: 1e-8, MaxGradientNorm: 1}}}
	writeCheckpoint(t, c, 1, "first")
	return c
}
func writeCheckpoint(t *testing.T, c Config, step uint64, id string) {
	t.Helper()
	directory := filepath.Join(c.CheckpointRoot, "step-00"+string(rune('0'+step)))
	if err := os.Mkdir(directory, 0700); err != nil {
		t.Fatal(err)
	}
	bytes := []byte("synthetic payload")
	for _, name := range []string{"adapter_model.safetensors", "optimizer_moments.safetensors"} {
		if err := os.WriteFile(filepath.Join(directory, name), bytes, 0600); err != nil {
			t.Fatal(err)
		}
	}
	m := stepManifest{Step: step, ExampleID: id, BaseRevision: c.Recipe.BaseRevision, RecipeSHA256: c.Recipe.Digest(), AdapterFileSHA: fmtHash(bytes), OptimizerFileSHA: fmtHash(bytes)}
	b, _ := json.Marshal(m)
	if err := os.WriteFile(filepath.Join(directory, "manifest.json"), b, 0600); err != nil {
		t.Fatal(err)
	}
}

func TestCheckpointReconciliationRequiresExactRecipeAndPayload(t *testing.T) {
	c := checkpointFixture(t)
	got, err := LatestCheckpoint(c)
	if err != nil || got.Step != 1 {
		t.Fatalf("initial checkpoint: %+v %v", got, err)
	}
	writeCheckpoint(t, c, 2, "second")
	got, err = LatestCheckpoint(c)
	if err != nil || got.Step != 2 {
		t.Fatalf("resume: %+v %v", got, err)
	}
	c.Recipe.Optimizer.LearningRate *= 2
	if _, err := LatestCheckpoint(c); err == nil {
		t.Fatal("changed recipe accepted")
	}
	c.Recipe.Optimizer.LearningRate /= 2
	if err := os.WriteFile(filepath.Join(c.CheckpointRoot, "step-002", "adapter_model.safetensors"), []byte("changed"), 0600); err != nil {
		t.Fatal(err)
	}
	if _, err := LatestCheckpoint(c); err == nil {
		t.Fatal("corrupt checkpoint accepted")
	}
}

func TestHistoricalCheckpointRequiresExplicitManifestAdmission(t *testing.T) {
	c := checkpointFixture(t)
	p := filepath.Join(c.InitialCheckpoint, "manifest.json")
	body, _ := os.ReadFile(p)
	var m stepManifest
	json.Unmarshal(body, &m)
	m.RecipeSHA256 = ""
	body, _ = json.Marshal(m)
	os.WriteFile(p, body, 0600)
	if _, err := LatestCheckpoint(c); err == nil {
		t.Fatal("unbound historical checkpoint accepted")
	}
	c.AdmittedCheckpoints = map[uint64]string{1: fmtHash(body)}
	if _, err := LatestCheckpoint(c); err != nil {
		t.Fatal(err)
	}
}

func TestConfigSnapshotOwnsNestedSlicesAndAdmissions(t *testing.T) {
	c := checkpointFixture(t)
	c.AdmittedCheckpoints = map[uint64]string{1: strings.Repeat("2", 64)}
	c.Recipe.Assembly.PersistentBytes = []int64{1024}
	c.Recipe.Assembly.DeviceByLayer = []int{0}
	c.Recipe.Initializer.Projections = []decoder.InitialProjection{{Name: "synthetic", Input: 2, Output: 2, Rank: 1}}
	owned, err := c.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	c.AdmittedCheckpoints[1] = ""
	c.Recipe.Assembly.PersistentBytes[0] = 0
	c.Recipe.Assembly.DeviceByLayer[0] = 99
	c.Recipe.Initializer.Projections[0].Name = "changed"
	if owned.AdmittedCheckpoints[1] == "" || owned.Recipe.Assembly.PersistentBytes[0] != 1024 || owned.Recipe.Assembly.DeviceByLayer[0] != 0 || owned.Recipe.Initializer.Projections[0].Name != "synthetic" {
		t.Fatal("configuration snapshot aliases caller state")
	}
}

func TestPublicConfigSnapshotAppliesExecutionValidation(t *testing.T) {
	base := checkpointFixture(t)
	cases := []struct {
		name   string
		mutate func(*Config)
	}{
		{"relative path", func(c *Config) { c.ModelDir = "relative" }},
		{"token bound", func(c *Config) { c.MaxTokens = 4097 }},
		{"step bound", func(c *Config) { c.MaxSteps = 21 }},
		{"method", func(c *Config) { c.Recipe.Method = "unsupported" }},
		{"identity", func(c *Config) { c.Recipe.Identity.ConfigSHA256 = "invalid" }},
		{"optimizer", func(c *Config) { c.Recipe.Optimizer.LearningRate = -1 }},
		{"historical admission", func(c *Config) { c.AdmittedCheckpoints = map[uint64]string{0: strings.Repeat("2", 64)} }},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			c := base
			tc.mutate(&c)
			_, publicErr := c.Snapshot()
			_, executionErr := c.snapshot()
			if publicErr == nil || executionErr == nil || publicErr.Error() != executionErr.Error() {
				t.Fatalf("validation differs: admission=%v execution=%v", publicErr, executionErr)
			}
		})
	}
}

// A declared coverage is validated and bound into the recipe digest, so a
// checkpoint written under one coverage is refused under another; an absent
// declaration keeps the q/v digest.
func TestRecipeCoverageIsValidatedAndBound(t *testing.T) {
	c := checkpointFixture(t)
	legacy := c.Recipe.Digest()
	c.Recipe.Assembly.AdapterCoverage = decoder.FullAdapterCoverage()
	if _, err := c.Snapshot(); err != nil {
		t.Fatal(err)
	}
	if c.Recipe.Digest() == legacy {
		t.Fatal("the declared coverage left the recipe digest unchanged")
	}
	if _, err := LatestCheckpoint(c); err == nil {
		t.Fatal("a q/v checkpoint was admitted under the full declaration")
	}
	c.Recipe.Assembly.AdapterCoverage = &decoder.AdapterCoverage{Version: 1, MLP: []string{"down_proj", "gate_proj"}}
	if _, err := c.Snapshot(); err == nil {
		t.Fatal("an out-of-order declaration was admitted")
	}
	c.Recipe.Assembly.AdapterCoverage = nil
	if c.Recipe.Digest() != legacy {
		t.Fatal("removing the declaration did not restore the q/v digest")
	}
}
