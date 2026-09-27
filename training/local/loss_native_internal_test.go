//go:build libtorch && cgo

package local

import (
	"context"
	"errors"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/tayi-ai/arandu-llama/training/decoder"
	"github.com/tayi-ai/arandu-llama/training/pipeline"
	"github.com/tayi-ai/arandu-llama/training/torch"
)

// lossTrajectory runs the fresh three-step stage and returns the readout
// configuration over what it wrote, with the step directories in order.
func lossTrajectory(t *testing.T) (LossConfig, []string, []example, decoder.InitialAdapterSpec, StageConfig) {
	t.Helper()
	c, stage, rows, spec := sftFreshFixture(t)
	calls, closed := 0, 0
	h, e := newStage(c, sftExecutor(t, &calls, &closed))
	if e != nil {
		t.Fatal(e)
	}
	var results []pipeline.StageResult
	if e := h.Run(context.Background(), stage, sftCommit(t, h, &stage, &results)); e != nil || len(results) != 3 {
		t.Fatal("the fresh trajectory did not complete", len(results), e)
	}
	var steps []string
	for step := uint64(1); step <= 3; step++ {
		steps = append(steps, filepath.Join(stage.ArtifactDirectory, h.stepName(step)))
	}
	cfg := LossConfig{BundleDir: c.BundleDirectory, ModelDir: c.ModelDirectory,
		DataPath: filepath.Join(c.DataDirectory, c.Protocol.Data.Artifact.Path), DataSHA256: c.Protocol.Data.Artifact.SHA256,
		CheckpointRoot: filepath.Join(stage.ArtifactDirectory, h.outputName()), Recipe: c.Protocol.Local}
	owned, e := cfg.snapshot()
	if e != nil {
		t.Fatal(e)
	}
	return owned, steps, rows, spec, c
}

func readLosses(t *testing.T, c LossConfig, requests []LossRequest) ([]LossResult, error) {
	t.Helper()
	ctx := context.Background()
	rows, e := lossRows(ctx, c, requests)
	if e != nil {
		t.Fatal(e)
	}
	manifests, e := lossManifests(ctx, c, requests)
	if e != nil {
		t.Fatal(e)
	}
	model, close := sftCPUModel(t)
	defer close()
	return measureLosses(ctx, model, c, rows, manifests, requests, torch.CPUDevice())
}

// Checkpoint t is the adapter step t+1 starts from, and the initializer the one
// step 1 starts from, so each reading must be the loss_before that step logged.
// Both come from the same forward on CPU, so they are held bitwise; the float64
// survives the manifest's JSON because encoding/json writes the shortest
// round-tripping form.
func TestNativeLossReadoutReproducesEveryLoggedLossBefore(t *testing.T) {
	c, steps, rows, spec, stage := lossTrajectory(t)
	independent, _, _ := freshTrajectory(t, stage, rows, spec, 3)
	var logged []stepManifest
	for _, step := range steps {
		logged = append(logged, readLossManifest(t, step))
	}
	before := treeDigest(t, filepath.Dir(c.CheckpointRoot))
	// Interleaved and repeated, so each adapter is installed over another one
	// and results must still come back in request order.
	requests := []LossRequest{{steps[1], rows[2].ID}, {"", rows[0].ID}, {steps[0], rows[1].ID}, {steps[1], rows[2].ID}, {"", rows[0].ID}, {steps[2], rows[3].ID}}
	results, e := readLosses(t, c, requests)
	if e != nil {
		t.Fatal(e)
	}
	want := []struct {
		step    uint64
		loss    float64
		adapter string
	}{
		{2, logged[2].LossBefore, logged[1].UpdatedAdapterSHA},
		{0, logged[0].LossBefore, spec.ExpectedSHA256},
		{1, logged[1].LossBefore, logged[0].UpdatedAdapterSHA},
		{2, logged[2].LossBefore, logged[1].UpdatedAdapterSHA},
		{0, logged[0].LossBefore, spec.ExpectedSHA256},
		{3, math.NaN(), logged[2].UpdatedAdapterSHA},
	}
	for i, r := range results {
		w := want[i]
		if r.Checkpoint != requests[i].Checkpoint || r.ExampleID != requests[i].ExampleID || r.Step != w.step || r.AdapterSHA256 != w.adapter || r.CompletionTokens != 1 {
			t.Fatal("reading identity differs", i, r)
		}
		if w.step < 3 && math.Float64bits(r.Loss) != math.Float64bits(w.loss) {
			t.Fatalf("reading %d at step %d: %.17g != logged loss_before %.17g", i, w.step, r.Loss, w.loss)
		}
		if w.step < 3 && math.Float64bits(r.Loss) != math.Float64bits(independent[w.step]) {
			t.Fatalf("reading %d at step %d: %.17g != independent trajectory %.17g", i, w.step, r.Loss, independent[w.step])
		}
	}
	// Past the logged interval there is nothing to compare with; the reading
	// must still be a finite loss that moved with the adapter.
	if last := results[5].Loss; math.IsNaN(last) || math.IsInf(last, 0) || last == results[2].Loss {
		t.Fatal("the final checkpoint was not read", last)
	}
	if treeDigest(t, filepath.Dir(c.CheckpointRoot)) != before {
		t.Fatal("the readout changed the stage's artifacts")
	}
}

// A checkpoint of another base is refused before anything is loaded. The
// restore checks both digests itself, so an adapter swapped after its manifest
// was verified, or bytes other than the parameters the manifest names, are
// refused at install time rather than measured.
func TestNativeLossReadoutRefusesAnotherBaseAndATamperedAdapter(t *testing.T) {
	c, steps, rows, _, _ := lossTrajectory(t)
	ctx := context.Background()
	requests := []LossRequest{{steps[1], rows[2].ID}}
	rowsByID, e := lossRows(ctx, c, requests)
	if e != nil {
		t.Fatal(e)
	}
	verified, e := lossManifests(ctx, c, requests)
	if e != nil {
		t.Fatal(e)
	}
	measure := func(manifests map[string]stepManifest) error {
		model, close := sftCPUModel(t)
		defer close()
		_, e := measureLosses(ctx, model, c, rowsByID, manifests, requests, torch.CPUDevice())
		return e
	}
	named := verified[steps[1]]
	named.UpdatedAdapterSHA = fmtHash([]byte("other parameters"))
	if e := measure(map[string]stepManifest{steps[1]: named}); !errors.Is(e, ErrLoss) || !errors.Is(e, decoder.ErrAdapterCheckpoint) {
		t.Fatal("parameters other than the manifest's were measured", e)
	}
	path := filepath.Join(steps[1], "adapter_model.safetensors")
	body, e := os.ReadFile(path)
	if e != nil {
		t.Fatal(e)
	}
	body[len(body)-1] ^= 1
	if e := os.WriteFile(path, body, 0600); e != nil {
		t.Fatal(e)
	}
	if _, e := lossManifests(ctx, c, requests); !errors.Is(e, ErrLoss) {
		t.Fatal("a tampered adapter was admitted", e)
	}
	if e := measure(verified); !errors.Is(e, ErrLoss) || !errors.Is(e, decoder.ErrAdapterCheckpoint) {
		t.Fatal("an adapter swapped after verification was measured", e)
	}
	m := readLossManifest(t, steps[0])
	m.BaseRevision = "another-revision"
	writeLossManifest(t, steps[0], m)
	if _, e := lossManifests(ctx, c, []LossRequest{{steps[0], rows[1].ID}}); !errors.Is(e, ErrLoss) {
		t.Fatal("a checkpoint of another base was admitted", e)
	}
}
