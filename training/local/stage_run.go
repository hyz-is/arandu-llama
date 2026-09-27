package local

import (
	"context"
	"errors"
	"fmt"
	"path/filepath"
	"reflect"

	"github.com/tayi-ai/arandu-llama/training/pipeline"
)

// Verify rechecks exact checkpoint state and new-step count without computation.
// It proves the admitted causal update artifacts, not heldout model quality.
func (h *Stage) Verify(ctx context.Context, c pipeline.StageContext, receipt pipeline.StageReceipt) error {
	identity, e := h.identity(ctx, c)
	if e != nil {
		return e
	}
	pin := receipt.SHA256
	receipt.SHA256 = ""
	if receipt.Version != 1 || pin != sessionDigest(receipt) || receipt.StageID != h.stage.ID || !sameStageIdentity(receipt.Identity, identity) || receipt.Identity.Generation == 0 || receipt.Identity.Generation > identity.Generation || receipt.Result.Steps < 1 || receipt.Result.Steps > int64(h.stage.MaxSteps) {
		return ErrStage
	}
	root, e := stageRoot(c.ArtifactDirectory)
	if e != nil {
		return e
	}
	rows, e := h.dataset(ctx, root)
	closeErr := root.Close()
	if e != nil || closeErr != nil {
		return errors.Join(e, closeErr)
	}
	target := h.config.Protocol.Initial.Step + uint64(receipt.Result.Steps)
	verified := c
	verified.Execution.Generation = receipt.Identity.Generation
	last, results, e := h.scan(ctx, verified, rows, target)
	if e != nil || last != target || len(results) == 0 || !reflect.DeepEqual(results[len(results)-1], receipt.Result) {
		return errors.Join(ErrStage, e)
	}
	return nil
}
func (h *Stage) acquire(ctx context.Context) error {
	if h == nil || ctx == nil || h.gate == nil || h.execute == nil {
		return ErrStage
	}
	select {
	case h.gate <- struct{}{}:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}
func (h *Stage) completed(ctx context.Context, c pipeline.StageContext, rows []example, commit func(context.Context, pipeline.StageResult) error) (uint64, []pipeline.StageResult, error) {
	if commit == nil {
		return 0, nil, ErrStage
	}
	if c.Previous != nil {
		if e := h.Verify(ctx, c, *c.Previous); e != nil {
			return 0, nil, e
		}
	}
	last, results, e := h.scan(ctx, c, rows, h.config.Protocol.TargetStep)
	if e != nil {
		return last, results, e
	}
	var published int64
	if c.Previous != nil {
		published = c.Previous.Result.Steps
	}
	if last < h.config.Protocol.Initial.Step+uint64(published) {
		return last, results, ErrStage
	}
	for _, result := range results {
		if result.Steps <= published {
			continue
		}
		if e := h.syncResult(ctx, c, result); e != nil {
			return last, results, e
		}
		if e := commit(ctx, result); e != nil {
			return last, results, e
		}
		published = result.Steps
	}
	return last, results, nil
}

// Reconcile publishes a durable checkpoint whose callback was interrupted. An
// intent without a complete checkpoint is uncertain and cannot be retried.
func (h *Stage) Reconcile(ctx context.Context, c pipeline.StageContext, commit func(context.Context, pipeline.StageResult) error) error {
	if e := h.acquire(ctx); e != nil {
		return e
	}
	defer func() { <-h.gate }()
	rows, e := h.prepare(ctx, c)
	if e != nil {
		return e
	}
	_, _, e = h.completed(ctx, c, rows, commit)
	return e
}

// Run continues the admitted interval, committing every synced checkpoint before
// starting the next example. Each bounded delivery closes its native model.
func (h *Stage) Run(ctx context.Context, c pipeline.StageContext, commit func(context.Context, pipeline.StageResult) error) error {
	if e := h.acquire(ctx); e != nil {
		return e
	}
	defer func() { <-h.gate }()
	rows, e := h.prepare(ctx, c)
	if e != nil {
		return e
	}
	last, results, e := h.completed(ctx, c, rows, commit)
	if e != nil {
		return fmt.Errorf("local SFT: reconciling continuation: %w", e)
	}
	// A fresh start has no manifest before step 1, so its chain begins empty.
	previousSHA := h.config.Protocol.Initial.Manifest.SHA256
	if len(results) > 0 {
		previousSHA = h.stepManifestArtifact(results[len(results)-1]).SHA256
	}
	root, e := stageRoot(c.ArtifactDirectory)
	if e != nil {
		return e
	}
	var prior stepManifest
	var readErr error
	if last > h.config.Protocol.Initial.Step {
		prior, _, readErr = h.checkpoint(ctx, root, h.stepName(last), last, rows)
	} else {
		prior, readErr = h.initialState(ctx, root, rows)
	}
	if e := errors.Join(readErr, root.Close()); e != nil {
		return e
	}
	previousAdapter := prior.UpdatedAdapterSHA
	for last < h.config.Protocol.TargetStep {
		if e := ctx.Err(); e != nil {
			return e
		}
		start := last
		cfg := h.localConfig(c.ArtifactDirectory, last)
		owned, e := cfg.Snapshot()
		if e != nil {
			return e
		}
		hooks := &stepHooks{
			storage: &stepStorage{checkpoint: h.config.Protocol.Limits.Checkpoint, tensorBytes: h.config.Protocol.Limits.MaxTensorFileBytes, manifestBytes: h.config.Protocol.Limits.MaxManifestBytes, metadataBytes: h.config.Protocol.Limits.MaxMetadataBytes},
			before: func(call context.Context, step uint64, id string) error {
				if step != last+1 || step > h.config.Protocol.TargetStep || id != rows[step-1].ID {
					return ErrStage
				}
				return h.writeIntent(call, c, step, id, previousSHA)
			},
			after: func(call context.Context, p Progress) error {
				if p.Step != last+1 || p.Step > h.config.Protocol.TargetStep || p.ExampleID != rows[p.Step-1].ID || p.Checkpoint != filepath.Join(c.ArtifactDirectory, h.stepName(p.Step)) {
					return ErrStage
				}
				root, e := stageRoot(c.ArtifactDirectory)
				if e != nil {
					return e
				}
				intent, e := h.readIntent(call, root, c, p.Step, p.ExampleID, previousSHA)
				if e != nil {
					root.Close()
					return e
				}
				manifest, artifacts, e := h.checkpoint(call, root, h.stepName(p.Step), p.Step, rows)
				closeErr := root.Close()
				if e != nil || closeErr != nil {
					return errors.Join(e, closeErr)
				}
				if manifest.UpdatedAdapterSHA == previousAdapter {
					return ErrStage
				}
				result := h.result(p.Step, artifacts, intent)
				if e := h.syncResult(call, c, result); e != nil {
					return e
				}
				if e := commit(call, result); e != nil {
					return e
				}
				previousSHA = artifacts[0].SHA256
				previousAdapter = manifest.UpdatedAdapterSHA
				last = p.Step
				return nil
			},
		}
		if e := h.execute(ctx, owned, hooks); e != nil {
			return e
		}
		if last != start+uint64(owned.MaxSteps) {
			return errors.New("local SFT: native delivery stopped before its admitted target")
		}
	}
	return ctx.Err()
}
