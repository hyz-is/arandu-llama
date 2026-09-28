//go:build libtorch && cgo

package local

import (
	"context"
	"errors"
	"fmt"
	"github.com/tayi-ai/arandu-llama/training/decoder"
	"github.com/tayi-ai/arandu-llama/training/torch"
	"path/filepath"
)

func runCurriculum(ctx context.Context, loaded *decoder.LoadedTextModel, data, previous, output string, maxTokens, maxSteps int, c Config) error {
	return runCurriculumWithHooks(ctx, loaded, data, previous, output, maxTokens, maxSteps, c, nil, torch.MPSDevice())
}

func runCurriculumWithHooks(ctx context.Context, loaded *decoder.LoadedTextModel, data, previous, output string, maxTokens, maxSteps int, c Config, hooks *stepHooks, device torch.Device) error {
	return runCurriculumWithSignals(ctx, loaded, data, previous, output, maxTokens, maxSteps, c, hooks, device, nil)
}

// runCurriculumWithSignals runs the admitted steps. signals are the recipe's
// distillation caches when the caller already read them, and are read here
// otherwise; a recipe without distillation has none.
func runCurriculumWithSignals(ctx context.Context, loaded *decoder.LoadedTextModel, data, previous, output string, maxTokens, maxSteps int, c Config, hooks *stepHooks, device torch.Device, signals *distillationSignals) error {
	rows, prior, next, err := curriculumPosition(data, previous, c)
	if err != nil {
		return err
	}
	if signals == nil {
		if signals, err = loadDistillation(ctx, c, rows); err != nil {
			return err
		}
	}
	if err := distillationStudent(loaded.Model, c.Recipe.Distillation); err != nil {
		return err
	}
	completed := 0
	for i := next; i < len(rows) && completed < maxSteps; i++ {
		if err := ctx.Err(); err != nil {
			return err
		}
		if len(rows[i].InputIDs) > maxTokens {
			break
		}
		row, err := selectedID(data, rows[i].ID)
		if err != nil {
			return err
		}
		// A row the caches do not match is refused before its intent exists,
		// so the refusal leaves no uncertain attempt behind.
		teachers, err := signals.teachersFor(i, row)
		if err != nil {
			return err
		}
		if hooks != nil {
			if err := hooks.before(ctx, prior.Step+1, row.ID); err != nil {
				return err
			}
		}
		tables, err := decoder.TextRotary(ctx, len(row.InputIDs), device, c.Recipe.Rotary)
		if err != nil {
			return err
		}
		for layer := range loaded.Model.Layers {
			if loaded.Model.Layers[layer].Weights.Full != nil {
				loaded.Model.Layers[layer].Cosine, loaded.Model.Layers[layer].Sine = tables.Cosine, tables.Sine
			}
		}
		target := filepath.Join(output, fmt.Sprintf("step-%03d", prior.Step+1))
		var storage *stepStorage
		if hooks != nil {
			storage = hooks.storage
		}
		stepErr := resumeNextWithStorage(ctx, loaded, row, previous, target, c.Recipe, storage, teachers)
		for layer := range loaded.Model.Layers {
			if loaded.Model.Layers[layer].Weights.Full != nil {
				loaded.Model.Layers[layer].Cosine, loaded.Model.Layers[layer].Sine = nil, nil
			}
		}
		closeErr := tables.Close()
		if err := errors.Join(stepErr, closeErr); err != nil {
			return err
		}
		previous = target
		prior.Step++
		completed++
		if hooks != nil {
			if err := hooks.after(ctx, Progress{Step: prior.Step, ExampleID: row.ID, Checkpoint: target}); err != nil {
				return err
			}
		}
	}
	if completed == 0 {
		return errors.New("no admitted curriculum step was run")
	}
	fmt.Printf("phase=bounded_curriculum_complete steps=%d last_step=%d max_tokens=%d\n", completed, prior.Step, maxTokens)
	return nil
}
