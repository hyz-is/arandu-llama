//go:build libtorch && cgo

package local

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"os"
	"path/filepath"

	"github.com/tayi-ai/arandu-llama/checkpoint"
	"github.com/tayi-ai/arandu-llama/training/decoder"
	"github.com/tayi-ai/arandu-llama/training/torch"
)

// MeasureLosses reads the completion loss of each request without training.
// Configuration, rows and every named manifest and adapter file are verified
// before the base is loaded, once, onto MPS. Each checkpoint's adapter is then
// installed once through the digest-verified restore, or rebuilt from the
// recipe's pinned initializer for an empty Checkpoint, and every example that
// names it is scored with rotary tables sized to that example, exactly as the
// step that logs loss_before scores it. Nothing is written, no optimizer state
// is read, and checkpoint directories are only read.
func MeasureLosses(ctx context.Context, c LossConfig, requests []LossRequest) (results []LossResult, err error) {
	if ctx == nil {
		return nil, ErrLoss
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	c, err = c.snapshot()
	if err != nil {
		return nil, err
	}
	requests = append([]LossRequest(nil), requests...)
	rows, err := lossRows(ctx, c, requests)
	if err != nil {
		return nil, err
	}
	manifests, err := lossManifests(ctx, c, requests)
	if err != nil {
		return nil, err
	}
	read := func(name, expected string) ([]byte, error) {
		body, err := os.ReadFile(filepath.Join(c.BundleDir, name))
		if err != nil {
			return nil, err
		}
		if actual := sha256.Sum256(body); hex.EncodeToString(actual[:]) != expected {
			return nil, errors.Join(ErrLoss, errors.New(name+" digest differs"))
		}
		return body, nil
	}
	index, err := read("model.safetensors.index.json", c.Recipe.Identity.IndexSHA256)
	if err != nil {
		return nil, err
	}
	config, err := read("config.json", c.Recipe.Identity.ConfigSHA256)
	if err != nil {
		return nil, err
	}
	reference, err := read("initial-reference.json", c.Recipe.Identity.ReferenceSHA256)
	if err != nil {
		return nil, err
	}
	plan, err := decoder.PlanLocalMPSAssembly(index, config, reference, c.Recipe.Identity, c.Recipe.Assembly, c.Recipe.MaxMPSBytes)
	if err != nil {
		return nil, err
	}
	geometry := plan.Geometry()
	if c.Recipe.Rotary.Theta != geometry.RoPE.Theta || c.Recipe.Rotary.Dimension != int(float64(geometry.Dimension)*geometry.RoPE.Partial) {
		return nil, errors.Join(ErrLoss, errors.New("rotary admission differs from geometry"))
	}
	provider := files{root: c.ModelDir}
	if _, err := decoder.InspectAssemblySources(ctx, plan, provider); err != nil {
		return nil, err
	}
	initial, err := decoder.InitializeAdapter(ctx, c.Recipe.Initializer)
	if err != nil {
		return nil, err
	}
	loaded, err := decoder.LoadTextAssembly(ctx, plan, provider, initial)
	defer func() {
		// A release failure leaves the backend in doubt; no reading survives it.
		if err = errors.Join(err, loaded.Close(), initial.Close()); err != nil {
			results = nil
		}
	}()
	if err != nil {
		return nil, err
	}
	return measureLosses(ctx, loaded, c, rows, manifests, requests, torch.MPSDevice())
}

// measureLosses scores verified requests on a loaded model and leaves the last
// installed adapter in place. rows and manifests come from lossRows and
// lossManifests for the same requests.
func measureLosses(ctx context.Context, loaded *decoder.LoadedTextModel, c LossConfig, rows map[string]example, manifests map[string]stepManifest, requests []LossRequest, device torch.Device) ([]LossResult, error) {
	names := make([]string, len(loaded.Parameters))
	shapes := make([][]int64, len(loaded.Parameters))
	var payload, largest int64
	for i, parameter := range loaded.Parameters {
		info, err := parameter.Value.Info()
		if err != nil || len(info.Shape) != 2 {
			return nil, errors.Join(ErrLoss, err)
		}
		names[i], shapes[i] = parameter.Name, info.Shape
		payload += info.Elements * 4
		largest = max(largest, info.Elements*4)
	}
	// The file holds the registry's payload behind a header; the working
	// budget is what RestoreAdapter keeps live for a file of that size.
	limits := checkpoint.DefaultLimits()
	fileBytes := 8 + limits.MaxHeaderBytes + payload
	working := fileBytes + min(limits.MaxHeaderBytes, fileBytes-8) + 3*payload + 2*largest
	results := make([]LossResult, len(requests))
	for _, group := range lossOrder(requests) {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		name := requests[group[0]].Checkpoint
		var step uint64
		var digest string
		if name == "" {
			adapter, err := freshAdapter(ctx, c.Recipe, names, shapes)
			if err != nil {
				return nil, err
			}
			var flat []float32
			for _, values := range adapter {
				flat = append(flat, values...)
			}
			digest, err = loaded.ReplaceParameters(ctx, flat)
			if err != nil || digest != c.Recipe.Initializer.ExpectedSHA256 {
				return nil, errors.Join(ErrLoss, errors.New("initializer adapter identity differs"), err)
			}
		} else {
			manifest, ok := manifests[name]
			if !ok {
				return nil, ErrLoss
			}
			var err error
			digest, err = loaded.RestoreAdapter(ctx, decoder.AdapterCheckpoint{
				Path: filepath.Join(name, "adapter_model.safetensors"), FileSHA256: manifest.AdapterFileSHA,
				ParametersSHA256: manifest.UpdatedAdapterSHA, MaxBytes: fileBytes, MaxWorkingBytes: working, Limits: limits,
			})
			if err != nil || digest != manifest.UpdatedAdapterSHA {
				return nil, errors.Join(ErrLoss, errors.New("checkpoint adapter identity differs"), err)
			}
			step = manifest.Step
		}
		for _, index := range group {
			row, ok := rows[requests[index].ExampleID]
			if !ok {
				return nil, ErrLoss
			}
			loss, err := rowLoss(ctx, loaded, row, c.Recipe, device)
			if err != nil {
				return nil, err
			}
			results[index] = LossResult{Checkpoint: name, ExampleID: row.ID, Step: step, Loss: loss,
				CompletionTokens: len(row.InputIDs) - row.PromptTokens, AdapterSHA256: digest}
		}
	}
	return results, nil
}

// rowLoss scores one example with the tables and limits the step uses for it.
func rowLoss(ctx context.Context, loaded *decoder.LoadedTextModel, row example, recipe Recipe, device torch.Device) (float64, error) {
	tables, err := decoder.TextRotary(ctx, len(row.InputIDs), device, recipe.Rotary)
	if err != nil {
		return 0, err
	}
	for layer := range loaded.Model.Layers {
		if loaded.Model.Layers[layer].Weights.Full != nil {
			loaded.Model.Layers[layer].Cosine, loaded.Model.Layers[layer].Sine = tables.Cosine, tables.Sine
		}
	}
	tokens := int64(len(row.InputIDs))
	loss, lossErr := decoder.CompletionLoss(ctx, loaded.Model, row.InputIDs, row.PromptTokens,
		decoder.Limits{MaxTokens: tokens, LogitRows: tokens - int64(row.PromptTokens) + 1, MaxCheckpointBytes: recipe.MaxCheckpointBytes})
	for layer := range loaded.Model.Layers {
		if loaded.Model.Layers[layer].Weights.Full != nil {
			loaded.Model.Layers[layer].Cosine, loaded.Model.Layers[layer].Sine = nil, nil
		}
	}
	if err := errors.Join(lossErr, tables.Close()); err != nil {
		return 0, err
	}
	return loss, nil
}
