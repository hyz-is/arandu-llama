//go:build libtorch && cgo

package local

import (
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
	"time"

	"github.com/tayi-ai/arandu-llama/checkpoint"
	"github.com/tayi-ai/arandu-llama/training/decoder"
	"github.com/tayi-ai/arandu-llama/training/optim"
	"github.com/tayi-ai/arandu-llama/training/torch"
)

func readFrozenTensorFile(ctx context.Context, path, expectedSHA string, names []string, shapes [][]int64) ([][]float32, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	stat, err := file.Stat()
	if err != nil || !stat.Mode().IsRegular() {
		return nil, errors.New("checkpoint tensor file is not regular")
	}
	h := sha256.New()
	if _, err := io.Copy(h, file); err != nil || hex.EncodeToString(h.Sum(nil)) != expectedSHA {
		return nil, errors.New("checkpoint tensor file hash differs")
	}
	index, err := checkpoint.OpenSafetensors(file, stat.Size(), checkpoint.DefaultLimits())
	if err != nil || len(index.Tensors()) != len(names) {
		return nil, errors.New("checkpoint tensor index differs")
	}
	out := make([][]float32, len(names))
	for i, name := range names {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		item, ok := index.Tensor(name)
		if !ok || item.DType != "F32" || len(item.Shape) != len(shapes[i]) {
			return nil, fmt.Errorf("checkpoint tensor geometry differs: %s", name)
		}
		count := int64(1)
		for j, dimension := range shapes[i] {
			if item.Shape[j] != uint64(dimension) || dimension <= 0 {
				return nil, fmt.Errorf("checkpoint tensor shape differs: %s", name)
			}
			count *= dimension
		}
		if item.Size() != count*4 {
			return nil, fmt.Errorf("checkpoint tensor bytes differ: %s", name)
		}
		reader, err := index.TensorReader(name)
		if err != nil {
			return nil, err
		}
		data, err := io.ReadAll(reader)
		if err != nil || int64(len(data)) != count*4 {
			return nil, errors.New("checkpoint tensor data incomplete")
		}
		values := make([]float32, count)
		for j := range values {
			values[j] = math.Float32frombits(binary.LittleEndian.Uint32(data[j*4:]))
			if math.IsNaN(float64(values[j])) || math.IsInf(float64(values[j]), 0) {
				return nil, errors.New("nonfinite checkpoint tensor")
			}
		}
		out[i] = values
	}
	return out, nil
}

// freshAdapter materialises the recipe's pinned initializer in the model's
// parameter order. The values come from the recipe, never from whatever the
// model already holds; the caller proves the result through the digest that
// ReplaceParameters returns, which must equal the initializer's pin.
func freshAdapter(ctx context.Context, recipe Recipe, names []string, shapes [][]int64) (_ [][]float32, err error) {
	initial, err := decoder.InitializeAdapter(ctx, recipe.Initializer)
	if err != nil {
		return nil, err
	}
	defer func() { err = errors.Join(err, initial.Close()) }()
	values := make(map[string][]float32, len(initial.Parameters))
	for _, parameter := range initial.Parameters {
		v, err := parameter.Value.Float32Values()
		if err != nil {
			return nil, err
		}
		values[parameter.Name] = v
	}
	if len(values) != len(names) {
		return nil, errors.New("fresh adapter layout differs from the model")
	}
	adapter := make([][]float32, len(names))
	for i, name := range names {
		v, ok := values[name]
		if !ok || int64(len(v)) != shapes[i][0]*shapes[i][1] {
			return nil, errors.New("fresh adapter layout differs from the model")
		}
		adapter[i] = v
	}
	return adapter, nil
}

func resumeNext(ctx context.Context, loaded *decoder.LoadedTextModel, row example, previous, output string, recipe Recipe) error {
	return resumeNextWithStorage(ctx, loaded, row, previous, output, recipe, nil, nil)
}

// resumeNextWithStorage takes teachers, the row's distillation signals, which
// a recipe with distillation requires and any other recipe refuses.
func resumeNextWithStorage(ctx context.Context, loaded *decoder.LoadedTextModel, row example, previous, output string, recipe Recipe, storage *stepStorage, teachers []decoder.FusionTeacher) error {
	started := time.Now()
	memory := func(phase string) {
		stats, err := torch.ReadMPSMemory()
		if err == nil {
			fmt.Printf("phase=mps_memory stage=%s current=%d driver=%d recommended=%d\n", phase, stats.CurrentAllocatedBytes, stats.DriverAllocatedBytes, stats.RecommendedMaxBytes)
		}
	}
	names := make([]string, len(loaded.Parameters))
	shapes := make([][]int64, len(loaded.Parameters))
	for i, parameter := range loaded.Parameters {
		info, err := parameter.Value.Info()
		if err != nil || len(info.Shape) != 2 {
			return errors.New("model parameter metadata differs")
		}
		names[i], shapes[i] = parameter.Name, info.Shape
	}
	momentNames, momentShapes := make([]string, 0, 2*len(names)), make([][]int64, 0, 2*len(names))
	for i, name := range names {
		momentNames = append(momentNames, "m."+name, "v."+name)
		momentShapes = append(momentShapes, shapes[i], shapes[i])
	}
	var body []byte
	var prior stepManifest
	var adapter [][]float32
	var state optim.AdamWState
	if previous == "" {
		// A fresh start is step 0 of the same AdamW recipe: the pinned
		// initializer's values, zero moments and a zero step count, so the
		// first update takes t=1 bias correction exactly as a new optimizer
		// would. It must write step 1; there is nothing earlier to verify.
		if output == "" {
			return errors.New("a fresh start must write its first checkpoint")
		}
		var err error
		if adapter, err = freshAdapter(ctx, recipe, names, shapes); err != nil {
			return err
		}
		prior = stepManifest{BaseRevision: recipe.BaseRevision, UpdatedAdapterSHA: recipe.Initializer.ExpectedSHA256}
		for i := range names {
			state.Parameters = append(state.Parameters, adapter[i]...)
		}
		state.First = make([]float32, len(state.Parameters))
		state.Second = make([]float32, len(state.Parameters))
	} else {
		var err error
		if body, err = os.ReadFile(filepath.Join(previous, "manifest.json")); err != nil {
			return err
		}
		if err := json.Unmarshal(body, &prior); err != nil || prior.Step == 0 || prior.BaseRevision != recipe.BaseRevision || prior.AdapterFileSHA == "" || prior.OptimizerFileSHA == "" || output == "" && prior.ExampleID != row.ID || output != "" && prior.ExampleID == row.ID {
			return errors.New("previous checkpoint identity differs")
		}
		if adapter, err = readFrozenTensorFile(ctx, filepath.Join(previous, "adapter_model.safetensors"), prior.AdapterFileSHA, names, shapes); err != nil {
			return err
		}
		moments, err := readFrozenTensorFile(ctx, filepath.Join(previous, "optimizer_moments.safetensors"), prior.OptimizerFileSHA, momentNames, momentShapes)
		if err != nil {
			return err
		}
		state.Step = prior.Step
		for i := range names {
			state.Parameters = append(state.Parameters, adapter[i]...)
			state.First = append(state.First, moments[2*i]...)
			state.Second = append(state.Second, moments[2*i+1]...)
		}
	}
	if err := optim.ValidateAdamWState(state); err != nil {
		return err
	}
	// For a fresh start this is the check that the model's adapter layout is
	// the one the initializer describes: a reordered or reshaped registry
	// hashes differently from the pin even when every value is right.
	digest, err := loaded.ReplaceParameters(ctx, state.Parameters)
	if err != nil || digest != prior.UpdatedAdapterSHA {
		return errors.New("previous adapter identity differs")
	}
	if previous == "" {
		fmt.Printf("phase=fresh_start adapter_sha256=%s example=%s\n", digest, row.ID)
	} else {
		fmt.Printf("phase=resumed step=%d adapter_sha256=%s example=%s\n", prior.Step, digest, row.ID)
	}
	memory("resumed")
	if output == "" {
		observed, err := forwardFingerprint(ctx, loaded.Model, row.InputIDs, int64(row.PromptTokens), recipe.MaxCheckpointBytes)
		if err != nil || observed != prior.LogitsAfterSHA {
			return errors.Join(errors.New("reloaded checkpoint inference differs"), err)
		}
		fmt.Printf("phase=checkpoint_reload_verified step=%d logits_sha256=%s elapsed=%s\n", prior.Step, observed, time.Since(started))
		return nil
	}
	gradient, distilled, err := completionStep(ctx, loaded.Model, row, recipe, teachers)
	if err != nil || gradient.Tokens != len(row.InputIDs)-row.PromptTokens || len(gradient.Gradients) != len(names) {
		return errors.Join(errors.New("resumed completion gradient incomplete"), err)
	}
	memory("gradient")
	flat := make([]float32, 0, len(state.Parameters))
	for i, item := range gradient.Gradients {
		if item.Name != names[i] || len(item.ValuesF32) != len(adapter[i]) {
			return errors.New("resumed gradient geometry differs")
		}
		flat = append(flat, item.ValuesF32...)
	}
	config := recipe.Optimizer
	next, receipt, err := optim.UpdateAdamW(state, flat, config)
	if err != nil {
		return err
	}
	nextDigest, err := loaded.ReplaceParameters(ctx, next.Parameters)
	if err != nil || nextDigest == digest {
		return errors.Join(errors.New("resumed update did not install"), err)
	}
	after, err := forwardFingerprint(ctx, loaded.Model, row.InputIDs, int64(row.PromptTokens), recipe.MaxCheckpointBytes)
	if err != nil {
		return err
	}
	memory("updated_inference")
	parent := filepath.Dir(output)
	if err := os.MkdirAll(parent, 0700); err != nil {
		return err
	}
	if _, err := os.Lstat(output); err == nil {
		return errors.New("next checkpoint already exists")
	} else if !errors.Is(err, os.ErrNotExist) {
		return err
	}
	temporary, err := os.MkdirTemp(parent, ".training-step-")
	if err != nil {
		return err
	}
	defer os.RemoveAll(temporary)
	newAdapter := make([]checkpoint.Float32Tensor, 0, len(names))
	newMoments := make([]checkpoint.Float32Tensor, 0, len(momentNames))
	position := 0
	for i, name := range names {
		count := len(adapter[i])
		shape := []uint64{uint64(shapes[i][0]), uint64(shapes[i][1])}
		newAdapter = append(newAdapter, checkpoint.Float32Tensor{Name: name, Shape: shape, Values: next.Parameters[position : position+count]})
		newMoments = append(newMoments, checkpoint.Float32Tensor{Name: "m." + name, Shape: shape, Values: next.First[position : position+count]}, checkpoint.Float32Tensor{Name: "v." + name, Shape: shape, Values: next.Second[position : position+count]})
		position += count
	}
	adapterReceipt, err := writeAndReadbackWithStorage(ctx, filepath.Join(temporary, "adapter_model.safetensors"), newAdapter, storage)
	if err != nil {
		return err
	}
	momentsReceipt, err := writeAndReadbackWithStorage(ctx, filepath.Join(temporary, "optimizer_moments.safetensors"), newMoments, storage)
	if err != nil {
		return err
	}
	manifest := stepManifest{RecipeSHA256: recipe.Digest(), Step: next.Step, ExampleID: row.ID, SupervisedTokens: gradient.Tokens, LossBefore: gradient.Loss,
		UpdatedAdapterSHA: nextDigest, AdapterFileSHA: adapterReceipt.SHA256, OptimizerFileSHA: momentsReceipt.SHA256,
		LogitsAfterSHA: after, BaseRevision: prior.BaseRevision}
	if distilled != nil {
		manifest.DistillationLossBefore, manifest.TeacherMass, manifest.TeacherLosses = &distilled.loss, distilled.teacherMass, distilled.teacherLosses
	}
	body, err = json.MarshalIndent(manifest, "", "  ")
	if err != nil {
		return err
	}
	if storage != nil && int64(len(body)+1) > storage.manifestBytes {
		return ErrStage
	}
	file, err := os.OpenFile(filepath.Join(temporary, "manifest.json"), os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
	if err != nil {
		return err
	}
	if _, err := file.Write(append(body, '\n')); err != nil {
		file.Close()
		return err
	}
	if err := errors.Join(file.Sync(), file.Close()); err != nil {
		return err
	}
	if err := syncLocalDirectory(temporary); err != nil {
		return err
	}
	if err := os.Rename(temporary, output); err != nil {
		return err
	}
	directory, err := os.Open(parent)
	if err == nil {
		err = errors.Join(directory.Sync(), directory.Close())
	}
	if err != nil {
		return err
	}
	if distilled != nil {
		fmt.Printf("phase=distillation step=%d hard_loss=%g distillation_loss=%g\n", receipt.Step, gradient.Loss, distilled.loss)
	}
	fmt.Printf("phase=step_verified step=%d loss=%g tokens=%d changed=%d elapsed=%s checkpoint=%s\n", receipt.Step, gradient.Loss, gradient.Tokens, receipt.ChangedParameter, time.Since(started), output)
	return nil
}
