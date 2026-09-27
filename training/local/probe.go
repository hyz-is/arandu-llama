//go:build libtorch && cgo

// Local causal SFT through bounded completion updates with explicit admission.
// Job ownership and queue state remain with the application.
package local

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/tayi-ai/arandu-llama/training/decoder"
	"github.com/tayi-ai/arandu-llama/training/torch"
)

type fileShard struct {
	*os.File
	size int64
}

func (s *fileShard) Size() int64 { return s.size }

type files struct{ root string }

func (f files) OpenShard(ctx context.Context, name string) (decoder.Shard, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if name == "" || name != filepath.Base(name) || strings.ContainsAny(name, "/\\") || !strings.HasSuffix(name, ".safetensors") {
		return nil, errors.New("unadmitted shard basename")
	}
	path := filepath.Join(f.root, name)
	before, err := os.Lstat(path)
	if err != nil || !before.Mode().IsRegular() {
		return nil, fmt.Errorf("shard is not a regular file: %s: %w", name, err)
	}
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	after, err := file.Stat()
	if err != nil || !os.SameFile(before, after) {
		_ = file.Close()
		return nil, errors.New("shard changed during open")
	}
	return &fileShard{File: file, size: after.Size()}, nil
}

func selectedID(path, id string) (example, error) {
	file, err := os.Open(path)
	if err != nil {
		return example{}, err
	}
	defer file.Close()
	decoder := json.NewDecoder(file)
	for {
		var row example
		if err := decoder.Decode(&row); err != nil {
			if errors.Is(err, io.EOF) {
				return example{}, errors.New("selected example absent")
			}
			return example{}, err
		}
		if row.ID != id {
			continue
		}
		if len(row.InputIDs) < 2 || len(row.InputIDs) > 4096 || len(row.Labels) != len(row.InputIDs) || row.PromptTokens < 1 || row.PromptTokens >= len(row.InputIDs) {
			return example{}, errors.New("selected example geometry differs")
		}
		for index, label := range row.Labels {
			if index < row.PromptTokens && label != -100 || index >= row.PromptTokens && label != row.InputIDs[index] {
				return example{}, errors.New("selected example mask differs")
			}
		}
		return row, nil
	}
}

func run(ctx context.Context, bundle, modelDir, data, output, reload, nextID string, curriculumMax, curriculumSteps int, c Config, hooks *stepHooks) (err error) {
	read := func(name, expected string) ([]byte, error) {
		if hooks != nil && hooks.storage != nil {
			root, err := stageRoot(bundle)
			if err != nil {
				return nil, err
			}
			body, readErr := stageRead(ctx, root, name, hooks.storage.metadataBytes, expected)
			return body, errors.Join(readErr, root.Close())
		}
		body, err := os.ReadFile(filepath.Join(bundle, name))
		if err != nil {
			return nil, err
		}
		actual := sha256.Sum256(body)
		if hex.EncodeToString(actual[:]) != expected {
			return nil, fmt.Errorf("%s digest differs", name)
		}
		return body, nil
	}
	index, err := read("model.safetensors.index.json", c.Recipe.Identity.IndexSHA256)
	if err != nil {
		return err
	}
	config, err := read("config.json", c.Recipe.Identity.ConfigSHA256)
	if err != nil {
		return err
	}
	reference, err := read("initial-reference.json", c.Recipe.Identity.ReferenceSHA256)
	if err != nil {
		return err
	}
	rowID, err := nextCurriculumID(data, reload, curriculumMax, c)
	if err != nil {
		return err
	}
	if nextID != "" {
		rowID = nextID
	}
	row, err := selectedID(data, rowID)
	if err != nil {
		return err
	}
	tokens := int64(len(row.InputIDs))
	if curriculumMax > 0 {
		tokens = int64(curriculumMax)
	}
	plan, err := decoder.PlanLocalMPSAssembly(index, config, reference, c.Recipe.Identity, c.Recipe.Assembly, c.Recipe.MaxMPSBytes)
	if err != nil {
		return err
	}
	geometry := plan.Geometry()
	if c.Recipe.Rotary.Theta != geometry.RoPE.Theta || c.Recipe.Rotary.Dimension != int(float64(geometry.Dimension)*geometry.RoPE.Partial) || c.Recipe.Assembly.Sequence.MaxTokens < int64(c.MaxTokens) {
		return errors.New("local training: rotary or sequence admission differs from geometry")
	}
	fmt.Printf("phase=plan tensors=%d local_mps_bytes=%d\n", len(plan.Tensors()), plan.Summary().LocalMPSBytes)
	provider := files{root: modelDir}
	inspection, err := decoder.InspectAssemblySources(ctx, plan, provider)
	if err != nil {
		return err
	}
	fmt.Printf("phase=inspect shards=%d tensors=%d\n", inspection.Shards, inspection.BaseTensors)
	initial, err := decoder.InitializeAdapter(ctx, c.Recipe.Initializer)
	if err != nil {
		return err
	}
	defer func() { err = errors.Join(err, initial.Close()) }()
	start := time.Now()
	loaded, err := decoder.LoadTextAssembly(ctx, plan, provider, initial)
	if err != nil {
		return err
	}
	defer func() { err = errors.Join(err, loaded.Close()) }()
	fmt.Printf("phase=loaded elapsed=%s receipts=%d\n", time.Since(start), len(loaded.Receipts))
	if curriculumMax > 0 {
		return runCurriculumWithHooks(ctx, loaded, data, reload, output, curriculumMax, curriculumSteps, c, hooks, torch.MPSDevice())
	}
	tables, err := decoder.TextRotary(ctx, len(row.InputIDs), torch.MPSDevice(), c.Recipe.Rotary)
	if err != nil {
		return err
	}
	defer func() { err = errors.Join(err, tables.Close()) }()
	for index := range loaded.Model.Layers {
		if loaded.Model.Layers[index].Weights.Full != nil {
			loaded.Model.Layers[index].Cosine, loaded.Model.Layers[index].Sine = tables.Cosine, tables.Sine
		}
	}
	if nextID != "" {
		// Resuming a named next example always continues a checkpoint. An empty
		// reload here would reach the fresh-start path of resumeNext, which only
		// the admitted curriculum may enter.
		if reload == "" {
			return errors.New("resuming the next example requires its previous checkpoint")
		}
		return resumeNext(ctx, loaded, row, reload, output, c.Recipe)
	}
	if reload != "" {
		if err := reloadAndVerifyAdapter(ctx, loaded, row, reload, c.Recipe); err != nil {
			return err
		}
		fmt.Printf("phase=reloaded_adapter_verified elapsed=%s\n", time.Since(start))
		return nil
	}
	var before string
	if output != "" {
		before, err = forwardFingerprint(ctx, loaded.Model, row.InputIDs, int64(row.PromptTokens), c.Recipe.MaxCheckpointBytes)
		if err != nil {
			return err
		}
		fmt.Printf("phase=base_forward logits_sha256=%s elapsed=%s\n", before, time.Since(start))
	}
	gradient, err := decoder.CompletionGradient(ctx, loaded.Model, row.InputIDs, row.PromptTokens,
		decoder.Limits{MaxTokens: tokens, LogitRows: tokens - int64(row.PromptTokens) + 1, MaxCheckpointBytes: c.Recipe.MaxCheckpointBytes}, c.Recipe.LossScale)
	if err != nil {
		return err
	}
	var nonzero int
	for _, item := range gradient.Gradients {
		for _, value := range item.ValuesF32 {
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				return errors.New("nonfinite gradient")
			}
			if value != 0 {
				nonzero++
			}
		}
	}
	if math.IsNaN(gradient.Loss) || math.IsInf(gradient.Loss, 0) || nonzero == 0 {
		return errors.New("invalid completion result")
	}
	fmt.Printf("phase=gradient example=%s loss=%g tokens=%d gradients=%d nonzero=%d elapsed=%s\n",
		row.ID, gradient.Loss, gradient.Tokens, len(gradient.Gradients), nonzero, time.Since(start))
	if output != "" {
		if err := applyAndVerifyFirstUpdate(ctx, loaded, row, gradient, before, output, c.Recipe); err != nil {
			return err
		}
		fmt.Printf("phase=step_1_verified elapsed=%s\n", time.Since(start))
	}
	return nil
}
