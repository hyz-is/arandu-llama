package local

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"io"
	"os"
	"path/filepath"
)

// ErrLoss refuses a loss readout whose inputs cannot be tied to what they name.
var ErrLoss = errors.New("local training: loss readout refused")

// maxLossRequests bounds one model residency; each request is one forward.
const maxLossRequests = 1 << 16

// LossConfig fixes the frozen base, the tokenized rows and the directory every
// named checkpoint must live under. The recipe identifies the base, the pinned
// initializer and the rotary tables; its optimizer and loss scale play no part
// in a reading. DataSHA256 pins the rows, which need not be the curriculum.
type LossConfig struct {
	BundleDir, ModelDir  string
	DataPath, DataSHA256 string
	CheckpointRoot       string
	Recipe               Recipe
}

// LossRequest names one reading. An empty Checkpoint is the recipe's pinned
// initializer, which is the base model because every LoRA B starts at zero;
// otherwise it is the absolute step directory holding manifest.json and
// adapter_model.safetensors, under LossConfig.CheckpointRoot.
type LossRequest struct {
	Checkpoint string
	ExampleID  string
}

// LossResult is one reading, returned in request order. Loss is the mean
// completion NLL a step logs as loss_before when it starts from this adapter;
// CompletionTokens is its weight in any aggregate, which is summed, never
// averaged over means. Step is 0 for the initializer. AdapterSHA256 is the
// digest of the adapter installed for the reading.
type LossResult struct {
	Checkpoint, ExampleID string
	Step                  uint64
	Loss                  float64
	CompletionTokens      int
	AdapterSHA256         string
}

// snapshot owns every slice before validation, as Config.snapshot does.
func (c LossConfig) snapshot() (LossConfig, error) {
	body, err := json.Marshal(c)
	if err != nil {
		return LossConfig{}, ErrLoss
	}
	var owned LossConfig
	if err := json.Unmarshal(body, &owned); err != nil {
		return LossConfig{}, errors.Join(ErrLoss, err)
	}
	for _, path := range []string{owned.BundleDir, owned.ModelDir, owned.DataPath, owned.CheckpointRoot} {
		if !filepath.IsAbs(path) || filepath.Clean(path) != path {
			return LossConfig{}, ErrLoss
		}
	}
	if !validHash(owned.DataSHA256) {
		return LossConfig{}, ErrLoss
	}
	if err := owned.Recipe.validate(); err != nil {
		return LossConfig{}, errors.Join(ErrLoss, err)
	}
	return owned, nil
}

// lossCheckpoint returns the root-relative directory of a named checkpoint, or
// "" for the initializer. Anything that is not a clean absolute path strictly
// inside the root is refused before a file is opened.
func (c LossConfig) lossCheckpoint(checkpoint string) (string, error) {
	if checkpoint == "" {
		return "", nil
	}
	if !filepath.IsAbs(checkpoint) || filepath.Clean(checkpoint) != checkpoint {
		return "", ErrLoss
	}
	relative, err := filepath.Rel(c.CheckpointRoot, checkpoint)
	if err != nil || !filepath.IsLocal(relative) || relative == "." {
		return "", ErrLoss
	}
	return relative, nil
}

// lossOrder groups request indices by checkpoint in first-appearance order, so
// each adapter is installed once while results keep the request order.
func lossOrder(requests []LossRequest) [][]int {
	var groups [][]int
	position := map[string]int{}
	for index, request := range requests {
		group, ok := position[request.Checkpoint]
		if !ok {
			group = len(groups)
			position[request.Checkpoint] = group
			groups = append(groups, nil)
		}
		groups[group] = append(groups[group], index)
	}
	return groups
}

// lossRows reads the pinned rows the requests name, holding every row to the
// stage's geometry and label mask. Any length is admitted up to the recipe's
// rotary and sequence caps, since a readout is not a curriculum.
func lossRows(ctx context.Context, c LossConfig, requests []LossRequest) (map[string]example, error) {
	if len(requests) < 1 || len(requests) > maxLossRequests {
		return nil, ErrLoss
	}
	for _, request := range requests {
		if request.ExampleID == "" {
			return nil, ErrLoss
		}
		if _, err := c.lossCheckpoint(request.Checkpoint); err != nil {
			return nil, err
		}
	}
	before, err := os.Lstat(c.DataPath)
	if err != nil || !before.Mode().IsRegular() {
		return nil, errors.Join(ErrLoss, err)
	}
	file, err := os.Open(c.DataPath)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	if after, err := file.Stat(); err != nil || !os.SameFile(before, after) {
		return nil, errors.Join(ErrLoss, err)
	}
	limit := int(min(c.Recipe.Rotary.MaxTokens, 4096))
	if c.Recipe.Assembly.Sequence.MaxTokens < int64(limit) {
		limit = int(c.Recipe.Assembly.Sequence.MaxTokens)
	}
	wanted := map[string]bool{}
	for _, request := range requests {
		wanted[request.ExampleID] = true
	}
	hash := sha256.New()
	decoder := json.NewDecoder(io.TeeReader(stageContextReader{ctx, file}, hash))
	seen := map[string]bool{}
	rows := make(map[string]example, len(wanted))
	for {
		var row example
		if err := decoder.Decode(&row); err != nil {
			if errors.Is(err, io.EOF) {
				break
			}
			return nil, errors.Join(ErrLoss, err)
		}
		if row.ID == "" || seen[row.ID] || len(row.InputIDs) < 2 || len(row.Labels) != len(row.InputIDs) ||
			row.PromptTokens < 1 || row.PromptTokens >= len(row.InputIDs) {
			return nil, ErrLoss
		}
		for i, label := range row.Labels {
			if row.InputIDs[i] < 0 || i < row.PromptTokens && label != -100 || i >= row.PromptTokens && label != row.InputIDs[i] {
				return nil, ErrLoss
			}
		}
		seen[row.ID] = true
		if wanted[row.ID] {
			rows[row.ID] = row
		}
	}
	if hex.EncodeToString(hash.Sum(nil)) != c.DataSHA256 {
		return nil, ErrLoss
	}
	for id := range wanted {
		if row, ok := rows[id]; !ok || len(row.InputIDs) > limit {
			return nil, ErrLoss
		}
	}
	return rows, ctx.Err()
}

// lossManifests reads the manifest of every named checkpoint through the root,
// so a symlinked component or a non-regular file is refused, and holds it to
// the recipe's base and to the adapter bytes it names. It reads nothing else:
// a readout needs no optimizer moments. The restore verifies both digests
// again when it installs the adapter, so a file swapped after this check is
// refused there rather than measured.
func lossManifests(ctx context.Context, c LossConfig, requests []LossRequest) (map[string]stepManifest, error) {
	manifests := map[string]stepManifest{}
	var root *os.Root
	defer func() {
		if root != nil {
			_ = root.Close()
		}
	}()
	for _, request := range requests {
		relative, err := c.lossCheckpoint(request.Checkpoint)
		if err != nil {
			return nil, err
		}
		if _, done := manifests[request.Checkpoint]; done || relative == "" {
			continue
		}
		if root == nil {
			if root, err = stageRoot(c.CheckpointRoot); err != nil {
				return nil, errors.Join(ErrLoss, err)
			}
		}
		body, err := stageRead(ctx, root, filepath.Join(relative, "manifest.json"), 1<<20, "")
		if err != nil {
			return nil, errors.Join(ErrLoss, err)
		}
		var manifest stepManifest
		if err := json.Unmarshal(body, &manifest); err != nil || manifest.Step == 0 ||
			manifest.BaseRevision != c.Recipe.BaseRevision || !validHash(manifest.AdapterFileSHA) || !validHash(manifest.UpdatedAdapterSHA) {
			return nil, errors.Join(ErrLoss, err)
		}
		file, err := stageOpen(root, filepath.Join(relative, "adapter_model.safetensors"))
		if err != nil {
			return nil, errors.Join(ErrLoss, err)
		}
		hash := sha256.New()
		_, copyErr := io.Copy(hash, stageContextReader{ctx, file})
		if err := errors.Join(copyErr, file.Close()); err != nil {
			return nil, errors.Join(ErrLoss, err)
		}
		if hex.EncodeToString(hash.Sum(nil)) != manifest.AdapterFileSHA {
			return nil, ErrLoss
		}
		manifests[request.Checkpoint] = manifest
	}
	return manifests, ctx.Err()
}
