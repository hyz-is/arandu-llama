package local

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"io"
	"os"
)

func curriculumPosition(data, previous string, c Config) ([]example, stepManifest, int, error) {
	file, err := os.Open(data)
	if err != nil {
		return nil, stepManifest{}, 0, err
	}
	defer file.Close()
	hash := sha256.New()
	decoder := json.NewDecoder(io.TeeReader(file, hash))
	rows := make([]example, 0)
	seen := map[string]bool{}
	lastLength := 0
	for {
		var row example
		if err := decoder.Decode(&row); err != nil {
			if errors.Is(err, io.EOF) {
				break
			}
			return nil, stepManifest{}, 0, err
		}
		if row.ID == "" || seen[row.ID] || c.Recipe.RequireLengthOrder && len(row.InputIDs) < lastLength || len(row.InputIDs) != len(row.Labels) || row.PromptTokens < 1 || row.PromptTokens >= len(row.InputIDs) {
			return nil, stepManifest{}, 0, errors.New("curriculum order or geometry differs")
		}
		if len(rows) >= c.Recipe.ExampleCount {
			return nil, stepManifest{}, 0, errors.New("curriculum count exceeds admission")
		}
		for i, label := range row.Labels {
			if row.InputIDs[i] < 0 || i < row.PromptTokens && label != -100 || i >= row.PromptTokens && label != row.InputIDs[i] {
				return nil, stepManifest{}, 0, errors.New("curriculum token or mask differs")
			}
		}
		seen[row.ID] = true
		lastLength = len(row.InputIDs)
		rows = append(rows, row)
	}
	if hex.EncodeToString(hash.Sum(nil)) != c.Recipe.DataSHA256 || len(rows) != c.Recipe.ExampleCount {
		return nil, stepManifest{}, 0, errors.New("admitted curriculum digest or count differs")
	}
	if previous == "" {
		// Only an explicit fresh start stands before the first example; a
		// continuation without its checkpoint is refused, not restarted.
		if !c.FreshStart {
			return nil, stepManifest{}, 0, errors.New("previous checkpoint required unless starting fresh")
		}
		return rows, stepManifest{}, 0, nil
	}
	prior, err := readManifest(previous, c)
	if err != nil {
		return nil, stepManifest{}, 0, err
	}
	for i, row := range rows {
		if row.ID == prior.ExampleID {
			if prior.Step != uint64(i+1) {
				return nil, stepManifest{}, 0, errors.New("checkpoint step differs from dataset cursor")
			}
			return rows, prior, i + 1, nil
		}
	}
	return nil, stepManifest{}, 0, errors.New("checkpoint example absent from admitted curriculum")
}

func nextCurriculumID(data, previous string, maxTokens int, c Config) (string, error) {
	rows, _, next, err := curriculumPosition(data, previous, c)
	if err != nil {
		return "", err
	}
	if next >= len(rows) || len(rows[next].InputIDs) > maxTokens {
		return "", errors.New("no next complete example within admitted token cap")
	}
	return rows[next].ID, nil
}
