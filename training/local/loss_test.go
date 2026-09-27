package local

import (
	"context"
	"encoding/json"
	"errors"
	"io/fs"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

// treeDigest records every path, mode, link target and content under root, so
// a readout that wrote, touched or replaced anything there is caught.
func treeDigest(t *testing.T, root string) string {
	t.Helper()
	var entries []string
	err := filepath.WalkDir(root, func(path string, entry fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		info, err := os.Lstat(path)
		if err != nil {
			return err
		}
		relative, _ := filepath.Rel(root, path)
		line := relative + " " + info.Mode().String() + " " + info.ModTime().String()
		switch {
		case info.Mode()&os.ModeSymlink != 0:
			target, err := os.Readlink(path)
			if err != nil {
				return err
			}
			line += " -> " + target
		case info.Mode().IsRegular():
			body, err := os.ReadFile(path)
			if err != nil {
				return err
			}
			line += " " + fmtHash(body)
		}
		entries = append(entries, line)
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	return fmtHash([]byte(strings.Join(entries, "\n")))
}

// lossFixture is the synthetic checkpoint fixture turned into a readout: two
// steps under their own root, with the adapter digests a readout depends on.
func lossFixture(t *testing.T) (LossConfig, []string) {
	t.Helper()
	base := checkpointFixture(t)
	base.Recipe.Assembly.Sequence.MaxTokens = 2
	root := filepath.Join(filepath.Dir(base.DataPath), "steps")
	if err := os.Mkdir(root, 0700); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(base.DataPath)
	if err != nil {
		t.Fatal(err)
	}
	c := LossConfig{BundleDir: base.BundleDir, ModelDir: base.ModelDir, DataPath: base.DataPath, DataSHA256: fmtHash(data), CheckpointRoot: root, Recipe: base.Recipe}
	var steps []string
	for step, id := range []string{"first", "second"} {
		directory := filepath.Join(root, "step-00"+string(rune('1'+step)))
		if err := os.Mkdir(directory, 0700); err != nil {
			t.Fatal(err)
		}
		adapter := []byte("synthetic adapter " + id)
		if err := os.WriteFile(filepath.Join(directory, "adapter_model.safetensors"), adapter, 0600); err != nil {
			t.Fatal(err)
		}
		writeLossManifest(t, directory, stepManifest{Step: uint64(step + 1), ExampleID: id, BaseRevision: c.Recipe.BaseRevision,
			AdapterFileSHA: fmtHash(adapter), UpdatedAdapterSHA: fmtHash([]byte("parameters " + id))})
		steps = append(steps, directory)
	}
	return c, steps
}

func writeLossManifest(t *testing.T, directory string, m stepManifest) {
	t.Helper()
	body, err := json.Marshal(m)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(directory, "manifest.json"), body, 0600); err != nil {
		t.Fatal(err)
	}
}

func readLossManifest(t *testing.T, directory string) stepManifest {
	t.Helper()
	body, err := os.ReadFile(filepath.Join(directory, "manifest.json"))
	if err != nil {
		t.Fatal(err)
	}
	var m stepManifest
	if err := json.Unmarshal(body, &m); err != nil {
		t.Fatal(err)
	}
	return m
}

func TestLossOrderInstallsEachCheckpointOnceAndKeepsRequestOrder(t *testing.T) {
	requests := []LossRequest{{"/a", "x"}, {"/b", "x"}, {"/a", "y"}, {"", "x"}, {"/b", "y"}, {"", "y"}}
	if got := lossOrder(requests); !reflect.DeepEqual(got, [][]int{{0, 2}, {1, 4}, {3, 5}}) {
		t.Fatal("requests are not grouped by first appearance", got)
	}
}

func TestLossReadoutReadsPinnedRowsAndVerifiedManifestsOnly(t *testing.T) {
	c, steps := lossFixture(t)
	ctx := context.Background()
	owned, err := c.snapshot()
	if err != nil {
		t.Fatal(err)
	}
	requests := []LossRequest{{steps[1], "second"}, {"", "first"}, {steps[0], "second"}, {steps[1], "first"}}
	before := treeDigest(t, filepath.Dir(c.DataPath))
	rows, err := lossRows(ctx, owned, requests)
	if err != nil || len(rows) != 2 || rows["second"].InputIDs[1] != 3 {
		t.Fatal("pinned rows were not read", rows, err)
	}
	manifests, err := lossManifests(ctx, owned, requests)
	if err != nil || len(manifests) != 2 || manifests[steps[0]].Step != 1 || manifests[steps[1]].Step != 2 {
		t.Fatal("named manifests were not read", manifests, err)
	}
	if _, named := manifests[""]; named {
		t.Fatal("the initializer was read as a checkpoint")
	}
	if treeDigest(t, filepath.Dir(c.DataPath)) != before {
		t.Fatal("a readout changed its sources")
	}
}

func TestLossReadoutRefusesUnboundConfigurationRowsAndPaths(t *testing.T) {
	for name, mutate := range map[string]func(*LossConfig){
		"relative data":        func(c *LossConfig) { c.DataPath = "data.jsonl" },
		"unclean root":         func(c *LossConfig) { c.CheckpointRoot += "/." },
		"unpinned data":        func(c *LossConfig) { c.DataSHA256 = "" },
		"recipe without base":  func(c *LossConfig) { c.Recipe.BaseRevision = "" },
		"recipe without pin":   func(c *LossConfig) { c.Recipe.Identity.IndexSHA256 = "" },
		"initializer mismatch": func(c *LossConfig) { c.Recipe.Initializer.ExpectedSHA256 = strings.Repeat("2", 64) },
	} {
		t.Run(name, func(t *testing.T) {
			c, _ := lossFixture(t)
			mutate(&c)
			if _, err := c.snapshot(); !errors.Is(err, ErrLoss) {
				t.Fatal("unbound readout configuration admitted", err)
			}
		})
	}
	for name, request := range map[string]func(LossConfig, []string) []LossRequest{
		"no request":      func(LossConfig, []string) []LossRequest { return nil },
		"unnamed example": func(_ LossConfig, s []string) []LossRequest { return []LossRequest{{s[0], ""}} },
		"unknown example": func(_ LossConfig, s []string) []LossRequest { return []LossRequest{{s[0], "absent"}} },
		"relative step":   func(LossConfig, []string) []LossRequest { return []LossRequest{{"step-001", "first"}} },
		"the root itself": func(c LossConfig, _ []string) []LossRequest { return []LossRequest{{c.CheckpointRoot, "first"}} },
		"beside the root": func(c LossConfig, _ []string) []LossRequest {
			return []LossRequest{{c.CheckpointRoot + "-other/step-001", "first"}}
		},
		"above the root": func(c LossConfig, _ []string) []LossRequest {
			return []LossRequest{{filepath.Dir(c.CheckpointRoot), "first"}}
		},
		"unclean traversal": func(c LossConfig, _ []string) []LossRequest {
			return []LossRequest{{c.CheckpointRoot + "/../steps/step-001", "first"}}
		},
		"too many to bound":  func(LossConfig, []string) []LossRequest { return make([]LossRequest, maxLossRequests+1) },
		"longer than rotary": func(LossConfig, []string) []LossRequest { return []LossRequest{{"", "long"}} },
	} {
		t.Run(name, func(t *testing.T) {
			c, steps := lossFixture(t)
			if name == "longer than rotary" {
				appendLossRow(t, &c, `{"id":"long","input_ids":[1,2,3],"labels":[-100,2,3],"prompt_tokens":1}`)
			}
			if _, err := lossRows(context.Background(), c, request(c, steps)); !errors.Is(err, ErrLoss) {
				t.Fatal("unbound readout request admitted", err)
			}
		})
	}
	for name, row := range map[string]string{
		"duplicate id":      `{"id":"first","input_ids":[1,2],"labels":[-100,2],"prompt_tokens":1}`,
		"prompt label kept": `{"id":"third","input_ids":[1,2],"labels":[1,2],"prompt_tokens":1}`,
		"completion masked": `{"id":"third","input_ids":[1,2],"labels":[-100,-100],"prompt_tokens":1}`,
		"no completion":     `{"id":"third","input_ids":[1,2],"labels":[-100,-100],"prompt_tokens":2}`,
		"negative token":    `{"id":"third","input_ids":[1,-2],"labels":[-100,-2],"prompt_tokens":1}`,
	} {
		t.Run(name, func(t *testing.T) {
			c, _ := lossFixture(t)
			appendLossRow(t, &c, row)
			if _, err := lossRows(context.Background(), c, []LossRequest{{"", "first"}}); !errors.Is(err, ErrLoss) {
				t.Fatal("invalid row admitted", err)
			}
		})
	}
	c, _ := lossFixture(t)
	c.DataSHA256 = fmtHash([]byte("other rows"))
	if _, err := lossRows(context.Background(), c, []LossRequest{{"", "first"}}); !errors.Is(err, ErrLoss) {
		t.Fatal("rows other than the pinned ones were read", err)
	}
}

// appendLossRow adds a row and repins the data, so only the row is at fault.
func appendLossRow(t *testing.T, c *LossConfig, row string) {
	t.Helper()
	body, err := os.ReadFile(c.DataPath)
	if err != nil {
		t.Fatal(err)
	}
	body = append(body, []byte(row+"\n")...)
	if err := os.WriteFile(c.DataPath, body, 0600); err != nil {
		t.Fatal(err)
	}
	c.DataSHA256 = fmtHash(body)
}

func TestLossReadoutRefusesManifestsOtherThanTheRecipesAndTheirAdapter(t *testing.T) {
	for name, mutate := range map[string]func(*testing.T, LossConfig, string){
		"another base revision": func(t *testing.T, c LossConfig, step string) {
			m := readLossManifest(t, step)
			m.BaseRevision = "another-revision"
			writeLossManifest(t, step, m)
		},
		"adapter bytes changed": func(t *testing.T, c LossConfig, step string) {
			if err := os.WriteFile(filepath.Join(step, "adapter_model.safetensors"), []byte("tampered"), 0600); err != nil {
				t.Fatal(err)
			}
		},
		"manifest names other bytes": func(t *testing.T, c LossConfig, step string) {
			m := readLossManifest(t, step)
			m.AdapterFileSHA = fmtHash([]byte("another adapter"))
			writeLossManifest(t, step, m)
		},
		"parameters unnamed": func(t *testing.T, c LossConfig, step string) {
			m := readLossManifest(t, step)
			m.UpdatedAdapterSHA = ""
			writeLossManifest(t, step, m)
		},
		"step zero": func(t *testing.T, c LossConfig, step string) {
			m := readLossManifest(t, step)
			m.Step = 0
			writeLossManifest(t, step, m)
		},
		"manifest missing": func(t *testing.T, c LossConfig, step string) {
			if err := os.Remove(filepath.Join(step, "manifest.json")); err != nil {
				t.Fatal(err)
			}
		},
		"adapter symlinked": func(t *testing.T, c LossConfig, step string) {
			path := filepath.Join(step, "adapter_model.safetensors")
			if err := os.Rename(path, path+".original"); err != nil {
				t.Fatal(err)
			}
			if err := os.Symlink(path+".original", path); err != nil {
				t.Fatal(err)
			}
		},
		"manifest symlinked": func(t *testing.T, c LossConfig, step string) {
			path := filepath.Join(step, "manifest.json")
			if err := os.Rename(path, path+".original"); err != nil {
				t.Fatal(err)
			}
			if err := os.Symlink(path+".original", path); err != nil {
				t.Fatal(err)
			}
		},
		"step symlinked": func(t *testing.T, c LossConfig, step string) {
			if err := os.Rename(step, step+"-original"); err != nil {
				t.Fatal(err)
			}
			if err := os.Symlink(step+"-original", step); err != nil {
				t.Fatal(err)
			}
		},
	} {
		t.Run(name, func(t *testing.T) {
			c, steps := lossFixture(t)
			mutate(t, c, steps[0])
			if _, err := lossManifests(context.Background(), c, []LossRequest{{steps[1], "second"}, {steps[0], "second"}}); !errors.Is(err, ErrLoss) {
				t.Fatal("unbound checkpoint admitted", err)
			}
		})
	}
}
