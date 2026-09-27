package pipeline_test

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"testing"

	"github.com/tayi-ai/arandu-llama/training/pipeline"
)

func sftRecipe() pipeline.Recipe {
	r := pipeline.Recipe{Version: 3, Scope: "sft", ID: "master-fresh-sft", Method: "causal-sft-v1", Student: ref("student"), Training: ref("train")}
	r.Stages = []pipeline.Stage{{ID: "sft", Phase: pipeline.PhaseSFT, Inputs: []pipeline.ArtifactRef{r.Training}, MaxSteps: 20, MaxTokens: 4096, TimeoutSeconds: 7200}}
	return r
}

func TestSFTScopeAdmitsOneStageOverStudentAndTrainingOnly(t *testing.T) {
	r := sftRecipe()
	if err := r.Validate(); err != nil {
		t.Fatal(err)
	}
	// The student may also be a declared input: the stage reads its weights.
	withStudent := sftRecipe()
	withStudent.Stages[0].Inputs = append(withStudent.Stages[0].Inputs, withStudent.Student)
	if err := withStudent.Validate(); err != nil {
		t.Fatal("student refused as an input of its own SFT stage", err)
	}
	body, err := json.Marshal(r)
	if err != nil {
		t.Fatal(err)
	}
	sum := sha256.Sum256(body)
	decoded, err := pipeline.DecodeRecipe(body, hex.EncodeToString(sum[:]))
	if err != nil || decoded.Version != 3 || decoded.Scope != "sft" || len(decoded.Stages) != 1 {
		t.Fatal("v3 did not survive its own wire form", decoded, err)
	}
}

// Absent and empty teachers are the same recipe, so they must be one digest;
// otherwise two runs of the identical protocol would look like two protocols.
func TestSFTScopeDigestDoesNotDependOnHowNoTeachersWasWritten(t *testing.T) {
	absent := sftRecipe()
	empty := sftRecipe()
	empty.Teachers = []pipeline.ArtifactRef{}
	a, err := absent.Digest()
	if err != nil {
		t.Fatal(err)
	}
	e, err := empty.Digest()
	if err != nil || a != e {
		t.Fatal("no teachers written two ways produced two digests", a, e, err)
	}
	for _, body := range []string{`null`, `[]`} {
		raw, err := json.Marshal(absent)
		if err != nil {
			t.Fatal(err)
		}
		raw = bytes.Replace(raw, []byte(`"teachers":null`), []byte(`"teachers":`+body), 1)
		sum := sha256.Sum256(raw)
		decoded, err := pipeline.DecodeRecipe(raw, hex.EncodeToString(sum[:]))
		if err != nil {
			t.Fatal(body, err)
		}
		d, err := decoded.Digest()
		if err != nil || d != a {
			t.Fatal("decoded teachers", body, "changed the typed digest", d, a, err)
		}
	}
}

// Every role this scope excludes is excluded by being refused, not ignored: a
// recipe that carried a teacher or heldout and still validated would let a
// preparatory run be described as more than it is.
func TestSFTScopeRefusesEveryRoleAndPhaseBeyondSFT(t *testing.T) {
	cases := map[string]func(*pipeline.Recipe){
		"teacher":      func(r *pipeline.Recipe) { r.Teachers = []pipeline.ArtifactRef{ref("teacher")} },
		"heldout":      func(r *pipeline.Recipe) { r.Heldout = ref("heldout") },
		"recovery":     func(r *pipeline.Recipe) { r.Recovery = ref("recovery") },
		"protection":   func(r *pipeline.Recipe) { r.Protection = ref("protection") },
		"calibration":  func(r *pipeline.Recipe) { r.Calibration = ref("calibration") },
		"method":       func(r *pipeline.Recipe) { r.Method = "generational-fusion-v1" },
		"scope master": func(r *pipeline.Recipe) { r.Scope = "master" },
		"scope empty":  func(r *pipeline.Recipe) { r.Scope = "" },
		"no stage":     func(r *pipeline.Recipe) { r.Stages = nil },
		"second stage": func(r *pipeline.Recipe) {
			s := r.Stages[0]
			s.ID = "again"
			s.ParentStage = "sft"
			r.Stages = append(r.Stages, s)
		},
		"teacher cache":    func(r *pipeline.Recipe) { r.Stages[0].Phase = pipeline.PhaseTeacherCache },
		"master phase":     func(r *pipeline.Recipe) { r.Stages[0].Phase = pipeline.PhaseMaster },
		"quantize phase":   func(r *pipeline.Recipe) { r.Stages[0].Phase = pipeline.PhaseQuantize; r.Stages[0].Format = "Q8_0" },
		"format":           func(r *pipeline.Recipe) { r.Stages[0].Format = "Q8_0" },
		"parent":           func(r *pipeline.Recipe) { r.Stages[0].ParentStage = "earlier" },
		"unregistered":     func(r *pipeline.Recipe) { r.Stages[0].Inputs = []pipeline.ArtifactRef{r.Training, ref("elsewhere")} },
		"training absent":  func(r *pipeline.Recipe) { r.Stages[0].Inputs = []pipeline.ArtifactRef{r.Student} },
		"training altered": func(r *pipeline.Recipe) { r.Stages[0].Inputs[0].SHA256 = ref("other").SHA256 },
		"repeated input":   func(r *pipeline.Recipe) { r.Stages[0].Inputs = []pipeline.ArtifactRef{r.Training, r.Training} },
		"student is data": func(r *pipeline.Recipe) {
			r.Training.ID = r.Student.ID
			r.Stages[0].Inputs = []pipeline.ArtifactRef{r.Training}
		},
		"student digest":    func(r *pipeline.Recipe) { r.Student.SHA256 = "not-a-digest" },
		"budget steps":      func(r *pipeline.Recipe) { r.Stages[0].MaxSteps = 0 },
		"budget tokens":     func(r *pipeline.Recipe) { r.Stages[0].MaxTokens = 1 },
		"budget timeout":    func(r *pipeline.Recipe) { r.Stages[0].TimeoutSeconds = 86401 },
		"stage identifier":  func(r *pipeline.Recipe) { r.Stages[0].ID = "../sft" },
		"recipe identifier": func(r *pipeline.Recipe) { r.ID = "" },
	}
	for name, mutate := range cases {
		t.Run(name, func(t *testing.T) {
			r := sftRecipe()
			mutate(&r)
			if err := r.Validate(); err == nil {
				t.Fatal("admitted")
			}
			if _, err := r.Digest(); err == nil {
				t.Fatal("digested an invalid recipe")
			}
		})
	}
}

// The new scope must not be reachable from the older schemas, and the older
// schemas must not learn to stop early because a new one exists.
func TestSFTScopeIsNotReachableFromV1OrV2(t *testing.T) {
	v2 := sftRecipe()
	v2.Version = 2
	if err := v2.Validate(); err == nil {
		t.Fatal("v2 admitted the sft scope")
	}
	v1 := sftRecipe()
	v1.Version, v1.Scope = 1, ""
	if err := v1.Validate(); err == nil {
		t.Fatal("v1 admitted a single SFT stage")
	}
	master := masterRecipe()
	master.Stages = master.Stages[:1]
	if err := master.Validate(); err == nil {
		t.Fatal("v2 Master scope stopped after SFT")
	}
	for _, r := range []pipeline.Recipe{recipe(), masterRecipe()} {
		if err := r.Validate(); err != nil {
			t.Fatal("an existing schema stopped validating", r.Version, err)
		}
	}
}

// V2 digests are frozen. This independent wire definition fixes what v2
// serialises to, so a change to the shared Recipe type shows here and not only
// as a digest that quietly moved.
type masterWire struct {
	Version     int                    `json:"schema_version"`
	Scope       string                 `json:"scope,omitempty"`
	ID          string                 `json:"id"`
	Method      string                 `json:"method"`
	Student     pipeline.ArtifactRef   `json:"student"`
	Teachers    []pipeline.ArtifactRef `json:"teachers"`
	Training    pipeline.ArtifactRef   `json:"training"`
	Recovery    pipeline.ArtifactRef   `json:"recovery"`
	Calibration pipeline.ArtifactRef   `json:"calibration"`
	Protection  pipeline.ArtifactRef   `json:"protection"`
	Heldout     pipeline.ArtifactRef   `json:"heldout"`
	Stages      []pipeline.Stage       `json:"stages"`
}

func TestSFTScopeLeavesTheV2MasterDigestUnchanged(t *testing.T) {
	r := masterRecipe()
	wire, err := json.Marshal(masterWire{r.Version, r.Scope, r.ID, r.Method, r.Student, r.Teachers, r.Training, r.Recovery, r.Calibration, r.Protection, r.Heldout, r.Stages})
	if err != nil {
		t.Fatal(err)
	}
	sum := sha256.Sum256(wire)
	got, err := r.Digest()
	if err != nil || got != hex.EncodeToString(sum[:]) {
		t.Fatal("v2 Master digest changed", got, err)
	}
}
