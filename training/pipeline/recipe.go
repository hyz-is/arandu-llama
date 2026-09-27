package pipeline

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"io"
	"slices"
)

// Phase is a scientific operation, never an executable command.
type Phase string

const (
	PhaseSFT             Phase = "sft"
	PhaseTeacherCache    Phase = "teacher_cache"
	PhaseAlignment       Phase = "feature_alignment"
	PhaseFusion          Phase = "constrained_fusion"
	PhaseRecovery        Phase = "recovery_protection"
	PhaseMaster          Phase = "master_evaluation"
	PhaseCalibration     Phase = "calibration"
	PhaseQuantize        Phase = "quantization"
	PhaseVariantRecovery Phase = "variant_recovery_protection"
)

// ArtifactRef names immutable data in an installation's artifact store. It is
// resolved there, not interpreted as a filesystem path or a remote URL.
type ArtifactRef struct {
	ID     string `json:"id"`
	SHA256 string `json:"sha256"`
}

// Stage describes a bounded phase and the exact inputs its backend must admit.
type Stage struct {
	ID             string        `json:"id"`
	Phase          Phase         `json:"phase"`
	Inputs         []ArtifactRef `json:"inputs"`
	MaxSteps       int           `json:"max_steps"`
	MaxTokens      int           `json:"max_tokens"`
	TimeoutSeconds int           `json:"timeout_seconds"`
	Format         string        `json:"format,omitempty"`
	ParentStage    string        `json:"parent_stage,omitempty"`
}

// Recipe is the frozen, backend-independent protocol. Capabilities are checked
// against the selected executor before queueing; a valid recipe is not itself
// evidence that its model, projections or quantization have been qualified.
type Recipe struct {
	Version int `json:"schema_version"`
	// Scope is empty for v1's full method. V2 requires "master" and ends only
	// after the six complete phases through Master evaluation, before variants.
	// V3 requires "sft": one causal SFT stage, student and training data only.
	Scope    string        `json:"scope,omitempty"`
	ID       string        `json:"id"`
	Method   string        `json:"method"`
	Student  ArtifactRef   `json:"student"`
	Teachers []ArtifactRef `json:"teachers"`
	Training ArtifactRef   `json:"training"`
	Recovery ArtifactRef   `json:"recovery"`
	// Calibration may be the zero reference only in v2 Master scope. If pinned,
	// it remains separate from every other dataset and may feed alignment only.
	// It neither replaces final Heldout nor declares a calibration phase complete.
	Calibration ArtifactRef `json:"calibration"`
	Protection  ArtifactRef `json:"protection"`
	Heldout     ArtifactRef `json:"heldout"`
	Stages      []Stage     `json:"stages"`
}

// Validate refuses incomplete declared deliveries, reordering and mixed roles.
// V1 requires Master and all variants. V2 Master scope does not certify any
// calibration or quantized variant, and cannot reuse a v1 recipe identity.
// V3 SFT scope is preparatory and certifies nothing beyond its own stage.
func (r Recipe) Validate() error {
	if r.Version == 3 {
		return r.validateSFT()
	}
	master := r.Version == 2 && r.Scope == "master"
	full := r.Version == 1 && r.Scope == ""
	if (!master && !full) || !identifier(r.ID) || r.Method != "generational-fusion-v1" || len(r.Teachers) < 1 || len(r.Teachers) > 8 || (full && len(r.Stages) != 13) || (master && len(r.Stages) != 6) {
		return errors.New("training recipe: invalid version, method or stage count")
	}
	data := []ArtifactRef{r.Training, r.Recovery}
	if !master || r.Calibration != (ArtifactRef{}) {
		data = append(data, r.Calibration)
	}
	data = append(data, r.Protection, r.Heldout)
	artifacts := append([]ArtifactRef{r.Student}, data...)
	artifacts = append(artifacts, r.Teachers...)
	seenArtifacts := map[string]string{}
	for _, ref := range artifacts {
		if !identifier(ref.ID) || !digest(ref.SHA256) || seenArtifacts[ref.ID] != "" {
			return errors.New("training recipe: invalid or repeated artifact identity")
		}
		seenArtifacts[ref.ID] = ref.SHA256
	}
	dataDigests := map[string]bool{}
	for _, ref := range data {
		if dataDigests[ref.SHA256] {
			return errors.New("training recipe: data roles share an artifact")
		}
		dataDigests[ref.SHA256] = true
	}
	order := []Phase{PhaseSFT, PhaseTeacherCache, PhaseAlignment, PhaseFusion, PhaseRecovery, PhaseMaster}
	if full {
		order = append(order, PhaseCalibration)
	}
	index := 0
	seenStages := map[string]bool{}
	formats := map[string]bool{}
	for i, stage := range r.Stages {
		if !stageBudget(stage) || seenStages[stage.ID] {
			return errors.New("training recipe: stage identity or budget invalid")
		}
		seenStages[stage.ID] = true
		inputs := map[string]bool{}
		for _, ref := range stage.Inputs {
			if !identifier(ref.ID) || !digest(ref.SHA256) || inputs[ref.ID] || seenArtifacts[ref.ID] != ref.SHA256 {
				return errors.New("training recipe: stage input identity invalid or unregistered")
			}
			inputs[ref.ID] = true
		}
		if inputs[r.Heldout.ID] && stage.Phase != PhaseMaster {
			return errors.New("training recipe: heldout data cannot feed adaptation")
		}
		if master && r.Calibration != (ArtifactRef{}) && inputs[r.Calibration.ID] && stage.Phase != PhaseAlignment {
			return errors.New("training recipe: optional Master calibration data can feed alignment only")
		}
		if inputs[r.Protection.ID] && stage.Phase != PhaseFusion && stage.Phase != PhaseRecovery && stage.Phase != PhaseVariantRecovery && stage.Phase != PhaseMaster {
			return errors.New("training recipe: protection data cannot feed SFT, teacher cache or alignment")
		}
		if (stage.Phase == PhaseFusion || stage.Phase == PhaseRecovery || stage.Phase == PhaseVariantRecovery) && !inputs[r.Protection.ID] {
			return errors.New("training recipe: constrained fusion requires protection")
		}
		parent := ""
		if i > 0 {
			parent = r.Stages[i-1].ID
		}
		if stage.Phase == PhaseQuantize {
			parent = r.Stages[5].ID
		}
		if stage.ParentStage != parent {
			return errors.New("training recipe: stage lineage differs")
		}
		required := r.Training.ID
		switch stage.Phase {
		case PhaseRecovery, PhaseVariantRecovery:
			required = r.Recovery.ID
		case PhaseMaster:
			required = r.Heldout.ID
		case PhaseCalibration, PhaseQuantize:
			required = r.Calibration.ID
		}
		if !inputs[required] {
			return errors.New("training recipe: stage data role missing")
		}
		if index < len(order) {
			if stage.Phase != order[index] || stage.Format != "" {
				return errors.New("training recipe: required scientific phase missing or reordered")
			}
			index++
			continue
		}
		if (i-len(order))%2 == 0 {
			if stage.Phase != PhaseQuantize || !slices.Contains([]string{"Q8_0", "Q6_K", "Q4_K_M"}, stage.Format) || formats[stage.Format] {
				return errors.New("training recipe: quantization must derive independently from approved master")
			}
			formats[stage.Format] = true
		} else if stage.Phase != PhaseVariantRecovery || stage.Format != r.Stages[i-1].Format {
			return errors.New("training recipe: each precision requires recovery and protection")
		}
	}
	if (len(r.Stages)-len(order))%2 != 0 {
		return errors.New("training recipe: missing variant protection")
	}
	return nil
}

// validateSFT admits the preparatory delivery that trains the student on its
// own training data and nothing else. It is a separate schema rather than a
// shortened v2 because a Master scope that could stop after SFT would let a
// partial run be read as a Master: here the absence of teachers, heldout,
// recovery, protection and calibration is required, not merely tolerated, so
// the recipe itself says what the result is not.
func (r Recipe) validateSFT() error {
	if r.Scope != "sft" || !identifier(r.ID) || r.Method != "causal-sft-v1" || len(r.Teachers) != 0 || len(r.Stages) != 1 ||
		r.Recovery != (ArtifactRef{}) || r.Calibration != (ArtifactRef{}) || r.Protection != (ArtifactRef{}) || r.Heldout != (ArtifactRef{}) {
		return errors.New("training recipe: invalid sft scope, method or roles")
	}
	registered := map[string]string{}
	for _, ref := range []ArtifactRef{r.Student, r.Training} {
		if !identifier(ref.ID) || !digest(ref.SHA256) || registered[ref.ID] != "" {
			return errors.New("training recipe: invalid or repeated artifact identity")
		}
		registered[ref.ID] = ref.SHA256
	}
	stage := r.Stages[0]
	if !stageBudget(stage) {
		return errors.New("training recipe: stage identity or budget invalid")
	}
	if stage.Phase != PhaseSFT || stage.Format != "" || stage.ParentStage != "" {
		return errors.New("training recipe: sft scope admits exactly one causal SFT stage")
	}
	inputs := map[string]bool{}
	for _, ref := range stage.Inputs {
		if !identifier(ref.ID) || !digest(ref.SHA256) || inputs[ref.ID] || registered[ref.ID] != ref.SHA256 {
			return errors.New("training recipe: stage input identity invalid or unregistered")
		}
		inputs[ref.ID] = true
	}
	if !inputs[r.Training.ID] {
		return errors.New("training recipe: stage data role missing")
	}
	return nil
}

func stageBudget(s Stage) bool {
	return identifier(s.ID) && s.MaxSteps >= 1 && s.MaxSteps <= 1_000_000 && s.MaxTokens >= 2 && s.MaxTokens <= 1_048_576 &&
		s.TimeoutSeconds >= 1 && s.TimeoutSeconds <= 86400 && len(s.Inputs) > 0 && len(s.Inputs) <= 64
}

// Digest hashes the canonical typed protocol after validation.
func (r Recipe) Digest() (string, error) {
	if err := r.Validate(); err != nil {
		return "", err
	}
	if r.Version == 3 {
		// V3 has no teachers, and "absent" and "empty" must name one recipe.
		// Only v3 is normalised: v1 and v2 digests are frozen as written.
		r.Teachers = nil
	}
	body, err := json.Marshal(r)
	if err != nil {
		return "", err
	}
	sum := sha256.Sum256(body)
	return hex.EncodeToString(sum[:]), nil
}

// DecodeRecipe checks the exact artifact digest and refuses unknown fields.
func DecodeRecipe(body []byte, expected string) (Recipe, error) {
	var r Recipe
	if len(body) == 0 || len(body) > 1<<20 || !digest(expected) {
		return r, errors.New("training recipe: invalid artifact bounds")
	}
	sum := sha256.Sum256(body)
	if hex.EncodeToString(sum[:]) != expected {
		return r, errors.New("training recipe: artifact hash differs")
	}
	d := json.NewDecoder(bytes.NewReader(body))
	d.DisallowUnknownFields()
	if d.Decode(&r) != nil || d.Decode(new(any)) != io.EOF {
		return Recipe{}, errors.New("training recipe: invalid typed document")
	}
	if r.Version == 1 {
		// Scope was not a v1 wire field. Preserve its historical rejection even
		// when a caller explicitly supplies the new field as empty or null.
		var fields struct {
			Scope json.RawMessage `json:"scope"`
		}
		if json.Unmarshal(body, &fields) != nil || fields.Scope != nil {
			return Recipe{}, errors.New("training recipe: scope is not a v1 field")
		}
	}
	return r, r.Validate()
}

func identifier(v string) bool {
	if len(v) < 1 || len(v) > 128 {
		return false
	}
	for _, c := range v {
		if !(c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || c >= '0' && c <= '9' || c == '-' || c == '_' || c == '.') {
			return false
		}
	}
	return v != "." && v != ".."
}

func digest(v string) bool {
	if len(v) != 64 {
		return false
	}
	b, e := hex.DecodeString(v)
	return e == nil && hex.EncodeToString(b) == v
}
