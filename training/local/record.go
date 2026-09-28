package local

type example struct {
	ID           string  `json:"id"`
	InputIDs     []int64 `json:"input_ids"`
	Labels       []int64 `json:"labels"`
	PromptTokens int     `json:"prompt_tokens"`
}

// stepManifest is what a step writes beside its adapter and moments. The
// distillation fields are set only by a step whose alpha is positive, so an
// SFT step writes the same bytes it always did. LossBefore stays the hard
// completion NLL either way, comparable with an SFT step's; a distillation
// step takes it from FusionCompletionResult.HardLoss, which accumulates the
// same NLL in another order, so it equals MeasureLosses up to rounding rather
// than bit for bit. DistillationLossBefore is the blended objective the step
// minimized, and the maps are FusionCompletionResult's per-teacher readings.
type stepManifest struct {
	RecipeSHA256           string             `json:"recipe_sha256"`
	Step                   uint64             `json:"step"`
	ExampleID              string             `json:"example_id"`
	SupervisedTokens       int                `json:"supervised_tokens"`
	LossBefore             float64            `json:"loss_before"`
	UpdatedAdapterSHA      string             `json:"updated_adapter_sha256"`
	AdapterFileSHA         string             `json:"adapter_file_sha256"`
	OptimizerFileSHA       string             `json:"optimizer_file_sha256"`
	LogitsAfterSHA         string             `json:"logits_after_sha256"`
	BaseRevision           string             `json:"base_revision"`
	DistillationLossBefore *float64           `json:"distillation_loss_before,omitempty"`
	TeacherMass            map[string]float64 `json:"teacher_mass,omitempty"`
	TeacherLosses          map[string]float64 `json:"teacher_losses,omitempty"`
}
