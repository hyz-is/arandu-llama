package decoder

import (
	"fmt"
	"slices"
	"strings"

	"github.com/tayi-ai/arandu-llama/training/layers"
	"github.com/tayi-ai/arandu-llama/training/torch"
)

// adapterModules names each target's checkpoint module relative to its layer.
var adapterModules = [...]string{
	layers.LoRAQuery:        "self_attn.q_proj",
	layers.LoRAKey:          "self_attn.k_proj",
	layers.LoRAValue:        "self_attn.v_proj",
	layers.LoRAOutput:       "self_attn.o_proj",
	layers.LoRALinearOutput: "linear_attn.out_proj",
	layers.LoRALinearQKV:    "linear_attn.in_proj_qkv",
	layers.LoRALinearZ:      "linear_attn.in_proj_z",
	layers.LoRALinearBeta:   "linear_attn.in_proj_b",
	layers.LoRALinearAlpha:  "linear_attn.in_proj_a",
	layers.LoRAGate:         "mlp.gate_proj",
	layers.LoRAUp:           "mlp.up_proj",
	layers.LoRADown:         "mlp.down_proj",
}

// adapterSlot is one adapted tensor of a model: its layer, target and letter.
type adapterSlot struct {
	name   string
	layer  int
	target layers.LoRATarget
	b      bool
	value  *torch.Tensor
}

// adapterRegistry lists every adapted tensor in parameter order: layer, then
// canonical target order, then A before B. Names follow the PEFT namespace.
func adapterRegistry(model *TextModel) []adapterSlot {
	var result []adapterSlot
	for index, layer := range model.Layers {
		if layer.Adapter == nil {
			continue
		}
		prefix := fmt.Sprintf("base_model.model.model.language_model.layers.%d.", index)
		for _, target := range layer.Adapter.Targets() {
			a, b := layer.Adapter.Pair(target)
			module := prefix + adapterModules[target]
			result = append(result,
				adapterSlot{name: module + ".lora_A.default.weight", layer: index, target: target, value: *a},
				adapterSlot{name: module + ".lora_B.default.weight", layer: index, target: target, b: true, value: *b})
		}
	}
	return result
}

// AdapterCoverageVersion is the only admitted AdapterCoverage schema.
const AdapterCoverageVersion = 1

// AdapterCoverage declares which frozen projections carry a trainable LoRA
// pair, by layer kind. Each list names checkpoint modules in canonical order
// without repetition: FullAttention admits q_proj, k_proj, v_proj, o_proj;
// LinearAttention admits out_proj, in_proj_qkv, in_proj_z, in_proj_b,
// in_proj_a; MLP admits gate_proj, up_proj, down_proj on every layer. That is
// the checkpoint's registration order, and therefore the order of reference
// rows, initializer projections, parameters and gradients. Every pair uses
// AssemblyLimits' AdapterRank and AdapterAlpha.
//
// A nil coverage is q_proj and v_proj of every full-attention layer, with
// every plan, reference name, parameter order and digest that predates this
// declaration. MTP and vision modules are outside the text forward and are
// never covered.
type AdapterCoverage struct {
	Version         int      `json:"version"`
	FullAttention   []string `json:"full_attention,omitempty"`
	LinearAttention []string `json:"linear_attention,omitempty"`
	MLP             []string `json:"mlp,omitempty"`
}

// FullAdapterCoverage declares every projection of every text layer.
func FullAdapterCoverage() *AdapterCoverage {
	return &AdapterCoverage{Version: AdapterCoverageVersion,
		FullAttention:   []string{"q_proj", "k_proj", "v_proj", "o_proj"},
		LinearAttention: []string{"out_proj", "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"},
		MLP:             []string{"gate_proj", "up_proj", "down_proj"}}
}

// coverageLists pairs each declared list with its module prefix and the
// targets it may name, in canonical order.
func (c *AdapterCoverage) coverageLists() []struct {
	modules []string
	prefix  string
	kind    func(layers.LoRATarget) bool
} {
	return []struct {
		modules []string
		prefix  string
		kind    func(layers.LoRATarget) bool
	}{
		{c.FullAttention, "self_attn.", layers.LoRATarget.FullAttention},
		{c.LinearAttention, "linear_attn.", layers.LoRATarget.LinearAttention},
		{c.MLP, "mlp.", layers.LoRATarget.FeedForward},
	}
}

// Validate refuses another version, an unknown, repeated or out-of-order
// module and a declaration that adapts nothing. A nil coverage is valid.
func (c *AdapterCoverage) Validate() error {
	if c == nil {
		return nil
	}
	if c.Version != AdapterCoverageVersion {
		return fmt.Errorf("%w: adapter coverage version %d is not admitted", ErrAssembly, c.Version)
	}
	declared := 0
	for _, list := range c.coverageLists() {
		next := layers.LoRAQuery
		for _, module := range list.modules {
			found := false
			for target := next; target <= layers.LoRADown; target++ {
				if list.kind(target) && adapterModules[target] == list.prefix+module {
					next, found = target+1, true
					break
				}
			}
			if !found {
				return fmt.Errorf("%w: adapter coverage module %q is unknown, repeated or out of canonical order", ErrAssembly, module)
			}
			declared++
		}
	}
	if declared == 0 {
		return fmt.Errorf("%w: adapter coverage declares no module", ErrAssembly)
	}
	return nil
}

// targets lists the adapted targets of a layer kind in canonical order:
// its attention targets, then the MLP targets every layer shares.
func (c *AdapterCoverage) targets(layerType string) []layers.LoRATarget {
	if c == nil {
		if layerType == "full_attention" {
			return []layers.LoRATarget{layers.LoRAQuery, layers.LoRAValue}
		}
		return nil
	}
	var result []layers.LoRATarget
	for _, list := range c.coverageLists() {
		if list.prefix == "self_attn." && layerType != "full_attention" || list.prefix == "linear_attn." && layerType != "linear_attention" {
			continue
		}
		for _, module := range list.modules {
			for _, target := range layers.LoRATargets() {
				if list.kind(target) && adapterModules[target] == list.prefix+module {
					result = append(result, target)
				}
			}
		}
	}
	return result
}

func (c *AdapterCoverage) clone() *AdapterCoverage {
	if c == nil {
		return nil
	}
	return &AdapterCoverage{Version: c.Version, FullAttention: slices.Clone(c.FullAttention),
		LinearAttention: slices.Clone(c.LinearAttention), MLP: slices.Clone(c.MLP)}
}

// CoverageProjections derives the initializer projections that coverage
// implies for configJSON at rank, in the order PlanTextAssembly admits the
// adapters: the Projections of the InitialAdapterSpec for that plan.
func CoverageProjections(configJSON []byte, coverage *AdapterCoverage, rank int64) ([]InitialProjection, error) {
	geometry, err := validateAssemblyConfig(configJSON)
	if err != nil {
		return nil, err
	}
	if err := coverage.Validate(); err != nil {
		return nil, err
	}
	if rank < 1 || rank > 1<<20 {
		return nil, fmt.Errorf("%w: adapter rank outside bounds", ErrAssembly)
	}
	_, adapters := assemblyGeometry(geometry, AssemblyLimits{AdapterRank: rank, AdapterCoverage: coverage, DeviceByLayer: make([]int, geometry.Layers)})
	result := make([]InitialProjection, 0, len(adapters)/2)
	for i := 0; i+1 < len(adapters); i += 2 {
		a, b := adapters[i], adapters[i+1]
		result = append(result, InitialProjection{Name: strings.TrimSuffix(a.ReferenceName, ".lora_A.default.weight"), Input: a.Shape[1], Output: b.Shape[0], Rank: a.Shape[0]})
	}
	return result, nil
}
