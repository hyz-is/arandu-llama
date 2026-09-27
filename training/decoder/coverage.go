package decoder

import (
	"fmt"

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
