package layers

import (
	"errors"

	"github.com/tayi-ai/arandu-llama/training/torch"
)

// LoRATarget identifies one adaptable projection of a decoder block. The
// constants are declared in canonical parameter order: the block's attention
// projections in checkpoint registration order, then its MLP projections.
type LoRATarget int

// Every adaptable projection. Full-attention targets belong to AttentionLoRA
// directly, recurrent targets to its Linear pairs and MLP targets to its
// FeedForward pairs.
const (
	LoRAQuery LoRATarget = iota
	LoRAKey
	LoRAValue
	LoRAOutput
	LoRALinearOutput
	LoRALinearQKV
	LoRALinearZ
	LoRALinearBeta
	LoRALinearAlpha
	LoRAGate
	LoRAUp
	LoRADown
)

// LoRATargets lists every target in canonical order.
func LoRATargets() []LoRATarget {
	targets := make([]LoRATarget, 0, LoRADown+1)
	for target := LoRAQuery; target <= LoRADown; target++ {
		targets = append(targets, target)
	}
	return targets
}

// FullAttention reports a projection of the full-attention block.
func (t LoRATarget) FullAttention() bool { return t >= LoRAQuery && t <= LoRAOutput }

// LinearAttention reports a projection of the recurrent block.
func (t LoRATarget) LinearAttention() bool { return t >= LoRALinearOutput && t <= LoRALinearAlpha }

// FeedForward reports a projection of the MLP.
func (t LoRATarget) FeedForward() bool { return t >= LoRAGate && t <= LoRADown }

// LinearAttentionLoRA holds optional trainable pairs of a recurrent block.
// Field XA/XB adapts LinearAttentionWeights.X; a nil pair leaves it frozen.
type LinearAttentionLoRA struct {
	QKVA, QKVB, ZA, ZB, BetaA, BetaB, AlphaA, AlphaB, OutputA, OutputB *torch.Tensor
}

// FeedForwardLoRA holds optional trainable pairs of the SwiGLU projections.
// Field XA/XB adapts DecoderWeights.X; a nil pair leaves it frozen.
type FeedForwardLoRA struct {
	GateA, GateB, UpA, UpB, DownA, DownB *torch.Tensor
}

// Pair returns the locations of target's A and B tensors, or nil for a target
// outside this block kind, so a caller can read or replace one pair.
func (a *AttentionLoRA) Pair(target LoRATarget) (A, B **torch.Tensor) {
	return pairOf(target, [4][2]**torch.Tensor{{&a.QueryA, &a.QueryB}, {&a.KeyA, &a.KeyB}, {&a.ValueA, &a.ValueB}, {&a.OutputA, &a.OutputB}}, &a.Linear, &a.FeedForward)
}

// Pair returns the locations of the cotangents of target's A and B tensors.
func (g *DecoderGradients) Pair(target LoRATarget) (A, B **torch.Tensor) {
	return pairOf(target, [4][2]**torch.Tensor{{&g.QueryA, &g.QueryB}, {&g.KeyA, &g.KeyB}, {&g.ValueA, &g.ValueB}, {&g.OutputA, &g.OutputB}}, &g.Linear, &g.FeedForward)
}

// Targets lists the adapted projections in canonical order. A half-declared
// pair is reported as present so validation can refuse it.
func (a *AttentionLoRA) Targets() []LoRATarget {
	if a == nil {
		return nil
	}
	var targets []LoRATarget
	for _, target := range LoRATargets() {
		if x, y := a.Pair(target); *x != nil || *y != nil {
			targets = append(targets, target)
		}
	}
	return targets
}

func pairOf(target LoRATarget, full [4][2]**torch.Tensor, linear *LinearAttentionLoRA, feedForward *FeedForwardLoRA) (**torch.Tensor, **torch.Tensor) {
	switch target {
	case LoRAQuery, LoRAKey, LoRAValue, LoRAOutput:
		return full[target][0], full[target][1]
	case LoRALinearOutput:
		return &linear.OutputA, &linear.OutputB
	case LoRALinearQKV:
		return &linear.QKVA, &linear.QKVB
	case LoRALinearZ:
		return &linear.ZA, &linear.ZB
	case LoRALinearBeta:
		return &linear.BetaA, &linear.BetaB
	case LoRALinearAlpha:
		return &linear.AlphaA, &linear.AlphaB
	case LoRAGate:
		return &feedForward.GateA, &feedForward.GateB
	case LoRAUp:
		return &feedForward.UpA, &feedForward.UpB
	case LoRADown:
		return &feedForward.DownA, &feedForward.DownB
	}
	return nil, nil
}

// validate admits an adapter for one block kind: a positive finite alpha, at
// least one complete pair, only pairs of that block's projections, and Float32
// matrices on the block's device. Shapes are checked against each base weight
// where the pair is applied.
func (a *AttentionLoRA) validate(full bool, device torch.Device) error {
	if !positiveFinite(a.Alpha) {
		return errors.New("layers: LoRA alpha must be finite and positive")
	}
	targets := a.Targets()
	if len(targets) == 0 {
		return errors.New("layers: an adapter must adapt at least one projection")
	}
	for _, target := range targets {
		if full && target.LinearAttention() || !full && target.FullAttention() {
			return errors.New("layers: LoRA pair does not belong to this attention variant")
		}
		x, y := a.Pair(target)
		for _, value := range []*torch.Tensor{*x, *y} {
			if value == nil {
				return errors.New("layers: LoRA pair requires both A and B")
			}
			info, err := value.Info()
			if err != nil {
				return err
			}
			if len(info.Shape) != 2 || info.DType != torch.Float32 || info.Device != device {
				return errors.New("layers: LoRA pairs must be Float32 matrices on the block device")
			}
		}
	}
	return nil
}

// project applies one optionally adapted projection. A nil pair runs the
// frozen Linear exactly as an unadapted block does.
func project(x, weight, a, b *torch.Tensor, alpha float64) (*torch.Tensor, error) {
	if a == nil && b == nil {
		return Linear(x, weight)
	}
	return LoRALinear(x, weight, a, b, alpha)
}
