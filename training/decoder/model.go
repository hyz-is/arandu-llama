// Package decoder composes checkpointed text-model calculations from native
// primitives. It does not admit resources, control fleet jobs or qualify a model.
package decoder

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"runtime"

	"github.com/tayi-ai/arandu-llama/training/layers"
	"github.com/tayi-ai/arandu-llama/training/torch"
)

// Layer describes one decoder and its placement. Cosine and sine are the
// checkpoint-specific text rotary values on this layer's device, or nil for a
// recurrent layer. All values must remain unchanged between Forward and VJP.
type Layer struct {
	Weights      layers.DecoderWeights
	Adapter      *layers.AttentionLoRA
	Config       layers.DecoderConfig
	Device       torch.Device
	Cosine, Sine *torch.Tensor
}

// TextModel borrows a frozen token embedding, decoder stack and output head.
// The loader owns all weights and adapters. This calculation can represent tiny
// qualification fixtures; its shape is not proof of Decoder identity.
// It is not goroutine-safe: the caller must serialize Forward, VJP, parameter
// replacement and Close, and must not mutate borrowed tensors during a call.
type TextModel struct {
	Embedding, FinalNorm, Head *torch.Tensor
	Layers                     []Layer
	Epsilon                    float64
	generation                 uint64
}

// ErrStaleSnapshot means parameter replacement invalidated a forward snapshot.
var ErrStaleSnapshot = errors.New("decoder: forward snapshot uses an older parameter generation")

// Limits bounds persistent activation payload, token count and output rows.
// MaxCheckpointBytes covers decoder boundaries plus one transfer buffer only;
// it excludes weights, per-layer temporary graphs and native allocator overhead.
// Those require separate admission and measured phase guards.
type Limits struct {
	MaxTokens, LogitRows int64
	MaxCheckpointBytes   int64
}

// Snapshot owns detached activations at decoder boundaries and detached logits.
// Logits contain only the requested final token rows, before softmax, in Float32.
// Do not mutate or close these tensors until VJP completes.
type Snapshot struct {
	Logits     *torch.Tensor
	states     []*torch.Tensor
	owner      *TextModel
	limits     Limits
	generation uint64
}

// Close releases all activations and logits. Model weights remain caller-owned.
func (s *Snapshot) Close() error {
	if s == nil {
		return nil
	}
	var failures []error
	for _, value := range s.states {
		failures = append(failures, value.Close())
	}
	failures = append(failures, s.Logits.Close())
	s.states = nil
	s.owner = nil
	return errors.Join(failures...)
}

// ParameterGradient owns one adapter gradient. Names use the frozen PEFT
// parameter naming convention; order follows decoder, projection, then A/B.
type ParameterGradient struct {
	Name  string
	Value *torch.Tensor
}

// Gradients owns only trainable adapter gradients; no base gradients are stored.
type Gradients []ParameterGradient

// Close releases all owned parameter gradient handles.
func (g Gradients) Close() error {
	var failures []error
	for _, parameter := range g {
		failures = append(failures, parameter.Value.Close())
	}
	return errors.Join(failures...)
}

// Forward processes explicit token IDs without a cache or an implicit prompt
// template. Each decoder output is detached; only boundary activations survive.
// The caller supplies the exact tokenizer output and any appended answer token.
func (m *TextModel) Forward(ctx context.Context, tokenIDs []int64, limits Limits) (_ *Snapshot, err error) {
	return m.ForwardObserved(ctx, tokenIDs, limits, nil)
}

// ForwardObserved performs Forward with optional synchronous observations of
// copied activation statistics. A nil observer adds no tensor reads. Observing
// synchronizes native copies and can change timing; it does not change tensor
// values, precision, model parameters or allocation admission. Observer errors
// stop the forward and release its intermediate handles.
func (m *TextModel) ForwardObserved(ctx context.Context, tokenIDs []int64, limits Limits, observer ForwardObserver) (_ *Snapshot, err error) {
	info, err := m.validate(ctx, tokenIDs, limits)
	if err != nil {
		return nil, err
	}
	rawIDs := make([]byte, len(tokenIDs)*8)
	for index, token := range tokenIDs {
		binary.LittleEndian.PutUint64(rawIDs[index*8:], uint64(token))
	}
	ids, err := torch.FromBytes(rawIDs, []int64{int64(len(tokenIDs))}, torch.Int64, info.Device, false)
	if err != nil {
		return nil, err
	}
	defer ids.Close()
	rows, err := m.Embedding.IndexSelect(0, ids)
	if err != nil {
		return nil, err
	}
	current, err := rows.Reshape([]int64{1, int64(len(tokenIDs)), info.Shape[1]})
	_ = rows.Close()
	if err != nil {
		return nil, err
	}
	defer func() { _ = current.Close() }()
	snapshot := &Snapshot{owner: m, limits: limits, generation: m.generation}
	defer func() {
		if err != nil {
			_ = snapshot.Close()
		}
	}()
	if err = observeForward(ctx, observer, StageEmbedding, -1, current); err != nil {
		return nil, err
	}
	for index, layer := range m.Layers {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if err = observeForward(ctx, observer, StageBeforePlacement, index, current); err != nil {
			return nil, err
		}
		placed, err := current.To(layer.Device, info.DType)
		if err != nil {
			return nil, fmt.Errorf("decoder: layer %d input placement: %w", index, err)
		}
		_ = current.Close()
		snapshot.states = append(snapshot.states, placed)
		if err = observeForward(ctx, observer, StageAfterPlacement, index, placed); err != nil {
			return nil, err
		}
		current, err = layers.DecoderForward(ctx, placed, layer.Weights, layer.Adapter, layer.Cosine, layer.Sine, layer.Config)
		if err != nil {
			return nil, fmt.Errorf("decoder: layer %d forward: %w", index, err)
		}
		if err = observeForward(ctx, observer, StageAfterDecoder, index, current); err != nil {
			return nil, err
		}
	}
	final, err := current.Detach()
	if err != nil {
		return nil, err
	}
	snapshot.states = append(snapshot.states, final)
	logits, err := m.headObserved(ctx, final, limits.LogitRows, observer)
	if err != nil {
		return nil, err
	}
	defer logits.Close()
	snapshot.Logits, err = logits.Detach()
	if err != nil {
		return nil, err
	}
	if err = observeForward(ctx, observer, StageLogits, -1, snapshot.Logits); err != nil {
		return nil, err
	}
	finite, err := snapshot.Logits.AllFinite()
	if err != nil || !finite {
		return nil, errors.Join(errors.New("decoder: non-finite forward logits"), err)
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return snapshot, nil
}

// VJP recomputes one decoder at a time in reverse order and returns all LoRA
// gradients in parameter order. It traverses every frozen recurrent block
// between adapters. The loss and optimizer are deliberately outside this API.
func (m *TextModel) VJP(ctx context.Context, snapshot *Snapshot, logitCotangent *torch.Tensor) (_ Gradients, err error) {
	return m.vjp(ctx, snapshot, logitCotangent, nil)
}

func (m *TextModel) validateSnapshot(ctx context.Context, snapshot *Snapshot, logitCotangent *torch.Tensor) error {
	if ctx == nil || m == nil || snapshot == nil || snapshot.owner != m || len(snapshot.states) != len(m.Layers)+1 || logitCotangent == nil {
		return errors.New("decoder: VJP requires this model's live forward snapshot and cotangent")
	}
	if snapshot.generation != m.generation {
		return ErrStaleSnapshot
	}
	return ctx.Err()
}

func (m *TextModel) vjp(ctx context.Context, snapshot *Snapshot, logitCotangent *torch.Tensor, features map[int][]featureCotangent) (_ Gradients, err error) {
	// The head graph is built and differentiated here; like every decoder VJP
	// it stays on one OS thread, so its gradient order is the creation order.
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	if err := m.validateSnapshot(ctx, snapshot, logitCotangent); err != nil {
		return nil, err
	}
	firstTrainable := -1
	for index, layer := range m.Layers {
		if layer.Adapter != nil {
			firstTrainable = index
			break
		}
	}
	if firstTrainable < 0 {
		return nil, errors.New("decoder: model has no trainable adapters")
	}
	last, err := snapshot.states[len(m.Layers)].Detach()
	if err != nil {
		return nil, err
	}
	defer last.Close()
	leaf, err := last.SetRequiresGrad(true)
	if err != nil {
		return nil, err
	}
	defer leaf.Close()
	logits, err := m.head(leaf, snapshot.limits.LogitRows)
	if err != nil {
		return nil, err
	}
	defer logits.Close()
	dInfo, err := logitCotangent.Info()
	if err != nil {
		return nil, err
	}
	lInfo, err := logits.Info()
	if err != nil {
		return nil, err
	}
	if dInfo.DType != torch.Float32 || dInfo.Device != lInfo.Device || !sameShape(dInfo.Shape, lInfo.Shape) {
		return nil, errors.New("decoder: logit cotangent geometry, dtype or placement differs")
	}
	finite, err := logitCotangent.AllFinite()
	if err != nil || !finite {
		return nil, errors.Join(errors.New("decoder: non-finite logit cotangent"), err)
	}
	headGradients, err := torch.Grad([]*torch.Tensor{logits}, []*torch.Tensor{leaf}, []*torch.Tensor{logitCotangent}, false, false)
	if err != nil {
		return nil, err
	}
	current := headGradients[0]
	defer func() { _ = current.Close() }()
	byLayer := make([]*layers.DecoderGradients, len(m.Layers))
	defer func() {
		for _, gradient := range byLayer {
			_ = gradient.Close()
		}
	}()
	for index := len(m.Layers) - 1; index >= firstTrainable; index-- {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		layer := m.Layers[index]
		state := snapshot.states[index]
		stateInfo, err := state.Info()
		if err != nil {
			return nil, err
		}
		placed, err := current.To(layer.Device, stateInfo.DType)
		if err != nil {
			return nil, err
		}
		_ = current.Close()
		current = placed
		if rows := features[index]; len(rows) != 0 {
			combined, err := addFeatureCotangents(ctx, current, rows)
			if err != nil {
				return nil, fmt.Errorf("decoder: layer %d feature cotangent: %w", index, err)
			}
			_ = current.Close()
			current = combined
		}
		gradient, err := layers.DecoderVJP(ctx, state, layer.Weights, layer.Adapter, layer.Cosine, layer.Sine, current, layer.Config)
		if err != nil {
			return nil, fmt.Errorf("decoder: layer %d backward: %w", index, err)
		}
		byLayer[index] = gradient
		_ = current.Close()
		current, gradient.Input = gradient.Input, nil
	}
	// The registry fixes names and order; each gradient leaves its layer's
	// owner as it joins the result, so exactly one of them closes it.
	var result Gradients
	for _, slot := range adapterRegistry(m) {
		gradients := byLayer[slot.layer]
		if gradients == nil {
			_ = result.Close()
			return nil, fmt.Errorf("decoder: layer %d adapter gradients missing", slot.layer)
		}
		a, b := gradients.Pair(slot.target)
		value := a
		if slot.b {
			value = b
		}
		if *value == nil {
			_ = result.Close()
			return nil, fmt.Errorf("decoder: layer %d adapter gradient missing", slot.layer)
		}
		result = append(result, ParameterGradient{Name: slot.name, Value: *value})
		*value = nil
	}
	return result, nil
}

func (m *TextModel) head(hidden *torch.Tensor, count int64) (*torch.Tensor, error) {
	return m.headObserved(nil, hidden, count, nil)
}

func (m *TextModel) headObserved(ctx context.Context, hidden *torch.Tensor, count int64, observer ForwardObserver) (*torch.Tensor, error) {
	info, err := hidden.Info()
	if err != nil {
		return nil, err
	}
	normalized, err := layers.RMSNorm(hidden, m.FinalNorm, m.Epsilon)
	if err != nil {
		return nil, err
	}
	defer normalized.Close()
	if err = observeForward(ctx, observer, StageFinalNorm, -1, normalized); err != nil {
		return nil, err
	}
	last, err := normalized.Slice(1, info.Shape[1]-count, info.Shape[1], 1)
	if err != nil {
		return nil, err
	}
	defer last.Close()
	logits, err := layers.Linear(last, m.Head)
	if err != nil {
		return nil, err
	}
	defer logits.Close()
	return logits.To(info.Device, torch.Float32)
}

func (m *TextModel) validate(ctx context.Context, tokens []int64, limits Limits) (torch.Info, error) {
	if ctx == nil || m == nil || m.Embedding == nil || m.Head == nil || m.FinalNorm == nil || len(m.Layers) == 0 {
		return torch.Info{}, errors.New("decoder: incomplete text model")
	}
	if err := ctx.Err(); err != nil {
		return torch.Info{}, err
	}
	if len(tokens) == 0 || limits.MaxTokens <= 0 || int64(len(tokens)) > limits.MaxTokens || limits.LogitRows <= 0 || limits.LogitRows > int64(len(tokens)) || limits.MaxCheckpointBytes <= 0 {
		return torch.Info{}, errors.New("decoder: token or checkpoint limits invalid")
	}
	info, err := m.Embedding.Info()
	if err != nil {
		return torch.Info{}, err
	}
	if len(info.Shape) != 2 || info.Shape[0] <= 0 || info.Shape[1] <= 0 || info.RequiresGrad || (info.DType != torch.Float16 && info.DType != torch.Float32) {
		return torch.Info{}, errors.New("decoder: token embedding must be frozen Float16 or Float32")
	}
	bytes := int64(2)
	if info.DType == torch.Float32 {
		bytes = 4
	}
	for _, dimension := range []int64{int64(len(tokens)), info.Shape[1], int64(len(m.Layers)) + 2} {
		if bytes > limits.MaxCheckpointBytes/dimension {
			return torch.Info{}, errors.New("decoder: activation checkpoint budget exceeded")
		}
		bytes *= dimension
	}
	for _, id := range tokens {
		if id < 0 || id >= info.Shape[0] {
			return torch.Info{}, errors.New("decoder: token ID outside embedding vocabulary")
		}
	}
	if !(m.Epsilon > 0) || math.IsInf(m.Epsilon, 0) {
		return torch.Info{}, errors.New("decoder: normalization epsilon invalid")
	}
	for _, expected := range []struct {
		value *torch.Tensor
		shape []int64
	}{
		{m.Head, []int64{info.Shape[0], info.Shape[1]}}, {m.FinalNorm, []int64{info.Shape[1]}},
	} {
		actual, err := expected.value.Info()
		if err != nil {
			return torch.Info{}, err
		}
		if actual.RequiresGrad || actual.DType != info.DType || actual.Device != m.Layers[len(m.Layers)-1].Device || !sameShape(actual.Shape, expected.shape) {
			return torch.Info{}, errors.New("decoder: frozen head or final norm geometry, dtype or device differs")
		}
	}
	return info, nil
}

func sameShape(first, second []int64) bool {
	if len(first) != len(second) {
		return false
	}
	for index := range first {
		if first[index] != second[index] {
			return false
		}
	}
	return true
}
