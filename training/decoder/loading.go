package decoder

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"path/filepath"
	"slices"
	"sort"
	"strings"

	"github.com/tayi-ai/arandu-llama/checkpoint"
	"github.com/tayi-ai/arandu-llama/training/layers"
	"github.com/tayi-ai/arandu-llama/training/loading"
	"github.com/tayi-ai/arandu-llama/training/sequence"
	"github.com/tayi-ai/arandu-llama/training/tensor"
	"github.com/tayi-ai/arandu-llama/training/torch"
)

// ErrAssembly identifies invalid geometry, identities, placement or admission.
var ErrAssembly = errors.New("decoder: text assembly rejected")

// AssemblyIdentity contains mandatory caller-admitted identities. Document
// hashes cover exact bytes, and InitialAdapterSHA256 uses name/content order.
// The caller pins the intended checkpoint; matching geometry alone is not an
// identity. No default identity or alternate initialization is accepted.
type AssemblyIdentity struct {
	IndexSHA256, ConfigSHA256, ReferenceSHA256, InitialAdapterSHA256 string
}

// AssemblyLimits admits persistent weight bytes per GPU and temporary explicit
// tensor copies. TensorCopyBytes must cover three times the larger source/final
// tensor storage, including bounded hash copies. These are payload estimates,
// not measured RAM/VRAM caps. Headers, allocator overhead, caller-owned initial
// tensors, activations, RoPE and kernel workspaces require separate admission.
// All execution budgets are explicit; no sequence or attention defaults apply.
type AssemblyLimits struct {
	HeaderLimits                  checkpoint.Limits
	TensorCopyBytes               int64
	PersistentBytes               []int64
	DeviceByLayer                 []int
	EmbeddingDevice, OutputDevice int
	AdapterRank                   int64
	AdapterAlpha                  float64
	HashChunkBytes                int
	MaxInputElements              int64
	MaxScoreElements              int64
	MaxWorkingElements            int64
	Sequence                      sequence.SequenceLimits
	// AdapterCoverage declares the adapted projections. Nil is the q/v
	// coverage of full attention and leaves this field out of the encoding.
	AdapterCoverage *AdapterCoverage `json:",omitempty"`
}

// AssemblyTensor specifies a checkpoint weight and its required final identity.
// All base tensors pass through Float16 CPU storage before final placement,
// including A_log; attention tensors are then promoted to Float32. Adapter
// tensors have an empty SourceName/Shard and are copied from the admitted CPU set.
type AssemblyTensor struct {
	SourceName, ReferenceName, Shard, SHA256 string
	Shape                                    []int64
	DType                                    torch.DType
	Device                                   torch.Device
	Bytes                                    int64
	Trainable                                bool
}

// AssemblySummary records the admitted geometry without allocating tensor data.
type AssemblySummary struct {
	BaseTensors, AdapterTensors int
	AdapterElements             int64
	PersistentBytes             []int64
	LocalMPSBytes               int64
	Identity                    AssemblyIdentity
}

// AssemblyPlan is an immutable, validated text-only weight and placement plan.
// Vision and MTP weights are not executed by the text-only forward. This plan
// does not support image/video inputs or claim equivalence for multimodal calls.
type AssemblyPlan struct {
	weights, adapters []AssemblyTensor
	limits            AssemblyLimits
	summary           AssemblySummary
	localMPS          bool
	geometry          TextGeometry
}

// Summary returns value-only counts, budgets and document identities.
func (p *AssemblyPlan) Summary() AssemblySummary {
	if p == nil {
		return AssemblySummary{}
	}
	result := p.summary
	result.PersistentBytes = slices.Clone(result.PersistentBytes)
	return result
}

// Tensors returns detached metadata copies in base-then-adapter loading order.
func (p *AssemblyPlan) Tensors() []AssemblyTensor {
	if p == nil {
		return nil
	}
	result := append(append([]AssemblyTensor(nil), p.weights...), p.adapters...)
	for i := range result {
		result[i].Shape = slices.Clone(result[i].Shape)
	}
	return result
}

// Shard is an immutable opened checkpoint source. Size reports its full size;
// the loader closes each successfully opened source exactly once per operation.
type Shard interface {
	io.ReaderAt
	Size() int64
	Close() error
}

// ShardProvider opens a caller-authorized shard by its validated basename. It
// performs no implicit download or discovery. Separate opens must return handles
// with independent Close ownership over identical, immutable source bytes.
type ShardProvider interface {
	OpenShard(context.Context, string) (Shard, error)
}

// AssemblyInspection records header-only source checks, not content verification.
type AssemblyInspection struct {
	Shards, BaseTensors int
	SourceBytes         int64
	LargestCopyBytes    int64
}

// AssemblyTensorReceipt records both checkpoint payload and final native hashes.
// Initial adapter receipts have no checkpoint payload receipt.
type AssemblyTensorReceipt struct {
	Name, SHA256 string
	Source       *loading.Receipt
}

// LoadedTextModel owns every placed weight and adapter. Model and Parameters
// borrow those handles; close this owner only after all forward/VJP work ends.
// Rotary values remain nil until the caller supplies prompt-specific tables.
// The caller owns those rotary handles independently and must close them itself.
type LoadedTextModel struct {
	Model      *TextModel
	Parameters []InitialParameter
	Receipts   []AssemblyTensorReceipt
	Summary    AssemblySummary
	owned      []*torch.Tensor
}

// Close releases all assembled weights; borrowed CPU initial tensors and caller
// rotary values remain untouched. Closing this owner invalidates its TextModel.
func (m *LoadedTextModel) Close() error {
	if m == nil {
		return nil
	}
	var failures []error
	for _, value := range m.owned {
		failures = append(failures, value.Close())
	}
	m.owned, m.Parameters = nil, nil
	if m.Model != nil {
		*m.Model = TextModel{}
	}
	m.Model = nil
	return errors.Join(failures...)
}

// PlanTextAssembly validates exact document hashes, caller-admitted decoder geometry,
// every required reference tensor and both persistent budgets before any source
// is opened. No payload is read and no native backend is required by this call.
func PlanTextAssembly(indexJSON, configJSON, referenceJSON []byte, identity AssemblyIdentity, limits AssemblyLimits) (*AssemblyPlan, error) {
	for _, document := range []struct {
		data   []byte
		digest string
	}{
		{indexJSON, identity.IndexSHA256}, {configJSON, identity.ConfigSHA256}, {referenceJSON, identity.ReferenceSHA256},
	} {
		if len(document.data) == 0 || len(document.data) > 16<<20 || !validAssemblyHash(document.digest) || assemblyHash(document.data) != document.digest {
			return nil, fmt.Errorf("%w: document identity or size differs", ErrAssembly)
		}
	}
	if !validAssemblyHash(identity.InitialAdapterSHA256) {
		return nil, fmt.Errorf("%w: initial adapter identity is required", ErrAssembly)
	}
	if err := validateAssemblyLimits(limits); err != nil {
		return nil, err
	}
	geometry, err := validateAssemblyConfig(configJSON)
	if err != nil {
		return nil, err
	}
	if len(limits.DeviceByLayer) != geometry.Layers {
		return nil, fmt.Errorf("%w: placement must cover every layer", ErrAssembly)
	}
	var index struct {
		WeightMap map[string]string `json:"weight_map"`
	}
	if err := json.Unmarshal(indexJSON, &index); err != nil {
		return nil, fmt.Errorf("%w: index: %v", ErrAssembly, err)
	}
	var reference struct {
		Tensors []assemblyReference `json:"tensors"`
	}
	if err := json.Unmarshal(referenceJSON, &reference); err != nil {
		return nil, fmt.Errorf("%w: reference: %v", ErrAssembly, err)
	}
	if len(index.WeightMap) > 4096 || len(reference.Tensors) > 4096 {
		return nil, fmt.Errorf("%w: too many source/reference entries", ErrAssembly)
	}
	refs := make(map[string]assemblyReference, len(reference.Tensors))
	for _, row := range reference.Tensors {
		if _, exists := refs[row.Name]; exists {
			return nil, fmt.Errorf("%w: duplicate reference %q", ErrAssembly, row.Name)
		}
		refs[row.Name] = row
	}
	limits.PersistentBytes = slices.Clone(limits.PersistentBytes)
	limits.DeviceByLayer = slices.Clone(limits.DeviceByLayer)
	limits.AdapterCoverage = limits.AdapterCoverage.clone()
	p := &AssemblyPlan{limits: limits, geometry: geometry, summary: AssemblySummary{Identity: identity, PersistentBytes: make([]int64, len(limits.PersistentBytes))}}
	p.weights, p.adapters = assemblyGeometry(geometry, limits)
	admittedNames := make(map[string]bool, len(p.weights)+len(p.adapters))
	sourceNames := make(map[string]bool, len(p.weights))
	for _, group := range [][]AssemblyTensor{p.weights, p.adapters} {
		for i := range group {
			item := &group[i]
			row, exists := refs[item.ReferenceName]
			if !exists || row.Kind != "parameter" || row.RequiresGrad != item.Trainable || !sameShape(row.Shape, item.Shape) ||
				row.Bytes != item.Bytes || row.DType != assemblyDTypeName(item.DType) || row.Device != fmt.Sprintf("cuda:%d", item.Device.Index) || !validAssemblyHash(row.SHA256) {
				return nil, fmt.Errorf("%w: reference metadata differs for %s", ErrAssembly, item.ReferenceName)
			}
			item.SHA256 = row.SHA256
			admittedNames[item.ReferenceName] = true
			if !item.Trainable {
				shard, exists := index.WeightMap[item.SourceName]
				if !exists || !validAssemblyShard(shard) {
					return nil, fmt.Errorf("%w: invalid or missing shard for %s", ErrAssembly, item.SourceName)
				}
				item.Shard = shard
				sourceNames[item.SourceName] = true
			}
			if item.Bytes > math.MaxInt64-p.summary.PersistentBytes[item.Device.Index] {
				return nil, fmt.Errorf("%w: persistent byte count overflow", ErrAssembly)
			}
			p.summary.PersistentBytes[item.Device.Index] += item.Bytes
			if item.Trainable {
				p.summary.AdapterElements += item.Bytes / 4
			}
			if item.Bytes > limits.TensorCopyBytes/3 {
				return nil, fmt.Errorf("%w: tensor copy budget for %s", ErrAssembly, item.ReferenceName)
			}
		}
	}
	for name, shard := range index.WeightMap {
		if !validAssemblyShard(shard) {
			return nil, fmt.Errorf("%w: invalid shard basename", ErrAssembly)
		}
		if sourceNames[name] || strings.HasPrefix(name, "model.visual.") || strings.HasPrefix(name, "mtp.") {
			continue
		}
		return nil, fmt.Errorf("%w: unexpected source tensor %s", ErrAssembly, name)
	}
	for name, row := range refs {
		if admittedNames[name] || strings.HasPrefix(name, "base_model.model.model.visual.") {
			continue
		}
		if (name == "base_model.model.model.language_model.rotary_emb.inv_freq" || name == "base_model.model.model.language_model.rotary_emb.original_inv_freq") && row.Kind == "buffer" {
			continue
		}
		return nil, fmt.Errorf("%w: unexpected reference tensor %s", ErrAssembly, name)
	}
	for device, needed := range p.summary.PersistentBytes {
		if needed > limits.PersistentBytes[device] {
			return nil, fmt.Errorf("%w: GPU %d persistent budget needs %d bytes", ErrAssembly, device, needed)
		}
	}
	p.summary.BaseTensors, p.summary.AdapterTensors = len(p.weights), len(p.adapters)
	return p, nil
}

// PlanLocalMPSAssembly retains the admitted reference placement validation and
// content hashes, then explicitly places the same tensors on one Apple device.
// maxMPSBytes is a caller-admitted payload cap, not an available-memory probe;
// activations, allocator overhead and system reserve need separate admission.
func PlanLocalMPSAssembly(indexJSON, configJSON, referenceJSON []byte, identity AssemblyIdentity, limits AssemblyLimits, maxMPSBytes int64) (*AssemblyPlan, error) {
	p, err := PlanTextAssembly(indexJSON, configJSON, referenceJSON, identity, limits)
	if err != nil {
		return nil, err
	}
	var needed int64
	for _, size := range p.summary.PersistentBytes {
		if size > math.MaxInt64-needed {
			return nil, ErrAssembly
		}
		needed += size
	}
	if maxMPSBytes <= 0 || needed > maxMPSBytes {
		return nil, fmt.Errorf("%w: local MPS persistent budget needs %d bytes", ErrAssembly, needed)
	}
	for i := range p.weights {
		p.weights[i].Device = torch.MPSDevice()
	}
	for i := range p.adapters {
		p.adapters[i].Device = torch.MPSDevice()
	}
	p.summary.LocalMPSBytes = needed
	p.localMPS = true
	return p, nil
}

// InspectAssemblySources opens only required shards, validates headers and all
// source shapes/dtypes/copy admissions, then closes every source. No payload or
// native tensor is read/created. Successful inspection is not a content attestation.
func InspectAssemblySources(ctx context.Context, plan *AssemblyPlan, provider ShardProvider) (result AssemblyInspection, err error) {
	sources, result, err := openAssemblySources(ctx, plan, provider)
	return result, errors.Join(err, closeAssemblySources(sources))
}

// LoadTextAssembly assembles the fixed text model from caller-owned
// immutable shard providers and an exact borrowed CPU initializer. It verifies
// all source headers before payloads, then every final tensor hash after the
// historical Float16 intermediate and Float32 attention promotion. A mismatch
// closes all placed tensors; no partial model is returned. GPU resource caps,
// process deadlines and live reserve monitoring remain caller responsibilities.
// Initial tensors must remain open and unchanged until this operation returns.
func LoadTextAssembly(ctx context.Context, plan *AssemblyPlan, provider ShardProvider, initial *InitialAdapter) (_ *LoadedTextModel, err error) {
	if err := assemblyContext(ctx, plan, provider); err != nil {
		return nil, err
	}
	if err := validateAssemblyInitial(ctx, plan, initial); err != nil {
		return nil, err
	}
	if plan.localMPS {
		if !torch.MPSAvailable() {
			return nil, fmt.Errorf("%w: local MPS unavailable", ErrAssembly)
		}
		probe, probeErr := torch.FromFloat32([]float32{0}, []int64{1}, torch.MPSDevice(), false)
		if probeErr != nil {
			return nil, fmt.Errorf("%w: local MPS unavailable: %v", ErrAssembly, probeErr)
		}
		if closeErr := probe.Close(); closeErr != nil {
			return nil, closeErr
		}
	} else {
		if !torch.CUDAEnabled() {
			return nil, torch.ErrCUDAUnavailable
		}
		count, countErr := torch.CUDADeviceCount()
		if countErr != nil {
			return nil, countErr
		}
		if count < len(plan.limits.PersistentBytes) {
			return nil, fmt.Errorf("%w: admitted CUDA placement is unavailable", ErrAssembly)
		}
	}
	sources, _, err := openAssemblySources(ctx, plan, provider)
	if err != nil {
		return nil, errors.Join(err, closeAssemblySources(sources))
	}
	defer func() { err = errors.Join(err, closeAssemblySources(sources)) }()
	result := &LoadedTextModel{Model: &TextModel{Layers: make([]Layer, plan.geometry.Layers), Epsilon: plan.geometry.Epsilon}, Summary: plan.Summary()}
	complete := false
	defer func() {
		if !complete || err != nil {
			err = errors.Join(err, result.Close())
		}
	}()
	values := make(map[string]*torch.Tensor, len(plan.weights)+len(plan.adapters))
	for _, spec := range plan.weights {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		shard := sources[spec.Shard]
		storage, receipt, err := loading.LoadTensor(ctx, shard, shard.Size(), spec.SourceName, loading.Options{
			HeaderLimits: plan.limits.HeaderLimits, BudgetBytes: plan.limits.TensorCopyBytes,
			DType: torch.Float16, Device: torch.CPUDevice(), Expected: &loading.Expectation{Shape: spec.Shape},
		})
		if err != nil {
			return nil, fmt.Errorf("%w: %s: %w", ErrAssembly, spec.SourceName, err)
		}
		placed, err := storage.To(spec.Device, spec.DType)
		closeErr := storage.Close()
		if err != nil || closeErr != nil {
			if placed != nil {
				_ = placed.Close()
			}
			return nil, errors.Join(err, closeErr)
		}
		result.owned = append(result.owned, placed)
		if err := verifyAssemblyTensor(ctx, placed, spec, plan.limits.HashChunkBytes, nil); err != nil {
			return nil, err
		}
		values[spec.ReferenceName] = placed
		result.Receipts = append(result.Receipts, AssemblyTensorReceipt{Name: spec.ReferenceName, SHA256: spec.SHA256, Source: &receipt})
	}
	for i, spec := range plan.adapters {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		detached, err := initial.Parameters[i].Value.Detach()
		if err != nil {
			return nil, err
		}
		placed, err := detached.To(spec.Device, torch.Float32)
		closeErr := detached.Close()
		if err != nil || closeErr != nil {
			if placed != nil {
				_ = placed.Close()
			}
			return nil, errors.Join(err, closeErr)
		}
		leaf, err := placed.SetRequiresGrad(true)
		closeErr = placed.Close()
		if err != nil || closeErr != nil {
			if leaf != nil {
				_ = leaf.Close()
			}
			return nil, errors.Join(err, closeErr)
		}
		result.owned = append(result.owned, leaf)
		if err := verifyAssemblyTensor(ctx, leaf, spec, plan.limits.HashChunkBytes, nil); err != nil {
			return nil, err
		}
		values[spec.ReferenceName] = leaf
		result.Parameters = append(result.Parameters, InitialParameter{Name: spec.ReferenceName, Value: leaf})
		result.Receipts = append(result.Receipts, AssemblyTensorReceipt{Name: spec.ReferenceName, SHA256: spec.SHA256})
	}
	wireAssembly(result.Model, values, plan.geometry, plan.limits, plan.localMPS)
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	// Close sources before returning, so even a provider close failure cannot
	// return an apparently successful model or leak its owned native handles.
	if err := closeAssemblySources(sources); err != nil {
		return nil, err
	}
	complete = true
	return result, nil
}

type assemblyReference struct {
	Name         string  `json:"name"`
	Kind         string  `json:"kind"`
	DType        string  `json:"dtype"`
	Device       string  `json:"device"`
	Shape        []int64 `json:"shape"`
	Bytes        int64   `json:"bytes"`
	SHA256       string  `json:"content_sha256"`
	RequiresGrad bool    `json:"requires_grad"`
}

func assemblyGeometry(c TextGeometry, limits AssemblyLimits) (weights, adapters []AssemblyTensor) {
	// A covered projection keeps its weight under PEFT's base_layer name.
	add := func(source string, shape []int64, dtype torch.DType, device int, adapted bool) {
		ref := "base_model.model." + source
		if adapted {
			ref = strings.TrimSuffix(ref, ".weight") + ".base_layer.weight"
		}
		count := int64(1)
		for _, d := range shape {
			count *= d
		}
		width := int64(2)
		if dtype == torch.Float32 {
			width = 4
		}
		weights = append(weights, AssemblyTensor{SourceName: source, ReferenceName: ref, Shape: shape, DType: dtype, Device: torch.CUDADevice(device), Bytes: count * width})
	}
	type item struct {
		suffix string
		shape  []int64
	}
	add("model.language_model.embed_tokens.weight", []int64{c.Vocab, c.Hidden}, torch.Float16, limits.EmbeddingDevice, false)
	for layer := 0; layer < c.Layers; layer++ {
		device := limits.DeviceByLayer[layer]
		prefix := fmt.Sprintf("model.language_model.layers.%d.", layer)
		targets := limits.AdapterCoverage.targets(c.LayerTypes[layer])
		adapted := make(map[string]bool, len(targets))
		for _, target := range targets {
			adapted[adapterModules[target]] = true
		}
		shapes := make(map[string][]int64)
		addLayer := func(items []item, dtype torch.DType) {
			for _, item := range items {
				module := strings.TrimSuffix(item.suffix, ".weight")
				shapes[module] = item.shape
				add(prefix+item.suffix, item.shape, dtype, device, adapted[module])
			}
		}
		if c.LayerTypes[layer] == "full_attention" {
			addLayer([]item{
				{"self_attn.q_proj.weight", []int64{2 * c.Heads * c.Dimension, c.Hidden}}, {"self_attn.k_proj.weight", []int64{c.KVHeads * c.Dimension, c.Hidden}},
				{"self_attn.v_proj.weight", []int64{c.KVHeads * c.Dimension, c.Hidden}}, {"self_attn.o_proj.weight", []int64{c.Hidden, c.Heads * c.Dimension}},
				{"self_attn.q_norm.weight", []int64{c.Dimension}}, {"self_attn.k_norm.weight", []int64{c.Dimension}},
			}, torch.Float32)
		} else {
			addLayer([]item{
				{"linear_attn.dt_bias", []int64{c.VHeads}}, {"linear_attn.A_log", []int64{c.VHeads}}, {"linear_attn.conv1d.weight", []int64{2*c.KHeads*c.KDimension + c.VHeads*c.VDimension, 1, c.Conv}},
				{"linear_attn.norm.weight", []int64{c.VDimension}}, {"linear_attn.out_proj.weight", []int64{c.Hidden, c.VHeads * c.VDimension}},
				{"linear_attn.in_proj_qkv.weight", []int64{2*c.KHeads*c.KDimension + c.VHeads*c.VDimension, c.Hidden}}, {"linear_attn.in_proj_z.weight", []int64{c.VHeads * c.VDimension, c.Hidden}},
				{"linear_attn.in_proj_b.weight", []int64{c.VHeads, c.Hidden}}, {"linear_attn.in_proj_a.weight", []int64{c.VHeads, c.Hidden}},
			}, torch.Float32)
		}
		addLayer([]item{
			{"mlp.gate_proj.weight", []int64{c.MLP, c.Hidden}}, {"mlp.up_proj.weight", []int64{c.MLP, c.Hidden}}, {"mlp.down_proj.weight", []int64{c.Hidden, c.MLP}},
			{"input_layernorm.weight", []int64{c.Hidden}}, {"post_attention_layernorm.weight", []int64{c.Hidden}},
		}, torch.Float16)
		// A is [rank,input] and B is [output,rank] of the covered base weight.
		for _, target := range targets {
			module := adapterModules[target]
			base := shapes[module]
			for _, pair := range []struct {
				letter string
				shape  []int64
			}{{"A", []int64{limits.AdapterRank, base[1]}}, {"B", []int64{base[0], limits.AdapterRank}}} {
				name := "base_model.model." + prefix + module + ".lora_" + pair.letter + ".default.weight"
				adapters = append(adapters, AssemblyTensor{ReferenceName: name, Shape: pair.shape, DType: torch.Float32, Device: torch.CUDADevice(device), Bytes: pair.shape[0] * pair.shape[1] * 4, Trainable: true})
			}
		}
	}
	add("model.language_model.norm.weight", []int64{c.Hidden}, torch.Float16, limits.OutputDevice, false)
	add("lm_head.weight", []int64{c.Vocab, c.Hidden}, torch.Float16, limits.OutputDevice, false)
	return
}

func validateAssemblyLimits(limits AssemblyLimits) error {
	for _, bytes := range limits.PersistentBytes {
		if bytes <= 0 {
			return ErrAssembly
		}
	}
	for _, device := range append(slices.Clone(limits.DeviceByLayer), limits.EmbeddingDevice, limits.OutputDevice) {
		if device < 0 || device >= len(limits.PersistentBytes) {
			return ErrAssembly
		}
	}
	h := limits.HeaderLimits
	if h.MaxHeaderBytes <= 0 || h.MaxHeaderBytes > 100_000_000 || h.MaxTensors <= 0 || h.MaxDimensions < 3 || h.MaxMetadataEntries <= 0 || h.MaxChunkBytes <= 0 ||
		limits.HashChunkBytes < 4 || limits.HashChunkBytes > 4<<20 || limits.TensorCopyBytes <= 0 || len(limits.PersistentBytes) == 0 || len(limits.PersistentBytes) > 4096 || limits.AdapterRank < 1 || limits.AdapterRank > 1<<20 || limits.AdapterAlpha <= 0 || math.IsNaN(limits.AdapterAlpha) || math.IsInf(limits.AdapterAlpha, 0) ||
		limits.MaxInputElements <= 0 || limits.MaxScoreElements <= 0 || limits.MaxWorkingElements <= 0 || limits.Sequence.ChunkTokens <= 0 || limits.Sequence.ChunkTokens > tensor.MaxChunkTokens ||
		limits.Sequence.MaxTokens <= 0 || limits.Sequence.MaxOwnedElements <= 0 {
		return fmt.Errorf("%w: explicit positive parsing, copy, persistent and execution limits required", ErrAssembly)
	}
	return limits.AdapterCoverage.Validate()
}

func assemblyContext(ctx context.Context, plan *AssemblyPlan, provider ShardProvider) error {
	if ctx == nil || plan == nil || len(plan.weights) == 0 || len(plan.adapters) == 0 || provider == nil {
		return fmt.Errorf("%w: context, validated plan and provider required", ErrAssembly)
	}
	return ctx.Err()
}

func openAssemblySources(ctx context.Context, plan *AssemblyPlan, provider ShardProvider) (map[string]Shard, AssemblyInspection, error) {
	sources := make(map[string]Shard)
	result := AssemblyInspection{}
	if err := assemblyContext(ctx, plan, provider); err != nil {
		return sources, result, err
	}
	names := make(map[string]bool)
	for _, spec := range plan.weights {
		names[spec.Shard] = true
	}
	ordered := make([]string, 0, len(names))
	for name := range names {
		ordered = append(ordered, name)
	}
	sort.Strings(ordered)
	indices := make(map[string]*checkpoint.Safetensors)
	for _, name := range ordered {
		if err := ctx.Err(); err != nil {
			return sources, result, err
		}
		source, err := provider.OpenShard(ctx, name)
		if source != nil {
			sources[name] = source
		}
		if err != nil {
			return sources, result, err
		}
		if source == nil {
			return sources, result, fmt.Errorf("%w: provider returned nil shard", ErrAssembly)
		}
		index, err := checkpoint.OpenSafetensors(assemblyReader{ctx, source}, source.Size(), plan.limits.HeaderLimits)
		if err != nil {
			return sources, result, err
		}
		indices[name] = index
		result.Shards++
	}
	for _, spec := range plan.weights {
		if err := ctx.Err(); err != nil {
			return sources, result, err
		}
		row, found := indices[spec.Shard].Tensor(spec.SourceName)
		shape := make([]int64, len(row.Shape))
		for i, d := range row.Shape {
			if d > 1<<62 {
				return sources, result, fmt.Errorf("%w: oversized source shape", ErrAssembly)
			}
			shape[i] = int64(d)
		}
		if !found || !sameShape(shape, spec.Shape) || (row.DType != "F16" && row.DType != "BF16" && row.DType != "F32" && row.DType != "F64") {
			return sources, result, fmt.Errorf("%w: source metadata differs for %s", ErrAssembly, spec.SourceName)
		}
		needed := max(row.Size(), spec.Bytes)
		if needed > plan.limits.TensorCopyBytes/3 {
			return sources, result, fmt.Errorf("%w: source conversion exceeds copy budget for %s", ErrAssembly, spec.SourceName)
		}
		result.LargestCopyBytes = max(result.LargestCopyBytes, needed*3)
		result.SourceBytes += row.Size()
		result.BaseTensors++
	}
	return sources, result, nil
}

func closeAssemblySources(sources map[string]Shard) error {
	var failures []error
	for name, source := range sources {
		failures = append(failures, source.Close())
		delete(sources, name)
	}
	return errors.Join(failures...)
}

type assemblyReader struct {
	context context.Context
	source  io.ReaderAt
}

func (r assemblyReader) ReadAt(data []byte, offset int64) (int, error) {
	if err := r.context.Err(); err != nil {
		return 0, err
	}
	return r.source.ReadAt(data, offset)
}

func validateAssemblyInitial(ctx context.Context, plan *AssemblyPlan, initial *InitialAdapter) error {
	if initial == nil || len(initial.Parameters) != len(plan.adapters) || initial.SHA256 != plan.summary.Identity.InitialAdapterSHA256 {
		return fmt.Errorf("%w: exact initial adapter required", ErrAssembly)
	}
	digest := sha256.New()
	for i, spec := range plan.adapters {
		if initial.Parameters[i].Name != spec.ReferenceName {
			return fmt.Errorf("%w: adapter order/names differ", ErrAssembly)
		}
		cpuSpec := spec
		cpuSpec.Device = torch.CPUDevice()
		_, _ = digest.Write([]byte(spec.ReferenceName))
		if err := verifyAssemblyTensor(ctx, initial.Parameters[i].Value, cpuSpec, plan.limits.HashChunkBytes, digest); err != nil {
			return err
		}
	}
	if hex.EncodeToString(digest.Sum(nil)) != plan.summary.Identity.InitialAdapterSHA256 {
		return fmt.Errorf("%w: initial adapter aggregate differs", ErrAssembly)
	}
	return ctx.Err()
}

func verifyAssemblyTensor(ctx context.Context, value *torch.Tensor, spec AssemblyTensor, chunkBytes int, aggregate io.Writer) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if value == nil {
		return fmt.Errorf("%w: missing tensor", ErrAssembly)
	}
	info, err := value.Info()
	if err != nil {
		return err
	}
	if info.Device != spec.Device || info.DType != spec.DType || info.RequiresGrad != spec.Trainable || !sameShape(info.Shape, spec.Shape) {
		return fmt.Errorf("%w: native metadata differs for %s", ErrAssembly, spec.ReferenceName)
	}
	flat, err := value.Reshape([]int64{info.Elements})
	if err != nil {
		return err
	}
	defer flat.Close()
	width := int64(2)
	if spec.DType == torch.Float32 {
		width = 4
	}
	count := int64(chunkBytes) / width
	digest := sha256.New()
	for offset := int64(0); offset < info.Elements; {
		if err := ctx.Err(); err != nil {
			return err
		}
		end := min(info.Elements, offset+count)
		piece, err := flat.Slice(0, offset, end, 1)
		if err != nil {
			return err
		}
		finite, finiteErr := piece.AllFinite()
		content, copyErr := piece.Bytes()
		closeErr := piece.Close()
		if finiteErr != nil || copyErr != nil || closeErr != nil {
			return errors.Join(finiteErr, copyErr, closeErr)
		}
		if !finite {
			return fmt.Errorf("%w: nonfinite tensor %s", ErrAssembly, spec.ReferenceName)
		}
		_, _ = digest.Write(content)
		if aggregate != nil {
			if _, err := aggregate.Write(content); err != nil {
				return err
			}
		}
		offset = end
	}
	if hex.EncodeToString(digest.Sum(nil)) != spec.SHA256 {
		return fmt.Errorf("%w: materialized hash differs for %s", ErrAssembly, spec.ReferenceName)
	}
	return ctx.Err()
}

func wireAssembly(model *TextModel, values map[string]*torch.Tensor, c TextGeometry, limits AssemblyLimits, localMPS bool) {
	get := func(source string) *torch.Tensor { return values["base_model.model."+source] }
	model.Embedding = get("model.language_model.embed_tokens.weight")
	model.FinalNorm = get("model.language_model.norm.weight")
	model.Head = get("lm_head.weight")
	for i := range model.Layers {
		prefix := fmt.Sprintf("model.language_model.layers.%d.", i)
		g := func(s string) *torch.Tensor { return get(prefix + s) }
		layer := &model.Layers[i]
		layer.Device = torch.CUDADevice(limits.DeviceByLayer[i])
		if localMPS {
			layer.Device = torch.MPSDevice()
		}
		layer.Config = layers.DecoderConfig{Epsilon: c.Epsilon, MaxInputElements: limits.MaxInputElements,
			Full:   layers.AttentionConfig{Heads: c.Heads, KVHeads: c.KVHeads, HeadDimension: c.Dimension, RotaryDimension: int64(float64(c.Dimension) * c.RoPE.Partial), Epsilon: c.Epsilon, MaxScoreElements: limits.MaxScoreElements},
			Linear: layers.LinearAttentionConfig{KeyHeads: c.KHeads, ValueHeads: c.VHeads, KeyDimension: c.KDimension, ValueDimension: c.VDimension, Epsilon: c.Epsilon, MaxWorkingElements: limits.MaxWorkingElements, Sequence: limits.Sequence, FrozenWeightsValidated: true}}
		targets := limits.AdapterCoverage.targets(c.LayerTypes[i])
		adapted := make(map[string]bool, len(targets))
		for _, target := range targets {
			adapted[adapterModules[target]] = true
		}
		// base returns a projection under the name its coverage gave it.
		base := func(module string) *torch.Tensor {
			if adapted[module] {
				return g(module + ".base_layer.weight")
			}
			return g(module + ".weight")
		}
		layer.Weights = layers.DecoderWeights{InputNorm: g("input_layernorm.weight"), PostAttentionNorm: g("post_attention_layernorm.weight"), Gate: base("mlp.gate_proj"), Up: base("mlp.up_proj"), Down: base("mlp.down_proj")}
		if c.LayerTypes[i] == "full_attention" {
			layer.Weights.Full = &layers.AttentionWeights{Query: base("self_attn.q_proj"), Key: base("self_attn.k_proj"), Value: base("self_attn.v_proj"), Output: base("self_attn.o_proj"), QueryNorm: g("self_attn.q_norm.weight"), KeyNorm: g("self_attn.k_norm.weight")}
		} else {
			layer.Weights.Linear = &layers.LinearAttentionWeights{QKV: base("linear_attn.in_proj_qkv"), Z: base("linear_attn.in_proj_z"), Beta: base("linear_attn.in_proj_b"), Alpha: base("linear_attn.in_proj_a"), Convolution: g("linear_attn.conv1d.weight"), ALog: g("linear_attn.A_log"), DTBias: g("linear_attn.dt_bias"), Norm: g("linear_attn.norm.weight"), Output: base("linear_attn.out_proj")}
		}
		if len(targets) != 0 {
			adapter := &layers.AttentionLoRA{Alpha: limits.AdapterAlpha}
			for _, target := range targets {
				a, b := adapter.Pair(target)
				*a, *b = g(adapterModules[target]+".lora_A.default.weight"), g(adapterModules[target]+".lora_B.default.weight")
			}
			layer.Adapter = adapter
		}
	}
}

func validAssemblyHash(value string) bool {
	decoded, err := hex.DecodeString(value)
	return err == nil && len(decoded) == 32 && strings.ToLower(value) == value
}
func assemblyHash(value []byte) string { return fmt.Sprintf("%x", sha256.Sum256(value)) }
func assemblyDTypeName(dtype torch.DType) string {
	if dtype == torch.Float32 {
		return "torch.float32"
	}
	return "torch.float16"
}
func validAssemblyShard(name string) bool {
	return name != "" && name == filepath.Base(name) && !strings.ContainsAny(name, "/\\\\") && strings.HasSuffix(name, ".safetensors")
}

// Keep this compile-time assertion near the source adapter: no Seek, path,
// filesystem mutation or process interface is required by the loader.
var _ io.ReaderAt = assemblyReader{}

// Geometry returns an owned copy of the admitted text structure.
func (p *AssemblyPlan) Geometry() TextGeometry {
	if p == nil {
		return TextGeometry{}
	}
	c := p.geometry
	c.LayerTypes = slices.Clone(c.LayerTypes)
	c.RoPE.Sections = slices.Clone(c.RoPE.Sections)
	return c
}
