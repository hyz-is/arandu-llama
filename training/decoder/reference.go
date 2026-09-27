package decoder

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"slices"
	"strings"
)

// referenceRow is one initial-reference tensor row, with fields in the
// document's sorted key order.
type referenceRow struct {
	Bytes        int64   `json:"bytes"`
	SHA256       string  `json:"content_sha256"`
	Device       string  `json:"device"`
	DType        string  `json:"dtype"`
	Kind         string  `json:"kind"`
	Name         string  `json:"name"`
	RequiresGrad bool    `json:"requires_grad"`
	Shape        []int64 `json:"shape"`
}

// DeriveInitialReference rewrites an admitted initial-reference document for
// the adapter coverage in limits and an initializer that is not pinned yet.
// The source's own text-layer adapter rows are dropped and their bases take
// their plain names back; then every base the coverage adapts takes its PEFT
// base_layer name and is followed by its A and B rows: FP32, trainable, on the
// base row's device, with the content digests of the initializer's bytes.
// Every other row and top-level field is kept, parameter_tensors is recounted,
// and aggregate_sha256, which certified the source's own tensor bytes, is
// dropped. It returns the document and the aggregate to pin as the
// initializer's ExpectedSHA256 and the plan's InitialAdapterSHA256.
//
// initializer.ExpectedSHA256 must be empty and its projections must equal
// CoverageProjections for configJSON, the coverage and AdapterRank. Nothing
// here admits the result: PlanTextAssembly must still accept it by digest.
func DeriveInitialReference(ctx context.Context, referenceJSON, configJSON []byte, limits AssemblyLimits, initializer InitialAdapterSpec) ([]byte, string, error) {
	if ctx == nil || len(referenceJSON) == 0 || len(referenceJSON) > 16<<20 {
		return nil, "", fmt.Errorf("%w: reference document required", ErrAssembly)
	}
	if err := validateAssemblyLimits(limits); err != nil {
		return nil, "", err
	}
	geometry, err := validateAssemblyConfig(configJSON)
	if err != nil {
		return nil, "", err
	}
	if len(limits.DeviceByLayer) != geometry.Layers {
		return nil, "", fmt.Errorf("%w: placement must cover every layer", ErrAssembly)
	}
	projections, err := CoverageProjections(configJSON, limits.AdapterCoverage, limits.AdapterRank)
	if err != nil {
		return nil, "", err
	}
	if !slices.Equal(projections, initializer.Projections) {
		return nil, "", fmt.Errorf("%w: initializer projections differ from the coverage", ErrAssembly)
	}
	aggregate, digests, err := DescribeInitialAdapter(ctx, initializer)
	if err != nil {
		return nil, "", err
	}
	var document map[string]json.RawMessage
	if err := json.Unmarshal(referenceJSON, &document); err != nil {
		return nil, "", fmt.Errorf("%w: reference: %v", ErrAssembly, err)
	}
	var source []referenceRow
	decoder := json.NewDecoder(bytes.NewReader(document["tensors"]))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&source); err != nil || len(source) == 0 || len(source) > 4096 {
		return nil, "", fmt.Errorf("%w: reference rows unreadable or out of bounds: %v", ErrAssembly, err)
	}
	const layerPrefix = "base_model.model.model.language_model.layers."
	var rows []referenceRow
	position := make(map[string]int, len(source))
	for _, row := range source {
		if strings.HasPrefix(row.Name, layerPrefix) {
			if strings.HasSuffix(row.Name, ".lora_A.default.weight") || strings.HasSuffix(row.Name, ".lora_B.default.weight") {
				continue
			}
			row.Name = strings.Replace(row.Name, ".base_layer.weight", ".weight", 1)
		}
		if _, duplicate := position[row.Name]; duplicate {
			return nil, "", fmt.Errorf("%w: duplicate reference %q", ErrAssembly, row.Name)
		}
		position[row.Name] = len(rows)
		rows = append(rows, row)
	}
	// Adapters follow their base in plan order: A then B for each projection.
	weights, adapters := assemblyGeometry(geometry, limits)
	byName := make(map[string]InitialTensorDigest, len(digests))
	for _, digest := range digests {
		byName[digest.Name] = digest
	}
	after := make(map[int][]referenceRow)
	next := 0
	for _, weight := range weights {
		if !strings.HasSuffix(weight.ReferenceName, ".base_layer.weight") {
			continue
		}
		plain := strings.TrimSuffix(weight.ReferenceName, ".base_layer.weight")
		index, found := position[plain+".weight"]
		if !found {
			return nil, "", fmt.Errorf("%w: reference lacks the adapted base %s", ErrAssembly, plain)
		}
		rows[index].Name = weight.ReferenceName
		for _, letter := range []string{"A", "B"} {
			if next >= len(adapters) || adapters[next].ReferenceName != plain+".lora_"+letter+".default.weight" {
				return nil, "", fmt.Errorf("%w: adapter order differs from its bases", ErrAssembly)
			}
			spec := adapters[next]
			digest, found := byName[spec.ReferenceName]
			if !found || !slices.Equal(digest.Shape, spec.Shape) {
				return nil, "", fmt.Errorf("%w: initializer lacks %s", ErrAssembly, spec.ReferenceName)
			}
			after[index] = append(after[index], referenceRow{Bytes: spec.Bytes, SHA256: digest.SHA256, Device: rows[index].Device,
				DType: assemblyDTypeName(spec.DType), Kind: "parameter", Name: spec.ReferenceName, RequiresGrad: true, Shape: slices.Clone(spec.Shape)})
			next++
		}
	}
	if next != len(adapters) || len(digests) != len(adapters) {
		return nil, "", fmt.Errorf("%w: adapters and initializer differ in count", ErrAssembly)
	}
	derived := make([]referenceRow, 0, len(rows)+len(adapters))
	parameters := 0
	for index, row := range rows {
		derived = append(derived, row)
		derived = append(derived, after[index]...)
	}
	for _, row := range derived {
		if row.Kind == "parameter" {
			parameters++
		}
	}
	tensors, err := json.Marshal(derived)
	if err != nil {
		return nil, "", err
	}
	document["tensors"] = tensors
	delete(document, "aggregate_sha256")
	if _, counted := document["parameter_tensors"]; counted {
		document["parameter_tensors"] = json.RawMessage(fmt.Sprint(parameters))
	}
	body, err := json.MarshalIndent(document, "", "  ")
	if err != nil {
		return nil, "", err
	}
	return append(body, '\n'), aggregate, ctx.Err()
}
