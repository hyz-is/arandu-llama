package assembly_test

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"testing"

	"github.com/tayi-ai/arandu-llama/training/decoder"
)

// layoutDigest hashes every admitted tensor in plan order: names, reference
// identity, geometry, precision, placement and trainability. Shards are left
// out: the fixture assigns them in map order.
func layoutDigest(p *decoder.AssemblyPlan) string {
	h := sha256.New()
	for _, item := range p.Tensors() {
		fmt.Fprintf(h, "%s|%s|%s|%v|%d|%v|%d|%t\n", item.ReferenceName, item.SourceName, item.SHA256, item.Shape, item.DType, item.Device, item.Bytes, item.Trainable)
	}
	summary := p.Summary()
	fmt.Fprintf(h, "%d|%d|%d|%v\n", summary.BaseTensors, summary.AdapterTensors, summary.AdapterElements, summary.PersistentBytes)
	return fmt.Sprintf("%x", h.Sum(nil))
}

// The q/v plan predates coverage declarations; any byte of its layout that
// moves fails here.
func TestPlanWithoutCoverageKeepsItsLayout(t *testing.T) {
	if digest := layoutDigest(plan(t, fixture())); digest != "9c8f4149e26aa9cdb4ddf9db0ae724a9ecb6c989122491b3b3fd6af09010ef44" {
		t.Fatalf("the q/v plan layout changed: %s", digest)
	}
}

// coveredModules lists, independently of the code under test, what the full
// declaration adapts on each layer kind, in canonical order.
var coveredModules = map[string][]string{
	"full_attention":   {"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"},
	"linear_attention": {"linear_attn.out_proj", "linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.in_proj_b", "linear_attn.in_proj_a", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"},
}

// fullyCovered rewrites the q/v fixture reference as a reference for the full
// declaration at rank 4: every adapted base row takes its base_layer name and
// gains FP32 trainable A and B rows on the same device, and the q/v rows of
// the historical declaration are rebuilt the same way.
func fullyCovered(d documents) documents {
	kinds := d.Config["text_config"].(map[string]any)["layer_types"].([]string)
	rows := map[string]reference{}
	var kept []reference
	for _, row := range d.Reference {
		if row.Grad {
			continue
		}
		row.Name = strings.Replace(row.Name, ".base_layer.weight", ".weight", 1)
		rows[row.Name] = row
		kept = append(kept, row)
	}
	d.Reference = kept
	for layer, kind := range kinds {
		for _, module := range coveredModules[kind] {
			name := fmt.Sprintf("base_model.model.model.language_model.layers.%d.%s", layer, module)
			base := rows[name+".weight"]
			for i := range d.Reference {
				if d.Reference[i].Name == name+".weight" {
					d.Reference[i].Name = name + ".base_layer.weight"
				}
			}
			for _, pair := range []struct {
				letter string
				shape  []int64
			}{{"A", []int64{4, base.Shape[1]}}, {"B", []int64{base.Shape[0], 4}}} {
				d.Reference = append(d.Reference, reference{name + ".lora_" + pair.letter + ".default.weight", "parameter", "torch.float32", base.Device, pair.shape, pair.shape[0] * pair.shape[1] * 4, strings.Repeat("0", 64), true})
			}
		}
	}
	return d
}

func fullLimits() decoder.AssemblyLimits {
	limits := admittedLimits()
	limits.AdapterCoverage = decoder.FullAdapterCoverage()
	limits.PersistentBytes = []int64{12 << 30, 12 << 30}
	return limits
}

func TestExplicitQVCoverageKeepsTheHistoricalLayout(t *testing.T) {
	limits := admittedLimits()
	limits.AdapterCoverage = &decoder.AdapterCoverage{Version: 1, FullAttention: []string{"q_proj", "v_proj"}}
	i, c, r, id := fixture().encode(t)
	p, err := decoder.PlanTextAssembly(i, c, r, id, limits)
	if err != nil {
		t.Fatal(err)
	}
	if digest := layoutDigest(p); digest != "9c8f4149e26aa9cdb4ddf9db0ae724a9ecb6c989122491b3b3fd6af09010ef44" {
		t.Fatalf("an explicit q/v declaration changed the plan: %s", digest)
	}
}

// The 32-layer fixture has the admitted hybrid geometry, so the counts here
// are the full declaration's parameter budget at rank 4, from shapes alone.
func TestFullCoveragePlanAdmitsEveryProjection(t *testing.T) {
	d := fullyCovered(fixture())
	i, c, r, id := d.encode(t)
	p, err := decoder.PlanTextAssembly(i, c, r, id, fullLimits())
	if err != nil {
		t.Fatal(err)
	}
	summary := p.Summary()
	byKind := map[string]int64{}
	adapters := 0
	for _, item := range p.Tensors() {
		if !item.Trainable {
			if strings.Contains(item.SourceName, "_proj") && !strings.HasSuffix(item.ReferenceName, ".base_layer.weight") {
				t.Fatalf("covered base %s kept its plain reference name", item.SourceName)
			}
			continue
		}
		adapters++
		for _, kind := range []string{"self_attn", "linear_attn", "mlp"} {
			if strings.Contains(item.ReferenceName, "."+kind+".") {
				byKind[kind] += item.Bytes / 4
			}
		}
	}
	if summary.BaseTensors != 427 || summary.AdapterTensors != 496 || adapters != 496 || summary.AdapterElements != 10_819_584 ||
		byKind["self_attn"] != 983_040 || byKind["linear_attn"] != 3_545_088 || byKind["mlp"] != 6_291_456 {
		t.Fatalf("full coverage geometry differs: %+v %v", summary, byKind)
	}
	projections, err := decoder.CoverageProjections(c, decoder.FullAdapterCoverage(), 4)
	if err != nil || len(projections)*2 != summary.AdapterTensors {
		t.Fatalf("projections differ from the plan: %d %v", len(projections), err)
	}
	tensors := p.Tensors()[summary.BaseTensors:]
	for k, projection := range projections {
		a, b := tensors[2*k], tensors[2*k+1]
		if a.ReferenceName != projection.Name+".lora_A.default.weight" || b.ReferenceName != projection.Name+".lora_B.default.weight" ||
			fmt.Sprint(a.Shape) != fmt.Sprint([]int64{projection.Rank, projection.Input}) || fmt.Sprint(b.Shape) != fmt.Sprint([]int64{projection.Output, projection.Rank}) {
			t.Fatalf("projection %d differs from plan order: %+v", k, projection)
		}
	}
	t.Logf("full coverage at rank 4: %d adapter tensors, %d parameters (full attention %d, linear attention %d, MLP %d); q/v: 32 tensors, 557056 parameters",
		summary.AdapterTensors, summary.AdapterElements, byKind["self_attn"], byKind["linear_attn"], byKind["mlp"])
}

func TestCoverageAndReferenceMustAgree(t *testing.T) {
	i, c, r, id := fixture().encode(t)
	if p, err := decoder.PlanTextAssembly(i, c, r, id, fullLimits()); p != nil || !errors.Is(err, decoder.ErrAssembly) {
		t.Fatalf("a q/v reference was admitted for the full declaration: %v", err)
	}
	i, c, r, id = fullyCovered(fixture()).encode(t)
	limits := fullLimits()
	limits.AdapterCoverage = nil
	if p, err := decoder.PlanTextAssembly(i, c, r, id, limits); p != nil || !errors.Is(err, decoder.ErrAssembly) {
		t.Fatalf("a full reference was admitted without its declaration: %v", err)
	}
}

func TestCoverageRefusesUndeclaredShapes(t *testing.T) {
	_, c, _, _ := fixture().encode(t)
	for name, coverage := range map[string]*decoder.AdapterCoverage{
		"no_version":       {FullAttention: []string{"q_proj"}},
		"future_version":   {Version: 2, FullAttention: []string{"q_proj"}},
		"unknown_module":   {Version: 1, FullAttention: []string{"qkv_proj"}},
		"repeated_module":  {Version: 1, MLP: []string{"up_proj", "up_proj"}},
		"out_of_order":     {Version: 1, FullAttention: []string{"v_proj", "q_proj"}},
		"other_kind":       {Version: 1, LinearAttention: []string{"q_proj"}},
		"mtp_module":       {Version: 1, MLP: []string{"mtp.fc"}},
		"declares_nothing": {Version: 1},
	} {
		t.Run(name, func(t *testing.T) {
			if err := coverage.Validate(); !errors.Is(err, decoder.ErrAssembly) {
				t.Fatalf("declaration admitted: %v", err)
			}
			if projections, err := decoder.CoverageProjections(c, coverage, 4); projections != nil || !errors.Is(err, decoder.ErrAssembly) {
				t.Fatalf("projections derived: %v", err)
			}
			limits := admittedLimits()
			limits.AdapterCoverage = coverage
			i, c, r, id := fixture().encode(t)
			if p, err := decoder.PlanTextAssembly(i, c, r, id, limits); p != nil || !errors.Is(err, decoder.ErrAssembly) {
				t.Fatalf("plan admitted: %v", err)
			}
		})
	}
}

// Without a declaration the projections are exactly the historical q/v
// initializer: eight full-attention layers, q then v, rank 4.
func TestCoverageProjectionsWithoutDeclarationAreTheHistoricalOnes(t *testing.T) {
	_, c, _, _ := fixture().encode(t)
	projections, err := decoder.CoverageProjections(c, nil, 4)
	if err != nil {
		t.Fatal(err)
	}
	var expected []decoder.InitialProjection
	for layer := 3; layer < 32; layer += 4 {
		expected = append(expected,
			decoder.InitialProjection{Name: fmt.Sprintf("base_model.model.model.language_model.layers.%d.self_attn.q_proj", layer), Input: 4096, Output: 8192, Rank: 4},
			decoder.InitialProjection{Name: fmt.Sprintf("base_model.model.model.language_model.layers.%d.self_attn.v_proj", layer), Input: 4096, Output: 1024, Rank: 4})
	}
	if fmt.Sprint(projections) != fmt.Sprint(expected) {
		t.Fatalf("historical projections differ:\n%v\n%v", projections, expected)
	}
}

func TestLimitsEncodeCoverageOnlyWhenDeclared(t *testing.T) {
	legacy, err := json.Marshal(admittedLimits())
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Contains(legacy, []byte("AdapterCoverage")) {
		t.Fatal("an undeclared coverage entered the encoding")
	}
	full, err := json.Marshal(fullLimits())
	if err != nil {
		t.Fatal(err)
	}
	const declared = `"AdapterCoverage":{"version":1,"full_attention":["q_proj","k_proj","v_proj","o_proj"],"linear_attention":["out_proj","in_proj_qkv","in_proj_z","in_proj_b","in_proj_a"],"mlp":["gate_proj","up_proj","down_proj"]}`
	if !bytes.Contains(full, []byte(declared)) {
		t.Fatalf("full declaration encodes differently: %s", full)
	}
}
