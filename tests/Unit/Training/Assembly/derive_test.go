//go:build libtorch && cgo && !libtorch_cuda

package assembly_test

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"testing"

	testfixture "github.com/tayi-ai/arandu-llama/tests/Unit/Training/Fixture"
	"github.com/tayi-ai/arandu-llama/training/decoder"
	"github.com/tayi-ai/arandu-llama/training/torch"
)

// derivedSpec is the historical schedule (seed 83, 24 prelude blocks of 32
// draws in [0.01,16)) over the projections a declaration implies.
func derivedSpec(t *testing.T, config []byte, coverage *decoder.AdapterCoverage) decoder.InitialAdapterSpec {
	t.Helper()
	projections, err := decoder.CoverageProjections(config, coverage, 4)
	if err != nil {
		t.Fatal(err)
	}
	return decoder.InitialAdapterSpec{Seed: 83, PreludeBlocks: 24, PreludeWidth: 32, PreludeLow: .01, PreludeHigh: 16, Projections: projections}
}

func derivedRows(t *testing.T, document []byte) map[string]reference {
	t.Helper()
	var parsed struct {
		Tensors []reference `json:"tensors"`
	}
	if err := json.Unmarshal(document, &parsed); err != nil {
		t.Fatal(err)
	}
	rows := map[string]reference{}
	for _, row := range parsed.Tensors {
		rows[row.Name] = row
	}
	return rows
}

// A q/v reference rewritten for the full declaration is admitted by the plan
// under its own digest, and the initializer pinned to the returned aggregate
// passes the loader's per-tensor check before any shard is opened. Frozen rows
// keep every byte of metadata; only covered bases change name.
func TestDerivedReferenceAdmitsTheFullDeclaration(t *testing.T) {
	d := fixture()
	index, config, source, _ := d.encode(t)
	limits := fullLimits()
	spec := derivedSpec(t, config, limits.AdapterCoverage)
	document, aggregate, err := decoder.DeriveInitialReference(context.Background(), source, config, limits, spec)
	if err != nil {
		t.Fatal(err)
	}
	if aggregate != testfixture.Digest(t, spec) {
		t.Fatal("derived aggregate differs from the independent reference")
	}
	identity := decoder.AssemblyIdentity{IndexSHA256: fmt.Sprintf("%x", sha256.Sum256(index)), ConfigSHA256: fmt.Sprintf("%x", sha256.Sum256(config)),
		ReferenceSHA256: fmt.Sprintf("%x", sha256.Sum256(document)), InitialAdapterSHA256: aggregate}
	p, err := decoder.PlanTextAssembly(index, config, document, identity, limits)
	if err != nil {
		t.Fatal(err)
	}
	if p.Summary().AdapterTensors != 496 || p.Summary().AdapterElements != 10_819_584 {
		t.Fatalf("derived plan differs: %+v", p.Summary())
	}
	derived := derivedRows(t, document)
	for _, row := range d.Reference {
		if row.Grad {
			if _, kept := derived[row.Name]; !kept {
				t.Fatalf("a q/v row of the full declaration is missing: %s", row.Name)
			}
			continue
		}
		name := strings.Replace(row.Name, ".base_layer.weight", ".weight", 1)
		got, found := derived[name]
		if !found {
			got, found = derived[strings.TrimSuffix(name, ".weight")+".base_layer.weight"]
		}
		row.Name = got.Name
		if !found || fmt.Sprint(got) != fmt.Sprint(row) {
			t.Fatalf("frozen row changed: %+v -> %+v", row, got)
		}
	}
	spec.ExpectedSHA256 = aggregate
	initial, err := decoder.InitializeAdapter(context.Background(), spec)
	if err != nil {
		t.Fatal(err)
	}
	defer initial.Close()
	shards := provider(t, p, nil)
	model, err := decoder.LoadTextAssembly(context.Background(), p, shards, initial)
	if model != nil || !errors.Is(err, torch.ErrCUDAUnavailable) || shards.opens != 0 {
		t.Fatalf("pinned full initializer was not admitted by the derived rows: %v", err)
	}
}

// Without a declaration the derivation rebuilds the q/v reference, and its
// aggregate is the historical pin: the q/v rows are that initializer's bytes.
func TestDerivedQVReferenceIsTheHistoricalInitializer(t *testing.T) {
	d := fixture()
	index, config, source, _ := d.encode(t)
	limits := admittedLimits()
	document, aggregate, err := decoder.DeriveInitialReference(context.Background(), source, config, limits, derivedSpec(t, config, nil))
	if err != nil {
		t.Fatal(err)
	}
	if aggregate != "75185d68fcfdd09bb2dcb6dffe0902a35300f52d9fda716b408189991773aa7c" {
		t.Fatalf("q/v derivation left the historical pin: %s", aggregate)
	}
	identity := decoder.AssemblyIdentity{IndexSHA256: fmt.Sprintf("%x", sha256.Sum256(index)), ConfigSHA256: fmt.Sprintf("%x", sha256.Sum256(config)),
		ReferenceSHA256: fmt.Sprintf("%x", sha256.Sum256(document)), InitialAdapterSHA256: aggregate}
	p, err := decoder.PlanTextAssembly(index, config, document, identity, limits)
	if err != nil {
		t.Fatal(err)
	}
	if layoutDigest(p) == "9c8f4149e26aa9cdb4ddf9db0ae724a9ecb6c989122491b3b3fd6af09010ef44" {
		t.Fatal("the fixture's placeholder adapter digests survived the derivation")
	}
	derived := derivedRows(t, document)
	for _, row := range d.Reference {
		got := derived[row.Name]
		if !row.Grad {
			if fmt.Sprint(got) != fmt.Sprint(row) {
				t.Fatalf("q/v derivation changed a frozen row: %+v", got)
			}
			continue
		}
		row.Hash = got.Hash
		if fmt.Sprint(got) != fmt.Sprint(row) || got.Hash == strings.Repeat("0", 64) {
			t.Fatalf("q/v adapter row differs: %+v", got)
		}
	}
}

func TestDerivationRefusesAnInitializerOtherThanTheCoverages(t *testing.T) {
	_, config, source, _ := fixture().encode(t)
	limits := fullLimits()
	for name, change := range map[string]func(*decoder.InitialAdapterSpec, *[]byte){
		"pinned":          func(s *decoder.InitialAdapterSpec, _ *[]byte) { s.ExpectedSHA256 = strings.Repeat("a", 64) },
		"q_v_projections": func(s *decoder.InitialAdapterSpec, _ *[]byte) { *s = derivedSpec(t, config, nil) },
		"another_rank":    func(s *decoder.InitialAdapterSpec, _ *[]byte) { s.Projections[0].Rank = 8 },
		"reordered_projections": func(s *decoder.InitialAdapterSpec, _ *[]byte) {
			s.Projections[0], s.Projections[1] = s.Projections[1], s.Projections[0]
		},
		"unknown_row_field": func(_ *decoder.InitialAdapterSpec, r *[]byte) {
			*r = []byte(strings.Replace(string(*r), `"kind"`, `"extra":1,"kind"`, 1))
		},
		"missing_adapted_base": func(_ *decoder.InitialAdapterSpec, r *[]byte) {
			*r = []byte(strings.Replace(string(*r), "layers.0.mlp.up_proj.weight", "layers.0.mlp.upper.weight", 1))
		},
		"duplicate_adapted_base": func(_ *decoder.InitialAdapterSpec, r *[]byte) {
			*r = []byte(strings.Replace(string(*r), "layers.3.self_attn.k_proj.weight", "layers.3.self_attn.q_proj.weight", 1))
		},
	} {
		t.Run(name, func(t *testing.T) {
			spec, reference := derivedSpec(t, config, limits.AdapterCoverage), append([]byte(nil), source...)
			change(&spec, &reference)
			if document, aggregate, err := decoder.DeriveInitialReference(context.Background(), reference, config, limits, spec); document != nil || aggregate != "" || err == nil {
				t.Fatal("derivation admitted", name)
			}
		})
	}
}
