package fusioncache_test

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"maps"
	"math"
	"math/rand/v2"
	"reflect"
	"slices"
	"strings"
	"testing"

	"github.com/tayi-ai/arandu-llama/training/fusioncache"
)

// routedPackageSHA256 pins the canonical encoding of routedPackage. A renamed
// JSON key, a reordered field or a changed number format moves it, and every
// frozen package would stop matching its own name.
const routedPackageSHA256 = "29294888bbbc7f0833c533079b2beaa39861ff5440709e93db28ab992831f0cb"

func hashOf(label string) string {
	sum := sha256.Sum256([]byte(label))
	return hex.EncodeToString(sum[:])
}

func packageLimits() fusioncache.PackageLimits {
	return fusioncache.PackageLimits{MaxBytes: 1 << 16, MaxTeachers: 4, MaxMaskSpans: 8, MaxArtifacts: 64, MaxPackages: 8}
}

func teacherModel(name string) fusioncache.ModelIdentity {
	return fusioncache.ModelIdentity{Name: name, Revision: hashOf(name + "/revision")[:40], WeightsSHA256: hashOf(name + "/weights"), TokenizerSHA256: hashOf(name + "/tokenizer"), TemplateSHA256: hashOf(name + "/template"), RuntimeSHA256: hashOf("runtime"), Vocabulary: 200000}
}

// artifactIndex labels every digest the package cites by the role it plays.
func artifactIndex(p fusioncache.CapabilityPackage) map[string]string {
	index := map[string]string{}
	add := func(digest, kind string) {
		if digest != "" {
			index[digest] = kind
		}
	}
	add(p.DatasetSHA256, "dataset")
	add(p.ContextSHA256, "context")
	add(p.Verifier.ReceiptSHA256, "verdict")
	add(p.Protection.ManifestSHA256, "protection-manifest")
	add(p.Signal.RefSHA256, "signal")
	if p.Signal.Mask != nil {
		add(p.Signal.Mask.TokenizerSHA256, "tokenizer")
	}
	if p.Lineage != nil {
		add(p.Lineage.RegistrySHA256, "registry")
	}
	for _, t := range p.Teachers {
		add(t.Model.WeightsSHA256, "weights")
		add(t.Model.TokenizerSHA256, "tokenizer")
		add(t.Model.TemplateSHA256, "template")
		add(t.Model.RuntimeSHA256, "runtime")
		add(t.License.ReceiptSHA256, "license-receipt")
		add(t.Outcome.GenerationReceiptSHA256, "generation-receipt")
	}
	if p.Router != nil {
		add(p.Router.ConfigSHA256, "router-config")
		add(p.Router.CompetenceSHA256, "competence-matrix")
		add(p.Router.RegistrySHA256, "registry")
	}
	return index
}

// routedPackage is an arm D unit: two teachers chosen by the router before
// generation, the first one budget-exhausted, the trajectory admitted from the
// one the verifier accepted.
func routedPackage(t *testing.T) fusioncache.CapabilityPackage {
	t.Helper()
	registry := hashOf("registry")
	p := fusioncache.CapabilityPackage{
		Schema:        fusioncache.CapabilityPackageSchema,
		Revision:      1,
		CapabilityID:  "reasoning.math.gsm8k",
		DatasetSHA256: hashOf("dataset"),
		ExampleID:     "gsm8k-train-1614",
		Role:          "train",
		ContextSHA256: fusioncache.ContextDigest([]byte("<|im_start|>user\nHow much time will he save?<|im_end|>\n<|im_start|>assistant\n")),
		Verifier:      fusioncache.Verification{ID: "math.numeric_exact_rational.v1", Correct: true, ReceiptSHA256: hashOf("verdict")},
		Teachers: []fusioncache.ConsultedTeacher{
			{ID: "gemma-4-12b-it-qat-q4-0", Family: "gemma", Model: teacherModel("gemma"), License: fusioncache.TeacherLicense{SPDX: "Apache-2.0", ReceiptSHA256: hashOf("gemma/license")}, Score: -0.25, Weight: 0.4, Outcome: fusioncache.TeacherOutcome{BudgetExhausted: true, GenerationReceiptSHA256: hashOf("gemma/generation")}},
			{ID: "gpt-oss-20b-mxfp4", Family: "gpt-oss", Model: teacherModel("gpt-oss"), License: fusioncache.TeacherLicense{SPDX: "Apache-2.0", ReceiptSHA256: hashOf("gpt-oss/license")}, Score: 0.5, Weight: 0.6, Outcome: fusioncache.TeacherOutcome{Correct: true, GenerationReceiptSHA256: hashOf("gpt-oss/generation")}},
		},
		Signal: fusioncache.AdmittedSignal{Kind: fusioncache.SignalTrajectory, Source: "gpt-oss-20b-mxfp4", RefSHA256: hashOf("sft-line"), Mask: &fusioncache.PositionMask{TokenizerSHA256: hashOf("student/tokenizer"), Tokens: 412, Spans: []fusioncache.MaskSpan{{Start: 96, End: 412}}}},
		Lineage: &fusioncache.LineageSnapshot{RegistrySHA256: registry, Roots: map[string][]string{
			"gemma-4-12b-it-qat-q4-0": {"gemma"},
			"gpt-oss-20b-mxfp4":       {"gpt-oss"},
		}},
		Protection: fusioncache.ProtectionCohort{ID: "tcap-protection-v1", ManifestSHA256: hashOf("protection")},
		Router: &fusioncache.RouterDecision{
			Method: "tcap.topk_lineage.v1", ConfigSHA256: hashOf("router/config"), CompetenceSHA256: hashOf("competence"), RegistrySHA256: registry,
			Ranking:   []fusioncache.RankedTeacher{{Teacher: "gpt-oss-20b-mxfp4", Score: 0.5}, {Teacher: "gemma-4-12b-it-qat-q4-0", Score: -0.25}},
			Reason:    "top-2 by reasoning.math.gsm8k competence across independent lineage roots",
			DecidedAt: fusioncache.LogicalInstant{Clock: "tcap-arm-d-seed-1", Tick: 17},
		},
	}
	p.Artifacts = artifactIndex(p)
	if err := fusioncache.ValidatePackage(p, packageLimits()); err != nil {
		t.Fatal(err)
	}
	return p
}

// groundTruthPackage is an arm A unit: the dataset's reference, no teacher.
func groundTruthPackage(t *testing.T) fusioncache.CapabilityPackage {
	t.Helper()
	p := fusioncache.CapabilityPackage{
		Schema: fusioncache.CapabilityPackageSchema, Revision: 1,
		CapabilityID: "reasoning.math.competition", DatasetSHA256: hashOf("dataset"), ExampleID: "math-train-0042", Role: "train",
		ContextSHA256: fusioncache.ContextDigest([]byte("What is 1/2 + 1/3?")),
		Verifier:      fusioncache.Verification{ID: "math.numeric_exact_rational.v1", Correct: true, ReceiptSHA256: hashOf("reference/verdict")},
		Signal:        fusioncache.AdmittedSignal{Kind: fusioncache.SignalGroundTruth, RefSHA256: hashOf("reference-line")},
		Protection:    fusioncache.ProtectionCohort{ID: "tcap-protection-v1", ManifestSHA256: hashOf("protection")},
	}
	p.Artifacts = artifactIndex(p)
	if err := fusioncache.ValidatePackage(p, packageLimits()); err != nil {
		t.Fatal(err)
	}
	return p
}

// weightedLogits turns the routed package into a weighted logits unit.
func weightedLogits(p *fusioncache.CapabilityPackage) {
	p.Signal = fusioncache.AdmittedSignal{Kind: fusioncache.SignalLogits, Weighted: true, RefSHA256: p.Signal.RefSHA256, Mask: p.Signal.Mask}
}

func packageDigest(t *testing.T, p fusioncache.CapabilityPackage) string {
	t.Helper()
	digest, err := fusioncache.PackageDigest(p, packageLimits())
	if err != nil {
		t.Fatal(err)
	}
	return digest
}

func encodePackage(t *testing.T, p fusioncache.CapabilityPackage) []byte {
	t.Helper()
	encoded, _, err := fusioncache.EncodePackage(p, packageLimits())
	if err != nil {
		t.Fatal(err)
	}
	return encoded
}

// shuffledJSON writes value with the keys of every object in a random order.
func shuffledJSON(t *testing.T, value any, r *rand.Rand) []byte {
	t.Helper()
	var b bytes.Buffer
	var write func(any)
	write = func(v any) {
		switch v := v.(type) {
		case map[string]any:
			keys := slices.Sorted(maps.Keys(v))
			r.Shuffle(len(keys), func(i, j int) { keys[i], keys[j] = keys[j], keys[i] })
			b.WriteByte('{')
			for i, key := range keys {
				if i > 0 {
					b.WriteByte(',')
				}
				encoded, _ := json.Marshal(key)
				b.Write(encoded)
				b.WriteByte(':')
				write(v[key])
			}
			b.WriteByte('}')
		case []any:
			b.WriteByte('[')
			for i, element := range v {
				if i > 0 {
					b.WriteByte(',')
				}
				write(element)
			}
			b.WriteByte(']')
		default:
			encoded, err := json.Marshal(v)
			if err != nil {
				t.Fatal(err)
			}
			b.Write(encoded)
		}
	}
	write(value)
	return b.Bytes()
}

func genericJSON(t *testing.T, data []byte) any {
	t.Helper()
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.UseNumber()
	var value any
	if err := decoder.Decode(&value); err != nil {
		t.Fatal(err)
	}
	return value
}

func TestThePackageDigestDoesNotDependOnMapOrKeyOrder(t *testing.T) {
	p := routedPackage(t)
	want := packageDigest(t, p)
	if want != routedPackageSHA256 {
		t.Fatalf("the canonical encoding moved: digest %s, pinned %s", want, routedPackageSHA256)
	}
	if plain := digest(t, p); plain != want {
		t.Fatalf("package digest %s differs from Digest %s", want, plain)
	}
	if sum := sha256.Sum256(encodePackage(t, p)); hex.EncodeToString(sum[:]) != want {
		t.Fatal("the digest is not the SHA-256 of the canonical bytes")
	}
	r := rand.New(rand.NewPCG(127, 7))
	canonical := encodePackage(t, p)
	differentBytes := 0
	for range 64 {
		shuffled := clone(t, p)
		keys := slices.Sorted(maps.Keys(p.Artifacts))
		r.Shuffle(len(keys), func(i, j int) { keys[i], keys[j] = keys[j], keys[i] })
		shuffled.Artifacts = map[string]string{}
		for _, key := range keys {
			shuffled.Artifacts[key] = p.Artifacts[key]
		}
		teachers := slices.Sorted(maps.Keys(p.Lineage.Roots))
		r.Shuffle(len(teachers), func(i, j int) { teachers[i], teachers[j] = teachers[j], teachers[i] })
		shuffled.Lineage.Roots = map[string][]string{}
		for _, teacher := range teachers {
			shuffled.Lineage.Roots[teacher] = p.Lineage.Roots[teacher]
		}
		if got := packageDigest(t, shuffled); got != want {
			t.Fatalf("map insertion order moved the digest: %s", got)
		}

		document := shuffledJSON(t, genericJSON(t, canonical), r)
		if !bytes.Equal(document, canonical) {
			differentBytes++
		}
		decoded, got, err := fusioncache.DecodePackage(document, packageLimits())
		if err != nil {
			t.Fatalf("shuffled keys refused: %v\n%s", err, document)
		}
		if got != want || !reflect.DeepEqual(decoded, p) {
			t.Fatalf("shuffled keys moved the digest to %s", got)
		}
	}
	if differentBytes == 0 {
		t.Fatal("no shuffled document differed from the canonical bytes")
	}
}

func TestThePackageRoundTripsThroughJSON(t *testing.T) {
	for name, p := range map[string]fusioncache.CapabilityPackage{"routed": routedPackage(t), "ground_truth": groundTruthPackage(t)} {
		t.Run(name, func(t *testing.T) {
			encoded, want, err := fusioncache.EncodePackage(p, packageLimits())
			if err != nil {
				t.Fatal(err)
			}
			decoded, got, err := fusioncache.DecodePackage(encoded, packageLimits())
			if err != nil {
				t.Fatal(err)
			}
			if got != want || !reflect.DeepEqual(decoded, p) {
				t.Fatalf("round trip changed the package: %s != %s", got, want)
			}
			indented, err := json.MarshalIndent(p, "", "  ")
			if err != nil {
				t.Fatal(err)
			}
			if _, got, err := fusioncache.DecodePackage(indented, packageLimits()); err != nil || got != want {
				t.Fatalf("whitespace changed the digest or was refused: %s, %v", got, err)
			}
		})
	}
}

func TestCanonicalPackageOrdersWhatHasNoOrder(t *testing.T) {
	p := routedPackage(t)
	want := packageDigest(t, p)
	scrambled := clone(t, p)
	scrambled.Teachers[0], scrambled.Teachers[1] = scrambled.Teachers[1], scrambled.Teachers[0]
	scrambled.Lineage.Roots["gpt-oss-20b-mxfp4"] = []string{"zeta", "gpt-oss"}
	scrambled.Signal.Mask.Spans = []fusioncache.MaskSpan{{Start: 200, End: 412}, {Start: 96, End: 150}, {Start: 150, End: 210}}
	if err := fusioncache.ValidatePackage(scrambled, packageLimits()); err == nil || !strings.Contains(err.Error(), "ordered by id") {
		t.Fatalf("an unordered package was admitted as it was: %v", err)
	}
	before := clone(t, scrambled)
	canonical := fusioncache.CanonicalPackage(scrambled)
	if !reflect.DeepEqual(scrambled, before) {
		t.Fatal("CanonicalPackage changed its argument")
	}
	if got := canonical.Signal.Mask.Spans; !slices.Equal(got, []fusioncache.MaskSpan{{Start: 96, End: 412}}) {
		t.Fatalf("mask spans were not merged: %v", got)
	}
	if got := canonical.Lineage.Roots["gpt-oss-20b-mxfp4"]; !slices.Equal(got, []string{"gpt-oss", "zeta"}) {
		t.Fatalf("roots were not sorted: %v", got)
	}
	canonical.Lineage.Roots["gpt-oss-20b-mxfp4"] = []string{"gpt-oss"}
	if got := packageDigest(t, canonical); got != want {
		t.Fatalf("canonical order moved the digest: %s", got)
	}
	duplicated := clone(t, p)
	duplicated.Teachers = append(duplicated.Teachers, duplicated.Teachers[0])
	if err := fusioncache.ValidatePackage(fusioncache.CanonicalPackage(duplicated), packageLimits()); err == nil {
		t.Fatal("CanonicalPackage hid a duplicated teacher")
	}
}

func TestThePackageRefusesWhatItCannotStandBehind(t *testing.T) {
	type change func(*fusioncache.CapabilityPackage)
	gpt, gemma := "gpt-oss-20b-mxfp4", "gemma-4-12b-it-qat-q4-0"
	cases := map[string]struct {
		change change
		want   string
	}{
		"schema":                  {func(p *fusioncache.CapabilityPackage) { p.Schema = 2 }, "unsupported capability package schema"},
		"revision_zero":           {func(p *fusioncache.CapabilityPackage) { p.Revision = 0 }, "revision 1 supersedes nothing"},
		"first_supersedes":        {func(p *fusioncache.CapabilityPackage) { p.Supersedes = hashOf("previous") }, "revision 1 supersedes nothing"},
		"second_names_nothing":    {func(p *fusioncache.CapabilityPackage) { p.Revision = 2 }, "revision 1 supersedes nothing"},
		"supersedes_not_a_digest": {func(p *fusioncache.CapabilityPackage) { p.Revision, p.Supersedes = 2, strings.Repeat("A", 64) }, "revision 1 supersedes nothing"},
		"capability_case":         {func(p *fusioncache.CapabilityPackage) { p.CapabilityID = "Reasoning.Math" }, "invalid capability id"},
		"capability_empty":        {func(p *fusioncache.CapabilityPackage) { p.CapabilityID = "" }, "invalid capability id"},
		"dataset_short":           {func(p *fusioncache.CapabilityPackage) { p.DatasetSHA256 = p.DatasetSHA256[:63] }, "invalid example identity"},
		"dataset_uppercase":       {func(p *fusioncache.CapabilityPackage) { p.DatasetSHA256 = strings.ToUpper(p.DatasetSHA256) }, "invalid example identity"},
		"example_space":           {func(p *fusioncache.CapabilityPackage) { p.ExampleID = "gsm8k train 1614" }, "invalid example identity"},
		"example_empty":           {func(p *fusioncache.CapabilityPackage) { p.ExampleID = "" }, "invalid example identity"},
		"role_sealed_final":       {func(p *fusioncache.CapabilityPackage) { p.Role = "sealed-final" }, "invalid example identity"},
		"context_not_hex":         {func(p *fusioncache.CapabilityPackage) { p.ContextSHA256 = strings.Repeat("g", 64) }, "invalid example identity"},
		"verifier_unversioned":    {func(p *fusioncache.CapabilityPackage) { p.Verifier.ID = "math.numeric_exact_rational" }, "a versioned verifier id"},
		"verifier_receipt":        {func(p *fusioncache.CapabilityPackage) { p.Verifier.ReceiptSHA256 = "" }, "a versioned verifier id"},
		"verifier_rejected":       {func(p *fusioncache.CapabilityPackage) { p.Verifier.Correct = false }, "correct verdict"},
		"protection_missing":      {func(p *fusioncache.CapabilityPackage) { p.Protection.ID = "" }, "protection cohort"},
		"protection_manifest":     {func(p *fusioncache.CapabilityPackage) { p.Protection.ManifestSHA256 = "0" }, "protection cohort"},
		"teachers_unordered": {func(p *fusioncache.CapabilityPackage) {
			p.Teachers[0], p.Teachers[1] = p.Teachers[1], p.Teachers[0]
		}, "unique and ordered by id"},
		"teacher_twice":            {func(p *fusioncache.CapabilityPackage) { p.Teachers[1].ID = p.Teachers[0].ID }, "unique and ordered by id"},
		"teacher_id":               {func(p *fusioncache.CapabilityPackage) { p.Teachers[0].ID = "Gemma 4" }, "invalid teacher or family id"},
		"teacher_family":           {func(p *fusioncache.CapabilityPackage) { p.Teachers[0].Family = "" }, "invalid teacher or family id"},
		"teacher_revision":         {func(p *fusioncache.CapabilityPackage) { p.Teachers[0].Model.Revision = "main" }, "invalid model identity"},
		"teacher_weights":          {func(p *fusioncache.CapabilityPackage) { p.Teachers[0].Model.WeightsSHA256 = "" }, "invalid model identity"},
		"teacher_template":         {func(p *fusioncache.CapabilityPackage) { p.Teachers[1].Model.TemplateSHA256 = "x" }, "invalid model identity"},
		"license_missing":          {func(p *fusioncache.CapabilityPackage) { p.Teachers[0].License.SPDX = "" }, "SPDX license"},
		"license_double_space":     {func(p *fusioncache.CapabilityPackage) { p.Teachers[0].License.SPDX = "Apache-2.0  AND MIT" }, "SPDX license"},
		"license_receipt":          {func(p *fusioncache.CapabilityPackage) { p.Teachers[0].License.ReceiptSHA256 = "" }, "SPDX license"},
		"score_nan":                {func(p *fusioncache.CapabilityPackage) { p.Teachers[0].Score = math.NaN() }, "finite score"},
		"score_infinite":           {func(p *fusioncache.CapabilityPackage) { p.Teachers[0].Score = math.Inf(-1) }, "finite score"},
		"score_negative_zero":      {func(p *fusioncache.CapabilityPackage) { p.Teachers[0].Score = math.Copysign(0, -1) }, "finite score"},
		"weight_negative":          {func(p *fusioncache.CapabilityPackage) { p.Teachers[0].Weight = -0.1 }, "weight in [0, 1]"},
		"weight_above_one":         {func(p *fusioncache.CapabilityPackage) { p.Teachers[0].Weight = 1.5 }, "weight in [0, 1]"},
		"weight_nan":               {func(p *fusioncache.CapabilityPackage) { p.Teachers[0].Weight = math.NaN() }, "weight in [0, 1]"},
		"weight_negative_zero":     {func(p *fusioncache.CapabilityPackage) { p.Teachers[0].Weight = math.Copysign(0, -1) }, "weight in [0, 1]"},
		"correct_after_exhaustion": {func(p *fusioncache.CapabilityPackage) { p.Teachers[1].Outcome.BudgetExhausted = true }, "cannot be correct after exhausting"},
		"generation_receipt":       {func(p *fusioncache.CapabilityPackage) { p.Teachers[1].Outcome.GenerationReceiptSHA256 = "" }, "generation receipt digest"},
		"one_model_twice":          {func(p *fusioncache.CapabilityPackage) { p.Teachers[0].Model = p.Teachers[1].Model }, "are one model"},
		"too_many_teachers": {func(p *fusioncache.CapabilityPackage) {
			for i := range 3 {
				extra := p.Teachers[1]
				extra.ID = "qwen-" + string(rune('a'+i))
				p.Teachers = append(p.Teachers, extra)
			}
		}, "exceed the bound"},
		"lineage_missing":  {func(p *fusioncache.CapabilityPackage) { p.Lineage = nil }, "need a lineage snapshot"},
		"lineage_registry": {func(p *fusioncache.CapabilityPackage) { p.Lineage.RegistrySHA256 = "" }, "roots of exactly the consulted teachers"},
		"lineage_extra":    {func(p *fusioncache.CapabilityPackage) { p.Lineage.Roots["qwen3-8-27b"] = []string{"qwen"} }, "roots of exactly the consulted teachers"},
		"lineage_other_teacher": {func(p *fusioncache.CapabilityPackage) {
			delete(p.Lineage.Roots, gemma)
			p.Lineage.Roots["qwen3-8-27b"] = []string{"qwen"}
		}, "has no roots for teacher"},
		"lineage_no_roots":   {func(p *fusioncache.CapabilityPackage) { p.Lineage.Roots[gpt] = nil }, "has no roots for teacher"},
		"lineage_unordered":  {func(p *fusioncache.CapabilityPackage) { p.Lineage.Roots[gpt] = []string{"zeta", "gpt-oss"} }, "unique and ascending"},
		"lineage_root_twice": {func(p *fusioncache.CapabilityPackage) { p.Lineage.Roots[gpt] = []string{"gpt-oss", "gpt-oss"} }, "unique and ascending"},
		"family_two_roots":   {func(p *fusioncache.CapabilityPackage) { p.Teachers[0].Family = "gpt-oss" }, "two different lineage roots"},
		"signal_digest":      {func(p *fusioncache.CapabilityPackage) { p.Signal.RefSHA256 = "" }, "signal needs its digest"},
		"signal_kind":        {func(p *fusioncache.CapabilityPackage) { p.Signal.Kind = "votes" }, "unknown signal kind"},
		"ground_truth_source": {func(p *fusioncache.CapabilityPackage) {
			p.Signal.Kind = fusioncache.SignalGroundTruth
		}, "ground truth signal has no teacher source"},
		"ground_truth_weighted": {func(p *fusioncache.CapabilityPackage) {
			p.Signal.Kind, p.Signal.Source, p.Signal.Weighted = fusioncache.SignalGroundTruth, "", true
		}, "ground truth signal has no teacher source"},
		"trajectory_outsider":  {func(p *fusioncache.CapabilityPackage) { p.Signal.Source = "qwen3-8-27b" }, "is not a consulted teacher"},
		"trajectory_unchecked": {func(p *fusioncache.CapabilityPackage) { p.Signal.Source = gemma }, "was not verified correct"},
		"trajectory_weighted":  {func(p *fusioncache.CapabilityPackage) { p.Signal.Weighted = true }, "never weighted"},
		"seqkd_unchecked": {func(p *fusioncache.CapabilityPackage) {
			p.Signal.Kind, p.Signal.Source = fusioncache.SignalSeqKD, gemma
		}, "was not verified correct"},
		"logits_without_mask": {func(p *fusioncache.CapabilityPackage) {
			p.Signal.Kind, p.Signal.Mask = fusioncache.SignalLogits, nil
			p.Artifacts = artifactIndex(*p)
		}, "needs its position mask"},
		"features_without_mask": {func(p *fusioncache.CapabilityPackage) {
			p.Signal.Kind, p.Signal.Mask = fusioncache.SignalFeatures, nil
			p.Artifacts = artifactIndex(*p)
		}, "needs its position mask"},
		"logits_outsider": {func(p *fusioncache.CapabilityPackage) {
			p.Signal.Kind, p.Signal.Source = fusioncache.SignalLogits, "qwen3-8-27b"
		}, "is not a consulted teacher"},
		"weighted_with_source": {func(p *fusioncache.CapabilityPackage) {
			weightedLogits(p)
			p.Signal.Source = gpt
		}, "names no single source"},
		"weights_above_one": {func(p *fusioncache.CapabilityPackage) {
			weightedLogits(p)
			p.Teachers[0].Weight = 0.6
		}, "sum to at most 1"},
		"weights_all_zero": {func(p *fusioncache.CapabilityPackage) {
			weightedLogits(p)
			p.Teachers[0].Weight, p.Teachers[1].Weight = 0, 0
		}, "sum to at most 1"},
		"mask_tokenizer":  {func(p *fusioncache.CapabilityPackage) { p.Signal.Mask.TokenizerSHA256 = "" }, "mask needs a tokenizer digest"},
		"mask_no_spans":   {func(p *fusioncache.CapabilityPackage) { p.Signal.Mask.Spans = nil }, "mask needs a tokenizer digest"},
		"mask_no_tokens":  {func(p *fusioncache.CapabilityPackage) { p.Signal.Mask.Tokens = 0 }, "mask needs a tokenizer digest"},
		"mask_empty_span": {func(p *fusioncache.CapabilityPackage) { p.Signal.Mask.Spans[0].End = 96 }, "mask spans must be nonempty"},
		"mask_beyond_end": {func(p *fusioncache.CapabilityPackage) { p.Signal.Mask.Spans[0].End = 413 }, "mask spans must be nonempty"},
		"mask_negative":   {func(p *fusioncache.CapabilityPackage) { p.Signal.Mask.Spans[0].Start = -1 }, "mask spans must be nonempty"},
		"mask_adjacent": {func(p *fusioncache.CapabilityPackage) {
			p.Signal.Mask.Spans = []fusioncache.MaskSpan{{Start: 0, End: 5}, {Start: 5, End: 9}}
		}, "mask spans must be nonempty"},
		"mask_overlapping": {func(p *fusioncache.CapabilityPackage) {
			p.Signal.Mask.Spans = []fusioncache.MaskSpan{{Start: 0, End: 6}, {Start: 5, End: 9}}
		}, "mask spans must be nonempty"},
		"mask_unordered": {func(p *fusioncache.CapabilityPackage) {
			p.Signal.Mask.Spans = []fusioncache.MaskSpan{{Start: 7, End: 9}, {Start: 0, End: 5}}
		}, "mask spans must be nonempty"},
		"mask_too_many_spans": {func(p *fusioncache.CapabilityPackage) {
			p.Signal.Mask.Spans = nil
			for start := 0; start < 18; start += 2 {
				p.Signal.Mask.Spans = append(p.Signal.Mask.Spans, fusioncache.MaskSpan{Start: start, End: start + 1})
			}
		}, "mask needs a tokenizer digest"},
		"router_unversioned": {func(p *fusioncache.CapabilityPackage) { p.Router.Method = "tcap.topk_lineage" }, "versioned id"},
		"router_config":      {func(p *fusioncache.CapabilityPackage) { p.Router.ConfigSHA256 = "" }, "router config, competence and registry"},
		"router_competence":  {func(p *fusioncache.CapabilityPackage) { p.Router.CompetenceSHA256 = "" }, "router config, competence and registry"},
		"router_no_reason":   {func(p *fusioncache.CapabilityPackage) { p.Router.Reason = "" }, "router reason"},
		"router_padded":      {func(p *fusioncache.CapabilityPackage) { p.Router.Reason += "\n" }, "router reason"},
		"router_tick":        {func(p *fusioncache.CapabilityPackage) { p.Router.DecidedAt.Tick = 0 }, "positive tick"},
		"router_clock":       {func(p *fusioncache.CapabilityPackage) { p.Router.DecidedAt.Clock = "" }, "positive tick"},
		"router_empty":       {func(p *fusioncache.CapabilityPackage) { p.Router.Ranking = nil }, "router ranking needs"},
		"router_outsider": {func(p *fusioncache.CapabilityPackage) {
			p.Router.Ranking = append(p.Router.Ranking, fusioncache.RankedTeacher{Teacher: "qwen3-8-27b", Score: 0.1})
		}, "not a consulted teacher"},
		"router_twice": {func(p *fusioncache.CapabilityPackage) {
			p.Router.Ranking = []fusioncache.RankedTeacher{p.Router.Ranking[0], p.Router.Ranking[0]}
		}, "twice"},
		"router_score": {func(p *fusioncache.CapabilityPackage) { p.Router.Ranking[1].Score = -0.5 }, "differs from the score its teacher records"},
		"router_unchosen_teacher": {func(p *fusioncache.CapabilityPackage) {
			p.Router.Ranking = p.Router.Ranking[:1]
		}, "consults only the teachers the router chose"},
		"router_registry":  {func(p *fusioncache.CapabilityPackage) { p.Router.RegistrySHA256 = hashOf("other registry") }, "different registries"},
		"artifact_missing": {func(p *fusioncache.CapabilityPackage) { delete(p.Artifacts, p.ContextSHA256) }, "exactly the"},
		"artifact_extra":   {func(p *fusioncache.CapabilityPackage) { p.Artifacts[hashOf("uncited")] = "weights" }, "exactly the"},
		"artifact_kind":    {func(p *fusioncache.CapabilityPackage) { p.Artifacts[p.ContextSHA256] = "Rendered Prompt" }, "kind labels"},
		"artifact_digest":  {func(p *fusioncache.CapabilityPackage) { p.Artifacts["context"] = "context" }, "kind labels"},
	}
	for name, c := range cases {
		t.Run(name, func(t *testing.T) {
			bad := clone(t, routedPackage(t))
			c.change(&bad)
			err := fusioncache.ValidatePackage(bad, packageLimits())
			if !errors.Is(err, fusioncache.ErrContract) || !strings.Contains(err.Error(), c.want) {
				t.Fatalf("want a refusal containing %q, got %v", c.want, err)
			}
			if _, err := fusioncache.PackageDigest(bad, packageLimits()); err == nil {
				t.Fatal("a refused package received a digest")
			}
		})
	}

	ground := groundTruthPackage(t)
	lineage := clone(t, ground)
	lineage.Lineage = &fusioncache.LineageSnapshot{RegistrySHA256: hashOf("registry"), Roots: map[string][]string{}}
	lineage.Artifacts = artifactIndex(lineage)
	if err := fusioncache.ValidatePackage(lineage, packageLimits()); err == nil || !strings.Contains(err.Error(), "without consulted teachers") {
		t.Fatalf("a lineage snapshot without teachers was admitted: %v", err)
	}
	routed := clone(t, ground)
	routed.Router = clone(t, routedPackage(t)).Router
	routed.Artifacts = artifactIndex(routed)
	if err := fusioncache.ValidatePackage(routed, packageLimits()); err == nil || !strings.Contains(err.Error(), "not a consulted teacher") {
		t.Fatalf("a router decision without consulted teachers was admitted: %v", err)
	}
	if err := fusioncache.ValidatePackage(ground, fusioncache.PackageLimits{}); err == nil || !strings.Contains(err.Error(), "explicit positive package bounds") {
		t.Fatalf("zero limits admitted: %v", err)
	}
}

func TestSignalKindsTheirReferencesAdmit(t *testing.T) {
	cases := map[string]func(*fusioncache.CapabilityPackage){
		"trajectory_without_mask": func(p *fusioncache.CapabilityPackage) {
			p.Signal.Mask = nil
			p.Artifacts = artifactIndex(*p)
		},
		"seqkd":                func(p *fusioncache.CapabilityPackage) { p.Signal.Kind = fusioncache.SignalSeqKD },
		"logits_single_source": func(p *fusioncache.CapabilityPackage) { p.Signal.Kind = fusioncache.SignalLogits },
		"features_weighted": func(p *fusioncache.CapabilityPackage) {
			weightedLogits(p)
			p.Signal.Kind = fusioncache.SignalFeatures
		},
		"weights_below_one": func(p *fusioncache.CapabilityPackage) {
			weightedLogits(p)
			p.Teachers[0].Weight, p.Teachers[1].Weight = 0, 0.25
		},
		"weights_rounded_above_one": func(p *fusioncache.CapabilityPackage) {
			weightedLogits(p)
			p.Teachers[0].Weight, p.Teachers[1].Weight = 0.1, 0.9000000000000001
		},
		"fallback_to_ground_truth": func(p *fusioncache.CapabilityPackage) {
			p.Signal.Kind, p.Signal.Source = fusioncache.SignalGroundTruth, ""
		},
		"unrouted": func(p *fusioncache.CapabilityPackage) {
			p.Router = nil
			p.Artifacts = artifactIndex(*p)
		},
	}
	for name, change := range cases {
		t.Run(name, func(t *testing.T) {
			p := clone(t, routedPackage(t))
			change(&p)
			if err := fusioncache.ValidatePackage(p, packageLimits()); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestDecodingRefusesJSONTwoReadersCouldReadDifferently(t *testing.T) {
	routed := encodePackage(t, routedPackage(t))
	ground := encodePackage(t, groundTruthPackage(t))
	var loose fusioncache.CapabilityPackage
	caseVariant := bytes.Replace(routed, []byte(`"capability_id"`), []byte(`"Capability_ID"`), 1)
	if err := json.Unmarshal(caseVariant, &loose); err != nil || loose.CapabilityID != "reasoning.math.gsm8k" {
		t.Fatalf("encoding/json no longer matches keys by case, so this case proves nothing: %v", err)
	}
	invalid := bytes.Replace(routed, []byte("independent lineage roots"), []byte("independent lineage \xff roots"), 1)
	cases := map[string]struct {
		data []byte
		want string
	}{
		"unknown_key":         {bytes.Replace(routed, []byte(`{"schema"`), []byte(`{"extra":1,"schema"`), 1), "unknown field"},
		"unknown_nested_key":  {bytes.Replace(routed, []byte(`"correct":true,"receipt_sha256"`), []byte(`"correct":true,"votes":3,"receipt_sha256"`), 1), "unknown field"},
		"case_variant_key":    {caseVariant, "canonical form"},
		"duplicate_key":       {bytes.Replace(routed, []byte(`{"schema"`), []byte(`{"capability_id":"coding.repository","schema"`), 1), "appears twice"},
		"null_optional":       {bytes.Replace(ground, []byte(`"signal"`), []byte(`"router":null,"signal"`), 1), "canonical form"},
		"empty_teachers":      {bytes.Replace(ground, []byte(`"signal"`), []byte(`"teachers":[],"signal"`), 1), "canonical form"},
		"empty_supersedes":    {bytes.Replace(ground, []byte(`"revision":1`), []byte(`"revision":1,"supersedes":""`), 1), "canonical form"},
		"trailing_data":       {append(slices.Clone(routed), []byte(`{}`)...), "trailing JSON"},
		"invalid_utf8":        {invalid, "not UTF-8"},
		"invalid_package":     {bytes.Replace(routed, []byte(`"weight":0.4`), []byte(`"weight":1.4`), 1), "weight in [0, 1]"},
		"string_number":       {bytes.Replace(routed, []byte(`"tick":17`), []byte(`"tick":"17"`), 1), "invalid JSON"},
		"not_an_object":       {[]byte(`[]`), "invalid JSON"},
		"over_the_byte_limit": {bytes.Repeat([]byte(" "), 1<<16+1), "byte limit"},
	}
	for name, c := range cases {
		t.Run(name, func(t *testing.T) {
			if bytes.Equal(c.data, routed) || bytes.Equal(c.data, ground) {
				t.Fatal("the case did not change the document")
			}
			_, _, err := fusioncache.DecodePackage(c.data, packageLimits())
			if !errors.Is(err, fusioncache.ErrContract) || !strings.Contains(err.Error(), c.want) {
				t.Fatalf("want a refusal containing %q, got %v", c.want, err)
			}
		})
	}
}

func TestSupersedeMakesTheNextRevision(t *testing.T) {
	first := routedPackage(t)
	firstDigest := packageDigest(t, first)
	corrected := clone(t, first)
	corrected.Signal.RefSHA256 = hashOf("sft-line normalized to the student template")
	corrected.Artifacts = artifactIndex(corrected)
	second, err := fusioncache.Supersede(first, corrected, packageLimits())
	if err != nil {
		t.Fatal(err)
	}
	if second.Revision != 2 || second.Supersedes != firstDigest || packageDigest(t, second) == firstDigest {
		t.Fatalf("second revision does not point at the first: %d %s", second.Revision, second.Supersedes)
	}
	if corrected.Revision != 1 || corrected.Supersedes != "" {
		t.Fatal("Supersede changed its argument")
	}
	again := clone(t, second)
	again.Protection.ManifestSHA256 = hashOf("protection v2")
	again.Artifacts = artifactIndex(again)
	third, err := fusioncache.Supersede(second, again, packageLimits())
	if err != nil {
		t.Fatal(err)
	}
	if third.Revision != 3 || third.Supersedes != packageDigest(t, second) {
		t.Fatalf("third revision does not point at the second: %d %s", third.Revision, third.Supersedes)
	}

	refusals := map[string]struct {
		change func(*fusioncache.CapabilityPackage)
		want   string
	}{
		"nothing_changed":  {func(*fusioncache.CapabilityPackage) {}, "must change"},
		"other_example":    {func(p *fusioncache.CapabilityPackage) { p.ExampleID = "gsm8k-train-2051" }, "keeps the capability"},
		"other_capability": {func(p *fusioncache.CapabilityPackage) { p.CapabilityID = "reasoning.math" }, "keeps the capability"},
		"other_dataset": {func(p *fusioncache.CapabilityPackage) {
			p.DatasetSHA256 = hashOf("other dataset")
			p.Artifacts = artifactIndex(*p)
		}, "keeps the capability"},
		"invalid_correction": {func(p *fusioncache.CapabilityPackage) { p.Verifier.Correct = false }, "correct verdict"},
	}
	for name, c := range refusals {
		t.Run(name, func(t *testing.T) {
			bad := clone(t, first)
			c.change(&bad)
			if _, err := fusioncache.Supersede(first, bad, packageLimits()); !errors.Is(err, fusioncache.ErrContract) || !strings.Contains(err.Error(), c.want) {
				t.Fatalf("want a refusal containing %q, got %v", c.want, err)
			}
		})
	}
}
