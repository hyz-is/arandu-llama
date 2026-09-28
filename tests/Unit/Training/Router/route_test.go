package router_test

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"math/rand/v2"
	"reflect"
	"slices"
	"strings"
	"testing"

	"github.com/tayi-ai/arandu-llama/training/router"
)

func mixedTable() []router.Teacher {
	merge := teacher("merge-qwen-llama", []string{"qwen", "llama"}, 250)
	merge.Cost = 2
	blocked := teacher("blocked", []string{"mistral"}, 299)
	blocked.Gates.Qualified = false
	return []router.Teacher{
		teacher("qwen3-8-27b", []string{"qwen"}, 270),
		teacher("bonsai-27b", []string{"qwen"}, 250),
		teacher("gpt-oss-20b", []string{"gpt-oss"}, 240),
		teacher("gemma-4-12b", []string{"gemma"}, 240),
		teacher("llama-4", []string{"llama"}, 200),
		merge,
		blocked,
	}
}

func mixedConfig() router.Config {
	config := fixture()
	config.K = 3
	config.Shortlist = 4
	config.Coefficients.Independence = 0.25
	config.Coefficients.Cost = 0.1
	config.Biases = map[string]float64{"llama-4": 0.05, "absent-teacher": 3}
	return config
}

func TestRouteIsDeterministicUnderAnyInputOrder(t *testing.T) {
	config, teachers := mixedConfig(), mixedTable()
	reference := route(t, config, teachers)
	want := encode(t, reference)
	if again := encode(t, route(t, config, teachers)); again != want {
		t.Fatal("the same input routed twice produced two decisions")
	}
	shortlist, err := router.Shortlist(config, teachers, capability)
	if err != nil {
		t.Fatal(err)
	}
	wantShortlist := encode(t, shortlist)
	source := rand.New(rand.NewPCG(128, 7))
	for i := range 500 {
		input := shuffled(teachers, source)
		if got := encode(t, route(t, config, input)); got != want {
			t.Fatalf("permutation %d changed the decision:\n%s\n%s", i, got, want)
		}
		got, err := router.Shortlist(config, input, capability)
		if err != nil || encode(t, got) != wantShortlist {
			t.Fatalf("permutation %d changed the shortlist: %v", i, err)
		}
	}
	// gemma-4-12b and gpt-oss-20b tie exactly on every factor: the lower id
	// comes first wherever both are standing.
	if score(reference, "gemma-4-12b").Base != score(reference, "gpt-oss-20b").Base {
		t.Fatal("the fixture no longer ties")
	}
	for _, round := range reference.Rounds {
		gemma := slices.IndexFunc(round.Standings, func(s router.Standing) bool { return s.Teacher == "gemma-4-12b" })
		gpt := slices.IndexFunc(round.Standings, func(s router.Standing) bool { return s.Teacher == "gpt-oss-20b" })
		if gemma >= 0 && gpt >= 0 && gemma > gpt {
			t.Fatalf("round %d broke the tie by something other than id: %+v", round.Rank, round.Standings)
		}
	}
}

func TestAnExactTieGoesToTheLowerID(t *testing.T) {
	config := fixture()
	config.K = 1
	for _, order := range [][]router.Teacher{
		{teacher("teacher-b", []string{"b"}, 240), teacher("teacher-a", []string{"a"}, 240)},
		{teacher("teacher-a", []string{"a"}, 240), teacher("teacher-b", []string{"b"}, 240)},
	} {
		if got := selected(route(t, config, order)); !slices.Equal(got, []string{"teacher-a"}) {
			t.Fatalf("tie resolved to %v", got)
		}
	}
}

func TestThreeIndependentFamiliesLeaveRedundancyInert(t *testing.T) {
	// The qualification v1 shape: three teachers, three disjoint lineage roots.
	teachers := []router.Teacher{
		teacher("gpt-oss-20b-mxfp4", []string{"gpt-oss"}, 240),
		teacher("gemma-4-12b-it-qat-q4-0", []string{"gemma"}, 210),
		teacher("qwen3-8-27b-q4-k-m", []string{"qwen"}, 255),
	}
	byCompetence := []string{"qwen3-8-27b-q4-k-m", "gpt-oss-20b-mxfp4", "gemma-4-12b-it-qat-q4-0"}
	config := fixture()
	config.K = 3
	config.Coefficients.Redundancy = 0
	reference := route(t, config, teachers)
	if got := selected(reference); !slices.Equal(got, byCompetence) {
		t.Fatalf("order %v is not the competence order %v", got, byCompetence)
	}
	for _, s := range reference.Scores {
		if s.Independence != 1 {
			t.Fatalf("%s has independence %v among disjoint lineages", s.Teacher, s.Independence)
		}
	}
	for _, round := range reference.Rounds {
		for _, s := range round.Standings {
			if s.Redundancy != 0 {
				t.Fatalf("round %d charged %s redundancy %v", round.Rank, s.Teacher, s.Redundancy)
			}
		}
	}
	for _, redundancy := range []float64{0.25, 5, 1000} {
		config.Coefficients.Redundancy = redundancy
		d := route(t, config, teachers)
		if !reflect.DeepEqual(d.Selected, reference.Selected) || !reflect.DeepEqual(d.Rounds, reference.Rounds) || !reflect.DeepEqual(d.Scores, reference.Scores) {
			t.Fatalf("a_l=%v moved a decision among independent families", redundancy)
		}
	}
	for _, independence := range []float64{0.25, 5} {
		config.Coefficients.Independence = independence
		if got := selected(route(t, config, teachers)); !slices.Equal(got, byCompetence) {
			t.Fatalf("a_i=%v reordered independent families: %v", independence, got)
		}
	}
	config.K = 2
	if got := selected(route(t, config, teachers)); !slices.Equal(got, byCompetence[:2]) {
		t.Fatalf("top-2 is %v", got)
	}
}

// TestARelativeYieldsTheSecondSlotInsideTheBand holds the case the
// qualification v1 cannot exercise: two teachers of one lineage beside an
// independent one. The relative takes the second slot only when its
// competence clears the band the coefficients define:
//
//	a_d ln((eps+D_rel)/(eps+D_ind)) + a_i ln((eps+I_rel)/(eps+I_ind)) > a_l
//
// with I_rel = 1/2 and I_ind = 1, so the tipping competence is
//
//	D* = (eps+D_ind) exp((a_l + a_i ln((eps+1)/(eps+1/2))) / a_d) - eps.
func TestARelativeYieldsTheSecondSlotInsideTheBand(t *testing.T) {
	const independent = 240 // gpt-oss at 0.80
	for _, independence := range []float64{0, 0.25} {
		t.Run(fmt.Sprintf("a_i=%v", independence), func(t *testing.T) {
			config := fixture()
			config.Coefficients.Independence = independence
			a, eps := config.Coefficients, config.Epsilon
			tipping := (eps+0.8)*math.Exp((a.Redundancy+a.Independence*math.Log((eps+1)/(eps+0.5)))/a.Competence) - eps
			want := int(math.Floor(tipping*items)) + 1
			first := 0
			for correct := independent + 1; correct < 270; correct++ {
				d := route(t, config, []router.Teacher{
					teacher("qwen3-8-27b", []string{"qwen"}, 270),
					teacher("bonsai-27b", []string{"qwen"}, correct),
					teacher("gpt-oss-20b", []string{"gpt-oss"}, independent),
				})
				got := selected(d)
				if got[0] != "qwen3-8-27b" {
					t.Fatalf("bonsai at %d/%d displaced the most competent teacher: %v", correct, items, got)
				}
				if score(d, "bonsai-27b").Independence != 0.5 || score(d, "qwen3-8-27b").Independence != 0.5 || score(d, "gpt-oss-20b").Independence != 1 {
					t.Fatalf("independence %+v", d.Scores)
				}
				for _, s := range d.Rounds[1].Standings {
					if want := map[string]float64{"bonsai-27b": 1, "gpt-oss-20b": 0}[s.Teacher]; s.Redundancy != want {
						t.Fatalf("round 2 charged %s redundancy %v", s.Teacher, s.Redundancy)
					}
				}
				switch {
				case got[1] == "bonsai-27b" && first == 0:
					first = correct
				case got[1] != "bonsai-27b" && first != 0:
					t.Fatalf("the relative won at %d and lost again at %d", first, correct)
				}
			}
			if first != want {
				t.Fatalf("the relative took the second slot at %d/%d, the formula says %d/%d (D* = %.6f)", first, items, want, items, tipping)
			}
			t.Logf("a_l=%v a_d=%v a_i=%v eps=%v: gpt-oss at %.4f keeps the second slot until bonsai reaches %d/%d = %.4f (D* = %.6f, band %.4f)",
				a.Redundancy, a.Competence, a.Independence, eps, 0.8, want, items, float64(want)/items, tipping, tipping-0.8)
		})
	}
	// Bonsai slightly more competent than gpt-oss: the independent teacher
	// keeps the slot with the redundancy on, and loses it with it off.
	teachers := []router.Teacher{
		teacher("qwen3-8-27b", []string{"qwen"}, 270),
		teacher("bonsai-27b", []string{"qwen"}, independent+1),
		teacher("gpt-oss-20b", []string{"gpt-oss"}, independent),
	}
	if got := selected(route(t, fixture(), teachers)); !slices.Equal(got, []string{"qwen3-8-27b", "gpt-oss-20b"}) {
		t.Fatalf("with the redundancy on: %v", got)
	}
	off := fixture()
	off.Coefficients.Redundancy = 0
	if got := selected(route(t, off, teachers)); !slices.Equal(got, []string{"qwen3-8-27b", "bonsai-27b"}) {
		t.Fatalf("with the redundancy off: %v", got)
	}
}

func TestRedundancyGrowsWithTheRootsSharedWithTheChosen(t *testing.T) {
	config := fixture()
	config.K, config.Shortlist = 5, 5
	config.Coefficients.Redundancy = 0
	config.Coefficients.Independence = 0
	roots := map[string][]string{
		"qwen-a": {"qwen"}, "llama": {"llama"}, "merge": {"llama", "qwen"}, "qwen-b": {"qwen"}, "gpt-oss": {"gpt-oss"},
	}
	teachers := []router.Teacher{
		teacher("qwen-a", roots["qwen-a"], 270),
		teacher("llama", roots["llama"], 255),
		teacher("merge", roots["merge"], 240),
		teacher("qwen-b", roots["qwen-b"], 225),
		teacher("gpt-oss", roots["gpt-oss"], 210),
	}
	d := route(t, config, teachers)
	if got := selected(d); !slices.Equal(got, []string{"qwen-a", "llama", "merge", "qwen-b", "gpt-oss"}) {
		t.Fatalf("order %v", got)
	}
	shared := func(a, b []string) int {
		n := 0
		for _, x := range a {
			if slices.Contains(b, x) {
				n++
			}
		}
		return n
	}
	for r, round := range d.Rounds {
		for _, s := range round.Standings {
			want := 0
			for _, picked := range d.Selected[:r] {
				want += shared(roots[s.Teacher], roots[picked.Teacher])
			}
			if s.Redundancy != float64(want) {
				t.Fatalf("round %d: %s has redundancy %v, the chosen prefix gives %d", round.Rank, s.Teacher, s.Redundancy, want)
			}
		}
	}
	at := func(rank int, id string) float64 {
		for _, s := range d.Rounds[rank-1].Standings {
			if s.Teacher == id {
				return s.Redundancy
			}
		}
		return math.NaN()
	}
	if at(3, "merge") != 2 || at(3, "qwen-b") != 1 || at(3, "gpt-oss") != 0 || at(4, "qwen-b") != 2 {
		t.Fatalf("redundancy does not count shared roots: %+v", d.Rounds)
	}
	for id, want := range map[string]float64{"qwen-a": 1.0 / 3, "qwen-b": 1.0 / 3, "llama": 0.5, "merge": 0.25, "gpt-oss": 1} {
		if got := score(d, id).Independence; got != want {
			t.Fatalf("%s independence %v, want %v", id, got, want)
		}
	}
}

func TestGatesExcludeAndRecordEveryReason(t *testing.T) {
	strong := func(id string) router.Teacher { return teacher(id, []string{id}, 300) }
	unlicensed := strong("a-unlicensed")
	unlicensed.Gates.LicenseVerified = false
	unpinned := strong("b-unpinned")
	unpinned.Gates.ArtifactSHA256 = ""
	short := strong("c-short-digest")
	short.Gates.ArtifactSHA256 = pin("c")[:63]
	upper := strong("d-upper-digest")
	upper.Gates.ArtifactSHA256 = strings.ToUpper(pin("d"))
	unqualified := strong("e-unqualified")
	unqualified.Gates.Qualified = false
	unmeasured := strong("f-unmeasured")
	unmeasured.Competence = []router.Competence{cell(capability, 300, items), cell(subcapability+".level-1", 300, items)}
	everything := strong("g-everything")
	everything.Gates = router.Gates{}
	everything.Competence = nil
	admitted := teacher("z-admitted", []string{"z"}, 150)
	teachers := []router.Teacher{admitted, unlicensed, unpinned, short, upper, unqualified, unmeasured, everything}

	d := route(t, fixture(), teachers)
	want := []router.Exclusion{
		{Teacher: "a-unlicensed", Reasons: []router.Reason{router.ReasonLicenseNotVerified}},
		{Teacher: "b-unpinned", Reasons: []router.Reason{router.ReasonArtifactNotPinned}},
		{Teacher: "c-short-digest", Reasons: []router.Reason{router.ReasonArtifactNotPinned}},
		{Teacher: "d-upper-digest", Reasons: []router.Reason{router.ReasonArtifactNotPinned}},
		{Teacher: "e-unqualified", Reasons: []router.Reason{router.ReasonNotQualified}},
		{Teacher: "f-unmeasured", Reasons: []router.Reason{router.ReasonCompetenceNotMeasured}},
		{Teacher: "g-everything", Reasons: []router.Reason{router.ReasonLicenseNotVerified, router.ReasonArtifactNotPinned, router.ReasonNotQualified, router.ReasonCompetenceNotMeasured}},
	}
	if !reflect.DeepEqual(d.Excluded, want) {
		t.Fatalf("exclusions\n got %+v\nwant %+v", d.Excluded, want)
	}
	if got := selected(d); !slices.Equal(got, []string{"z-admitted"}) || len(d.Scores) != 1 || len(d.Rounds) != 1 || len(d.Rounds[0].Standings) != 1 {
		t.Fatalf("an excluded teacher was scored: %+v", d)
	}

	blocked := fixture()
	blocked.K = 1
	none, err := router.Route(blocked, teachers[1:], example())
	if err != nil {
		t.Fatalf("all teachers excluded is a decision, not an error: %v", err)
	}
	if body := encode(t, none); len(none.Selected) != 0 || len(none.Excluded) != 7 || !strings.Contains(body, `"selected":[]`) {
		t.Fatalf("no admitted teacher: %s", body)
	}
}

func TestAnExcludedRelativeLowersNobodysIndependence(t *testing.T) {
	config := fixture()
	config.Coefficients.Independence = 0.25
	bonsai := teacher("bonsai-27b", []string{"qwen"}, 250)
	qwen, gpt := teacher("qwen3-8-27b", []string{"qwen"}, 270), teacher("gpt-oss-20b", []string{"gpt-oss"}, 240)
	if got := score(route(t, config, []router.Teacher{qwen, bonsai, gpt}), "qwen3-8-27b").Independence; got != 0.5 {
		t.Fatalf("admitted relative: qwen independence %v", got)
	}
	bonsai.Gates.LicenseVerified = false
	with := route(t, config, []router.Teacher{qwen, bonsai, gpt})
	without := route(t, config, []router.Teacher{qwen, gpt})
	if !reflect.DeepEqual(with.Selected, without.Selected) || !reflect.DeepEqual(with.Scores, without.Scores) || !reflect.DeepEqual(with.Rounds, without.Rounds) {
		t.Fatalf("an excluded teacher changed the decision:\n%+v\n%+v", with, without)
	}
	if score(with, "qwen3-8-27b").Independence != 1 {
		t.Fatal("an excluded relative still counted against independence")
	}
}

func TestNoSingleFactorSilencesATeacher(t *testing.T) {
	config := fixture()
	config.K = 3
	config.Coefficients.Independence = 0.25
	config.Coefficients.Cost = 0.1
	a, eps := config.Coefficients, config.Epsilon
	healthy := teacher("weak", []string{"weak"}, 240)
	floors := map[string]struct {
		degrade func(*router.Teacher)
		penalty float64
	}{
		"reliability_zero":     {func(x *router.Teacher) { x.Reliability = 0 }, a.Reliability * (math.Log(eps+1) - math.Log(eps))},
		"reproducibility_zero": {func(x *router.Teacher) { x.Reproducibility = 0 }, a.Reproducibility * (math.Log(eps+1) - math.Log(eps))},
		"competence_zero": {func(x *router.Teacher) {
			x.Competence = []router.Competence{cell(subcapability, 0, items), cell(capability, 0, items)}
		}, a.Competence * (math.Log(eps+0.8) - math.Log(eps))},
		"cost_high": {func(x *router.Teacher) { x.Cost = 50 }, a.Cost * 50},
	}
	others := []router.Teacher{teacher("qwen", []string{"qwen"}, 270), teacher("gpt-oss", []string{"gpt-oss"}, 255)}
	reference := route(t, config, append(slices.Clone(others), healthy))
	for name, floor := range floors {
		t.Run(name, func(t *testing.T) {
			weak := healthy
			floor.degrade(&weak)
			d := route(t, config, append(slices.Clone(others), weak))
			s := score(d, "weak")
			if math.IsInf(s.Base, 0) || math.IsNaN(s.Base) {
				t.Fatalf("base score %v", s.Base)
			}
			if lost := score(reference, "weak").Base - s.Base; math.Abs(lost-floor.penalty) > 1e-12 {
				t.Fatalf("the factor cost %v, the formula bounds it at %v", lost, floor.penalty)
			}
			index := slices.IndexFunc(d.Selected, func(x router.Selection) bool { return x.Teacher == "weak" })
			if index < 0 || !(d.Selected[index].Weight > 0) || math.IsInf(d.Selected[index].Weight, 0) {
				t.Fatalf("the degraded teacher was silenced: %+v", d.Selected)
			}
			sum := 0.0
			for _, x := range d.Selected {
				sum += x.Weight
			}
			if math.Abs(sum-1) > 1e-12 {
				t.Fatalf("weights sum to %v", sum)
			}
		})
	}
	var relatives []router.Teacher
	for i := range 10 {
		relatives = append(relatives, teacher(fmt.Sprintf("qwen-%d", i), []string{"qwen"}, 200+i))
	}
	config.K, config.Shortlist = 10, 10
	d := route(t, config, relatives)
	for _, s := range d.Selected {
		if score(d, s.Teacher).Independence != 0.1 || !(s.Weight > 0) {
			t.Fatalf("ten relatives: %+v", d.Selected)
		}
	}
}

func TestAWeightThatUnderflowsIsRefusedNotRecorded(t *testing.T) {
	config := fixture()
	config.Tau = 1e-9
	teachers := []router.Teacher{teacher("qwen", []string{"qwen"}, 270), teacher("gpt-oss", []string{"gpt-oss"}, 240)}
	d, err := router.Route(config, teachers, example())
	if !errors.Is(err, router.ErrScore) || d.Schema != "" {
		t.Fatalf("a zero weight was recorded: %+v %v", d.Selected, err)
	}
}

func TestTheCompetenceStatisticIsTheDeclaredOne(t *testing.T) {
	few := teacher("few-items", []string{"few"}, 0)
	few.Competence = []router.Competence{cell(subcapability, 24, 30)}
	many := teacher("many-items", []string{"many"}, 0)
	many.Competence = []router.Competence{cell(subcapability, 234, 300)}
	config := fixture()
	config.K = 1
	if got := selected(route(t, config, []router.Teacher{few, many})); !slices.Equal(got, []string{"few-items"}) {
		t.Fatalf("point accuracy chose %v", got)
	}
	config.Competence = router.StatisticLowerBound
	d := route(t, config, []router.Teacher{few, many})
	if got := selected(d); !slices.Equal(got, []string{"many-items"}) {
		t.Fatalf("the lower bound chose %v", got)
	}
	if score(d, "few-items").Competence != few.Competence[0].CILower {
		t.Fatal("the recorded competence is not the lower bound")
	}
}

func TestShortlistScoresTheCapabilityCell(t *testing.T) {
	qwen := teacher("qwen3-8-27b", []string{"qwen"}, 270)
	bonsai := teacher("bonsai-27b", []string{"qwen"}, 250)
	gpt := teacher("gpt-oss-20b", []string{"gpt-oss"}, 240)
	subOnly := teacher("sub-only", []string{"sub"}, 300)
	subOnly.Competence = []router.Competence{cell(subcapability, 300, items)}
	gpt.Competence[1] = cell(capability, 150, items)
	config := fixture()
	config.Shortlist = 2
	d, err := router.Shortlist(config, []router.Teacher{qwen, bonsai, gpt, subOnly}, capability)
	if err != nil {
		t.Fatal(err)
	}
	if d.Level != router.LevelCapability || d.ExampleID != "" || d.Subcapability != capability || d.K != 2 {
		t.Fatalf("shortlist identity %+v", d)
	}
	if score(d, "gpt-oss-20b").Competence != 0.5 {
		t.Fatal("the shortlist did not read the capability cell")
	}
	if !reflect.DeepEqual(d.Excluded, []router.Exclusion{{Teacher: "sub-only", Reasons: []router.Reason{router.ReasonCompetenceNotMeasured}}}) {
		t.Fatalf("a subcapability cell stood in for the capability: %+v", d.Excluded)
	}
	if got := selected(d); !slices.Equal(got, []string{"qwen3-8-27b", "bonsai-27b"}) {
		t.Fatalf("shortlist %v", got)
	}
	r := route(t, config, []router.Teacher{qwen, bonsai, gpt, subOnly})
	if r.Level != router.LevelExample || r.InputsSHA256 == d.InputsSHA256 || r.TeachersSHA256 != d.TeachersSHA256 {
		t.Fatal("the two levels share an inputs digest, or read different tables")
	}
}

func TestDigestsAreStableUnderShuffledInputAndBindEveryValue(t *testing.T) {
	config, teachers := mixedConfig(), mixedTable()
	reference := route(t, config, teachers)
	source := rand.New(rand.NewPCG(7, 128))
	for range 200 {
		d := route(t, config, shuffled(teachers, source))
		if d.TeachersSHA256 != reference.TeachersSHA256 || d.InputsSHA256 != reference.InputsSHA256 || d.ConfigSHA256 != reference.ConfigSHA256 {
			t.Fatal("the input order changed a digest")
		}
	}
	if sum, _ := config.Digest(); sum != reference.ConfigSHA256 {
		t.Fatal("the decision does not carry the configuration digest")
	}
	nudged := mixedTable()
	nudged[2].Reliability = math.Nextafter(nudged[2].Reliability, 0)
	if d := route(t, config, nudged); d.TeachersSHA256 == reference.TeachersSHA256 || d.InputsSHA256 == reference.InputsSHA256 {
		t.Fatal("one ulp of reliability left the digests unchanged")
	}
	rooted := mixedTable()
	rooted[5].Roots = []string{"qwen"}
	if d := route(t, config, rooted); d.TeachersSHA256 == reference.TeachersSHA256 {
		t.Fatal("a lineage change left the teacher digest unchanged")
	}
	other, err := router.Route(config, teachers, router.Example{ID: "gsm8k-train-0043", Subcapability: subcapability})
	if err != nil || other.InputsSHA256 == reference.InputsSHA256 || other.TeachersSHA256 != reference.TeachersSHA256 {
		t.Fatalf("the example is not bound by the inputs digest alone: %v", err)
	}
}

func TestADecisionSurvivesItsOwnEncoding(t *testing.T) {
	d := route(t, mixedConfig(), mixedTable())
	body, err := json.Marshal(d)
	if err != nil {
		t.Fatal(err)
	}
	var back router.Decision
	if err := json.Unmarshal(body, &back); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(back, d) || d.Schema != router.DecisionSchema {
		t.Fatalf("the decision does not survive its encoding:\n%s", body)
	}
}

func TestInvalidInputIsRefused(t *testing.T) {
	nan := math.NaN()
	cases := map[string]func([]router.Teacher) ([]router.Teacher, router.Example){
		"no_teachers": func([]router.Teacher) ([]router.Teacher, router.Example) { return nil, example() },
		"duplicate_id": func(ts []router.Teacher) ([]router.Teacher, router.Example) {
			return append(ts, teacher(ts[0].ID, []string{"other"}, 10)), example()
		},
		"empty_id":  func(ts []router.Teacher) ([]router.Teacher, router.Example) { ts[0].ID = ""; return ts, example() },
		"padded_id": func(ts []router.Teacher) ([]router.Teacher, router.Example) { ts[0].ID = " qwen"; return ts, example() },
		"no_roots":  func(ts []router.Teacher) ([]router.Teacher, router.Example) { ts[0].Roots = nil; return ts, example() },
		"repeated_root": func(ts []router.Teacher) ([]router.Teacher, router.Example) {
			ts[0].Roots = []string{"q", "q"}
			return ts, example()
		},
		"empty_root": func(ts []router.Teacher) ([]router.Teacher, router.Example) {
			ts[0].Roots = []string{""}
			return ts, example()
		},
		"reliability_nan": func(ts []router.Teacher) ([]router.Teacher, router.Example) {
			ts[0].Reliability = nan
			return ts, example()
		},
		"reliability_high": func(ts []router.Teacher) ([]router.Teacher, router.Example) {
			ts[0].Reliability = 1.5
			return ts, example()
		},
		"reproducibility_negative": func(ts []router.Teacher) ([]router.Teacher, router.Example) {
			ts[0].Reproducibility = -0.1
			return ts, example()
		},
		"cost_negative": func(ts []router.Teacher) ([]router.Teacher, router.Example) { ts[0].Cost = -1; return ts, example() },
		"cost_inf": func(ts []router.Teacher) ([]router.Teacher, router.Example) {
			ts[0].Cost = math.Inf(1)
			return ts, example()
		},
		"no_items": func(ts []router.Teacher) ([]router.Teacher, router.Example) {
			ts[0].Competence[0].Items = 0
			return ts, example()
		},
		"correct_over_items": func(ts []router.Teacher) ([]router.Teacher, router.Example) {
			ts[0].Competence[0].Correct = items + 1
			return ts, example()
		},
		"accuracy_mismatch": func(ts []router.Teacher) ([]router.Teacher, router.Example) {
			ts[0].Competence[0].Accuracy = math.Nextafter(ts[0].Competence[0].Accuracy, 1)
			return ts, example()
		},
		"interval_inverted": func(ts []router.Teacher) ([]router.Teacher, router.Example) {
			c := &ts[0].Competence[0]
			c.CILower, c.CIUpper = c.CIUpper, c.CILower
			return ts, example()
		},
		"interval_nan": func(ts []router.Teacher) ([]router.Teacher, router.Example) {
			ts[0].Competence[0].CILower = nan
			return ts, example()
		},
		"repeated_cell": func(ts []router.Teacher) ([]router.Teacher, router.Example) {
			ts[0].Competence = append(ts[0].Competence, ts[0].Competence[0])
			return ts, example()
		},
		"example_without_id": func(ts []router.Teacher) ([]router.Teacher, router.Example) {
			return ts, router.Example{Subcapability: subcapability}
		},
		"example_without_subcapability": func(ts []router.Teacher) ([]router.Teacher, router.Example) {
			return ts, router.Example{ID: "x"}
		},
	}
	for name, corrupt := range cases {
		t.Run(name, func(t *testing.T) {
			teachers, ex := corrupt([]router.Teacher{teacher("qwen", []string{"qwen"}, 270), teacher("gpt-oss", []string{"gpt-oss"}, 240)})
			if d, err := router.Route(fixture(), teachers, ex); !errors.Is(err, router.ErrInput) || d.Schema != "" {
				t.Fatalf("admitted: %+v %v", d, err)
			}
		})
	}
	if _, err := router.Shortlist(fixture(), []router.Teacher{teacher("qwen", []string{"qwen"}, 270)}, ""); !errors.Is(err, router.ErrInput) {
		t.Fatalf("empty capability admitted: %v", err)
	}
}
