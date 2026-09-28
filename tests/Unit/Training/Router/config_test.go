package router_test

import (
	"encoding/json"
	"errors"
	"math"
	"regexp"
	"testing"

	"github.com/tayi-ai/arandu-llama/training/router"
)

func TestInvalidConfigIsRefusedEverywhere(t *testing.T) {
	nan, inf := math.NaN(), math.Inf(1)
	cases := map[string]func(*router.Config){
		"a_v_nan":            func(c *router.Config) { c.Coefficients.Reliability = nan },
		"a_d_nan":            func(c *router.Config) { c.Coefficients.Competence = nan },
		"a_r_nan":            func(c *router.Config) { c.Coefficients.Reproducibility = nan },
		"a_i_nan":            func(c *router.Config) { c.Coefficients.Independence = nan },
		"a_l_nan":            func(c *router.Config) { c.Coefficients.Redundancy = nan },
		"a_k_nan":            func(c *router.Config) { c.Coefficients.Cost = nan },
		"a_d_inf":            func(c *router.Config) { c.Coefficients.Competence = inf },
		"a_l_negative":       func(c *router.Config) { c.Coefficients.Redundancy = -0.25 },
		"a_k_negative":       func(c *router.Config) { c.Coefficients.Cost = -1 },
		"epsilon_zero":       func(c *router.Config) { c.Epsilon = 0 },
		"epsilon_negative":   func(c *router.Config) { c.Epsilon = -0.01 },
		"epsilon_nan":        func(c *router.Config) { c.Epsilon = nan },
		"epsilon_inf":        func(c *router.Config) { c.Epsilon = inf },
		"tau_zero":           func(c *router.Config) { c.Tau = 0 },
		"tau_negative":       func(c *router.Config) { c.Tau = -1 },
		"tau_nan":            func(c *router.Config) { c.Tau = nan },
		"tau_inf":            func(c *router.Config) { c.Tau = inf },
		"k_zero":             func(c *router.Config) { c.K = 0 },
		"k_negative":         func(c *router.Config) { c.K = -2 },
		"shortlist_below_k":  func(c *router.Config) { c.Shortlist = 1 },
		"statistic_empty":    func(c *router.Config) { c.Competence = "" },
		"statistic_unknown":  func(c *router.Config) { c.Competence = "median" },
		"independence_empty": func(c *router.Config) { c.Independence = "" },
		"redundancy_empty":   func(c *router.Config) { c.Redundancy = "" },
		"redundancy_unknown": func(c *router.Config) { c.Redundancy = "unconditional" },
		"bias_nan":           func(c *router.Config) { c.Biases = map[string]float64{"qwen": nan} },
		"bias_inf":           func(c *router.Config) { c.Biases = map[string]float64{"qwen": -inf} },
		"bias_empty_key":     func(c *router.Config) { c.Biases = map[string]float64{"": 1} },
	}
	teachers := []router.Teacher{teacher("qwen", []string{"qwen"}, 270), teacher("gpt-oss", []string{"gpt-oss"}, 240)}
	for name, change := range cases {
		t.Run(name, func(t *testing.T) {
			config := fixture()
			change(&config)
			if err := config.Validate(); !errors.Is(err, router.ErrConfig) {
				t.Fatalf("validate admitted it: %v", err)
			}
			if sum, err := config.Digest(); !errors.Is(err, router.ErrConfig) || sum != "" {
				t.Fatalf("digest admitted it: %q %v", sum, err)
			}
			if d, err := router.Route(config, teachers, example()); !errors.Is(err, router.ErrConfig) || d.Schema != "" {
				t.Fatalf("route admitted it: %+v %v", d, err)
			}
			if d, err := router.Shortlist(config, teachers, capability); !errors.Is(err, router.ErrConfig) || d.Schema != "" {
				t.Fatalf("shortlist admitted it: %+v %v", d, err)
			}
		})
	}
	if err := fixture().Validate(); err != nil {
		t.Fatalf("the fixture itself is refused: %v", err)
	}
}

func TestConfigDigestIsCanonical(t *testing.T) {
	a := fixture()
	a.Biases = map[string]float64{}
	a.Biases["qwen"] = 0.5
	a.Biases["gpt-oss"] = -0.25
	a.Biases["gemma"] = 0
	b := fixture()
	b.Biases = map[string]float64{}
	b.Biases["gemma"] = 0
	b.Biases["gpt-oss"] = -0.25
	b.Biases["qwen"] = 0.5
	da, err := a.Digest()
	if err != nil {
		t.Fatal(err)
	}
	db, err := b.Digest()
	if err != nil {
		t.Fatal(err)
	}
	if da != db || !regexp.MustCompile(`^[0-9a-f]{64}$`).MatchString(da) {
		t.Fatalf("bias insertion order changed the digest: %s %s", da, db)
	}
	empty, none := fixture(), fixture()
	empty.Biases = map[string]float64{}
	de, _ := empty.Digest()
	dn, _ := none.Digest()
	if de != dn {
		t.Fatal("an empty bias map and an absent one digest differently")
	}
	changes := map[string]func(*router.Config){
		"a_v":          func(c *router.Config) { c.Coefficients.Reliability = math.Nextafter(c.Coefficients.Reliability, 2) },
		"a_d":          func(c *router.Config) { c.Coefficients.Competence = 3 },
		"a_r":          func(c *router.Config) { c.Coefficients.Reproducibility = 0.5 },
		"a_i":          func(c *router.Config) { c.Coefficients.Independence = 0.25 },
		"a_l":          func(c *router.Config) { c.Coefficients.Redundancy = 0.5 },
		"a_k":          func(c *router.Config) { c.Coefficients.Cost = 0.1 },
		"bias":         func(c *router.Config) { c.Biases = map[string]float64{"qwen": 0} },
		"epsilon":      func(c *router.Config) { c.Epsilon = 0.001 },
		"tau":          func(c *router.Config) { c.Tau = 0.5 },
		"k":            func(c *router.Config) { c.K = 3 },
		"shortlist":    func(c *router.Config) { c.Shortlist = 4 },
		"statistic":    func(c *router.Config) { c.Competence = router.StatisticLowerBound },
		"a_l_ulp_zero": func(c *router.Config) { c.Coefficients.Redundancy = math.Nextafter(c.Coefficients.Redundancy, 0) },
	}
	seen := map[string]string{dn: "fixture"}
	for name, change := range changes {
		c := fixture()
		change(&c)
		sum, err := c.Digest()
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		if other, dup := seen[sum]; dup {
			t.Fatalf("%s digests like %s", name, other)
		}
		seen[sum] = name
	}
}

func TestLoadConfigPinsTheRegisteredDigest(t *testing.T) {
	config := fixture()
	config.Biases = map[string]float64{"qwen": 0.5}
	want, err := config.Digest()
	if err != nil {
		t.Fatal(err)
	}
	body, err := json.Marshal(config)
	if err != nil {
		t.Fatal(err)
	}
	got, err := router.LoadConfig(body, want)
	if err != nil {
		t.Fatal(err)
	}
	if sum, _ := got.Digest(); sum != want || got.Biases["qwen"] != 0.5 || got.K != 2 {
		t.Fatalf("loaded a different configuration: %+v", got)
	}
	refusals := map[string]struct {
		body   string
		digest string
	}{
		"other_digest":  {string(body), pin("another")},
		"unknown_field": {`{"coefficients":{"a_d":4},"epsilon":0.01,"tau":1,"k":2,"shortlist":3,"competence":"accuracy","independence":"inverse_relatives","redundancy":"shared_roots","a_x":1}`, want},
		"trailing":      {string(body) + ` {}`, want},
		"invalid":       {`{"coefficients":{"a_d":4},"epsilon":0,"tau":1,"k":2,"shortlist":3,"competence":"accuracy","independence":"inverse_relatives","redundancy":"shared_roots"}`, want},
		"not_json":      {`coefficients`, want},
	}
	for name, refusal := range refusals {
		if _, err := router.LoadConfig([]byte(refusal.body), refusal.digest); !errors.Is(err, router.ErrConfig) {
			t.Fatalf("%s loaded: %v", name, err)
		}
	}
}
