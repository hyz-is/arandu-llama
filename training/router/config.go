// Package router decides which teachers supervise an example before any of
// them generates.
//
// Every teacher is scored from what was measured before the example existed:
// its gates, its reliability against the verifier, its competence on the
// subcapability, its reproducibility, its lineage and its cost. For teacher t
// on competence cell c,
//
//	S_t = b_t + a_v log(eps+V_t) + a_d log(eps+D_t,c) + a_r log(eps+R_t)
//	          + a_i log(eps+I_t) - a_l L_t - a_k K_t
//
// and the chosen teachers are weighted by softmax(S_t/tau). A failed gate
// excludes a teacher and names the reason; it is never a factor. The package
// holds no default coefficients: they are a hypothesis the caller registers
// in a Config and pins by its digest.
//
// No function here accepts a teacher's output. A decision is a function of the
// configuration, the teacher table and the example's identity, so it cannot
// depend on an answer it never received.
package router

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"maps"
	"math"
	"slices"
	"strings"
)

// ErrConfig identifies a configuration the router refuses to score with.
var ErrConfig = errors.New("router: invalid configuration")

// ErrInput identifies a teacher table or example the router refuses to read.
var ErrInput = errors.New("router: invalid input")

// ErrScore identifies a score or weight that is not a usable finite number.
// A weight that underflows to zero is refused rather than recorded, because
// a zero weight would silence a chosen teacher without saying so.
var ErrScore = errors.New("router: unusable score")

// Statistic names the reading of a competence cell that enters D_t,c.
type Statistic string

const (
	// StatisticAccuracy reads the point accuracy, correct over items.
	StatisticAccuracy Statistic = "accuracy"
	// StatisticLowerBound reads the lower bound of the cell's confidence
	// interval, which discounts a competence measured on few items.
	StatisticLowerBound Statistic = "ci_lower"
)

// IndependenceRule names how I_t is computed from lineage roots.
type IndependenceRule string

// IndependenceInverseRelatives sets I_t to one over the number of admitted
// candidates, t included, that share at least one lineage root with t. Three
// teachers with disjoint roots each have I = 1; two relatives beside an
// independent teacher have I = 1/2 and the independent one I = 1. The pool is
// the teachers admitted for the decision, so an excluded relative does not
// lower anybody's independence.
const IndependenceInverseRelatives IndependenceRule = "inverse_relatives"

// RedundancyRule names how L_t is computed from the teachers already chosen.
type RedundancyRule string

// RedundancySharedRoots sets L_t to the sum, over every teacher already
// chosen, of the number of lineage roots t shares with it. The first choice
// always has L = 0; a relative of one chosen teacher has L = 1, a relative of
// two has L = 2, and a merge of two chosen lineages has L = 2.
const RedundancySharedRoots RedundancyRule = "shared_roots"

// Coefficients are the non-negative multipliers of the score's factors. The
// JSON names are the symbols of the formula in the package documentation.
// A negative coefficient would invert what its factor means, so it is refused.
type Coefficients struct {
	Reliability     float64 `json:"a_v"`
	Competence      float64 `json:"a_d"`
	Reproducibility float64 `json:"a_r"`
	Independence    float64 `json:"a_i"`
	Redundancy      float64 `json:"a_l"`
	Cost            float64 `json:"a_k"`
}

// Config is the pre-registered scoring hypothesis. Its Digest is what a
// decision records and what a registration pins.
//
// Biases holds b_t by teacher id; a teacher it does not name has b_t = 0, and
// every score records the bias it used. K is the number of teachers chosen
// per example and Shortlist the number kept per capability, so Shortlist is
// at least K.
type Config struct {
	Coefficients Coefficients       `json:"coefficients"`
	Biases       map[string]float64 `json:"biases,omitempty"`
	Epsilon      float64            `json:"epsilon"`
	Tau          float64            `json:"tau"`
	K            int                `json:"k"`
	Shortlist    int                `json:"shortlist"`
	Competence   Statistic          `json:"competence"`
	Independence IndependenceRule   `json:"independence"`
	Redundancy   RedundancyRule     `json:"redundancy"`
}

const configSchema = "tayi.tcap.router.config.v1"

// Validate refuses a non-finite or negative coefficient, a non-finite bias or
// an empty bias key, eps or tau that is not a positive finite number, K below
// one, Shortlist below K, and any statistic or rule this version does not
// implement.
func (c Config) Validate() error {
	coefficients := []struct {
		name  string
		value float64
	}{
		{"a_v", c.Coefficients.Reliability},
		{"a_d", c.Coefficients.Competence},
		{"a_r", c.Coefficients.Reproducibility},
		{"a_i", c.Coefficients.Independence},
		{"a_l", c.Coefficients.Redundancy},
		{"a_k", c.Coefficients.Cost},
	}
	for _, coefficient := range coefficients {
		if !finite(coefficient.value) || coefficient.value < 0 {
			return fmt.Errorf("%w: coefficient %s is %v, not a finite non-negative number", ErrConfig, coefficient.name, coefficient.value)
		}
	}
	for _, teacher := range slices.Sorted(maps.Keys(c.Biases)) {
		bias := c.Biases[teacher]
		if !validID(teacher) {
			return fmt.Errorf("%w: bias key %q is not a teacher id", ErrConfig, teacher)
		}
		if !finite(bias) {
			return fmt.Errorf("%w: bias of %s is %v", ErrConfig, teacher, bias)
		}
	}
	if !finite(c.Epsilon) || c.Epsilon <= 0 {
		return fmt.Errorf("%w: epsilon is %v, not a positive finite number", ErrConfig, c.Epsilon)
	}
	if !finite(c.Tau) || c.Tau <= 0 {
		return fmt.Errorf("%w: tau is %v, not a positive finite number", ErrConfig, c.Tau)
	}
	if c.K < 1 {
		return fmt.Errorf("%w: k is %d, below one", ErrConfig, c.K)
	}
	if c.Shortlist < c.K {
		return fmt.Errorf("%w: shortlist %d is below k %d", ErrConfig, c.Shortlist, c.K)
	}
	switch c.Competence {
	case StatisticAccuracy, StatisticLowerBound:
	default:
		return fmt.Errorf("%w: competence statistic %q is not implemented", ErrConfig, c.Competence)
	}
	if c.Independence != IndependenceInverseRelatives {
		return fmt.Errorf("%w: independence rule %q is not implemented", ErrConfig, c.Independence)
	}
	if c.Redundancy != RedundancySharedRoots {
		return fmt.Errorf("%w: redundancy rule %q is not implemented", ErrConfig, c.Redundancy)
	}
	return nil
}

// Digest validates the configuration and returns the SHA-256 of its canonical
// encoding under the configuration schema. Bias keys are encoded in sorted
// order, and an empty bias map encodes as an absent one.
func (c Config) Digest() (string, error) {
	if err := c.Validate(); err != nil {
		return "", err
	}
	return digest(struct {
		Schema string `json:"schema"`
		Config Config `json:"config"`
	}{configSchema, c})
}

// LoadConfig decodes a registered configuration and refuses it unless it is
// valid and its canonical digest is want. Unknown fields and trailing data are
// refused, so a field this version does not read cannot pass as registered.
func LoadConfig(data []byte, want string) (Config, error) {
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	var c Config
	if err := decoder.Decode(&c); err != nil {
		return Config{}, fmt.Errorf("%w: decode: %v", ErrConfig, err)
	}
	if _, err := decoder.Token(); err != io.EOF {
		return Config{}, fmt.Errorf("%w: trailing data after the configuration", ErrConfig)
	}
	got, err := c.Digest()
	if err != nil {
		return Config{}, err
	}
	if got != want {
		return Config{}, fmt.Errorf("%w: digest %s is not the registered %s", ErrConfig, got, want)
	}
	return c, nil
}

// digest hashes deterministic Go JSON: struct fields in declaration order and
// map keys sorted. Callers validate first, so no non-finite value reaches it.
func digest(value any) (string, error) {
	encoded, err := json.Marshal(value)
	if err != nil {
		return "", fmt.Errorf("%w: encode digest: %v", ErrInput, err)
	}
	sum := sha256.Sum256(encoded)
	return hex.EncodeToString(sum[:]), nil
}

func finite(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0)
}

// validID admits a non-empty identifier without surrounding whitespace.
func validID(id string) bool {
	return id != "" && strings.TrimSpace(id) == id
}
