package router

import (
	"fmt"
	"regexp"
	"slices"
	"strings"
)

// Gates are the admission conditions of a teacher. A teacher that fails any
// of them is excluded with the reason recorded; a gate never enters the score.
//
// ArtifactSHA256 is the pinned digest of the exact artifact, 64 lowercase
// hexadecimal characters; any other value, empty included, is an unpinned
// artifact. Qualified reports that the teacher passed qualification,
// including its verifier checks.
type Gates struct {
	LicenseVerified bool   `json:"license_verified"`
	ArtifactSHA256  string `json:"artifact_sha256"`
	Qualified       bool   `json:"qualified"`
}

// Competence is one measured cell of the Teacher x Capability matrix: a
// teacher's result on the items of one subcapability, with the confidence
// interval of its accuracy. Accuracy must be exactly Correct/Items.
type Competence struct {
	Subcapability string  `json:"subcapability"`
	Items         int     `json:"items"`
	Correct       int     `json:"correct"`
	Accuracy      float64 `json:"accuracy"`
	CILower       float64 `json:"ci_lower"`
	CIUpper       float64 `json:"ci_upper"`
}

// Teacher is everything the router reads about a teacher, all of it measured
// before any example is routed.
//
// Roots are the teacher's lineage roots in the registry's lineage graph: the
// root families it descends from. Reliability (V) and Reproducibility (R) lie
// in [0, 1]; Cost (K) is a non-negative number in whatever unit the
// registered a_k was chosen for. Competence holds one cell per subcapability.
type Teacher struct {
	ID              string       `json:"id"`
	Roots           []string     `json:"roots"`
	Gates           Gates        `json:"gates"`
	Reliability     float64      `json:"reliability"`
	Reproducibility float64      `json:"reproducibility"`
	Cost            float64      `json:"cost"`
	Competence      []Competence `json:"competence"`
}

// Example identifies the item being routed: its id and the subcapability
// whose competence cell scores the teachers. It deliberately carries nothing
// a teacher produced.
type Example struct {
	ID            string `json:"id"`
	Subcapability string `json:"subcapability"`
}

// Reason is why a teacher was excluded from a decision.
type Reason string

const (
	// ReasonLicenseNotVerified excludes a teacher whose artifact license was
	// not checked for the intended use.
	ReasonLicenseNotVerified Reason = "license_not_verified"
	// ReasonArtifactNotPinned excludes a teacher without a pinned artifact
	// digest.
	ReasonArtifactNotPinned Reason = "artifact_not_pinned"
	// ReasonNotQualified excludes a teacher that did not pass qualification.
	ReasonNotQualified Reason = "not_qualified"
	// ReasonCompetenceNotMeasured excludes a teacher without a competence
	// cell for the routed subcapability. An absent measurement is not a low
	// one, and it is not borrowed from a parent capability.
	ReasonCompetenceNotMeasured Reason = "competence_not_measured"
)

var pinnedDigest = regexp.MustCompile(`^[0-9a-f]{64}$`)

const teachersSchema = "tayi.tcap.router.teachers.v1"

// canonical validates the teacher table and returns a copy sorted by id, with
// every teacher's roots and competence cells sorted, so that the order the
// caller built them in changes neither the decision nor any digest.
func canonical(teachers []Teacher) ([]Teacher, error) {
	if len(teachers) == 0 {
		return nil, fmt.Errorf("%w: no teachers", ErrInput)
	}
	out := make([]Teacher, len(teachers))
	seen := make(map[string]bool, len(teachers))
	for i, t := range teachers {
		if !validID(t.ID) {
			return nil, fmt.Errorf("%w: teacher %d has id %q", ErrInput, i, t.ID)
		}
		if seen[t.ID] {
			return nil, fmt.Errorf("%w: teacher %s appears twice", ErrInput, t.ID)
		}
		seen[t.ID] = true
		if err := checkTeacher(t); err != nil {
			return nil, err
		}
		t.Roots = slices.Clone(t.Roots)
		slices.Sort(t.Roots)
		t.Competence = slices.Clone(t.Competence)
		slices.SortFunc(t.Competence, func(a, b Competence) int { return strings.Compare(a.Subcapability, b.Subcapability) })
		out[i] = t
	}
	slices.SortFunc(out, func(a, b Teacher) int { return strings.Compare(a.ID, b.ID) })
	return out, nil
}

func checkTeacher(t Teacher) error {
	if len(t.Roots) == 0 {
		return fmt.Errorf("%w: teacher %s declares no lineage root", ErrInput, t.ID)
	}
	roots := make(map[string]bool, len(t.Roots))
	for _, root := range t.Roots {
		if !validID(root) || roots[root] {
			return fmt.Errorf("%w: teacher %s has an empty or repeated lineage root %q", ErrInput, t.ID, root)
		}
		roots[root] = true
	}
	if !unit(t.Reliability) || !unit(t.Reproducibility) {
		return fmt.Errorf("%w: teacher %s reliability %v or reproducibility %v is outside [0, 1]", ErrInput, t.ID, t.Reliability, t.Reproducibility)
	}
	if !finite(t.Cost) || t.Cost < 0 {
		return fmt.Errorf("%w: teacher %s cost is %v, not a finite non-negative number", ErrInput, t.ID, t.Cost)
	}
	cells := make(map[string]bool, len(t.Competence))
	for _, c := range t.Competence {
		if !validID(c.Subcapability) || cells[c.Subcapability] {
			return fmt.Errorf("%w: teacher %s has an empty or repeated competence cell %q", ErrInput, t.ID, c.Subcapability)
		}
		cells[c.Subcapability] = true
		if err := checkCompetence(c); err != nil {
			return fmt.Errorf("%w: teacher %s cell %s: %v", ErrInput, t.ID, c.Subcapability, err)
		}
	}
	return nil
}

func checkCompetence(c Competence) error {
	if c.Items < 1 || c.Correct < 0 || c.Correct > c.Items {
		return fmt.Errorf("%d correct of %d items", c.Correct, c.Items)
	}
	if !unit(c.Accuracy) || !unit(c.CILower) || !unit(c.CIUpper) {
		return fmt.Errorf("accuracy %v or interval [%v, %v] is outside [0, 1]", c.Accuracy, c.CILower, c.CIUpper)
	}
	if c.Accuracy != float64(c.Correct)/float64(c.Items) {
		return fmt.Errorf("accuracy %v is not %d/%d", c.Accuracy, c.Correct, c.Items)
	}
	if c.CILower > c.CIUpper {
		return fmt.Errorf("interval [%v, %v] is inverted", c.CILower, c.CIUpper)
	}
	return nil
}

// gates returns every reason the teacher is excluded when scored on the
// competence cell key, in a fixed order, and the cell when it exists.
func gates(t Teacher, key string) ([]Reason, Competence) {
	var reasons []Reason
	if !t.Gates.LicenseVerified {
		reasons = append(reasons, ReasonLicenseNotVerified)
	}
	if !pinnedDigest.MatchString(t.Gates.ArtifactSHA256) {
		reasons = append(reasons, ReasonArtifactNotPinned)
	}
	if !t.Gates.Qualified {
		reasons = append(reasons, ReasonNotQualified)
	}
	index := slices.IndexFunc(t.Competence, func(c Competence) bool { return c.Subcapability == key })
	if index < 0 {
		return append(reasons, ReasonCompetenceNotMeasured), Competence{}
	}
	return reasons, t.Competence[index]
}

// unit reports whether value is a finite number in [0, 1].
func unit(value float64) bool {
	return finite(value) && value >= 0 && value <= 1
}

// sharedRoots counts the lineage roots two sorted root lists have in common.
func sharedRoots(a, b []string) int {
	shared := 0
	for _, root := range a {
		if _, found := slices.BinarySearch(b, root); found {
			shared++
		}
	}
	return shared
}
