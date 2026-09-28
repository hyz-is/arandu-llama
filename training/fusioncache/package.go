package fusioncache

import (
	"bytes"
	"cmp"
	"encoding/json"
	"fmt"
	"io"
	"maps"
	"math"
	"reflect"
	"regexp"
	"slices"
	"strings"
	"unicode/utf8"
)

// CapabilityPackageSchema is the only capability package shape this package
// reads or writes.
const CapabilityPackageSchema = 1

// The kinds of admitted signal a capability package names. Ground truth is
// the dataset's own reference; every other kind comes from consulted teachers.
const (
	SignalGroundTruth = "ground_truth"
	SignalTrajectory  = "trajectory"
	SignalLogits      = "logits"
	SignalFeatures    = "features"
	SignalSeqKD       = "seqkd"
)

// weightTolerance absorbs the rounding of a sum of normalized weights, which
// can exceed one by a few units in the last place.
const weightTolerance = 1e-12

var (
	packageIDPattern  = regexp.MustCompile(`^[a-z0-9]+(?:[._-][a-z0-9]+)*$`)
	capabilityPattern = regexp.MustCompile(`^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)*$`)
	versionedPattern  = regexp.MustCompile(`^[a-z][a-z0-9_]*(?:\.[a-z0-9_]+)*\.v[1-9][0-9]*$`)
	spdxTermPattern   = regexp.MustCompile(`^[A-Za-z0-9.+:()-]+$`)
)

// CapabilityPackage is the immutable provenance of one admitted training unit:
// which capability it teaches, on which example and context, which verifier
// accepted it, which teachers were consulted and how, which signal was
// admitted, under which lineage snapshot and Protection cohort, and the digest
// of every artifact it cites. It records decisions; it computes none.
//
// A correction is a new package: Revision one more than its predecessor's, and
// Supersedes naming the predecessor's digest. Revision 1 supersedes nothing.
//
// ContextSHA256 is the SHA-256 of the exact rendered prompt bytes the unit is
// trained under, as ContextDigest computes it. Teachers are ordered by ID;
// Teachers and Lineage are absent together, when no teacher was consulted.
// Artifacts maps the digest of every artifact the package cites, and nothing
// else, to a kind label; CitedArtifacts lists what it must hold.
type CapabilityPackage struct {
	Schema        int                `json:"schema"`
	Revision      int                `json:"revision"`
	Supersedes    string             `json:"supersedes,omitempty"`
	CapabilityID  string             `json:"capability_id"`
	DatasetSHA256 string             `json:"dataset_sha256"`
	ExampleID     string             `json:"example_id"`
	Role          string             `json:"role"`
	ContextSHA256 string             `json:"context_sha256"`
	Verifier      Verification       `json:"verifier"`
	Teachers      []ConsultedTeacher `json:"teachers,omitempty"`
	Signal        AdmittedSignal     `json:"signal"`
	Lineage       *LineageSnapshot   `json:"lineage,omitempty"`
	Protection    ProtectionCohort   `json:"protection"`
	Router        *RouterDecision    `json:"router,omitempty"`
	Artifacts     map[string]string  `json:"artifacts"`
}

// Verification is the verdict that admitted the unit. ID names the verifier
// and its version together, as in math.numeric_exact_rational.v1, because a
// change to what a verifier accepts is a new identifier. An admitted package
// carries a correct verdict; ReceiptSHA256 names the record of that verdict.
type Verification struct {
	ID            string `json:"id"`
	Correct       bool   `json:"correct"`
	ReceiptSHA256 string `json:"receipt_sha256"`
}

// ConsultedTeacher is one teacher that generated for this example. ID is the
// registry artifact and Family its registry family. Score and Weight are the
// values the selection used, recorded as given. Outcome is the teacher's own
// verified result, whether or not its output was admitted.
type ConsultedTeacher struct {
	ID      string         `json:"id"`
	Family  string         `json:"family"`
	Model   ModelIdentity  `json:"model"`
	License TeacherLicense `json:"license"`
	Score   float64        `json:"score"`
	Weight  float64        `json:"weight"`
	Outcome TeacherOutcome `json:"outcome"`
}

// TeacherLicense is the license read on the exact teacher artifact, as an SPDX
// expression, and the digest of the receipt of that reading.
type TeacherLicense struct {
	SPDX          string `json:"spdx"`
	ReceiptSHA256 string `json:"receipt_sha256"`
}

// TeacherOutcome is one teacher's verified result on the example. A teacher
// that exhausted its generation budget is never correct, because no answer is
// extracted from its reasoning. GenerationReceiptSHA256 names the record of
// the generation itself.
type TeacherOutcome struct {
	Correct                 bool   `json:"correct"`
	BudgetExhausted         bool   `json:"budget_exhausted"`
	GenerationReceiptSHA256 string `json:"generation_receipt_sha256"`
}

// AdmittedSignal names the one signal the unit trains on. RefSHA256 is the
// digest of that signal: the SFT line, the logits cache entry, the feature
// entry. Source is the teacher it came from, empty for ground truth and for a
// weighted signal, which combines the consulted teachers by their Weight.
// Mask, required for logits and features, selects the supervised positions.
type AdmittedSignal struct {
	Kind      string        `json:"kind"`
	Source    string        `json:"source,omitempty"`
	Weighted  bool          `json:"weighted"`
	RefSHA256 string        `json:"ref_sha256"`
	Mask      *PositionMask `json:"mask,omitempty"`
}

// PositionMask selects positions of a sequence of Tokens tokens produced by
// the tokenizer TokenizerSHA256 names. Spans are half-open, ascending and
// separated by at least one unselected position, so one set of positions has
// exactly one encoding.
type PositionMask struct {
	TokenizerSHA256 string     `json:"tokenizer_sha256"`
	Tokens          int        `json:"tokens"`
	Spans           []MaskSpan `json:"spans"`
}

// MaskSpan selects positions [Start, End).
type MaskSpan struct {
	Start int `json:"start"`
	End   int `json:"end"`
}

// LineageSnapshot is the part of the lineage graph the package depends on:
// the registry it was read from, and the ascending lineage roots of every
// consulted teacher, keyed by teacher ID.
type LineageSnapshot struct {
	RegistrySHA256 string              `json:"registry_sha256"`
	Roots          map[string][]string `json:"roots"`
}

// ProtectionCohort names the Protection cohort that gates the transaction this
// unit belongs to, and the digest of that cohort's manifest.
type ProtectionCohort struct {
	ID             string `json:"id"`
	ManifestSHA256 string `json:"manifest_sha256"`
}

// RouterDecision records a routing decision as the router made it, before any
// teacher answered: the router and its version as one identifier, the digests
// of its configuration and of its inputs, the teachers it chose in its order
// with their scores, its reason, and the logical instant of the decision.
// The package checks the record against itself and never recomputes it.
type RouterDecision struct {
	Method           string          `json:"method"`
	ConfigSHA256     string          `json:"config_sha256"`
	CompetenceSHA256 string          `json:"competence_sha256"`
	RegistrySHA256   string          `json:"registry_sha256"`
	Ranking          []RankedTeacher `json:"ranking"`
	Reason           string          `json:"reason"`
	DecidedAt        LogicalInstant  `json:"decided_at"`
}

// RankedTeacher is one teacher the router chose, with the score it assigned.
type RankedTeacher struct {
	Teacher string  `json:"teacher"`
	Score   float64 `json:"score"`
}

// LogicalInstant is a positive tick of a named logical clock. It orders events
// of one run; it is not a wall-clock time.
type LogicalInstant struct {
	Clock string `json:"clock"`
	Tick  uint64 `json:"tick"`
}

// PackageLimits bounds package JSON, the collections inside a package and the
// number of packages one manifest names.
type PackageLimits struct {
	MaxBytes     int64
	MaxTeachers  int
	MaxMaskSpans int
	MaxArtifacts int
	MaxPackages  int
}

func refuse(format string, args ...any) error {
	return fmt.Errorf("%w: "+format, append([]any{ErrContract}, args...)...)
}

func validatePackageLimits(l PackageLimits) error {
	if l.MaxBytes <= 0 || l.MaxBytes == math.MaxInt64 || l.MaxTeachers <= 0 || l.MaxMaskSpans <= 0 || l.MaxArtifacts <= 0 || l.MaxPackages <= 0 {
		return refuse("explicit positive package bounds required")
	}
	return nil
}

func validPackageID(s string) bool { return len(s) <= 128 && packageIDPattern.MatchString(s) }

// validExampleID admits printable ASCII without spaces, so an example ID has
// one spelling and survives any encoding unchanged.
func validExampleID(s string) bool {
	if s == "" || len(s) > 256 {
		return false
	}
	for i := 0; i < len(s); i++ {
		if s[i] <= ' ' || s[i] > '~' {
			return false
		}
	}
	return true
}

// validSPDX admits an SPDX expression of terms separated by single spaces.
func validSPDX(s string) bool {
	if s == "" || len(s) > 256 {
		return false
	}
	for _, term := range strings.Split(s, " ") {
		if !spdxTermPattern.MatchString(term) {
			return false
		}
	}
	return true
}

// canonicalFloat refuses what has no single JSON spelling: NaN, infinities,
// and negative zero, which compares equal to zero but encodes as -0.
func canonicalFloat(v float64) bool { return finite(v) && !(v == 0 && math.Signbit(v)) }

// ValidatePackage refuses a capability package that is incomplete, ambiguous
// or incoherent. It checks what the package records against itself; it does
// not open a cited artifact, recompute a score or re-run a verifier.
func ValidatePackage(p CapabilityPackage, l PackageLimits) error {
	if err := validatePackageLimits(l); err != nil {
		return err
	}
	if p.Schema != CapabilityPackageSchema {
		return refuse("unsupported capability package schema %d", p.Schema)
	}
	if p.Revision < 1 || (p.Revision == 1) != (p.Supersedes == "") || (p.Supersedes != "" && !validSHA(p.Supersedes)) {
		return refuse("revision 1 supersedes nothing and a later revision names its predecessor's digest")
	}
	if !capabilityPattern.MatchString(p.CapabilityID) || len(p.CapabilityID) > 128 {
		return refuse("invalid capability id %q", p.CapabilityID)
	}
	if !validSHA(p.DatasetSHA256) || !validExampleID(p.ExampleID) || !allowedRole(p.Role) || !validSHA(p.ContextSHA256) {
		return refuse("invalid example identity, role or context digest")
	}
	if !versionedPattern.MatchString(p.Verifier.ID) || len(p.Verifier.ID) > 128 || !validSHA(p.Verifier.ReceiptSHA256) {
		return refuse("a versioned verifier id and its receipt digest required")
	}
	if !p.Verifier.Correct {
		return refuse("an admitted package carries a correct verdict")
	}
	if !validPackageID(p.Protection.ID) || !validSHA(p.Protection.ManifestSHA256) {
		return refuse("protection cohort id and manifest digest required")
	}
	teachers, err := validateTeachers(p.Teachers, l)
	if err != nil {
		return err
	}
	if err := validateLineage(p.Lineage, p.Teachers); err != nil {
		return err
	}
	if err := validateSignal(p.Signal, p.Teachers, teachers, l); err != nil {
		return err
	}
	if err := validateRouter(p.Router, p.Lineage, teachers, l); err != nil {
		return err
	}
	return validateArtifacts(p, l)
}

func validateTeachers(teachers []ConsultedTeacher, l PackageLimits) (map[string]ConsultedTeacher, error) {
	if len(teachers) > l.MaxTeachers {
		return nil, refuse("%d consulted teachers exceed the bound of %d", len(teachers), l.MaxTeachers)
	}
	byID := make(map[string]ConsultedTeacher, len(teachers))
	for i, t := range teachers {
		if !validPackageID(t.ID) || !validPackageID(t.Family) {
			return nil, refuse("invalid teacher or family id %q/%q", t.ID, t.Family)
		}
		if i > 0 && t.ID <= teachers[i-1].ID {
			return nil, refuse("consulted teachers must be unique and ordered by id")
		}
		if !validModel(t.Model) {
			return nil, refuse("teacher %s has an invalid model identity", t.ID)
		}
		if !validSPDX(t.License.SPDX) || !validSHA(t.License.ReceiptSHA256) {
			return nil, refuse("teacher %s needs an SPDX license and its receipt digest", t.ID)
		}
		if !canonicalFloat(t.Score) || !canonicalFloat(t.Weight) || t.Weight < 0 || t.Weight > 1 {
			return nil, refuse("teacher %s needs a finite score and a weight in [0, 1]", t.ID)
		}
		if t.Outcome.Correct && t.Outcome.BudgetExhausted {
			return nil, refuse("teacher %s cannot be correct after exhausting its budget", t.ID)
		}
		if !validSHA(t.Outcome.GenerationReceiptSHA256) {
			return nil, refuse("teacher %s needs its generation receipt digest", t.ID)
		}
		for _, other := range teachers[:i] {
			if other.Model == t.Model {
				return nil, refuse("teachers %s and %s are one model", other.ID, t.ID)
			}
		}
		byID[t.ID] = t
	}
	return byID, nil
}

func validateLineage(lineage *LineageSnapshot, teachers []ConsultedTeacher) error {
	if lineage == nil {
		if len(teachers) != 0 {
			return refuse("consulted teachers need a lineage snapshot")
		}
		return nil
	}
	if len(teachers) == 0 {
		return refuse("a lineage snapshot without consulted teachers")
	}
	if !validSHA(lineage.RegistrySHA256) || len(lineage.Roots) != len(teachers) {
		return refuse("lineage needs the registry digest and the roots of exactly the consulted teachers")
	}
	families := map[string][]string{}
	for _, t := range teachers {
		roots, ok := lineage.Roots[t.ID]
		if !ok || len(roots) == 0 {
			return refuse("lineage has no roots for teacher %s", t.ID)
		}
		for i, root := range roots {
			if !validPackageID(root) || (i > 0 && root <= roots[i-1]) {
				return refuse("lineage roots of teacher %s must be valid, unique and ascending", t.ID)
			}
		}
		if known, ok := families[t.Family]; ok && !slices.Equal(known, roots) {
			return refuse("family %s has two different lineage roots", t.Family)
		}
		families[t.Family] = roots
	}
	return nil
}

func validateMask(m *PositionMask, l PackageLimits) error {
	if !validSHA(m.TokenizerSHA256) || m.Tokens < 1 || len(m.Spans) == 0 || len(m.Spans) > l.MaxMaskSpans {
		return refuse("mask needs a tokenizer digest, a length and 1 to %d spans", l.MaxMaskSpans)
	}
	for i, span := range m.Spans {
		if span.Start < 0 || span.Start >= span.End || span.End > m.Tokens || (i > 0 && span.Start <= m.Spans[i-1].End) {
			return refuse("mask spans must be nonempty, within %d tokens, ascending and separated", m.Tokens)
		}
	}
	return nil
}

func validateSignal(s AdmittedSignal, teachers []ConsultedTeacher, byID map[string]ConsultedTeacher, l PackageLimits) error {
	if !validSHA(s.RefSHA256) {
		return refuse("the admitted signal needs its digest")
	}
	if s.Mask != nil {
		if err := validateMask(s.Mask, l); err != nil {
			return err
		}
	}
	switch s.Kind {
	case SignalGroundTruth:
		if s.Source != "" || s.Weighted {
			return refuse("a ground truth signal has no teacher source and no weights")
		}
	case SignalTrajectory, SignalSeqKD:
		if s.Weighted {
			return refuse("a %s signal is one teacher's output, never weighted", s.Kind)
		}
		source, ok := byID[s.Source]
		if !ok {
			return refuse("%s source %q is not a consulted teacher", s.Kind, s.Source)
		}
		if !source.Outcome.Correct {
			return refuse("%s source %s was not verified correct", s.Kind, s.Source)
		}
	case SignalLogits, SignalFeatures:
		if s.Mask == nil {
			return refuse("a %s signal needs its position mask", s.Kind)
		}
		if !s.Weighted {
			if _, ok := byID[s.Source]; !ok {
				return refuse("%s source %q is not a consulted teacher", s.Kind, s.Source)
			}
			return nil
		}
		if s.Source != "" {
			return refuse("a weighted %s signal combines teachers and names no single source", s.Kind)
		}
		sum, positive := 0.0, false
		for _, t := range teachers {
			sum += t.Weight
			positive = positive || t.Weight > 0
		}
		if !positive || sum > 1+weightTolerance {
			return refuse("weights of a weighted signal must include a positive one and sum to at most 1, got %v", sum)
		}
	default:
		return refuse("unknown signal kind %q", s.Kind)
	}
	return nil
}

func validateRouter(r *RouterDecision, lineage *LineageSnapshot, byID map[string]ConsultedTeacher, l PackageLimits) error {
	if r == nil {
		return nil
	}
	if !versionedPattern.MatchString(r.Method) || len(r.Method) > 128 {
		return refuse("router method must be a versioned id, got %q", r.Method)
	}
	if !validSHA(r.ConfigSHA256) || !validSHA(r.CompetenceSHA256) || !validSHA(r.RegistrySHA256) {
		return refuse("router config, competence and registry digests required")
	}
	if r.Reason == "" || strings.TrimSpace(r.Reason) != r.Reason || !utf8.ValidString(r.Reason) {
		return refuse("router reason must be nonempty, trimmed UTF-8")
	}
	if !validPackageID(r.DecidedAt.Clock) || r.DecidedAt.Tick == 0 {
		return refuse("router decision needs a named clock and a positive tick")
	}
	if len(r.Ranking) == 0 || len(r.Ranking) > l.MaxTeachers {
		return refuse("router ranking needs 1 to %d teachers", l.MaxTeachers)
	}
	ranked := make(map[string]bool, len(r.Ranking))
	for _, entry := range r.Ranking {
		teacher, ok := byID[entry.Teacher]
		if !ok {
			return refuse("router chose %q, which is not a consulted teacher", entry.Teacher)
		}
		if ranked[entry.Teacher] {
			return refuse("router ranked %s twice", entry.Teacher)
		}
		if !canonicalFloat(entry.Score) || entry.Score != teacher.Score {
			return refuse("router score of %s differs from the score its teacher records", entry.Teacher)
		}
		ranked[entry.Teacher] = true
	}
	if len(ranked) != len(byID) {
		return refuse("a routed package consults only the teachers the router chose")
	}
	if lineage == nil || lineage.RegistrySHA256 != r.RegistrySHA256 {
		return refuse("router and lineage snapshot read different registries")
	}
	return nil
}

// CitedArtifacts returns, ascending and without repetition, every artifact
// digest p cites outside its Artifacts index: the dataset, the context, the
// verdict, the Protection manifest, the signal and its mask tokenizer, the
// registry, every teacher's model files, license receipt and generation
// receipt, and the router's configuration and inputs. A model revision is a
// version-control pointer, not an artifact digest, and is not listed; neither
// is Supersedes, which names a package frozen in the same store.
func CitedArtifacts(p CapabilityPackage) []string {
	cited := []string{p.DatasetSHA256, p.ContextSHA256, p.Verifier.ReceiptSHA256, p.Protection.ManifestSHA256, p.Signal.RefSHA256}
	if p.Signal.Mask != nil {
		cited = append(cited, p.Signal.Mask.TokenizerSHA256)
	}
	if p.Lineage != nil {
		cited = append(cited, p.Lineage.RegistrySHA256)
	}
	for _, t := range p.Teachers {
		cited = append(cited, t.Model.WeightsSHA256, t.Model.TokenizerSHA256, t.Model.TemplateSHA256, t.Model.RuntimeSHA256, t.License.ReceiptSHA256, t.Outcome.GenerationReceiptSHA256)
	}
	if p.Router != nil {
		cited = append(cited, p.Router.ConfigSHA256, p.Router.CompetenceSHA256, p.Router.RegistrySHA256)
	}
	cited = slices.DeleteFunc(cited, func(s string) bool { return s == "" })
	slices.Sort(cited)
	return slices.Compact(cited)
}

func validateArtifacts(p CapabilityPackage, l PackageLimits) error {
	if len(p.Artifacts) > l.MaxArtifacts {
		return refuse("%d artifacts exceed the bound of %d", len(p.Artifacts), l.MaxArtifacts)
	}
	for digest, kind := range p.Artifacts {
		if !validSHA(digest) || !validPackageID(kind) {
			return refuse("artifact index needs digests mapped to kind labels")
		}
	}
	cited := CitedArtifacts(p)
	if !slices.Equal(cited, slices.Sorted(maps.Keys(p.Artifacts))) {
		return refuse("artifact index must hold exactly the %d digests the package cites", len(cited))
	}
	return nil
}

// ContextDigest returns the SHA-256 of rendered prompt bytes, the value
// CapabilityPackage.ContextSHA256 records.
func ContextDigest(rendered []byte) string { return byteDigest(rendered) }

// CanonicalPackage returns a copy of p in the order ValidatePackage admits:
// teachers sorted by ID, each teacher's lineage roots sorted, and mask spans
// sorted with overlapping or adjacent spans merged. The router ranking keeps
// its order, which is the router's. Nothing is removed, so a duplicate teacher
// or root is still there for validation to refuse.
func CanonicalPackage(p CapabilityPackage) CapabilityPackage {
	c := p
	c.Teachers = slices.Clone(p.Teachers)
	slices.SortStableFunc(c.Teachers, func(a, b ConsultedTeacher) int { return strings.Compare(a.ID, b.ID) })
	if p.Lineage != nil {
		lineage := LineageSnapshot{RegistrySHA256: p.Lineage.RegistrySHA256, Roots: make(map[string][]string, len(p.Lineage.Roots))}
		for teacher, roots := range p.Lineage.Roots {
			lineage.Roots[teacher] = slices.Sorted(slices.Values(roots))
		}
		c.Lineage = &lineage
	}
	if p.Signal.Mask != nil {
		mask := *p.Signal.Mask
		spans := slices.Clone(mask.Spans)
		slices.SortStableFunc(spans, func(a, b MaskSpan) int { return cmp.Compare(a.Start, b.Start) })
		mask.Spans = nil
		for _, span := range spans {
			// Only well-formed spans merge, so a malformed one stays visible.
			last := len(mask.Spans) - 1
			if last >= 0 && span.Start < span.End && mask.Spans[last].Start < mask.Spans[last].End && span.Start <= mask.Spans[last].End {
				mask.Spans[last].End = max(mask.Spans[last].End, span.End)
				continue
			}
			mask.Spans = append(mask.Spans, span)
		}
		c.Signal.Mask = &mask
	}
	if p.Router != nil {
		router := *p.Router
		router.Ranking = slices.Clone(p.Router.Ranking)
		c.Router = &router
	}
	c.Artifacts = maps.Clone(p.Artifacts)
	return c
}

// EncodePackage validates p and returns its canonical JSON with the SHA-256 of
// those bytes, which is the package digest. The encoding is encoding/json's:
// struct fields in declaration order, map keys sorted, no insignificant
// whitespace, no trailing newline. The digest therefore equals Digest(p).
func EncodePackage(p CapabilityPackage, l PackageLimits) ([]byte, string, error) {
	if err := ValidatePackage(p, l); err != nil {
		return nil, "", err
	}
	encoded, err := json.Marshal(p)
	if err != nil {
		return nil, "", refuse("encode package: %v", err)
	}
	if int64(len(encoded)) > l.MaxBytes {
		return nil, "", refuse("package JSON exceeds its byte limit")
	}
	return encoded, byteDigest(encoded), nil
}

// PackageDigest returns the digest of a valid package: the SHA-256 of its
// canonical JSON. Two packages that differ in anything recorded have
// different digests; map insertion order is not recorded.
func PackageDigest(p CapabilityPackage, l PackageLimits) (string, error) {
	_, digest, err := EncodePackage(p, l)
	return digest, err
}

// DecodePackage reads one capability package from JSON and returns it with its
// digest. Key order and insignificant whitespace do not matter; anything else
// that could make two readers disagree is refused: bytes that are not UTF-8,
// a key named twice, a key encoding/json would match only by ignoring case,
// an unknown key, an optional field written as null or empty instead of being
// omitted, trailing data, and every package ValidatePackage refuses.
func DecodePackage(data []byte, l PackageLimits) (CapabilityPackage, string, error) {
	if err := validatePackageLimits(l); err != nil {
		return CapabilityPackage{}, "", err
	}
	if int64(len(data)) > l.MaxBytes || !utf8.Valid(data) {
		return CapabilityPackage{}, "", refuse("package JSON exceeds its byte limit or is not UTF-8")
	}
	if err := refuseDuplicateKeys(data); err != nil {
		return CapabilityPackage{}, "", err
	}
	var p CapabilityPackage
	if err := decodeStrict(data, &p); err != nil {
		return CapabilityPackage{}, "", err
	}
	canonical, digest, err := EncodePackage(p, l)
	if err != nil {
		return CapabilityPackage{}, "", err
	}
	var read, want any
	if err := json.Unmarshal(data, &read); err != nil {
		return CapabilityPackage{}, "", refuse("invalid JSON: %v", err)
	}
	if err := json.Unmarshal(canonical, &want); err != nil {
		return CapabilityPackage{}, "", refuse("canonical JSON unreadable: %v", err)
	}
	if !reflect.DeepEqual(read, want) {
		return CapabilityPackage{}, "", refuse("package JSON holds keys or values outside its canonical form")
	}
	return p, digest, nil
}

// refuseDuplicateKeys walks the token stream and refuses an object that names
// a key twice, which encoding/json would otherwise resolve to the last value.
func refuseDuplicateKeys(data []byte) error {
	type frame struct {
		keys      map[string]bool
		expectKey bool
	}
	decoder := json.NewDecoder(bytes.NewReader(data))
	var stack []*frame
	for {
		token, err := decoder.Token()
		if err == io.EOF {
			return nil
		}
		if err != nil {
			return refuse("invalid JSON: %v", err)
		}
		var top *frame
		if len(stack) > 0 {
			top = stack[len(stack)-1]
		}
		inObject := top != nil && top.keys != nil
		switch value := token.(type) {
		case json.Delim:
			if value == '}' || value == ']' {
				stack = stack[:len(stack)-1]
				continue
			}
			if inObject {
				top.expectKey = true
			}
			next := &frame{}
			if value == '{' {
				next.keys, next.expectKey = map[string]bool{}, true
			}
			stack = append(stack, next)
		case string:
			if inObject && top.expectKey {
				if top.keys[value] {
					return refuse("JSON key %q appears twice in one object", value)
				}
				top.keys[value], top.expectKey = true, false
				continue
			}
			if inObject {
				top.expectKey = true
			}
		default:
			if inObject {
				top.expectKey = true
			}
		}
	}
}

func unitKey(p CapabilityPackage) string {
	return p.CapabilityID + "\x00" + p.DatasetSHA256 + "\x00" + p.ExampleID
}

// Supersede returns corrected as the revision after previous: Revision one
// more and Supersedes the digest of previous. The correction must keep the
// capability, dataset and example, and must change something else, since a
// correction that changes nothing is not one.
func Supersede(previous, corrected CapabilityPackage, l PackageLimits) (CapabilityPackage, error) {
	prior, err := PackageDigest(previous, l)
	if err != nil {
		return CapabilityPackage{}, err
	}
	if unitKey(previous) != unitKey(corrected) {
		return CapabilityPackage{}, refuse("a correction keeps the capability, dataset and example it corrects")
	}
	before, after := previous, corrected
	before.Revision, before.Supersedes, after.Revision, after.Supersedes = 1, "", 1, ""
	unchanged, err := PackageDigest(before, l)
	if err != nil {
		return CapabilityPackage{}, err
	}
	changed, err := PackageDigest(after, l)
	if err != nil {
		return CapabilityPackage{}, err
	}
	if unchanged == changed {
		return CapabilityPackage{}, refuse("a correction must change the package it corrects")
	}
	corrected.Revision, corrected.Supersedes = previous.Revision+1, prior
	if err := ValidatePackage(corrected, l); err != nil {
		return CapabilityPackage{}, err
	}
	return corrected, nil
}
