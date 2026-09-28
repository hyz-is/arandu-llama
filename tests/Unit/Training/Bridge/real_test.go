package bridge_test

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"math"
	"os"
	"sort"
	"strings"
	"testing"

	"github.com/tayi-ai/arandu-llama/training/bridge"
	"github.com/tayi-ai/arandu-llama/training/tokenizer"
)

// The real vocabularies are opt-in: each variable names a local file, and the
// test is skipped when one is unset. Only tokenizer metadata is read.
const (
	envGPTOSS     = "BRIDGE_GPTOSS_GGUF"
	envGemma      = "BRIDGE_GEMMA_GGUF"
	envOrnithJSON = "BRIDGE_ORNITH_TOKENIZER_JSON"
	envOrnithGGUF = "BRIDGE_ORNITH_GGUF"
)

func realPath(t *testing.T, name string) string {
	t.Helper()
	path := os.Getenv(name)
	if path == "" {
		t.Skipf("%s is unset", name)
	}
	return path
}

func loadReal(t *testing.T, name string) *bridge.Vocabulary {
	t.Helper()
	path := realPath(t, name)
	var (
		v   *bridge.Vocabulary
		err error
	)
	if strings.HasSuffix(path, ".json") {
		v, err = bridge.LoadTokenizerJSON(path)
	} else {
		v, err = bridge.LoadGGUF(path)
	}
	if err != nil {
		t.Fatalf("%s: %v", name, err)
	}
	return v
}

// textID finds the token whose source text is exactly text.
func textID(v *bridge.Vocabulary, text string) (int64, bool) {
	for id := 0; id < v.Size(); id++ {
		if v.Text(int64(id)) == text {
			return int64(id), true
		}
	}
	return 0, false
}

func show(v *bridge.Vocabulary, id int64) string {
	if b, ok := v.Bytes(id); ok {
		return fmt.Sprintf("%q", b)
	}
	return fmt.Sprintf("%s(%s)", v.Text(id), v.Kind(id))
}

func TestRealVocabulariesDecodeToBytes(t *testing.T) {
	for _, name := range []string{envGPTOSS, envGemma, envOrnithJSON} {
		v := loadReal(t, name)
		s := v.Stats()
		t.Logf("%s: scheme %s, %d ids, digest %s", name, v.Scheme(), s.Tokens, v.Digest())
		t.Logf("  kinds: normal %d, byte %d, user-defined %d, control %d, unknown %d, unused %d",
			s.Normal, s.Byte, s.UserDefined, s.Control, s.Unknown, s.Unused)
		t.Logf("  with bytes %d; without %d (control %d + unknown %d + unused %d + unrepresentable %d); ids sharing bytes with a lower id %d",
			s.WithBytes, s.Tokens-s.WithBytes, s.Control, s.Unknown, s.Unused, s.Unrepresentable, s.SharedBytes)
		if s.Unrepresentable != 0 {
			for _, id := range v.Unrepresentable() {
				t.Errorf("  normal token %d %q does not decode", id, v.Text(id))
			}
		}
		if s.WithBytes+s.Control+s.Unknown+s.Unused+s.Unrepresentable != s.Tokens {
			t.Errorf("  counts do not add up: %+v", s)
		}
	}
}

func TestOrnithTokenizerJSONAndGGUFAgreeOnBytes(t *testing.T) {
	json, gguf := loadReal(t, envOrnithJSON), loadReal(t, envOrnithGGUF)
	var differ []string
	for id := int64(0); id < int64(json.Size()); id++ {
		a, _ := json.Bytes(id)
		b, _ := gguf.Bytes(id)
		if string(a) != string(b) || json.Kind(id) != gguf.Kind(id) {
			differ = append(differ, fmt.Sprintf("%d %q json %s gguf %s", id, json.Text(id), json.Kind(id), gguf.Kind(id)))
		}
	}
	t.Logf("ids compared %d (json %d, gguf %d); differing in bytes or kind: %d", json.Size(), json.Size(), gguf.Size(), len(differ))
	for _, d := range differ {
		t.Logf("  %s", d)
	}
	for id := int64(json.Size()); id < int64(gguf.Size()); id++ {
		if _, ok := gguf.Bytes(id); ok {
			t.Errorf("gguf id %d beyond the json vocabulary has bytes: %q", id, gguf.Text(id))
		}
	}
}

type teacherCase struct {
	env      string
	stops    []string
	specials []string
}

var teacherCases = []teacherCase{
	{env: envGPTOSS, stops: []string{"<|return|>", "<|end|>", "<|call|>"}, specials: []string{"<|channel|>"}},
	{env: envGemma, stops: []string{"<eos>", "<turn|>"}, specials: []string{"<|turn>"}},
}

func realBridge(t *testing.T, student *bridge.Vocabulary, c teacherCase) (*bridge.Bridge, []int64, []int64) {
	t.Helper()
	teacher := loadReal(t, c.env)
	var stops, specials []int64
	for _, text := range c.stops {
		id, ok := textID(teacher, text)
		if !ok {
			t.Fatalf("%s: no stop token %q", c.env, text)
		}
		stops = append(stops, id)
	}
	for _, text := range c.specials {
		id, ok := textID(teacher, text)
		if !ok {
			t.Fatalf("%s: no control token %q", c.env, text)
		}
		specials = append(specials, id)
	}
	b, err := bridge.New(teacher, student, stops)
	if err != nil {
		t.Fatal(err)
	}
	return b, stops, specials
}

func TestRealTeachersRouteOntoOrnith(t *testing.T) {
	student := loadReal(t, envOrnithJSON)
	for _, c := range teacherCases {
		t.Run(c.env, func(t *testing.T) {
			b, stops, specials := realBridge(t, student, c)
			teacher := b.Teacher()
			var withBytes, bottom, surface int
			var coverage float64
			for id := int64(0); id < int64(teacher.Size()); id++ {
				w, ok := teacher.Bytes(id)
				if !ok {
					continue
				}
				withBytes++
				u := b.Phi(id)
				if u == bridge.NoToken {
					bottom++
					continue
				}
				s, _ := student.Bytes(u)
				if string(s) == string(w) {
					surface++
				}
				coverage += float64(len(s)) / float64(len(w))
			}
			t.Logf("phi over %d byte-bearing teacher tokens: bottom %d, same bytes as a student token %d (%.2f%%), mean byte coverage of phi %.4f",
				withBytes, bottom, surface, 100*float64(surface)/float64(withBytes), coverage/float64(withBytes))
			if bottom != 0 {
				t.Errorf("a byte-level student has every single byte, yet %d teacher tokens have no prefix", bottom)
			}

			// A plausible next-token distribution after "The answer is":
			// retained top-k plus a stop and a control token, the rest
			// uncovered.
			want := []struct {
				raw string
				p   float64
			}{{" the", 0.30}, {" The", 0.05}, {" their", 0.04}, {" 12", 0.10}, {"12", 0.08}, {" 1", 0.03}, {"\n\n", 0.12}, {".", 0.06}, {"\xc3", 0.01}}
			var d bridge.Distribution
			for _, w := range want {
				id, ok := teacher.Find([]byte(w.raw))
				if !ok {
					t.Logf("  teacher has no token for %q; left out", w.raw)
					continue
				}
				d.Probabilities = append(d.Probabilities, bridge.Probability{Token: id, Probability: w.p})
				d.RetainedMass += w.p
			}
			d.Probabilities = append(d.Probabilities, bridge.Probability{Token: stops[0], Probability: 0.05}, bridge.Probability{Token: specials[0], Probability: 0.02})
			d.RetainedMass += 0.07
			for _, p := range d.Probabilities {
				t.Logf("  teacher %-8d %-14s p=%.2f -> phi %s", p.Token, show(teacher, p.Token), p.Probability, show(student, b.Phi(p.Token)))
			}
			target, err := b.Marginalize(bridge.Query{Teacher: d})
			if err != nil {
				t.Fatal(err)
			}
			logTarget(t, student, "boundary target", target)
			expectPartition(t, target)

			// Inside a teacher token after "1": only tokens continuing "1".
			var inside bridge.Distribution
			for _, w := range []struct {
				raw string
				p   float64
			}{{"1", 0.2}, {"12", 0.3}, {"123", 0.1}, {"15", 0.1}, {"2", 0.1}} {
				if id, ok := teacher.Find([]byte(w.raw)); ok {
					inside.Probabilities = append(inside.Probabilities, bridge.Probability{Token: id, Probability: w.p})
					inside.RetainedMass += w.p
				}
			}
			target, err = b.Marginalize(bridge.Query{Prefix: []byte("1"), Teacher: inside})
			if err != nil {
				t.Fatal(err)
			}
			logTarget(t, student, "target inside a teacher token after \"1\"", target)
			expectPartition(t, target)
		})
	}
}

func logTarget(t *testing.T, student *bridge.Vocabulary, title string, target bridge.Target) {
	t.Helper()
	t.Logf("  %s (%s, support %.4f):", title, target.Bound, target.Support)
	for _, c := range target.Cells {
		t.Logf("    student %-8d %-12s %.6f", c.Token, show(student, c.Token), c.Probability)
	}
	t.Logf("    residual %.6f stop %.6f special %.6f uncovered %.6f chain %.6f -> mass %.17f",
		target.Residual, target.Stop, target.Special, target.Uncovered, target.Chain, target.Mass())
}

// greedy segments text by longest match over v: a valid segmentation of the
// same bytes, standing in for the teacher's own BPE, which this package does
// not implement.
func greedy(t *testing.T, v *bridge.Vocabulary, text string) []int64 {
	t.Helper()
	byBytes := map[string]int64{}
	for id := int64(v.Size()) - 1; id >= 0; id-- {
		if b, ok := v.Bytes(id); ok && v.Kind(id) == bridge.KindNormal {
			byBytes[string(b)] = id
		}
	}
	var out []int64
	for len(text) > 0 {
		n := min(len(text), 64)
		for ; n > 0; n-- {
			if id, ok := byBytes[text[:n]]; ok {
				out = append(out, id)
				break
			}
		}
		if n == 0 {
			t.Fatalf("no token prefixes %q", text)
		}
		text = text[n:]
	}
	return out
}

func TestRealSequenceRowsPreserveMass(t *testing.T) {
	path := realPath(t, envOrnithJSON)
	body, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	sum := sha256.Sum256(body)
	encoder, err := tokenizer.Load(context.Background(), strings.NewReader(string(body)), hex.EncodeToString(sum[:]), tokenizer.DefaultLimits())
	if err != nil {
		t.Fatal(err)
	}
	student := loadReal(t, envOrnithJSON)
	text := "<think>\nThe answer is 12.\n</think>\n\nThe answer is **12**.\n    return x"
	studentIDs, err := encoder.Encode(context.Background(), text)
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range teacherCases {
		t.Run(c.env, func(t *testing.T) {
			b, stops, _ := realBridge(t, student, c)
			teacher := b.Teacher()
			teacherIDs := greedy(t, teacher, text)
			dists := make([]bridge.Distribution, len(teacherIDs))
			for j, realized := range teacherIDs {
				d := bridge.Distribution{Probabilities: []bridge.Probability{{Token: realized, Probability: 0.6}}, RetainedMass: 0.6}
				w, _ := teacher.Bytes(realized)
				if len(w) > 1 {
					if shorter, ok := teacher.Find(w[:len(w)-1]); ok && shorter != realized {
						d.Probabilities = append(d.Probabilities, bridge.Probability{Token: shorter, Probability: 0.15})
						d.RetainedMass += 0.15
					}
				}
				d.Probabilities = append(d.Probabilities, bridge.Probability{Token: stops[0], Probability: 0.05})
				d.RetainedMass += 0.05
				dists[j] = d
			}
			a, err := b.Align(studentIDs, teacherIDs)
			if err != nil {
				t.Fatal(err)
			}
			t.Logf("%d student tokens (Ornith BPE), %d teacher tokens (greedy stand-in)", a.Students(), a.Teachers())
			rows, err := b.Rows(a, dists, bridge.RowOptions{})
			if err != nil {
				t.Fatal(err)
			}
			counts := map[string]int{}
			worst := 0.0
			for _, row := range rows {
				id := studentIDs[row.Position]
				label := row.Relation.String()
				if row.Masked {
					label += " masked"
				}
				counts[label]++
				if row.Target == nil {
					t.Logf("  row %2d %-12s %-10s no target", row.Position, show(student, id), label)
					continue
				}
				worst = max(worst, math.Abs(row.Target.Mass()-1))
				expectPartition(t, *row.Target)
				t.Logf("  row %2d %-12s %-10s %-11s teacher %-14s p(realized)=%.4f chain %.4f cells %d mass %.17f",
					row.Position, show(student, id), label, row.Target.Bound, show(teacher, teacherIDs[row.Teacher]),
					row.Target.Probability(id), row.Target.Chain, len(row.Target.Cells), row.Target.Mass())
			}
			keys := make([]string, 0, len(counts))
			for k := range counts {
				keys = append(keys, k)
			}
			sort.Strings(keys)
			var summary []string
			for _, k := range keys {
				summary = append(summary, fmt.Sprintf("%s=%d", k, counts[k]))
			}
			t.Logf("relations: %s; largest |mass-1| %.3g", strings.Join(summary, " "), worst)
		})
	}
}
