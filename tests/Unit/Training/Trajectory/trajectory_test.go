package trajectory_test

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	llama "github.com/tayi-ai/arandu-llama"
	"github.com/tayi-ai/arandu-llama/training/tokenizer"
	"github.com/tayi-ai/arandu-llama/training/trajectory"
)

// A verified teacher trajectory, turned into a row the student trains on --
// and the answer alone, the row that carries no trajectory.
//
// The trajectory is a real one: Qwen3.8's answer to gsm8k-train-1468 from the
// teacher smoke run, split into reasoning and answer by the parser Chat uses.
// The student is the Ornith, whose template is the file its repository
// publishes. The tests that always run hold the text, where the template
// decides everything; the tokens need the student's 12.8 MB tokenizer, which
// is not in this repository, so they run against a small byte-level one and,
// when TRAJECTORY_ORNITH_TOKENIZER_JSON names the file, against the real one.
//
// None of these run in parallel: a failed render writes the wrapper's one
// error slot.

const ornithEndOfTurn = "<|im_end|>"

func packageRoot(t *testing.T) string {
	t.Helper()

	root, err := filepath.Abs(filepath.Join("..", "..", "..", ".."))
	if err != nil {
		t.Fatalf("resolving the package root: %v", err)
	}
	return root
}

func readFile(t *testing.T, parts ...string) string {
	t.Helper()

	body, err := os.ReadFile(filepath.Join(append([]string{packageRoot(t)}, parts...)...))
	if err != nil {
		t.Fatalf("reading %s: %v", filepath.Join(parts...), err)
	}
	return string(body)
}

func ornithTemplate(t *testing.T) string {
	t.Helper()
	return readFile(t, "testdata", "chat_templates", "ornith-1.5-9b.jinja")
}

// The teacher outputs under testdata/chat_outputs that ended with the model's
// own stop. Each is a different family's markup around the same kind of text.
var finishedOutputs = []string{
	"qwen3.8-27b.gsm8k-train-1468",
	"gpt-oss-20b.gsm8k-train-1468",
	"gemma-4-12b-it.gsm8k-train-6546",
}

// recordedTrajectory is a teacher smoke output read back the way Chat reads
// it, as the Example the student would be trained on. The qualification of
// which trajectories deserve a row is the verifier's, and is not made here.
func recordedTrajectory(t *testing.T, name string) trajectory.Example {
	t.Helper()

	var recorded struct {
		Template string `json:"template"`
		ID       string `json:"id"`
		User     string `json:"user"`
		Finish   string `json:"finish"`
		Output   string `json:"output"`
	}
	if err := json.Unmarshal([]byte(readFile(t, "testdata", "chat_outputs", name+".json")), &recorded); err != nil {
		t.Fatalf("decoding the %s output: %v", name, err)
	}
	messages := []llama.ChatMessage{{Role: "user", Content: recorded.User}}
	opts := llama.ChatOptions{
		ChatTemplate:    readFile(t, "testdata", "chat_templates", recorded.Template+".jinja"),
		ReasoningFormat: llama.ReasoningFormatAuto,
	}
	response, err := llama.ParseChatOutput(messages, opts, recorded.Output, llama.FinishReason(recorded.Finish))
	if err != nil {
		t.Fatalf("parsing the %s output: %v", name, err)
	}
	return trajectory.Example{
		ID:        recorded.ID + "/" + recorded.Template,
		Messages:  messages,
		Reasoning: response.ReasoningContent,
		Answer:    response.Content,
	}
}

func abbreviate(s string) string {
	if len(s) <= 160 {
		return s
	}
	return s[:80] + " [...] " + s[len(s)-80:]
}

func TestTheOrnithRendersAVerifiedTrajectory(t *testing.T) {
	example := recordedTrajectory(t, "qwen3.8-27b.gsm8k-train-1468")
	text, err := trajectory.Render(ornithTemplate(t), ornithEndOfTurn, example)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("prompt     %q", abbreviate(text.Prompt))
	t.Logf("completion %q", abbreviate(text.Completion))
	t.Logf("tail       %q", text.Tail)

	// The prompt is what the student is given with thinking on: the question,
	// then an assistant turn open inside <think>. It carries no </think>, so
	// nothing the student is given closes the block for it.
	question := strings.TrimSpace(example.Messages[0].Content)
	if want := "<|im_start|>user\n" + question + "<|im_end|>\n<|im_start|>assistant\n<think>\n"; text.Prompt != want {
		t.Fatalf("the prompt is\n%q\nwant\n%q", text.Prompt, want)
	}
	if strings.Contains(text.Prompt, "</think>") {
		t.Errorf("the prompt closes the thinking block: %q", text.Prompt)
	}

	// The completion is the reasoning, the template's close of the block, the
	// answer and the end of turn -- and nothing after it. The template trims
	// both texts, which the teacher's reasoning ends with a newline to show.
	reasoning, answer := strings.TrimSpace(example.Reasoning), strings.TrimSpace(example.Answer)
	if reasoning == example.Reasoning {
		t.Errorf("the recorded reasoning has no surrounding whitespace to trim; the fixture no longer shows the template trimming")
	}
	if want := reasoning + "\n</think>\n\n" + answer + ornithEndOfTurn; text.Completion != want {
		t.Fatalf("the completion is\n%q\nwant\n%q", abbreviate(text.Completion), abbreviate(want))
	}
	if strings.Contains(text.Completion, "<think>") {
		t.Errorf("the completion opens a second thinking block")
	}
	if got, want := strings.Index(text.Completion, "</think>"), len(reasoning)+1; got != want || strings.Count(text.Completion, "</think>") != 1 {
		t.Errorf("</think> is at byte %d of the completion (%d times), want once at %d, right after the reasoning",
			got, strings.Count(text.Completion, "</think>"), want)
	}
	if !strings.HasSuffix(text.Completion, ornithEndOfTurn) || strings.Count(text.Completion, ornithEndOfTurn) != 1 {
		t.Errorf("the completion does not end with its one %s", ornithEndOfTurn)
	}

	// What the template writes after the turn stays out: generation stops on
	// <|im_end|>, and the student never writes the newline behind it.
	if text.Tail != "\n" {
		t.Errorf("the tail is %q, want the template's newline", text.Tail)
	}
}

// chatMLTurns renders every turn the same way, reasoning never. The two
// templates built on it each add a generation prompt that breaks the row in
// one place.
const (
	chatMLTurns = "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}"
	// thinkingPrompt opens a thinking block the assistant's own turn never
	// writes, so the prompt is not where the conversation continues.
	thinkingPrompt = chatMLTurns + "{% if add_generation_prompt %}<|im_start|>assistant\n<think>\n{% endif %}"
	// plainPrompt continues the conversation but writes no reasoning.
	plainPrompt = chatMLTurns + "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)

var arithmetic = trajectory.Example{
	ID:        "arithmetic",
	Messages:  []llama.ChatMessage{{Role: "user", Content: "What is 2+3?"}},
	Reasoning: "Two plus three is five.",
	Answer:    "#### 5",
}

func TestAPromptTheConversationDoesNotContinueIsRefused(t *testing.T) {
	// Trained on this template's completion, the student would learn to
	// answer after "<|im_start|>assistant\n" and be given "<think>\n" too at
	// inference: a row whose prompt is not the one it is asked from.
	text, err := trajectory.Render(thinkingPrompt, ornithEndOfTurn, arithmetic)
	if err == nil {
		t.Fatalf("a prompt the conversation does not begin with gave %+v, want an error", text)
	}
	if !strings.Contains(err.Error(), "does not begin with the prompt") || !strings.Contains(err.Error(), "<think>") {
		t.Errorf("the refusal %q does not show where the renders part", err)
	}
}

func TestATemplateThatDropsTheReasoningIsRefused(t *testing.T) {
	// The prefix holds, and the completion is a fine answer -- without the
	// reasoning the row was built to teach.
	text, err := trajectory.Render(plainPrompt, ornithEndOfTurn, arithmetic)
	if err == nil {
		t.Fatalf("a template without reasoning_content gave %+v, want an error", text)
	}
	if !strings.Contains(err.Error(), "dropped the reasoning") {
		t.Errorf("the refusal %q does not say the reasoning is missing", err)
	}
}

func TestAnExampleWithoutAQuestionReasoningOrAnswerIsRefused(t *testing.T) {
	template := ornithTemplate(t)
	for name, change := range map[string]func(*trajectory.Example){
		"no ID":       func(e *trajectory.Example) { e.ID = "" },
		"no messages": func(e *trajectory.Example) { e.Messages = nil },
		"an answered question": func(e *trajectory.Example) {
			e.Messages = append(slices.Clone(e.Messages), llama.ChatMessage{Role: "assistant", Content: "5"})
		},
		"empty reasoning":       func(e *trajectory.Example) { e.Reasoning = "" },
		"whitespace reasoning":  func(e *trajectory.Example) { e.Reasoning = " \n\t" },
		"empty answer":          func(e *trajectory.Example) { e.Answer = "" },
		"whitespace answer":     func(e *trajectory.Example) { e.Answer = "\n\n" },
		"an answer ending turn": func(e *trajectory.Example) { e.Answer = "#### 5" + ornithEndOfTurn + "\nmore" },
	} {
		example := arithmetic
		change(&example)
		if text, err := trajectory.Render(template, ornithEndOfTurn, example); err == nil {
			t.Errorf("%s rendered %+v, want an error", name, text)
		}
	}
	for _, end := range []string{"", "<|endoftext|>"} {
		if text, err := trajectory.Render(template, end, arithmetic); err == nil {
			t.Errorf("end of turn %q rendered %+v, want an error", end, text)
		}
	}
}

func TestATrajectoryCutAtTheCeilingIsRefused(t *testing.T) {
	// Gemma 4 spent its 1024 tokens thinking and never answered. Its reasoning
	// is real; a row from it would teach the student to think and then reply
	// with nothing.
	example := recordedTrajectory(t, "gemma-4-12b-it.gsm8k-train-1468")
	if example.Reasoning == "" || example.Answer != "" {
		t.Fatalf("the cut output parsed to reasoning %q and answer %q, want reasoning only", abbreviate(example.Reasoning), example.Answer)
	}
	text, err := trajectory.Render(ornithTemplate(t), ornithEndOfTurn, example)
	if err == nil || !strings.Contains(err.Error(), "empty answer") {
		t.Fatalf("a trajectory without an answer gave %+v and %v, want the empty answer refused", text, err)
	}
}

// answerOnly is example's row without its trajectory: the same question, the
// level declared, and answer as the whole of the assistant's turn.
func answerOnly(example trajectory.Example, answer string) trajectory.Example {
	return trajectory.Example{ID: example.ID, Messages: example.Messages, AnswerOnly: true, Answer: answer}
}

// gsm8k1468Answer is the reference answer of gsm8k-train-1468 in the form the
// verifier reads: 23 meters farther each way, twice a day, for five days.
const gsm8k1468Answer = "#### 230"

func TestAnAnswerOnlyRowIsGivenTheTrajectoryRowsPrompt(t *testing.T) {
	template := ornithTemplate(t)
	full := recordedTrajectory(t, "qwen3.8-27b.gsm8k-train-1468")
	withReasoning, err := trajectory.Render(template, ornithEndOfTurn, full)
	if err != nil {
		t.Fatal(err)
	}
	text, err := trajectory.Render(template, ornithEndOfTurn, answerOnly(full, gsm8k1468Answer))
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("prompt     %q", abbreviate(text.Prompt))
	t.Logf("completion %q", text.Completion)
	t.Logf("tail       %q", text.Tail)

	// The two rows differ in what the student writes, never in what it is
	// given: the same bytes, thinking on, open inside <think>.
	if text.Prompt != withReasoning.Prompt {
		at := 0
		for at < len(text.Prompt) && at < len(withReasoning.Prompt) && text.Prompt[at] == withReasoning.Prompt[at] {
			at++
		}
		t.Fatalf("the answer-only prompt parts from the trajectory's at byte %d of %d and %d", at, len(text.Prompt), len(withReasoning.Prompt))
	}
	if !strings.HasSuffix(text.Prompt, "<|im_start|>assistant\n<think>\n") {
		t.Errorf("the answer-only prompt does not end inside the thinking block: %q", abbreviate(text.Prompt))
	}

	// What the Ornith's template writes for an assistant turn with empty
	// reasoning: a newline that leaves the block the prompt opened empty, its
	// close, a blank line, the answer and the end of turn.
	if want := "\n</think>\n\n" + gsm8k1468Answer + ornithEndOfTurn; text.Completion != want {
		t.Fatalf("the answer-only completion is\n%q\nwant\n%q", text.Completion, want)
	}
	if text.Tail != withReasoning.Tail || text.Tail != "\n" {
		t.Errorf("the answer-only tail is %q, want the template's newline", text.Tail)
	}
}

func TestTheOrnithWritesTheAnswerOnlyTurnItIsGiven(t *testing.T) {
	// The template trims the answer, so surrounding space does not reach the
	// row; the answer is otherwise written as it is.
	template := ornithTemplate(t)
	for answer, want := range map[string]string{
		"#### 5":      "\n</think>\n\n#### 5<|im_end|>",
		"#### 42":     "\n</think>\n\n#### 42<|im_end|>",
		"  #### 42\n": "\n</think>\n\n#### 42<|im_end|>",
		"#### -3/4":   "\n</think>\n\n#### -3/4<|im_end|>",
	} {
		text, err := trajectory.Render(template, ornithEndOfTurn, answerOnly(arithmetic, answer))
		if err != nil {
			t.Errorf("answer %q: %v", answer, err)
			continue
		}
		if text.Completion != want {
			t.Errorf("answer %q gave the completion %q, want %q", answer, text.Completion, want)
		}
	}
}

func TestTheLevelIsDeclaredAndTheReasoningAgreesWithIt(t *testing.T) {
	template := ornithTemplate(t)
	for name, c := range map[string]struct {
		example trajectory.Example
		refusal string
	}{
		"an answer-only example with reasoning": {
			trajectory.Example{ID: "a", Messages: arithmetic.Messages, AnswerOnly: true, Reasoning: arithmetic.Reasoning, Answer: arithmetic.Answer},
			"reasoning in an answer-only example"},
		"an answer-only example with whitespace reasoning": {
			trajectory.Example{ID: "a", Messages: arithmetic.Messages, AnswerOnly: true, Reasoning: " \n", Answer: arithmetic.Answer},
			"reasoning in an answer-only example"},
		"a trajectory without reasoning": {
			trajectory.Example{ID: "a", Messages: arithmetic.Messages, Answer: arithmetic.Answer},
			"empty reasoning in an example that does not declare AnswerOnly"},
		"a trajectory with whitespace reasoning": {
			trajectory.Example{ID: "a", Messages: arithmetic.Messages, Reasoning: "\n\t", Answer: arithmetic.Answer},
			"empty reasoning in an example that does not declare AnswerOnly"},
		"an answer-only example without an answer": {
			trajectory.Example{ID: "a", Messages: arithmetic.Messages, AnswerOnly: true},
			"empty answer"},
		"an answer-only example with a whitespace answer": {
			trajectory.Example{ID: "a", Messages: arithmetic.Messages, AnswerOnly: true, Answer: " \n"},
			"empty answer"},
	} {
		text, err := trajectory.Render(template, ornithEndOfTurn, c.example)
		if err == nil || !strings.Contains(err.Error(), c.refusal) {
			t.Errorf("%s gave %+v and %v, want %q", name, text, err, c.refusal)
		}
	}
}

func TestAnAnswerOnlyTurnHoldsTheAnswerAndNothingElse(t *testing.T) {
	template := ornithTemplate(t)

	// Handed no reasoning_content, the Ornith's template takes what precedes
	// a </think> in the content as reasoning: this answer renders as a
	// trajectory, the reasoning inside the block the prompt opened.
	smuggled := answerOnly(arithmetic, arithmetic.Reasoning+"\n</think>\n\n#### 5")
	conversation := append(slices.Clone(smuggled.Messages), llama.ChatMessage{Role: "assistant", Content: smuggled.Answer})
	full, err := llama.RenderChatTemplate(conversation, llama.ChatOptions{ChatTemplate: template, EnableThinking: llama.Bool(true), AddGenerationPrompt: llama.Bool(false)})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(full, "<think>\n"+arithmetic.Reasoning+"\n</think>\n\n#### 5<|im_end|>") {
		t.Fatalf("the template no longer takes reasoning out of the content, and this case shows nothing: %q", full)
	}
	text, err := trajectory.Render(template, ornithEndOfTurn, smuggled)
	if err == nil || !strings.Contains(err.Error(), "does not open with") || !strings.Contains(err.Error(), arithmetic.Reasoning) {
		t.Errorf("an answer the template turns into a trajectory gave %+v and %v, want the reasoning refused", text, err)
	}

	// A thinking template that drops the content: the turn opens as the empty
	// turn does, and there is no answer after the opening.
	const droppedAnswer = "{% for m in messages %}{% if m.role == 'user' %}<|im_start|>user\n{{ m.content }}<|im_end|>\n" +
		"{% else %}<|im_start|>assistant\n<think>\n{{ m.reasoning_content }}\n</think>\n\n<|im_end|>\n{% endif %}{% endfor %}" +
		"{% if add_generation_prompt %}<|im_start|>assistant\n<think>\n{% endif %}"
	text, err = trajectory.Render(droppedAnswer, ornithEndOfTurn, answerOnly(arithmetic, "#### 5"))
	if err == nil || !strings.Contains(err.Error(), "want the answer") {
		t.Errorf("a template that drops the answer gave %+v and %v, want a refusal", text, err)
	}

	// The opening is the template's, not the Ornith's: one that writes no
	// thinking has none, refuses the trajectory, and takes the answer alone.
	if _, err := trajectory.Render(plainPrompt, ornithEndOfTurn, arithmetic); err == nil {
		t.Errorf("a template without reasoning took the trajectory")
	}
	text, err = trajectory.Render(plainPrompt, ornithEndOfTurn, answerOnly(arithmetic, "#### 5"))
	if err != nil {
		t.Fatal(err)
	}
	if want := "#### 5" + ornithEndOfTurn; text.Completion != want {
		t.Errorf("a template without thinking gave the completion %q, want %q", text.Completion, want)
	}
}

// byteLevel builds a tokenizer in the schema training/tokenizer admits: one
// token per byte, the added tokens given, and merges spelling each word.
func byteLevel(t *testing.T, added map[string]int64, words ...string) *tokenizer.Tokenizer {
	t.Helper()

	var alphabet [256]string
	extra := rune(256)
	for b := range 256 {
		r := rune(b)
		if b < 33 || b > 126 && b < 161 || b == 173 {
			r = extra
			extra++
		}
		alphabet[b] = string(r)
	}
	symbol := func(raw string) string {
		var out strings.Builder
		for i := 0; i < len(raw); i++ {
			out.WriteString(alphabet[raw[i]])
		}
		return out.String()
	}
	vocab := map[string]int64{}
	for b := range 256 {
		vocab[alphabet[b]] = int64(b)
	}
	merges := []string{}
	next := int64(256)
	for _, word := range words {
		for i := 2; i <= len(word); i++ {
			if _, ok := vocab[symbol(word[:i])]; ok {
				continue
			}
			merges = append(merges, symbol(word[:i-1])+" "+symbol(word[i-1:i]))
			vocab[symbol(word[:i])] = next
			next++
		}
	}
	tokens := []any{}
	for content, id := range added {
		tokens = append(tokens, map[string]any{"id": id, "content": content, "single_word": false,
			"lstrip": false, "rstrip": false, "normalized": false, "special": true})
	}
	level := map[string]any{"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false}
	pattern := `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`
	config := map[string]any{"version": "1.0", "truncation": nil, "padding": nil, "added_tokens": tokens,
		"normalizer": map[string]any{"type": "NFC"},
		"pre_tokenizer": map[string]any{"type": "Sequence", "pretokenizers": []any{
			map[string]any{"type": "Split", "pattern": map[string]any{"Regex": pattern}, "behavior": "Isolated", "invert": false}, level}},
		"post_processor": level, "decoder": level, "model": map[string]any{"type": "BPE", "dropout": nil,
			"unk_token": nil, "continuing_subword_prefix": "", "end_of_word_suffix": "", "fuse_unk": false,
			"byte_fallback": false, "ignore_merges": false, "vocab": vocab, "merges": merges}}
	body, err := json.Marshal(config)
	if err != nil {
		t.Fatal(err)
	}
	sum := sha256.Sum256(body)
	loaded, err := tokenizer.Load(context.Background(), bytes.NewReader(body), hex.EncodeToString(sum[:]), tokenizer.DefaultLimits())
	if err != nil {
		t.Fatalf("loading the byte-level tokenizer: %v", err)
	}
	return loaded
}

// ornithSpecials are the Ornith's ids for the pieces its template writes, as
// its tokenizer.json numbers them; the opt-in test checks them against it.
var ornithSpecials = map[string]int64{"<|im_start|>": 248045, "<|im_end|>": 248046, "<think>": 248068, "</think>": 248069}

// ornithEmbeddingRows is the vocabulary size the Ornith-1.5-9B GGUF records,
// 250 rows past the last id its tokenizer.json assigns.
const ornithEmbeddingRows = 248320

// curriculumRow is training/local's example, whose type is unexported: the
// same four fields under the same JSON names.
type curriculumRow struct {
	ID           string  `json:"id"`
	InputIDs     []int64 `json:"input_ids"`
	Labels       []int64 `json:"labels"`
	PromptTokens int     `json:"prompt_tokens"`
}

// admitted applies to one encoded line what training/local's
// curriculumPosition applies to every line it reads. That function is
// unexported and reads a file bound to a full recipe, so the rule is replicated
// here, and TestTheReplicatedRuleIsTheCurriculums holds the replica to the
// source.
func admitted(t *testing.T, line []byte) (curriculumRow, error) {
	t.Helper()

	decoder := json.NewDecoder(bytes.NewReader(line))
	decoder.DisallowUnknownFields()
	var row curriculumRow
	if err := decoder.Decode(&row); err != nil {
		return row, err
	}
	if row.ID == "" || len(row.InputIDs) != len(row.Labels) || row.PromptTokens < 1 || row.PromptTokens >= len(row.InputIDs) {
		return row, errors.New("curriculum order or geometry differs")
	}
	for i, label := range row.Labels {
		if row.InputIDs[i] < 0 || i < row.PromptTokens && label != -100 || i >= row.PromptTokens && label != row.InputIDs[i] {
			return row, errors.New("curriculum token or mask differs")
		}
	}
	return row, nil
}

func TestTheReplicatedRuleIsTheCurriculums(t *testing.T) {
	// If training/local changes what it admits, this replica is stale, and
	// every row these tests call admitted is admitted by a rule nobody runs.
	source := readFile(t, "training", "local", "curriculum.go")
	for _, rule := range []string{
		`row.ID == "" || seen[row.ID] || c.Recipe.RequireLengthOrder && len(row.InputIDs) < lastLength || len(row.InputIDs) != len(row.Labels) || row.PromptTokens < 1 || row.PromptTokens >= len(row.InputIDs)`,
		`row.InputIDs[i] < 0 || i < row.PromptTokens && label != -100 || i >= row.PromptTokens && label != row.InputIDs[i]`,
	} {
		if !strings.Contains(source, rule) {
			t.Errorf("training/local/curriculum.go no longer holds the rule replicated here:\n%s", rule)
		}
	}
	record := readFile(t, "training", "local", "record.go")
	for _, field := range []string{"`json:\"id\"`", "`json:\"input_ids\"`", "`json:\"labels\"`", "`json:\"prompt_tokens\"`"} {
		if !strings.Contains(record, field) {
			t.Errorf("training/local/record.go no longer names the field %s", field)
		}
	}
	if trajectory.IgnoreLabel != -100 {
		t.Errorf("IgnoreLabel is %d, and the curriculum requires -100", trajectory.IgnoreLabel)
	}
}

// checkRow holds what every row must be, whatever the tokenizer: the line the
// curriculum admits, the prompt's tokens then the completion's, and the end of
// turn last.
func checkRow(t *testing.T, student trajectory.Student, row trajectory.Row, report trajectory.Report) {
	t.Helper()

	line, err := json.Marshal(row)
	if err != nil {
		t.Fatal(err)
	}
	var keys map[string]json.RawMessage
	if err := json.Unmarshal(line, &keys); err != nil {
		t.Fatal(err)
	}
	if got := slices.Sorted(func(yield func(string) bool) {
		for key := range keys {
			if !yield(key) {
				return
			}
		}
	}); !slices.Equal(got, []string{"id", "input_ids", "labels", "prompt_tokens"}) {
		t.Errorf("the line has keys %v, want the curriculum's four", got)
	}
	if _, err := admitted(t, line); err != nil {
		t.Fatalf("the curriculum refuses the row: %v", err)
	}

	ctx := context.Background()
	prompt, err := student.Tokenizer.Encode(ctx, report.Text.Prompt)
	if err != nil {
		t.Fatal(err)
	}
	completion, err := student.Tokenizer.Encode(ctx, report.Text.Completion)
	if err != nil {
		t.Fatal(err)
	}
	if row.PromptTokens != len(prompt) {
		t.Errorf("prompt_tokens is %d, and the prompt alone is %d tokens", row.PromptTokens, len(prompt))
	}
	if want := append(slices.Clone(prompt), completion...); !slices.Equal(row.InputIDs, want) {
		t.Errorf("input_ids is not the prompt's tokens followed by the completion's")
	}
	for i, label := range row.Labels {
		if want := row.InputIDs[i]; i < len(prompt) && label != trajectory.IgnoreLabel || i >= len(prompt) && label != want {
			t.Fatalf("label %d at position %d; the prompt has %d tokens", label, i, len(prompt))
		}
	}
	end, err := student.Tokenizer.Encode(ctx, student.EndOfTurn)
	if err != nil || len(end) != 1 || row.InputIDs[len(row.InputIDs)-1] != end[0] {
		t.Errorf("the row ends with %d, want the end of turn %v", row.InputIDs[len(row.InputIDs)-1], end)
	}
	joint, err := student.Tokenizer.Encode(ctx, report.Text.Prompt+report.Text.Completion)
	if err != nil {
		t.Fatal(err)
	}
	if report.JointAgrees != slices.Equal(joint, row.InputIDs) {
		t.Errorf("the report says the joint encoding agrees: %t, and it does: %t", report.JointAgrees, !report.JointAgrees)
	}
}

func TestTheRowIsWhatTheCurriculumAdmits(t *testing.T) {
	student := trajectory.Student{
		ChatTemplate: ornithTemplate(t),
		EndOfTurn:    ornithEndOfTurn,
		Tokenizer:    byteLevel(t, ornithSpecials),
		Vocabulary:   ornithEmbeddingRows,
	}
	examples := []trajectory.Example{arithmetic}
	for _, name := range finishedOutputs {
		examples = append(examples, recordedTrajectory(t, name))
	}
	for _, example := range examples {
		row, report, err := trajectory.Build(context.Background(), student, example)
		if err != nil {
			t.Fatalf("%s: %v", example.ID, err)
		}
		if row.ID != example.ID {
			t.Errorf("the row is %q, want %q", row.ID, example.ID)
		}
		checkRow(t, student, row, report)
		// The Ornith's prompt ends with a newline and its completion starts
		// with a letter, a boundary no piece crosses.
		if !report.JointAgrees {
			t.Errorf("%s: the joint encoding disagrees at a boundary no piece crosses", example.ID)
		}
		if got := row.InputIDs[row.PromptTokens-2]; got != ornithSpecials["<think>"] {
			t.Errorf("%s: the prompt's next-to-last token is %d, want <think>", example.ID, got)
		}
		t.Logf("%s: %d tokens, %d of them prompt", example.ID, len(row.InputIDs), row.PromptTokens)
	}
}

func TestAMergeAcrossTheBoundaryIsReportedNotHidden(t *testing.T) {
	// A template that opens the answer mid-word: the prompt ends with "re" and
	// the reasoning continues it. Tokenized whole, "reason" is one token;
	// tokenized as the student meets it, "re" and then "ason".
	const template = "{% for m in messages %}{% if m.role == 'user' %}<|im_start|>user\n{{ m.content }}<|im_end|>\n" +
		"{% else %}<|im_start|>assistant\nre{{ m.reasoning_content }} so {{ m.content }}<|im_end|>\n{% endif %}{% endfor %}" +
		"{% if add_generation_prompt %}<|im_start|>assistant\nre{% endif %}"
	student := trajectory.Student{
		ChatTemplate: template,
		EndOfTurn:    ornithEndOfTurn,
		Tokenizer:    byteLevel(t, ornithSpecials, "reason"),
		Vocabulary:   ornithEmbeddingRows,
	}
	example := trajectory.Example{ID: "boundary", Messages: arithmetic.Messages, Reasoning: "ason", Answer: "5"}
	row, report, err := trajectory.Build(context.Background(), student, example)
	if err != nil {
		t.Fatal(err)
	}
	checkRow(t, student, row, report)
	if report.JointAgrees {
		t.Fatalf("a merge across the boundary was reported as agreeing: %q + %q", report.Text.Prompt, report.Text.Completion)
	}
}

func TestTheEndOfTurnMustBeOneTokenAndEveryIDInTheVocabulary(t *testing.T) {
	template := ornithTemplate(t)
	ctx := context.Background()

	// Without <|im_end|> as an added token the tokenizer spells it in bytes:
	// the student would learn to write the stop rather than to stop.
	spelled := map[string]int64{"<|im_start|>": 248045, "<think>": 248068, "</think>": 248069}
	student := trajectory.Student{ChatTemplate: template, EndOfTurn: ornithEndOfTurn, Tokenizer: byteLevel(t, spelled), Vocabulary: ornithEmbeddingRows}
	if row, _, err := trajectory.Build(ctx, student, arithmetic); err == nil || !strings.Contains(err.Error(), "tokens, want the one token") {
		t.Errorf("an end of turn spelled in bytes gave row %v and %v, want a refusal", row.InputIDs, err)
	}

	// The highest id a row holds is </think>'s, 248069, so 248070 rows is the
	// smallest vocabulary that has a row for every token.
	student.Tokenizer = byteLevel(t, ornithSpecials)
	for _, rows := range []int{248046, 248069} {
		student.Vocabulary = rows
		if row, _, err := trajectory.Build(ctx, student, arithmetic); err == nil || !strings.Contains(err.Error(), "outside the student's vocabulary") {
			t.Errorf("a vocabulary of %d rows gave row %v and %v, want a refusal", rows, row.InputIDs, err)
		}
	}
	student.Vocabulary = 248070
	if _, _, err := trajectory.Build(ctx, student, arithmetic); err != nil {
		t.Errorf("every id inside the vocabulary was refused: %v", err)
	}

	for name, broken := range map[string]trajectory.Student{
		"no tokenizer":     {ChatTemplate: template, EndOfTurn: ornithEndOfTurn, Vocabulary: ornithEmbeddingRows},
		"no vocabulary":    {ChatTemplate: template, EndOfTurn: ornithEndOfTurn, Tokenizer: student.Tokenizer},
		"no end of turn":   {ChatTemplate: template, Tokenizer: student.Tokenizer, Vocabulary: ornithEmbeddingRows},
		"no chat template": {EndOfTurn: ornithEndOfTurn, Tokenizer: student.Tokenizer, Vocabulary: ornithEmbeddingRows},
	} {
		if row, _, err := trajectory.Build(ctx, broken, arithmetic); err == nil {
			t.Errorf("a student with %s gave row %v, want an error", name, row.InputIDs)
		}
	}
}

func TestAnAnswerOnlyRowIsWhatTheCurriculumAdmits(t *testing.T) {
	ctx := context.Background()
	student := trajectory.Student{
		ChatTemplate: ornithTemplate(t),
		EndOfTurn:    ornithEndOfTurn,
		Tokenizer:    byteLevel(t, ornithSpecials),
		Vocabulary:   ornithEmbeddingRows,
	}
	full := recordedTrajectory(t, "qwen3.8-27b.gsm8k-train-1468")
	withReasoning, _, err := trajectory.Build(ctx, student, full)
	if err != nil {
		t.Fatal(err)
	}
	row, report, err := trajectory.Build(ctx, student, answerOnly(full, gsm8k1468Answer))
	if err != nil {
		t.Fatal(err)
	}
	checkRow(t, student, row, report)

	// The prompt's tokens are the trajectory row's, and every one of them is
	// masked; only the completion is trained.
	if row.PromptTokens != withReasoning.PromptTokens || !slices.Equal(row.InputIDs[:row.PromptTokens], withReasoning.InputIDs[:withReasoning.PromptTokens]) {
		t.Errorf("the answer-only row's prompt is %d tokens, the trajectory row's %d, and they differ", row.PromptTokens, withReasoning.PromptTokens)
	}
	want := []int64{'\n', ornithSpecials["</think>"], '\n', '\n'}
	for _, b := range []byte(gsm8k1468Answer) {
		want = append(want, int64(b))
	}
	want = append(want, ornithSpecials["<|im_end|>"])
	if completion := row.InputIDs[row.PromptTokens:]; !slices.Equal(completion, want) {
		t.Errorf("the answer-only completion is %v, want %v", completion, want)
	}
	if !report.JointAgrees {
		t.Errorf("a tokenizer with no piece longer than a byte disagreed with itself at the boundary")
	}

	// With a piece for the blank line, which the Ornith's tokenizer has, the
	// prompt's last newline and the completion's first are that piece when
	// the text is tokenized whole. The row keeps the boundary the student
	// meets, and the report says the whole text would not. The trajectory row
	// is untouched: its reasoning begins with a letter.
	student.Tokenizer = byteLevel(t, ornithSpecials, "\n\n")
	row, report, err = trajectory.Build(ctx, student, answerOnly(full, gsm8k1468Answer))
	if err != nil {
		t.Fatal(err)
	}
	checkRow(t, student, row, report)
	if report.JointAgrees {
		t.Errorf("the blank line across the answer-only boundary was reported as agreeing")
	}
	if _, fullReport, err := trajectory.Build(ctx, student, full); err != nil || !fullReport.JointAgrees {
		t.Errorf("the trajectory row disagrees at its boundary (%v): %t", err, fullReport.JointAgrees)
	}
}

// ornithTokenizerSHA256 is the Ornith's tokenizer.json at the revision its
// template comes from. The file is not downloaded or kept here:
// TRAJECTORY_ORNITH_TOKENIZER_JSON names a copy, and Load refuses any other.
const ornithTokenizerSHA256 = "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42"

// ornithTokenizer loads the copy TRAJECTORY_ORNITH_TOKENIZER_JSON names and
// holds its ids for the template's pieces to ornithSpecials. The test is
// skipped when the variable is unset.
func ornithTokenizer(t *testing.T) *tokenizer.Tokenizer {
	t.Helper()

	path := os.Getenv("TRAJECTORY_ORNITH_TOKENIZER_JSON")
	if path == "" {
		t.Skip("set TRAJECTORY_ORNITH_TOKENIZER_JSON to the Ornith-1.5-9B tokenizer.json, sha256 " + ornithTokenizerSHA256)
	}
	file, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	ctx := context.Background()
	loaded, err := tokenizer.Load(ctx, file, ornithTokenizerSHA256, tokenizer.DefaultLimits())
	if err != nil {
		t.Fatal(err)
	}
	for piece, want := range ornithSpecials {
		if ids, err := loaded.Encode(ctx, piece); err != nil || !slices.Equal(ids, []int64{want}) {
			t.Fatalf("the Ornith tokenizes %q as %v (%v), want [%d]", piece, ids, err, want)
		}
	}
	return loaded
}

func TestTheOrnithTokenizerWritesTheRowOptIn(t *testing.T) {
	loaded := ornithTokenizer(t)
	ctx := context.Background()

	newline, err := loaded.Encode(ctx, "\n")
	if err != nil || len(newline) != 1 {
		t.Fatalf("the Ornith tokenizes a newline as %v (%v), want one token", newline, err)
	}

	student := trajectory.Student{ChatTemplate: ornithTemplate(t), EndOfTurn: ornithEndOfTurn, Tokenizer: loaded, Vocabulary: ornithEmbeddingRows}
	for _, name := range finishedOutputs {
		example := recordedTrajectory(t, name)
		row, report, err := trajectory.Build(ctx, student, example)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		checkRow(t, student, row, report)

		// The prompt ends with <think> and its newline, and the student's
		// first trained token is the reasoning's; </think> is trained once,
		// and never given.
		promptTail := row.InputIDs[row.PromptTokens-2 : row.PromptTokens]
		completion := row.InputIDs[row.PromptTokens:]
		if !slices.Equal(promptTail, []int64{ornithSpecials["<think>"], newline[0]}) {
			t.Errorf("%s: the prompt ends with %v, want <think> and its newline", name, promptTail)
		}
		closing := ornithSpecials["</think>"]
		if n := slices.Index(completion, closing); n < 0 || slices.Contains(completion[n+1:], closing) || slices.Contains(row.InputIDs[:row.PromptTokens], closing) {
			t.Errorf("%s: </think> is not in the completion exactly once", name)
		}
		t.Logf("%s: %d tokens = %d prompt + %d completion; prompt ends %v, completion starts %v and ends %v; joint encoding agrees: %t",
			row.ID, len(row.InputIDs), row.PromptTokens, len(completion), promptTail, completion[:4], completion[len(completion)-4:], report.JointAgrees)
	}

	// A vocabulary that ends at <|im_end|>'s id has no row to train it with,
	// nor the thinking tags numbered after it.
	student.Vocabulary = int(ornithSpecials["<|im_end|>"])
	if _, _, err := trajectory.Build(ctx, student, arithmetic); err == nil || !strings.Contains(err.Error(), "outside the student's vocabulary") {
		t.Errorf("a vocabulary without a row for <|im_end|> gave %v, want a refusal", err)
	}
}

func TestTheOrnithTokenizerWritesTheAnswerOnlyRowOptIn(t *testing.T) {
	loaded := ornithTokenizer(t)
	ctx := context.Background()
	student := trajectory.Student{ChatTemplate: ornithTemplate(t), EndOfTurn: ornithEndOfTurn, Tokenizer: loaded, Vocabulary: ornithEmbeddingRows}
	full := recordedTrajectory(t, "qwen3.8-27b.gsm8k-train-1468")
	withReasoning, _, err := trajectory.Build(ctx, student, full)
	if err != nil {
		t.Fatal(err)
	}
	row, report, err := trajectory.Build(ctx, student, answerOnly(full, gsm8k1468Answer))
	if err != nil {
		t.Fatal(err)
	}
	checkRow(t, student, row, report)
	if row.PromptTokens != withReasoning.PromptTokens || !slices.Equal(row.InputIDs[:row.PromptTokens], withReasoning.InputIDs[:withReasoning.PromptTokens]) {
		t.Errorf("the answer-only row's prompt is %d tokens, the trajectory row's %d, and they differ", row.PromptTokens, withReasoning.PromptTokens)
	}

	// The completion as the Ornith's tokenizer writes it: the newline 198,
	// </think>, the blank line 271, "####" 794, the space 220, the digits of
	// 230 one token each, and <|im_end|>.
	const blankLine = 271
	completion := row.InputIDs[row.PromptTokens:]
	want := []int64{198, ornithSpecials["</think>"], blankLine, 794, 220, 17, 18, 15, ornithSpecials["<|im_end|>"]}
	if !slices.Equal(completion, want) {
		t.Errorf("the answer-only completion is %v, want %v", completion, want)
	}
	if last := completion[len(completion)-1]; last != 248046 {
		t.Errorf("the answer-only completion ends with %d, want <|im_end|> 248046", last)
	}
	if ids, err := loaded.Encode(ctx, "\n\n"); err != nil || !slices.Equal(ids, []int64{blankLine}) {
		t.Fatalf("the Ornith tokenizes a blank line as %v (%v), want [%d]", ids, err, blankLine)
	}

	// Tokenized whole, "<think>\n" and "\n</think>" meet in the blank line:
	// the joint encoding is the row with the prompt's last newline and the
	// completion's first replaced by 271, and nowhere else different.
	joint, err := loaded.Encode(ctx, report.Text.Prompt+report.Text.Completion)
	if err != nil {
		t.Fatal(err)
	}
	merged := append(slices.Clone(row.InputIDs[:row.PromptTokens-1]), blankLine)
	merged = append(merged, completion[1:]...)
	if report.JointAgrees || !slices.Equal(joint, merged) {
		t.Errorf("the joint encoding agrees: %t; it is the row with the two newlines merged: %t", report.JointAgrees, slices.Equal(joint, merged))
	}
	t.Logf("%s answer-only: %d tokens = %d prompt + %d completion; prompt ends %v; completion %v; joint encoding agrees: %t, joint at the boundary %v",
		row.ID, len(row.InputIDs), row.PromptTokens, len(completion), row.InputIDs[row.PromptTokens-2:row.PromptTokens], completion,
		report.JointAgrees, joint[row.PromptTokens-2:row.PromptTokens+1])
}
