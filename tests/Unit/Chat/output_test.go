package chat_test

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	llama "github.com/tayi-ai/arandu-llama"
)

// How a chat output is read back, before a model is involved.
//
// The outputs under testdata/chat_outputs are what three teachers generated in
// a smoke run with ReasoningFormatNone, so each is the model's text with its
// special tokens written out. At the pinned llama.cpp every parse is a PEG
// parse, and the wrapper used to call it with no parser: the output came back
// whole as Content, and every teacher's reasoning was scored as its answer.
// ParseChatOutput reads an output the way Chat does, with the parser llama.cpp
// builds beside the prompt, so these tests hold what Chat would return for
// those outputs.
//
// Like the render tests, none of these run in parallel: a failed parse writes
// the wrapper's one error slot.

// chatOutput is one recorded generation; testdata/chat_outputs/README.md says
// where each came from.
type chatOutput struct {
	Template string `json:"template"`
	ID       string `json:"id"`
	User     string `json:"user"`
	Finish   string `json:"finish"`
	Output   string `json:"output"`
}

func recordedOutput(t *testing.T, name string) chatOutput {
	t.Helper()

	body, err := os.ReadFile(filepath.Join(packageRoot(t), "testdata", "chat_outputs", name+".json"))
	if err != nil {
		t.Fatalf("reading the %s output: %v", name, err)
	}
	var recorded chatOutput
	if err := json.Unmarshal(body, &recorded); err != nil {
		t.Fatalf("decoding the %s output: %v", name, err)
	}
	return recorded
}

// messages is the conversation the smoke run sent: one user turn.
func (o chatOutput) messages() []llama.ChatMessage {
	return []llama.ChatMessage{{Role: "user", Content: o.User}}
}

// parse reads the output as Chat would have with format, and checks what
// every response must carry whatever the parser made of it.
func parse(t *testing.T, o chatOutput, format llama.ReasoningFormat) *llama.ChatResponse {
	t.Helper()

	opts := llama.ChatOptions{ChatTemplate: chatTemplate(t, o.Template), ReasoningFormat: format}
	finish := llama.FinishReason(o.Finish)
	response, err := llama.ParseChatOutput(o.messages(), opts, o.Output, finish)
	if err != nil {
		t.Fatalf("parsing the %s output for %s with format %s: %v", o.Template, o.ID, format, err)
	}
	if response.Output != o.Output {
		t.Errorf("%s: Output is not the text that was parsed", o.Template)
	}
	if response.FinishReason != finish {
		t.Errorf("%s: FinishReason = %q, want %q", o.Template, response.FinishReason, finish)
	}
	if response.GeneratedTokens != 0 {
		t.Errorf("%s: GeneratedTokens = %d without a generation, want 0", o.Template, response.GeneratedTokens)
	}
	return response
}

// requireNoMarkup fails when a template's control tokens survived the parse
// into text a scorer reads.
func requireNoMarkup(t *testing.T, where, text string, markup ...string) {
	t.Helper()

	for _, token := range markup {
		if strings.Contains(text, token) {
			t.Errorf("%s holds %q: %q", where, token, text)
		}
	}
}

var (
	harmonyMarkup = []string{"<|channel|>", "<|message|>", "<|start|>", "<|end|>", "assistantfinal"}
	qwenMarkup    = []string{"<think>", "</think>", "<|im_start|>", "<|im_end|>"}
	gemmaMarkup   = []string{"<|channel>", "<channel|>", "<|turn>", "<turn|>"}
)

func TestGptOssAnalysisIsReasoningAndFinalIsContent(t *testing.T) {
	// gpt-oss writes harmony channels: its thinking on analysis, its answer on
	// final. The whole output used to be Content, so a verifier reading the
	// last line found "#### 230" -- and a verifier reading anything else found
	// the analysis first.
	o := recordedOutput(t, "gpt-oss-20b.gsm8k-train-1468")
	response := parse(t, o, llama.ReasoningFormatAuto)

	if !strings.HasPrefix(response.ReasoningContent, "We need to parse the problem: Troy's home is 75 meters away from school.") ||
		!strings.HasSuffix(response.ReasoningContent, "Provide final numeric answer as '#### 230'.") {
		t.Errorf("gpt-oss reasoning is %q, want the analysis channel", response.ReasoningContent)
	}
	if !strings.HasPrefix(response.Content, "Troy walks \\(75 \\text{ m}\\) each way") || !strings.HasSuffix(response.Content, "\n\n#### 230") {
		t.Errorf("gpt-oss content is %q, want the final channel", response.Content)
	}
	if strings.Contains(response.Content, "We need to parse") {
		t.Errorf("gpt-oss content holds the analysis: %q", response.Content)
	}
	requireNoMarkup(t, "gpt-oss content", response.Content, harmonyMarkup...)
	requireNoMarkup(t, "gpt-oss reasoning", response.ReasoningContent, harmonyMarkup[:4]...)
}

func TestQwenThinkingIsReasoning(t *testing.T) {
	// The prompt opens <think>, so the output starts inside the thinking block
	// and its first tag is the closing one. Without the generation prompt in
	// front of it the parser has no opening tag to match.
	o := recordedOutput(t, "qwen3.8-27b.gsm8k-train-1468")
	response := parse(t, o, llama.ReasoningFormatAuto)

	if !strings.HasPrefix(response.ReasoningContent, "We need answer math problem.") ||
		!strings.HasSuffix(response.ReasoningContent, "Ensure final line exact.\n") {
		t.Errorf("Qwen3.8 reasoning is %q, want the thinking block", response.ReasoningContent)
	}
	if !strings.HasPrefix(response.Content, "Emily walks 98 meters to school") || !strings.HasSuffix(response.Content, "\n\n#### 230") {
		t.Errorf("Qwen3.8 content is %q, want the answer after </think>", response.Content)
	}
	requireNoMarkup(t, "Qwen3.8 content", response.Content, qwenMarkup...)
	requireNoMarkup(t, "Qwen3.8 reasoning", response.ReasoningContent, qwenMarkup...)
}

func TestGemmaThoughtChannelIsReasoning(t *testing.T) {
	// 629 tokens as the smoke run counted them, by tokenizing the text again:
	// the one Gemma 4 answer that ended before the ceiling.
	o := recordedOutput(t, "gemma-4-12b-it.gsm8k-train-6546")
	response := parse(t, o, llama.ReasoningFormatAuto)

	if !strings.HasPrefix(response.ReasoningContent, "*   Total people invited: 30\n") ||
		!strings.HasSuffix(response.ReasoningContent, "Number of people who ordered chicken = 6.") {
		t.Errorf("Gemma 4 reasoning is %q, want the thought channel", response.ReasoningContent)
	}
	if !strings.HasPrefix(response.Content, "To find the number of people who ordered chicken") || !strings.HasSuffix(response.Content, "\n\n#### 6") {
		t.Errorf("Gemma 4 content is %q, want the answer after <channel|>", response.Content)
	}
	requireNoMarkup(t, "Gemma 4 content", response.Content, gemmaMarkup...)
	requireNoMarkup(t, "Gemma 4 reasoning", response.ReasoningContent, gemmaMarkup...)
}

func TestGemmaCutAtTheCeilingHasNoAnswer(t *testing.T) {
	// The smoke run stopped this one at MaxTokens, 1024, inside the thought
	// channel. Scored as Content it was the model's last calculation, "5",
	// against a reference of 230; there is no answer in it, and the response
	// says so with an empty Content.
	o := recordedOutput(t, "gemma-4-12b-it.gsm8k-train-1468")
	if o.Finish != string(llama.FinishReasonLength) {
		t.Fatalf("the fixture records finish %q, want %q", o.Finish, llama.FinishReasonLength)
	}
	response := parse(t, o, llama.ReasoningFormatAuto)

	if response.Content != "" {
		t.Errorf("Gemma 4 cut in its thought has content %q, want none", response.Content)
	}
	const opening = "<|channel>thought\n"
	if !strings.HasPrefix(o.Output, opening) || response.ReasoningContent != strings.TrimPrefix(o.Output, opening) {
		t.Errorf("Gemma 4 cut in its thought has reasoning %q, want everything after %q", response.ReasoningContent, opening)
	}
}

func TestReasoningFormatNoneLeavesTheOutputAsContent(t *testing.T) {
	// None is the smoke run's own setting, and it has to keep meaning what it
	// meant there. The parser llama.cpp builds for None is not a no-op -- for
	// Qwen3.8 it would put the prompt's "<think>\n" in front of the content --
	// so None does not parse at all.
	for _, name := range []string{
		"gpt-oss-20b.gsm8k-train-1468",
		"qwen3.8-27b.gsm8k-train-1468",
		"gemma-4-12b-it.gsm8k-train-6546",
		"gemma-4-12b-it.gsm8k-train-1468",
	} {
		o := recordedOutput(t, name)
		response := parse(t, o, llama.ReasoningFormatNone)
		if response.Content != o.Output || response.ReasoningContent != "" {
			t.Errorf("%s with ReasoningFormatNone gave content %q and reasoning %q, want the output as content",
				name, response.Content, response.ReasoningContent)
		}
	}
}

func TestAnOutputTheParserRejectsIsAnError(t *testing.T) {
	// gpt-oss writes nothing outside a channel. An output that did, parsed as
	// complete, is an error: returning it as Content is how the reasoning got
	// scored before.
	opts := llama.ChatOptions{ChatTemplate: chatTemplate(t, "gpt-oss-20b"), ReasoningFormat: llama.ReasoningFormatAuto}
	const stray = "#### 230"
	response, err := llama.ParseChatOutput(hello, opts, stray, llama.FinishReasonStop)
	if err == nil {
		t.Fatalf("gpt-oss output %q outside any channel parsed to content %q and reasoning %q, want an error",
			stray, response.Content, response.ReasoningContent)
	}
	if response != nil {
		t.Errorf("a failed parse still returned %+v", response)
	}
	if !strings.Contains(err.Error(), "does not match the expected") {
		t.Errorf("the parse failed with %q, want llama.cpp's format mismatch", err)
	}

	// Cut at the ceiling the same text is a partial parse, which llama.cpp
	// accepts with nothing extracted. What matters is that it is not the raw
	// output either.
	response, err = llama.ParseChatOutput(hello, opts, stray, llama.FinishReasonLength)
	if err != nil {
		t.Fatalf("a partial parse of %q failed: %v", stray, err)
	}
	if response.Content == stray {
		t.Errorf("a partial parse returned the unparsed output %q as content", stray)
	}
}

func TestParseChatOutputRefusesWhatItCannotRender(t *testing.T) {
	qwen := chatTemplate(t, "qwen3.8-27b")
	for name, call := range map[string]struct {
		opts   llama.ChatOptions
		finish llama.FinishReason
	}{
		"no template":              {llama.ChatOptions{ReasoningFormat: llama.ReasoningFormatAuto}, llama.FinishReasonStop},
		"a format name":            {llama.ChatOptions{ChatTemplate: "chatml", ReasoningFormat: llama.ReasoningFormatAuto}, llama.FinishReasonStop},
		"no finish reason":         {llama.ChatOptions{ChatTemplate: qwen, ReasoningFormat: llama.ReasoningFormatAuto}, ""},
		"an unknown finish reason": {llama.ChatOptions{ChatTemplate: qwen, ReasoningFormat: llama.ReasoningFormatAuto}, "eos"},
		"an unknown format":        {llama.ChatOptions{ChatTemplate: qwen, ReasoningFormat: llama.ReasoningFormat(9)}, llama.FinishReasonStop},
	} {
		if response, err := llama.ParseChatOutput(hello, call.opts, "done", call.finish); err == nil {
			t.Errorf("ParseChatOutput with %s returned %+v, want an error", name, response)
		}
	}
}

func TestTheReasoningFormatDoesNotChangeThePrompt(t *testing.T) {
	// The format now reaches llama.cpp's render, where it decides the parser.
	// Measured for these templates, it does not decide the prompt: a teacher
	// sent the same text under None and under Auto generates the same way, so
	// outputs recorded under None can be read back under Auto.
	conversations := map[string][]llama.ChatMessage{
		"hello":   hello,
		"smoke":   recordedOutput(t, "gpt-oss-20b.gsm8k-train-1468").messages(),
		"history": {{Role: "user", Content: "What is 2+2?"}, {Role: "assistant", Content: "4"}, {Role: "user", Content: "And 3+3?"}},
	}
	for _, template := range []string{"gemma-4-12b-it", "gpt-oss-20b", "qwen3.8-27b"} {
		for conversation, messages := range conversations {
			for _, thinking := range []*bool{nil, llama.Bool(true), llama.Bool(false)} {
				none := render(t, template, messages, llama.ChatOptions{EnableThinking: thinking, ReasoningFormat: llama.ReasoningFormatNone})
				for _, format := range []llama.ReasoningFormat{llama.ReasoningFormatAuto, llama.ReasoningFormatDeepSeekLegacy, llama.ReasoningFormatDeepSeek} {
					got := render(t, template, messages, llama.ChatOptions{EnableThinking: thinking, ReasoningFormat: format})
					if got != none {
						t.Errorf("%s, %s conversation, thinking %v: %s rendered\n%q\nand none\n%q",
							template, conversation, describe(thinking), format, got, none)
					}
				}
			}
		}
	}
}

func describe(b *bool) string {
	if b == nil {
		return "default"
	}
	if *b {
		return "on"
	}
	return "off"
}
