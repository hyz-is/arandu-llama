package chat_test

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	llama "github.com/tayi-ai/arandu-llama"
)

// What a chat prompt is made of, before a model is involved.
//
// Chat sends a model its own GGUF template rendered by llama.cpp's Jinja
// engine. The three templates under testdata/chat_templates are copied from the
// GGUFs a teacher run loads, so what these tests assert is what those models
// are sent, short of the BOS their tokenizer adds. The legacy formatter this
// replaced guessed a format from the template's text and wrote its own
// rendering: an error on every Gemma 4 request, a gpt-oss prompt without the
// harmony system preamble, a Qwen3.8 prompt without its thinking tag.
//
// None of these run in parallel: a failed render writes the wrapper's one
// error slot, a C++ string with no lock around it.

func chatTemplate(t *testing.T, name string) string {
	t.Helper()

	body, err := os.ReadFile(filepath.Join(packageRoot(t), "testdata", "chat_templates", name+".jinja"))
	if err != nil {
		t.Fatalf("reading the %s template: %v", name, err)
	}
	return string(body)
}

func packageRoot(t *testing.T) string {
	t.Helper()

	root, err := filepath.Abs(filepath.Join("..", "..", ".."))
	if err != nil {
		t.Fatalf("resolving the package root: %v", err)
	}
	return root
}

func render(t *testing.T, name string, messages []llama.ChatMessage, opts llama.ChatOptions) string {
	t.Helper()

	opts.ChatTemplate = chatTemplate(t, name)
	prompt, err := llama.RenderChatTemplate(messages, opts)
	if err != nil {
		t.Fatalf("rendering the %s template: %v", name, err)
	}
	return prompt
}

var hello = []llama.ChatMessage{{Role: "user", Content: "Hello"}}

func TestTheGemma4TemplateRenders(t *testing.T) {
	// The legacy formatter answered every Gemma 4 request with "Failed to apply
	// chat template (template detection or application error)": it knew no
	// format containing <|turn>, so there was no prompt to generate from.
	if got, want := render(t, "gemma-4-12b-it", hello, llama.ChatOptions{EnableThinking: llama.Bool(false)}),
		"<|turn>user\nHello<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"; got != want {
		t.Fatalf("Gemma 4 without thinking rendered\n%q\nwant\n%q", got, want)
	}

	// Left unset, thinking follows llama-server: on, because llama.cpp reports
	// that this template supports it. The template alone would default it off,
	// which is why the choice is written down rather than inherited.
	got := render(t, "gemma-4-12b-it", hello, llama.ChatOptions{})
	if !strings.HasPrefix(got, "<|turn>system\n<|think|>\n") || !strings.HasSuffix(got, "<|turn>user\nHello<turn|>\n<|turn>model\n") {
		t.Fatalf("Gemma 4 with the default thinking rendered %q, want the <|think|> system turn and an open model turn", got)
	}
}

func TestTheGptOssTemplateRendersItsSystemPreamble(t *testing.T) {
	// The legacy formatter rendered "<|start|>user<|message|>Hello<|end|>
	// <|start|>assistant" for this template: harmony's turn markers without the
	// system message the template writes first. The date line is the render
	// day, so the assertion stops short of it.
	got := render(t, "gpt-oss-20b", hello, llama.ChatOptions{})
	for _, want := range []string{
		"<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.\nKnowledge cutoff: 2024-06\nCurrent date: ",
		"\n\nReasoning: medium\n\n# Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|>",
	} {
		if !strings.Contains(got, want) {
			t.Fatalf("gpt-oss rendered %q, which lacks %q", got, want)
		}
	}
	if !strings.HasPrefix(got, "<|start|>system<|message|>") || !strings.HasSuffix(got, "<|start|>user<|message|>Hello<|end|><|start|>assistant") {
		t.Fatalf("gpt-oss rendered %q, want the system turn first and an open assistant turn last", got)
	}
}

func TestTheQwenTemplateRendersItsThinkingTag(t *testing.T) {
	// The legacy formatter read <|im_start|> as ChatML and stopped at
	// "<|im_start|>assistant\n"; the template opens the thinking block there.
	if got, want := render(t, "qwen3.8-27b", hello, llama.ChatOptions{EnableThinking: llama.Bool(false)}),
		"<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"; got != want {
		t.Fatalf("Qwen3.8 without thinking rendered\n%q\nwant\n%q", got, want)
	}

	got := render(t, "qwen3.8-27b", hello, llama.ChatOptions{})
	if !strings.HasPrefix(got, "<|im_start|>system\nReasoning effort is set to xhigh.") || !strings.HasSuffix(got, "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n<think>\n") {
		t.Fatalf("Qwen3.8 with the default thinking rendered %q, want the reasoning-effort system turn and an open <think>", got)
	}
}

func TestAConversationEndsWithTheGenerationPrompt(t *testing.T) {
	// An earlier assistant turn is history, and the prompt has to end where the
	// next answer begins: after the last user turn is closed and the
	// assistant's is opened. A prompt ending anywhere else scores or generates
	// from the wrong position without failing.
	conversation := []llama.ChatMessage{
		{Role: "user", Content: "What is the capital of France?"},
		{Role: "assistant", Content: "Paris."},
		{Role: "user", Content: "And of Italy?"},
	}
	for _, family := range []struct {
		template string
		ending   string
	}{
		{"gemma-4-12b-it", "And of Italy?<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"},
		{"gpt-oss-20b", "And of Italy?<|end|><|start|>assistant"},
		{"qwen3.8-27b", "And of Italy?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"},
	} {
		got := render(t, family.template, conversation, llama.ChatOptions{EnableThinking: llama.Bool(false)})
		if !strings.HasSuffix(got, family.ending) {
			t.Errorf("%s rendered %q, want it to end with %q", family.template, got, family.ending)
			continue
		}
		previous := -1
		for _, message := range conversation {
			at := strings.Index(got, message.Content)
			if at <= previous {
				t.Errorf("%s rendered %q with %q out of order", family.template, got, message.Content)
				break
			}
			previous = at
		}
	}
}

func TestChatTemplateKwargsReachTheTemplate(t *testing.T) {
	if got := render(t, "gpt-oss-20b", hello, llama.ChatOptions{ChatTemplateKwargs: map[string]interface{}{"reasoning_effort": "low"}}); !strings.Contains(got, "\n\nReasoning: low\n\n") {
		t.Fatalf("gpt-oss with reasoning_effort low rendered %q", got)
	}

	// enable_thinking as a kwarg sets what EnableThinking sets, as it does in
	// llama-server, so both spellings render the same prompt.
	byOption := render(t, "qwen3.8-27b", hello, llama.ChatOptions{EnableThinking: llama.Bool(false)})
	byKwarg := render(t, "qwen3.8-27b", hello, llama.ChatOptions{ChatTemplateKwargs: map[string]interface{}{"enable_thinking": false}})
	if byOption != byKwarg {
		t.Fatalf("enable_thinking as a kwarg rendered\n%q\nand as EnableThinking\n%q", byKwarg, byOption)
	}

	qwen := chatTemplate(t, "qwen3.8-27b")
	for name, opts := range map[string]llama.ChatOptions{
		"contradicting EnableThinking": {ChatTemplate: qwen, EnableThinking: llama.Bool(true), ChatTemplateKwargs: map[string]interface{}{"enable_thinking": false}},
		"not a bool":                   {ChatTemplate: qwen, ChatTemplateKwargs: map[string]interface{}{"enable_thinking": "false"}},
	} {
		if prompt, err := llama.RenderChatTemplate(hello, opts); err == nil {
			t.Errorf("an enable_thinking kwarg %s rendered %q, want an error", name, prompt)
		}
	}
}

func TestATemplateThatFailsIsAnErrorAndNotAnotherPrompt(t *testing.T) {
	// The first template is ChatML with its closing endif missing. The legacy
	// formatter would have found <|im_start|> in it and rendered its own ChatML;
	// any prompt coming back here would be one the template did not write.
	chatML := "{% for message in messages %}<|im_start|>{{ message.role }}\n{{ message.content }}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n"
	for _, failure := range []struct {
		template string
		reason   string
	}{
		{chatML, "Failed to parse chat template"},
		{"{{ messages[0].content }", "Failed to parse chat template"},
		{"{{ raise_exception('refused by the template') }}", "refused by the template"},
	} {
		prompt, err := llama.RenderChatTemplate(hello, llama.ChatOptions{ChatTemplate: failure.template})
		if err == nil {
			t.Errorf("template %q rendered %q, want an error", failure.template, prompt)
			continue
		}
		if prompt != "" {
			t.Errorf("template %q failed with %v and still returned %q", failure.template, err, prompt)
		}
		if !strings.Contains(err.Error(), failure.reason) {
			t.Errorf("template %q failed with %q, want it to say %q", failure.template, err, failure.reason)
		}
	}
}

func TestAFormatNameIsNotATemplate(t *testing.T) {
	// A name selected a built-in format of the legacy formatter. To the Jinja
	// engine it is a template without tags, and its rendering would be the name.
	for _, name := range []string{"chatml", "llama3", "gemma"} {
		if prompt, err := llama.RenderChatTemplate(hello, llama.ChatOptions{ChatTemplate: name}); err == nil {
			t.Errorf("format name %q rendered %q, want an error", name, prompt)
		}
	}
}

func TestRenderingNeedsATemplateAndMessages(t *testing.T) {
	if prompt, err := llama.RenderChatTemplate(hello, llama.ChatOptions{}); err == nil {
		t.Errorf("no template rendered %q, want an error", prompt)
	}
	if prompt, err := llama.RenderChatTemplate(nil, llama.ChatOptions{ChatTemplate: chatTemplate(t, "qwen3.8-27b")}); err == nil {
		t.Errorf("no messages rendered %q, want an error", prompt)
	}
}

func TestChatTemplateEngineNamesThePinnedEngine(t *testing.T) {
	// The Jinja engine is llama.cpp's, so moving the pin can move every prompt
	// while each template keeps its digest. The gitlink is the pin; the
	// constant has to follow it.
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git is not on PATH: the gitlink cannot be read")
	}
	command := exec.Command("git", "ls-tree", "HEAD", "llama.cpp")
	command.Dir = packageRoot(t)
	output, err := command.Output()
	if err != nil {
		t.Skipf("not a git checkout: %v", err)
	}
	// "160000 commit <sha>\tllama.cpp"
	fields := strings.Fields(string(output))
	if len(fields) < 3 || fields[1] != "commit" {
		t.Skipf("llama.cpp is not a gitlink at HEAD: %q", strings.TrimSpace(string(output)))
	}
	if len(fields[2]) < 8 {
		t.Fatalf("gitlink %q is not a commit id", fields[2])
	}
	want := "llama.cpp." + fields[2][:8]
	if !strings.HasSuffix(llama.ChatTemplateEngine, want) {
		t.Fatalf("ChatTemplateEngine %q does not end with %q: the submodule moved and the constant did not, so a prompt rendered by this engine is labelled as another's",
			llama.ChatTemplateEngine, want)
	}
}
