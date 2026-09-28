package chat_test

import (
	"strings"
	"testing"

	llama "github.com/tayi-ai/arandu-llama"
)

// A finished conversation, rendered the way a training row needs it.
//
// A prompt ends where the next answer begins. A row for supervised training
// needs the other render as well: the conversation with the assistant's turn
// written out -- thinking included -- and nothing opened after it. The
// student's template is what decides how that turn looks, so these tests read
// it rather than assume ChatML.

var ornithTurn = []llama.ChatMessage{
	{Role: "user", Content: "What is 2+3?"},
	{Role: "assistant", Content: "#### 5", ReasoningContent: "Two plus three is five."},
}

func TestTheOrnithTemplateOpensItsThinkingBlock(t *testing.T) {
	// The student's prompt ends inside <think>, so whatever the student writes
	// first is thinking, and a completion trained against this prompt starts
	// with the reasoning rather than with the tag.
	if got, want := render(t, "ornith-1.5-9b", hello, llama.ChatOptions{EnableThinking: llama.Bool(true)}),
		"<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n<think>\n"; got != want {
		t.Fatalf("Ornith with thinking rendered\n%q\nwant\n%q", got, want)
	}
	if got, want := render(t, "ornith-1.5-9b", hello, llama.ChatOptions{EnableThinking: llama.Bool(false)}),
		"<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"; got != want {
		t.Fatalf("Ornith without thinking rendered\n%q\nwant\n%q", got, want)
	}
}

func TestReasoningContentReachesTheAssistantTurn(t *testing.T) {
	// Before ReasoningContent the wrapper handed llama.cpp the content alone,
	// and the Ornith template wrote an empty <think></think> block for every
	// assistant turn: a teacher's reasoning had no way into the student's
	// render.
	got := render(t, "ornith-1.5-9b", ornithTurn, llama.ChatOptions{EnableThinking: llama.Bool(true), AddGenerationPrompt: llama.Bool(false)})
	want := "<|im_start|>user\nWhat is 2+3?<|im_end|>\n" +
		"<|im_start|>assistant\n<think>\nTwo plus three is five.\n</think>\n\n#### 5<|im_end|>\n"
	if got != want {
		t.Fatalf("Ornith with the assistant's reasoning rendered\n%q\nwant\n%q", got, want)
	}

	// Without the field the same turn is what it always was: an empty block.
	without := []llama.ChatMessage{ornithTurn[0], {Role: "assistant", Content: ornithTurn[1].Content}}
	got = render(t, "ornith-1.5-9b", without, llama.ChatOptions{EnableThinking: llama.Bool(true), AddGenerationPrompt: llama.Bool(false)})
	if want := "<|im_start|>assistant\n<think>\n\n</think>\n\n#### 5<|im_end|>\n"; !strings.HasSuffix(got, want) {
		t.Fatalf("Ornith without reasoning rendered\n%q\nwant it to end with\n%q", got, want)
	}
}

func TestWithoutTheGenerationPromptTheRenderEndsWithTheLastMessage(t *testing.T) {
	// nil is the default and must stay the prompt Chat generates from; false
	// is the one value that changes anything.
	for _, family := range []struct {
		template string
		history  string
	}{
		{"ornith-1.5-9b", "<|im_start|>user\nHello<|im_end|>\n"},
		{"qwen3.8-27b", "<|im_end|>\n<|im_start|>user\nHello<|im_end|>\n"},
	} {
		unset := render(t, family.template, hello, llama.ChatOptions{EnableThinking: llama.Bool(true)})
		set := render(t, family.template, hello, llama.ChatOptions{EnableThinking: llama.Bool(true), AddGenerationPrompt: llama.Bool(true)})
		if unset != set {
			t.Errorf("%s: AddGenerationPrompt nil rendered\n%q\nand true\n%q", family.template, unset, set)
		}
		history := render(t, family.template, hello, llama.ChatOptions{EnableThinking: llama.Bool(true), AddGenerationPrompt: llama.Bool(false)})
		if !strings.HasSuffix(history, family.history) {
			t.Errorf("%s without the generation prompt rendered %q, want it to end with %q", family.template, history, family.history)
		}
		if !strings.HasPrefix(unset, history) || len(unset) == len(history) {
			t.Errorf("%s: the prompt %q does not extend the history %q", family.template, unset, history)
		}
	}
}

func TestParseChatOutputRefusesARenderWithoutTheGenerationPrompt(t *testing.T) {
	// Output is read from where the generation prompt ends. Chat and
	// ChatStream refuse the same option before they render; they need a model
	// to reach, so this is the refusal a test can hold without one.
	opts := llama.ChatOptions{
		ChatTemplate:        chatTemplate(t, "ornith-1.5-9b"),
		ReasoningFormat:     llama.ReasoningFormatAuto,
		AddGenerationPrompt: llama.Bool(false),
	}
	response, err := llama.ParseChatOutput(hello, opts, "Hi.\n</think>\n\nHello.", llama.FinishReasonStop)
	if err == nil {
		t.Fatalf("ParseChatOutput without the generation prompt returned %+v, want an error", response)
	}
	if !strings.Contains(err.Error(), "AddGenerationPrompt") {
		t.Errorf("the refusal %q does not name the option", err)
	}
}
