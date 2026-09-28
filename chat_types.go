package llama

// ChatMessage represents a message in a chat conversation.
//
// Common roles include "system", "user", "assistant", "tool", and "function".
// The role is not validated by this library - the model's chat template will
// handle role interpretation and any unknown roles.
//
// Example:
//
//	messages := []llama.ChatMessage{
//	    {Role: "system", Content: "You are a helpful assistant."},
//	    {Role: "user", Content: "What is the capital of France?"},
//	}
type ChatMessage struct {
	Role    string // Message role (e.g., "system", "user", "assistant")
	Content string // Message content

	// ReasoningContent is an assistant turn's thinking, kept apart from its
	// answer in Content the way ChatResponse keeps them. It reaches the
	// template as the message's reasoning_content, and a template that
	// writes thinking into the turn puts it there -- between <think> and
	// </think> for the Ornith. Empty is no reasoning, and the message renders
	// as it did before the field existed. A template that never reads
	// reasoning_content drops it without an error.
	ReasoningContent string
}

// ChatResponse represents the complete response from a chat completion.
//
// With ReasoningFormatNone, Content is Output unchanged. With any other
// format, Output is split by the parser llama.cpp builds for the template:
// thinking goes to ReasoningContent and the answer to Content, and output the
// parser does not accept is an error rather than a Content holding both.
//
// Example:
//
//	response, err := model.Chat(ctx, messages, opts)
//	if err != nil {
//	    log.Fatal(err)
//	}
//	fmt.Println("Response:", response.Content)
//	if response.ReasoningContent != "" {
//	    fmt.Println("Reasoning:", response.ReasoningContent)
//	}
type ChatResponse struct {
	Content          string // Regular response content
	ReasoningContent string // Extracted reasoning/thinking (if reasoning model)

	// Output is the generated text before any parse, special tokens written
	// as text: "<|channel|>analysis<|message|>..." for gpt-oss. It is what
	// ParseChatOutput takes to reproduce Content and ReasoningContent.
	Output string

	// FinishReason is why generation stopped. With FinishReasonLength the
	// answer was cut at MaxTokens, and Content may be empty or unfinished
	// while ReasoningContent holds the thinking that used the budget.
	FinishReason FinishReason

	// GeneratedTokens is the number of tokens whose pieces make up Output,
	// counted by the generation loop rather than by tokenizing Output again.
	// The end-of-generation token is not among them.
	GeneratedTokens int
}

// FinishReason is why a chat completion stopped, named as in the OpenAI chat
// API that llama-server serves.
type FinishReason string

const (
	// FinishReasonStop means the model ended the answer itself, with its
	// end-of-generation token or one of ChatOptions.StopWords.
	FinishReasonStop FinishReason = "stop"

	// FinishReasonLength means ChatOptions.MaxTokens tokens were generated
	// before the model ended the answer.
	FinishReasonLength FinishReason = "length"
)

// ChatDelta represents a streaming chunk from chat completion.
//
// During streaming, deltas arrive progressively. For standard models,
// only Content is populated with token(s). For reasoning models with
// extraction enabled, tokens may appear in either Content or
// ReasoningContent depending on whether they're inside reasoning tags.
//
// Example:
//
//	deltaCh, errCh := model.ChatStream(ctx, messages, opts)
//	for {
//	    select {
//	    case delta, ok := <-deltaCh:
//	        if !ok {
//	            return
//	        }
//	        if delta.Content != "" {
//	            fmt.Print(delta.Content)
//	        }
//	        if delta.ReasoningContent != "" {
//	            fmt.Print("[thinking: ", delta.ReasoningContent, "]")
//	        }
//	    case err := <-errCh:
//	        if err != nil {
//	            log.Fatal(err)
//	        }
//	    }
//	}
type ChatDelta struct {
	Content          string // Regular content token(s)
	ReasoningContent string // Reasoning token(s)
	// Future fields: ToolCalls, Role, FinishReason, etc.
}
