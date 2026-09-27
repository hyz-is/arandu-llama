package llama

/*
#include "wrapper.h"
#include <stdlib.h>
*/
import "C"

import (
	gocontext "context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"unsafe"
)

// Chat implementation for Context is in context.go
// This file contains shared types, options, and helpers

// ChatTemplateEngine names how Chat, ChatStream, Model.FormatChatPrompt and
// RenderChatTemplate turn messages into a prompt: the template's own Jinja
// source rendered by llama.cpp's Jinja engine at the pinned commit. Keep it
// beside a template's digest wherever a prompt is recorded, because the same
// template bytes rendered by another engine are another prompt -- the legacy
// formatter this replaced wrote no system preamble for gpt-oss, no thinking
// tag for Qwen3.8 and nothing at all for Gemma 4. tests/Unit/Chat checks the
// suffix against `git ls-tree HEAD llama.cpp`.
const ChatTemplateEngine = "arandu-llama.chat-jinja.v1/llama.cpp.90c26fcd"

// RenderChatTemplate renders messages through opts.ChatTemplate with the
// engine ChatTemplateEngine names, ending with the generation prompt, without
// a model.
//
// With no vocabulary behind it the template sees empty bos_token and
// eos_token. For a model whose vocabulary adds BOS itself, Gemma 4 among them,
// that is also what Model.FormatChatPrompt returns: llama.cpp drops a leading
// BOS the template writes so the tokenizer does not write it twice.
func RenderChatTemplate(messages []ChatMessage, opts ChatOptions) (string, error) {
	if opts.ChatTemplate == "" {
		return "", errors.New("no chat template: RenderChatTemplate renders ChatOptions.ChatTemplate and has no model to read one from")
	}
	prompt, _, err := renderChat(nil, messages, opts)
	return prompt, err
}

// formatChatMessages renders messages into the prompt Chat generates from, and
// returns the chat format llama.cpp detected for the template it rendered.
//
// The template is opts.ChatTemplate when set and the model's own GGUF template
// otherwise, and both go through the Jinja engine. A model without a template
// is an error rather than ChatML, and a template that fails to render is an
// error rather than a prompt from the legacy formatter: either would generate
// from a prompt the model was not trained on. For raw completion without a
// template, use Generate() instead of Chat().
func formatChatMessages(model *Model, messages []ChatMessage, opts ChatOptions) (string, int, error) {
	// The write lock is for the parsed templates cached on the model, and it
	// keeps Close from freeing them under a render.
	model.mu.Lock()
	defer model.mu.Unlock()

	if model.closed {
		return "", 0, fmt.Errorf("model is closed")
	}
	return renderChat(model, messages, opts)
}

// renderChat is the one render path. model is nil when there is no model
// behind the template; otherwise the caller holds model.mu for writing.
func renderChat(model *Model, messages []ChatMessage, opts ChatOptions) (string, int, error) {
	if len(messages) == 0 {
		return "", 0, errors.New("messages cannot be empty")
	}
	thinking, names, values, err := chatTemplateInputs(opts)
	if err != nil {
		return "", 0, err
	}

	var modelPtr unsafe.Pointer
	if model != nil {
		modelPtr = model.modelPtr
	}

	var templates unsafe.Pointer
	if opts.ChatTemplate != "" {
		// A format name such as "llama3" selected the legacy formatter. To the
		// Jinja engine it is a template with no tags, whose rendering is the
		// name itself -- a prompt, and a wrong one, with no error.
		if !strings.Contains(opts.ChatTemplate, "{{") && !strings.Contains(opts.ChatTemplate, "{%") {
			return "", 0, errors.New("ChatOptions.ChatTemplate is not a Jinja template (no {{ or {%): built-in format names are not accepted")
		}
		templates, err = initChatTemplates(modelPtr, opts.ChatTemplate)
		if err != nil {
			return "", 0, err
		}
		defer C.llama_wrapper_chat_templates_free(templates)
	} else {
		if modelPtr == nil || C.llama_wrapper_get_chat_template(modelPtr) == nil {
			return "", 0, errors.New("no chat template available: provide ChatOptions.ChatTemplate or use a model with embedded template (or use Generate() for raw completion)")
		}
		if model.chatTemplates == nil {
			parsed, err := initChatTemplates(modelPtr, "")
			if err != nil {
				return "", 0, err
			}
			model.chatTemplates = parsed
		}
		templates = model.chatTemplates
	}

	roles, freeRoles := cStrings(len(messages), func(i int) string { return messages[i].Role })
	defer freeRoles()
	contents, freeContents := cStrings(len(messages), func(i int) string { return messages[i].Content })
	defer freeContents()
	cNames, freeNames := cStrings(len(names), func(i int) string { return names[i] })
	defer freeNames()
	cValues, freeValues := cStrings(len(values), func(i int) string { return values[i] })
	defer freeValues()

	var namesPtr, valuesPtr **C.char
	if len(names) > 0 {
		namesPtr, valuesPtr = &cNames[0], &cValues[0]
	}
	var format C.int
	result := C.llama_wrapper_chat_templates_render(templates, &roles[0], &contents[0], C.int(len(messages)),
		C.bool(true), C.int(thinking), namesPtr, valuesPtr, C.int(len(names)), &format)
	if result == nil {
		return "", 0, errors.New(C.GoString(C.llama_wrapper_last_error()))
	}
	defer C.llama_wrapper_free_result(result)

	return C.GoString(result), int(format), nil
}

// initChatTemplates parses a template for the Jinja engine; the caller frees
// the result with llama_wrapper_chat_templates_free.
func initChatTemplates(modelPtr unsafe.Pointer, override string) (unsafe.Pointer, error) {
	var cOverride *C.char
	if override != "" {
		cOverride = C.CString(override)
		defer C.free(unsafe.Pointer(cOverride))
	}
	templates := C.llama_wrapper_chat_templates_init(modelPtr, cOverride)
	if templates == nil {
		return nil, errors.New(C.GoString(C.llama_wrapper_last_error()))
	}
	return templates, nil
}

// chatTemplateInputs is what the template is given besides the messages:
// enable_thinking as -1 (llama-server's default), 0 or 1, and every
// ChatTemplateKwargs entry as JSON. An enable_thinking kwarg sets the same
// input EnableThinking does, as it does in llama-server, so the two
// disagreeing is an error rather than a precedence rule.
func chatTemplateInputs(opts ChatOptions) (int, []string, []string, error) {
	thinking := -1
	if opts.EnableThinking != nil {
		thinking = 0
		if *opts.EnableThinking {
			thinking = 1
		}
	}
	names := make([]string, 0, len(opts.ChatTemplateKwargs))
	values := make([]string, 0, len(opts.ChatTemplateKwargs))
	for name, value := range opts.ChatTemplateKwargs {
		if name == "enable_thinking" {
			enabled, ok := value.(bool)
			if !ok {
				return 0, nil, nil, fmt.Errorf("ChatTemplateKwargs[%q] is %T, want bool", name, value)
			}
			if opts.EnableThinking != nil && *opts.EnableThinking != enabled {
				return 0, nil, nil, fmt.Errorf("ChatTemplateKwargs[%q] = %t contradicts EnableThinking = %t", name, enabled, *opts.EnableThinking)
			}
			thinking = 0
			if enabled {
				thinking = 1
			}
		}
		encoded, err := json.Marshal(value)
		if err != nil {
			return 0, nil, nil, fmt.Errorf("ChatTemplateKwargs[%q]: %w", name, err)
		}
		names = append(names, name)
		values = append(values, string(encoded))
	}
	return thinking, names, values, nil
}

// cStrings copies n strings into C memory for the length of a call.
func cStrings(n int, at func(int) string) ([]*C.char, func()) {
	out := make([]*C.char, n)
	for i := range out {
		out[i] = C.CString(at(i))
	}
	return out, func() {
		for _, s := range out {
			C.free(unsafe.Pointer(s))
		}
	}
}

// parseReasoning extracts reasoning/thinking content from model output.
// Returns content and reasoning_content separately.
func parseReasoning(text string, format ReasoningFormat, chatFormat int) (content, reasoningContent string, err error) {
	if format == ReasoningFormatNone || text == "" {
		return text, "", nil
	}

	cText := C.CString(text)
	defer C.free(unsafe.Pointer(cText))

	cFormat := C.llama_wrapper_reasoning_format(format)
	cChatFormat := C.int(chatFormat)

	// Parse with is_partial=true for streaming
	result := C.llama_wrapper_parse_reasoning(cText, C.bool(true), cFormat, cChatFormat)
	if result == nil {
		return "", "", fmt.Errorf("failed to parse reasoning: %s", C.GoString(C.llama_wrapper_last_error()))
	}
	defer C.llama_wrapper_free_parsed_message(result)

	content = C.GoString(result.content)
	if result.reasoning_content != nil {
		reasoningContent = C.GoString(result.reasoning_content)
	}

	return content, reasoningContent, nil
}

// chatWithContext implements non-streaming chat completion using a specific context.
//
// This is an internal helper called by Context.Chat().
func (m *Model) chatWithContext(ctx gocontext.Context, c *Context, messages []ChatMessage, opts ChatOptions) (*ChatResponse, error) {
	// Build prompt from messages using chat template
	prompt, chatFormat, err := formatChatMessages(m, messages, opts)
	if err != nil {
		return nil, err
	}

	// Build generation options from chat options
	// Use user-provided stop words (no defaults - template handles this)
	genOpts := []GenerateOption{
		WithStopWords(opts.StopWords...),
	}

	if opts.MaxTokens != nil {
		genOpts = append(genOpts, WithMaxTokens(*opts.MaxTokens))
	}
	if opts.Temperature != nil {
		genOpts = append(genOpts, WithTemperature(*opts.Temperature))
	}
	if opts.TopP != nil {
		genOpts = append(genOpts, WithTopP(*opts.TopP))
	}
	if opts.TopK != nil {
		genOpts = append(genOpts, WithTopK(*opts.TopK))
	}
	if opts.Seed != nil {
		genOpts = append(genOpts, WithSeed(*opts.Seed))
	}

	// Generate using context's GenerateChannel
	tokenCh, errCh := c.GenerateChannel(ctx, prompt, genOpts...)

	var content strings.Builder

Loop:
	for {
		select {
		case token, ok := <-tokenCh:
			if !ok {
				break Loop
			}
			content.WriteString(token)
		case err := <-errCh:
			if err != nil {
				return nil, err
			}
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}

	// Parse final output to extract reasoning
	fullOutput := content.String()
	parsedContent, reasoning, err := parseReasoning(fullOutput, opts.ReasoningFormat, chatFormat)
	if err != nil {
		// If parsing fails, return content as-is without reasoning extraction
		return &ChatResponse{Content: fullOutput}, nil
	}

	return &ChatResponse{
		Content:          parsedContent,
		ReasoningContent: reasoning,
	}, nil
}

// chatStreamWithContext implements streaming chat completion using a specific context.
//
// This is an internal helper called by Context.ChatStream().
func (m *Model) chatStreamWithContext(ctx gocontext.Context, c *Context, messages []ChatMessage, opts ChatOptions) (<-chan ChatDelta, <-chan error) {
	bufferSize := 256
	if opts.StreamBufferSize > 0 {
		bufferSize = opts.StreamBufferSize
	}

	deltaCh := make(chan ChatDelta, bufferSize)
	errCh := make(chan error, 1)

	go func() {
		defer close(deltaCh)
		defer close(errCh)

		// Build prompt from messages using chat template
		prompt, chatFormat, err := formatChatMessages(m, messages, opts)
		if err != nil {
			select {
			case errCh <- err:
			default:
			}
			return
		}

		// Build generation options from chat options
		// Use user-provided stop words (no defaults - template handles this)
		genOpts := []GenerateOption{
			WithStopWords(opts.StopWords...),
		}

		if opts.MaxTokens != nil {
			genOpts = append(genOpts, WithMaxTokens(*opts.MaxTokens))
		}
		if opts.Temperature != nil {
			genOpts = append(genOpts, WithTemperature(*opts.Temperature))
		}
		if opts.TopP != nil {
			genOpts = append(genOpts, WithTopP(*opts.TopP))
		}
		if opts.TopK != nil {
			genOpts = append(genOpts, WithTopK(*opts.TopK))
		}
		if opts.Seed != nil {
			genOpts = append(genOpts, WithSeed(*opts.Seed))
		}

		// Use context's GenerateChannel
		tokenCh, genErrCh := c.GenerateChannel(ctx, prompt, genOpts...)

		// Track accumulated output and previous parsed state for delta computation
		var accumulated strings.Builder
		var prevContent, prevReasoning string

	Loop:
		for {
			select {
			case token, ok := <-tokenCh:
				if !ok {
					break Loop
				}

				// Accumulate token
				accumulated.WriteString(token)

				// Parse accumulated output to extract reasoning
				content, reasoning, err := parseReasoning(accumulated.String(), opts.ReasoningFormat, chatFormat)
				if err != nil {
					// If parsing fails, send token as-is without reasoning extraction
					select {
					case deltaCh <- ChatDelta{Content: token}:
					case <-ctx.Done():
						return
					}
					continue
				}

				// Compute deltas (what's new since last parse)
				contentDelta := content[len(prevContent):]
				reasoningDelta := reasoning[len(prevReasoning):]

				// Send delta if there's new content or reasoning
				if contentDelta != "" || reasoningDelta != "" {
					select {
					case deltaCh <- ChatDelta{
						Content:          contentDelta,
						ReasoningContent: reasoningDelta,
					}:
					case <-ctx.Done():
						return
					}
				}

				// Update previous state
				prevContent = content
				prevReasoning = reasoning

			case err := <-genErrCh:
				if err != nil {
					select {
					case errCh <- err:
					default:
					}
					return
				}
			case <-ctx.Done():
				return
			}
		}
	}()

	return deltaCh, errCh
}

// Int returns a pointer to the given int value.
// This is a convenience helper for setting optional ChatOptions fields.
//
// Example:
//
//	opts := llama.ChatOptions{
//	    MaxTokens: llama.Int(100),  // Instead of &100
//	}
func Int(v int) *int {
	return &v
}

// Float32 returns a pointer to the given float32 value.
// This is a convenience helper for setting optional ChatOptions fields.
//
// Example:
//
//	opts := llama.ChatOptions{
//	    Temperature: llama.Float32(0.7),  // Instead of &0.7
//	}
func Float32(v float32) *float32 {
	return &v
}

// Bool returns a pointer to the given bool value.
// This is a convenience helper for setting optional ChatOptions fields.
//
// Example:
//
//	opts := llama.ChatOptions{
//	    EnableThinking: llama.Bool(true),  // Instead of &true
//	}
func Bool(v bool) *bool {
	return &v
}
