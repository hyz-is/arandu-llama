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
// a model. With opts.AddGenerationPrompt false it ends with the last message
// instead, which is how a conversation with its assistant turn is rendered
// for training.
//
// With no vocabulary behind it the template sees empty bos_token and
// eos_token. For a model whose vocabulary adds BOS itself, Gemma 4 among them,
// that is also what Model.FormatChatPrompt returns: llama.cpp drops a leading
// BOS the template writes so the tokenizer does not write it twice.
func RenderChatTemplate(messages []ChatMessage, opts ChatOptions) (string, error) {
	if opts.ChatTemplate == "" {
		return "", errors.New("no chat template: RenderChatTemplate renders ChatOptions.ChatTemplate and has no model to read one from")
	}
	rendered, err := renderChat(nil, messages, opts)
	if err != nil {
		return "", err
	}
	return rendered.prompt, nil
}

// ParseChatOutput splits output, generated from messages rendered through
// opts.ChatTemplate, into Content and ReasoningContent the way Chat does,
// without a model. It exists to read a recorded Output again: the same
// messages, options and output give Chat's Content and ReasoningContent,
// because both render the template for its parser and generation prompt and
// both parse through the same function.
//
// finish is the FinishReason the output was generated with, since Chat parses
// FinishReasonLength output as partial and FinishReasonStop output as
// complete. The response carries output, finish and a GeneratedTokens of 0,
// because no tokens were counted here.
//
// Like RenderChatTemplate, it renders with empty bos_token and eos_token. The
// generation prompt is where the assistant turn opens, at the end of the
// render, so a template that writes those tokens only before it gets the same
// parser and generation prompt it would get with a model.
func ParseChatOutput(messages []ChatMessage, opts ChatOptions, output string, finish FinishReason) (*ChatResponse, error) {
	if opts.ChatTemplate == "" {
		return nil, errors.New("no chat template: ParseChatOutput renders ChatOptions.ChatTemplate and has no model to read one from")
	}
	if finish != FinishReasonStop && finish != FinishReasonLength {
		return nil, fmt.Errorf("finish reason %q is neither %q nor %q", finish, FinishReasonStop, FinishReasonLength)
	}
	if err := requireGenerationPrompt(opts, "ParseChatOutput"); err != nil {
		return nil, err
	}
	rendered, err := renderChat(nil, messages, opts)
	if err != nil {
		return nil, err
	}
	return rendered.response(output, finish, 0, opts.ReasoningFormat)
}

// requireGenerationPrompt refuses opts that render without the generation
// prompt where output is generated or read back. The output begins where the
// generation prompt ends; a render that stops after the last message has no
// such place, and a model given it would write the next turn's header rather
// than an answer.
func requireGenerationPrompt(opts ChatOptions, caller string) error {
	if opts.AddGenerationPrompt != nil && !*opts.AddGenerationPrompt {
		return fmt.Errorf("%s needs the generation prompt: ChatOptions.AddGenerationPrompt false renders a finished conversation, which RenderChatTemplate returns", caller)
	}
	return nil
}

// chatRender is what one render of the messages produces: the prompt to
// generate from, and what llama.cpp built beside it to read the output back.
type chatRender struct {
	prompt string
	// format is the common_chat_format llama.cpp detected. The PEG parse picks
	// its mapper by it: Gemma 4's thought channel has one of its own.
	format int
	// generationPrompt is the text the template appends to open the answer,
	// "<|start|>assistant" for gpt-oss. The parser starts there, and the output
	// does not repeat it.
	generationPrompt string
	// parser is the serialized PEG arena llama.cpp built for the template and
	// the reasoning format. Without it every parse is content only.
	parser string
}

// formatChatMessages renders messages into the prompt Chat generates from, and
// returns the chat format llama.cpp detected for the template it rendered.
func formatChatMessages(model *Model, messages []ChatMessage, opts ChatOptions) (string, int, error) {
	rendered, err := formatChat(model, messages, opts)
	if err != nil {
		return "", 0, err
	}
	return rendered.prompt, rendered.format, nil
}

// formatChat renders messages for a model: the prompt Chat generates from and
// the parser it reads the output with.
//
// The template is opts.ChatTemplate when set and the model's own GGUF template
// otherwise, and both go through the Jinja engine. A model without a template
// is an error rather than ChatML, and a template that fails to render is an
// error rather than a prompt from the legacy formatter: either would generate
// from a prompt the model was not trained on. For raw completion without a
// template, use Generate() instead of Chat().
func formatChat(model *Model, messages []ChatMessage, opts ChatOptions) (chatRender, error) {
	// The write lock is for the parsed templates cached on the model, and it
	// keeps Close from freeing them under a render.
	model.mu.Lock()
	defer model.mu.Unlock()

	if model.closed {
		return chatRender{}, fmt.Errorf("model is closed")
	}
	return renderChat(model, messages, opts)
}

// renderChat is the one render path. model is nil when there is no model
// behind the template; otherwise the caller holds model.mu for writing.
func renderChat(model *Model, messages []ChatMessage, opts ChatOptions) (chatRender, error) {
	if len(messages) == 0 {
		return chatRender{}, errors.New("messages cannot be empty")
	}
	// The format is handed to llama.cpp as its own enum, where a value past
	// the last one would still be "not none" and extract.
	if opts.ReasoningFormat < ReasoningFormatNone || opts.ReasoningFormat > ReasoningFormatDeepSeek {
		return chatRender{}, fmt.Errorf("unknown ReasoningFormat %d", int(opts.ReasoningFormat))
	}
	thinking, names, values, err := chatTemplateInputs(opts)
	if err != nil {
		return chatRender{}, err
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
			return chatRender{}, errors.New("ChatOptions.ChatTemplate is not a Jinja template (no {{ or {%): built-in format names are not accepted")
		}
		templates, err = initChatTemplates(modelPtr, opts.ChatTemplate)
		if err != nil {
			return chatRender{}, err
		}
		defer C.llama_wrapper_chat_templates_free(templates)
	} else {
		if modelPtr == nil || C.llama_wrapper_get_chat_template(modelPtr) == nil {
			return chatRender{}, errors.New("no chat template available: provide ChatOptions.ChatTemplate or use a model with embedded template (or use Generate() for raw completion)")
		}
		if model.chatTemplates == nil {
			parsed, err := initChatTemplates(modelPtr, "")
			if err != nil {
				return chatRender{}, err
			}
			model.chatTemplates = parsed
		}
		templates = model.chatTemplates
	}

	roles, freeRoles := cStrings(len(messages), func(i int) string { return messages[i].Role })
	defer freeRoles()
	contents, freeContents := cStrings(len(messages), func(i int) string { return messages[i].Content })
	defer freeContents()
	// Every message gets an entry, empty for no reasoning: llama.cpp hands the
	// template a reasoning_content only when it is not empty, so a message
	// without one renders as it did before the field existed.
	reasoning, freeReasoning := cStrings(len(messages), func(i int) string { return messages[i].ReasoningContent })
	defer freeReasoning()
	cNames, freeNames := cStrings(len(names), func(i int) string { return names[i] })
	defer freeNames()
	cValues, freeValues := cStrings(len(values), func(i int) string { return values[i] })
	defer freeValues()

	var namesPtr, valuesPtr **C.char
	if len(names) > 0 {
		namesPtr, valuesPtr = &cNames[0], &cValues[0]
	}
	addGenerationPrompt := opts.AddGenerationPrompt == nil || *opts.AddGenerationPrompt
	var format C.int
	var generationPrompt, parser *C.char
	result := C.llama_wrapper_chat_templates_render(templates, &roles[0], &contents[0], &reasoning[0], C.int(len(messages)),
		C.bool(addGenerationPrompt), C.int(thinking), C.llama_wrapper_reasoning_format(opts.ReasoningFormat),
		namesPtr, valuesPtr, C.int(len(names)), &format, &generationPrompt, &parser)
	if result == nil {
		return chatRender{}, errors.New(C.GoString(C.llama_wrapper_last_error()))
	}
	defer C.llama_wrapper_free_result(result)
	defer C.llama_wrapper_free_result(generationPrompt)
	defer C.llama_wrapper_free_result(parser)

	return chatRender{
		prompt:           C.GoString(result),
		format:           int(format),
		generationPrompt: C.GoString(generationPrompt),
		parser:           C.GoString(parser),
	}, nil
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

// response turns output generated from r.prompt into the ChatResponse that
// Chat and ParseChatOutput both return, so the two cannot disagree about the
// same output.
//
// With ReasoningFormatNone, Content is output unchanged. With any other format
// a failed parse is an error: the previous behaviour, returning output as
// Content, handed whoever scored Content the model's reasoning as its answer.
//
// Output cut at the token ceiling is parsed as partial, as a stream in
// progress is: the model did not finish it. llama.cpp's PEG parse is lenient,
// and for the recorded gpt-oss, Qwen3.8 and Gemma 4 outputs the split is the
// same either way, an unclosed thinking block included. Where partial differs
// is output the parser rejects, which it returns empty instead of failing.
func (r chatRender) response(output string, finish FinishReason, tokens int, format ReasoningFormat) (*ChatResponse, error) {
	response := &ChatResponse{Output: output, FinishReason: finish, GeneratedTokens: tokens}
	if format == ReasoningFormatNone {
		response.Content = output
		return response, nil
	}
	content, reasoning, err := r.parse(output, format, finish == FinishReasonLength)
	if err != nil {
		return nil, fmt.Errorf("output that stopped with finish reason %q: %w", finish, err)
	}
	response.Content = content
	response.ReasoningContent = reasoning
	return response, nil
}

// parse splits text into content and reasoning with the parser and generation
// prompt the render built. partial is true for text the model did not finish:
// a stream in progress, or output cut at the token ceiling.
func (r chatRender) parse(text string, format ReasoningFormat, partial bool) (content, reasoningContent string, err error) {
	if format == ReasoningFormatNone {
		return text, "", nil
	}

	cText := C.CString(text)
	defer C.free(unsafe.Pointer(cText))
	cGenerationPrompt := C.CString(r.generationPrompt)
	defer C.free(unsafe.Pointer(cGenerationPrompt))
	cParser := C.CString(r.parser)
	defer C.free(unsafe.Pointer(cParser))

	result := C.llama_wrapper_parse_reasoning(cText, C.bool(partial), C.llama_wrapper_reasoning_format(format),
		C.int(r.format), cGenerationPrompt, cParser)
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

// chatGenerateOptions is the generation half of opts, shared by Chat and
// ChatStream. Stop words are the caller's only: the template ends a turn with
// the model's end-of-generation token.
func chatGenerateOptions(opts ChatOptions) []GenerateOption {
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
	if opts.MinP != nil {
		genOpts = append(genOpts, WithMinP(*opts.MinP))
	}
	if opts.Seed != nil {
		genOpts = append(genOpts, WithSeed(*opts.Seed))
	}
	return genOpts
}

// chatWithContext implements non-streaming chat completion using a specific context.
//
// This is an internal helper called by Context.Chat().
//
// It generates through the wrapper's loop directly rather than through
// GenerateChannel, because only the loop knows why it stopped: a channel that
// closes looks the same after the model's end-of-generation token, after
// MaxTokens and after a failed decode, and the last two are not answers.
func (m *Model) chatWithContext(ctx gocontext.Context, c *Context, messages []ChatMessage, opts ChatOptions) (*ChatResponse, error) {
	if err := requireGenerationPrompt(opts, "Chat"); err != nil {
		return nil, err
	}
	rendered, err := formatChat(m, messages, opts)
	if err != nil {
		return nil, err
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	config := defaultGenerateConfig
	for _, opt := range chatGenerateOptions(opts) {
		opt(&config)
	}

	// The callback is how ctx reaches the loop: it runs once per token, and
	// refusing a token stops generation without appending it.
	var report generationReport
	output, err := c.generate(rendered.prompt, config, func(string) bool {
		return ctx.Err() == nil
	}, &report)
	if err != nil {
		return nil, err
	}

	var finish FinishReason
	switch report.stop {
	case generationStopEOG, generationStopWord:
		finish = FinishReasonStop
	case generationStopLength:
		finish = FinishReasonLength
	case generationStopCallback:
		// The callback refuses a token only once ctx is done.
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		return nil, fmt.Errorf("generation was stopped by its callback after %d tokens with the context still live", report.tokens)
	case generationStopDecodeFailed:
		return nil, fmt.Errorf("generation failed: llama_decode failed after %d tokens", report.tokens)
	default:
		return nil, fmt.Errorf("generation stopped after %d tokens for a reason the loop did not report", report.tokens)
	}

	return rendered.response(output, finish, report.tokens, opts.ReasoningFormat)
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

		fail := func(err error) {
			select {
			case errCh <- err:
			default:
			}
		}

		if err := requireGenerationPrompt(opts, "ChatStream"); err != nil {
			fail(err)
			return
		}

		// Build prompt from messages using chat template
		rendered, err := formatChat(m, messages, opts)
		if err != nil {
			fail(err)
			return
		}

		// Use context's GenerateChannel
		tokenCh, genErrCh := c.GenerateChannel(ctx, rendered.prompt, chatGenerateOptions(opts)...)

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

				// Parse accumulated output to extract reasoning. Every parse is
				// partial: the stream does not know which token is the last.
				content, reasoning, err := rendered.parse(accumulated.String(), opts.ReasoningFormat, true)
				if err != nil {
					// If parsing fails, send token as-is without reasoning extraction
					select {
					case deltaCh <- ChatDelta{Content: token}:
					case <-ctx.Done():
						return
					}
					continue
				}

				// Compute deltas (what's new since last parse). With a parser
				// that extracts, a partial parse can read a tag's first bytes as
				// text and give them back once the tag completes, so the new
				// parse is not always an extension of the last one.
				contentDelta, err := streamDelta("content", prevContent, content)
				if err != nil {
					fail(err)
					return
				}
				reasoningDelta, err := streamDelta("reasoning", prevReasoning, reasoning)
				if err != nil {
					fail(err)
					return
				}

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
					fail(err)
					return
				}
			case <-ctx.Done():
				return
			}
		}
	}()

	return deltaCh, errCh
}

// streamDelta is what current adds to what was already streamed as last, by
// the rule llama.cpp's string_diff follows for llama-server's deltas: a
// current that is a prefix of last sends nothing, and one that neither
// extends nor shortens last is an error, since what was sent cannot be taken
// back.
func streamDelta(field, last, current string) (string, error) {
	if strings.HasPrefix(current, last) {
		return current[len(last):], nil
	}
	if strings.HasPrefix(last, current) {
		return "", nil
	}
	return "", fmt.Errorf("the %d bytes of %s already streamed are not a prefix of the next parse's %d", len(last), field, len(current))
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
