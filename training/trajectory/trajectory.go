// Package trajectory turns a verified teacher trajectory -- the question, the
// teacher's reasoning and its final answer -- into one row of the curriculum
// training/local reads. The student's own chat template writes the text and
// the student's own tokenizer writes the tokens: a teacher's text does not
// depend on its tokenizer, so any family can supply it, but a row the student
// trains on has to be what the student sees at inference.
//
// Choosing which trajectories deserve a row is the verifier's job, not this
// package's: Build writes a row for whatever it is given.
package trajectory

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"strings"

	llama "github.com/tayi-ai/arandu-llama"
	"github.com/tayi-ai/arandu-llama/training/tokenizer"
)

// IgnoreLabel is the label of every prompt position, the value training/local
// requires there so the loss reads the completion only.
const IgnoreLabel int64 = -100

// Example is one verified trajectory.
type Example struct {
	// ID names the row; the curriculum refuses a repeated one.
	ID string
	// Messages is the conversation the teacher answered, ending with the user
	// turn the answer replies to.
	Messages []llama.ChatMessage
	// Reasoning is the teacher's thinking, ChatResponse.ReasoningContent.
	Reasoning string
	// Answer is the teacher's final answer, ChatResponse.Content.
	Answer string
}

// Student is the model a row is built for.
type Student struct {
	// ChatTemplate is the student's Jinja chat template.
	ChatTemplate string
	// EndOfTurn is the piece that closes the assistant's turn, the one the
	// student's generation stops on: "<|im_end|>" for the Ornith. The
	// completion ends with it, and it has to be a single token, or the
	// student would learn to spell the stop rather than to emit it.
	EndOfTurn string
	// Tokenizer is the student's, loaded with tokenizer.Load against the
	// SHA256 the student's admission expects.
	Tokenizer *tokenizer.Tokenizer
	// Vocabulary is the number of rows in the student's token embedding. An
	// id at or past it has no row to train, so a row holding one is refused.
	Vocabulary int
}

// Text is an Example rendered through the student's template.
type Text struct {
	// Prompt is the conversation with the generation prompt and thinking on:
	// what the student is given at inference, ending where its answer begins.
	// For the Ornith that is inside the thinking block, after "<think>\n".
	Prompt string
	// Completion is what the conversation with the assistant's turn adds
	// after Prompt, through EndOfTurn: the reasoning, the template's close of
	// the thinking block, the answer and EndOfTurn itself.
	Completion string
	// Tail is what the template writes after EndOfTurn -- "\n" for the Ornith.
	// It is not trained: generation stops on EndOfTurn, so the student never
	// writes what comes after it.
	Tail string
}

// Row is one line of the curriculum training/local reads. InputIDs is the
// prompt's tokens followed by the completion's, PromptTokens counts the
// prompt's, and Labels holds IgnoreLabel over the prompt and the token itself
// over the completion.
type Row struct {
	ID           string  `json:"id"`
	InputIDs     []int64 `json:"input_ids"`
	Labels       []int64 `json:"labels"`
	PromptTokens int     `json:"prompt_tokens"`
}

// Report is what Build measured on the way to a Row, to keep beside it.
type Report struct {
	Text Text
	// JointAgrees is whether tokenizing Prompt and Completion as one text
	// gives Row.InputIDs. The row does not depend on it: InputIDs is the
	// prompt's tokens and then the completion's, because at inference the
	// prompt is tokenized alone and the student continues from its last
	// token. false means the tokenizer merges across that boundary, so a
	// corpus tokenized whole would disagree with these rows.
	JointAgrees bool
}

// Render writes example through chatTemplate: the prompt the student is given
// and the completion it is trained to write, which ends with endOfTurn.
//
// The prompt is the messages with the generation prompt and thinking on. The
// completion is what rendering the messages with the assistant's turn --
// Reasoning as its ReasoningContent, Answer as its Content -- and without the
// generation prompt adds after the prompt. A render that does not begin with
// the prompt is an error: the student would be trained on a prefix it is never
// given. So are a completion in which endOfTurn does not appear exactly once,
// and one that lacks the reasoning or the answer, which is what a template
// that drops reasoning_content produces.
//
// The template decides whitespace. The Ornith's trims the reasoning and the
// answer, so the text in the completion is theirs without surrounding space.
func Render(chatTemplate, endOfTurn string, example Example) (Text, error) {
	if err := example.validate(); err != nil {
		return Text{}, err
	}
	if endOfTurn == "" {
		return Text{}, errors.New("trajectory: no end of turn: the completion has to end where the student stops")
	}
	opts := llama.ChatOptions{ChatTemplate: chatTemplate, EnableThinking: llama.Bool(true)}
	prompt, err := llama.RenderChatTemplate(example.Messages, opts)
	if err != nil {
		return Text{}, fmt.Errorf("trajectory %q: rendering the prompt: %w", example.ID, err)
	}
	conversation := append(slices.Clone(example.Messages), llama.ChatMessage{
		Role:             "assistant",
		Content:          example.Answer,
		ReasoningContent: example.Reasoning,
	})
	opts.AddGenerationPrompt = llama.Bool(false)
	full, err := llama.RenderChatTemplate(conversation, opts)
	if err != nil {
		return Text{}, fmt.Errorf("trajectory %q: rendering the conversation: %w", example.ID, err)
	}
	if !strings.HasPrefix(full, prompt) {
		at := 0
		for at < len(full) && at < len(prompt) && full[at] == prompt[at] {
			at++
		}
		return Text{}, fmt.Errorf("trajectory %q: the conversation with the assistant's turn does not begin with the prompt: they part at byte %d, prompt %q, conversation %q",
			example.ID, at, excerpt(prompt, at), excerpt(full, at))
	}
	rest := full[len(prompt):]
	if n := strings.Count(rest, endOfTurn); n != 1 {
		return Text{}, fmt.Errorf("trajectory %q: the assistant's turn holds the end of turn %q %d times, want once", example.ID, endOfTurn, n)
	}
	end := strings.Index(rest, endOfTurn) + len(endOfTurn)
	completion := rest[:end]
	reasoning, answer := strings.TrimSpace(example.Reasoning), strings.TrimSpace(example.Answer)
	at := strings.Index(completion, reasoning)
	if at < 0 {
		return Text{}, fmt.Errorf("trajectory %q: the template dropped the reasoning from the assistant's turn", example.ID)
	}
	if !strings.Contains(completion[at+len(reasoning):], answer) {
		return Text{}, fmt.Errorf("trajectory %q: the assistant's turn does not hold the answer after the reasoning", example.ID)
	}
	return Text{Prompt: prompt, Completion: completion, Tail: rest[end:]}, nil
}

// Build renders example for student with Render and tokenizes it into a Row.
//
// The tokens are the prompt's and then the completion's, each tokenized
// alone, which is the boundary the student meets at inference; Report says
// whether tokenizing the whole text agrees. The completion's last token must
// be EndOfTurn's one token, and every id must fall inside Vocabulary.
//
// It reads nothing and writes nothing: the same student and example give the
// same row.
func Build(ctx context.Context, student Student, example Example) (Row, Report, error) {
	if ctx == nil {
		return Row{}, Report{}, errors.New("trajectory: context required")
	}
	if student.Tokenizer == nil {
		return Row{}, Report{}, errors.New("trajectory: the student's tokenizer is required")
	}
	if student.Vocabulary < 1 {
		return Row{}, Report{}, fmt.Errorf("trajectory: student vocabulary %d, want the embedding's row count", student.Vocabulary)
	}
	text, err := Render(student.ChatTemplate, student.EndOfTurn, example)
	if err != nil {
		return Row{}, Report{}, err
	}
	endOfTurn, err := student.Tokenizer.Encode(ctx, student.EndOfTurn)
	if err != nil {
		return Row{}, Report{}, fmt.Errorf("trajectory: tokenizing the end of turn: %w", err)
	}
	if len(endOfTurn) != 1 {
		return Row{}, Report{}, fmt.Errorf("trajectory: the end of turn %q is %d tokens, want the one token the student stops on", student.EndOfTurn, len(endOfTurn))
	}
	prompt, err := student.Tokenizer.Encode(ctx, text.Prompt)
	if err != nil {
		return Row{}, Report{}, fmt.Errorf("trajectory %q: tokenizing the prompt: %w", example.ID, err)
	}
	completion, err := student.Tokenizer.Encode(ctx, text.Completion)
	if err != nil {
		return Row{}, Report{}, fmt.Errorf("trajectory %q: tokenizing the completion: %w", example.ID, err)
	}
	if len(prompt) == 0 || len(completion) == 0 {
		return Row{}, Report{}, fmt.Errorf("trajectory %q: %d prompt and %d completion tokens, want both", example.ID, len(prompt), len(completion))
	}
	if last := completion[len(completion)-1]; last != endOfTurn[0] {
		return Row{}, Report{}, fmt.Errorf("trajectory %q: the completion ends with token %d, not the end of turn's %d", example.ID, last, endOfTurn[0])
	}

	inputIDs := append(slices.Clip(prompt), completion...)
	labels := make([]int64, len(inputIDs))
	for i, id := range inputIDs {
		if id < 0 || id >= int64(student.Vocabulary) {
			return Row{}, Report{}, fmt.Errorf("trajectory %q: token %d at position %d is outside the student's vocabulary of %d", example.ID, id, i, student.Vocabulary)
		}
		labels[i] = id
		if i < len(prompt) {
			labels[i] = IgnoreLabel
		}
	}

	joint, err := student.Tokenizer.Encode(ctx, text.Prompt+text.Completion)
	if err != nil {
		return Row{}, Report{}, fmt.Errorf("trajectory %q: tokenizing the prompt and completion together: %w", example.ID, err)
	}
	row := Row{ID: example.ID, InputIDs: inputIDs, Labels: labels, PromptTokens: len(prompt)}
	return row, Report{Text: text, JointAgrees: slices.Equal(joint, inputIDs)}, nil
}

// validate refuses what cannot become a row whatever the template: a row the
// curriculum cannot name, a conversation that does not end with the question,
// and an empty reasoning or answer. The Ornith's template trims either to
// nothing, and the row would teach the student an empty thinking block or an
// empty reply -- the one thing a verified trajectory was chosen not to be.
func (e Example) validate() error {
	if e.ID == "" {
		return errors.New("trajectory: no ID")
	}
	if len(e.Messages) == 0 {
		return fmt.Errorf("trajectory %q: no messages", e.ID)
	}
	if last := e.Messages[len(e.Messages)-1].Role; last != "user" {
		return fmt.Errorf("trajectory %q: the conversation ends with a %q turn, want the user's question", e.ID, last)
	}
	if strings.TrimSpace(e.Reasoning) == "" {
		return fmt.Errorf("trajectory %q: empty reasoning", e.ID)
	}
	if strings.TrimSpace(e.Answer) == "" {
		return fmt.Errorf("trajectory %q: empty answer", e.ID)
	}
	return nil
}

// excerpt is up to 40 bytes of s from at, for an error that shows where two
// renders part without printing either whole.
func excerpt(s string, at int) string {
	end := min(at+40, len(s))
	return s[at:end]
}
