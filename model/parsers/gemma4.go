package parsers

import (
	"encoding/json"
	"errors"
	"log/slog"
	"regexp"
	"strings"
	"unicode"

	"github.com/ollama/ollama/api"
)

type Gemma4ParserState int

const (
	Gemma4CollectingContent Gemma4ParserState = iota
	Gemma4CollectingThinking
	Gemma4CollectingToolCall
)

const (
	gemma4ThinkingOpenTag  = "<|channel>"
	gemma4ThinkingCloseTag = "<channel|>"
	gemma4ToolCallOpenTag  = "<|tool_call>"
	gemma4ToolCallCloseTag = "<tool_call|>"
)

var (
	gemma4QuotedStringRe = regexp.MustCompile(`(?s)<\|"\|>(.*?)<\|"\|>`)
	gemma4BareKeyRe      = regexp.MustCompile(`([,{])(\w+):`)
)

type Gemma4Parser struct {
	state                 Gemma4ParserState
	buffer                strings.Builder
	hasThinkingSupport    bool
	thinkingEnabled       bool // true when both model supports and user requested thinking
	needsChannelNameStrip bool // true when we just entered thinking and need to strip "thought\n"
}

func (p *Gemma4Parser) HasToolSupport() bool {
	return true
}

func (p *Gemma4Parser) HasThinkingSupport() bool {
	return p.hasThinkingSupport
}

func (p *Gemma4Parser) Init(tools []api.Tool, lastMessage *api.Message, thinkValue *api.ThinkValue) []api.Tool {
	prefill := lastMessage != nil && lastMessage.Role == "assistant"

	p.thinkingEnabled = p.HasThinkingSupport() && (thinkValue != nil && thinkValue.Bool())

	if !p.thinkingEnabled {
		p.state = Gemma4CollectingContent
		return tools
	}

	if prefill && lastMessage.Content != "" {
		p.state = Gemma4CollectingContent
		return tools
	}

	// When thinking is enabled, start in content mode but we'll switch to
	// thinking when we see <|channel>. The model typically starts with
	// <|channel> immediately when thinking is enabled.
	p.state = Gemma4CollectingContent
	return tools
}

type gemma4Event interface {
	isGemma4Event()
}

type gemma4EventThinkingContent struct {
	content string
}

type gemma4EventContent struct {
	content string
}

type gemma4EventToolCall struct {
	toolCall api.ToolCall
}

func (gemma4EventThinkingContent) isGemma4Event() {}
func (gemma4EventContent) isGemma4Event()         {}
func (gemma4EventToolCall) isGemma4Event()        {}

func (p *Gemma4Parser) Add(s string, done bool) (content string, thinking string, calls []api.ToolCall, err error) {
	p.buffer.WriteString(s)
	events := p.parseEvents(done)

	var toolCalls []api.ToolCall
	var contentSb strings.Builder
	var thinkingSb strings.Builder
	for _, event := range events {
		switch event := event.(type) {
		case gemma4EventToolCall:
			toolCalls = append(toolCalls, event.toolCall)
		case gemma4EventThinkingContent:
			if p.thinkingEnabled {
				thinkingSb.WriteString(event.content)
			}
			// When thinking is disabled, silently discard channel content
		case gemma4EventContent:
			contentSb.WriteString(event.content)
		}
	}

	return contentSb.String(), thinkingSb.String(), toolCalls, nil
}

func (p *Gemma4Parser) parseEvents(done bool) []gemma4Event {
	var all []gemma4Event

	keepLooping := true
	for keepLooping {
		var events []gemma4Event
		events, keepLooping = p.eat(done)
		if len(events) > 0 {
			all = append(all, events...)
		}
	}

	return all
}

// longestOverlap returns the longest overlap between the suffix of bufStr and
// a prefix of any of the given tags.
func longestOverlap(bufStr string, tags ...string) int {
	maxOverlap := 0
	for _, tag := range tags {
		if o := overlap(bufStr, tag); o > maxOverlap {
			maxOverlap = o
		}
	}
	return maxOverlap
}

func (p *Gemma4Parser) eat(done bool) ([]gemma4Event, bool) {
	var events []gemma4Event
	bufStr := p.buffer.String()
	if bufStr == "" {
		return events, false
	}

	switch p.state {
	case Gemma4CollectingContent:
		// Check for thinking open tag
		if idx := strings.Index(bufStr, gemma4ThinkingOpenTag); idx != -1 {
			contentBefore := bufStr[:idx]
			remaining := bufStr[idx+len(gemma4ThinkingOpenTag):]

			p.buffer.Reset()
			p.buffer.WriteString(remaining)
			p.state = Gemma4CollectingThinking
			p.needsChannelNameStrip = true

			if contentBefore = strings.TrimRightFunc(contentBefore, unicode.IsSpace); len(contentBefore) > 0 {
				events = append(events, gemma4EventContent{content: contentBefore})
			}
			return events, true
		}

		// Check for tool call open tag
		if idx := strings.Index(bufStr, gemma4ToolCallOpenTag); idx != -1 {
			contentBefore := bufStr[:idx]
			remaining := bufStr[idx+len(gemma4ToolCallOpenTag):]

			p.buffer.Reset()
			p.buffer.WriteString(remaining)
			p.state = Gemma4CollectingToolCall

			if contentBefore = strings.TrimRightFunc(contentBefore, unicode.IsSpace); len(contentBefore) > 0 {
				events = append(events, gemma4EventContent{content: contentBefore})
			}
			return events, true
		}

		// Check for partial tag overlap
		if !done {
			if overlapLen := longestOverlap(bufStr, gemma4ThinkingOpenTag, gemma4ToolCallOpenTag); overlapLen > 0 {
				beforePartialTag := bufStr[:len(bufStr)-overlapLen]
				trailingLen := trailingWhitespaceLen(beforePartialTag)
				ambiguousStart := len(beforePartialTag) - trailingLen

				unambiguous := bufStr[:ambiguousStart]
				ambiguous := bufStr[ambiguousStart:]
				p.buffer.Reset()
				p.buffer.WriteString(ambiguous)
				if len(unambiguous) > 0 {
					events = append(events, gemma4EventContent{content: unambiguous})
				}
				return events, false
			}
		}

		// No tags found, emit all content
		p.buffer.Reset()
		if len(bufStr) > 0 {
			events = append(events, gemma4EventContent{content: bufStr})
		}
		return events, false

	case Gemma4CollectingThinking:
		// Strip channel name (e.g., "thought\n") after <|channel>.
		// Gemma 4 format: <|channel>thought\n...content...<channel|>
		// In streaming mode, "thought" and "\n" may arrive in separate chunks.
		if p.needsChannelNameStrip {
			if strings.HasPrefix(bufStr, "thought\n") {
				bufStr = bufStr[len("thought\n"):]
				p.buffer.Reset()
				p.buffer.WriteString(bufStr)
				p.needsChannelNameStrip = false
			} else if !done && (bufStr == "thought" || strings.HasPrefix("thought\n", bufStr)) {
				// Partial match — wait for more data.
				return events, false
			} else {
				// No match (different channel name or no newline) — don't strip.
				p.needsChannelNameStrip = false
			}
		}

		if strings.Contains(bufStr, gemma4ThinkingCloseTag) {
			split := strings.SplitN(bufStr, gemma4ThinkingCloseTag, 2)
			thinking := strings.TrimRightFunc(split[0], unicode.IsSpace)
			remaining := strings.TrimLeftFunc(split[1], unicode.IsSpace)

			p.buffer.Reset()
			p.buffer.WriteString(remaining)
			p.state = Gemma4CollectingContent

			if len(thinking) > 0 {
				events = append(events, gemma4EventThinkingContent{content: thinking})
			}
			return events, true
		}

		// Check for partial close tag
		if !done {
			if overlapLen := overlap(bufStr, gemma4ThinkingCloseTag); overlapLen > 0 {
				beforePartialTag := bufStr[:len(bufStr)-overlapLen]
				trailingLen := trailingWhitespaceLen(beforePartialTag)
				ambiguousStart := len(beforePartialTag) - trailingLen

				unambiguous := bufStr[:ambiguousStart]
				ambiguous := bufStr[ambiguousStart:]
				p.buffer.Reset()
				p.buffer.WriteString(ambiguous)
				if len(unambiguous) > 0 {
					events = append(events, gemma4EventThinkingContent{content: unambiguous})
				}
				return events, false
			}
		}

		// No close tag, emit thinking content (hold back trailing whitespace)
		if !done {
			whitespaceLen := trailingWhitespaceLen(bufStr)
			ambiguousStart := len(bufStr) - whitespaceLen

			unambiguous := bufStr[:ambiguousStart]
			ambiguous := bufStr[ambiguousStart:]
			p.buffer.Reset()
			p.buffer.WriteString(ambiguous)
			if len(unambiguous) > 0 {
				events = append(events, gemma4EventThinkingContent{content: unambiguous})
			}
		} else {
			p.buffer.Reset()
			if len(bufStr) > 0 {
				events = append(events, gemma4EventThinkingContent{content: bufStr})
			}
		}
		return events, false

	case Gemma4CollectingToolCall:
		if idx := strings.Index(bufStr, gemma4ToolCallCloseTag); idx != -1 {
			toolCallContent := bufStr[:idx]
			remaining := bufStr[idx+len(gemma4ToolCallCloseTag):]
			remaining = strings.TrimLeftFunc(remaining, unicode.IsSpace)

			p.buffer.Reset()
			p.buffer.WriteString(remaining)
			p.state = Gemma4CollectingContent

			if toolCall, err := parseGemma4ToolCall(toolCallContent); err == nil {
				events = append(events, gemma4EventToolCall{toolCall: toolCall})
			} else {
				slog.Warn("gemma4 tool call parsing failed", "error", err, "content", toolCallContent)
			}
			return events, true
		}

		// If done, flush any accumulated tool call content even without closing tag.
		// The model may hit a stop token before emitting <tool_call|>.
		if done && len(bufStr) > 0 {
			p.buffer.Reset()
			p.state = Gemma4CollectingContent
			if toolCall, err := parseGemma4ToolCall(bufStr); err == nil {
				events = append(events, gemma4EventToolCall{toolCall: toolCall})
			} else {
				slog.Warn("gemma4 tool call flush on done failed", "error", err, "content", bufStr)
			}
			return events, false
		}

		// Wait for closing tag
		return events, false
	}

	return events, false
}

// parseGemma4ToolCall parses a tool call in Gemma 4 format:
// call:NAME{key:value,key:value}
func parseGemma4ToolCall(content string) (api.ToolCall, error) {
	// Expected format: call:NAME{args}
	if !strings.HasPrefix(content, "call:") {
		return api.ToolCall{}, errors.New("expected 'call:' prefix")
	}
	content = content[len("call:"):]

	// Find the opening brace for args
	braceIdx := strings.Index(content, "{")
	if braceIdx == -1 {
		return api.ToolCall{}, errors.New("expected '{' in tool call")
	}

	toolName := strings.TrimSpace(content[:braceIdx])
	argsStr := content[braceIdx:]

	// Convert Gemma 4 argument format to JSON
	jsonStr := gemma4ArgsToJSON(argsStr)

	var args api.ToolCallFunctionArguments
	if err := json.Unmarshal([]byte(jsonStr), &args); err != nil {
		return api.ToolCall{}, err
	}

	return api.ToolCall{
		Function: api.ToolCallFunction{
			Name:      toolName,
			Arguments: args,
		},
	}, nil
}

// gemma4ArgsToJSON converts Gemma 4's custom argument format to valid JSON.
//
// Gemma 4 uses <|"|> as string delimiters in tool call arguments. The content
// between these delimiters can itself contain <|"|> tokens (e.g. when the model
// quotes descriptions or outputs markup). A simple non-greedy regex fails for
// this case because it matches the first inner <|"|> as the closing delimiter.
//
// Instead, we parse structurally: we find top-level key-value pairs by tracking
// brace/bracket depth and <|"|> nesting, then convert each value appropriately.
func gemma4ArgsToJSON(s string) string {
	const delim = "<|\"|>"

	var quotedStrings []string
	var sb strings.Builder

	i := 0
	for i < len(s) {
		// Check for opening <|"|> delimiter
		if i+len(delim) <= len(s) && s[i:i+len(delim)] == delim {
			// Find the matching closing <|"|> at the top level.
			// We need to find the LAST <|"|> before a structural boundary
			// (comma at depth 0, closing brace/bracket at depth 0, or end of string).
			start := i + len(delim)
			closeIdx := findClosingDelim(s, start)
			if closeIdx == -1 {
				// No closing delimiter found — treat rest as the quoted string
				quotedStrings = append(quotedStrings, s[start:])
				sb.WriteString(gemma4Placeholder(len(quotedStrings) - 1))
				i = len(s)
			} else {
				quotedStrings = append(quotedStrings, s[start:closeIdx])
				sb.WriteString(gemma4Placeholder(len(quotedStrings) - 1))
				i = closeIdx + len(delim)
			}
		} else {
			sb.WriteByte(s[i])
			i++
		}
	}

	text := sb.String()

	// Quote bare keys: {key: or ,key: → {"key": or ,"key":
	text = gemma4BareKeyRe.ReplaceAllString(text, `$1"$2":`)

	// Replace placeholders with JSON-escaped strings
	for idx, value := range quotedStrings {
		escaped, _ := json.Marshal(value)
		text = strings.ReplaceAll(text, gemma4Placeholder(idx), string(escaped))
	}

	return text
}

// gemma4Placeholder returns a unique placeholder string for the given index.
// Uses characters from the private use area (U+E000+) to avoid collisions
// with any content the model might generate.
func gemma4Placeholder(i int) string {
	return string(rune(0xE000+i)) + "\x00"
}

// findClosingDelim finds the position of the closing <|"|> delimiter that matches
// the opening one, accounting for nested <|"|> pairs within the value.
//
// The strategy: scan forward for <|"|> tokens. Each <|"|> could be either:
// (a) an inner opening delimiter (followed by content and another <|"|>)
// (b) the closing delimiter for our value
//
// We identify the closing delimiter as the <|"|> that is followed by a structural
// character: } ] , or end of string (after optional whitespace).
func findClosingDelim(s string, start int) int {
	const delim = "<|\"|>"

	i := start
	for i < len(s) {
		idx := strings.Index(s[i:], delim)
		if idx == -1 {
			return -1
		}

		pos := i + idx
		after := pos + len(delim)

		// Check what follows this <|"|>
		// Skip whitespace
		j := after
		for j < len(s) && (s[j] == ' ' || s[j] == '\t' || s[j] == '\n' || s[j] == '\r') {
			j++
		}

		if j >= len(s) {
			// End of string — this is the closing delimiter
			return pos
		}

		// If followed by a structural character (} ] ,) this is the closing delimiter
		switch s[j] {
		case '}', ']', ',':
			return pos
		}

		// Otherwise this is an inner <|"|> — skip past it and continue
		i = after
	}

	return -1
}
