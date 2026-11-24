// Package contextbudget assembles the messages that one request sends to a
// model inside a token budget.
//
// The budget comes from the configured model context window. The output quota
// is reserved first, because the model needs room to answer. What remains is
// the input budget, shared in this fixed priority:
//
//  1. mode and system messages (never dropped);
//  2. the current user input (dropped only when one message alone cannot fit,
//     and then truncated on a character boundary with a marker);
//  3. the conversation history, newest first, tool calls and tool results
//     always kept or dropped as a pair.
//
// The estimator is a deterministic heuristic: no network, no randomness, so
// the same input and budget always assemble to the same result.
package contextbudget

import (
	"github.com/danielmiessler/fabric/internal/chat"
)

// OverheadTokens covers the per-request and per-message framing that every
// provider adds on top of the visible text (role tags, field names,
// separators). It is part of the deterministic estimate.
const OverheadTokens = 4

// MessageTokens estimates the token cost of one message. It counts visible
// text only: role names, tool call ids and image URLs are structural framing
// and are covered by OverheadTokens, except that an image message is never
// free, so image parts each cost ImageTokens.
//
// The estimate is deliberately simple and stable:
//   - CJK and other wide code points count as one token each;
//   - runs of word characters count as ceil(run length / 4);
//   - every other run of punctuation/whitespace counts as one token.
func MessageTokens(message *chat.ChatCompletionMessage) int {
	if message == nil {
		return 0
	}
	tokens := 0
	tokens += countText(message.Content)
	tokens += countText(message.Refusal)
	tokens += countText(message.ReasoningContent)
	for _, call := range message.ToolCalls {
		tokens += countText(call.Function.Name)
		tokens += countText(call.Function.Arguments)
	}
	for _, part := range message.MultiContent {
		switch part.Type {
		case chat.ChatMessagePartTypeText:
			tokens += countText(part.Text)
		case chat.ChatMessagePartTypeImageURL:
			// Images cannot be measured in text tokens. Charge a fixed,
			// conservative reserve so a history of images is trimmed like
			// any other large message.
			tokens += ImageTokens
			if part.ImageURL != nil {
				tokens += countText(part.ImageURL.URL)
			}
		}
	}
	return tokens + OverheadTokens
}

// ImageTokens is the fixed reserve for one image part.
const ImageTokens = 1000

// countText returns the estimated token count of s using run classes, so the
// estimate is identical on every platform and does not depend on a tokenizer.
func countText(s string) int {
	if s == "" {
		return 0
	}
	tokens := 0
	var wordRun int
	flushWord := func() {
		if wordRun > 0 {
			tokens += (wordRun + 3) / 4
			wordRun = 0
		}
	}
	for _, r := range s {
		switch {
		case isWordRune(r):
			wordRun++
		case isWideRune(r):
			flushWord()
			tokens++
		default:
			flushWord()
			tokens++
		}
	}
	flushWord()
	return tokens
}

func isWordRune(r rune) bool {
	return r == '_' ||
		(r >= 'a' && r <= 'z') ||
		(r >= 'A' && r <= 'Z') ||
		(r >= '0' && r <= '9')
}

// isWideRune covers the scripts whose tokenizers tend to emit about one token
// per code point: CJK ideographs, kana, hangul and full-width forms.
func isWideRune(r rune) bool {
	return (r >= 0x1100 && r <= 0x11FF) || // Hangul Jamo
		(r >= 0x2E80 && r <= 0x9FFF) || // CJK radicals, ideographs, kana
		(r >= 0xAC00 && r <= 0xD7AF) || // Hangul syllables
		(r >= 0xF900 && r <= 0xFAFF) || // CJK compatibility ideographs
		(r >= 0xFF00 && r <= 0xFFEF) || // full-width forms
		(r >= 0x3040 && r <= 0x30FF) // hiragana + katakana (also inside 2E80-9FFF on some tables)
}

// TruncateText shortens s to at most limit estimated tokens, cutting on a
// Unicode character boundary (never inside a multi-byte rune) and appending
// marker when anything was removed. The marker is counted inside the limit,
// so the result always costs at most limit tokens. The text portion is always
// a prefix of s, so no content can change position.
//
// limit <= 0 returns the marker alone when s is non-empty.
func TruncateText(s string, limit int, marker string) string {
	if countText(s) <= limit {
		return s
	}
	if limit <= 0 {
		return marker
	}
	markerCost := countText(marker)
	budget := limit - markerCost
	if budget <= 0 {
		// The marker alone already fills the budget; keep it, it is the
		// signal that content was removed.
		return marker
	}

	runes := []rune(s)
	// Binary search for the longest prefix whose own estimated cost fits.
	// The estimator is the single source of truth, so the result cannot
	// disagree with it by a boundary token. Search over rune counts, which
	// makes every candidate a valid character boundary.
	lo, hi := 0, len(runes)
	for lo < hi {
		mid := (lo + hi + 1) / 2
		if countText(string(runes[:mid])) <= budget {
			lo = mid
		} else {
			hi = mid - 1
		}
	}
	return string(runes[:lo]) + marker
}
