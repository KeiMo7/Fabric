package contextbudget

import (
	"errors"
	"fmt"
	"strings"

	"github.com/danielmiessler/fabric/internal/chat"
	"github.com/danielmiessler/fabric/internal/i18n"
)

// TruncationMarker is appended to a user message that the planner shortened.
// It tells the reader exactly where content disappeared.
const TruncationMarker = "\n\n[... fabric: input truncated to fit the model context window ...]"

// DefaultContextWindow is the conservative window used when the vendor gives
// none. Most small local models and the cheapest hosted tier sit at or above
// 8k, so budgeting against 8k rather than sending blind avoids the upstream
// "context length exceeded" failure without assuming a large window.
const DefaultContextWindow = 8192

// DefaultOutputReservation is the output quota reserved when the caller does
// not configure MaxTokens: 10% of the window, clamped to [256, 2048].
const (
	DefaultOutputReservationMin = 256
	DefaultOutputReservationMax = 2048
	DefaultOutputReservationPct = 10
)

// ErrInvalidConfiguration means a configured budget value is illegal. The
// request is rejected; the caller must not silently fall back to another value.
var ErrInvalidConfiguration = errors.New("invalid context budget configuration")

// ErrBudgetExceeded means the request still does not fit after every legal
// trim. It carries the concrete overflow so the user knows how much to remove.
var ErrBudgetExceeded = errors.New("context budget exceeded")

// BudgetExceededError reports a request that cannot fit the window. Overflow
// is the estimated token count that remains after all trimming, measured
// against InputBudget.
type BudgetExceededError struct {
	Window       int
	InputBudget  int
	ReservedOut  int
	EstimatedIn  int
	Overflow     int
	DroppedPairs int
}

func (e *BudgetExceededError) Error() string {
	return fmt.Sprintf(i18n.T("context_budget_error_exceeded"),
		e.Window, e.InputBudget, e.EstimatedIn, e.Overflow, e.DroppedPairs)
}

func (*BudgetExceededError) Is(target error) bool { return target == ErrBudgetExceeded }

// InvalidConfigurationError names the offending option and its value.
type InvalidConfigurationError struct {
	Option string
	Value  int
}

func (e *InvalidConfigurationError) Error() string {
	return fmt.Sprintf(i18n.T("context_budget_error_invalid_config"), e.Option, e.Value)
}

func (*InvalidConfigurationError) Is(target error) bool { return target == ErrInvalidConfiguration }

// Options configures a Plan call.
type Options struct {
	// ContextWindow is the model's total context window in tokens. Zero means
	// unknown: the planner then uses DefaultContextWindow and flags the plan.
	ContextWindow int
	// MaxTokens is the configured output limit. Zero means "reserve a
	// conservative default instead".
	MaxTokens int
	// OutputReservationPct overrides DefaultOutputReservationPct for the
	// default reserve. Zero keeps the default. Mainly used by tests.
	OutputReservationPct int
}

// Notice is a human-readable fact about the plan that the caller should show:
// an unknown window, a truncated input, or dropped history. Notices never
// change the assembled bytes; they only explain them.
type Notice struct {
	Kind    NoticeKind
	Message string
}

type NoticeKind string

const (
	NoticeUnknownWindow NoticeKind = "unknown_window"
	NoticeTruncated     NoticeKind = "truncated_input"
	NoticeDropped       NoticeKind = "dropped_history"
)

// Plan is the deterministic result of budgeting one request.
type Plan struct {
	// Messages is exactly what should be sent to the vendor: system first,
	// then the surviving history (original order), then the current input.
	Messages []*chat.ChatCompletionMessage
	// Window is the total window the plan used (a configured value or the
	// conservative default).
	Window int
	// OutputReservation is the token quota held for the model's answer.
	OutputReservation int
	// InputBudget is Window - OutputReservation.
	InputBudget int
	// EstimatedInput is the estimated token cost of Messages.
	EstimatedInput int
	// UnknownWindow is true when no vendor/configured window was available.
	UnknownWindow bool
	// DroppedMessageCount counts every history message removed, including
	// both halves of a tool pair.
	DroppedMessageCount int
	// DroppedPairs counts tool call/result pairs removed together.
	DroppedPairs int
	// InputTruncated is true when the current user input was shortened.
	InputTruncated bool
	Notices        []Notice
}

// group is one history unit. A plain message is a group of one; a tool call
// (assistant message carrying ToolCalls) plus every tool result that answers
// it forms one group, so the pair is always kept or dropped together.
type group struct {
	messages []*chat.ChatCompletionMessage
	cost     int
	// isToolPair marks a group that contains a tool call and its result(s).
	isToolPair bool
}

// PlanMessages budgets one request.
//
// system holds mode and system instructions (including developer messages).
// history holds prior turns in chronological order. current is the new user
// input, possibly a multi-content message. None of the input slices or
// messages are modified: the plan only references or copies them.
func PlanMessages(system, history []*chat.ChatCompletionMessage, current *chat.ChatCompletionMessage, opts Options) (*Plan, error) {
	if err := validateOptions(opts); err != nil {
		return nil, err
	}

	window := opts.ContextWindow
	unknown := false
	if window == 0 {
		window = DefaultContextWindow
		unknown = true
	}

	reservation := outputReservation(window, opts)
	if reservation >= window {
		return nil, &BudgetExceededError{
			Window:      window,
			InputBudget: 0,
			ReservedOut: reservation,
			// The output reservation alone fills the window; no input can fit.
			EstimatedIn:  MessageTokens(current),
			Overflow:     MessageTokens(current),
			DroppedPairs: 0,
		}
	}
	inputBudget := window - reservation

	plan := &Plan{
		Window:            window,
		OutputReservation: reservation,
		InputBudget:       inputBudget,
		UnknownWindow:     unknown,
	}
	if unknown {
		plan.Notices = append(plan.Notices, Notice{
			Kind:    NoticeUnknownWindow,
			Message: fmt.Sprintf(i18n.T("context_budget_notice_unknown_window"), DefaultContextWindow),
		})
	}

	systemCost := 0
	for _, msg := range system {
		systemCost += MessageTokens(msg)
	}

	// Priority 1: system/mode messages always stay. If they alone exceed the
	// input budget, no trim of history or the user input can fix it, so fail
	// with a concrete overflow.
	if systemCost > inputBudget {
		plan.EstimatedInput = systemCost
		return nil, plan.exceeded(systemCost-inputBudget, 0)
	}

	// Group the history so tool calls and tool results move as one unit.
	groups := groupHistory(history)

	// Priority 2: reserve room for the current input. It may itself be
	// truncated, but only after history has made as much room as possible.
	currentCopy := cloneMessage(current)
	currentCost := MessageTokens(currentCopy)

	// First pass: keep the newest groups that fit alongside the full current
	// input and the system block.
	remaining := inputBudget - systemCost
	keptGroups, droppedGroups := selectGroups(groups, remaining-currentCost)

	// Second pass: if even with all history dropped the current input does not
	// fit, shorten the current input on a character boundary.
	keptCost := systemCost + costOfGroups(keptGroups)
	if keptCost+currentCost > inputBudget {
		// This is the whole-message budget for the current input;
		// truncateCurrent deducts the per-message framing itself.
		availableForInput := inputBudget - keptCost
		currentCopy = truncateCurrent(currentCopy, availableForInput)
		currentCost = MessageTokens(currentCopy)
		plan.InputTruncated = true
		plan.Notices = append(plan.Notices, Notice{
			Kind:    NoticeTruncated,
			Message: fmt.Sprintf(i18n.T("context_budget_notice_input_truncated"), availableForInput),
		})
	}

	// A pathological case: the marker and framing alone still do not fit.
	if keptCost+currentCost > inputBudget {
		plan.DroppedMessageCount = countGroupMessages(droppedGroups)
		plan.DroppedPairs = countToolPairs(droppedGroups)
		plan.EstimatedInput = keptCost + currentCost
		return nil, plan.exceeded(keptCost+currentCost-inputBudget, plan.DroppedPairs)
	}

	plan.Messages = assemble(system, keptGroups, currentCopy)
	plan.DroppedMessageCount = countGroupMessages(droppedGroups)
	plan.DroppedPairs = countToolPairs(droppedGroups)
	if plan.DroppedMessageCount > 0 {
		plan.Notices = append(plan.Notices, Notice{
			Kind: NoticeDropped,
			Message: fmt.Sprintf(i18n.T("context_budget_notice_dropped"),
				plan.DroppedMessageCount, plan.DroppedPairs),
		})
	}
	plan.EstimatedInput = systemCost + costOfGroups(keptGroups) + currentCost
	return plan, nil
}

func validateOptions(opts Options) error {
	if opts.ContextWindow < 0 {
		return &InvalidConfigurationError{Option: "contextWindow", Value: opts.ContextWindow}
	}
	if opts.MaxTokens < 0 {
		return &InvalidConfigurationError{Option: "maxTokens", Value: opts.MaxTokens}
	}
	if opts.OutputReservationPct < 0 || opts.OutputReservationPct > 100 {
		return &InvalidConfigurationError{Option: "outputReservationPct", Value: opts.OutputReservationPct}
	}
	return nil
}

func outputReservation(window int, opts Options) int {
	if opts.MaxTokens > 0 {
		// A configured output limit is the real reservation. If it claims the
		// whole window, the caller gets the explicit exceeded error below
		// rather than a silent clamp.
		return opts.MaxTokens
	}
	pct := opts.OutputReservationPct
	if pct == 0 {
		pct = DefaultOutputReservationPct
	}
	r := window * pct / 100
	r = max(r, DefaultOutputReservationMin)
	r = min(r, DefaultOutputReservationMax)
	return min(r, window)
}

// groupHistory walks the chronological history and binds each assistant tool
// call to the tool result messages that follow it. A result without a call
// (malformed history) becomes its own single-message group and is droppable,
// but a call is never kept without its results.
func groupHistory(history []*chat.ChatCompletionMessage) []group {
	var groups []group
	i := 0
	for i < len(history) {
		msg := history[i]
		if msg != nil && len(msg.ToolCalls) > 0 {
			g := group{isToolPair: true, messages: []*chat.ChatCompletionMessage{msg}}
			cost := MessageTokens(msg)
			i++
			// Collect every consecutive tool result that answers this call.
			for i < len(history) && history[i] != nil && history[i].Role == chat.ChatMessageRoleTool {
				g.messages = append(g.messages, history[i])
				cost += MessageTokens(history[i])
				i++
			}
			g.cost = cost
			groups = append(groups, g)
			continue
		}
		groups = append(groups, group{messages: []*chat.ChatCompletionMessage{msg}, cost: MessageTokens(msg)})
		i++
	}
	return groups
}

// selectGroups keeps a contiguous tail of the history: it walks newest to
// oldest and stops at the first group that does not fit the remaining budget.
// The tail is returned in chronological order, every dropped group before it.
//
// Contiguity is deliberate: keeping a newer assistant answer while dropping its
// user question, or keeping one half of a user/assistant turn, produces role
// adjacency that several providers reject. Tool call/result groups are already
// atomic from groupHistory, so the pair rule and the turn rule share one
// mechanism.
//
// The current request always ends with a user message, so the kept tail must
// start with a user message too; an assistant-only tail would make two user
// messages frame one answer. The boundary is therefore walked forward until a
// user-led group after any trimming.
func selectGroups(groups []group, budget int) (kept, dropped []group) {
	if budget < 0 {
		budget = 0
	}
	used := 0
	cutoff := len(groups) // index of the oldest surviving group
	for i := len(groups) - 1; i >= 0; i-- {
		if used+groups[i].cost > budget {
			break
		}
		used += groups[i].cost
		cutoff = i
	}
	// Correct a tail that begins mid-turn: advance past leading assistant
	// (or tool) groups until a user-led group opens the history.
	for cutoff < len(groups) && !groupLeadsWithUser(groups[cutoff]) {
		cutoff++
	}
	dropped = groups[:cutoff]
	kept = groups[cutoff:]
	return kept, dropped
}

// groupLeadsWithUser reports whether a group starts with a user message. A
// tool-pair group starts with an assistant tool call, so it cannot open a
// kept tail on its own.
func groupLeadsWithUser(g group) bool {
	return len(g.messages) > 0 && g.messages[0] != nil && g.messages[0].Role == chat.ChatMessageRoleUser
}

func costOfGroups(groups []group) int {
	total := 0
	for _, g := range groups {
		total += g.cost
	}
	return total
}

func countGroupMessages(groups []group) int {
	n := 0
	for _, g := range groups {
		n += len(g.messages)
	}
	return n
}

func countToolPairs(groups []group) int {
	n := 0
	for _, g := range groups {
		if g.isToolPair {
			n++
		}
	}
	return n
}

func assemble(system []*chat.ChatCompletionMessage, kept []group, current *chat.ChatCompletionMessage) []*chat.ChatCompletionMessage {
	out := make([]*chat.ChatCompletionMessage, 0, len(system)+countGroupMessages(kept)+1)
	out = append(out, system...)
	for _, g := range kept {
		out = append(out, g.messages...)
	}
	if current != nil {
		out = append(out, current)
	}
	return out
}

func (p *Plan) exceeded(overflow, droppedPairs int) *BudgetExceededError {
	return &BudgetExceededError{
		Window:       p.Window,
		InputBudget:  p.InputBudget,
		ReservedOut:  p.OutputReservation,
		EstimatedIn:  p.EstimatedInput,
		Overflow:     overflow,
		DroppedPairs: droppedPairs,
	}
}

// cloneMessage returns a shallow copy with a fresh MultiContent slice, so
// truncation never mutates the session's stored message.
func cloneMessage(msg *chat.ChatCompletionMessage) *chat.ChatCompletionMessage {
	if msg == nil {
		return nil
	}
	cp := *msg
	if msg.MultiContent != nil {
		cp.MultiContent = append([]chat.ChatMessagePart(nil), msg.MultiContent...)
	}
	if msg.ToolCalls != nil {
		cp.ToolCalls = append([]chat.ToolCall(nil), msg.ToolCalls...)
	}
	return &cp
}

// truncateCurrent shortens the current message so its full estimated cost
// (framing included) fits budget tokens. Plain text messages are cut with the
// token-aware TruncateText, which stays on character boundaries.
// Multi-content messages are shortened in two deterministic passes: text
// parts are truncated first, then image parts (the largest fixed-cost units)
// are removed oldest first, each removal recorded by a text marker.
func truncateCurrent(msg *chat.ChatCompletionMessage, budget int) *chat.ChatCompletionMessage {
	if budget <= 0 {
		return &chat.ChatCompletionMessage{
			Role:    chat.ChatMessageRoleUser,
			Content: strings.TrimSpace(TruncationMarker),
		}
	}
	if len(msg.MultiContent) == 0 {
		cp := cloneMessage(msg)
		// MessageTokens adds OverheadTokens on top of the text, so the text
		// itself may only spend budget minus framing.
		cp.Content = TruncateText(msg.Content, budget-OverheadTokens, TruncationMarker)
		return cp
	}

	cp := cloneMessage(msg)
	// Pass 1: shrink every text part against the whole-message budget. The
	// marker cost is deducted up front so a truncated part cannot overshoot.
	textBudget := budget - OverheadTokens - countText(TruncationMarker)
	for i := range cp.MultiContent {
		if cp.MultiContent[i].Type == chat.ChatMessagePartTypeText {
			cp.MultiContent[i].Text = TruncateText(cp.MultiContent[i].Text, textBudget, TruncationMarker)
		}
	}
	// Pass 2: drop image parts, oldest first, until the whole message fits.
	// A marker text part records how many images disappeared.
	removedImages := 0
	for MessageTokens(cp) > budget {
		dropped := false
		for i := range cp.MultiContent {
			if cp.MultiContent[i].Type == chat.ChatMessagePartTypeImageURL {
				cp.MultiContent = append(cp.MultiContent[:i], cp.MultiContent[i+1:]...)
				removedImages++
				dropped = true
				break
			}
		}
		if !dropped {
			break
		}
	}
	if removedImages > 0 {
		cp.MultiContent = append(cp.MultiContent, chat.ChatMessagePart{
			Type: chat.ChatMessagePartTypeText,
			Text: fmt.Sprintf(i18n.T("context_budget_notice_images_dropped"), removedImages),
		})
	}
	return cp
}
