package contextbudget

import (
	"strings"
	"testing"
	"unicode/utf8"

	"github.com/danielmiessler/fabric/internal/chat"
)

func textMsg(role, content string) *chat.ChatCompletionMessage {
	return &chat.ChatCompletionMessage{Role: role, Content: content}
}

func userMsg(content string) *chat.ChatCompletionMessage {
	return textMsg(chat.ChatMessageRoleUser, content)
}
func systemMsg(content string) *chat.ChatCompletionMessage {
	return textMsg(chat.ChatMessageRoleSystem, content)
}
func assistantMsg(content string) *chat.ChatCompletionMessage {
	return textMsg(chat.ChatMessageRoleAssistant, content)
}

func toolPair(name, args, result string) []*chat.ChatCompletionMessage {
	return []*chat.ChatCompletionMessage{
		{
			Role: chat.ChatMessageRoleAssistant,
			ToolCalls: []chat.ToolCall{{
				ID:       "call-1",
				Type:     chat.ToolTypeFunction,
				Function: chat.FunctionCall{Name: name, Arguments: args},
			}},
			Content: "",
		},
		{
			Role:       chat.ChatMessageRoleTool,
			ToolCallID: "call-1",
			Content:    result,
		},
	}
}

// TestPlan_NoTrimWhenWithinBudget: a request that fits must come out
// byte-for-byte identical, in the same order.
func TestPlan_NoTrimWhenWithinBudget(t *testing.T) {
	system := []*chat.ChatCompletionMessage{systemMsg("system instructions")}
	history := []*chat.ChatCompletionMessage{
		userMsg("first question"),
		assistantMsg("first answer"),
	}
	current := userMsg("second question")
	original := make([]*chat.ChatCompletionMessage, 0)
	original = append(original, system...)
	original = append(original, history...)
	original = append(original, current)

	plan, err := PlanMessages(system, history, current, Options{ContextWindow: 100000, MaxTokens: 1000})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if plan.DroppedMessageCount != 0 || plan.InputTruncated || plan.UnknownWindow {
		t.Fatalf("expected no trimming, got dropped=%d truncated=%v unknown=%v",
			plan.DroppedMessageCount, plan.InputTruncated, plan.UnknownWindow)
	}
	if len(plan.Messages) != len(original) {
		t.Fatalf("message count changed: got %d want %d", len(plan.Messages), len(original))
	}
	for i := range original {
		if plan.Messages[i].Role != original[i].Role || plan.Messages[i].Content != original[i].Content {
			t.Fatalf("message %d changed; within-budget path must not copy or reorder content", i)
		}
	}
	if got := joinContents(plan.Messages); got != joinContents(original) {
		t.Fatalf("content changed:\n got %q", got)
	}
}

// TestPlan_DeterministicReassembly: the same input and budget assembled twice
// must give identical results, including when history is presented in
// different construction orders.
func TestPlan_DeterministicReassembly(t *testing.T) {
	system := []*chat.ChatCompletionMessage{systemMsg("mode " + strings.Repeat("s", 200))}
	history := []*chat.ChatCompletionMessage{
		userMsg(strings.Repeat("old-question ", 40)),
		assistantMsg(strings.Repeat("old-answer ", 40)),
		userMsg(strings.Repeat("newer-question ", 40)),
		assistantMsg(strings.Repeat("newer-answer ", 40)),
	}
	current := userMsg(strings.Repeat("current-input ", 40))
	opts := Options{ContextWindow: 900, MaxTokens: 200}

	first, err := PlanMessages(system, history, current, opts)
	if err != nil {
		t.Fatalf("first plan: %v", err)
	}
	second, err := PlanMessages(system, history, current, opts)
	if err != nil {
		t.Fatalf("second plan: %v", err)
	}
	if joinContents(first.Messages) != joinContents(second.Messages) {
		t.Fatal("two plans of the same input differ")
	}
	if first.DroppedMessageCount != second.DroppedMessageCount ||
		first.EstimatedInput != second.EstimatedInput {
		t.Fatal("plan statistics differ between runs")
	}

	// Rebuild equivalent history in a fresh slice (different allocation order)
	// and require the same surviving contents.
	rebuiltHistory := []*chat.ChatCompletionMessage{
		userMsg(strings.Repeat("old-question ", 40)),
		assistantMsg(strings.Repeat("old-answer ", 40)),
		userMsg(strings.Repeat("newer-question ", 40)),
		assistantMsg(strings.Repeat("newer-answer ", 40)),
	}
	third, err := PlanMessages(system, rebuiltHistory, userMsg(strings.Repeat("current-input ", 40)), opts)
	if err != nil {
		t.Fatalf("third plan: %v", err)
	}
	if joinContents(third.Messages) != joinContents(first.Messages) {
		t.Fatal("rebuilt history assembled differently")
	}
}

// TestPlan_PriorityOrder: system first, current input last; history kept
// newest-first and output back in chronological order.
func TestPlan_PriorityOrder(t *testing.T) {
	system := []*chat.ChatCompletionMessage{systemMsg("SYS")}
	history := []*chat.ChatCompletionMessage{
		userMsg("h1 " + strings.Repeat("old ", 30)), assistantMsg("a1 " + strings.Repeat("old ", 30)),
		userMsg("h2 " + strings.Repeat("mid ", 30)), assistantMsg("a2 " + strings.Repeat("mid ", 30)),
		userMsg("h3 " + strings.Repeat("new ", 30)), assistantMsg("a3 " + strings.Repeat("new ", 30)),
	}
	current := userMsg("NOW")
	plan, err := PlanMessages(system, history, current, Options{ContextWindow: 250, MaxTokens: 60})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if plan.Messages[0].Role != chat.ChatMessageRoleSystem || plan.Messages[0].Content != "SYS" {
		t.Fatal("system message must remain first")
	}
	last := plan.Messages[len(plan.Messages)-1]
	if last.Role != chat.ChatMessageRoleUser || last.Content != "NOW" {
		t.Fatalf("current input must remain last, got %s:%q", last.Role, last.Content)
	}
	// Surviving history must stay chronological: any h2 kept must precede h3.
	contents := joinContents(plan.Messages)
	idx2, idx3 := strings.Index(contents, "h2"), strings.Index(contents, "h3")
	if idx2 >= 0 && idx3 >= 0 && idx2 > idx3 {
		t.Fatal("history order reversed: h2 appears after h3")
	}
	if plan.DroppedMessageCount == 0 {
		t.Fatal("expected some history to be dropped at this budget")
	}
	// Newest history survives preferentially.
	if !strings.Contains(contents, "h3") {
		t.Fatalf("newest history was dropped first: %q", contents)
	}
}

// TestPlan_ToolPairsKeptAtomic: a tool call and its tool result are never split.
func TestPlan_ToolPairsAtomic(t *testing.T) {
	system := []*chat.ChatCompletionMessage{systemMsg("SYS")}
	pair := toolPair("search", `{"q":"x"}`, strings.Repeat("tool-result ", 60))
	history := []*chat.ChatCompletionMessage{
		userMsg("warmup"), assistantMsg("warmup-answer"),
	}
	history = append(history, pair...)
	history = append(history, userMsg("after-tool"), assistantMsg("after-answer"))
	current := userMsg("NOW")

	plan, err := PlanMessages(system, history, current, Options{ContextWindow: 500, MaxTokens: 100})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var sawCall, sawResult bool
	for _, msg := range plan.Messages {
		if len(msg.ToolCalls) > 0 {
			sawCall = true
		}
		if msg.Role == chat.ChatMessageRoleTool {
			sawResult = true
		}
	}
	if sawCall != sawResult {
		t.Fatalf("tool pair split: call=%v result=%v", sawCall, sawResult)
	}
	if sawCall && plan.DroppedPairs != 0 {
		t.Fatalf("pair kept but DroppedPairs=%d", plan.DroppedPairs)
	}
	if !sawCall && plan.DroppedPairs != 1 {
		t.Fatalf("pair dropped but DroppedPairs=%d, want 1", plan.DroppedPairs)
	}
}

// TestPlan_UnknownWindowUsesConservativeDefault: no window means the
// conservative default plus a notice, never a blind full-size send.
func TestPlan_UnknownWindowUsesConservativeDefault(t *testing.T) {
	system := []*chat.ChatCompletionMessage{systemMsg(strings.Repeat("S", 300))}
	history := []*chat.ChatCompletionMessage{
		userMsg(strings.Repeat("history ", 2000)),
		assistantMsg(strings.Repeat("answer ", 2000)),
	}
	current := userMsg("NOW")
	plan, err := PlanMessages(system, history, current, Options{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !plan.UnknownWindow || plan.Window != DefaultContextWindow {
		t.Fatalf("expected unknown window default %d, got window=%d unknown=%v",
			DefaultContextWindow, plan.Window, plan.UnknownWindow)
	}
	if len(plan.Notices) == 0 || plan.Notices[0].Kind != NoticeUnknownWindow {
		t.Fatal("expected an unknown-window notice")
	}
	if plan.DroppedMessageCount == 0 {
		t.Fatal("large history must be trimmed under the conservative default window")
	}
}

// TestPlan_CurrentInputLargerThanWindow: the current input alone exceeds the
// window; it must be truncated on a character boundary and carry the marker.
func TestPlan_CurrentInputLargerThanWindow(t *testing.T) {
	system := []*chat.ChatCompletionMessage{systemMsg("SYS")}
	current := userMsg(strings.Repeat("contract-clause ", 2000))
	plan, err := PlanMessages(system, nil, current, Options{ContextWindow: 600, MaxTokens: 100})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !plan.InputTruncated {
		t.Fatal("expected the current input to be truncated")
	}
	last := plan.Messages[len(plan.Messages)-1]
	if !strings.Contains(last.Content, strings.TrimSpace(TruncationMarker)) {
		t.Fatalf("truncated input lacks marker: %q", last.Content)
	}
	if plan.EstimatedInput > plan.InputBudget {
		t.Fatalf("plan still over budget: estimated=%d budget=%d", plan.EstimatedInput, plan.InputBudget)
	}
	// The stored session message must not be mutated.
	if current.Content == last.Content {
		t.Fatal("truncation mutated the original message")
	}
}

// TestPlan_MultibyteTruncationOnCharacterBoundary: truncation never splits a
// multi-byte rune, including CJK and emoji, and the result stays valid UTF-8.
func TestPlan_MultibyteTruncationOnCharacterBoundary(t *testing.T) {
	multibyte := strings.Repeat("合同条款内容", 200) + "🎉" + strings.Repeat("代码块```go\nx := 1\n```\n", 50)
	truncated := TruncateText(multibyte, 120, TruncationMarker)
	if !utf8.ValidString(truncated) {
		t.Fatal("truncated text is not valid UTF-8")
	}
	if !strings.HasSuffix(truncated, strings.TrimSpace(TruncationMarker)) {
		t.Fatal("truncated text lacks the marker")
	}
	if countText(truncated) > 120 {
		t.Fatalf("truncation exceeded its token limit: %d > 120", countText(truncated))
	}
	// Every rune of the kept text must be a prefix of the original; the
	// marker (whose text begins with newlines) is appended after the cut.
	keptText := strings.TrimSuffix(truncated, TruncationMarker)
	if !strings.HasPrefix(multibyte, keptText) {
		t.Fatal("truncation cut inside the text instead of taking a prefix")
	}
}

// TestPlan_MultiContentImageMessages: image parts carry a fixed cost and are
// dropped oldest-first when the input cannot otherwise fit.
func TestPlan_MultiContentImageMessages(t *testing.T) {
	system := []*chat.ChatCompletionMessage{systemMsg("SYS")}
	imageMsg := func(url string) *chat.ChatCompletionMessage {
		return &chat.ChatCompletionMessage{
			Role: chat.ChatMessageRoleUser,
			MultiContent: []chat.ChatMessagePart{
				{Type: chat.ChatMessagePartTypeText, Text: "look at this"},
				{Type: chat.ChatMessagePartTypeImageURL, ImageURL: &chat.ChatMessageImageURL{URL: url}},
			},
		}
	}
	history := []*chat.ChatCompletionMessage{
		imageMsg("https://example.com/1.png"),
		assistantMsg("seen 1"),
		imageMsg("https://example.com/2.png"),
		assistantMsg("seen 2"),
	}
	current := userMsg("NOW")
	plan, err := PlanMessages(system, history, current, Options{ContextWindow: 1800, MaxTokens: 200})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if plan.DroppedMessageCount == 0 {
		t.Fatal("expected image history to be trimmed")
	}
	for _, msg := range plan.Messages {
		for _, part := range msg.MultiContent {
			if part.Type == chat.ChatMessagePartTypeImageURL && part.ImageURL != nil {
				if part.ImageURL.URL == "https://example.com/1.png" {
					t.Fatal("oldest image must be dropped first")
				}
			}
		}
	}
}

// TestPlan_CurrentMultiContentTruncation: an over-budget current multi-content
// message loses image parts with an explicit marker and stays valid.
func TestPlan_CurrentMultiContentTruncation(t *testing.T) {
	system := []*chat.ChatCompletionMessage{systemMsg("SYS")}
	current := &chat.ChatCompletionMessage{
		Role: chat.ChatMessageRoleUser,
		MultiContent: []chat.ChatMessagePart{
			{Type: chat.ChatMessagePartTypeText, Text: strings.Repeat("explain ", 500)},
			{Type: chat.ChatMessagePartTypeImageURL, ImageURL: &chat.ChatMessageImageURL{URL: "https://example.com/big.png"}},
		},
	}
	plan, err := PlanMessages(system, nil, current, Options{ContextWindow: 400, MaxTokens: 100})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	last := plan.Messages[len(plan.Messages)-1]
	if plan.EstimatedInput > plan.InputBudget {
		t.Fatalf("plan over budget: %d > %d", plan.EstimatedInput, plan.InputBudget)
	}
	// Original message untouched.
	if len(current.MultiContent) != 2 {
		t.Fatal("original current message was mutated")
	}
	var hasImage, hasDropNotice bool
	for _, part := range last.MultiContent {
		if part.Type == chat.ChatMessagePartTypeImageURL {
			hasImage = true
		}
		if strings.Contains(part.Text, "removed") {
			hasDropNotice = true
		}
	}
	if hasImage {
		t.Fatal("image part should have been removed")
	}
	if !hasDropNotice {
		t.Fatal("expected a text marker recording the removed image part")
	}
}

// TestPlan_RepeatedTrimmingAfterPriorTrim: trimming again on an already
// trimmed, smaller request stays idempotent and never grows.
func TestPlan_RepeatedTrimmingAfterPriorTrim(t *testing.T) {
	system := []*chat.ChatCompletionMessage{systemMsg("SYS")}
	history := []*chat.ChatCompletionMessage{
		userMsg(strings.Repeat("old ", 300)), assistantMsg(strings.Repeat("ans ", 300)),
		userMsg(strings.Repeat("new ", 300)), assistantMsg(strings.Repeat("rep ", 300)),
	}
	current := userMsg(strings.Repeat("input ", 100))
	opts := Options{ContextWindow: 700, MaxTokens: 150}

	first, err := PlanMessages(system, history, current, opts)
	if err != nil {
		t.Fatalf("first plan: %v", err)
	}
	// Feed the surviving messages back in as the "new" full state.
	var secondHistory []*chat.ChatCompletionMessage
	secondHistory = append(secondHistory, first.Messages[1:len(first.Messages)-1]...)
	second, err := PlanMessages(system, secondHistory, first.Messages[len(first.Messages)-1], opts)
	if err != nil {
		t.Fatalf("second plan: %v", err)
	}
	if second.EstimatedInput > first.EstimatedInput {
		t.Fatalf("second plan grew: %d > %d", second.EstimatedInput, first.EstimatedInput)
	}
	if joinContents(second.Messages) != joinContents(first.Messages) {
		t.Fatal("re-planning an already trimmed request changed its contents")
	}
}

// TestPlan_ExceededAfterTrimmingReportsOverflow: when even the system block
// plus the marker cannot fit, the planner errors with the overflow count
// instead of sending anything.
func TestPlan_ExceededAfterTrimmingReportsOverflow(t *testing.T) {
	system := []*chat.ChatCompletionMessage{systemMsg(strings.Repeat("system-", 500))}
	current := userMsg("hi")
	_, err := PlanMessages(system, nil, current, Options{ContextWindow: 200, MaxTokens: 50})
	if err == nil {
		t.Fatal("expected BudgetExceededError")
	}
	exceeded, ok := err.(*BudgetExceededError)
	if !ok {
		t.Fatalf("expected *BudgetExceededError, got %T: %v", err, err)
	}
	if exceeded.Overflow <= 0 {
		t.Fatalf("overflow must be positive, got %d", exceeded.Overflow)
	}
	if exceeded.InputBudget != 150 {
		t.Fatalf("input budget = %d, want 150", exceeded.InputBudget)
	}
	if !strings.Contains(err.Error(), "overflow") {
		t.Fatalf("error must explain the overflow, got %q", err.Error())
	}
}

// TestPlan_InvalidConfigurationRejected: illegal values fail the request
// explicitly rather than silently falling back.
func TestPlan_InvalidConfigurationRejected(t *testing.T) {
	cases := []struct {
		name string
		opts Options
		want string
	}{
		{"negative window", Options{ContextWindow: -1}, "contextWindow"},
		{"negative max tokens", Options{ContextWindow: 1000, MaxTokens: -5}, "maxTokens"},
		{"negative reservation pct", Options{ContextWindow: 1000, OutputReservationPct: -1}, "outputReservationPct"},
		{"pct over 100", Options{ContextWindow: 1000, OutputReservationPct: 101}, "outputReservationPct"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := PlanMessages(nil, nil, userMsg("hi"), tc.opts)
			if err == nil {
				t.Fatal("expected InvalidConfigurationError")
			}
			inv, ok := err.(*InvalidConfigurationError)
			if !ok {
				t.Fatalf("expected *InvalidConfigurationError, got %T", err)
			}
			if inv.Option != tc.want {
				t.Fatalf("option = %q, want %q", inv.Option, tc.want)
			}
		})
	}
}

// TestPlan_OutputReservationPrecedence: an explicit MaxTokens is the output
// quota; without one, the clamped default applies.
func TestPlan_OutputReservationPrecedence(t *testing.T) {
	withMax, err := PlanMessages(nil, nil, userMsg("hi"), Options{ContextWindow: 10000, MaxTokens: 3000})
	if err != nil {
		t.Fatal(err)
	}
	if withMax.OutputReservation != 3000 || withMax.InputBudget != 7000 {
		t.Fatalf("explicit max tokens not respected: reserve=%d budget=%d",
			withMax.OutputReservation, withMax.InputBudget)
	}

	def, err := PlanMessages(nil, nil, userMsg("hi"), Options{ContextWindow: 10000})
	if err != nil {
		t.Fatal(err)
	}
	want := 10000 * DefaultOutputReservationPct / 100
	if def.OutputReservation != want {
		t.Fatalf("default reservation = %d, want %d", def.OutputReservation, want)
	}

	small, err := PlanMessages(nil, nil, userMsg("hi"), Options{ContextWindow: 1000})
	if err != nil {
		t.Fatal(err)
	}
	if small.OutputReservation != DefaultOutputReservationMin {
		t.Fatalf("small window reservation = %d, want clamp %d",
			small.OutputReservation, DefaultOutputReservationMin)
	}
}

// TestPlan_WindowSmallerThanRealConfigured: a misconfigured window that is too
// small triggers explicit trimming/errors rather than an upstream failure.
func TestPlan_WindowSmallerThanRealConfigured(t *testing.T) {
	system := []*chat.ChatCompletionMessage{systemMsg("SYS")}
	history := []*chat.ChatCompletionMessage{userMsg("h"), assistantMsg("a")}
	// The configured window is far smaller than the messages' real need.
	plan, err := PlanMessages(system, history, userMsg(strings.Repeat("x ", 200)),
		Options{ContextWindow: 220, MaxTokens: 60})
	if err != nil {
		t.Fatalf("planner should trim before erroring: %v", err)
	}
	if plan.InputTruncated || plan.DroppedMessageCount > 0 {
		// At least one trimming mechanism engaged, which is the point: the
		// request is reshaped locally instead of being rejected upstream.
		return
	}
	t.Fatal("expected trimming with an undersized configured window")
}

// TestPlan_WindowLargerThanRealConfigured: an oversized configured window
// simply leaves the request untouched (no artificial trimming).
func TestPlan_WindowLargerThanRealConfigured(t *testing.T) {
	system := []*chat.ChatCompletionMessage{systemMsg("SYS")}
	current := userMsg("small")
	plan, err := PlanMessages(system, nil, current, Options{ContextWindow: 200000, MaxTokens: 4096})
	if err != nil {
		t.Fatal(err)
	}
	if plan.DroppedMessageCount != 0 || plan.InputTruncated {
		t.Fatal("an oversized window must not trigger trimming")
	}
}

// TestTruncateText_CharacterBoundaries checks plain ASCII word runs and edge
// cases directly.
func TestTruncateText_CharacterBoundaries(t *testing.T) {
	if got := TruncateText("short", 100, "M"); got != "short" {
		t.Fatalf("unaffected text changed: %q", got)
	}
	got := TruncateText(strings.Repeat("abcdefgh ", 100), 20, "…")
	if !utf8.ValidString(got) {
		t.Fatal("invalid UTF-8 after truncation")
	}
	if !strings.HasSuffix(got, "…") {
		t.Fatalf("marker missing: %q", got)
	}
	if countText(got) > 20 {
		t.Fatalf("limit broken: %d > 20", countText(got))
	}
	if TruncateText("abc", 0, "M") != "M" {
		t.Fatal("zero limit should return the marker alone")
	}
}

// TestMessageTokens_StructuralParts locks the estimator inputs: image parts
// cost the fixed reserve, text is counted, framing is included.
func TestMessageTokens_StructuralParts(t *testing.T) {
	plain := MessageTokens(userMsg("hello world"))
	if plain <= OverheadTokens {
		t.Fatalf("plain text message must cost more than framing: %d", plain)
	}
	image := MessageTokens(&chat.ChatCompletionMessage{
		Role: chat.ChatMessageRoleUser,
		MultiContent: []chat.ChatMessagePart{
			{Type: chat.ChatMessagePartTypeImageURL, ImageURL: &chat.ChatMessageImageURL{URL: "u"}},
		},
	})
	if image < ImageTokens+OverheadTokens {
		t.Fatalf("image message under-reserved: %d", image)
	}
	if MessageTokens(nil) != 0 {
		t.Fatal("nil message must cost zero")
	}
	// Determinism: same input, same count.
	if MessageTokens(userMsg(strings.Repeat("token ", 100))) != MessageTokens(userMsg(strings.Repeat("token ", 100))) {
		t.Fatal("estimator is not deterministic")
	}
}

// TestGroupHistory_ToolPairBinding covers the grouping invariant directly,
// including a tool result whose call is missing in malformed history.
func TestGroupHistory_ToolPairBinding(t *testing.T) {
	pair := toolPair("f", "{}", "result")
	history := append(pair, userMsg("later"))
	groups := groupHistory(history)
	if len(groups) != 2 {
		t.Fatalf("expected 2 groups, got %d", len(groups))
	}
	if !groups[0].isToolPair || len(groups[0].messages) != 2 {
		t.Fatal("tool call and result not bound into one group")
	}

	// Orphan tool result becomes its own droppable group.
	orphan := groupHistory([]*chat.ChatCompletionMessage{
		{Role: chat.ChatMessageRoleTool, ToolCallID: "missing", Content: "x"},
	})
	if len(orphan) != 1 || orphan[0].isToolPair {
		t.Fatal("orphan tool result must be a plain single-message group")
	}
}

func joinContents(msgs []*chat.ChatCompletionMessage) string {
	var b strings.Builder
	for _, msg := range msgs {
		if msg == nil {
			continue
		}
		b.WriteString(msg.Role)
		b.WriteString(":")
		b.WriteString(msg.Content)
		for _, part := range msg.MultiContent {
			b.WriteString("|")
			b.WriteString(string(part.Type))
			b.WriteString(":")
			b.WriteString(part.Text)
			if part.ImageURL != nil {
				b.WriteString(part.ImageURL.URL)
			}
		}
		b.WriteString("\n")
	}
	return b.String()
}
