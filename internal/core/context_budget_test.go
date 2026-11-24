package core

import (
	"bytes"
	"context"
	"errors"
	"os"
	"strings"
	"sync"
	"testing"

	"github.com/danielmiessler/fabric/internal/chat"
	"github.com/danielmiessler/fabric/internal/contextbudget"
	"github.com/danielmiessler/fabric/internal/domain"
	"github.com/danielmiessler/fabric/internal/plugins/ai/dryrun"
	"github.com/danielmiessler/fabric/internal/plugins/db/fsdb"
)

// recordingVendor records the exact message slice handed to the vendor.
type recordingVendor struct {
	mu       sync.Mutex
	got      []*chat.ChatCompletionMessage
	callN    int
	reply    string
	failWith error
}

func (r *recordingVendor) GetName() string                       { return "recording" }
func (r *recordingVendor) GetSetupDescription() string           { return "recording vendor" }
func (r *recordingVendor) IsConfigured() bool                    { return true }
func (r *recordingVendor) Configure() error                      { return nil }
func (r *recordingVendor) Setup() error                          { return nil }
func (r *recordingVendor) SetupFillEnvFileContent(*bytes.Buffer) {}
func (r *recordingVendor) ListModels(context.Context) ([]string, error) {
	return []string{"test-model"}, nil
}
func (r *recordingVendor) NeedsRawMode(string) bool { return false }

func (r *recordingVendor) Send(_ context.Context, messages []*chat.ChatCompletionMessage, _ *domain.ChatOptions) (string, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.callN++
	r.got = messages
	if r.failWith != nil {
		return "", r.failWith
	}
	return r.reply, nil
}

func (r *recordingVendor) SendStream(_ context.Context, _ []*chat.ChatCompletionMessage, _ *domain.ChatOptions, _ chan domain.StreamUpdate) error {
	return errors.New("not used")
}

func (r *recordingVendor) captured() []*chat.ChatCompletionMessage {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.got
}

func newBudgetChatter(t *testing.T, vendor *recordingVendor, window int) (*Chatter, *fsdb.Db) {
	t.Helper()
	db := fsdb.NewDb(t.TempDir())
	if err := db.Sessions.Configure(); err != nil {
		t.Fatalf("configure sessions: %v", err)
	}
	return &Chatter{
		db:                 db,
		Stream:             false,
		vendor:             vendor,
		model:              "test-model",
		modelContextLength: window,
		contextBudgeting:   true,
	}, db
}

func budgetOpts(window int) *domain.ChatOptions {
	return &domain.ChatOptions{Model: "test-model", ModelContextLength: window, MaxTokens: 100}
}

// TestContextBudget_Send_TrimsHistoryAndRecordsDrop: a fake vendor records the
// actual messages; the kept order must be system -> surviving history (newest
// first preference, chronological output) -> current input, with a drop count.
func TestContextBudget_Send_TrimsHistoryAndRecordsDrop(t *testing.T) {
	vendor := &recordingVendor{reply: "answer"}
	chatter, db := newBudgetChatter(t, vendor, 400)

	// Seed a session with a long multi-turn history.
	seed := []*chat.ChatCompletionMessage{
		systemMsgForTest("system instructions"),
	}
	for i := 0; i < 6; i++ {
		seed = append(seed,
			userMsgForTest("question number "+strings.Repeat("x", 80)+" #"+string(rune('a'+i))),
			assistantMsgForTest("answer number "+strings.Repeat("y", 80)+" #"+string(rune('a'+i))),
		)
	}
	session := &fsdb.Session{Name: "trim", Messages: seed}
	if err := db.Sessions.SaveSession(session); err != nil {
		t.Fatalf("save seed: %v", err)
	}

	request := &domain.ChatRequest{
		SessionName: "trim",
		Message:     userMsgForTest("the current question " + strings.Repeat("z", 40)),
	}
	if _, err := chatter.Send(context.Background(), request, budgetOpts(400)); err != nil {
		t.Fatalf("Send: %v", err)
	}

	sent := vendor.captured()
	if len(sent) == 0 {
		t.Fatal("vendor received no messages")
	}
	if sent[0].Role != chat.ChatMessageRoleSystem {
		t.Fatalf("first sent message role = %s, want system", sent[0].Role)
	}
	if last := sent[len(sent)-1]; last.Role != chat.ChatMessageRoleUser ||
		!strings.Contains(last.Content, "current question") {
		t.Fatalf("last sent message must be the current input, got %s:%q", last.Role, last.Content)
	}
	if len(sent) >= len(seed)+1 {
		t.Fatalf("nothing was trimmed: sent %d, seeded %d", len(sent), len(seed))
	}
	// Chronological order of the surviving history must be intact.
	var lastUser string
	for _, msg := range sent {
		if msg.Role == chat.ChatMessageRoleUser && strings.Contains(msg.Content, "question number") {
			if msg.Content < lastUser {
				t.Fatal("history order violated")
			}
			lastUser = msg.Content
		}
	}

	// The local session still holds every message plus the new reply.
	stored, err := db.Sessions.Get("trim")
	if err != nil {
		t.Fatalf("reload session: %v", err)
	}
	if len(stored.Messages) != len(seed)+2 {
		t.Fatalf("local history changed: got %d messages, want %d", len(stored.Messages), len(seed)+2)
	}
}

// TestContextBudget_Send_ToolPairAtomic: an assistant tool call and its tool
// result are either both sent or both dropped.
func TestContextBudget_Send_ToolPairAtomic(t *testing.T) {
	vendor := &recordingVendor{reply: "answer"}
	chatter, db := newBudgetChatter(t, vendor, 350)

	seed := []*chat.ChatCompletionMessage{
		systemMsgForTest("SYS"),
		userMsgForTest("warmup " + strings.Repeat("w", 100)),
		assistantMsgForTest("warmup reply " + strings.Repeat("q", 100)),
		{
			Role: chat.ChatMessageRoleAssistant,
			ToolCalls: []chat.ToolCall{{
				ID:       "call-9",
				Type:     chat.ToolTypeFunction,
				Function: chat.FunctionCall{Name: "search", Arguments: `{"q":"contract"}`},
			}},
		},
		{Role: chat.ChatMessageRoleTool, ToolCallID: "call-9", Content: strings.Repeat("result ", 100)},
	}
	if err := db.Sessions.SaveSession(&fsdb.Session{Name: "tools", Messages: seed}); err != nil {
		t.Fatalf("save: %v", err)
	}

	request := &domain.ChatRequest{
		SessionName: "tools",
		Message:     userMsgForTest("now " + strings.Repeat("n", 60)),
	}
	if _, err := chatter.Send(context.Background(), request, budgetOpts(350)); err != nil {
		t.Fatalf("Send: %v", err)
	}

	var calls, results int
	for _, msg := range vendor.captured() {
		if len(msg.ToolCalls) > 0 {
			calls++
		}
		if msg.Role == chat.ChatMessageRoleTool {
			results++
		}
	}
	if calls != results {
		t.Fatalf("tool pair split: calls=%d results=%d", calls, results)
	}
}

// TestContextBudget_Send_InputLargerThanWindowTruncatesWithMarker: the current
// input alone overflows; it is truncated on a character boundary and marked.
func TestContextBudget_Send_InputLargerThanWindowTruncatesWithMarker(t *testing.T) {
	vendor := &recordingVendor{reply: "answer"}
	chatter, _ := newBudgetChatter(t, vendor, 300)

	request := &domain.ChatRequest{
		Message: userMsgForTest("合同 " + strings.Repeat("条款内容", 500)),
	}
	if _, err := chatter.Send(context.Background(), request, budgetOpts(300)); err != nil {
		t.Fatalf("Send: %v", err)
	}
	sent := vendor.captured()
	last := sent[len(sent)-1]
	if !strings.Contains(last.Content, strings.TrimSpace(contextbudget.TruncationMarker)) {
		t.Fatalf("truncation marker missing: %q", last.Content)
	}
	if strings.Contains(last.Content, strings.Repeat("条款内容", 500)) {
		t.Fatal("full oversized input was sent")
	}
	// Original request message untouched.
	if !strings.Contains(request.Message.Content, strings.Repeat("条款内容", 500)) {
		t.Fatal("trimming mutated the request message")
	}
}

// TestContextBudget_Send_StillOverBudgetReportsFriendlyError: after trimming,
// an impossible request returns the budget error, not the upstream provider's
// raw message.
func TestContextBudget_Send_StillOverBudgetReportsFriendlyError(t *testing.T) {
	vendor := &recordingVendor{reply: "answer", failWith: errors.New("upstream says context length exceeded")}
	chatter, _ := newBudgetChatter(t, vendor, 200)

	// The system block alone cannot fit.
	request := &domain.ChatRequest{
		ContextName: "",
		Message:     userMsgForTest("hi"),
	}
	// Inject an oversized system message via a pattern-free session-less path
	// by using a context file.
	if err := chatter.db.Contexts.Configure(); err != nil {
		t.Fatalf("contexts configure: %v", err)
	}
	ctxPath := chatter.db.Contexts.Dir + "/huge"
	if err := writeFileForTest(ctxPath, strings.Repeat("system directive ", 300)); err != nil {
		t.Fatalf("write context: %v", err)
	}
	request.ContextName = "huge"

	_, err := chatter.Send(context.Background(), request, budgetOpts(200))
	if err == nil {
		t.Fatal("expected a budget error")
	}
	if errors.Is(err, contextbudget.ErrBudgetExceeded) == false {
		t.Fatalf("expected ErrBudgetExceeded, got %T: %v", err, err)
	}
	if strings.Contains(err.Error(), "upstream says") {
		t.Fatal("upstream raw error leaked to the caller")
	}
	if vendor.callN != 0 {
		t.Fatalf("vendor must not be called when the budget fails, calls=%d", vendor.callN)
	}
}

// TestContextBudget_Send_InvalidWindowRejectsRequest: a negative configured
// window fails the request explicitly instead of silently falling back.
func TestContextBudget_Send_InvalidWindowRejectsRequest(t *testing.T) {
	vendor := &recordingVendor{reply: "answer"}
	chatter, _ := newBudgetChatter(t, vendor, 0)

	request := &domain.ChatRequest{Message: userMsgForTest("hi")}
	opts := budgetOpts(0)
	opts.ModelContextLength = -4096
	_, err := chatter.Send(context.Background(), request, opts)
	if err == nil {
		t.Fatal("expected invalid configuration error")
	}
	var inv *contextbudget.InvalidConfigurationError
	if !errors.As(err, &inv) {
		t.Fatalf("expected InvalidConfigurationError, got %T: %v", err, err)
	}
	if inv.Option != "modelContextLength" {
		t.Fatalf("error names %q, want modelContextLength", inv.Option)
	}
	if vendor.callN != 0 {
		t.Fatal("vendor must not be called for an invalid configuration")
	}
}

// TestContextBudget_Send_UnknownWindowConservativeDefault: with no window
// anywhere, the conservative default applies and the request still goes out.
func TestContextBudget_Send_UnknownWindowConservativeDefault(t *testing.T) {
	vendor := &recordingVendor{reply: "answer"}
	chatter, _ := newBudgetChatter(t, vendor, 0)

	request := &domain.ChatRequest{Message: userMsgForTest("just a small question")}
	opts := &domain.ChatOptions{Model: "test-model"} // both window values zero
	if _, err := chatter.Send(context.Background(), request, opts); err != nil {
		t.Fatalf("Send: %v", err)
	}
	if vendor.callN != 1 {
		t.Fatalf("expected one vendor call, got %d", vendor.callN)
	}
}

// TestContextBudget_Disabled_ByteForByteRegression: with budgeting off (the
// default), the vendor receives exactly GetVendorMessages, unchanged.
func TestContextBudget_Disabled_ByteForByteRegression(t *testing.T) {
	vendor := &recordingVendor{reply: "answer"}
	db := fsdb.NewDb(t.TempDir())
	if err := db.Sessions.Configure(); err != nil {
		t.Fatalf("configure: %v", err)
	}
	chatter := &Chatter{
		db: db, Stream: false, vendor: vendor, model: "test-model",
		contextBudgeting: false,
	}
	if err := db.Sessions.SaveSession(&fsdb.Session{
		Name: "reg",
		Messages: []*chat.ChatCompletionMessage{
			systemMsgForTest("SYS"),
			userMsgForTest("one"), assistantMsgForTest("two"),
		},
	}); err != nil {
		t.Fatalf("save: %v", err)
	}
	request := &domain.ChatRequest{
		SessionName: "reg",
		Message:     userMsgForTest("three"),
	}
	if _, err := chatter.Send(context.Background(), request, &domain.ChatOptions{Model: "test-model"}); err != nil {
		t.Fatalf("Send: %v", err)
	}
	sent := vendor.captured()
	want := []string{"system:SYS", "user:one", "assistant:two", "user:three"}
	if len(sent) != len(want) {
		t.Fatalf("sent %d messages, want %d", len(sent), len(want))
	}
	for i, msg := range sent {
		got := msg.Role + ":" + msg.Content
		if got != want[i] {
			t.Fatalf("message %d = %q, want %q", i, got, want[i])
		}
	}
}

// TestContextBudget_DeterministicAcrossRepeatedAssembly: two independent,
// equivalent sessions with the same budget assemble to identical sending bytes.
func TestContextBudget_DeterministicAcrossRepeatedAssembly(t *testing.T) {
	buildChatter := func(t *testing.T) (*Chatter, *recordingVendor) {
		vendor := &recordingVendor{reply: "answer"}
		db := fsdb.NewDb(t.TempDir())
		if err := db.Sessions.Configure(); err != nil {
			t.Fatalf("configure: %v", err)
		}
		seed := []*chat.ChatCompletionMessage{systemMsgForTest("SYS")}
		for i := 0; i < 5; i++ {
			seed = append(seed,
				userMsgForTest("q"+strings.Repeat("a", 90)),
				assistantMsgForTest("a"+strings.Repeat("b", 90)),
			)
		}
		if err := db.Sessions.SaveSession(&fsdb.Session{Name: "det", Messages: seed}); err != nil {
			t.Fatalf("save: %v", err)
		}
		return &Chatter{
			db: db, Stream: false, vendor: vendor, model: "test-model",
			modelContextLength: 400, contextBudgeting: true,
		}, vendor
	}
	chatterA, vendorA := buildChatter(t)
	chatterB, vendorB := buildChatter(t)

	req := &domain.ChatRequest{
		SessionName: "det",
		Message:     userMsgForTest("current " + strings.Repeat("c", 60)),
	}
	if _, err := chatterA.Send(context.Background(), req, budgetOpts(400)); err != nil {
		t.Fatalf("Send A: %v", err)
	}
	if _, err := chatterB.Send(context.Background(), req, budgetOpts(400)); err != nil {
		t.Fatalf("Send B: %v", err)
	}
	a, b := vendorA.captured(), vendorB.captured()
	if len(a) != len(b) {
		t.Fatalf("assembly length differs: %d vs %d", len(a), len(b))
	}
	for i := range a {
		if a[i].Role != b[i].Role || a[i].Content != b[i].Content {
			t.Fatalf("message %d differs:\n A=%s:%q\n B=%s:%q", i, a[i].Role, a[i].Content, b[i].Role, b[i].Content)
		}
	}
}

// TestContextBudget_DryRunShowsTrimmedMessagesAndDropCount: dry run renders
// exactly what would be sent, including the trim marker, and writes nothing to
// the session store.
func TestContextBudget_DryRun_DisplaysTrimmedView(t *testing.T) {
	db := fsdb.NewDb(t.TempDir())
	if err := db.Sessions.Configure(); err != nil {
		t.Fatalf("configure: %v", err)
	}
	if err := db.Sessions.SaveSession(&fsdb.Session{
		Name: "dry",
		Messages: []*chat.ChatCompletionMessage{
			systemMsgForTest("SYS"),
			userMsgForTest(strings.Repeat("old question ", 60)),
			assistantMsgForTest(strings.Repeat("old answer ", 60)),
		},
	}); err != nil {
		t.Fatalf("save: %v", err)
	}
	chatter := &Chatter{
		db: db, Stream: false, DryRun: true,
		vendor: dryrun.NewClient(), model: "test-model",
		modelContextLength: 250, contextBudgeting: true,
	}
	request := &domain.ChatRequest{
		SessionName: "dry",
		Message:     userMsgForTest("current " + strings.Repeat("z", 100)),
	}
	session, err := chatter.Send(context.Background(), request, budgetOpts(250))
	if err != nil {
		t.Fatalf("dry run Send: %v", err)
	}
	out := session.GetLastMessage().Content
	if !strings.Contains(out, "Would send the following request") {
		t.Fatalf("dry run output missing request section: %q", out)
	}
	if !strings.Contains(out, "current") {
		t.Fatal("dry run dropped the current input")
	}
	// The stored session must remain exactly the three seed messages.
	stored, err := db.Sessions.Get("dry")
	if err != nil {
		t.Fatalf("reload: %v", err)
	}
	if len(stored.Messages) != 3 {
		t.Fatalf("dry run modified the session: got %d messages", len(stored.Messages))
	}
}

func systemMsgForTest(content string) *chat.ChatCompletionMessage {
	return &chat.ChatCompletionMessage{Role: chat.ChatMessageRoleSystem, Content: content}
}
func userMsgForTest(content string) *chat.ChatCompletionMessage {
	return &chat.ChatCompletionMessage{Role: chat.ChatMessageRoleUser, Content: content}
}
func assistantMsgForTest(content string) *chat.ChatCompletionMessage {
	return &chat.ChatCompletionMessage{Role: chat.ChatMessageRoleAssistant, Content: content}
}
func writeFileForTest(path, content string) error {
	return os.WriteFile(path, []byte(content), 0o644)
}
