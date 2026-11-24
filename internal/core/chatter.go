package core

import (
	"cmp"
	"context"
	"crypto/rand"
	"errors"
	"fmt"
	"os"
	"strings"

	"github.com/danielmiessler/fabric/internal/chat"

	"github.com/danielmiessler/fabric/internal/contextbudget"
	"github.com/danielmiessler/fabric/internal/domain"
	"github.com/danielmiessler/fabric/internal/i18n"
	debuglog "github.com/danielmiessler/fabric/internal/log"
	"github.com/danielmiessler/fabric/internal/plugins/ai"
	"github.com/danielmiessler/fabric/internal/plugins/db/fsdb"
	"github.com/danielmiessler/fabric/internal/plugins/strategy"
	"github.com/danielmiessler/fabric/internal/plugins/template"
)

type Chatter struct {
	db *fsdb.Db

	Stream bool
	DryRun bool

	model              string
	modelContextLength int
	vendor             ai.Vendor

	// contextBudgeting gates the pre-send budget pass. It stays on the
	// Chatter rather than the options so one code path serves the CLI, the
	// REST server and dry runs.
	contextBudgeting bool
}

// NewChatter returns a Chatter without a vendor, for BuildSession only.
// Use GetChatter to get a Chatter that can Send.
func NewChatter(db *fsdb.Db) *Chatter {
	return &Chatter{db: db}
}

// SetContextBudget turns the pre-send context-budget pass on or off. The
// registry builds Chatters in several flows; the flag-bearing callers opt in
// explicitly so the default path stays byte-for-byte unchanged.
func (o *Chatter) SetContextBudget(enabled bool) *Chatter {
	o.contextBudgeting = enabled
	return o
}

// recordFirstStreamError sends err to errChan when the channel has space. It discards later errors.
func recordFirstStreamError(errChan chan error, err error) {
	if err == nil {
		return
	}

	select {
	case errChan <- err:
	default:
		debuglog.Debug(debuglog.Wire, "additional stream error discarded: %v\n", err)
	}
}

// joinPromptSections trims each part, drops empty ones, and joins the rest with newline separators.
func joinPromptSections(parts ...string) string {
	sections := make([]string, 0, len(parts))
	for _, part := range parts {
		trimmed := strings.TrimSpace(part)
		if trimmed != "" {
			sections = append(sections, trimmed)
		}
	}

	return strings.Join(sections, "\n")
}

// NeedsRawMode tells if the vendor needs raw mode for the model.
// It tests o.model, not opts.Model. GetChatter sets o.model to the vendor's spelling of the name.
func (o *Chatter) NeedsRawMode() bool {
	return o.vendor != nil && o.vendor.NeedsRawMode(o.model)
}

// Send processes a chat request and applies file changes for create_coding_feature pattern
func (o *Chatter) Send(ctx context.Context, request *domain.ChatRequest, opts *domain.ChatOptions) (session *fsdb.Session, err error) {
	if o.NeedsRawMode() {
		opts.Raw = true
	}

	// Hold one lock for each session name from the read in BuildSession
	// until SaveSession. Two requests on the same session then cannot
	// lose the messages of the other request.
	if request.SessionName != "" {
		// Validate the name before Lock. Lock keeps a mutex for each name.
		if err = fsdb.ValidateStorageName(request.SessionName); err != nil {
			return
		}
		unlock := o.db.Sessions.Lock(request.SessionName)
		defer unlock()
	}

	// A dry run does not save the session, so it does not announce a new one.
	if session, err = o.BuildSession(request, opts.Raw, !o.DryRun); err != nil {
		return
	}

	// Set one stable session ID for each conversation. Providers that route by session send it as a header.
	if opts.SessionID = cmp.Or(opts.SessionID, session.Name); opts.SessionID == "" {
		opts.SessionID = rand.Text()
	}

	vendorMessages := session.GetVendorMessages()

	// Pre-send budget pass: reserve output room, then trim the *sending*
	// copy only. The session keeps its full history; nothing here is saved.
	// When trimming is disabled (the historical default path) the messages
	// go out byte-for-byte unchanged.
	if o.contextBudgeting {
		var budgetErr error
		vendorMessages, budgetErr = o.applyContextBudget(session, vendorMessages, opts)
		if budgetErr != nil {
			err = budgetErr
			return
		}
	}

	if debuglog.GetLevel() >= debuglog.Wire {
		debuglog.Debug(debuglog.Wire, "FABRIC->LLM request messages (%d)\n", len(vendorMessages))
		for i, msg := range vendorMessages {
			debuglog.Debug(debuglog.Wire, "FABRIC->LLM [%d] role=%s content=%q\n", i, msg.Role, msg.Content)
			if len(msg.MultiContent) > 0 {
				debuglog.Debug(debuglog.Wire, "FABRIC->LLM [%d] parts=%d\n", i, len(msg.MultiContent))
			}
		}
	}
	if len(vendorMessages) == 0 {
		if session.Name != "" {
			err = o.db.Sessions.SaveSession(session)
			if err != nil {
				return
			}
		}
		err = errors.New(i18n.T("chatter_error_no_messages_provided"))
		return
	}

	// Send the vendor's spelling of the model name, not the one the user typed.
	opts.Model = o.model

	if opts.ModelContextLength == 0 {
		opts.ModelContextLength = o.modelContextLength
	}

	message := ""

	if o.Stream {
		responseChan := make(chan domain.StreamUpdate)
		errChan := make(chan error, 1)
		done := make(chan struct{})
		printedStream := false
		// held is the end of the displayed stream that SplitOpenEscape keeps
		// for the next chunk.
		held := ""

		go func() {
			defer close(done)
			if streamErr := o.vendor.SendStream(ctx, vendorMessages, opts, responseChan); streamErr != nil {
				recordFirstStreamError(errChan, &domain.ProviderError{Err: streamErr})
			}
		}()

		for update := range responseChan {
			if debuglog.GetLevel() >= debuglog.Wire {
				debuglog.Debug(debuglog.Wire, "LLM->FABRIC stream update type=%s content=%q\n", update.Type, update.Content)
				if update.Usage != nil {
					debuglog.Debug(debuglog.Wire, "LLM->FABRIC stream usage input=%d output=%d total=%d\n", update.Usage.InputTokens, update.Usage.OutputTokens, update.Usage.TotalTokens)
				}
			}
			if opts.UpdateChan != nil {
				select {
				case opts.UpdateChan <- update:
				case <-ctx.Done():
					recordFirstStreamError(errChan, ctx.Err())
					// Do not stop the loop. If nothing reads responseChan,
					// the vendor goroutine blocks on its next send.
					continue
				}
			}
			switch update.Type {
			case domain.StreamTypeContent:
				message += update.Content
				if !opts.SuppressThink && !opts.BufferStream && !opts.Quiet {
					// Remove terminal control sequences from the displayed copy only.
					var done string
					done, held = domain.SplitOpenEscape(held + update.Content)
					fmt.Print(domain.SanitizeTerminalOutput(done))
					printedStream = true
				}
			case domain.StreamTypeUsage:
				if opts.ShowMetadata && update.Usage != nil && !opts.Quiet {
					fmt.Fprintf(
						os.Stderr,
						"\n%s\n",
						fmt.Sprintf(
							i18n.T("chatter_log_stream_usage_metadata"),
							update.Usage.InputTokens,
							update.Usage.OutputTokens,
							update.Usage.TotalTokens,
						),
					)
				}
			case domain.StreamTypeError:
				if !opts.Quiet {
					fmt.Fprintf(os.Stderr, "%s\n", fmt.Sprintf(i18n.T("chatter_error_stream_update"), update.Content))
				}
				recordFirstStreamError(errChan, &domain.ProviderError{Err: errors.New(update.Content)})
			}
		}

		fmt.Print(domain.SanitizeTerminalOutput(held))
		if printedStream && !opts.SuppressThink && !strings.HasSuffix(message, "\n") && !opts.Quiet {
			fmt.Println()
		}

		<-done

		select {
		case streamErr := <-errChan:
			if streamErr != nil {
				err = streamErr
				return
			}
		default:
		}
	} else {
		if message, err = o.vendor.Send(ctx, vendorMessages, opts); err != nil {
			err = &domain.ProviderError{Err: err}
			return
		}
		if debuglog.GetLevel() >= debuglog.Wire {
			debuglog.Debug(debuglog.Wire, "LLM->FABRIC response content=%q\n", message)
		}
	}

	if opts.SuppressThink && !o.DryRun {
		message = domain.StripThinkBlocks(message, opts.ThinkStartTag, opts.ThinkEndTag)
	}

	if message == "" {
		session = nil
		err = errors.New(i18n.T("chatter_error_empty_response"))
		return
	}

	// Apply file changes only in CLI mode, where UpdateChan is nil.
	// API mode streams responses and must not write files without supervision.
	// A dry run reply is the request text, and the pattern in it has an example
	// of the file changes format. Do not apply that example.
	if request.PatternName == "create_coding_feature" && opts.UpdateChan == nil && !o.DryRun {
		summary, fileChanges, parseErr := domain.ParseFileChanges(message)
		if parseErr != nil {
			// The error can contain a path or an operation from the model.
			fmt.Printf("%s\n", domain.SanitizeTerminalOutput(fmt.Sprintf(i18n.T("chatter_warning_parse_file_changes_failed"), parseErr)))
		} else if len(fileChanges) > 0 {
			projectRoot, err := os.Getwd()
			if err != nil {
				fmt.Printf("%s\n", fmt.Sprintf(i18n.T("chatter_warning_get_current_directory_failed"), err))
			} else {
				if applyErr := domain.ApplyFileChanges(projectRoot, fileChanges); applyErr != nil {
					fmt.Printf("%s\n", fmt.Sprintf(i18n.T("chatter_warning_apply_file_changes_failed"), applyErr))
				} else {
					fmt.Println(i18n.T("chatter_info_file_changes_applied_successfully"))
					fmt.Printf("%s\n\n", i18n.T("chatter_help_review_changes_with_git_diff"))
				}
			}
		}
		message = summary
	}

	session.Append(&chat.ChatCompletionMessage{Role: chat.ChatMessageRoleAssistant, Content: message})

	// A dry run reply is not a model reply. If the session keeps it, the next
	// request sends it to the model.
	if session.Name != "" && !o.DryRun {
		err = o.db.Sessions.SaveSession(session)
	}
	return
}

// BuildSession prints a notice for a new named session when announceNewSession is true.
func (o *Chatter) BuildSession(
	request *domain.ChatRequest, raw bool, announceNewSession bool) (session *fsdb.Session, err error) {
	if request.SessionName != "" {
		var sess *fsdb.Session
		if sess, err = o.db.Sessions.GetWithNotice(request.SessionName, announceNewSession); err != nil {
			err = fmt.Errorf(i18n.T("chatter_error_find_session"), request.SessionName, err)
			return
		}
		session = sess
	} else {
		session = &fsdb.Session{}
	}

	if request.Meta != "" {
		session.Append(&chat.ChatCompletionMessage{Role: domain.ChatMessageRoleMeta, Content: request.Meta})
	}

	var contextContent string
	if request.ContextName != "" {
		var ctx *fsdb.Context
		if ctx, err = o.db.Contexts.Get(request.ContextName); err != nil {
			err = fmt.Errorf(i18n.T("chatter_error_find_context"), request.ContextName, err)
			return
		}
		contextContent = ctx.Content
	}

	if request.Message == nil {
		request.Message = &chat.ChatCompletionMessage{
			Role:    chat.ChatMessageRoleUser,
			Content: "",
		}
	}

	if request.InputHasVars && !request.NoVariableReplacement {
		request.Message.Content, err = template.ApplyTemplateInput(request.Message.Content, request.PatternVariables)
		if err != nil {
			return nil, err
		}
	}

	var patternContent string
	inputUsed := false
	if request.PatternName != "" {
		var pattern *fsdb.Pattern
		if request.NoVariableReplacement {
			pattern, err = o.db.Patterns.GetWithoutVariables(request.PatternName, request.Message.Content)
		} else {
			pattern, err = o.db.Patterns.GetApplyVariables(request.PatternName, request.PatternVariables, request.Message.Content)
		}

		if err != nil {
			return nil, fmt.Errorf(i18n.T("chatter_error_get_pattern"), request.PatternName, err)
		}
		patternContent = pattern.Pattern
		inputUsed = pattern.InputUsed
	}

	systemMessage := joinPromptSections(contextContent, patternContent)

	if request.StrategyName != "" {
		strategy, err := strategy.LoadStrategy(request.StrategyName)
		if err != nil {
			return nil, fmt.Errorf(i18n.T("chatter_error_load_strategy"), request.StrategyName, err)
		}
		if strategy != nil && strategy.Prompt != "" {
			systemMessage = joinPromptSections(strategy.Prompt, systemMessage)
		}
	}

	if request.Language != "" && request.Language != "en" {
		// The prompt tells the model to run the instructions first, then write the full response in request.Language.
		systemMessage = fmt.Sprintf(i18n.T("chatter_prompt_enforce_response_language"), systemMessage, request.Language)
	}

	// The request must end with a user message: some backends reject a
	// request with system messages only. The input goes to the model one time.
	msg := request.Message
	hasInput := msg.Content != "" || len(msg.MultiContent) > 0
	if !raw && !inputUsed && hasInput {
		// The usual shape: instructions in the system message, input in the user message.
		if systemMessage != "" {
			session.Append(&chat.ChatCompletionMessage{Role: chat.ChatMessageRoleSystem, Content: systemMessage})
		}
		session.Append(msg)
	} else {
		// Raw mode, a pattern that contains the input, or no input:
		// send all of the text in one user message.
		text := systemMessage
		if !inputUsed {
			text = joinPromptSections(systemMessage, msg.Content)
		}
		merged := &chat.ChatCompletionMessage{Role: chat.ChatMessageRoleUser, Content: text}
		if len(msg.MultiContent) > 0 {
			// Attachments: the text goes first, then the parts from the request.
			merged.Content = ""
			if text != "" {
				merged.MultiContent = []chat.ChatMessagePart{{Type: chat.ChatMessagePartTypeText, Text: text}}
			}
			merged.MultiContent = append(merged.MultiContent, msg.MultiContent...)
		}
		session.Append(merged)
	}

	if session.IsEmpty() {
		session = nil
		err = errors.New(i18n.T("chatter_error_no_session_pattern_user_messages"))
	}
	return
}

// ApplyContextBudgetForExport runs the same budget pass that Send runs, for
// flows that render the request without sending it (--print-prompt). It keeps
// the same deterministic rules and notice behavior.
func (o *Chatter) ApplyContextBudgetForExport(
	messages []*chat.ChatCompletionMessage, opts *domain.ChatOptions) ([]*chat.ChatCompletionMessage, error) {
	return o.applyContextBudget(nil, messages, opts)
}

// applyContextBudget runs the deterministic budget pass over one request and
// returns the messages that may actually leave the process. It never mutates
// the session: system messages, history and the current input are split out of
// the vendor view, planned, and reassembled as new slices. Notices (unknown
// window, dropped history, truncated input) go to stderr in CLI mode; the REST
// server passes Quiet, which suppresses them because it streams its own events.
func (o *Chatter) applyContextBudget(
	_ *fsdb.Session,
	vendorMessages []*chat.ChatCompletionMessage,
	opts *domain.ChatOptions) ([]*chat.ChatCompletionMessage, error) {

	window := opts.ModelContextLength
	if window < 0 {
		return nil, &contextbudget.InvalidConfigurationError{Option: "modelContextLength", Value: window}
	}
	maxTokens := opts.MaxTokens
	if maxTokens < 0 {
		return nil, &contextbudget.InvalidConfigurationError{Option: "maxTokens", Value: maxTokens}
	}

	system, history, current := splitForBudget(vendorMessages)
	plan, err := contextbudget.PlanMessages(system, history, current, contextbudget.Options{
		ContextWindow: window,
		MaxTokens:     maxTokens,
	})
	if err != nil {
		return nil, err
	}

	if !opts.Quiet {
		for _, notice := range plan.Notices {
			fmt.Fprintf(os.Stderr, "%s\n", notice.Message)
		}
	}

	debuglog.Debug(debuglog.Basic,
		"context budget: window=%d unknown=%v output_reserved=%d input_budget=%d estimated_input=%d dropped_messages=%d dropped_tool_pairs=%d input_truncated=%v\n",
		plan.Window, plan.UnknownWindow, plan.OutputReservation, plan.InputBudget,
		plan.EstimatedInput, plan.DroppedMessageCount, plan.DroppedPairs, plan.InputTruncated)

	return plan.Messages, nil
}

// splitForBudget divides the vendor view into the fixed-priority sections.
// System and developer messages are the mode/instructions section; the final
// user message is the current input; everything between is history in
// chronological order.
func splitForBudget(messages []*chat.ChatCompletionMessage) (
	system, history []*chat.ChatCompletionMessage, current *chat.ChatCompletionMessage) {

	currentIdx := -1
	for i := len(messages) - 1; i >= 0; i-- {
		if messages[i] != nil && messages[i].Role == chat.ChatMessageRoleUser {
			currentIdx = i
			break
		}
	}
	for i, msg := range messages {
		switch {
		case msg == nil:
			continue
		case msg.Role == chat.ChatMessageRoleSystem || msg.Role == chat.ChatMessageRoleDeveloper:
			system = append(system, msg)
		case i == currentIdx:
			current = msg
		default:
			history = append(history, msg)
		}
	}
	return system, history, current
}
