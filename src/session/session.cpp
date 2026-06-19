#include <ai/session/session.hpp>

#include <ai/prompt/content_part.hpp>
#include <ai/util/json.hpp>

#include <random>
#include <sstream>
#include <utility>

namespace ai {

namespace {

/// Reconstruct the new assistant + tool messages produced by a single agent
/// turn from `result.steps` (authoritative) plus the terminal `result.text`.
///
/// We deliberately do NOT slice `result.response_messages` by a positional
/// offset, because the agent may inject or transform messages (e.g. prepend a
/// system message from its instructions) that would misalign the offset.
/// Building from `steps` keeps the turn's tool-call/tool-result pairs together
/// and always appends the final assistant text turn (`generate_text` reports
/// the terminal Stop text in `result.text` but does not append it to its
/// internal prompt).
Prompt extract_turn_messages(const GenerateTextResult& result) {
    Prompt delta;

    for (const auto& step : result.steps) {
        const auto& res = step.result;

        // If this step produced tool calls, the assistant tool-call message
        // and its tool results belong to history as a matched pair.
        const auto tool_calls_content = res.tool_calls();
        if (!tool_calls_content.empty()) {
            AssistantContent content;
            for (auto& c : res.content) {
                if (auto* text = std::get_if<TextContent>(&c)) {
                    if (!text->text.empty()) {
                        content.push_back(TextPart{.text = text->text});
                    }
                } else if (auto* reasoning = std::get_if<ReasoningContent>(&c)) {
                    content.push_back(ReasoningPart{.text = reasoning->text});
                } else if (auto* tc = std::get_if<ToolCallContent>(&c)) {
                    content.push_back(ToolCallPart{
                        .tool_call_id = tc->tool_call_id,
                        .tool_name = tc->tool_name,
                        .input = ai::json::safe_parse(tc->input)
                                     .value_or(boost::json::value(tc->input)),
                    });
                }
            }
            if (!content.empty()) {
                delta.push_back(AssistantMessage{.content = std::move(content)});
            }

            // Matching tool results for this step's tool calls.
            ToolContent tc_content;
            for (auto& r : step.tool_results) {
                ToolResultOutput output = r.is_error
                    ? ToolResultOutput{ErrorJsonOutput{.value = r.output}}
                    : ToolResultOutput{JsonOutput{.value = r.output}};
                tc_content.push_back(ToolResultPart{
                    .tool_call_id = r.tool_call_id,
                    .tool_name = r.tool_name,
                    .output = std::move(output),
                });
            }
            if (!tc_content.empty()) {
                delta.push_back(ToolMessage{.content = std::move(tc_content)});
            }
        }
    }

    // Terminal assistant text turn (Stop), if not already represented.
    bool ends_with_assistant_text = false;
    if (!delta.empty()) {
        if (auto* asst = std::get_if<AssistantMessage>(&delta.back())) {
            for (auto& part : asst->content) {
                if (auto* t = std::get_if<TextPart>(&part)) {
                    if (!t->text.empty()) {
                        ends_with_assistant_text = true;
                    }
                }
            }
        }
    }
    if (!ends_with_assistant_text && !result.text.empty()) {
        AssistantContent content;
        content.push_back(TextPart{.text = result.text});
        delta.push_back(AssistantMessage{.content = std::move(content)});
    }

    return delta;
}

/// Generate a random-ish session id (e.g. "sess-a1b2c3d4e5f6g7h8").
std::string new_session_id() {
    std::random_device rd;
    std::uniform_int_distribution<unsigned> dist(0, 0xFFFFFFFFu);
    std::ostringstream ss;
    ss << "sess-" << std::hex << dist(rd) << dist(rd);
    return ss.str();
}

} // namespace

Session::Session(
    Agent& agent,
    ContextWindow window,
    std::shared_ptr<TokenCounter> counter,
    std::shared_ptr<ContextStrategy> strategy,
    Summarizer summarizer
)
    : agent_(&agent),
      window_(window),
      counter_(counter ? std::move(counter)
                       : std::make_shared<ApproximateTokenCounter>()),
      strategy_(strategy ? std::move(strategy)
                         : std::make_shared<SlidingWindowStrategy>()),
      summarizer_(std::move(summarizer)),
      id_(new_session_id()) {}

Session::Session(Agent& agent, ContextWindow window)
    : Session(
          agent,
          window,
          std::make_shared<ApproximateTokenCounter>(),
          std::make_shared<SlidingWindowStrategy>(),
          {}
      ) {}

Session::~Session() = default;
Session::Session(Session&&) noexcept = default;
Session& Session::operator=(Session&&) noexcept = default;

void Session::add_user(std::string text) {
    UserContent content;
    content.push_back(TextPart{.text = std::move(text)});
    history_.push_back(UserMessage{.content = std::move(content)});
}

void Session::add_assistant(std::string text) {
    AssistantContent content;
    content.push_back(TextPart{.text = std::move(text)});
    history_.push_back(AssistantMessage{.content = std::move(content)});
}

void Session::set_system(std::string text) {
    // Replace an existing leading system message if present, else prepend.
    if (!history_.empty() && std::holds_alternative<SystemMessage>(history_.front())) {
        history_.front() = SystemMessage{.content = std::move(text)};
    } else {
        history_.insert(history_.begin(),
                        SystemMessage{.content = std::move(text)});
    }
}

Task<GenerateTextResult> Session::send(
    std::string user_msg,
    CancellationToken cancel
) {
    // 1. Append the user message.
    add_user(std::move(user_msg));

    // 2. Apply the context strategy to fit the window. `managed` becomes the
    //    agent's input AND the new base for history_ (compaction is sticky).
    Prompt managed = co_await strategy_->manage(
        history_, window_, *counter_, summarizer_);

    // 3. Run the (stateless) agent on the managed history. Pass a copy so we
    //    can rebuild history_ from the managed slice + the turn delta below.
    auto result = co_await agent_->call(Prompt(managed), std::move(cancel));

    // 4. Reconstruct the turn's new assistant + tool messages from the
    //    result's steps + terminal text. Built so tool-call/tool-result
    //    pairs stay together and the final Stop assistant text is recorded.
    Prompt delta = extract_turn_messages(result);

    // 5. Rebuild history_ from the managed slice + delta, reflecting any
    //    compaction that happened and the new turn.
    history_ = std::move(managed);
    for (auto& m : delta) {
        history_.push_back(std::move(m));
    }

    // 6. Update bookkeeping.
    ++metadata_.turns;
    if (result.usage.input_tokens.total) {
        metadata_.total_input_tokens += *result.usage.input_tokens.total;
    }
    if (result.usage.output_tokens.total) {
        metadata_.total_output_tokens += *result.usage.output_tokens.total;
    }
    // Record the most recent compaction summary produced by a summarizing
    // strategy, if any.
    for (auto& m : history_) {
        if (auto* sys = std::get_if<SystemMessage>(&m)) {
            if (sys->content.rfind("## Earlier in this session", 0) == 0) {
                metadata_.compaction_summary = sys->content;
            }
        }
    }

    // Best-effort post-turn hook (e.g. checkpoint writer).
    if (on_turn_finish_) {
        try {
            co_await on_turn_finish_(*this);
        } catch (...) {
            // A hook failure must not break the agent loop.
        }
    }

    co_return result;
}

Task<StreamTextResult> Session::send_stream(
    std::string user_msg,
    CancellationToken cancel
) {
    add_user(std::move(user_msg));

    Prompt managed = co_await strategy_->manage(
        history_, window_, *counter_, summarizer_);

    history_ = managed; // commit compaction immediately for streaming

    co_return co_await agent_->stream(std::move(managed), std::move(cancel));
}

SessionSnapshot Session::snapshot(std::string model_id, std::string provider_id) const {
    return SessionSnapshot{
        .id = id_,
        .history = history_,  // copy
        .metadata = metadata_,
        .model_id = std::move(model_id),
        .provider_id = std::move(provider_id),
    };
}

void Session::restore(SessionSnapshot snapshot) {
    id_ = std::move(snapshot.id);
    history_ = std::move(snapshot.history);
    metadata_ = std::move(snapshot.metadata);
}

} // namespace ai
