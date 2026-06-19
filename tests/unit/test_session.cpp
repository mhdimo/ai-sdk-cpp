#include <catch2/catch_test_macros.hpp>

#include <ai/session/session.hpp>
#include <ai/session/context_strategy.hpp>
#include <ai/agent/tool_loop_agent.hpp>
#include <ai/prompt/message.hpp>
#include <ai/prompt/content_part.hpp>
#include <ai/tool/tool.hpp>
#include <ai/tool/tool_set.hpp>
#include <ai/schema/json_schema.hpp>
#include <ai/model/language_model.hpp>
#include <ai/test/mock_model.hpp>

#include <boost/asio.hpp>

#include <memory>
#include <string>
#include <vector>

namespace {

template <typename T>
T run(ai::Task<T> task, boost::asio::io_context& ioc) {
    task.start();
    int guard = 0;
    while (!task.done() && guard++ < 100'000) {
        ioc.run_one();
    }
    REQUIRE(task.done());
    return task.get();
}

/// Deterministic token counter that scores each message as a fixed cost.
/// Used to force compaction on small windows predictably.
class FixedTokenCounter : public ai::TokenCounter {
public:
    explicit FixedTokenCounter(int per_message) : per_message_(per_message) {}

    int count_tokens(std::string_view) const override { return per_message_; }

    int count_message_tokens(const ai::Message&) const override {
        return per_message_;
    }

    int count_messages_tokens(const ai::Prompt& messages) const override {
        return static_cast<int>(messages.size()) * per_message_;
    }

private:
    int per_message_;
};

/// Build a flat vector of plain user messages (handy for sizing history).
ai::Prompt make_user_history(std::size_t n, const std::string& prefix = "msg") {
    ai::Prompt p;
    for (std::size_t i = 0; i < n; ++i) {
        ai::UserContent c;
        c.push_back(ai::TextPart{.text = prefix + std::to_string(i)});
        p.push_back(ai::UserMessage{.content = std::move(c)});
    }
    return p;
}

/// Append an assistant message carrying a single tool call.
void append_assistant_tool_call(ai::Prompt& p, std::string id) {
    ai::AssistantContent c;
    c.push_back(ai::ToolCallPart{
        .tool_call_id = id,
        .tool_name = "lookup",
        .input = boost::json::value(boost::json::object{{"q", id}}),
    });
    p.push_back(ai::AssistantMessage{.content = std::move(c)});
}

/// Append a tool message carrying the result for a tool_call_id.
void append_tool_result(ai::Prompt& p, std::string id) {
    ai::ToolContent c;
    c.push_back(ai::ToolResultPart{
        .tool_call_id = id,
        .tool_name = "lookup",
        .output = ai::TextOutput{.value = std::string("result-") + id},
    });
    p.push_back(ai::ToolMessage{.content = std::move(c)});
}

} // namespace

// ---------------------------------------------------------------------------
// SlidingWindowStrategy / evict_turns_from_front
// ---------------------------------------------------------------------------

TEST_CASE("SlidingWindowStrategy trims to fit a small window", "[session]") {
    FixedTokenCounter counter(/*per_message=*/10);
    ai::ContextWindow window{.max_tokens = 35, .reserved_output_tokens = 0};
    // available = 35 -> at most 3 messages.

    ai::SlidingWindowStrategy strategy;
    ai::Summarizer noop;

    ai::Prompt history = make_user_history(6);
    REQUIRE(history.size() == 6);

    boost::asio::io_context ioc;
    ai::Prompt managed = run(strategy.manage(history, window, counter, noop), ioc);

    REQUIRE(managed.size() <= 3);
    // The most recent messages must be retained.
    ai::UserContent& last =
        std::get<ai::UserMessage>(managed.back()).content;
    REQUIRE(std::get<ai::TextPart>(last.front()).text == "msg5");
}

TEST_CASE("SlidingWindowStrategy protects the system message", "[session]") {
    FixedTokenCounter counter(/*per_message=*/10);
    ai::ContextWindow window{.max_tokens = 25, .reserved_output_tokens = 0};

    ai::SlidingWindowStrategy strategy;
    ai::Summarizer noop;

    ai::Prompt history;
    history.push_back(ai::SystemMessage{.content = "you are helpful"});
    for (auto& m : make_user_history(4)) {
        history.push_back(m);
    }

    boost::asio::io_context ioc;
    ai::Prompt managed = run(strategy.manage(history, window, counter, noop), ioc);

    REQUIRE_FALSE(managed.empty());
    REQUIRE(std::holds_alternative<ai::SystemMessage>(managed.front()));
    REQUIRE(std::get<ai::SystemMessage>(managed.front()).content ==
            "you are helpful");
}

TEST_CASE("SlidingWindowStrategy preserves tool-call/tool-result pair integrity",
          "[session]") {
    FixedTokenCounter counter(/*per_message=*/10);
    ai::ContextWindow window{.max_tokens = 45, .reserved_output_tokens = 0};
    // available = 45 -> at most 4 messages.

    ai::SlidingWindowStrategy strategy;
    ai::Summarizer noop;

    ai::Prompt history;
    // Old turn (ought to be evicted as a unit).
    history.push_back(ai::SystemMessage{.content = "sys"});
    append_assistant_tool_call(history, "old1");
    append_tool_result(history, "old1");
    // Middle user turn.
    for (auto& m : make_user_history(1, "mid")) {
        history.push_back(m);
    }
    // Current turn: assistant tool call + matching result (must stay together).
    append_assistant_tool_call(history, "cur1");
    append_tool_result(history, "cur1");

    boost::asio::io_context ioc;
    ai::Prompt managed = run(strategy.manage(history, window, counter, noop), ioc);

    // Gather every referenced tool_call_id and every result id present.
    auto ids_in_assistants =
        ai::collect_tool_call_ids(managed.begin(), managed.end());

    // For every ToolMessage present, every tool_call_id it references must
    // also appear in an AssistantMessage (no orphaned tool result), and
    // every AssistantMessage tool_call_id that is kept must have its matching
    // tool result kept (no orphaned tool_use).
    for (auto& msg : managed) {
        if (auto* tool = std::get_if<ai::ToolMessage>(&msg)) {
            for (auto& part : tool->content) {
                INFO("orphaned tool result: " << part.tool_call_id);
                REQUIRE(ids_in_assistants.count(part.tool_call_id));
            }
        }
    }
    for (auto& msg : managed) {
        if (auto* asst = std::get_if<ai::AssistantMessage>(&msg)) {
            for (auto& part : asst->content) {
                if (auto* tc = std::get_if<ai::ToolCallPart>(&part)) {
                    // A kept assistant tool call must have its result kept.
                    bool found = false;
                    for (auto& m2 : managed) {
                        if (auto* t2 = std::get_if<ai::ToolMessage>(&m2)) {
                            for (auto& p2 : t2->content) {
                                if (p2.tool_call_id == tc->tool_call_id) {
                                    found = true;
                                }
                            }
                        }
                    }
                    INFO("orphaned tool_use: " << tc->tool_call_id);
                    REQUIRE(found);
                }
            }
        }
    }

    // The protected current turn must survive intact.
    bool has_cur_use = false;
    bool has_cur_result = false;
    for (auto& msg : managed) {
        if (auto* asst = std::get_if<ai::AssistantMessage>(&msg)) {
            for (auto& part : asst->content) {
                if (auto* tc = std::get_if<ai::ToolCallPart>(&part)) {
                    if (tc->tool_call_id == "cur1") has_cur_use = true;
                }
            }
        }
        if (auto* tool = std::get_if<ai::ToolMessage>(&msg)) {
            for (auto& part : tool->content) {
                if (part.tool_call_id == "cur1") has_cur_result = true;
            }
        }
    }
    REQUIRE(has_cur_use);
    REQUIRE(has_cur_result);
}

// ---------------------------------------------------------------------------
// SummarizationStrategy
// ---------------------------------------------------------------------------

TEST_CASE("SummarizationStrategy invokes the Summarizer and keeps recent turns",
          "[session]") {
    FixedTokenCounter counter(/*per_message=*/10);
    ai::ContextWindow window{.max_tokens = 40, .reserved_output_tokens = 0};
    // available = 40 -> at most 4 messages, but the summary counts as one of
    // them. kept_turns = 1 below.

    ai::SummarizationStrategy strategy(/*kept_turns=*/1);

    bool summarizer_called = false;
    ai::Summarizer summarizer = [&](const ai::Prompt& evicted)
        -> ai::Task<std::string> {
        summarizer_called = true;
        // Confirm the evicted tail is exactly the non-protected middle.
        std::string out = "summarized-" +
            std::to_string(evicted.size()) + "-msgs";
        co_return out;
    };

    ai::Prompt history;
    history.push_back(ai::SystemMessage{.content = "sys"});
    for (auto& m : make_user_history(6)) { // 6 evictable user turns
        history.push_back(m);
    }

    boost::asio::io_context ioc;
    ai::Prompt managed = run(strategy.manage(history, window, counter, summarizer), ioc);

    REQUIRE(summarizer_called);
    // System message preserved at the front.
    REQUIRE(std::holds_alternative<ai::SystemMessage>(managed.front()));
    // A summary block was injected after the system message.
    bool has_summary = false;
    for (auto& m : managed) {
        if (auto* sys = std::get_if<ai::SystemMessage>(&m)) {
            if (sys->content.rfind("## Earlier in this session", 0) == 0) {
                has_summary = true;
                REQUIRE(sys->content.find("summarized-") != std::string::npos);
            }
        }
    }
    REQUIRE(has_summary);
    // Recent turn kept verbatim.
    ai::UserContent& last =
        std::get<ai::UserMessage>(managed.back()).content;
    REQUIRE(std::get<ai::TextPart>(last.front()).text == "msg5");
    // Fits the window.
    REQUIRE(counter.count_messages_tokens(managed) <=
            window.available_input_tokens());
}

// ---------------------------------------------------------------------------
// Session (end-to-end with MockLanguageModel + ToolLoopAgent)
// ---------------------------------------------------------------------------

namespace {

class TestAgent : public ai::Agent {
public:
    explicit TestAgent(ai::LanguageModelPtr model)
        : agent_(ai::ToolLoopAgentOptions{
              .model = std::move(model),
              .instructions = "",
              .max_steps = 1,
          }) {}

    ai::Task<ai::GenerateTextResult> call(
        std::string prompt, ai::CancellationToken cancel = {}) override {
        co_return co_await agent_.call(std::move(prompt), std::move(cancel));
    }

    ai::Task<ai::GenerateTextResult> call(
        std::vector<ai::Message> messages,
        ai::CancellationToken cancel = {}) override {
        co_return co_await agent_.call(std::move(messages), std::move(cancel));
    }

private:
    ai::ToolLoopAgent agent_;
};

} // namespace

TEST_CASE("Session history grows across two sends", "[session]") {
    boost::asio::io_context ioc;
    auto model = std::make_shared<ai::test::MockLanguageModel>();
    model->queue_text("answer-one");
    model->queue_text("answer-two");

    TestAgent agent(model);
    ai::Session session(agent);

    auto r1 = run(session.send("hello one"), ioc);
    REQUIRE(r1.text == "answer-one");
    REQUIRE(session.history().size() == 2); // user + assistant

    auto r2 = run(session.send("hello two"), ioc);
    REQUIRE(r2.text == "answer-two");
    REQUIRE(session.history().size() == 4); // u a u a

    REQUIRE(session.metadata().turns == 2);
}

TEST_CASE("Session send appends turn assistant+tool messages coherently", "[session]") {
    boost::asio::io_context ioc;
    auto model = std::make_shared<ai::test::MockLanguageModel>();

    ai::ToolSet tools;
    tools.add(ai::tool(
        "lookup",
        ai::schema::JsonSchema::object({{"q", ai::schema::JsonSchema::string()}}),
        "look something up",
        [](const boost::json::value& v, ai::ToolExecutionContext) -> ai::Task<boost::json::value> {
            co_return boost::json::value(std::string("ok-") +
                std::string(v.at("q").as_string()));
        }
    ));

    ai::ToolLoopAgent agent(ai::ToolLoopAgentOptions{
        .model = model,
        .tools = std::move(tools),
        .max_steps = 3,
    });

    // Step 1: model requests a tool call; step 2: model produces final text.
    model->queue_tool_call("lookup", R"({"q":"x"})");
    model->queue_text("final");

    ai::Session session(agent);

    auto r = run(session.send("please look up x"), ioc);
    REQUIRE(r.text == "final");

    // History: user, assistant(tool_call), tool(result), assistant(text).
    REQUIRE(session.history().size() == 4);

    REQUIRE(std::holds_alternative<ai::UserMessage>(session.history()[0]));
    REQUIRE(std::holds_alternative<ai::AssistantMessage>(session.history()[1]));
    REQUIRE(std::holds_alternative<ai::ToolMessage>(session.history()[2]));
    REQUIRE(std::holds_alternative<ai::AssistantMessage>(session.history()[3]));

    // The tool result must reference the same tool_call_id as the assistant
    // tool call (pair integrity preserved through the Session layer).
    auto& asst = std::get<ai::AssistantMessage>(session.history()[1]);
    auto& tool_msg = std::get<ai::ToolMessage>(session.history()[2]);
    std::string use_id;
    for (auto& part : asst.content) {
        if (auto* tc = std::get_if<ai::ToolCallPart>(&part)) {
            use_id = tc->tool_call_id;
        }
    }
    REQUIRE_FALSE(use_id.empty());
    bool matched = false;
    for (auto& part : tool_msg.content) {
        if (part.tool_call_id == use_id) matched = true;
    }
    REQUIRE(matched);
}
