#include <catch2/catch_test_macros.hpp>

#include <ai/stream/async_generator.hpp>
#include <ai/stream/stream_part.hpp>
#include <ai/model/language_model.hpp>
#include <ai/model/call_options.hpp>
#include <ai/core/generate_object.hpp>
#include <ai/core/stream_text.hpp>
#include <ai/schema/json_schema.hpp>
#include <ai/error/ai_error.hpp>
#include <ai/prompt/message.hpp>
#include <ai/test/mock_model.hpp>
#include <ai/tool/tool.hpp>
#include <ai/tool/tool_set.hpp>

#include <boost/asio.hpp>
#include <boost/json.hpp>

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

/// A minimal LanguageModel whose do_stream yields a caller-supplied sequence of
/// StreamParts verbatim. Used to exercise the AsyncGenerator handshake with
/// arbitrary (incl. multi-chunk) part sequences, offline.
class FakeStreamModel : public ai::LanguageModel {
public:
    explicit FakeStreamModel(std::vector<ai::StreamPart> parts)
        : parts_(std::move(parts)) {}

    std::string_view provider() const override { return "fake"; }
    std::string_view model_id() const override { return "fake-model"; }

    ai::Task<ai::GenerateResult> do_generate(ai::CallOptions) override {
        throw std::runtime_error("FakeStreamModel: do_generate not implemented");
    }

    ai::Task<ai::StreamResult> do_stream(ai::CallOptions options) override {
        ++call_count_;
        auto gen = [](std::vector<ai::StreamPart> parts) -> ai::AsyncGenerator<ai::StreamPart> {
            for (auto& p : parts) {
                co_yield p;
            }
        }(std::move(parts_));
        (void)options;
        co_return ai::StreamResult{.stream = std::move(gen)};
    }

    int call_count() const { return call_count_; }

private:
    std::vector<ai::StreamPart> parts_;
    int call_count_ = 0;
};

/// Drive a root Task to completion on an io_context, with a guard against
/// pathological infinite loops (would only trigger if a coroutine deadlocks).
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

} // namespace

// ---------------------------------------------------------------------------
// Proof of the AsyncGenerator handshake fix (item 1).
// Before the fix, every consume-loop deadlocked on the first iteration because
// yield_value returned suspend_always and never resumed the consumer.
// ---------------------------------------------------------------------------

TEST_CASE("AsyncGenerator drains all parts in order", "[streaming]") {
    boost::asio::io_context ioc;
    auto model = std::make_shared<FakeStreamModel>(std::vector<ai::StreamPart>{
        ai::TextStart{.id = "0"},
        ai::TextDelta{.id = "0", .delta = "Hello"},
        ai::TextDelta{.id = "0", .delta = " "},
        ai::TextDelta{.id = "0", .delta = "world"},
        ai::TextEnd{.id = "0"},
        ai::FinishPart{.reason = ai::FinishReason::Stop},
    });

    auto collect = [](ai::LanguageModelPtr m, boost::asio::io_context&)
        -> ai::Task<std::vector<std::string>> {
        auto r = co_await m->do_stream(ai::CallOptions{});
        std::vector<std::string> deltas;
        while (auto part = co_await r.stream.next()) {
            if (auto* d = std::get_if<ai::TextDelta>(&*part)) {
                deltas.push_back(d->delta);
            }
        }
        co_return deltas;
    }(model, ioc);

    auto deltas = run(std::move(collect), ioc);

    REQUIRE(deltas.size() == 3);
    REQUIRE(deltas[0] == "Hello");
    REQUIRE(deltas[1] == " ");
    REQUIRE(deltas[2] == "world");
}

TEST_CASE("MockLanguageModel stream drains end to end", "[streaming]") {
    boost::asio::io_context ioc;
    auto model = std::make_shared<ai::test::MockLanguageModel>();
    model->queue_text("Hello world");

    auto collect = [](ai::LanguageModelPtr m) -> ai::Task<std::string> {
        auto r = co_await m->do_stream(ai::CallOptions{});
        std::string text;
        while (auto part = co_await r.stream.next()) {
            if (auto* d = std::get_if<ai::TextDelta>(&*part)) {
                text += d->delta;
            }
        }
        co_return text;
    }(model);

    REQUIRE(run(std::move(collect), ioc) == "Hello world");
    REQUIRE(model->call_count() == 1);
}

// ---------------------------------------------------------------------------
// stream_object (item 4)
// ---------------------------------------------------------------------------

namespace {

ai::Task<ai::StreamObjectResult> start_stream_object(
    ai::LanguageModelPtr m, ai::schema::JsonSchema s
) {
    co_return co_await ai::stream_object(ai::StreamObjectOptions{
        .model = m,
        .schema = s,
    });
}

ai::Task<std::vector<boost::json::value>> drain_values(ai::AsyncGenerator<boost::json::value> s) {
    std::vector<boost::json::value> out;
    while (auto p = co_await s.next()) {
        out.push_back(*p);
    }
    co_return out;
}

} // namespace

TEST_CASE("stream_object populates final state and validates", "[streaming]") {
    boost::asio::io_context ioc;
    auto model = std::make_shared<ai::test::MockLanguageModel>();
    ai::Usage usage;
    usage.output_tokens.total = 7;
    model->queue_response(ai::test::MockResponse{
        .text = R"({"answer":"42","count":7})",
        .usage = usage,
    });

    auto schema = ai::schema::JsonSchema::object({
        {"answer", ai::schema::JsonSchema::string()},
        {"count", ai::schema::JsonSchema::integer()},
    }).required({"answer", "count"});

    auto result = run(start_stream_object(model, schema), ioc);
    auto partials = run(drain_values(std::move(result.partial_object_stream)), ioc);

    REQUIRE_FALSE(partials.empty());
    REQUIRE(result.final_state->validated);
    REQUIRE(result.final_state->had_output);
    auto& obj = result.final_state->object.as_object();
    REQUIRE(obj.at("answer").as_string() == "42");
    REQUIRE(obj.at("count").as_int64() == 7);
    REQUIRE(result.final_state->usage.output_tokens.total.value_or(0) == 7);
}

TEST_CASE("stream_object yields incremental partials across chunks", "[streaming]") {
    boost::asio::io_context ioc;
    auto model = std::make_shared<FakeStreamModel>(std::vector<ai::StreamPart>{
        ai::TextStart{.id = "0"},
        ai::TextDelta{.id = "0", .delta = R"({"a":1)"},
        ai::TextDelta{.id = "0", .delta = R"(,"b":2})"},
        ai::TextEnd{.id = "0"},
        ai::FinishPart{.reason = ai::FinishReason::Stop},
    });

    auto schema = ai::schema::JsonSchema::object({
        {"a", ai::schema::JsonSchema::integer()},
        {"b", ai::schema::JsonSchema::integer()},
    }).required({"a", "b"});

    auto result = run(start_stream_object(model, schema), ioc);
    auto partials = run(drain_values(std::move(result.partial_object_stream)), ioc);

    // Two distinct parseable prefixes: {"a":1} then {"a":1,"b":2}.
    REQUIRE(partials.size() == 2);
    REQUIRE(partials[0].as_object().at("a").as_int64() == 1);
    REQUIRE(partials[1].as_object().at("a").as_int64() == 1);
    REQUIRE(partials[1].as_object().at("b").as_int64() == 2);
    REQUIRE(result.final_state->object.as_object().at("b").as_int64() == 2);
}

TEST_CASE("stream_object throws TypeValidationError on schema mismatch", "[streaming]") {
    boost::asio::io_context ioc;
    auto model = std::make_shared<ai::test::MockLanguageModel>();
    // Valid JSON, but count is a string where the schema requires an integer.
    model->queue_text(R"({"answer":"x","count":"not-int"})");

    auto schema = ai::schema::JsonSchema::object({
        {"answer", ai::schema::JsonSchema::string()},
        {"count", ai::schema::JsonSchema::integer()},
    }).required({"answer", "count"});

    auto result = run(start_stream_object(model, schema), ioc);

    // The error is raised inside the generator while draining the final object.
    REQUIRE_THROWS_AS(
        run(drain_values(std::move(result.partial_object_stream)), ioc),
        ai::error::TypeValidationError
    );
}

TEST_CASE("stream_object throws NoOutputGeneratedError on empty output", "[streaming]") {
    boost::asio::io_context ioc;
    auto model = std::make_shared<ai::test::MockLanguageModel>();
    model->queue_text(""); // no text, no tool calls

    auto schema = ai::schema::JsonSchema::object({
        {"answer", ai::schema::JsonSchema::string()},
    }).required({"answer"});

    auto result = run(start_stream_object(model, schema), ioc);

    REQUIRE_THROWS_AS(
        run(drain_values(std::move(result.partial_object_stream)), ioc),
        ai::error::NoOutputGeneratedError
    );
}

TEST_CASE("stream_text full_result resolves after draining", "[streaming]") {
    boost::asio::io_context ioc;
    auto model = std::make_shared<ai::test::MockLanguageModel>();
    ai::Usage usage;
    usage.input_tokens.total = 5;
    usage.output_tokens.total = 8;
    model->queue_response(ai::test::MockResponse{
        .text = "hello stream",
        .usage = usage,
    });

    auto start = [](ai::LanguageModelPtr m) -> ai::Task<ai::StreamTextResult> {
        co_return co_await ai::stream_text(ai::StreamTextOptions{
            .model = m,
            .prompt = std::string("hi"),
        });
    }(model);

    auto result = run(std::move(start), ioc);

    // Drain the stream first.
    auto drain = [](ai::AsyncGenerator<ai::StreamPart> s) -> ai::Task<std::string> {
        std::string t;
        while (auto p = co_await s.next()) {
            if (auto* d = std::get_if<ai::TextDelta>(&*p)) t += d->delta;
        }
        co_return t;
    }(std::move(result.stream));
    REQUIRE(run(std::move(drain), ioc) == "hello stream");

    // full_result must now resolve to the complete result.
    auto fr = run(std::move(result.full_result), ioc);
    REQUIRE(fr.text == "hello stream");
    REQUIRE(fr.usage.input_tokens.total.value_or(0) == 5);
    REQUIRE(fr.usage.output_tokens.total.value_or(0) == 8);
}

// ---------------------------------------------------------------------------
// stream_text's multi-step path.
//
// stream_text branches on `max_steps <= 1 || tools.empty()` and only then enters
// the loop that executes tools and re-invokes the model. Every streaming case
// above takes the single-step branch, so the loop — and the tool-call
// accumulation, reasoning replay and step bookkeeping inside it — went
// unexecuted. These cases are the ones that reach it.
// ---------------------------------------------------------------------------

namespace {

/// A LanguageModel that answers with a different scripted stream per call and
/// remembers the prompts it was handed. A model that can only answer once cannot
/// reach the multi-step loop, which re-invokes it once per tool round trip.
///
/// Past the end of the script the last step replays, which is how a model that
/// refuses to stop asking for tools is expressed.
class ScriptedStreamModel : public ai::LanguageModel {
public:
    explicit ScriptedStreamModel(std::vector<std::vector<ai::StreamPart>> script)
        : script_(std::move(script)) {
        REQUIRE_FALSE(script_.empty());
    }

    std::string_view provider() const override { return "scripted"; }
    std::string_view model_id() const override { return "scripted-model"; }

    ai::Task<ai::GenerateResult> do_generate(ai::CallOptions) override {
        throw std::runtime_error("ScriptedStreamModel: do_generate not implemented");
    }

    ai::Task<ai::StreamResult> do_stream(ai::CallOptions options) override {
        prompts.push_back(std::move(options.prompt));
        size_t index = std::min<size_t>(static_cast<size_t>(call_count_++), script_.size() - 1);
        auto gen = [](std::vector<ai::StreamPart> parts) -> ai::AsyncGenerator<ai::StreamPart> {
            for (auto& p : parts) {
                co_yield p;
            }
        }(script_[index]);
        co_return ai::StreamResult{.stream = std::move(gen)};
    }

    int call_count() const { return call_count_; }
    std::vector<ai::Prompt> prompts;

private:
    std::vector<std::vector<ai::StreamPart>> script_;
    int call_count_ = 0;
};

/// Drain a stream, keeping the parts so the caller can assert on what the
/// consumer actually saw (as opposed to what the internal accumulator built).
ai::Task<std::vector<ai::StreamPart>> drain_parts(ai::AsyncGenerator<ai::StreamPart> s) {
    std::vector<ai::StreamPart> out;
    while (auto p = co_await s.next()) {
        out.push_back(*p);
    }
    co_return out;
}

ai::Usage usage_of(int in, int out) {
    ai::Usage u{};
    u.input_tokens.total = in;
    u.output_tokens.total = out;
    return u;
}

/// A tool that always succeeds, recording the input it was handed.
ai::ToolDefinition recording_tool(std::shared_ptr<std::vector<std::string>> log) {
    return ai::tool(
        "lookup", ai::schema::JsonSchema::object({}).additional_properties(false),
        "look something up",
        [log](boost::json::value input, ai::ToolExecutionContext) -> ai::Task<boost::json::value> {
            log->push_back(boost::json::serialize(input));
            co_return boost::json::value("found");
        });
}

/// Stream the parts of one tool call, then finish asking for tools.
std::vector<ai::StreamPart> tool_call_step(std::vector<ai::StreamPart> prefix, int in, int out) {
    prefix.push_back(ai::ToolInputStart{.id = "tc1", .tool_name = "lookup"});
    prefix.push_back(ai::ToolInputDelta{.id = "tc1", .delta = "{\"k\":"});
    prefix.push_back(ai::ToolInputDelta{.id = "tc1", .delta = "\"v\"}"});
    prefix.push_back(ai::ToolInputEnd{.id = "tc1"});
    prefix.push_back(ai::FinishPart{.reason = ai::FinishReason::ToolCalls, .usage = usage_of(in, out)});
    return prefix;
}

} // namespace

TEST_CASE("stream_text runs a tool step, then streams the final answer", "[streaming]") {
    boost::asio::io_context ioc;
    auto log = std::make_shared<std::vector<std::string>>();

    auto model = std::make_shared<ScriptedStreamModel>(std::vector<std::vector<ai::StreamPart>>{
        tool_call_step({ai::TextDelta{.id = "0", .delta = "Checking"}}, 10, 5),
        {
            ai::TextDelta{.id = "0", .delta = "The answer"},
            ai::FinishPart{.reason = ai::FinishReason::Stop, .usage = usage_of(7, 3)},
        },
    });

    ai::ToolSet tools;
    tools.add(recording_tool(log));

    auto started = run(ai::stream_text(ai::StreamTextOptions{
        .model = model,
        .tools = std::move(tools),
        .prompt = std::string("go"),
        .max_steps = 3,
    }), ioc);

    auto seen = run(drain_parts(std::move(started.stream)), ioc);
    auto final = run(std::move(started.full_result), ioc);

    // The loop executed the tool and went back to the model for a second step.
    REQUIRE(model->call_count() == 2);
    REQUIRE(log->size() == 1);
    REQUIRE((*log)[0] == "{\"k\":\"v\"}");

    REQUIRE(final.steps.size() == 2);
    REQUIRE(final.steps[0].step_number == 0);
    REQUIRE(final.steps[1].step_number == 1);
    REQUIRE(final.steps[0].result.finish_reason == ai::FinishReason::ToolCalls);
    REQUIRE(final.steps[0].tool_results.size() == 1);
    REQUIRE(final.steps[0].tool_results[0].tool_call_id == "tc1");
    REQUIRE(final.steps[0].tool_results[0].output.as_string() == "found");
    REQUIRE_FALSE(final.steps[0].tool_results[0].is_error);

    // The last step is what the caller sees as "the" text and finish reason.
    REQUIRE(final.finish_reason == ai::FinishReason::Stop);
    REQUIRE(final.text == "The answer");

    // Usage is the sum over steps, not just the last one.
    REQUIRE(final.usage.input_tokens.total.value_or(0) == 17);
    REQUIRE(final.usage.output_tokens.total.value_or(0) == 8);

    // generate_text populates result.tool_calls; the streaming path must agree,
    // or the same field means two different things depending on which call the
    // caller used.
    REQUIRE(final.tool_calls.size() == 1);
    REQUIRE(final.tool_calls[0].tool_call_id == "tc1");
    REQUIRE(final.tool_calls[0].output.as_string() == "found");

    // The tool-call parts reached the consumer, not just the accumulator.
    size_t starts = 0;
    for (auto& p : seen) {
        if (std::get_if<ai::ToolInputStart>(&p)) {
            ++starts;
        }
    }
    REQUIRE(starts == 1);
}

TEST_CASE("stream_text replays reasoning with its signature into the next step", "[streaming]") {
    boost::asio::io_context ioc;
    auto log = std::make_shared<std::vector<std::string>>();

    auto model = std::make_shared<ScriptedStreamModel>(std::vector<std::vector<ai::StreamPart>>{
        tool_call_step(
            {
                ai::ReasoningStart{.id = "r0"},
                ai::ReasoningDelta{.id = "r0", .delta = "weighing"},
                ai::ReasoningDelta{.id = "r0", .delta = " options"},
                ai::ReasoningEnd{.id = "r0", .signature = std::string("sig-1")},
            },
            1, 2),
        {
            ai::TextDelta{.id = "0", .delta = "done"},
            ai::FinishPart{.reason = ai::FinishReason::Stop},
        },
    });

    ai::ToolSet tools;
    tools.add(recording_tool(log));

    auto started = run(ai::stream_text(ai::StreamTextOptions{
        .model = model,
        .tools = std::move(tools),
        .prompt = std::string("go"),
        .max_steps = 3,
    }), ioc);

    run(drain_parts(std::move(started.stream)), ioc);
    auto final = run(std::move(started.full_result), ioc);

    REQUIRE(model->call_count() == 2);
    REQUIRE(model->prompts.size() == 2);

    // The assistant turn replayed to the model must carry the reasoning block
    // with its signature intact: a provider that signs its thinking rejects a
    // replayed block whose signature was dropped, which is what breaks
    // multi-turn tool loops with extended thinking on.
    const ai::AssistantMessage* assistant = nullptr;
    for (auto& m : model->prompts[1]) {
        if (auto* a = std::get_if<ai::AssistantMessage>(&m)) {
            assistant = a;
        }
    }
    REQUIRE(assistant != nullptr);

    const ai::ReasoningPart* replayed = nullptr;
    for (auto& c : assistant->content) {
        if (auto* r = std::get_if<ai::ReasoningPart>(&c)) {
            replayed = r;
        }
    }
    REQUIRE(replayed != nullptr);
    REQUIRE(replayed->text == "weighing options");
    REQUIRE(replayed->signature.has_value());
    REQUIRE(*replayed->signature == "sig-1");

    // ...and the caller sees the reasoning on the step that produced it.
    const ai::ReasoningContent* step_reasoning = nullptr;
    for (auto& c : final.steps[0].result.content) {
        if (auto* r = std::get_if<ai::ReasoningContent>(&c)) {
            step_reasoning = r;
        }
    }
    REQUIRE(step_reasoning != nullptr);
    REQUIRE(step_reasoning->text == "weighing options");
    REQUIRE(step_reasoning->signature.has_value());
    REQUIRE(*step_reasoning->signature == "sig-1");
}

TEST_CASE("stream_text stops at max_steps when the model keeps calling tools", "[streaming]") {
    boost::asio::io_context ioc;
    auto log = std::make_shared<std::vector<std::string>>();

    // One scripted step; the model replays it forever.
    auto model = std::make_shared<ScriptedStreamModel>(
        std::vector<std::vector<ai::StreamPart>>{tool_call_step({}, 1, 1)});

    ai::ToolSet tools;
    tools.add(recording_tool(log));

    auto started = run(ai::stream_text(ai::StreamTextOptions{
        .model = model,
        .tools = std::move(tools),
        .prompt = std::string("go"),
        .max_steps = 2,
    }), ioc);

    run(drain_parts(std::move(started.stream)), ioc);
    auto final = run(std::move(started.full_result), ioc);

    // Nothing in the transcript stops the loop, so the bound has to.
    REQUIRE(model->call_count() == 2);
    REQUIRE(final.steps.size() == 2);
    REQUIRE(log->size() == 2);
    REQUIRE(final.finish_reason == ai::FinishReason::ToolCalls);
    REQUIRE(final.tool_calls.size() == 2);
}

TEST_CASE("stream_text reports each step and marks a tool that fails", "[streaming]") {
    boost::asio::io_context ioc;

    auto model = std::make_shared<ScriptedStreamModel>(std::vector<std::vector<ai::StreamPart>>{
        tool_call_step({}, 1, 1),
        {
            ai::TextDelta{.id = "0", .delta = "recovered"},
            ai::FinishPart{.reason = ai::FinishReason::Stop},
        },
    });

    ai::ToolSet tools;
    tools.add(ai::tool(
        "lookup", ai::schema::JsonSchema::object({}).additional_properties(false),
        "always fails",
        [](boost::json::value, ai::ToolExecutionContext) -> ai::Task<boost::json::value> {
            throw std::runtime_error("boom");
        }));

    std::vector<int> reported;
    auto started = run(ai::stream_text(ai::StreamTextOptions{
        .model = model,
        .tools = std::move(tools),
        .prompt = std::string("go"),
        .max_steps = 3,
        .on_step_finish = [&reported](const ai::StepResult& s) {
            reported.push_back(s.step_number);
        },
    }), ioc);

    run(drain_parts(std::move(started.stream)), ioc);
    auto final = run(std::move(started.full_result), ioc);

    // A throwing tool is reported, not propagated: the loop still completes.
    REQUIRE(reported == std::vector<int>{0, 1});
    REQUIRE(model->call_count() == 2);
    REQUIRE(final.steps[0].tool_results.size() == 1);
    REQUIRE(final.steps[0].tool_results[0].is_error);
    REQUIRE(final.steps[0].tool_results[0].output.as_string().find("boom") != std::string::npos);

    // The failure is still appended to the transcript as an error result, so the
    // model can see why and recover rather than repeating the call.
    const ai::ToolMessage* tool_turn = nullptr;
    for (auto& m : model->prompts[1]) {
        if (auto* t = std::get_if<ai::ToolMessage>(&m)) {
            tool_turn = t;
        }
    }
    REQUIRE(tool_turn != nullptr);
    REQUIRE(tool_turn->content.size() == 1);
    REQUIRE(std::holds_alternative<ai::ErrorJsonOutput>(tool_turn->content[0].output));
}
