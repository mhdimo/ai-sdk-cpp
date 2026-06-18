#include <catch2/catch_test_macros.hpp>

#include <ai/stream/async_generator.hpp>
#include <ai/stream/stream_part.hpp>
#include <ai/model/language_model.hpp>
#include <ai/model/call_options.hpp>
#include <ai/core/generate_object.hpp>
#include <ai/core/stream_text.hpp>
#include <ai/schema/json_schema.hpp>
#include <ai/error/ai_error.hpp>
#include <ai/test/mock_model.hpp>

#include <boost/asio.hpp>

#include <memory>
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
