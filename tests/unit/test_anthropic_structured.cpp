#include <catch2/catch_test_macros.hpp>

#include <ai/providers/anthropic/anthropic.hpp>
#include <ai/providers/anthropic/anthropic_model.hpp>
#include <ai/model/call_options.hpp>
#include <ai/schema/json_schema.hpp>
#include <ai/prompt/message.hpp>
#include <ai/http/client.hpp>
#include <ai/http/multipart.hpp>
#include <ai/stream/async_generator.hpp>
#include <ai/stream/stream_part.hpp>

#include <boost/asio.hpp>
#include <boost/json.hpp>

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace anthropic = ai::providers::anthropic;

// Holds the io_context, provider, and model as members (declaration order =
// init order) so the provider's internal reference to the io_context stays
// valid. Not moved, so all addresses are stable.
struct AnthropicEnv {
    boost::asio::io_context ioc;
    std::shared_ptr<anthropic::AnthropicProvider> provider;
    anthropic::AnthropicLanguageModel model;

    AnthropicEnv()
        : ioc()
        , provider(std::make_shared<anthropic::AnthropicProvider>(
              anthropic::AnthropicOptions{
                  .api_key = std::string("test-key"),
                  .base_url = "https://example.test",
                  .io_context = ioc,
              }))
        , model("claude-sonnet-4-5", provider) {}
};

// Exercises the structured-output request construction (the fix for the former
// no-op response_format branch). The Anthropic Messages API has no top-level
// response_format, so schema'd JSON must be requested via a forced tool.
TEST_CASE("Anthropic injects forced tool for structured output", "[anthropic][structured]") {
    AnthropicEnv env;

    ai::Prompt prompt;
    ai::UserContent uc;
    uc.push_back(ai::TextPart{.text = "extract"});
    prompt.push_back(ai::UserMessage{.content = std::move(uc)});

    ai::CallOptions co;
    co.prompt = prompt;
    co.response_format = ai::ResponseFormat{
        .type = "json",
        .schema = ai::schema::JsonSchema::object({
            {"answer", ai::schema::JsonSchema::string()},
        }).required({"answer"}),
        .name = std::string("extracted"),
    };

    auto body = env.model.build_request_body(co, false);
    auto& obj = body.as_object();

    // A single forced tool carrying the schema.
    REQUIRE(obj.contains("tools"));
    auto& tools = obj["tools"].as_array();
    REQUIRE(tools.size() == 1);
    REQUIRE(tools[0].as_object().at("name").as_string() == "extracted");
    REQUIRE(tools[0].as_object().contains("input_schema"));

    // tool_choice forces that tool so the model returns schema'd JSON.
    REQUIRE(obj.contains("tool_choice"));
    auto& tc = obj["tool_choice"].as_object();
    REQUIRE(tc.at("type").as_string() == "tool");
    REQUIRE(tc.at("name").as_string() == "extracted");
}

TEST_CASE("Anthropic leaves non-structured calls untouched", "[anthropic][structured]") {
    AnthropicEnv env;

    ai::Prompt prompt;
    ai::UserContent uc;
    uc.push_back(ai::TextPart{.text = "hi"});
    prompt.push_back(ai::UserMessage{.content = std::move(uc)});

    ai::CallOptions co;
    co.prompt = prompt;

    auto body = env.model.build_request_body(co, false);
    auto& obj = body.as_object();

    REQUIRE_FALSE(obj.contains("tool_choice"));
    REQUIRE_FALSE(obj.contains("tools"));
}

TEST_CASE("Anthropic provider exposes a batch processor (C binding dispatch)", "[anthropic][batch]") {
    AnthropicEnv env;
    // The polymorphic seam the C ai_batch_create relies on: returns non-null for
    // a provider that supports batching (no network needed to construct it).
    auto proc = env.provider->batch_processor("claude-sonnet-4-5");
    REQUIRE(proc != nullptr);
}

// ---------------------------------------------------------------------------
// Structured output response side (do_generate / do_stream) via a fake HTTP
// client — verifies the tool-use extraction end to end, no network.
// ---------------------------------------------------------------------------

namespace {

template <typename T>
T drive(ai::Task<T> task, boost::asio::io_context& ioc) {
    task.start();
    int guard = 0;
    while (!task.done() && guard++ < 100'000) {
        ioc.run_one();
    }
    REQUIRE(task.done());
    return task.get();
}

class FakeHttpClient : public ai::http::IHttpClient {
public:
    std::string json_body;    // returned by post_json
    std::string stream_body;  // returned by post_streaming (raw SSE bytes)
    int post_json_calls = 0;
    int post_stream_calls = 0;
    boost::asio::io_context& ioc;

    explicit FakeHttpClient(boost::asio::io_context& ctx) : ioc(ctx) {}

    ai::Task<ai::http::HttpResponse> post_json(
        std::string_view, const boost::json::value&, ai::http::Headers,
        ai::CancellationToken
    ) override {
        ++post_json_calls;
        co_return ai::http::HttpResponse{200, {}, json_body};
    }

    ai::Task<ai::http::StreamingResponse> post_streaming(
        std::string_view, const boost::json::value&, ai::http::Headers,
        ai::CancellationToken
    ) override {
        ++post_stream_calls;
        auto gen = [](std::string body) -> ai::AsyncGenerator<std::vector<uint8_t>> {
            std::vector<uint8_t> bytes(body.begin(), body.end());
            co_yield bytes;
        }(stream_body);
        co_return ai::http::StreamingResponse{200, {}, std::move(gen)};
    }

    ai::Task<ai::http::HttpResponse> get(
        std::string_view, ai::http::Headers, ai::CancellationToken
    ) override {
        throw std::runtime_error("FakeHttpClient::get not used");
    }

    ai::Task<ai::http::HttpResponse> post_multipart(
        std::string_view, ai::http::MultipartFormData, ai::http::Headers,
        ai::CancellationToken
    ) override {
        throw std::runtime_error("FakeHttpClient::post_multipart not used");
    }

    boost::asio::io_context& get_io_context() const override { return ioc; }
};

ai::CallOptions structured_call_options() {
    ai::CallOptions co;
    ai::UserContent uc;
    uc.push_back(ai::TextPart{.text = "extract"});
    co.prompt.push_back(ai::UserMessage{.content = std::move(uc)});
    co.response_format = ai::ResponseFormat{
        .type = "json",
        .schema = ai::schema::JsonSchema::object({
            {"answer", ai::schema::JsonSchema::string()},
            {"count", ai::schema::JsonSchema::integer()},
        }).required({"answer", "count"}),
        .name = std::string("extracted"),
    };
    return co;
}

} // namespace

TEST_CASE("Anthropic do_generate surfaces structured tool input as text", "[anthropic][structured]") {
    boost::asio::io_context ioc;
    auto fake = std::make_shared<FakeHttpClient>(ioc);
    fake->json_body = R"({"id":"msg_1","type":"message","role":"assistant","model":"claude-sonnet-4-5","content":[{"type":"tool_use","id":"toolu_1","name":"extracted","input":{"answer":"42","count":7}}],"stop_reason":"tool_use","usage":{"input_tokens":10,"output_tokens":5}})";

    anthropic::AnthropicOptions opts{
        .api_key = std::string("test-key"),
        .base_url = "https://example.test",
        .io_context = ioc,
        .http_client = fake,
    };
    auto provider = std::make_shared<anthropic::AnthropicProvider>(std::move(opts));
    anthropic::AnthropicLanguageModel model("claude-sonnet-4-5", provider);

    auto result = drive(model.do_generate(structured_call_options()), ioc);

    REQUIRE(fake->post_json_calls == 1);
    // The forced tool's input must be surfaced as text so generate_object works.
    auto parsed = boost::json::parse(result.text());
    REQUIRE(parsed.as_object().at("answer").as_string() == "42");
    REQUIRE(parsed.as_object().at("count").as_int64() == 7);
    REQUIRE(result.usage.input_tokens.total.value_or(0) == 10);
}

TEST_CASE("Anthropic do_stream maps structured tool input to text deltas", "[anthropic][structured]") {
    boost::asio::io_context ioc;
    auto fake = std::make_shared<FakeHttpClient>(ioc);
    fake->stream_body = R"(
event: message_start
data: {"type":"message_start","message":{"id":"msg_1","model":"claude-sonnet-4-5","usage":{"input_tokens":10}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_1","name":"extracted"}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"answer\":\"42\",\"count\":7}"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":5}}

event: message_stop
data: {"type":"message_stop"}

)";

    anthropic::AnthropicOptions opts{
        .api_key = std::string("test-key"),
        .base_url = "https://example.test",
        .io_context = ioc,
        .http_client = fake,
    };
    auto provider = std::make_shared<anthropic::AnthropicProvider>(std::move(opts));
    anthropic::AnthropicLanguageModel model("claude-sonnet-4-5", provider);

    auto stream_result = drive(model.do_stream(structured_call_options()), ioc);

    REQUIRE(fake->post_stream_calls == 1);

    // Drain and collect text deltas; structured mode maps tool input_json_delta
    // onto the text channel.
    auto collect = [](ai::AsyncGenerator<ai::StreamPart> s) -> ai::Task<std::string> {
        std::string text;
        while (auto part = co_await s.next()) {
            if (auto* d = std::get_if<ai::TextDelta>(&*part)) {
                text += d->delta;
            }
        }
        co_return text;
    }(std::move(stream_result.stream));
    std::string text = drive(std::move(collect), ioc);

    auto parsed = boost::json::parse(text);
    REQUIRE(parsed.as_object().at("answer").as_string() == "42");
    REQUIRE(parsed.as_object().at("count").as_int64() == 7);
}

// ---------------------------------------------------------------------------
// Thinking signature + redacted_thinking round-trip (extended thinking).
// ---------------------------------------------------------------------------
TEST_CASE("Anthropic round-trips thinking signature + redacted_thinking", "[anthropic][reasoning]") {
    AnthropicEnv env;

    SECTION("parse_response captures signature") {
        auto resp = boost::json::parse(
            R"({"content":[{"type":"thinking","thinking":"hm","signature":"sig123"},)"
            R"({"type":"text","text":"answer"}],)"
            R"("stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":2}})");
        auto result = env.model.parse_response(resp);
        REQUIRE(result.content.size() == 2);
        auto* r = std::get_if<ai::ReasoningContent>(&result.content[0]);
        REQUIRE(r != nullptr);
        REQUIRE(r->text == "hm");
        REQUIRE(r->signature.value_or("") == "sig123");
    }

    SECTION("parse_response captures redacted_thinking") {
        auto resp = boost::json::parse(
            R"({"content":[{"type":"redacted_thinking","data":"opaque"},)"
            R"({"type":"text","text":"answer"}],)"
            R"("stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":2}})");
        auto result = env.model.parse_response(resp);
        auto* r = std::get_if<ai::ReasoningContent>(&result.content[0]);
        REQUIRE(r != nullptr);
        REQUIRE(r->redacted_data.value_or("") == "opaque");
    }

    SECTION("build_request_body re-emits thinking+signature and redacted_thinking") {
        auto build_with = [&](ai::AssistantContent ac) -> boost::json::array {
            ai::Prompt p;
            p.push_back(ai::AssistantMessage{.content = std::move(ac)});
            ai::CallOptions co;
            co.prompt = p;
            return env.model.build_request_body(co, false)
                .as_object().at("messages").as_array()[0]
                .as_object().at("content").as_array();
        };

        ai::AssistantContent ac1;
        ac1.push_back(ai::ReasoningPart{.text = "hm", .signature = std::string("sig123")});
        auto c1 = build_with(std::move(ac1));
        REQUIRE(c1[0].as_object().at("type").as_string() == "thinking");
        REQUIRE(c1[0].as_object().at("signature").as_string() == "sig123");

        ai::AssistantContent ac2;
        ac2.push_back(ai::ReasoningPart{.redacted_data = std::string("opaque")});
        auto c2 = build_with(std::move(ac2));
        REQUIRE(c2[0].as_object().at("type").as_string() == "redacted_thinking");
        REQUIRE(c2[0].as_object().at("data").as_string() == "opaque");
    }
}

TEST_CASE("Anthropic stop_reason / tool_choice:none / metadata", "[anthropic][reasoning]") {
    AnthropicEnv env;

    SECTION("stop_reason refusal -> ContentFilter, pause_turn -> Other") {
        auto mk = [&](const char* reason) {
            auto resp = boost::json::parse(
                std::string(R"({"content":[{"type":"text","text":"x"}],"stop_reason":")")
                + reason + R"(","usage":{"input_tokens":1,"output_tokens":1}})");
            return env.model.parse_response(resp).finish_reason;
        };
        REQUIRE(mk("refusal") == ai::FinishReason::ContentFilter);
        REQUIRE(mk("pause_turn") == ai::FinishReason::Other);
        REQUIRE(mk("tool_use") == ai::FinishReason::ToolCalls);
    }

    SECTION("tool_choice none emitted") {
        ai::CallOptions co;
        ai::UserContent uc;
        uc.push_back(ai::TextPart{.text = "hi"});
        co.prompt.push_back(ai::UserMessage{.content = std::move(uc)});
        co.tools.push_back(ai::FunctionTool{.name = "t", .input_schema = ai::schema::JsonSchema::object({})});
        co.tool_choice = ai::ToolChoiceNone{};
        auto body = env.model.build_request_body(co, false);
        REQUIRE(body.as_object().at("tool_choice").as_object().at("type").as_string() == "none");
    }

    SECTION("metadata.user_id emitted") {
        ai::CallOptions co;
        ai::UserContent uc;
        uc.push_back(ai::TextPart{.text = "hi"});
        co.prompt.push_back(ai::UserMessage{.content = std::move(uc)});
        co.user_id = std::string("user-42");
        auto body = env.model.build_request_body(co, false);
        REQUIRE(body.as_object().at("metadata").as_object().at("user_id").as_string() == "user-42");
    }
}
