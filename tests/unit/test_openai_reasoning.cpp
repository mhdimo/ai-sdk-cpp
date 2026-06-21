// OpenAI provider reasoning semantics — verified offline against a fake HTTP
// client. Covers three behaviors that matter for OpenAI-compatible reasoning
// models (DeepSeek v4, R1-style gateways) but are gated independently of
// OpenAI's own reasoning-SKU allowlist:
//   1. reasoning_content is surfaced on read-back (non-stream + stream)
//   2. reasoning_effort is emitted on explicit opt-in (options.reasoning)
//   3. DeepSeek-only knobs (thinking toggle, raw effort override) via
//      provider_options.openai
// Built whenever AI_SDK_PROVIDER_OPENAI is ON (no Google dependency).

#include <catch2/catch_test_macros.hpp>

#include "fake_http_client.hpp"

#include <ai/providers/openai/openai.hpp>
#include <ai/model/language_model.hpp>
#include <ai/model/call_options.hpp>
#include <ai/prompt/message.hpp>
#include <ai/stream/async_generator.hpp>
#include <ai/stream/stream_part.hpp>

#include <boost/asio.hpp>
#include <boost/json.hpp>

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

ai::CallOptions simple_prompt(const std::string& text) {
    ai::CallOptions co;
    ai::UserContent uc;
    uc.push_back(ai::TextPart{.text = text});
    co.prompt.push_back(ai::UserMessage{.content = std::move(uc)});
    return co;
}

std::shared_ptr<ai::providers::openai::OpenAIProvider>
make_provider(std::shared_ptr<ai::test::FakeHttpClient> fake, boost::asio::io_context& ioc,
              std::string base_url = "https://example.test") {
    return std::make_shared<ai::providers::openai::OpenAIProvider>(
        ai::providers::openai::OpenAIOptions{
            .api_key = std::string("test-key"),
            .base_url = std::move(base_url),
            .io_context = ioc,
            .http_client = fake,
        });
}

} // namespace

// DeepSeek v4 (thinking mode) returns reasoning_content before the answer.
TEST_CASE("OpenAI surfaces reasoning_content (DeepSeek-style) on do_generate", "[provider][reasoning]") {
    boost::asio::io_context ioc;
    auto fake = std::make_shared<ai::test::FakeHttpClient>(ioc);
    fake->json_body = R"({"id":"chatcmpl-1","object":"chat.completion",)"
                      R"("choices":[{"index":0,"message":{"role":"assistant",)"
                      R"("reasoning_content":"let me think","content":"hello world"},)"
                      R"("finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":2}})";

    auto model = make_provider(fake, ioc)->language_model("deepseek-v4-flash");
    auto result = run(model->do_generate(simple_prompt("hi")), ioc);

    // Reasoning precedes the answer (matches stream order); text() still works.
    REQUIRE(result.content.size() == 2);
    REQUIRE(std::get_if<ai::ReasoningContent>(&result.content[0]) != nullptr);
    REQUIRE(std::get_if<ai::ReasoningContent>(&result.content[0])->text == "let me think");
    REQUIRE(std::get_if<ai::TextContent>(&result.content[1]) != nullptr);
    REQUIRE(result.text() == "hello world");
}

TEST_CASE("OpenAI emits reasoning_effort + thinking on explicit opt-in", "[provider][reasoning]") {
    boost::asio::io_context ioc;
    auto fake = std::make_shared<ai::test::FakeHttpClient>(ioc);
    fake->json_body = R"({"id":"chatcmpl-1","object":"chat.completion",)"
                      R"("choices":[{"index":0,"message":{"role":"assistant","content":"ok"},)"
                      R"("finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1}})";
    auto provider = make_provider(fake, ioc);

    SECTION("mapped effort + max_tokens for a non-allowlisted reasoning model") {
        auto model = provider->language_model("deepseek-v4-flash");
        auto co = simple_prompt("hi");
        co.reasoning = "high";
        co.max_output_tokens = 256;

        run(model->do_generate(co), ioc);

        REQUIRE(fake->post_json_calls == 1);
        const auto& body = fake->last_request_body.as_object();
        REQUIRE(body.at("reasoning_effort").as_string() == "high");
        // DeepSeek isn't in OpenAI's reasoning-SKU allowlist, so it must use
        // max_tokens (OpenAI's own reasoning SKUs would use max_completion_tokens).
        REQUIRE(body.at("max_tokens").as_int64() == 256);
        REQUIRE_FALSE(body.contains("max_completion_tokens"));
    }

    SECTION("raw provider_options override (DeepSeek max + thinking)") {
        auto model = provider->language_model("deepseek-v4-pro");
        auto co = simple_prompt("hi");
        co.reasoning = "xhigh";
        co.provider_options["openai"] = boost::json::object{
            {"reasoning_effort", "max"},
            {"thinking", "enabled"},
        };

        run(model->do_generate(co), ioc);

        const auto& body = fake->last_request_body.as_object();
        REQUIRE(body.at("reasoning_effort").as_string() == "max");
        REQUIRE(body.at("thinking").as_object().at("type").as_string() == "enabled");
    }

    SECTION("no reasoning_effort when options.reasoning is unset") {
        auto model = provider->language_model("gpt-4o");
        run(model->do_generate(simple_prompt("hi")), ioc);
        REQUIRE_FALSE(fake->last_request_body.as_object().contains("reasoning_effort"));
    }
}

// DeepSeek streams reasoning_content deltas before the answer content.
TEST_CASE("OpenAI streams reasoning_content before content", "[provider][reasoning]") {
    boost::asio::io_context ioc;
    auto fake = std::make_shared<ai::test::FakeHttpClient>(ioc);
    fake->stream_body =
        R"(data: {"choices":[{"index":0,"delta":{"role":"assistant","reasoning_content":"thinking..."}}]})" "\n\n"
        R"(data: {"choices":[{"index":0,"delta":{"content":"answer"}}]})" "\n\n"
        "data: [DONE]\n\n";

    auto model = make_provider(fake, ioc)->language_model("deepseek-v4-flash");

    auto collect = [](ai::LanguageModelPtr m, boost::asio::io_context&)
        -> ai::Task<std::vector<std::string>> {
        auto r = co_await m->do_stream(ai::CallOptions{});
        std::vector<std::string> out;
        while (auto part = co_await r.stream.next()) {
            if (std::get_if<ai::ReasoningStart>(&*part)) out.push_back("R+");
            else if (auto* d = std::get_if<ai::ReasoningDelta>(&*part)) out.push_back("R:" + d->delta);
            else if (std::get_if<ai::ReasoningEnd>(&*part)) out.push_back("R-");
            else if (std::get_if<ai::TextStart>(&*part)) out.push_back("T+");
            else if (auto* d = std::get_if<ai::TextDelta>(&*part)) out.push_back("T:" + d->delta);
            else if (std::get_if<ai::TextEnd>(&*part)) out.push_back("T-");
        }
        co_return out;
    }(model, ioc);

    auto parts = run(std::move(collect), ioc);

    REQUIRE(parts.size() == 6);
    REQUIRE(parts[0] == "R+");
    REQUIRE(parts[1] == "R:thinking...");
    REQUIRE(parts[2] == "R-");
    REQUIRE(parts[3] == "T+");
    REQUIRE(parts[4] == "T:answer");
    REQUIRE(parts[5] == "T-");
}

// ---------------------------------------------------------------------------
// Streaming usage: the final chunk carries usage with an empty choices array.
// Previously dropped (choices.empty() early-return), so FinishPart.usage was
// always zeros across every OpenAI-compatible provider.
// ---------------------------------------------------------------------------
TEST_CASE("OpenAI streaming surfaces usage from final chunk", "[provider][reasoning]") {
    boost::asio::io_context ioc;
    auto fake = std::make_shared<ai::test::FakeHttpClient>(ioc);
    fake->stream_body =
        R"(data: {"choices":[{"index":0,"delta":{"content":"hi"}}]})" "\n\n"
        R"(data: {"choices":[],"usage":{"prompt_tokens":7,"completion_tokens":3,)"
        R"("prompt_tokens_details":{"cached_tokens":2},)"
        R"("completion_tokens_details":{"reasoning_tokens":1}}})" "\n\n"
        "data: [DONE]\n\n";

    auto model = make_provider(fake, ioc)->language_model("gpt-4o");

    auto collect = [](ai::LanguageModelPtr m, boost::asio::io_context&)
        -> ai::Task<ai::Usage> {
        auto r = co_await m->do_stream(ai::CallOptions{});
        ai::Usage usage{};
        while (auto part = co_await r.stream.next()) {
            if (auto* f = std::get_if<ai::FinishPart>(&*part)) usage = f->usage;
        }
        co_return usage;
    }(model, ioc);

    auto usage = run(std::move(collect), ioc);
    REQUIRE(usage.input_tokens.total.value_or(0) == 7);
    REQUIRE(usage.output_tokens.total.value_or(0) == 3);
    REQUIRE(usage.input_tokens.cache_read.value_or(0) == 2);
    REQUIRE(usage.output_tokens.reasoning.value_or(0) == 1);
}

// reasoning_effort: real OpenAI rejects it on non-reasoning SKUs (gpt-4o) with
// a 400; suppress there, emit for reasoning SKUs and non-OpenAI hosts.
TEST_CASE("OpenAI guards reasoning_effort by host + model", "[provider][reasoning]") {
    boost::asio::io_context ioc;
    auto fake = std::make_shared<ai::test::FakeHttpClient>(ioc);
    fake->json_body = R"({"id":"x","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},)"
                      R"("finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1}})";

    SECTION("suppressed for non-reasoning OpenAI SKU") {
        auto model = make_provider(fake, ioc, "https://api.openai.com/v1")->language_model("gpt-4o");
        auto co = simple_prompt("hi");
        co.reasoning = "high";
        run(model->do_generate(co), ioc);
        REQUIRE_FALSE(fake->last_request_body.as_object().contains("reasoning_effort"));
    }

    SECTION("emitted for reasoning OpenAI SKU") {
        auto model = make_provider(fake, ioc, "https://api.openai.com/v1")->language_model("gpt-5");
        auto co = simple_prompt("hi");
        co.reasoning = "high";
        run(model->do_generate(co), ioc);
        REQUIRE(fake->last_request_body.as_object().at("reasoning_effort").as_string() == "high");
    }
}

// Refusal: content is empty and the reason is in message.refusal.
TEST_CASE("OpenAI surfaces message.refusal as text", "[provider][reasoning]") {
    boost::asio::io_context ioc;
    auto fake = std::make_shared<ai::test::FakeHttpClient>(ioc);
    fake->json_body = R"({"id":"x","choices":[{"index":0,"message":{"role":"assistant",)"
                      R"("content":"","refusal":"I can't help with that."},)"
                      R"("finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1}})";
    auto model = make_provider(fake, ioc)->language_model("gpt-4o");
    auto result = run(model->do_generate(simple_prompt("hi")), ioc);
    REQUIRE(result.text() == "I can't help with that.");
}

// Structured output: json_schema on real OpenAI; auto-downgrade to json_object
// (+ prompt schema injection) on OpenAI-compatible gateways (DeepSeek, z.ai).
namespace {
ai::ResponseFormat number_json_format() {
    return ai::ResponseFormat{
        .type = "json",
        .schema = ai::schema::JsonSchema(boost::json::object{
            {"type", "object"},
            {"properties", boost::json::object{{"n", boost::json::object{{"type", "integer"}}}}},
            {"required", boost::json::array{"n"}},
        }),
        .name = "num",
    };
}
} // namespace

TEST_CASE("OpenAI structured output mode follows host", "[provider][reasoning]") {
    boost::asio::io_context ioc;
    auto fake = std::make_shared<ai::test::FakeHttpClient>(ioc);
    fake->json_body = R"({"id":"x","choices":[{"index":0,"message":{"role":"assistant",)"
                      R"("content":"{\"n\":1}"},"finish_reason":"stop"}],)"
                      R"("usage":{"prompt_tokens":1,"completion_tokens":1}})";

    SECTION("json_schema on OpenAI host") {
        auto model = make_provider(fake, ioc, "https://api.openai.com/v1")->language_model("gpt-4o");
        auto co = simple_prompt("give a number");
        co.response_format = number_json_format();
        run(model->do_generate(co), ioc);
        REQUIRE(fake->last_request_body.as_object().at("response_format").as_object().at("type").as_string() == "json_schema");
    }

    SECTION("json_object + prompt injection on DeepSeek host") {
        auto model = make_provider(fake, ioc, "https://api.deepseek.com")->language_model("deepseek-v4-flash");
        auto co = simple_prompt("give a number");
        co.response_format = number_json_format();
        run(model->do_generate(co), ioc);
        const auto& body = fake->last_request_body.as_object();
        REQUIRE(body.at("response_format").as_object().at("type").as_string() == "json_object");
        const auto& msgs = body.at("messages").as_array();
        REQUIRE_FALSE(msgs.empty());
        REQUIRE(msgs[0].as_object().at("role").as_string() == "system");
    }
}
