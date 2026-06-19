#include <catch2/catch_test_macros.hpp>

#include "fake_http_client.hpp"

#include <ai/providers/openai/openai.hpp>
#include <ai/providers/google/google.hpp>
#include <ai/model/language_model.hpp>
#include <ai/model/call_options.hpp>
#include <ai/prompt/message.hpp>
#include <ai/stream/async_generator.hpp>

#include <boost/asio.hpp>
#include <boost/json.hpp>

#include <memory>
#include <string>

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

} // namespace

TEST_CASE("OpenAI do_generate parses a chat completion response", "[provider]") {
    boost::asio::io_context ioc;
    auto fake = std::make_shared<ai::test::FakeHttpClient>(ioc);
    fake->json_body = R"({"id":"chatcmpl-1","object":"chat.completion",)"
                      R"("choices":[{"index":0,"message":{"role":"assistant","content":"hello world"},)"
                      R"("finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":2}})";

    auto provider = std::make_shared<ai::providers::openai::OpenAIProvider>(
        ai::providers::openai::OpenAIOptions{
            .api_key = std::string("test-key"),
            .base_url = "https://example.test",
            .io_context = ioc,
            .http_client = fake,
        });
    auto model = provider->language_model("gpt-test");

    auto result = run(model->do_generate(simple_prompt("hi")), ioc);

    REQUIRE(fake->post_json_calls == 1);
    REQUIRE(result.text() == "hello world");
    REQUIRE(result.usage.input_tokens.total.value_or(0) == 5);
    REQUIRE(result.usage.output_tokens.total.value_or(0) == 2);
}

TEST_CASE("Google do_generate parses a generateContent response", "[provider]") {
    boost::asio::io_context ioc;
    auto fake = std::make_shared<ai::test::FakeHttpClient>(ioc);
    fake->json_body = R"({"candidates":[{"content":{"role":"model",)"
                      R"("parts":[{"text":"hi from gemini"}]},"finishReason":"STOP"}],)"
                      R"("usageMetadata":{"promptTokenCount":4,"candidatesTokenCount":3}})";

    auto provider = std::make_shared<ai::providers::google::GoogleProvider>(
        ai::providers::google::GoogleOptions{
            .api_key = std::string("test-key"),
            .base_url = "https://example.test",
            .io_context = ioc,
            .http_client = fake,
        });
    auto model = provider->language_model("gemini-test");

    auto result = run(model->do_generate(simple_prompt("hi")), ioc);

    REQUIRE(fake->post_json_calls == 1);
    REQUIRE(result.text() == "hi from gemini");
    REQUIRE(result.usage.input_tokens.total.value_or(0) == 4);
    REQUIRE(result.usage.output_tokens.total.value_or(0) == 3);
}
