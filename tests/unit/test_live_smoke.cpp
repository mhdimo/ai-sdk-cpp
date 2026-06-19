// Live (network) smoke test. OFFLINE BY DEFAULT: it SKIPs unless the env var
// AI_SDK_RUN_LIVE=1 is set (and a provider API key is available). This keeps the
// normal ctest run fully offline while giving CI/developers an easy way to
// verify a real end-to-end call.
//
// Run it manually:
//   AI_SDK_RUN_LIVE=1 ANTHROPIC_API_KEY=sk-... \
//     ./build/tests/unit/ai-sdk-tests "[live]"

#include <catch2/catch_test_macros.hpp>

#include <ai/coding_agent.hpp>
#include <ai/providers/anthropic/anthropic.hpp>

#include <boost/asio.hpp>

#include <cstdlib>
#include <iostream>
#include <string>

namespace {

template <typename T>
T run(ai::Task<T> task, boost::asio::io_context& ioc) {
    task.start();
    while (!task.done()) {
        ioc.run_one();
    }
    return task.get();
}

bool live_enabled() {
    const char* flag = std::getenv("AI_SDK_RUN_LIVE");
    return flag && std::string(flag) == "1";
}

std::string env_or(const char* name, const char* fallback) {
    const char* v = std::getenv(name);
    return (v && *v) ? std::string(v) : std::string(fallback);
}

} // namespace

TEST_CASE("CodingAgent end-to-end against a live provider", "[live][smoke]") {
    if (!live_enabled()) {
        SKIP("set AI_SDK_RUN_LIVE=1 and a provider API key to run live tests");
    }

    boost::asio::io_context ioc;

    ai::providers::anthropic::AnthropicOptions opts{.io_context = ioc};
    std::string base = env_or("ANTHROPIC_BASE_URL", "https://api.anthropic.com");
    opts.base_url = base;

    auto provider = ai::providers::anthropic::create_anthropic(std::move(opts));
    std::string model_id = env_or("ANTHROPIC_MODEL", "claude-sonnet-4-5");
    auto model = provider->language_model(model_id);

    ai::CodingAgentOptions copts;
    copts.model = model;
    copts.instructions = "You are terse. Follow instructions exactly.";
    ai::CodingAgent agent(copts);

    auto result = run(agent.send("Reply with exactly one word: pong"), ioc);

    REQUIRE_FALSE(result.text.empty());
    std::cout << "[live] model replied: " << result.text << "\n";
    std::cout << "[live] usage: "
              << result.usage.input_tokens.total.value_or(0) << " in / "
              << result.usage.output_tokens.total.value_or(0) << " out\n";
}
