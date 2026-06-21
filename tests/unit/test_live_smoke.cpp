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
#include <ai/providers/openai/openai.hpp>
#include <ai/providers/zai/zai.hpp>
#include <ai/providers/moonshotai/moonshotai.hpp>
#include <ai/mcp/mcp_client.hpp>
#include <ai/model/call_options.hpp>
#include <ai/prompt/message.hpp>
#include <boost/json.hpp>

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

TEST_CASE("OpenAI provider surfaces DeepSeek reasoning_content live", "[live][deepseek]") {
    if (!live_enabled()) {
        SKIP("set AI_SDK_RUN_LIVE=1 and DEEPSEEK_API_KEY to run the live DeepSeek test");
    }
    const char* key = std::getenv("DEEPSEEK_API_KEY");
    if (!key || !*key) {
        SKIP("set DEEPSEEK_API_KEY for the live DeepSeek reasoning test");
    }

    boost::asio::io_context ioc;
    std::string base = env_or("DEEPSEEK_BASE_URL", "https://api.deepseek.com");
    auto provider = ai::providers::openai::create_openai(
        ai::providers::openai::OpenAIOptions{
            .api_key = std::string(key),
            .base_url = base,
            .io_context = ioc,
        });
    std::string model_id = env_or("DEEPSEEK_MODEL", "deepseek-reasoner");
    auto model = provider->language_model(model_id);

    ai::CallOptions co;
    ai::UserContent uc;
    uc.push_back(ai::TextPart{.text = "What is 7 * 13? Show your work."});
    co.prompt.push_back(ai::UserMessage{.content = std::move(uc)});

    auto result = run(model->do_generate(std::move(co)), ioc);

    REQUIRE_FALSE(result.text().empty());
    bool has_reasoning = false;
    for (auto& c : result.content) {
        if (std::get_if<ai::ReasoningContent>(&c)) has_reasoning = true;
    }
    std::cout << "[live] DeepSeek " << model_id << " replied: " << result.text() << "\n";
    std::cout << "[live] reasoning_content surfaced: " << (has_reasoning ? "yes" : "no") << "\n";
    std::cout << "[live] usage: " << result.usage.input_tokens.total.value_or(0)
              << " in / " << result.usage.output_tokens.total.value_or(0) << " out\n";
}

TEST_CASE("MCP client against a real streamable-HTTP server (zread)", "[live][mcp]") {
    if (!live_enabled()) {
        SKIP("set AI_SDK_RUN_LIVE=1, MCP_SERVER_URL, and MCP_SERVER_AUTH to run live MCP");
    }
    const char* url_env = std::getenv("MCP_SERVER_URL");
    const char* auth_env = std::getenv("MCP_SERVER_AUTH");
    if (!url_env) {
        SKIP("set MCP_SERVER_URL (and MCP_SERVER_AUTH) for the live MCP test");
    }

    boost::asio::io_context ioc;

    ai::mcp::McpServerConfig cfg;
    cfg.name = "zread";
    cfg.transport = "http";
    cfg.url = url_env;
    if (auth_env) {
        std::string auth = auth_env;
        cfg.headers["Authorization"] =
            (auth.rfind("Bearer ", 0) == 0) ? auth : ("Bearer " + auth);
    }

    ai::mcp::McpClient client(cfg);
    run(client.connect(), ioc);  // initialize handshake
    REQUIRE(client.is_connected());

    auto tools = run(client.list_tools(), ioc);
    std::cout << "[live] MCP tools: " << tools.size() << "\n";
    for (auto& t : tools) {
        std::cout << "[live]   - " << t.name << "\n";
    }
    REQUIRE(!tools.empty());

    auto result = run(client.call_tool(
        "get_repo_structure",
        boost::json::object{{"repo_name", "anthropics/anthropic-sdk-python"}}), ioc);

    std::string serialized = boost::json::serialize(result);
    std::cout << "[live] get_repo_structure -> "
              << serialized.substr(0, std::min(serialized.size(), size_t(200))) << "\n";
    REQUIRE_FALSE(serialized.empty());
}

TEST_CASE("MCP stdio client against a real subprocess server", "[live][mcp][stdio]") {
    if (!live_enabled()) {
        SKIP("set AI_SDK_RUN_LIVE=1 and Z_AI_API_KEY to run the live stdio MCP test");
    }
    const char* api_key = std::getenv("Z_AI_API_KEY");
    if (!api_key || !*api_key) {
        SKIP("set Z_AI_API_KEY for the live stdio MCP test");
    }

    boost::asio::io_context ioc;

    ai::mcp::McpServerConfig cfg;
    cfg.name = "zai-mcp-server";
    cfg.transport = "stdio";
    cfg.command = "npx";
    cfg.args = {"-y", "@z_ai/mcp-server"};
    cfg.env = {{"Z_AI_API_KEY", std::string(api_key)}, {"Z_AI_MODE", "ZAI"}};

    ai::mcp::McpClient client(cfg);
    run(client.connect(), ioc);  // spawn subprocess + initialize handshake
    REQUIRE(client.is_connected());

    auto tools = run(client.list_tools(), ioc);
    std::cout << "[live stdio] MCP tools: " << tools.size() << "\n";
    for (auto& t : tools) {
        std::cout << "[live stdio]   - " << t.name << "\n";
    }
    REQUIRE(!tools.empty());
}

// z.ai GLM via the first-class zai wrapper. The [1m] Claude-Code alias must be
// stripped (else [1211] Unknown Model); auth is Bearer via ZAI_API_KEY.
TEST_CASE("zai provider strips [1m] alias and reaches GLM", "[live][zai]") {
    if (!live_enabled()) SKIP("set AI_SDK_RUN_LIVE=1 and ZAI_API_KEY");
    const char* key = std::getenv("ZAI_API_KEY");
    if (!key || !*key) SKIP("set ZAI_API_KEY for the live zai test");

    boost::asio::io_context ioc;
    auto provider = ai::providers::zai::create_zai(
        ai::providers::zai::ZaiOptions{.io_context = ioc});
    // [1m] alias — the zai provider must strip it to glm-5.2.
    auto model = provider->language_model("glm-5.2[1m]");

    ai::CallOptions co;
    ai::UserContent uc;
    uc.push_back(ai::TextPart{.text = "Reply with exactly one word: pong"});
    co.prompt.push_back(ai::UserMessage{.content = std::move(uc)});

    auto result = run(model->do_generate(std::move(co)), ioc);
    REQUIRE_FALSE(result.text().empty());
    std::cout << "[live] zai glm-5.2[1m] -> " << result.text() << "\n";
}

// DeepSeek has no json_schema support; the SDK auto-downgrades to json_object +
// prompt-injected schema. The model must still return valid JSON.
TEST_CASE("DeepSeek structured output downgrades to json_object", "[live][deepseek]") {
    if (!live_enabled()) SKIP("set AI_SDK_RUN_LIVE=1 and DEEPSEEK_API_KEY");
    const char* key = std::getenv("DEEPSEEK_API_KEY");
    if (!key || !*key) SKIP("set DEEPSEEK_API_KEY");

    boost::asio::io_context ioc;
    std::string base = env_or("DEEPSEEK_BASE_URL", "https://api.deepseek.com");
    auto provider = ai::providers::openai::create_openai(
        ai::providers::openai::OpenAIOptions{
            .api_key = std::string(key), .base_url = base, .io_context = ioc});
    auto model = provider->language_model(env_or("DEEPSEEK_MODEL", "deepseek-v4-flash"));

    ai::CallOptions co;
    ai::UserContent uc;
    uc.push_back(ai::TextPart{.text = "Pick a small integer for n."});
    co.prompt.push_back(ai::UserMessage{.content = std::move(uc)});
    co.response_format = ai::ResponseFormat{
        .type = "json",
        .schema = ai::schema::JsonSchema::object({{"n", ai::schema::JsonSchema::integer()}}).required({"n"}),
        .name = std::string("num"),
    };

    auto result = run(model->do_generate(std::move(co)), ioc);
    auto parsed = boost::json::parse(result.text());
    REQUIRE(parsed.as_object().contains("n"));
    std::cout << "[live] DeepSeek structured -> " << result.text() << "\n";
}

// Moonshot (Kimi) via the OpenAI-compatible wrapper — same code path as
// OpenAI/DeepSeek, so this also exercises reasoning_content surfacing etc.
TEST_CASE("Moonshot (Kimi) via OpenAI-compatible wrapper", "[live][moonshot]") {
    if (!live_enabled()) SKIP("set AI_SDK_RUN_LIVE=1 and MOONSHOT_API_KEY");
    const char* key = std::getenv("MOONSHOT_API_KEY");
    if (!key || !*key) SKIP("set MOONSHOT_API_KEY");

    boost::asio::io_context ioc;
    ai::providers::moonshotai::MoonshotAIOptions o{.io_context = ioc};
    std::string base = env_or("MOONSHOT_BASE_URL", "https://api.moonshot.ai/v1");
    if (!base.empty()) o.base_url = base;
    auto provider = ai::providers::moonshotai::create_moonshotai(std::move(o));
    auto model = provider->language_model(env_or("MOONSHOT_MODEL", "kimi-k2.6"));

    ai::CallOptions co;
    ai::UserContent uc;
    uc.push_back(ai::TextPart{.text = "Reply with exactly one word: pong"});
    co.prompt.push_back(ai::UserMessage{.content = std::move(uc)});

    auto result = run(model->do_generate(std::move(co)), ioc);
    REQUIRE_FALSE(result.text().empty());
    std::cout << "[live] moonshot replied: " << result.text() << "\n";
}
