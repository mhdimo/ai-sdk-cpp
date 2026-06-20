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
#include <ai/mcp/mcp_client.hpp>
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
