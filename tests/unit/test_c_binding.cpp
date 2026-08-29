#include <catch2/catch_test_macros.hpp>

#include "ai_sdk.h"

#include <string>

namespace {

struct Stack {
    ai_context_t ctx = nullptr;
    ai_provider_t provider = nullptr;
    ai_model_t model = nullptr;
    ai_tool_set_t tools = nullptr;

    Stack() {
        ctx = ai_context_create();
        if (!ctx) return;
        ai_provider_options_t popts{};
        popts.api_key = "test-key";
        provider = ai_provider_create(ctx, "anthropic", popts);
        if (!provider) return;
        model = ai_model_create(provider, "claude-sonnet-4-5");
        if (!model) return;
        tools = ai_tool_set_create();
    }

    ~Stack() {
        if (tools) ai_tool_set_destroy(tools);
        if (model) ai_model_destroy(model);
        if (provider) ai_provider_destroy(provider);
        if (ctx) ai_context_destroy(ctx);
    }

    ai_agent_t make_agent(const char* provider_options_json) {
        ai_agent_options_t aopts{};
        aopts.model = model;
        aopts.tools = tools;
        aopts.instructions = "test";
        aopts.max_steps = 3;
        aopts.on_event = nullptr;
        aopts.user_data = nullptr;
        aopts.provider_options_json = provider_options_json;
        return ai_agent_create(aopts);
    }
};

} // namespace

TEST_CASE("ai_agent_create accepts provider_options_json", "[c_binding]") {
    Stack s;
    REQUIRE(s.ctx);
    REQUIRE(s.provider);
    REQUIRE(s.model);
    REQUIRE(s.tools);

    ai_agent_t agent = s.make_agent(R"({"anthropic":{"builtinTools":[]}})");
    REQUIRE(agent != nullptr);
    ai_agent_destroy(agent);
}

TEST_CASE("ai_agent_create treats NULL provider_options_json as none", "[c_binding]") {
    Stack s;
    REQUIRE(s.model);

    ai_agent_t agent = s.make_agent(nullptr);
    REQUIRE(agent != nullptr);
    ai_agent_destroy(agent);
}

TEST_CASE("ai_agent_create rejects malformed provider_options_json", "[c_binding]") {
    Stack s;
    REQUIRE(s.model);

    ai_agent_t agent = s.make_agent("{not json");
    CHECK(agent == nullptr);
    REQUIRE(ai_last_error(s.ctx) != nullptr);
    CHECK(std::string(ai_last_error(s.ctx)).empty() == false);
}

TEST_CASE("ai_agent_create rejects non-object provider_options_json", "[c_binding]") {
    Stack s;
    REQUIRE(s.model);

    ai_agent_t agent = s.make_agent("[1,2,3]");
    CHECK(agent == nullptr);
    REQUIRE(ai_last_error(s.ctx) != nullptr);
    CHECK(std::string(ai_last_error(s.ctx)).find("JSON object") != std::string::npos);
}
