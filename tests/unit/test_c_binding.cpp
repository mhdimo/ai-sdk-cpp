#include <catch2/catch_test_macros.hpp>

#include "ai_sdk.h"
#include "local_http_server.hpp"

#include <boost/json.hpp>

#include <array>
#include <atomic>
#include <ctime>
#include <string>
#include <vector>

namespace json = boost::json;

namespace {

struct Stack {
    ai_context_t ctx = nullptr;
    ai_provider_t provider = nullptr;
    ai_model_t model = nullptr;
    ai_tool_set_t tools = nullptr;

    /// `base_url` NULL keeps the provider's default host; a test that needs to
    /// watch a call fail points it at an address with nothing behind it.
    explicit Stack(const char* base_url = nullptr) {
        ctx = ai_context_create();
        if (!ctx) return;
        ai_provider_options_t popts{};
        popts.api_key = "test-key";
        popts.base_url = base_url;
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

/// A base URL with nothing listening on it. Binding to port 0 and closing hands
/// back a port the OS just proved was free, so the connect is refused at once
/// rather than hanging until the request times out.
std::string unreachable_base_url() {
    boost::asio::io_context ioc;
    const boost::asio::ip::tcp::acceptor acc(
        ioc, boost::asio::ip::tcp::endpoint(boost::asio::ip::tcp::v4(), 0));
    return "http://127.0.0.1:" + std::to_string(acc.local_endpoint().port());
}

/// What a stream callback was handed, reduced to the questions that matter:
/// was it told the turn failed, was it told the turn finished, and what did
/// the failure say.
struct StreamTally {
    std::vector<ai_stream_event_type_t> types;
    std::string error_text;

    static void on_event(ai_stream_event_t event, void* user_data) {
        auto* self = static_cast<StreamTally*>(user_data);
        self->types.push_back(event.type);
        if (event.type == AI_STREAM_ERROR && event.text) {
            self->error_text = event.text;
        }
    }

    bool saw(ai_stream_event_type_t type) const {
        return std::find(types.begin(), types.end(), type) != types.end();
    }

    /// A turn that failed has to say so, and has to *end* on the failure: an
    /// error followed by a finish would leave a consumer that stops at the
    /// first terminal event believing the turn completed.
    void check_reports_failure() const {
        INFO("events emitted: " << types.size() << ", error text: " << error_text);
        // REQUIRE, not CHECK: back() below needs an element, and the failure
        // this is here to catch is a stream that emitted nothing at all. Left
        // as a CHECK the test does not report that, it segfaults on the empty
        // vector and takes the rest of the run with it.
        REQUIRE_FALSE(types.empty());
        CHECK(saw(AI_STREAM_ERROR));
        CHECK_FALSE(saw(AI_STREAM_FINISH));
        CHECK(types.back() == AI_STREAM_ERROR);
        CHECK_FALSE(error_text.empty());
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

TEST_CASE("an MCP config without a usable name cannot produce a toolset", "[c_binding][mcp]") {
    // The name qualifies every tool the server exposes, and that qualified
    // string is what a policy or approver is keyed on. A toolset whose tools
    // have only bare names is one another server can shadow, so the binding
    // refuses to build it rather than handing back something unaddressable.
    ai_context_t ctx = ai_context_create();
    REQUIRE(ctx != nullptr);

    for (const char* config : {
             R"({"transport":"stdio","command":"/bin/cat"})",          // absent
             R"({"name":"","transport":"stdio","command":"/bin/cat"})", // empty
             R"({"name":42,"transport":"stdio","command":"/bin/cat"})",  // wrong type
         }) {
        INFO("config: " << config);
        CHECK(ai_mcp_toolset_from_server(ctx, config) == nullptr);
        REQUIRE(ai_last_error(ctx) != nullptr);
        CHECK(std::string(ai_last_error(ctx)).find("name") != std::string::npos);
    }

    ai_context_destroy(ctx);
}

/* ---------------------------------------------------------------------------
 * Permission gating, driven end to end through the C API.
 *
 * The decision constants are an enum in the header and an int in the callback
 * signature, so every mapping between the two is a place where a comparison can
 * be inverted without anything failing to compile. The only way to catch that
 * is to run a tool and see whether it ran — which needs a model that asks for
 * one, which needs a socket. Hence the loopback server.
 * ------------------------------------------------------------------------ */

namespace {

/// Canned OpenAI chat-completion bodies. The agent loop is generate-only
/// (ToolLoopAgent::generate -> generate_text), so these are plain JSON rather
/// than SSE frames.
const char* kToolCallBody = R"({
  "id": "chatcmpl-1", "object": "chat.completion", "created": 1, "model": "gpt-4o",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": null,
      "tool_calls": [{"id": "call_1", "type": "function",
                      "function": {"name": "probe", "arguments": "{}"}}]},
    "finish_reason": "tool_calls"
  }],
  "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
})";

const char* kFinalBody = R"({
  "id": "chatcmpl-2", "object": "chat.completion", "created": 2, "model": "gpt-4o",
  "choices": [{"index": 0,
    "message": {"role": "assistant", "content": "done"},
    "finish_reason": "stop"}],
  "usage": {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15}
})";

const char* kProbeSchema = R"({"type":"object","properties":{},"additionalProperties":false})";

/// What the tool itself records. `runs` is the whole point: "was the tool
/// allowed to execute" is not otherwise observable from outside the agent.
struct Probe {
    std::atomic<int> runs{0};
};

ai_tool_result_t probe_tool(const char*, const char*, void* user_data) {
    static_cast<Probe*>(user_data)->runs.fetch_add(1);
    return ai_tool_result_t{R"({"ok":true})", 0};
}

TEST_CASE("a tool set describes its tools as JSON", "[c_binding][toolset]") {
    ai_tool_set_t tools = ai_tool_set_create();
    REQUIRE(tools != nullptr);
    REQUIRE(ai_tool_set_add(tools, "probe", "records that it ran", kProbeSchema, probe_tool,
                            nullptr) == AI_OK);

    ai_tool_set_description_t desc{};
    REQUIRE(ai_tool_set_describe_json(tools, &desc) == AI_OK);
    CHECK(desc.count == 1);

    // Parsed rather than substring-matched: the claim is about the shape a
    // consumer can rely on, and "does the text contain this word" would pass
    // for output that is not JSON at all.
    auto parsed = json::parse(desc.json);
    REQUIRE(parsed.is_array());
    REQUIRE(parsed.as_array().size() == 1);

    const auto& entry = parsed.as_array()[0].as_object();
    CHECK(entry.at("name") == "probe");
    CHECK(entry.at("description") == "records that it ran");
    CHECK(entry.at("input_schema").as_object().at("type") == "object");

    ai_tool_set_description_free(&desc);
    CHECK(desc.json == nullptr);
    CHECK(desc.count == 0);
    ai_tool_set_destroy(tools);
}

TEST_CASE("a tool set describes its tools in name order", "[c_binding][toolset]") {
    // A ToolSet is an unordered_map underneath, so without the sort this comes
    // out in a different order every run and no two descriptions can be
    // compared.
    ai_tool_set_t tools = ai_tool_set_create();
    REQUIRE(tools != nullptr);
    for (const char* name : {"zeta", "alpha", "mid"}) {
        REQUIRE(ai_tool_set_add(tools, name, "", kProbeSchema, probe_tool, nullptr) == AI_OK);
    }

    ai_tool_set_description_t desc{};
    REQUIRE(ai_tool_set_describe_json(tools, &desc) == AI_OK);
    REQUIRE(desc.count == 3);

    auto parsed = json::parse(desc.json).as_array();
    CHECK(parsed[0].as_object().at("name") == "alpha");
    CHECK(parsed[1].as_object().at("name") == "mid");
    CHECK(parsed[2].as_object().at("name") == "zeta");

    ai_tool_set_description_free(&desc);
    ai_tool_set_destroy(tools);
}

TEST_CASE("an empty tool set describes itself as an empty array", "[c_binding][toolset]") {
    ai_tool_set_t tools = ai_tool_set_create();
    REQUIRE(tools != nullptr);

    ai_tool_set_description_t desc{};
    REQUIRE(ai_tool_set_describe_json(tools, &desc) == AI_OK);
    CHECK(desc.count == 0);
    CHECK(std::string(desc.json) == "[]");

    ai_tool_set_description_free(&desc);
    ai_tool_set_destroy(tools);
}

TEST_CASE("describing a tool set rejects null arguments and frees safely",
          "[c_binding][toolset]") {
    ai_tool_set_t tools = ai_tool_set_create();
    REQUIRE(tools != nullptr);

    ai_tool_set_description_t desc{};
    CHECK(ai_tool_set_describe_json(nullptr, &desc) == AI_ERROR_INVALID_ARGUMENT);
    CHECK(ai_tool_set_describe_json(tools, nullptr) == AI_ERROR_INVALID_ARGUMENT);

    // Free gets called on a zeroed struct by callers who checked the status and
    // bailed, and on an already-freed one by callers who did not. Neither may
    // touch the heap.
    ai_tool_set_description_t zeroed{};
    ai_tool_set_description_free(&zeroed);
    ai_tool_set_description_free(nullptr);

    REQUIRE(ai_tool_set_describe_json(tools, &desc) == AI_OK);
    ai_tool_set_description_free(&desc);
    ai_tool_set_description_free(&desc);

    ai_tool_set_destroy(tools);
}

/// An agent whose provider talks to a loopback server, wired up with a
/// permission-gated copy of a single `probe` tool.
struct GatedAgent {
    ai::test::LocalHttpServer server;
    ai_context_t ctx = nullptr;
    ai_provider_t provider = nullptr;
    ai_model_t model = nullptr;
    ai_tool_set_t raw_tools = nullptr;
    ai_tool_set_t gated_tools = nullptr;
    ai_agent_t agent = nullptr;
    Probe probe;

    GatedAgent(std::vector<std::string> bodies, ai_permission_policy_fn policy, void* policy_ud,
               ai_approver_fn approver, void* approver_ud,
               std::chrono::milliseconds think_time = std::chrono::milliseconds{0})
        : server(std::move(bodies), think_time) {
        ctx = ai_context_create();
        if (!ctx) return;

        // base_url has to outlive the create call: the C struct takes a bare
        // const char*, so a temporary .c_str() here would dangle.
        std::string base = server.base_url();
        ai_provider_options_t popts{};
        popts.api_key = "test-key";
        popts.base_url = base.c_str();
        provider = ai_provider_create(ctx, "openai", popts);
        if (!provider) return;

        model = ai_model_create(provider, "gpt-4o");
        if (!model) return;

        raw_tools = ai_tool_set_create();
        if (!raw_tools) return;
        if (ai_tool_set_add(raw_tools, "probe", "records that it ran", kProbeSchema, probe_tool,
                            &probe) != AI_OK) {
            return;
        }

        gated_tools = ai_with_permissions_approver(raw_tools, policy, policy_ud, approver,
                                                   approver_ud);
        if (!gated_tools) return;

        ai_agent_options_t aopts{};
        aopts.model = model;
        aopts.tools = gated_tools;
        aopts.instructions = "test";
        aopts.max_steps = 4;
        agent = ai_agent_create(aopts);
    }

    ~GatedAgent() {
        if (agent) ai_agent_destroy(agent);
        if (gated_tools) ai_tool_set_destroy(gated_tools);
        if (raw_tools) ai_tool_set_destroy(raw_tools);
        if (model) ai_model_destroy(model);
        if (provider) ai_provider_destroy(provider);
        if (ctx) ai_context_destroy(ctx);
    }

    GatedAgent(const GatedAgent&) = delete;
    GatedAgent& operator=(const GatedAgent&) = delete;

    bool ready() const { return agent != nullptr; }

    /// One full agent run: tool-call turn, then a final answer.
    ai_status_t call(std::string* out_text = nullptr) {
        ai_generate_result_t r{};
        ai_status_t status = ai_agent_call(agent, "go", &r);
        if (out_text && r.text) *out_text = r.text;
        ai_generate_result_free(&r);
        return status;
    }
};

/// A policy that always asks, so the approver is the only thing deciding.
int always_ask(const char*, const char*, char*, void*) { return AI_PERMISSION_ASK; }

/// Writes into the SDK's reason buffer the way the header tells callers to.
void write_reason(char* buf, const std::string& text) {
    if (buf) {
        std::snprintf(buf, AI_REASON_MAX, "%s", text.c_str());
    }
}

/// Counts how many times the approver was consulted, on top of the answer.
struct ApproverState {
    std::atomic<int> calls{0};
    int answer = AI_PERMISSION_DENY;
    /// Written into the reason buffer when non-empty.
    std::string reason;
};

int counting_approver(const char*, const char*, const char*, char* reason, void* user_data) {
    auto* st = static_cast<ApproverState*>(user_data);
    st->calls.fetch_add(1);
    if (!st->reason.empty()) {
        write_reason(reason, st->reason);
    }
    return st->answer;
}

} // namespace

TEST_CASE("a policy that allows runs the tool", "[c_binding][permission]") {
    int policy_calls = 0;
    // The `-> int` is load-bearing: AI_PERMISSION_ALLOW belongs to an unnamed
    // enum, so without it the lambda deduces that enum as its return type and
    // stops converting to ai_permission_policy_fn.
    auto allow = [](const char*, const char*, char*, void* ud) -> int {
        ++*static_cast<int*>(ud);
        return AI_PERMISSION_ALLOW;
    };

    GatedAgent a({kToolCallBody, kFinalBody}, allow, &policy_calls, nullptr, nullptr);
    REQUIRE(a.ready());

    CHECK(a.call() == AI_OK);
    CHECK(a.probe.runs.load() == 1);
    CHECK(policy_calls == 1);
}

TEST_CASE("a policy that denies keeps the tool from running", "[c_binding][permission]") {
    auto deny = [](const char*, const char*, char*, void*) -> int { return AI_PERMISSION_DENY; };

    GatedAgent a({kToolCallBody, kFinalBody}, deny, nullptr, nullptr, nullptr);
    REQUIRE(a.ready());

    // The run still succeeds: a denied tool comes back to the model as an error
    // result, it does not abort the agent.
    CHECK(a.call() == AI_OK);
    CHECK(a.probe.runs.load() == 0);
}

TEST_CASE("a reason written to the buffer reaches the model", "[c_binding][permission]") {
    // The reason is produced inside the callback and consumed by the core, so
    // the only proof that the C layer transfers it correctly is to read the
    // request the model was sent. A stub that echoes the buffer back would pass
    // even if the binding dropped it or copied the wrong bytes.
    auto deny = [](const char*, const char*, char* reason, void*) -> int {
        write_reason(reason, "the path is outside the project root");
        return AI_PERMISSION_DENY;
    };

    GatedAgent a({kToolCallBody, kFinalBody}, deny, nullptr, nullptr, nullptr);
    REQUIRE(a.ready());

    CHECK(a.call() == AI_OK);
    CHECK(a.probe.runs.load() == 0);
    REQUIRE(a.server.requests_served() == 2);

    // Second request = the one carrying the tool result back to the model.
    const std::string followup = a.server.request_body(1);
    REQUIRE(followup.find("permission_denied") != std::string::npos);
    CHECK(followup.find("the path is outside the project root") != std::string::npos);
}

TEST_CASE("a callback that fills the buffer exactly does not over-read", "[c_binding][permission]") {
    // The buffer arrives zeroed but a caller can legitimately write every byte
    // of it with no NUL -- strncpy of a full-length string, say. Reading that
    // with strlen would run off the end, so the SDK caps the copy. This asserts
    // the cap holds and the run survives it.
    auto deny = [](const char*, const char*, char* reason, void*) -> int {
        if (reason) {
            for (int i = 0; i < AI_REASON_MAX; ++i) {
                reason[i] = 'x';
            }
        }
        return AI_PERMISSION_DENY;
    };

    GatedAgent a({kToolCallBody, kFinalBody}, deny, nullptr, nullptr, nullptr);
    REQUIRE(a.ready());

    CHECK(a.call() == AI_OK);
    CHECK(a.probe.runs.load() == 0);

    // Truncated to one byte short of the buffer, and stopping at the first NUL
    // the SDK guarantees.
    const std::string followup = a.server.request_body(1);
    REQUIRE(followup.find("permission_denied") != std::string::npos);
    CHECK(followup.find(std::string(AI_REASON_MAX, 'x')) == std::string::npos);
    CHECK(followup.find(std::string(AI_REASON_MAX - 1, 'x')) != std::string::npos);
}

TEST_CASE("an approver decides when the policy asks", "[c_binding][permission]") {
    SECTION("allow") {
        ApproverState st;
        st.answer = AI_PERMISSION_ALLOW;

        GatedAgent a({kToolCallBody, kFinalBody}, always_ask, nullptr, counting_approver, &st);
        REQUIRE(a.ready());
        CHECK(a.call() == AI_OK);
        CHECK(a.probe.runs.load() == 1);
        CHECK(st.calls.load() == 1);
    }

    SECTION("deny") {
        ApproverState st;
        st.answer = AI_PERMISSION_DENY;

        GatedAgent a({kToolCallBody, kFinalBody}, always_ask, nullptr, counting_approver, &st);
        REQUIRE(a.ready());
        CHECK(a.call() == AI_OK);
        CHECK(a.probe.runs.load() == 0);
        CHECK(st.calls.load() == 1);
    }

    SECTION("no approver at all fails closed") {
        // Documented behaviour of ai_with_permissions: Ask with nothing to ask
        // is a Deny, not an Allow.
        GatedAgent a({kToolCallBody, kFinalBody}, always_ask, nullptr, nullptr, nullptr);
        REQUIRE(a.ready());
        CHECK(a.call() == AI_OK);
        CHECK(a.probe.runs.load() == 0);
    }
}

TEST_CASE("ALLOW_ALWAYS approves the tool for the rest of the toolset's life",
          "[c_binding][permission]") {
    ApproverState st;
    st.answer = AI_PERMISSION_ALLOW_ALWAYS;

    // Four bodies: two complete runs, each asking for the tool once.
    GatedAgent a({kToolCallBody, kFinalBody, kToolCallBody, kFinalBody}, always_ask, nullptr,
                 counting_approver, &st);
    REQUIRE(a.ready());

    CHECK(a.call() == AI_OK);
    CHECK(a.probe.runs.load() == 1);
    CHECK(st.calls.load() == 1);

    // Second run, same toolset: the allowlist short-circuits before the policy
    // and the approver, so both counts stay put while the tool still executes.
    CHECK(a.call() == AI_OK);
    CHECK(a.probe.runs.load() == 2);
    CHECK(st.calls.load() == 1);
}

TEST_CASE("an out-of-contract approver answer is treated as deny", "[c_binding][permission]") {
    // The header promises Allow / Allow-Always / Deny and says anything else is
    // Deny. These are the values a caller reaches by accident — a -1 default, a
    // stray status code, or the policy's own ASK constant returned from the
    // wrong callback. Each must fail closed.
    for (const int answer : std::array<int, 4>{-1, 42, 7, static_cast<int>(AI_PERMISSION_ASK)}) {
        ApproverState st;
        st.answer = answer;

        GatedAgent a({kToolCallBody, kFinalBody}, always_ask, nullptr, counting_approver, &st);
        REQUIRE(a.ready());

        INFO("approver returned: " << answer);
        CHECK(a.call() == AI_OK);
        CHECK(a.probe.runs.load() == 0);
        CHECK(st.calls.load() == 1);
    }
}

TEST_CASE("the approver is given the tool name and a rationale", "[c_binding][permission]") {
    struct Seen {
        std::string tool;
        std::string rationale;
    };
    Seen seen;

    auto approver = [](const char* tool, const char*, const char* rationale, char*,
                       void* ud) -> int {
        auto* s = static_cast<Seen*>(ud);
        s->tool = tool ? tool : "";
        s->rationale = rationale ? rationale : "";
        return AI_PERMISSION_ALLOW;
    };

    GatedAgent a({kToolCallBody, kFinalBody}, always_ask, nullptr, approver, &seen);
    REQUIRE(a.ready());
    CHECK(a.call() == AI_OK);

    CHECK(seen.tool == "probe");
    CHECK(seen.rationale.find("probe") != std::string::npos);
}

TEST_CASE("waiting on a slow model costs almost no CPU", "[c_binding][perf]") {
    // The C API drives a coroutine to completion by polling an io_context,
    // because a step that is waiting on the network is resumed from the thread
    // that owns the pending future rather than from the loop. That makes the
    // loop unable to simply block, and the failure mode when it is written
    // naively is a tight spin: right answers, a full core for every concurrent
    // request, for as long as the model takes to answer.
    //
    // Measured at ~0.5% of a core against a server that holds each response.
    // The bound is deliberately loose — this is a smoke alarm, not a
    // benchmark. What it catches is the ~100% of a core a spin costs.
    auto allow = [](const char*, const char*, char*, void*) -> int {
        return AI_PERMISSION_ALLOW;
    };

    constexpr auto kThink = std::chrono::milliseconds{200};
    GatedAgent a({kToolCallBody, kFinalBody}, allow, nullptr, nullptr, nullptr, kThink);
    REQUIRE(a.ready());

    const auto cpu_before = std::clock();
    const auto wall_before = std::chrono::steady_clock::now();
    CHECK(a.call() == AI_OK);
    const std::chrono::duration<double> wall = std::chrono::steady_clock::now() - wall_before;
    const double cpu_s = static_cast<double>(std::clock() - cpu_before) / CLOCKS_PER_SEC;
    const double wall_s = wall.count();

    CHECK(a.probe.runs.load() == 1);

    // Two turns, each held up by the server. Without this the ratio below is
    // only testing that a fast machine is fast.
    INFO("wall " << wall_s << "s, cpu " << cpu_s << "s");
    REQUIRE(wall_s > 0.3);
    CHECK(cpu_s < wall_s / 2);
}

// Every streaming entry point has to report a failure that the stream itself
// cannot see. A transport failure produces no response at all, so nothing
// downstream can turn it into an event; if the entry point returns its error
// status and emits nothing, the consumer is handed a stream that simply ended.
//
// That is what `ai_session_send_stream` did. The Node binding, whose fallback
// exists to stop a consumer waiting forever, filled the gap with a *finish* —
// so the caller got a turn that had succeeded and produced nothing: zero
// tokens, empty finishReason, no error, and nothing to tell it apart from a
// model that answered with silence.
//
// Only *this* entry point was wrong; the other two already emitted. They are
// covered anyway, because the contract is what is being pinned here, and a
// future edit to either of them should have to fail a test.
TEST_CASE("a stream that cannot reach the provider reports the failure",
          "[c_binding][stream]") {
    const std::string url = unreachable_base_url();
    Stack s(url.c_str());
    REQUIRE(s.ctx);
    REQUIRE(s.model);

    SECTION("ai_stream_text") {
        StreamTally tally;
        ai_generate_options_t opts{};
        opts.model = s.model;
        opts.prompt = "hello";
        opts.max_steps = 1;

        CHECK(ai_stream_text(opts, &StreamTally::on_event, &tally) != AI_OK);
        tally.check_reports_failure();
    }

    SECTION("ai_agent_call_stream") {
        ai_agent_t agent = s.make_agent(nullptr);
        REQUIRE(agent);

        StreamTally tally;
        CHECK(ai_agent_call_stream(agent, "hello", &StreamTally::on_event, &tally) != AI_OK);
        tally.check_reports_failure();

        ai_agent_destroy(agent);
    }

    SECTION("ai_session_send_stream") {
        ai_agent_t agent = s.make_agent(nullptr);
        REQUIRE(agent);
        ai_session_t session = ai_session_create(agent);
        REQUIRE(session);

        StreamTally tally;
        CHECK(ai_session_send_stream(session, "hello", &StreamTally::on_event, &tally) != AI_OK);
        tally.check_reports_failure();

        ai_session_destroy(session);
        ai_agent_destroy(agent);
    }
}
