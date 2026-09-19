#include <catch2/catch_test_macros.hpp>

#include <ai/permission/permission.hpp>
#include <ai/tool/tool.hpp>
#include <ai/tool/tool_set.hpp>
#include <ai/schema/json_schema.hpp>

#include <boost/asio.hpp>

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

/// A tool that increments a shared counter when it actually runs and returns
/// input.x + 1. Lets tests distinguish "ran" from "blocked".
ai::ToolDefinition make_counting_tool(std::string name, std::shared_ptr<int> counter) {
    return ai::tool(
        std::move(name),
        ai::schema::JsonSchema::object({{"x", ai::schema::JsonSchema::integer()}}),
        "counts",
        [counter](const boost::json::value& input, ai::ToolExecutionContext)
            -> ai::Task<boost::json::value> {
            ++*counter;
            co_return boost::json::value(input.at("x").as_int64() + 1);
        }
    );
}

bool is_denied(const boost::json::value& out) {
    return out.is_object() &&
           out.as_object().contains("error") &&
           out.as_object().at("error") == "permission_denied";
}

/// The explanation attached to a denial, or "" when the payload carries none.
std::string deny_message(const boost::json::value& out) {
    if (!out.is_object()) return {};
    const auto& obj = out.as_object();
    if (!obj.contains("message")) return {};
    const auto& msg = obj.at("message");
    return msg.is_string() ? std::string(msg.as_string()) : std::string{};
}

/// Whether the payload has a `message` key at all. Distinct from the message
/// being empty: a denial with no explanation must not look like one that
/// explained itself with a blank string.
bool has_message(const boost::json::value& out) {
    return out.is_object() && out.as_object().contains("message");
}

} // namespace

TEST_CASE("with_permissions Allow runs the tool", "[permission]") {
    boost::asio::io_context ioc;
    auto counter = std::make_shared<int>(0);

    ai::ToolSet tools;
    tools.add(make_counting_tool("inc", counter));

    ai::ToolSet gated = ai::with_permissions(
        std::move(tools),
        [](const std::string&, const boost::json::value&) {
            return ai::PermissionDecision::Allow;
        });

    const auto* def = gated.find("inc");
    REQUIRE(def);
    REQUIRE(def->execute);

    auto out = run((*def->execute)(
        boost::json::object{{"x", 1}}, ai::ToolExecutionContext{}), ioc);
    REQUIRE(out.as_int64() == 2);
    REQUIRE(*counter == 1);
}

TEST_CASE("with_permissions Deny blocks and returns an error result", "[permission]") {
    boost::asio::io_context ioc;
    auto counter = std::make_shared<int>(0);

    ai::ToolSet tools;
    tools.add(make_counting_tool("inc", counter));

    ai::ToolSet gated = ai::with_permissions(
        std::move(tools),
        [](const std::string&, const boost::json::value&) {
            return ai::PermissionDecision::Deny;
        });

    const auto* def = gated.find("inc");
    auto out = run((*def->execute)(
        boost::json::object{{"x", 1}}, ai::ToolExecutionContext{}), ioc);

    REQUIRE(is_denied(out));
    REQUIRE(*counter == 0);  // never ran
}

TEST_CASE("with_permissions Ask without approver defaults to Deny", "[permission]") {
    boost::asio::io_context ioc;
    auto counter = std::make_shared<int>(0);

    ai::ToolSet tools;
    tools.add(make_counting_tool("inc", counter));

    // Ask policy, NO approver -> fail closed.
    ai::ToolSet gated = ai::with_permissions(
        std::move(tools),
        [](const std::string&, const boost::json::value&) {
            return ai::PermissionDecision::Ask;
        });

    const auto* def = gated.find("inc");
    auto out = run((*def->execute)(
        boost::json::object{{"x", 1}}, ai::ToolExecutionContext{}), ioc);

    REQUIRE(is_denied(out));
    REQUIRE(*counter == 0);
}

TEST_CASE("with_permissions Ask approver Deny blocks", "[permission]") {
    boost::asio::io_context ioc;
    auto counter = std::make_shared<int>(0);

    ai::ToolSet tools;
    tools.add(make_counting_tool("inc", counter));

    ai::Approver approver = [](const std::string&, const boost::json::value&,
                               const std::string&) -> ai::Task<ai::Approval> {
        co_return ai::Approval{ai::PermissionDecision::Deny, false};
    };

    ai::ToolSet gated = ai::with_permissions(
        std::move(tools),
        [](const std::string&, const boost::json::value&) {
            return ai::PermissionDecision::Ask;
        },
        approver);

    const auto* def = gated.find("inc");
    auto out = run((*def->execute)(
        boost::json::object{{"x", 1}}, ai::ToolExecutionContext{}), ioc);

    REQUIRE(is_denied(out));
    REQUIRE(*counter == 0);
}

TEST_CASE("with_permissions caches always_allow and skips the approver", "[permission]") {
    boost::asio::io_context ioc;
    auto counter = std::make_shared<int>(0);
    auto approver_calls = std::make_shared<int>(0);

    ai::ToolSet tools;
    tools.add(make_counting_tool("inc", counter));

    ai::Approver approver = [approver_calls](
        const std::string&, const boost::json::value&, const std::string&
    ) -> ai::Task<ai::Approval> {
        ++*approver_calls;
        // Approve once, and remember the decision for the session.
        co_return ai::Approval{ai::PermissionDecision::Allow, /*always_allow=*/true};
    };

    ai::ToolSet gated = ai::with_permissions(
        std::move(tools),
        [](const std::string&, const boost::json::value&) {
            return ai::PermissionDecision::Ask;
        },
        approver);

    const auto* def = gated.find("inc");
    REQUIRE(def);
    REQUIRE(def->execute);

    // First call: policy -> Ask -> approver -> {Allow, always_allow} -> runs.
    auto out1 = run((*def->execute)(
        boost::json::object{{"x", 10}}, ai::ToolExecutionContext{}), ioc);
    REQUIRE(out1.as_int64() == 11);
    REQUIRE(*counter == 1);
    REQUIRE(*approver_calls == 1);

    // Second call: tool is now allowlisted -> runs WITHOUT the approver.
    auto out2 = run((*def->execute)(
        boost::json::object{{"x", 20}}, ai::ToolExecutionContext{}), ioc);
    REQUIRE(out2.as_int64() == 21);
    REQUIRE(*counter == 2);
    REQUIRE(*approver_calls == 1);  // not consulted again
}

TEST_CASE("a policy can explain its denial to the model", "[permission]") {
    boost::asio::io_context ioc;
    auto counter = std::make_shared<int>(0);

    ai::ToolSet tools;
    tools.add(make_counting_tool("inc", counter));

    ai::ToolSet gated = ai::with_permissions(
        std::move(tools),
        [](const std::string&, const boost::json::value&) {
            return ai::PermissionVerdict{
                ai::PermissionDecision::Deny,
                "this path is outside the project root"};
        });

    const auto* def = gated.find("inc");
    auto out = run((*def->execute)(
        boost::json::object{{"x", 1}}, ai::ToolExecutionContext{}), ioc);

    REQUIRE(is_denied(out));
    REQUIRE(*counter == 0);
    CHECK(deny_message(out) == "this path is outside the project root");
}

TEST_CASE("a denial with no reason carries no message key", "[permission]") {
    boost::asio::io_context ioc;
    auto counter = std::make_shared<int>(0);

    ai::ToolSet tools;
    tools.add(make_counting_tool("inc", counter));

    ai::ToolSet gated = ai::with_permissions(
        std::move(tools),
        [](const std::string&, const boost::json::value&) {
            return ai::PermissionDecision::Deny;
        });

    const auto* def = gated.find("inc");
    auto out = run((*def->execute)(
        boost::json::object{{"x", 1}}, ai::ToolExecutionContext{}), ioc);

    REQUIRE(is_denied(out));
    // Absent, not empty: a consumer must be able to tell the two apart.
    CHECK_FALSE(has_message(out));
}

TEST_CASE("an approver that explains itself overrides the policy's reason", "[permission]") {
    boost::asio::io_context ioc;
    auto counter = std::make_shared<int>(0);

    ai::ToolSet tools;
    tools.add(make_counting_tool("inc", counter));

    ai::Approver approver = [](const std::string&, const boost::json::value&,
                               const std::string&) -> ai::Task<ai::Approval> {
        ai::Approval a;
        a.decision = ai::PermissionDecision::Deny;
        a.reason = "the user declined";
        co_return a;
    };

    ai::ToolSet gated = ai::with_permissions(
        std::move(tools),
        [](const std::string&, const boost::json::value&) {
            return ai::PermissionVerdict{
                ai::PermissionDecision::Ask, "policy wanted a human to look"};
        },
        approver);

    const auto* def = gated.find("inc");
    auto out = run((*def->execute)(
        boost::json::object{{"x", 1}}, ai::ToolExecutionContext{}), ioc);

    REQUIRE(is_denied(out));
    REQUIRE(*counter == 0);
    // The approver answered second and saw the same input, so its reason wins.
    CHECK(deny_message(out) == "the user declined");
}

TEST_CASE("a silent approver leaves the policy's reason standing", "[permission]") {
    boost::asio::io_context ioc;
    auto counter = std::make_shared<int>(0);

    ai::ToolSet tools;
    tools.add(make_counting_tool("inc", counter));

    // Denies without a reason: a bare `Approval{Deny, false}`.
    ai::Approver approver = [](const std::string&, const boost::json::value&,
                               const std::string&) -> ai::Task<ai::Approval> {
        co_return ai::Approval{ai::PermissionDecision::Deny, false};
    };

    ai::ToolSet gated = ai::with_permissions(
        std::move(tools),
        [](const std::string&, const boost::json::value&) {
            return ai::PermissionVerdict{
                ai::PermissionDecision::Ask, "touches a credentials file"};
        },
        approver);

    const auto* def = gated.find("inc");
    auto out = run((*def->execute)(
        boost::json::object{{"x", 1}}, ai::ToolExecutionContext{}), ioc);

    REQUIRE(is_denied(out));
    // The policy's reason is the only explanation available, and it still
    // describes why the call was questionable. Drop it and the model learns
    // nothing about why the answer was no.
    CHECK(deny_message(out) == "touches a credentials file");
}

TEST_CASE("Ask with no approver explains the fail-closed denial", "[permission]") {
    boost::asio::io_context ioc;
    auto counter = std::make_shared<int>(0);

    ai::ToolSet tools;
    tools.add(make_counting_tool("inc", counter));

    ai::ToolSet gated = ai::with_permissions(
        std::move(tools),
        [](const std::string&, const boost::json::value&) {
            return ai::PermissionDecision::Ask;
        });

    const auto* def = gated.find("inc");
    auto out = run((*def->execute)(
        boost::json::object{{"x", 1}}, ai::ToolExecutionContext{}), ioc);

    REQUIRE(is_denied(out));
    REQUIRE(*counter == 0);
    // Nobody is left to explain this one, so the SDK must.
    CHECK_FALSE(deny_message(out).empty());
}

TEST_CASE("an allowed tool's reason never reaches the result", "[permission]") {
    boost::asio::io_context ioc;
    auto counter = std::make_shared<int>(0);

    ai::ToolSet tools;
    tools.add(make_counting_tool("inc", counter));

    ai::ToolSet gated = ai::with_permissions(
        std::move(tools),
        [](const std::string&, const boost::json::value&) {
            // A reason on an Allow is meaningless; it must not leak.
            return ai::PermissionVerdict{
                ai::PermissionDecision::Allow, "looks fine to me"};
        });

    const auto* def = gated.find("inc");
    auto out = run((*def->execute)(
        boost::json::object{{"x", 41}}, ai::ToolExecutionContext{}), ioc);

    REQUIRE(out.is_int64());
    CHECK(out.as_int64() == 42);
    REQUIRE(*counter == 1);
}
