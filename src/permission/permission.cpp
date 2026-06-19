#include <ai/permission/permission.hpp>

#include <memory>
#include <unordered_set>
#include <utility>

namespace ai {

namespace {

// Per-session permission state shared by every wrapped tool execute in the
// returned ToolSet (held via shared_ptr so the coroutine lambdas stay copyable
// into std::function).
struct PermissionState {
    PermissionPolicy policy;
    Approver approver;
    std::unordered_set<std::string> always_allowed;
};

std::string rationale_for(const std::string& tool, const boost::json::value& input) {
    std::string s = "Allow tool '" + tool + "' to run";
    // Attach a compact preview of the input when it is small enough to be
    // useful in a prompt.
    auto serialized = boost::json::serialize(input);
    if (serialized.size() <= 200) {
        s += " with input " + serialized;
    } else {
        s += " with input (" + std::to_string(serialized.size()) + " bytes)";
    }
    return s;
}

} // namespace

ToolSet with_permissions(ToolSet tools, PermissionPolicy policy, Approver approver) {
    auto state = std::make_shared<PermissionState>();
    state->policy = std::move(policy);
    state->approver = std::move(approver);

    ToolSet wrapped;
    for (auto& [name, tool] : tools) {
        ToolDefinition def = tool;  // ToolSet exposes const definitions.

        if (!def.execute) {
            // Schema-only tool: nothing to gate.
            wrapped.add(std::move(def));
            continue;
        }

        auto original =
            std::make_shared<ToolDefinition::ExecuteFn>(std::move(*def.execute));
        std::string tool_name = def.name;

        def.execute = [state, original, tool_name](
            boost::json::value input,
            ToolExecutionContext ctx
        ) -> Task<boost::json::value> {
            // Fast path: previously approved "always".
            if (state->always_allowed.count(tool_name)) {
                co_return co_await (*original)(std::move(input), std::move(ctx));
            }

            PermissionDecision decision = PermissionDecision::Allow;
            if (state->policy) {
                decision = state->policy(tool_name, input);
            }

            if (decision == PermissionDecision::Ask) {
                if (state->approver) {
                    Approval approval = co_await state->approver(
                        tool_name, input, rationale_for(tool_name, input));
                    decision = approval.decision;
                    if (approval.decision == PermissionDecision::Allow &&
                        approval.always_allow) {
                        state->always_allowed.insert(tool_name);
                    }
                } else {
                    // No interactive approver: fail closed.
                    decision = PermissionDecision::Deny;
                }
            }

            if (decision == PermissionDecision::Deny) {
                boost::json::object err;
                err["error"] = "permission_denied";
                err["tool"] = tool_name;
                co_return boost::json::value(std::move(err));
            }

            co_return co_await (*original)(std::move(input), std::move(ctx));
        };

        wrapped.add(std::move(def));
    }

    return wrapped;
}

} // namespace ai
