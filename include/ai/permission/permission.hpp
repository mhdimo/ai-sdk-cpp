#pragma once

#include <ai/tool/tool.hpp>
#include <ai/tool/tool_set.hpp>
#include <ai/stream/async_generator.hpp>
#include <boost/json.hpp>
#include <functional>
#include <string>

namespace ai {

/// Outcome of a permission check.
enum class PermissionDecision {
    Allow,  ///< run the tool
    Deny,   ///< refuse and return an error result without running
    Ask,    ///< escalate to the interactive Approver (if any)
};

/// Result of an interactive approval. When `decision == Allow` and
/// `always_allow` is true, the tool is added to a session-local allowlist so
/// subsequent calls skip the approver.
struct Approval {
    PermissionDecision decision;
    bool always_allow = false;
};

/// Synchronous permission rule: inspect (tool, input) and return a decision
/// with no I/O. Acts as a fast path before any interactive Approver. A missing
/// (empty) policy means "Allow everything".
using PermissionPolicy = std::function<PermissionDecision(
    const std::string& tool,
    const boost::json::value& input
)>;

/// Async interactive approver: may prompt the user (e.g. a CLI y/n), await
/// input, or consult any out-of-band source. Receives a human-readable
/// `rationale` to display. Invoked only when the policy returns `Ask`.
using Approver = std::function<Task<Approval>(
    const std::string& tool,
    const boost::json::value& input,
    const std::string& rationale
)>;

/// Return a copy of `tools` in which every executable tool's `execute` is
/// wrapped in a permission gate. The gate applies `policy` first:
///   - Allow -> run the tool;
///   - Deny  -> return `{ "error": "permission_denied", "tool": <name> }`
///              without running;
///   - Ask   -> `co_await approver` (if provided). An approval of
///              `{Allow, always_allow}` adds the tool to a session-local
///              allowlist so later calls skip the approver. With no approver
///              wired, `Ask` is treated as `Deny` (safe default).
/// Tools without an `execute` (schema-only) are copied unchanged. A missing
/// (empty) policy makes the gate pass-through (Allow).
ToolSet with_permissions(ToolSet tools, PermissionPolicy policy, Approver approver = {});

} // namespace ai
