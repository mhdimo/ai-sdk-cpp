#pragma once

#include <ai/tool/tool.hpp>
#include <ai/tool/tool_set.hpp>

#include <chrono>
#include <cstddef>
#include <string>

namespace ai {

/// Tunables for the standard toolkit.
struct ToolkitOptions {
    /// Maximum bytes returned by `read_file` before truncation.
    std::size_t max_read_bytes = 256 * 1024;
    /// Maximum bytes captured from a shell command before truncation.
    std::size_t max_command_output = 10'000;
    /// Maximum number of paths returned by `glob`.
    std::size_t max_glob_results = 500;
    /// Maximum number of matches returned by `grep`.
    std::size_t max_grep_results = 50;
    /// Advisory upper bound on command execution time. The current `bash`
    /// implementation runs synchronously via popen; a hard, portable timeout
    /// (fork/exec/wait) is future work.
    std::chrono::milliseconds command_timeout{60'000};
};

/// A tested, built-in toolkit for filesystem/shell agents, in the spirit of
/// Claude Code:
///   - `read_file`     {path, offset?, limit?}  -> file contents (line-sliced, byte-capped)
///   - `write_file`    {path, content}          -> create/overwrite (mkdir -p)
///   - `edit_file`     {path, old_string, new_string, replace_all?}  -> exact string-replace
///   - `glob`          {pattern, path?}         -> matching file paths
///   - `grep`          {pattern, path, file_pattern?} -> content matches
///   - `bash`          {command, working_dir?}  -> captured stdout+stderr (byte-capped)
///
/// `edit_file` requires `old_string` to match exactly and uniquely unless
/// `replace_all` is set; it throws (-> tool error result) on a missing or
/// ambiguous match, the Claude-Code semantics.
ToolSet standard_toolkit(ToolkitOptions opts = {});

} // namespace ai
