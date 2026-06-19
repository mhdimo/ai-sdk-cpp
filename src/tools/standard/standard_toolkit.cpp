#include <ai/tools/standard/standard_toolkit.hpp>

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace ai {

namespace {

// --- defensive input accessors (never throw on missing fields) -------------

std::string get_string(const boost::json::value& input, const std::string& key) {
    const auto* obj = input.if_object();
    if (!obj) return {};
    auto it = obj->find(key);
    if (it == obj->end() || !it->value().is_string()) return {};
    return std::string(it->value().as_string());
}

std::optional<long long> get_int(const boost::json::value& input, const std::string& key) {
    const auto* obj = input.if_object();
    if (!obj) return std::nullopt;
    auto it = obj->find(key);
    if (it == obj->end() || !it->value().is_int64()) return std::nullopt;
    return it->value().as_int64();
}

std::optional<bool> get_bool(const boost::json::value& input, const std::string& key) {
    const auto* obj = input.if_object();
    if (!obj) return std::nullopt;
    auto it = obj->find(key);
    if (it == obj->end() || !it->value().is_bool()) return std::nullopt;
    return it->value().as_bool();
}

// --- file helpers ----------------------------------------------------------

std::string read_file_contents(const fs::path& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        throw std::runtime_error("cannot read file: " + path.string());
    }
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

std::vector<std::string> split_lines(const std::string& s) {
    std::vector<std::string> lines;
    std::string current;
    for (char c : s) {
        if (c == '\n') {
            lines.push_back(std::move(current));
            current.clear();
        } else if (c != '\r') {
            current += c;
        }
    }
    lines.push_back(std::move(current));
    return lines;
}

std::size_t count_occurrences(const std::string& haystack, const std::string& needle) {
    if (needle.empty()) return 0;
    std::size_t n = 0, pos = 0;
    while ((pos = haystack.find(needle, pos)) != std::string::npos) {
        ++n;
        pos += needle.size();
    }
    return n;
}

std::string replace_all_copy(std::string s, const std::string& from, const std::string& to) {
    if (from.empty()) return s;
    std::size_t pos = 0;
    while ((pos = s.find(from, pos)) != std::string::npos) {
        s.replace(pos, from.size(), to);
        pos += to.size();
    }
    return s;
}

std::string truncate_with_note(std::string s, std::size_t max_bytes) {
    if (s.size() <= max_bytes) return s;
    s.resize(max_bytes);
    s += "\n... (truncated)";
    return s;
}

// --- glob -> regex ---------------------------------------------------------

std::regex glob_to_regex(const std::string& glob) {
    std::string r = "^";
    for (std::size_t i = 0; i < glob.size(); ++i) {
        char c = glob[i];
        if (c == '*' && i + 1 < glob.size() && glob[i + 1] == '*') {
            r += ".*";
            ++i;
            if (i + 1 < glob.size() && glob[i + 1] == '/') ++i;  // swallow **/
        } else if (c == '*') {
            r += "[^/]*";
        } else if (c == '?') {
            r += "[^/]";
        } else if (std::strchr(".+()[]{}^$|\\", c)) {
            r += '\\';
            r += c;
        } else {
            r += c;
        }
    }
    r += '$';
    return std::regex(r);
}

// --- command execution -----------------------------------------------------

std::string run_command_capture(const std::string& command, const std::string& working_dir) {
    std::string full = command;
    if (!working_dir.empty()) {
        full = "cd " + working_dir + " && " + command;
    }
    full += " 2>&1";

    std::string result;
    std::array<char, 8192> buffer{};  // NOLINT: sized buffer for fgets
    // NOLINTNEXTLINE: popen returns a raw FILE*; managed via unique_ptr below.
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(full.c_str(), "r"), pclose);
    if (!pipe) {
        throw std::runtime_error("failed to start command: " + command);
    }
    while (std::fgets(buffer.data(), static_cast<int>(buffer.size()), pipe.get())) {
        result += buffer.data();
    }
    return result;
}

// --- per-tool factories ----------------------------------------------------

ToolDefinition read_file_tool(ToolkitOptions opts) {
    return tool(
        "read_file",
        schema::JsonSchema::object({
            {"path", schema::JsonSchema::string("Absolute or relative file path")},
            {"offset", schema::JsonSchema::integer("1-based line number to start at (optional)")},
            {"limit", schema::JsonSchema::integer("Number of lines to read (optional)")},
        }).required({"path"}),
        "Read the contents of a file, optionally a line range.",
        [opts](const boost::json::value& input, ToolExecutionContext)
            -> Task<boost::json::value> {
            std::string path = get_string(input, "path");
            if (path.empty()) throw std::runtime_error("read_file: 'path' is required");

            std::string contents = read_file_contents(path);
            auto offset = get_int(input, "offset");
            auto limit = get_int(input, "limit");
            if (offset || limit) {
                auto lines = split_lines(contents);
                std::size_t begin = offset ? static_cast<std::size_t>(std::max<long long>(0, *offset - 1)) : 0;
                begin = std::min(begin, lines.size());
                std::size_t end = lines.size();
                if (limit) end = std::min(lines.size(), begin + static_cast<std::size_t>(std::max<long long>(0, *limit)));
                std::string sliced;
                for (std::size_t i = begin; i < end; ++i) {
                    sliced += lines[i];
                    sliced += '\n';
                }
                contents = sliced;
            }
            co_return boost::json::value(truncate_with_note(std::move(contents), opts.max_read_bytes));
        });
}

ToolDefinition write_file_tool() {
    return tool(
        "write_file",
        schema::JsonSchema::object({
            {"path", schema::JsonSchema::string("File path to create or overwrite")},
            {"content", schema::JsonSchema::string("Full file content")},
        }).required({"path", "content"}),
        "Create or overwrite a file with the given content (mkdir -p the parent).",
        [](const boost::json::value& input, ToolExecutionContext) -> Task<boost::json::value> {
            std::string path = get_string(input, "path");
            std::string content = get_string(input, "content");
            if (path.empty()) throw std::runtime_error("write_file: 'path' is required");
            fs::path p(path);
            if (p.has_parent_path()) fs::create_directories(p.parent_path());
            std::ofstream f(p, std::ios::binary | std::ios::trunc);
            if (!f) throw std::runtime_error("cannot write file: " + path);
            f << content;
            co_return boost::json::value(
                "Wrote " + std::to_string(content.size()) + " bytes to " + path);
        });
}

ToolDefinition edit_file_tool() {
    return tool(
        "edit_file",
        schema::JsonSchema::object({
            {"path", schema::JsonSchema::string("File to edit")},
            {"old_string", schema::JsonSchema::string("Exact text to find")},
            {"new_string", schema::JsonSchema::string("Replacement text")},
            {"replace_all", schema::JsonSchema::boolean("Replace every occurrence (default false)")},
        }).required({"path", "old_string", "new_string"}),
        "Replace exact text in a file. old_string must match exactly and uniquely "
        "unless replace_all is true.",
        [](const boost::json::value& input, ToolExecutionContext) -> Task<boost::json::value> {
            std::string path = get_string(input, "path");
            std::string old_s = get_string(input, "old_string");
            std::string new_s = get_string(input, "new_string");
            bool replace_all = get_bool(input, "replace_all").value_or(false);
            if (path.empty()) throw std::runtime_error("edit_file: 'path' is required");
            if (old_s.empty()) throw std::runtime_error("edit_file: 'old_string' is required");

            std::string contents = read_file_contents(path);
            std::size_t matches = count_occurrences(contents, old_s);
            if (matches == 0) {
                throw std::runtime_error("old_string not found in " + path);
            }
            if (matches > 1 && !replace_all) {
                throw std::runtime_error(
                    "old_string matches " + std::to_string(matches) +
                    " locations in " + path + "; set replace_all to replace all");
            }

            std::string updated;
            if (replace_all) {
                updated = replace_all_copy(contents, old_s, new_s);
            } else {
                auto pos = contents.find(old_s);
                updated = contents;
                updated.replace(pos, old_s.size(), new_s);
            }
            std::ofstream f(path, std::ios::binary | std::ios::trunc);
            if (!f) throw std::runtime_error("cannot write file: " + path);
            f << updated;

            std::size_t replaced = replace_all ? matches : 1;
            co_return boost::json::value(
                "Edited " + path + " (" + std::to_string(replaced) + " replacement" +
                (replaced == 1 ? "" : "s") + ").");
        });
}

ToolDefinition glob_tool(ToolkitOptions opts) {
    return tool(
        "glob",
        schema::JsonSchema::object({
            {"pattern", schema::JsonSchema::string("Glob pattern, e.g. '**/*.cpp'")},
            {"path", schema::JsonSchema::string("Directory to search (default cwd)")},
        }).required({"pattern"}),
        "List file paths matching a glob pattern.",
        [opts](const boost::json::value& input, ToolExecutionContext) -> Task<boost::json::value> {
            std::string pattern = get_string(input, "pattern");
            std::string base = get_string(input, "path");
            if (pattern.empty()) throw std::runtime_error("glob: 'pattern' is required");
            fs::path root = base.empty() ? fs::current_path() : fs::path(base);
            std::regex rx = glob_to_regex(pattern);

            boost::json::array matches;
            std::error_code ec;
            for (auto it = fs::recursive_directory_iterator(root, ec);
                 it != fs::recursive_directory_iterator(); it.increment(ec)) {
                if (ec) { ec.clear(); continue; }
                if (!it->is_regular_file()) continue;
                std::string rel = fs::relative(it->path(), root).string();
                try {
                    if (std::regex_match(rel, rx)) {
                        matches.push_back(boost::json::value(it->path().string()));
                        if (matches.size() >= opts.max_glob_results) break;
                    }
                } catch (...) {
                    // Ignore regex/parsing oddities on individual paths.
                }
            }
            co_return boost::json::value(matches);
        });
}

ToolDefinition grep_tool(ToolkitOptions opts) {
    return tool(
        "grep",
        schema::JsonSchema::object({
            {"pattern", schema::JsonSchema::string("Regex pattern to search for")},
            {"path", schema::JsonSchema::string("Directory or file to search")},
            {"file_pattern", schema::JsonSchema::string("Glob to restrict files (optional)")},
        }).required({"pattern", "path"}),
        "Search file contents for a regex pattern; returns path:line:match entries.",
        [opts](const boost::json::value& input, ToolExecutionContext) -> Task<boost::json::value> {
            std::string pattern = get_string(input, "pattern");
            std::string path = get_string(input, "path");
            std::string file_pattern = get_string(input, "file_pattern");
            if (pattern.empty()) throw std::runtime_error("grep: 'pattern' is required");
            if (path.empty()) throw std::runtime_error("grep: 'path' is required");

            std::regex rx;
            try {
                rx = std::regex(pattern);
            } catch (const std::regex_error& e) {
                throw std::runtime_error(std::string("invalid regex: ") + e.what());
            }
            std::optional<std::regex> file_rx;
            if (!file_pattern.empty()) file_rx = glob_to_regex(file_pattern);

            boost::json::array results;
            std::error_code ec;
            auto handle_file = [&](const fs::path& file) {
                if (file_rx) {
                    if (!std::regex_match(file.filename().string(), *file_rx)) return;
                }
                std::ifstream f(file);
                if (!f) return;
                std::string line;
                for (long long lineno = 1; std::getline(f, line); ++lineno) {
                    try {
                        if (std::regex_search(line, rx)) {
                            std::string entry = file.string() + ":" +
                                std::to_string(lineno) + ":" + line;
                            results.push_back(boost::json::value(entry));
                            if (results.size() >= opts.max_grep_results) return;
                        }
                    } catch (...) {
                        return;
                    }
                }
            };

            fs::path root(path);
            if (fs::is_regular_file(root)) {
                handle_file(root);
            } else {
                for (auto it = fs::recursive_directory_iterator(root, ec);
                     it != fs::recursive_directory_iterator(); it.increment(ec)) {
                    if (ec) { ec.clear(); continue; }
                    if (it->is_regular_file()) {
                        handle_file(it->path());
                        if (results.size() >= opts.max_grep_results) break;
                    }
                }
            }
            co_return boost::json::value(results);
        });
}

ToolDefinition bash_tool(ToolkitOptions opts) {
    return tool(
        "bash",
        schema::JsonSchema::object({
            {"command", schema::JsonSchema::string("Shell command to execute")},
            {"working_dir", schema::JsonSchema::string("Working directory (optional)")},
        }).required({"command"}),
        "Execute a shell command and return its combined stdout/stderr (capped).",
        [opts](const boost::json::value& input, ToolExecutionContext) -> Task<boost::json::value> {
            std::string command = get_string(input, "command");
            std::string working_dir = get_string(input, "working_dir");
            if (command.empty()) throw std::runtime_error("bash: 'command' is required");
            (void)opts;  // command_timeout is advisory only for the popen impl
            std::string output = run_command_capture(command, working_dir);
            co_return boost::json::value(truncate_with_note(std::move(output), opts.max_command_output));
        });
}

} // namespace

ToolSet standard_toolkit(ToolkitOptions opts) {
    ToolSet tools;
    tools.add(read_file_tool(opts));
    tools.add(write_file_tool());
    tools.add(edit_file_tool());
    tools.add(glob_tool(opts));
    tools.add(grep_tool(opts));
    tools.add(bash_tool(opts));
    return tools;
}

} // namespace ai
