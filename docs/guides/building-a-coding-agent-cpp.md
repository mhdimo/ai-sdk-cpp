# Building a Coding Agent in C++

This guide walks you through building a fully functional coding agent -- the kind of tool that powers products like Claude Code, Codex, and Kimi Code -- using the `ai-sdk-cpp` framework. By the end, you will have a working agent that can read your codebase, make edits, run commands, and iterate on problems autonomously.

## Why C++ for a Coding Agent?

Most AI agent frameworks are written in Python or TypeScript. Building one in C++ gives you distinct advantages:

- **Performance**: Sub-millisecond tool execution. No interpreter overhead. When your agent is running grep across a million-line codebase or applying 50 patches in a row, you feel the difference.
- **Memory efficiency**: Agents that run for hours (long refactoring sessions, multi-file migrations) benefit from deterministic memory management and zero-copy string views.
- **Native system access**: Direct POSIX APIs, no subprocess overhead for file I/O, and tight integration with system facilities like `inotify` or `kqueue`.
- **Single binary deployment**: Ship one statically-linked executable. No runtime, no virtualenv, no `node_modules`.
- **C++20 coroutines**: The `co_return` / `co_await` model maps naturally to the agent loop pattern -- each tool execution is a coroutine that the event loop drives forward.

## Prerequisites

Before you begin, make sure you have:

| Requirement | Version | Notes |
|---|---|---|
| C++ compiler | GCC 12+ or Clang 15+ | Must support C++20 coroutines |
| CMake | 3.20+ | Build system |
| Boost | 1.82+ | We use Boost.Asio, Boost.JSON, Boost.Beast |
| OpenSSL | 1.1+ or 3.x | TLS for API calls |
| An API key | -- | ANTHROPIC_API_KEY, OPENAI_API_KEY, etc. |

On macOS:
```bash
brew install cmake boost openssl
```

On Ubuntu/Debian:
```bash
sudo apt install cmake libboost-all-dev libssl-dev g++-12
```

## Project Setup

Create a new directory for your agent project and set up the build:

```
my-coding-agent/
  CMakeLists.txt
  main.cpp
```

### CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.20)
project(my-coding-agent LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find ai-sdk-cpp (adjust path to your local checkout or installed location)
add_subdirectory(path/to/ai-sdk-cpp EXCLUDE_FROM_ALL)

add_executable(coding-agent main.cpp)
target_link_libraries(coding-agent PRIVATE ai-sdk-core ai-sdk-anthropic)
```

If you installed `ai-sdk-cpp` system-wide, you can use `find_package` instead:

```cmake
find_package(AiSdk REQUIRED)
target_link_libraries(coding-agent PRIVATE AiSdk::core AiSdk::anthropic)
```

Build it:

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)
```

## Core Concepts

Before diving into code, let's understand the four building blocks:

### Providers

A provider connects to an AI service. You create one, then ask it for a model:

```cpp
#include <ai/ai.hpp>
#include <ai/providers/anthropic/anthropic.hpp>

boost::asio::io_context ioc;
auto anthropic = ai::providers::anthropic::create_anthropic({.io_context = ioc});
auto model = anthropic->language_model("claude-sonnet-4-20250514");
```

The SDK ships providers for Anthropic, OpenAI, Google, Groq, xAI, Mistral, Fireworks, Together AI, Perplexity, Cohere, DeepSeek, and Amazon Bedrock. They all return a `LanguageModelPtr` -- the same interface.

### Tools

Tools are functions the model can call. Each tool has a name, a JSON Schema describing its inputs, a description for the model, and an async execution function:

```cpp
ai::ToolSet tools;

tools.add(ai::tool("tool_name",
    ai::schema::JsonSchema::object({
        {"param", ai::schema::JsonSchema::string("What this parameter is for")}
    }).required({"param"}).additional_properties(false),
    "What this tool does (the model reads this)",
    [](boost::json::value input, ai::ToolExecutionContext ctx) -> ai::Task<boost::json::value> {
        auto val = std::string(input.at("param").as_string());
        // Do work...
        co_return boost::json::value("result");
    }
));
```

The schema types available are:
- `JsonSchema::string(description)` -- a string value
- `JsonSchema::number(description)` -- a numeric value
- `JsonSchema::integer(description)` -- an integer
- `JsonSchema::boolean(description)` -- true/false
- `JsonSchema::object({...})` -- an object with named properties
- `JsonSchema::array(item_schema)` -- an array of items
- `JsonSchema::enum_of({"a", "b", "c"})` -- one of a fixed set of strings

### The Agent Loop

A `ToolLoopAgent` runs the core loop: send messages to the model, receive tool calls, execute them, feed results back, repeat until the model says it's done or you hit the step limit.

```cpp
ai::ToolLoopAgent agent({
    .model = model,
    .tools = std::move(tools),
    .instructions = "System prompt that shapes behavior...",
    .max_steps = 50,
    .on_step_finish = [](const ai::StepResult& step) { /* observe progress */ },
});
```

### Driving the Event Loop

The SDK uses Boost.Asio for async I/O. You start a task and pump the event loop:

```cpp
auto task = agent.call("User's request");
task.start();
while (!task.done()) {
    ioc.run_one();  // Process one async event
}
auto result = task.get();  // Get the final result (or throw)
```

This gives you full control -- you can interleave other work, handle signals, or drive multiple agents concurrently on the same `io_context`.

## Building the Agent Step by Step

Let's build a coding agent incrementally. We'll start with the skeleton and add tools one at a time.

### Step 1: The Skeleton

```cpp
#include <ai/ai.hpp>
#include <ai/providers/anthropic/anthropic.hpp>
#include <boost/asio.hpp>
#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;

int main() {
    boost::asio::io_context ioc;

    auto anthropic = ai::providers::anthropic::create_anthropic({.io_context = ioc});
    auto model = anthropic->language_model("claude-sonnet-4-20250514");

    ai::ToolSet tools;
    // We'll add tools here...

    std::string cwd = fs::current_path().string();

    ai::ToolLoopAgent agent({
        .model = model,
        .tools = std::move(tools),
        .instructions = "You are a coding agent. CWD: " + cwd,
        .max_steps = 50,
    });

    auto task = agent.call("Read the README and summarize this project");
    task.start();
    while (!task.done()) ioc.run_one();

    std::cout << task.get().text << "\n";
    return 0;
}
```

This compiles and runs but cannot do anything yet -- the model has no tools. Let's fix that.

### Step 2: Add the `read_file` Tool

The most fundamental capability. Your agent needs to see code:

```cpp
tools.add(ai::tool("read_file",
    ai::schema::JsonSchema::object({
        {"path", ai::schema::JsonSchema::string("Absolute path to the file to read")},
        {"offset", ai::schema::JsonSchema::integer("Line number to start reading from (0-indexed)")},
        {"limit", ai::schema::JsonSchema::integer("Maximum number of lines to read")}
    }).required({"path"}),
    "Read the contents of a file. Returns the file content with line numbers. "
    "For large files, use offset and limit to read specific sections.",
    [](boost::json::value input, ai::ToolExecutionContext) -> ai::Task<boost::json::value> {
        auto path = std::string(input.at("path").as_string());

        std::ifstream f(path);
        if (!f.is_open()) {
            co_return boost::json::value("Error: cannot open file: " + path);
        }

        int offset = 0, limit = 2000;
        if (auto it = input.as_object().find("offset"); it != input.as_object().end())
            offset = static_cast<int>(it->value().as_int64());
        if (auto it = input.as_object().find("limit"); it != input.as_object().end())
            limit = static_cast<int>(it->value().as_int64());

        std::string result;
        std::string line;
        int line_num = 0;
        while (std::getline(f, line)) {
            if (line_num >= offset && line_num < offset + limit) {
                result += std::to_string(line_num + 1) + "\t" + line + "\n";
            }
            if (line_num >= offset + limit) break;
            line_num++;
        }

        if (result.empty()) {
            co_return boost::json::value("File is empty or offset is past end of file");
        }
        co_return boost::json::value(std::move(result));
    }
));
```

### Step 3: Add the `write_file` Tool

For creating new files or complete rewrites:

```cpp
tools.add(ai::tool("write_file",
    ai::schema::JsonSchema::object({
        {"path", ai::schema::JsonSchema::string("Absolute path for the file to write")},
        {"content", ai::schema::JsonSchema::string("Complete file content to write")}
    }).required({"path", "content"}).additional_properties(false),
    "Create or overwrite a file with the given content. Creates parent directories as needed.",
    [](boost::json::value input, ai::ToolExecutionContext) -> ai::Task<boost::json::value> {
        auto path = std::string(input.at("path").as_string());
        auto content = std::string(input.at("content").as_string());

        // Create parent directories
        fs::create_directories(fs::path(path).parent_path());

        std::ofstream f(path);
        if (!f.is_open()) {
            co_return boost::json::value("Error: cannot write to " + path);
        }
        f << content;
        f.close();

        co_return boost::json::value(
            "Written " + std::to_string(content.size()) + " bytes to " + path
        );
    }
));
```

### Step 4: Add the `edit_file` Tool (Surgical Edits)

This is the most important tool for a coding agent. Instead of rewriting entire files, the model makes targeted edits using exact string matching. This minimizes context window usage and prevents accidental changes to surrounding code:

```cpp
tools.add(ai::tool("edit_file",
    ai::schema::JsonSchema::object({
        {"path", ai::schema::JsonSchema::string("Absolute path to the file to edit")},
        {"old_string", ai::schema::JsonSchema::string("Exact string to find and replace (must match uniquely)")},
        {"new_string", ai::schema::JsonSchema::string("Replacement string")}
    }).required({"path", "old_string", "new_string"}).additional_properties(false),
    "Replace an exact string in a file with a new string. "
    "The old_string must appear exactly once in the file. "
    "Use read_file first to see the exact content including whitespace.",
    [](boost::json::value input, ai::ToolExecutionContext) -> ai::Task<boost::json::value> {
        auto path = std::string(input.at("path").as_string());
        auto old_str = std::string(input.at("old_string").as_string());
        auto new_str = std::string(input.at("new_string").as_string());

        // Read current content
        std::ifstream in(path);
        if (!in.is_open()) {
            co_return boost::json::value("Error: cannot open " + path);
        }
        std::string content((std::istreambuf_iterator<char>(in)),
                             std::istreambuf_iterator<char>());
        in.close();

        // Find the old string
        auto pos = content.find(old_str);
        if (pos == std::string::npos) {
            co_return boost::json::value(
                "Error: old_string not found in file. "
                "Make sure it matches exactly (including whitespace and indentation)."
            );
        }

        // Check it's unique
        if (content.find(old_str, pos + 1) != std::string::npos) {
            co_return boost::json::value(
                "Error: old_string appears multiple times. "
                "Provide more surrounding context to make it unique."
            );
        }

        // Apply the edit
        content.replace(pos, old_str.size(), new_str);

        std::ofstream out(path);
        out << content;
        out.close();

        co_return boost::json::value("Edit applied successfully to " + path);
    }
));
```

### Step 5: Add the `run_command` Tool

Shell access is essential for running tests, building projects, and exploring:

```cpp
#include <array>
#include <memory>

namespace {
std::string exec_command(const std::string& cmd, int timeout_sec = 30) {
    std::string full_cmd = "timeout " + std::to_string(timeout_sec) + " "
                          + cmd + " 2>&1";
    std::array<char, 8192> buffer;
    std::string result;
    std::unique_ptr<FILE, decltype(&pclose)> pipe(
        popen(full_cmd.c_str(), "r"), pclose
    );
    if (!pipe) return "Error: failed to execute command";
    while (fgets(buffer.data(), buffer.size(), pipe.get())) {
        result += buffer.data();
        // Safety limit: don't let output grow unbounded
        if (result.size() > 100000) {
            result += "\n... (output truncated at 100KB)";
            break;
        }
    }
    return result;
}
} // namespace

// Then in your tool setup:
tools.add(ai::tool("run_command",
    ai::schema::JsonSchema::object({
        {"command", ai::schema::JsonSchema::string("The shell command to execute")},
        {"working_dir", ai::schema::JsonSchema::string("Working directory (defaults to CWD)")}
    }).required({"command"}),
    "Execute a shell command and return stdout+stderr. "
    "Commands are terminated after 30 seconds. Use for: running tests, "
    "building projects, git operations, installing packages.",
    [cwd](boost::json::value input, ai::ToolExecutionContext) -> ai::Task<boost::json::value> {
        auto cmd = std::string(input.at("command").as_string());
        std::string dir = cwd;
        if (auto it = input.as_object().find("working_dir");
            it != input.as_object().end() && it->value().is_string()) {
            dir = std::string(it->value().as_string());
        }

        std::string full = "cd " + dir + " && " + cmd;
        auto output = exec_command(full);
        co_return boost::json::value(std::move(output));
    }
));
```

### Step 6: Add the `list_files` Tool

For exploring project structure:

```cpp
tools.add(ai::tool("list_files",
    ai::schema::JsonSchema::object({
        {"path", ai::schema::JsonSchema::string("Directory path to list")},
        {"recursive", ai::schema::JsonSchema::boolean("If true, list recursively (max 500 entries)")},
    }).required({"path"}),
    "List files and directories at the given path.",
    [](boost::json::value input, ai::ToolExecutionContext) -> ai::Task<boost::json::value> {
        auto path = std::string(input.at("path").as_string());
        bool recursive = false;
        if (auto it = input.as_object().find("recursive");
            it != input.as_object().end() && it->value().is_bool()) {
            recursive = it->value().as_bool();
        }

        boost::json::array entries;
        try {
            if (recursive) {
                for (auto& entry : fs::recursive_directory_iterator(path)) {
                    entries.push_back(boost::json::value(entry.path().string()));
                    if (entries.size() >= 500) {
                        entries.push_back(boost::json::value("... (truncated at 500 entries)"));
                        break;
                    }
                }
            } else {
                for (auto& entry : fs::directory_iterator(path)) {
                    std::string suffix = entry.is_directory() ? "/" : "";
                    entries.push_back(boost::json::value(entry.path().filename().string() + suffix));
                }
            }
        } catch (const fs::filesystem_error& e) {
            co_return boost::json::value(std::string("Error: ") + e.what());
        }

        co_return boost::json::value(std::move(entries));
    }
));
```

### Step 7: Add the `search_files` Tool

Fast text search is critical for navigating unfamiliar codebases:

```cpp
tools.add(ai::tool("search_files",
    ai::schema::JsonSchema::object({
        {"pattern", ai::schema::JsonSchema::string("Regex pattern to search for")},
        {"path", ai::schema::JsonSchema::string("Directory to search in")},
        {"file_glob", ai::schema::JsonSchema::string("File pattern to filter (e.g. *.cpp, *.hpp)")},
        {"max_results", ai::schema::JsonSchema::integer("Maximum matches to return (default 50)")}
    }).required({"pattern", "path"}),
    "Search for a regex pattern across files. Returns matching lines with file paths and line numbers.",
    [](boost::json::value input, ai::ToolExecutionContext) -> ai::Task<boost::json::value> {
        auto pattern = std::string(input.at("pattern").as_string());
        auto path = std::string(input.at("path").as_string());

        int max_results = 50;
        if (auto it = input.as_object().find("max_results");
            it != input.as_object().end() && it->value().is_int64()) {
            max_results = static_cast<int>(it->value().as_int64());
        }

        std::string cmd = "grep -rn --include='*'";
        if (auto it = input.as_object().find("file_glob");
            it != input.as_object().end() && it->value().is_string()) {
            cmd = "grep -rn --include='" + std::string(it->value().as_string()) + "'";
        }
        cmd += " '" + pattern + "' " + path + " | head -" + std::to_string(max_results);

        auto output = exec_command(cmd);
        if (output.empty()) {
            co_return boost::json::value("No matches found");
        }
        co_return boost::json::value(std::move(output));
    }
));
```

## Advanced Topics

### Streaming Output with `on_step_finish`

A good coding agent shows you what it's doing in real time. The `on_step_finish` callback fires after every model turn, letting you display text and tool results as they happen:

```cpp
ai::ToolLoopAgent agent({
    .model = model,
    .tools = std::move(tools),
    .instructions = system_prompt,
    .max_steps = 50,
    .on_step_finish = [](const ai::StepResult& step) {
        // Print any text the model produced this step
        auto text = step.result.text();
        if (!text.empty()) {
            std::cout << text << std::flush;
        }

        // Print tool execution summaries
        for (auto& tr : step.tool_results) {
            std::string output_str = boost::json::serialize(tr.output);
            // Show a preview -- don't dump entire file contents to terminal
            if (output_str.size() > 200) {
                output_str = output_str.substr(0, 200) + "...";
            }
            std::cout << "\n  \033[36m" << tr.tool_name << "\033[0m → "
                      << output_str << "\n" << std::flush;
        }
    },
});
```

### Conversation History (Multi-Turn)

For an interactive agent (like a REPL), you want to maintain conversation history across multiple user inputs. Use the message-based overload of `call`:

```cpp
std::vector<ai::Message> history;

while (true) {
    std::string user_input = get_user_input();
    if (user_input.empty()) break;

    // Add user message to history
    history.push_back(ai::Message::user(user_input));

    auto task = agent.call(history);
    task.start();
    while (!task.done()) ioc.run_one();

    auto result = task.get();

    // Append assistant response messages to history for next turn
    for (auto& msg : result.response_messages) {
        history.push_back(msg);
    }
}
```

### Multi-Model Strategy

Real coding agents use different models for different tasks. Use a fast, cheap model for simple operations (listing files, reading) and a powerful model for complex reasoning (architecture decisions, multi-file refactors):

```cpp
auto anthropic = ai::providers::anthropic::create_anthropic({.io_context = ioc});

// Fast model for routine operations
auto fast_model = anthropic->language_model("claude-haiku-3-20250514");

// Powerful model for complex reasoning
auto strong_model = anthropic->language_model("claude-sonnet-4-20250514");

// Router: use the strong model for the main agent
ai::ToolLoopAgent agent({
    .model = strong_model,
    .tools = std::move(tools),
    .instructions = system_prompt,
    .max_steps = 50,
});

// You could also create a "sub-agent" with the fast model for
// lightweight tasks like summarization:
ai::ToolLoopAgent summarizer({
    .model = fast_model,
    .tools = {},  // No tools needed for summarization
    .instructions = "Summarize the given code concisely.",
    .max_steps = 1,
});
```

### Error Handling

The SDK throws specific exception types you should catch:

```cpp
#include <ai/error/api_call_error.hpp>
#include <ai/error/ai_error.hpp>

try {
    auto result = task.get();
    std::cout << result.text << "\n";
} catch (const ai::APICallError& e) {
    // Network error, rate limit, authentication failure
    std::cerr << "API error: " << e.what() << "\n";
    std::cerr << "Status: " << e.status_code() << "\n";
    if (e.is_retryable()) {
        std::cerr << "(this error is retryable)\n";
    }
} catch (const ai::OperationCancelled&) {
    std::cerr << "Operation was cancelled\n";
} catch (const ai::AISDKError& e) {
    std::cerr << "SDK error: " << e.what() << "\n";
} catch (const std::exception& e) {
    std::cerr << "Unexpected error: " << e.what() << "\n";
}
```

### Cancellation

For a CLI agent, you want Ctrl+C to gracefully stop the current operation:

```cpp
#include <ai/util/cancellation.hpp>
#include <csignal>

ai::CancellationSource cancel_source;

// Wire up SIGINT
std::signal(SIGINT, [](int) {
    cancel_source.cancel();
});

// Pass the token to the agent
auto task = agent.call("Do something complex", cancel_source.token());
task.start();
while (!task.done()) ioc.run_one();

try {
    auto result = task.get();
} catch (const ai::OperationCancelled&) {
    std::cout << "\nCancelled by user.\n";
}
```

### Provider Options

Pass provider-specific options (like enabling extended thinking on Anthropic):

```cpp
ai::ToolLoopAgent agent({
    .model = model,
    .tools = std::move(tools),
    .instructions = system_prompt,
    .max_steps = 50,
    .temperature = 0.0,        // Deterministic for code
    .max_output_tokens = 16384,
    .provider_options = {
        {"anthropic", {
            {"thinking", {{"type", "enabled"}, {"budget_tokens", 10000}}}
        }}
    },
});
```

## Full Working Example

Here's a complete, self-contained coding agent in ~150 lines. Copy this into `main.cpp`:

```cpp
#include <ai/ai.hpp>
#include <ai/providers/anthropic/anthropic.hpp>
#include <boost/asio.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <array>
#include <memory>
#include <filesystem>

namespace fs = std::filesystem;

static std::string exec_cmd(const std::string& cmd) {
    std::array<char, 8192> buf;
    std::string out;
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd.c_str(), "r"), pclose);
    if (!pipe) return "Error: popen failed";
    while (fgets(buf.data(), buf.size(), pipe.get())) {
        out += buf.data();
        if (out.size() > 100000) { out += "\n...(truncated)"; break; }
    }
    return out;
}

int main() {
    boost::asio::io_context ioc;

    auto provider = ai::providers::anthropic::create_anthropic({.io_context = ioc});
    auto model = provider->language_model("claude-sonnet-4-20250514");
    std::string cwd = fs::current_path().string();

    ai::ToolSet tools;

    // -- read_file --
    tools.add(ai::tool("read_file",
        ai::schema::JsonSchema::object({
            {"path", ai::schema::JsonSchema::string("Absolute file path")}
        }).required({"path"}).additional_properties(false),
        "Read file contents",
        [](boost::json::value in, ai::ToolExecutionContext) -> ai::Task<boost::json::value> {
            std::ifstream f(std::string(in.at("path").as_string()));
            if (!f) co_return boost::json::value("Error: cannot open file");
            std::stringstream ss; ss << f.rdbuf();
            co_return boost::json::value(ss.str());
        }));

    // -- write_file --
    tools.add(ai::tool("write_file",
        ai::schema::JsonSchema::object({
            {"path", ai::schema::JsonSchema::string("Absolute file path")},
            {"content", ai::schema::JsonSchema::string("File content")}
        }).required({"path", "content"}).additional_properties(false),
        "Write content to a file (creates parent dirs)",
        [](boost::json::value in, ai::ToolExecutionContext) -> ai::Task<boost::json::value> {
            auto p = std::string(in.at("path").as_string());
            fs::create_directories(fs::path(p).parent_path());
            std::ofstream f(p);
            if (!f) co_return boost::json::value("Error: cannot write");
            auto content = std::string(in.at("content").as_string());
            f << content;
            co_return boost::json::value("Written " + std::to_string(content.size()) + " bytes");
        }));

    // -- edit_file --
    tools.add(ai::tool("edit_file",
        ai::schema::JsonSchema::object({
            {"path", ai::schema::JsonSchema::string("File path")},
            {"old_string", ai::schema::JsonSchema::string("Exact text to find")},
            {"new_string", ai::schema::JsonSchema::string("Replacement text")}
        }).required({"path", "old_string", "new_string"}).additional_properties(false),
        "Replace exact text in a file (must be unique match)",
        [](boost::json::value in, ai::ToolExecutionContext) -> ai::Task<boost::json::value> {
            auto path = std::string(in.at("path").as_string());
            auto old_s = std::string(in.at("old_string").as_string());
            auto new_s = std::string(in.at("new_string").as_string());
            std::ifstream fi(path); std::string content((std::istreambuf_iterator<char>(fi)), {});
            fi.close();
            auto pos = content.find(old_s);
            if (pos == std::string::npos) co_return boost::json::value("Error: string not found");
            if (content.find(old_s, pos+1) != std::string::npos)
                co_return boost::json::value("Error: multiple matches, provide more context");
            content.replace(pos, old_s.size(), new_s);
            std::ofstream fo(path); fo << content;
            co_return boost::json::value("Edit applied");
        }));

    // -- run_command --
    tools.add(ai::tool("run_command",
        ai::schema::JsonSchema::object({
            {"command", ai::schema::JsonSchema::string("Shell command")},
            {"working_dir", ai::schema::JsonSchema::string("Working directory")}
        }).required({"command"}),
        "Run a shell command, returns stdout+stderr",
        [cwd](boost::json::value in, ai::ToolExecutionContext) -> ai::Task<boost::json::value> {
            auto cmd = std::string(in.at("command").as_string());
            std::string dir = cwd;
            if (auto it = in.as_object().find("working_dir");
                it != in.as_object().end() && it->value().is_string())
                dir = std::string(it->value().as_string());
            co_return boost::json::value(exec_cmd("cd " + dir + " && " + cmd + " 2>&1"));
        }));

    // -- list_files --
    tools.add(ai::tool("list_files",
        ai::schema::JsonSchema::object({
            {"path", ai::schema::JsonSchema::string("Directory path")},
            {"recursive", ai::schema::JsonSchema::boolean("Recurse into subdirs")}
        }).required({"path"}),
        "List directory contents",
        [](boost::json::value in, ai::ToolExecutionContext) -> ai::Task<boost::json::value> {
            auto path = std::string(in.at("path").as_string());
            bool rec = false;
            if (auto it = in.as_object().find("recursive");
                it != in.as_object().end() && it->value().is_bool()) rec = it->value().as_bool();
            boost::json::array files;
            try {
                if (rec) for (auto& e : fs::recursive_directory_iterator(path)) {
                    files.push_back(boost::json::value(e.path().string()));
                    if (files.size() > 500) break;
                } else for (auto& e : fs::directory_iterator(path))
                    files.push_back(boost::json::value(e.path().filename().string()));
            } catch (const fs::filesystem_error& e) {
                co_return boost::json::value(std::string("Error: ") + e.what());
            }
            co_return boost::json::value(std::move(files));
        }));

    // -- search_files --
    tools.add(ai::tool("search_files",
        ai::schema::JsonSchema::object({
            {"pattern", ai::schema::JsonSchema::string("Grep pattern")},
            {"path", ai::schema::JsonSchema::string("Directory to search")},
            {"file_glob", ai::schema::JsonSchema::string("File filter (e.g. *.cpp)")}
        }).required({"pattern", "path"}),
        "Grep for a pattern in files",
        [](boost::json::value in, ai::ToolExecutionContext) -> ai::Task<boost::json::value> {
            auto pat = std::string(in.at("pattern").as_string());
            auto path = std::string(in.at("path").as_string());
            std::string inc = "*";
            if (auto it = in.as_object().find("file_glob");
                it != in.as_object().end() && it->value().is_string())
                inc = std::string(it->value().as_string());
            auto out = exec_cmd("grep -rn --include='" + inc + "' '" + pat + "' " + path + " | head -50");
            co_return boost::json::value(out.empty() ? "No matches" : out);
        }));

    // -- Agent setup --
    ai::ToolLoopAgent agent({
        .model = model,
        .tools = std::move(tools),
        .instructions =
            "You are an expert coding agent. You can read, write, edit, and search files, "
            "and run shell commands. Current working directory: " + cwd + "\n\n"
            "Guidelines:\n"
            "- Always read a file before editing it\n"
            "- Make minimal, targeted edits\n"
            "- Run tests after changes to verify correctness\n"
            "- If a command fails, diagnose the issue before retrying\n"
            "- Explain your reasoning briefly before acting\n",
        .max_steps = 50,
        .on_step_finish = [](const ai::StepResult& step) {
            if (auto t = step.result.text(); !t.empty()) std::cout << t << std::flush;
            for (auto& tr : step.tool_results) {
                auto s = boost::json::serialize(tr.output);
                if (s.size() > 120) s = s.substr(0, 120) + "...";
                std::cout << "\n  \033[36m[" << tr.tool_name << "]\033[0m " << s << "\n" << std::flush;
            }
        },
    });

    // -- Interactive loop --
    std::cout << "Coding Agent (ai-sdk-cpp) | Model: claude-sonnet-4 | CWD: " << cwd << "\n";
    std::cout << "Enter your request (blank line to send, Ctrl+D to quit):\n\n> " << std::flush;

    std::string line, buffer;
    while (std::getline(std::cin, line)) {
        if (line.empty() && !buffer.empty()) {
            std::cout << "\n";
            auto task = agent.call(buffer);
            task.start();
            while (!task.done()) ioc.run_one();
            try {
                auto res = task.get();
                if (!res.text.empty()) std::cout << "\n" << res.text << "\n";
                std::cout << "\n--- " << res.steps.size() << " steps | "
                          << res.usage.input_tokens.total.value_or(0) << " in / "
                          << res.usage.output_tokens.total.value_or(0) << " out tokens ---\n\n> ";
            } catch (const std::exception& e) {
                std::cerr << "\nError: " << e.what() << "\n\n> ";
            }
            buffer.clear();
        } else {
            if (!buffer.empty()) buffer += "\n";
            buffer += line;
            std::cout << "  " << std::flush;
        }
    }
    return 0;
}
```

## Building and Running

### Build

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)
```

### Set Your API Key

The provider reads credentials from environment variables:

```bash
# Anthropic
export ANTHROPIC_API_KEY="sk-ant-..."

# OpenAI (if using OpenAI provider instead)
export OPENAI_API_KEY="sk-..."

# Google (if using Google provider)
export GOOGLE_GENERATIVE_AI_API_KEY="..."

# Or any other supported provider:
# XAI_API_KEY, GROQ_API_KEY, MISTRAL_API_KEY, FIREWORKS_API_KEY,
# TOGETHER_AI_API_KEY, PERPLEXITY_API_KEY, COHERE_API_KEY, DEEPSEEK_API_KEY
```

### Run

```bash
./coding-agent
```

Then try prompts like:

- "Read the CMakeLists.txt and explain the build structure"
- "Find all TODO comments in this project"
- "Add error handling to the parse_config function in src/config.cpp"
- "Run the test suite and fix any failures"
- "Refactor the Logger class to use the builder pattern"

## Design Decisions and Tips

**Keep tool descriptions precise.** The model decides which tool to use based on descriptions. Vague descriptions lead to wrong tool choices.

**Truncate large outputs.** If `read_file` returns a 10,000-line file, you waste context tokens. Implement pagination (offset/limit) and let the model request specific sections.

**The `edit_file` tool is better than `write_file` for modifications.** It uses less context (only the changed lines, not the entire file), and it naturally prevents the model from accidentally dropping content it wasn't thinking about.

**Set `max_steps` high enough.** Complex tasks (multi-file refactors, debugging test failures) can easily take 20-40 steps. A limit of 50 is a good default; for simple Q&A you could use 10.

**Instrument with `on_step_finish`.** Even if you don't display to a terminal, log steps somewhere. When the agent gets stuck in a loop, you need visibility into what it's doing.

**Handle the event loop carefully.** `ioc.run_one()` processes exactly one handler. This means your main thread remains responsive. For GUI applications, you can integrate the `io_context` with your UI event loop.

## Next Steps

From here, you can extend your agent with:

- **MCP (Model Context Protocol)** integration via `ai/mcp/mcp_client.hpp` to connect to external tool servers
- **Middleware** (`ai/model/middleware.hpp`) for logging, caching, or request transformation
- **Structured output** via `generate_object` for cases where you need typed responses
- **Parallel tool execution** -- the agent loop already supports concurrent tool calls when the model requests multiple tools in one turn
- **Token budget management** -- use `ai/util/token_count.hpp` to track usage and compact history when approaching limits

Happy building.
