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

namespace {

std::string exec(const std::string& cmd) {
    std::array<char, 8192> buffer;
    std::string result;
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd.c_str(), "r"), pclose);
    if (!pipe) return "Error: popen failed";
    while (fgets(buffer.data(), buffer.size(), pipe.get())) {
        result += buffer.data();
    }
    return result;
}

} // namespace

int main() {
    boost::asio::io_context ioc;

    auto anthropic = ai::providers::anthropic::create_anthropic({
        .base_url = "https://api.z.ai/api/anthropic",
        .io_context = ioc,
    });

    auto model = anthropic->language_model("glm-5.1");

    ai::ToolSet tools;

    tools.add(ai::tool("read_file",
        ai::schema::JsonSchema::object({
            {"path", ai::schema::JsonSchema::string("Absolute file path")}
        }).required({"path"}).additional_properties(false),
        "Read the contents of a file",
        [](boost::json::value input, ai::ToolExecutionContext) -> ai::Task<boost::json::value> {
            auto path = std::string(input.at("path").as_string());
            std::ifstream f(path);
            if (!f.is_open()) co_return boost::json::value("Error: cannot open " + path);
            std::stringstream ss;
            ss << f.rdbuf();
            co_return boost::json::value(ss.str());
        }
    ));

    tools.add(ai::tool("write_file",
        ai::schema::JsonSchema::object({
            {"path", ai::schema::JsonSchema::string("Absolute file path")},
            {"content", ai::schema::JsonSchema::string("File content to write")}
        }).required({"path", "content"}).additional_properties(false),
        "Create or overwrite a file with the given content",
        [](boost::json::value input, ai::ToolExecutionContext) -> ai::Task<boost::json::value> {
            auto path = std::string(input.at("path").as_string());
            auto content = std::string(input.at("content").as_string());
            fs::create_directories(fs::path(path).parent_path());
            std::ofstream f(path);
            if (!f.is_open()) co_return boost::json::value("Error: cannot write " + path);
            f << content;
            co_return boost::json::value("Written " + std::to_string(content.size()) + " bytes to " + path);
        }
    ));

    tools.add(ai::tool("run_command",
        ai::schema::JsonSchema::object({
            {"command", ai::schema::JsonSchema::string("Shell command to execute")},
            {"working_dir", ai::schema::JsonSchema::string("Working directory (optional)")}
        }).required({"command"}),
        "Execute a shell command and return its output",
        [](boost::json::value input, ai::ToolExecutionContext) -> ai::Task<boost::json::value> {
            auto cmd = std::string(input.at("command").as_string());
            std::string full_cmd = cmd;
            if (auto wd = input.as_object().find("working_dir"); wd != input.as_object().end() && wd->value().is_string()) {
                full_cmd = "cd " + std::string(wd->value().as_string()) + " && " + cmd;
            }
            auto output = exec(full_cmd + " 2>&1");
            if (output.size() > 10000) {
                output = output.substr(0, 10000) + "\n... (truncated)";
            }
            co_return boost::json::value(output);
        }
    ));

    tools.add(ai::tool("list_files",
        ai::schema::JsonSchema::object({
            {"path", ai::schema::JsonSchema::string("Directory path")},
            {"recursive", ai::schema::JsonSchema::boolean("List recursively")}
        }).required({"path"}),
        "List files in a directory",
        [](boost::json::value input, ai::ToolExecutionContext) -> ai::Task<boost::json::value> {
            auto path = std::string(input.at("path").as_string());
            bool recursive = false;
            if (auto r = input.as_object().find("recursive"); r != input.as_object().end() && r->value().is_bool()) {
                recursive = r->value().as_bool();
            }

            boost::json::array files;
            try {
                if (recursive) {
                    for (auto& entry : fs::recursive_directory_iterator(path)) {
                        files.push_back(boost::json::value(entry.path().string()));
                        if (files.size() > 500) break;
                    }
                } else {
                    for (auto& entry : fs::directory_iterator(path)) {
                        files.push_back(boost::json::value(entry.path().string()));
                    }
                }
            } catch (const fs::filesystem_error& e) {
                co_return boost::json::value(std::string("Error: ") + e.what());
            }
            co_return boost::json::value(std::move(files));
        }
    ));

    tools.add(ai::tool("search_files",
        ai::schema::JsonSchema::object({
            {"pattern", ai::schema::JsonSchema::string("Grep pattern to search for")},
            {"path", ai::schema::JsonSchema::string("Directory to search in")},
            {"file_pattern", ai::schema::JsonSchema::string("File glob pattern (e.g. *.cpp)")}
        }).required({"pattern", "path"}),
        "Search for a pattern in files using grep",
        [](boost::json::value input, ai::ToolExecutionContext) -> ai::Task<boost::json::value> {
            auto pattern = std::string(input.at("pattern").as_string());
            auto path = std::string(input.at("path").as_string());
            std::string cmd = "grep -rn --include='*' '" + pattern + "' " + path + " | head -50";
            if (auto fp = input.as_object().find("file_pattern"); fp != input.as_object().end() && fp->value().is_string()) {
                cmd = "grep -rn --include='" + std::string(fp->value().as_string()) + "' '" + pattern + "' " + path + " | head -50";
            }
            auto output = exec(cmd);
            co_return boost::json::value(output);
        }
    ));

    std::string cwd = fs::current_path().string();

    ai::ToolLoopAgent agent({
        .model = model,
        .tools = std::move(tools),
        .instructions = "You are an expert coding assistant, similar to Claude Code. "
                        "You have access to the filesystem and can read, write, and search files, "
                        "as well as run shell commands. The current working directory is: " + cwd + "\n\n"
                        "When helping with code:\n"
                        "- Read existing code before making changes\n"
                        "- Make targeted, minimal edits\n"
                        "- Run tests after making changes\n"
                        "- Explain what you're doing\n",
        .max_steps = 50,
        .on_step_finish = [](const ai::StepResult& step) {
            auto text = step.result.text();
            if (!text.empty()) {
                std::cout << text << std::flush;
            }
            for (auto& tr : step.tool_results) {
                auto output_str = boost::json::serialize(tr.output);
                std::string preview = output_str.size() > 100
                    ? output_str.substr(0, 100) + "..."
                    : output_str;
                std::cout << "\n  → [" << tr.tool_name << "] " << preview << "\n" << std::flush;
            }
        },
    });

    std::cout << "╔══════════════════════════════════════════╗\n";
    std::cout << "║  AI Coding Agent (C++ AI SDK)            ║\n";
    std::cout << "║  Model: " << model << "         ║\n";
    std::cout << "║  CWD: " << cwd.substr(0, 33) << "        ║\n";
    std::cout << "╚══════════════════════════════════════════╝\n\n";
    std::cout << "Type your request (empty line to send, Ctrl+D to exit):\n\n";

    std::string line;
    std::string input_buffer;

    std::cout << "› " << std::flush;
    while (std::getline(std::cin, line)) {
        if (line.empty() && !input_buffer.empty()) {
            std::cout << "\n" << std::flush;
            auto task = agent.call(input_buffer);
            task.start();
            while (!task.done()) {
                ioc.run_one();
            }

            try {
                auto result = task.get();
                if (!result.text.empty()) {
                    std::cout << "\n" << result.text << "\n";
                }
                std::cout << "\n━━━ "
                          << result.steps.size() << " steps | "
                          << result.usage.input_tokens.total.value_or(0) << "→"
                          << result.usage.output_tokens.total.value_or(0) << " tokens"
                          << " ━━━\n\n";
            } catch (const std::exception& e) {
                std::cerr << "\nError: " << e.what() << "\n\n";
            }

            input_buffer.clear();
            std::cout << "› " << std::flush;
        } else {
            if (!input_buffer.empty()) input_buffer += "\n";
            input_buffer += line;
            std::cout << "  " << std::flush;
        }
    }

    std::cout << "\n";
    return 0;
}
