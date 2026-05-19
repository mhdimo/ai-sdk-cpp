#include <ai/ai.hpp>
#include <ai/providers/anthropic/anthropic.hpp>
#include <ai/providers/openai/openai.hpp>
#include <ai/providers/bedrock/bedrock.hpp>
#include <boost/asio.hpp>
#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <array>
#include <memory>
#include <filesystem>
#include <termios.h>
#include <unistd.h>
#include <csignal>

namespace fs = std::filesystem;

namespace {

struct termios g_orig_termios;
bool g_raw_active = false;

void restore_terminal() {
    if (g_raw_active) {
        write(STDOUT_FILENO, "\x1b[?2004l", 8);
        write(STDOUT_FILENO, "\x1b[<u", 4);
        write(STDOUT_FILENO, "\x1b[>4;0m", 7);
        tcsetattr(STDIN_FILENO, TCSAFLUSH, &g_orig_termios);
        g_raw_active = false;
    }
}

void signal_handler(int) {
    restore_terminal();
    write(STDOUT_FILENO, "\n", 1);
    _exit(0);
}

std::string g_cwd;

std::string exec_command(const std::string& cmd) {
    std::string full_cmd = "cd " + g_cwd + " && " + cmd + " 2>&1";
    std::array<char, 4096> buffer;
    std::string result;
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(full_cmd.c_str(), "r"), pclose);
    if (!pipe) return "Error: failed to run command";
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        result += buffer.data();
    }
    if (result.size() > 50000) {
        result = result.substr(0, 50000) + "\n... (truncated)";
    }
    return result;
}

std::string resolve_path(const std::string& path) {
    if (path.empty()) return path;
    if (path[0] == '/') return path;
    return g_cwd + "/" + path;
}

std::string read_file_impl(const std::string& path) {
    auto resolved = resolve_path(path);
    std::ifstream f(resolved);
    if (!f.is_open()) return "Error: cannot open file " + resolved;
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

std::string write_file_impl(const std::string& path, const std::string& content) {
    auto resolved = resolve_path(path);
    fs::create_directories(fs::path(resolved).parent_path());
    std::ofstream f(resolved);
    if (!f.is_open()) return "Error: cannot write file " + resolved;
    f << content;
    return "Written " + std::to_string(content.size()) + " bytes to " + resolved;
}

// --- Terminal input handling ---

struct Terminal {
    struct termios orig;
    bool raw_mode = false;

    void enable_raw() {
        tcgetattr(STDIN_FILENO, &orig);
        g_orig_termios = orig;
        struct termios r = orig;
        r.c_lflag &= ~(ICANON | ECHO | ISIG | IEXTEN);
        r.c_iflag &= ~(ICRNL | INLCR | IGNCR | IXON | IXOFF);
        r.c_oflag &= ~(OPOST);
        r.c_cc[VMIN] = 1;
        r.c_cc[VTIME] = 0;
        tcsetattr(STDIN_FILENO, TCSAFLUSH, &r);
        raw_mode = true;
        g_raw_active = true;

        // Enable bracketed paste mode
        write(STDOUT_FILENO, "\x1b[?2004h", 8);
        // Enable Kitty keyboard protocol (shift+enter → CSI 13;2u)
        write(STDOUT_FILENO, "\x1b[>1u", 5);
        // Enable xterm modifyOtherKeys mode 2 (shift+enter → CSI 27;2;13~)
        write(STDOUT_FILENO, "\x1b[>4;2m", 7);
    }

    void disable_raw() {
        if (!raw_mode) return;
        write(STDOUT_FILENO, "\x1b[?2004l", 8);
        write(STDOUT_FILENO, "\x1b[<u", 4);
        write(STDOUT_FILENO, "\x1b[>4;0m", 7); // disable modifyOtherKeys
        tcsetattr(STDIN_FILENO, TCSAFLUSH, &orig);
        raw_mode = false;
        g_raw_active = false;
    }

    void pause_raw() {
        tcsetattr(STDIN_FILENO, TCSAFLUSH, &orig);
    }

    void resume_raw() {
        struct termios r = orig;
        r.c_lflag &= ~(ICANON | ECHO | ISIG | IEXTEN);
        r.c_iflag &= ~(ICRNL | INLCR | IGNCR | IXON | IXOFF);
        r.c_oflag &= ~(OPOST);
        r.c_cc[VMIN] = 1;
        r.c_cc[VTIME] = 0;
        tcsetattr(STDIN_FILENO, TCSAFLUSH, &r);
    }

    int read_byte(int timeout_ms = 0) {
        if (timeout_ms > 0) {
            struct termios tmp;
            tcgetattr(STDIN_FILENO, &tmp);
            tmp.c_cc[VMIN] = 0;
            tmp.c_cc[VTIME] = (timeout_ms + 99) / 100; // deciseconds
            tcsetattr(STDIN_FILENO, TCSAFLUSH, &tmp);
            char c;
            int n = read(STDIN_FILENO, &c, 1);
            resume_raw();
            return n == 1 ? (unsigned char)c : -1;
        }
        char c;
        return read(STDIN_FILENO, &c, 1) == 1 ? (unsigned char)c : -1;
    }

    // Read a full CSI sequence after seeing ESC [
    std::string read_csi() {
        std::string seq;
        while (true) {
            int c = read_byte(50);
            if (c < 0) break;
            seq += (char)c;
            // Final byte of CSI is in range 0x40-0x7E (excluding [ and ;)
            if (c >= 0x40 && c <= 0x7E) break;
        }
        return seq;
    }
};

enum class InputEvent {
    Send,       // User pressed Enter — send the buffer
    Newline,    // User wants a newline in the buffer
    Char,       // Regular character
    Backspace,
    Eof,
    None,
};

struct InputResult {
    InputEvent event;
    char ch = 0;
    std::string paste_text;
};

InputResult read_input(Terminal& term) {
    int c = term.read_byte();
    if (c < 0) return {InputEvent::Eof};

    if (c == 4) return {InputEvent::Eof}; // Ctrl+D
    if (c == 3) return {InputEvent::Eof}; // Ctrl+C

    if (c == '\x1b') {
        int next = term.read_byte(50);
        if (next < 0) return {InputEvent::None}; // bare Escape

        if (next == '[') {
            auto seq = term.read_csi();
            // Shift+Enter via Kitty protocol: CSI 13;2u
            if (seq == "13;2u") return {InputEvent::Newline};
            // Ctrl+Enter: CSI 13;5u
            if (seq == "13;5u") return {InputEvent::Newline};
            // Shift+Enter via xterm modifyOtherKeys: CSI 27;2;13~
            if (seq == "27;2;13~") return {InputEvent::Newline};
            // Bracketed paste start: CSI 200~
            if (seq == "200~") {
                std::string paste;
                while (true) {
                    int pc = term.read_byte();
                    if (pc < 0) break;
                    if (pc == '\x1b') {
                        int pn = term.read_byte(50);
                        if (pn == '[') {
                            auto pseq = term.read_csi();
                            if (pseq == "201~") break; // paste end
                            paste += "\x1b[" + pseq;
                        } else if (pn >= 0) {
                            paste += '\x1b';
                            paste += (char)pn;
                        }
                    } else {
                        paste += (char)pc;
                    }
                }
                InputResult r{InputEvent::Char};
                r.paste_text = std::move(paste);
                return r;
            }
            // Ignore other CSI sequences (arrows, etc.)
            return {InputEvent::None};
        }

        // Option+Enter (ESC followed by CR or LF) → newline
        if (next == '\r' || next == '\n') return {InputEvent::Newline};

        // Other Alt+ combos — ignore
        return {InputEvent::None};
    }

    if (c == '\r' || c == '\n') return {InputEvent::Newline};
    if (c == 127 || c == 8) return {InputEvent::Backspace};

    if (c >= 32 || c == '\t') {
        InputResult r{InputEvent::Char};
        r.ch = (char)c;
        return r;
    }

    return {InputEvent::None};
}

// Redraw the current input line(s)
void display_input(const std::string& buffer, int cursor_line) {
    // Simple: just show what's there. Multi-line shows continuation prompt.
    (void)buffer;
    (void)cursor_line;
}

} // namespace

int main(int argc, char* argv[]) {
    boost::asio::io_context ioc;

    std::string provider_name = "anthropic";
    std::string model_name = "claude-sonnet-4-20250514";
    g_cwd = fs::current_path().string();

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--cwd" && i + 1 < argc) {
            g_cwd = fs::absolute(argv[++i]).string();
        } else if (provider_name == "anthropic" && model_name == "claude-sonnet-4-20250514") {
            if (arg == "anthropic" || arg == "openai" || arg == "bedrock") {
                provider_name = arg;
            } else {
                model_name = arg;
            }
        } else {
            model_name = arg;
        }
    }

    ai::LanguageModelPtr model;
    ai::ProviderPtr provider_instance;

    if (provider_name == "anthropic") {
        provider_instance = ai::providers::anthropic::create_anthropic({.io_context = ioc});
    } else if (provider_name == "openai") {
        provider_instance = ai::providers::openai::create_openai({.io_context = ioc});
    } else if (provider_name == "bedrock") {
        provider_instance = ai::providers::bedrock::create_bedrock({.io_context = ioc});
    } else {
        std::cerr << "Unknown provider: " << provider_name << "\n";
        return 1;
    }

    model = provider_instance->language_model(model_name);

    ai::ToolSet tools;

    tools.add(ai::tool("read_file",
        ai::schema::JsonSchema::object({
            {"path", ai::schema::JsonSchema::string("File path (relative to cwd or absolute)")}
        }).required({"path"}).additional_properties(false),
        "Read the contents of a file",
        [](boost::json::value input, ai::ToolExecutionContext) -> ai::Task<boost::json::value> {
            auto path = std::string(input.at("path").as_string());
            co_return boost::json::value(read_file_impl(path));
        }
    ));

    tools.add(ai::tool("write_file",
        ai::schema::JsonSchema::object({
            {"path", ai::schema::JsonSchema::string("File path (relative to cwd or absolute)")},
            {"content", ai::schema::JsonSchema::string("Content to write")}
        }).required({"path", "content"}).additional_properties(false),
        "Create or overwrite a file",
        [](boost::json::value input, ai::ToolExecutionContext) -> ai::Task<boost::json::value> {
            auto path = std::string(input.at("path").as_string());
            auto content = std::string(input.at("content").as_string());
            co_return boost::json::value(write_file_impl(path, content));
        }
    ));

    tools.add(ai::tool("run_command",
        ai::schema::JsonSchema::object({
            {"command", ai::schema::JsonSchema::string("Shell command to execute")}
        }).required({"command"}).additional_properties(false),
        "Run a shell command in the working directory. Returns stdout+stderr.",
        [](boost::json::value input, ai::ToolExecutionContext) -> ai::Task<boost::json::value> {
            auto cmd = std::string(input.at("command").as_string());
            co_return boost::json::value(exec_command(cmd));
        }
    ));

    tools.add(ai::tool("list_directory",
        ai::schema::JsonSchema::object({
            {"path", ai::schema::JsonSchema::string("Directory path (relative to cwd or absolute)")},
            {"recursive", ai::schema::JsonSchema::boolean("List recursively")}
        }).required({"path"}),
        "List files in a directory",
        [](boost::json::value input, ai::ToolExecutionContext) -> ai::Task<boost::json::value> {
            auto path = resolve_path(std::string(input.at("path").as_string()));
            bool recursive = false;
            if (auto it = input.as_object().find("recursive"); it != input.as_object().end() && it->value().is_bool()) {
                recursive = it->value().as_bool();
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
            {"pattern", ai::schema::JsonSchema::string("Text pattern to search for")},
            {"path", ai::schema::JsonSchema::string("Directory to search in")},
            {"file_pattern", ai::schema::JsonSchema::string("File glob (e.g. *.cpp)")}
        }).required({"pattern"}),
        "Search for a text pattern in files (grep -rn)",
        [](boost::json::value input, ai::ToolExecutionContext) -> ai::Task<boost::json::value> {
            auto pattern = std::string(input.at("pattern").as_string());
            std::string path = ".";
            if (auto it = input.as_object().find("path"); it != input.as_object().end() && it->value().is_string())
                path = std::string(it->value().as_string());
            std::string cmd = "grep -rn '" + pattern + "' " + path;
            if (auto it = input.as_object().find("file_pattern"); it != input.as_object().end() && it->value().is_string())
                cmd += " --include='" + std::string(it->value().as_string()) + "'";
            cmd += " | head -50";
            co_return boost::json::value(exec_command(cmd));
        }
    ));

    ai::ToolLoopAgent agent({
        .model = model,
        .tools = std::move(tools),
        .instructions = "You are a coding agent. You have access to read/write files, "
                        "run shell commands, list directories, and search code.\n\n"
                        "Working directory: " + g_cwd + "\n\n"
                        "IMPORTANT RULES:\n"
                        "- Your tools ALWAYS work. If a tool returned data, it succeeded. NEVER claim tools returned empty results.\n"
                        "- Do NOT repeat tool calls. Each tool call returns complete results on the first try.\n"
                        "- To understand a project, read CLAUDE.md or README.md first, then explore from there.\n"
                        "- Ignore build artifacts (CMakeFiles/, _deps/, *.o, CMakeCache.txt) when exploring.\n"
                        "- Use list_directory for structure, read_file for content, run_command for actions.\n"
                        "- All relative paths resolve from the working directory.\n"
                        "- Read files before editing them.\n"
                        "- Keep responses concise.\n",
        .max_steps = 50,
        .on_step_finish = [](const ai::StepResult& step) {
            auto text = step.result.text();
            if (!text.empty()) {
                std::cout << text << std::flush;
            }
            for (auto& tr : step.tool_results) {
                std::cout << "\n  \x1b[90m[" << tr.tool_name << "] "
                          << boost::json::serialize(tr.output)
                          << "\x1b[0m\n" << std::flush;
            }
        },
    });

    // --- UI ---
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    Terminal term;
    term.enable_raw();

    auto write_str = [](const char* s) { write(STDOUT_FILENO, s, strlen(s)); };

    write_str("\x1b[1mAI Agent\x1b[0m (");
    write_str(provider_name.c_str());
    write_str("/");
    write_str(model_name.c_str());
    write_str(")\r\n\x1b[90mcwd: ");
    write_str(g_cwd.c_str());
    write_str("\x1b[0m\r\n");
    write_str("Enter for newline │ double-Enter to send │ Ctrl+C to exit\r\n\r\n");
    write_str("\x1b[1m>\x1b[0m ");

    std::string input_buffer;

    auto do_send = [&]() {
        // Trim trailing newlines from the buffer
        while (!input_buffer.empty() && input_buffer.back() == '\n')
            input_buffer.pop_back();

        if (input_buffer.empty()) {
            write_str("\x1b[1m>\x1b[0m ");
            return;
        }

        term.pause_raw();
        {
            auto task = agent.call(input_buffer);
            task.start();
            while (!task.done()) {
                ioc.run_one();
            }
            try {
                auto res = task.get();
                std::cout << "\n\x1b[90m[" << res.steps.size() << " steps | "
                          << res.usage.input_tokens.total.value_or(0) << "→"
                          << res.usage.output_tokens.total.value_or(0) << " tokens]\x1b[0m\n";
            } catch (const std::exception& e) {
                std::cerr << "\n\x1b[31mError: " << e.what() << "\x1b[0m\n";
            }
        }
        term.resume_raw();

        input_buffer.clear();
        write_str("\r\n\x1b[1m>\x1b[0m ");
    };

    while (true) {
        auto result = read_input(term);

        switch (result.event) {
        case InputEvent::Eof:
            goto done;

        case InputEvent::Send:
            // shouldn't happen anymore, but handle it
            write_str("\r\n");
            do_send();
            break;

        case InputEvent::Newline:
            // Double-Enter: if buffer ends with \n, send
            if (!input_buffer.empty() && input_buffer.back() == '\n') {
                write_str("\r\n");
                do_send();
            } else {
                input_buffer += '\n';
                write_str("\r\n\x1b[90m│\x1b[0m ");
            }
            break;

        case InputEvent::Char:
            if (!result.paste_text.empty()) {
                // Pasted text — add it all, showing newlines as continuation
                for (char pc : result.paste_text) {
                    if (pc == '\n' || pc == '\r') {
                        input_buffer += '\n';
                        write_str("\r\n\x1b[90m│\x1b[0m ");
                    } else {
                        input_buffer += pc;
                        write(STDOUT_FILENO, &pc, 1);
                    }
                }
            } else {
                input_buffer += result.ch;
                write(STDOUT_FILENO, &result.ch, 1);
            }
            break;

        case InputEvent::Backspace:
            if (!input_buffer.empty()) {
                if (input_buffer.back() == '\n') {
                    // TODO: proper multi-line backspace with cursor movement
                    input_buffer.pop_back();
                    write_str("\x1b[A\x1b[999C"); // move up, end of line
                } else {
                    input_buffer.pop_back();
                    write_str("\b \b");
                }
            }
            break;

        case InputEvent::None:
            break;
        }
    }

done:
    term.disable_raw();
    std::cout << "\n";
    return 0;
}
