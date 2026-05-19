#include <ai/mcp/mcp_client.hpp>
#include <ai/util/json.hpp>
#include <boost/json.hpp>

#include <array>
#include <atomic>
#include <cstdio>
#include <iostream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

#ifdef _WIN32
#include <windows.h>
#else
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace ai::mcp {

namespace {

/// Reads a single JSON-RPC message from a FILE* stream.
/// MCP uses Content-Length header framing (like LSP).
std::string read_message(FILE* stream) {
    // Read headers until empty line
    std::string headers;
    int content_length = -1;

    while (true) {
        char buf[1024];
        if (!fgets(buf, sizeof(buf), stream)) {
            throw std::runtime_error("MCP: Failed to read from server (EOF or error)");
        }
        std::string line(buf);

        // Strip trailing \r\n or \n
        while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) {
            line.pop_back();
        }

        if (line.empty()) {
            // Empty line marks end of headers
            break;
        }

        // Parse Content-Length header
        if (line.substr(0, 16) == "Content-Length: ") {
            content_length = std::stoi(line.substr(16));
        }
    }

    if (content_length < 0) {
        throw std::runtime_error("MCP: Missing Content-Length header in server response");
    }

    // Read exactly content_length bytes
    std::string body(content_length, '\0');
    size_t bytes_read = fread(body.data(), 1, content_length, stream);
    if (static_cast<int>(bytes_read) != content_length) {
        throw std::runtime_error("MCP: Incomplete message body from server");
    }

    return body;
}

/// Writes a JSON-RPC message to a FILE* stream with Content-Length framing.
void write_message(FILE* stream, const std::string& body) {
    std::string header = "Content-Length: " + std::to_string(body.size()) + "\r\n\r\n";
    fwrite(header.data(), 1, header.size(), stream);
    fwrite(body.data(), 1, body.size(), stream);
    fflush(stream);
}

} // namespace

struct McpClient::Impl {
    McpServerConfig config;
    std::atomic<bool> connected{false};
    std::atomic<int> next_id{1};
    std::mutex io_mutex;

#ifdef _WIN32
    HANDLE child_process = nullptr;
    FILE* child_stdin = nullptr;
    FILE* child_stdout = nullptr;
#else
    pid_t child_pid = -1;
    FILE* child_stdin = nullptr;
    FILE* child_stdout = nullptr;
    int stdin_fd = -1;
    int stdout_fd = -1;
#endif

    boost::json::value send_request(const std::string& method, boost::json::value params = boost::json::value()) {
        if (!connected) {
            throw std::runtime_error("MCP: Not connected to server");
        }

        int id = next_id.fetch_add(1);

        boost::json::object request;
        request["jsonrpc"] = "2.0";
        request["id"] = id;
        request["method"] = method;
        if (!params.is_null()) {
            request["params"] = params;
        }

        std::string body = boost::json::serialize(request);

        std::lock_guard lock(io_mutex);
        write_message(child_stdin, body);

        // Read response (skip notifications)
        while (true) {
            std::string response_body = read_message(child_stdout);
            auto response = ai::json::parse(response_body);

            if (!response.is_object()) {
                throw std::runtime_error("MCP: Invalid JSON-RPC response");
            }

            auto& obj = response.as_object();

            // Skip notifications (no "id" field)
            auto id_it = obj.find("id");
            if (id_it == obj.end() || id_it->value().is_null()) {
                // This is a notification, skip it
                continue;
            }

            // Check if this is our response
            if (id_it->value().is_int64() && id_it->value().as_int64() == id) {
                // Check for error
                auto err_it = obj.find("error");
                if (err_it != obj.end() && !err_it->value().is_null()) {
                    auto& err = err_it->value().as_object();
                    std::string err_msg = "MCP error";
                    auto msg_it = err.find("message");
                    if (msg_it != err.end() && msg_it->value().is_string()) {
                        err_msg = std::string(msg_it->value().as_string());
                    }
                    throw std::runtime_error("MCP: Server error: " + err_msg);
                }

                auto result_it = obj.find("result");
                if (result_it != obj.end()) {
                    return result_it->value();
                }
                return boost::json::value();
            }
        }
    }

    void send_notification(const std::string& method, boost::json::value params = boost::json::value()) {
        if (!connected) return;

        boost::json::object notification;
        notification["jsonrpc"] = "2.0";
        notification["method"] = method;
        if (!params.is_null()) {
            notification["params"] = params;
        }

        std::string body = boost::json::serialize(notification);

        std::lock_guard lock(io_mutex);
        write_message(child_stdin, body);
    }

    void spawn_process() {
        if (config.transport != "stdio") {
            throw std::runtime_error("MCP: Only 'stdio' transport is currently supported");
        }

        if (config.command.empty()) {
            throw std::runtime_error("MCP: No command specified for stdio transport");
        }

#ifdef _WIN32
        // Windows implementation using CreateProcess with pipes
        SECURITY_ATTRIBUTES sa;
        sa.nLength = sizeof(SECURITY_ATTRIBUTES);
        sa.bInheritHandle = TRUE;
        sa.lpSecurityDescriptor = nullptr;

        HANDLE stdin_read, stdin_write, stdout_read, stdout_write;
        CreatePipe(&stdin_read, &stdin_write, &sa, 0);
        CreatePipe(&stdout_read, &stdout_write, &sa, 0);

        SetHandleInformation(stdin_write, HANDLE_FLAG_INHERIT, 0);
        SetHandleInformation(stdout_read, HANDLE_FLAG_INHERIT, 0);

        STARTUPINFOA si = {};
        si.cb = sizeof(si);
        si.hStdInput = stdin_read;
        si.hStdOutput = stdout_write;
        si.hStdError = GetStdHandle(STD_ERROR_HANDLE);
        si.dwFlags |= STARTF_USESTDHANDLES;

        std::string cmd_line = config.command;
        for (auto& arg : config.args) {
            cmd_line += " " + arg;
        }

        PROCESS_INFORMATION pi = {};
        if (!CreateProcessA(nullptr, cmd_line.data(), nullptr, nullptr, TRUE,
                           0, nullptr, nullptr, &si, &pi)) {
            throw std::runtime_error("MCP: Failed to spawn server process");
        }

        CloseHandle(stdin_read);
        CloseHandle(stdout_write);
        CloseHandle(pi.hThread);

        child_process = pi.hProcess;
        int stdin_fd_w = _open_osfhandle((intptr_t)stdin_write, 0);
        int stdout_fd_r = _open_osfhandle((intptr_t)stdout_read, 0);
        child_stdin = _fdopen(stdin_fd_w, "w");
        child_stdout = _fdopen(stdout_fd_r, "r");
#else
        // Unix implementation using fork/exec with pipes
        int stdin_pipe[2];   // parent writes to stdin_pipe[1], child reads from stdin_pipe[0]
        int stdout_pipe[2];  // child writes to stdout_pipe[1], parent reads from stdout_pipe[0]

        if (pipe(stdin_pipe) != 0 || pipe(stdout_pipe) != 0) {
            throw std::runtime_error("MCP: Failed to create pipes");
        }

        pid_t pid = fork();
        if (pid < 0) {
            throw std::runtime_error("MCP: Failed to fork process");
        }

        if (pid == 0) {
            // Child process
            close(stdin_pipe[1]);
            close(stdout_pipe[0]);

            dup2(stdin_pipe[0], STDIN_FILENO);
            dup2(stdout_pipe[1], STDOUT_FILENO);

            close(stdin_pipe[0]);
            close(stdout_pipe[1]);

            // Set environment variables
            for (auto& [key, val] : config.env) {
                setenv(key.c_str(), val.c_str(), 1);
            }

            // Build argv
            std::vector<const char*> argv;
            argv.push_back(config.command.c_str());
            for (auto& arg : config.args) {
                argv.push_back(arg.c_str());
            }
            argv.push_back(nullptr);

            execvp(config.command.c_str(), const_cast<char* const*>(argv.data()));
            // If exec fails
            _exit(127);
        }

        // Parent process
        close(stdin_pipe[0]);
        close(stdout_pipe[1]);

        child_pid = pid;
        stdin_fd = stdin_pipe[1];
        stdout_fd = stdout_pipe[0];
        child_stdin = fdopen(stdin_fd, "w");
        child_stdout = fdopen(stdout_fd, "r");

        if (!child_stdin || !child_stdout) {
            kill(child_pid, SIGTERM);
            throw std::runtime_error("MCP: Failed to open file streams for pipes");
        }
#endif
    }

    void kill_process() {
#ifdef _WIN32
        if (child_process) {
            TerminateProcess(child_process, 0);
            CloseHandle(child_process);
            child_process = nullptr;
        }
        if (child_stdin) { fclose(child_stdin); child_stdin = nullptr; }
        if (child_stdout) { fclose(child_stdout); child_stdout = nullptr; }
#else
        if (child_stdin) { fclose(child_stdin); child_stdin = nullptr; }
        if (child_stdout) { fclose(child_stdout); child_stdout = nullptr; }
        if (child_pid > 0) {
            kill(child_pid, SIGTERM);
            int status;
            waitpid(child_pid, &status, 0);
            child_pid = -1;
        }
#endif
    }
};

McpClient::McpClient(McpServerConfig config)
    : impl_(std::make_unique<Impl>()) {
    impl_->config = std::move(config);
}

McpClient::~McpClient() {
    disconnect();
}

McpClient::McpClient(McpClient&& other) noexcept = default;
McpClient& McpClient::operator=(McpClient&& other) noexcept = default;

Task<void> McpClient::connect() {
    // Spawn the subprocess
    impl_->spawn_process();
    impl_->connected = true;

    // Send initialize request
    boost::json::object init_params;
    init_params["protocolVersion"] = "2024-11-05";
    boost::json::object client_info;
    client_info["name"] = "ai-sdk-cpp";
    client_info["version"] = "0.1.0";
    init_params["clientInfo"] = client_info;

    boost::json::object capabilities;
    init_params["capabilities"] = capabilities;

    impl_->send_request("initialize", boost::json::value(init_params));

    // Send initialized notification
    impl_->send_notification("notifications/initialized");

    co_return;
}

void McpClient::disconnect() {
    if (impl_ && impl_->connected) {
        impl_->connected = false;
        impl_->kill_process();
    }
}

Task<std::vector<McpTool>> McpClient::list_tools() {
    auto result = impl_->send_request("tools/list");
    std::vector<McpTool> tools;

    if (!result.is_object()) {
        co_return tools;
    }

    auto& obj = result.as_object();
    auto tools_it = obj.find("tools");
    if (tools_it == obj.end() || !tools_it->value().is_array()) {
        co_return tools;
    }

    for (auto& tool_val : tools_it->value().as_array()) {
        if (!tool_val.is_object()) continue;
        auto& tool_obj = tool_val.as_object();

        McpTool tool;

        auto name_it = tool_obj.find("name");
        if (name_it != tool_obj.end() && name_it->value().is_string()) {
            tool.name = std::string(name_it->value().as_string());
        }

        auto desc_it = tool_obj.find("description");
        if (desc_it != tool_obj.end() && desc_it->value().is_string()) {
            tool.description = std::string(desc_it->value().as_string());
        }

        auto schema_it = tool_obj.find("inputSchema");
        if (schema_it != tool_obj.end() && schema_it->value().is_object()) {
            tool.input_schema = schema::JsonSchema(
                boost::json::object(schema_it->value().as_object())
            );
        }

        tools.push_back(std::move(tool));
    }

    co_return tools;
}

Task<boost::json::value> McpClient::call_tool(std::string name, boost::json::value input) {
    boost::json::object params;
    params["name"] = name;
    if (input.is_object()) {
        params["arguments"] = input;
    } else {
        params["arguments"] = boost::json::object{};
    }

    auto result = impl_->send_request("tools/call", boost::json::value(params));

    // Parse the result - MCP tools/call returns { content: [...] }
    if (!result.is_object()) {
        co_return result;
    }

    auto& obj = result.as_object();
    auto content_it = obj.find("content");
    if (content_it == obj.end() || !content_it->value().is_array()) {
        co_return result;
    }

    // Concatenate text content parts
    auto& content_arr = content_it->value().as_array();
    std::string text_result;
    for (auto& part : content_arr) {
        if (!part.is_object()) continue;
        auto& part_obj = part.as_object();
        auto type_it = part_obj.find("type");
        if (type_it == part_obj.end()) continue;

        if (type_it->value().is_string() && type_it->value().as_string() == "text") {
            auto text_it = part_obj.find("text");
            if (text_it != part_obj.end() && text_it->value().is_string()) {
                if (!text_result.empty()) text_result += "\n";
                text_result += std::string(text_it->value().as_string());
            }
        }
    }

    if (!text_result.empty()) {
        // Try to parse as JSON, otherwise return as string
        auto parsed = ai::json::safe_parse(text_result);
        if (parsed) {
            co_return *parsed;
        }
        co_return boost::json::value(text_result);
    }

    // Return the raw result if no text content found
    co_return result;
}

bool McpClient::is_connected() const {
    return impl_ && impl_->connected;
}

const std::string& McpClient::server_name() const {
    return impl_->config.name;
}

ai::ToolSet mcp_tools_to_toolset(std::shared_ptr<McpClient> client, const std::vector<McpTool>& tools) {
    ToolSet toolset;

    for (auto& mcp_tool : tools) {
        auto tool_name = mcp_tool.name;
        auto captured_client = client;

        ToolDefinition def{
            .name = mcp_tool.name,
            .description = mcp_tool.description,
            .input_schema = mcp_tool.input_schema,
            .strict = false,
            .execute = [captured_client, tool_name](
                boost::json::value input,
                ToolExecutionContext ctx
            ) -> Task<boost::json::value> {
                co_return co_await captured_client->call_tool(tool_name, std::move(input));
            },
        };

        toolset.add(std::move(def));
    }

    return toolset;
}

} // namespace ai::mcp
