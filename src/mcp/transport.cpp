#include <ai/mcp/transport.hpp>
#include <ai/util/json.hpp>

#include <boost/json.hpp>

#include <array>
#include <cstdio>
#include <cstring>
#include <sstream>
#include <stdexcept>

#ifdef _WIN32
#include <windows.h>
#else
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace ai::mcp {

namespace {

// --- Content-Length framing (LSP-style) for the stdio transport ------------

std::string read_framed(FILE* stream) {
    int content_length = -1;
    while (true) {
        char buf[1024];
        if (!fgets(buf, sizeof(buf), stream)) {
            throw std::runtime_error("MCP: EOF reading from server");
        }
        std::string line(buf);
        while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) {
            line.pop_back();
        }
        if (line.empty()) break;  // end of headers
        if (line.starts_with("Content-Length: ")) {
            content_length = std::stoi(line.substr(16));
        }
    }
    if (content_length < 0) {
        throw std::runtime_error("MCP: missing Content-Length header");
    }
    std::string body(static_cast<size_t>(content_length), '\0');
    size_t n = std::fread(body.data(), 1, static_cast<size_t>(content_length), stream);
    if (static_cast<int>(n) != content_length) {
        throw std::runtime_error("MCP: incomplete message body");
    }
    return body;
}

void write_framed(FILE* stream, const std::string& body) {
    std::string header =
        "Content-Length: " + std::to_string(body.size()) + "\r\n\r\n";
    std::fwrite(header.data(), 1, header.size(), stream);
    std::fwrite(body.data(), 1, body.size(), stream);
    std::fflush(stream);
}

} // namespace

// ---------------------------------------------------------------------------
// StdioTransport
// ---------------------------------------------------------------------------

struct StdioTransport::Impl {
    std::string command;
    std::vector<std::string> args;
    std::unordered_map<std::string, std::string> env;

#ifdef _WIN32
    HANDLE child_process = nullptr;
#else
    pid_t child_pid = -1;
#endif
    FILE* child_stdin = nullptr;
    FILE* child_stdout = nullptr;

    ~Impl() { stop(); }

    void spawn() {
#ifdef _WIN32
        SECURITY_ATTRIBUTES sa{};
        sa.nLength = sizeof(sa);
        sa.bInheritHandle = TRUE;
        HANDLE stdin_read, stdin_write, stdout_read, stdout_write;
        CreatePipe(&stdin_read, &stdin_write, &sa, 0);
        CreatePipe(&stdout_read, &stdout_write, &sa, 0);
        SetHandleInformation(stdin_write, HANDLE_FLAG_INHERIT, 0);
        SetHandleInformation(stdout_read, HANDLE_FLAG_INHERIT, 0);

        STARTUPINFOA si{};
        si.cb = sizeof(si);
        si.hStdInput = stdin_read;
        si.hStdOutput = stdout_write;
        si.hStdError = GetStdHandle(STD_ERROR_HANDLE);
        si.dwFlags |= STARTF_USESTDHANDLES;

        std::string cmd_line = command;
        for (auto& a : args) cmd_line += " " + a;

        PROCESS_INFORMATION pi{};
        if (!CreateProcessA(nullptr, cmd_line.data(), nullptr, nullptr, TRUE,
                            0, nullptr, nullptr, &si, &pi)) {
            throw std::runtime_error("MCP: failed to spawn server process");
        }
        CloseHandle(stdin_read);
        CloseHandle(stdout_write);
        CloseHandle(pi.hThread);
        child_process = pi.hProcess;
        child_stdin = _fdopen(_open_osfhandle(reinterpret_cast<intptr_t>(stdin_write), 0), "w");
        child_stdout = _fdopen(_open_osfhandle(reinterpret_cast<intptr_t>(stdout_read), 0), "r");
#else
        int stdin_pipe[2];
        int stdout_pipe[2];
        if (pipe(stdin_pipe) != 0 || pipe(stdout_pipe) != 0) {
            throw std::runtime_error("MCP: failed to create pipes");
        }
        pid_t pid = fork();
        if (pid < 0) throw std::runtime_error("MCP: failed to fork");
        if (pid == 0) {
            close(stdin_pipe[1]);
            close(stdout_pipe[0]);
            dup2(stdin_pipe[0], STDIN_FILENO);
            dup2(stdout_pipe[1], STDOUT_FILENO);
            close(stdin_pipe[0]);
            close(stdout_pipe[1]);
            for (auto& [k, v] : env) setenv(k.c_str(), v.c_str(), 1);
            std::vector<const char*> argv;
            argv.push_back(command.c_str());
            for (auto& a : args) argv.push_back(a.c_str());
            argv.push_back(nullptr);
            execvp(command.c_str(), const_cast<char* const*>(argv.data()));
            _exit(127);
        }
        close(stdin_pipe[0]);
        close(stdout_pipe[1]);
        child_pid = pid;
        child_stdin = fdopen(stdin_pipe[1], "w");
        child_stdout = fdopen(stdout_pipe[0], "r");
        if (!child_stdin || !child_stdout) {
            kill(child_pid, SIGTERM);
            throw std::runtime_error("MCP: failed to open pipe streams");
        }
#endif
    }

    void stop() {
#ifdef _WIN32
        if (child_stdin) { fclose(child_stdin); child_stdin = nullptr; }
        if (child_stdout) { fclose(child_stdout); child_stdout = nullptr; }
        if (child_process) {
            TerminateProcess(child_process, 0);
            CloseHandle(child_process);
            child_process = nullptr;
        }
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

StdioTransport::StdioTransport(std::string command, std::vector<std::string> args,
                               std::unordered_map<std::string, std::string> env)
    : impl_(std::make_unique<Impl>()) {
    impl_->command = std::move(command);
    impl_->args = std::move(args);
    impl_->env = std::move(env);
}

StdioTransport::~StdioTransport() = default;

void StdioTransport::start() {
    if (impl_->child_stdin) return;  // already started
    impl_->spawn();
}

void StdioTransport::stop() {
    if (impl_) impl_->stop();
}

void StdioTransport::send(const std::string& json_message) {
    if (!impl_ || !impl_->child_stdin) {
        throw std::runtime_error("MCP: transport not started");
    }
    write_framed(impl_->child_stdin, json_message);
}

std::string StdioTransport::receive() {
    if (!impl_ || !impl_->child_stdout) {
        throw std::runtime_error("MCP: transport not started");
    }
    return read_framed(impl_->child_stdout);
}

// ---------------------------------------------------------------------------
// StreamableHttpTransport
// ---------------------------------------------------------------------------

StreamableHttpTransport::StreamableHttpTransport(std::string url,
                                                 std::unordered_map<std::string, std::string> headers,
                                                 std::string api_key)
    : url_(std::move(url)),
      headers_(std::move(headers)),
      api_key_(std::move(api_key)),
      client_(ioc_) {}

void StreamableHttpTransport::start() {}  // connectionless
void StreamableHttpTransport::stop() {}

void StreamableHttpTransport::send(const std::string& json_message) {
    http::Headers h = headers_;
    h["Content-Type"] = "application/json";
    h["Accept"] = "application/json, text/event-stream";
    if (!api_key_.empty()) {
        h["Authorization"] = "Bearer " + api_key_;
    }

    auto body_value = ai::json::parse(json_message);

    auto task = client_.post_json(url_, body_value, h, {});
    ioc_.restart();
    task.start();
    while (!task.done()) {
        ioc_.run_one();
    }
    auto resp = task.get();

    if (resp.body.empty()) return;  // e.g. 202 for a notification

    // Determine framing from the response content type.
    std::string content_type;
    if (auto it = resp.headers.find("content-type"); it != resp.headers.end()) {
        content_type = it->second;
    }

    if (content_type.find("text/event-stream") != std::string::npos) {
        // SSE: queue each `data:` event so notifications + final response are
        // each returned by a separate receive().
        std::istringstream ss(resp.body);
        std::string line;
        std::string data;
        while (std::getline(ss, line)) {
            while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) line.pop_back();
            if (line.starts_with("data:")) {
                data = line.substr(5);
                if (!data.empty() && data.front() == ' ') data.erase(0, 1);
                auto parsed = ai::json::safe_parse(data);
                if (parsed) incoming_.push(boost::json::serialize(*parsed));
            }
        }
    } else {
        // Single JSON object.
        auto parsed = ai::json::safe_parse(resp.body);
        if (parsed) incoming_.push(boost::json::serialize(*parsed));
    }
}

std::string StreamableHttpTransport::receive() {
    if (incoming_.empty()) return {};
    auto out = incoming_.front();
    incoming_.pop();
    return out;
}

// ---------------------------------------------------------------------------
// InMemoryTransport
// ---------------------------------------------------------------------------

void InMemoryTransport::send(const std::string& json_message) {
    sent.push_back(json_message);
    if (handler) {
        if (auto resp = handler(json_message)) {
            incoming_.push(*resp);
        }
    }
}

std::string InMemoryTransport::receive() {
    if (incoming_.empty()) return {};
    auto out = incoming_.front();
    incoming_.pop();
    return out;
}

void InMemoryTransport::inject(const std::string& json_message) {
    incoming_.push(json_message);
}

} // namespace ai::mcp
