#pragma once

#include <ai/http/client.hpp>

#include <boost/asio.hpp>

#include <functional>
#include <memory>
#include <optional>
#include <queue>
#include <string>
#include <unordered_map>
#include <vector>

namespace ai::mcp {

/// Abstract JSON-RPC transport. McpClient is synchronous/blocking, so send()
/// writes a message and receive() blocks until the next complete inbound message
/// is available. Each implementation owns its framing.
class Transport {
public:
    virtual ~Transport() = default;
    virtual void start() = 0;   // open the connection / spawn the process
    virtual void stop() = 0;    // tear down
    virtual void send(const std::string& json_message) = 0;
    virtual std::string receive() = 0;  // blocking; "" if none/closed
};

/// stdio transport: spawns a subprocess and frames messages with the
/// Content-Length header (like LSP). Cross-platform (POSIX fork/exec + Windows
/// CreateProcess).
class StdioTransport : public Transport {
public:
    StdioTransport(std::string command, std::vector<std::string> args,
                   std::unordered_map<std::string, std::string> env = {});
    ~StdioTransport() override;
    StdioTransport(const StdioTransport&) = delete;
    StdioTransport& operator=(const StdioTransport&) = delete;

    void start() override;
    void stop() override;
    void send(const std::string& json_message) override;
    std::string receive() override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/// Streamable HTTP transport (MCP 2025-03 spec): each JSON-RPC message is POSTed
/// to a single endpoint. The response may be a JSON object or an SSE stream;
/// SSE events are queued so notifications (e.g. progress) and the final response
/// are each returned by separate receive() calls. Blocking: owns its io_context.
class StreamableHttpTransport : public Transport {
public:
    StreamableHttpTransport(std::string url,
                            std::unordered_map<std::string, std::string> headers = {},
                            std::string api_key = {});
    void start() override;  // connectionless: no-op
    void stop() override;
    void send(const std::string& json_message) override;  // POST + queue responses
    std::string receive() override;

private:
    std::string url_;
    std::unordered_map<std::string, std::string> headers_;
    std::string api_key_;
    boost::asio::io_context ioc_;
    http::HttpClient client_;
    std::queue<std::string> incoming_;
    std::string session_id_;  // streamable-HTTP session, echoed on requests
};

/// In-memory transport for tests: send() records the message and, if a handler
/// is set, queues any response it returns; receive() pops the queue. Lets the
/// MCP client be exercised end to end with no subprocess or network.
class InMemoryTransport : public Transport {
public:
    /// Inspects an outbound message and returns a response to queue (nullopt for
    /// notifications / no response).
    std::function<std::optional<std::string>(const std::string&)> handler;

    /// Every message passed to send(), in order.
    std::vector<std::string> sent;

    void start() override {}
    void stop() override {}
    void send(const std::string& json_message) override;
    std::string receive() override;

    /// Push a message as if the server sent it (for tests to simulate
    /// server-initiated requests / notifications mid-conversation).
    void inject(const std::string& json_message);

private:
    std::queue<std::string> incoming_;
};

} // namespace ai::mcp
