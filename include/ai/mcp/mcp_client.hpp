#pragma once

#include <ai/mcp/transport.hpp>
#include <ai/schema/json_schema.hpp>
#include <ai/tool/tool_set.hpp>
#include <ai/stream/async_generator.hpp>
#include <boost/json.hpp>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace ai::mcp {

/// A tool exposed by an MCP server.
struct McpTool {
    std::string name;
    std::string description;
    schema::JsonSchema input_schema;
};

/// Configuration for an MCP server connection.
struct McpServerConfig {
    std::string name;
    std::string transport;  // "stdio" | "http" | "streamable-http"
    // stdio:
    std::string command;
    std::vector<std::string> args;
    std::unordered_map<std::string, std::string> env;
    // http / streamable-http:
    std::string url;
    std::unordered_map<std::string, std::string> headers;
    std::string api_key;
};

/// A resource exposed by an MCP server.
struct McpResource {
    std::string uri;
    std::string name;
    std::string description;
    std::string mime_type;
};

/// Contents of a read resource (text or base64 blob).
struct McpResourceContent {
    std::string uri;
    std::string mime_type;
    std::string text;
    std::optional<std::string> blob_b64;
};

/// A prompt template exposed by an MCP server.
struct McpPromptArgument {
    std::string name;
    std::string description;
    bool required = false;
};
struct McpPrompt {
    std::string name;
    std::string description;
    std::vector<McpPromptArgument> arguments;
};

/// One message within a resolved prompt (text content only, simplified).
struct McpPromptMessage {
    std::string role;
    std::string text;
};

/// Handler for server-initiated LLM sampling (sampling/createMessage). Returns
/// the createMessage result object: { role, content: { type, text }, model }.
/// Called synchronously from inside the request loop.
using SamplingHandler =
    std::function<boost::json::value(const boost::json::value& params)>;

/// Handler for progress notifications (notifications/progress).
using ProgressHandler = std::function<void(
    const boost::json::value& progress_token,
    double progress,
    std::optional<double> total,
    const std::string& message
)>;

/// Client for an MCP (Model Context Protocol) server using JSON-RPC 2.0 over a
/// pluggable Transport (stdio, streamable HTTP, or an injected test transport).
class McpClient {
public:
    /// Build a client from a config; the transport is selected by
    /// `config.transport` ("stdio" | "http" | "streamable-http").
    explicit McpClient(McpServerConfig config);
    /// Build a client over an injected transport (tests / custom transports).
    explicit McpClient(std::unique_ptr<Transport> transport, std::string name = {});
    ~McpClient();

    McpClient(const McpClient&) = delete;
    McpClient& operator=(const McpClient&) = delete;
    McpClient(McpClient&&) noexcept;
    McpClient& operator=(McpClient&&) noexcept;

    /// Connect (start transport, perform initialize handshake).
    Task<void> connect();
    void disconnect();

    // Tools
    Task<std::vector<McpTool>> list_tools();
    Task<boost::json::value> call_tool(std::string name, boost::json::value input);

    // Resources
    Task<std::vector<McpResource>> list_resources();
    Task<std::vector<McpResourceContent>> read_resource(std::string uri);

    // Prompts
    Task<std::vector<McpPrompt>> list_prompts();
    Task<std::vector<McpPromptMessage>> get_prompt(std::string name,
                                                   boost::json::object arguments = {});

    // Server-initiated sampling + progress
    void set_sampling_handler(SamplingHandler handler);
    void set_progress_handler(ProgressHandler handler);

    bool is_connected() const;
    const std::string& server_name() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/// Convert MCP tools to an AI SDK ToolSet whose execute functions call the client.
ai::ToolSet mcp_tools_to_toolset(std::shared_ptr<McpClient> client,
                                 const std::vector<McpTool>& tools);

} // namespace ai::mcp
