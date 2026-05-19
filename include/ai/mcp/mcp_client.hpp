#pragma once

#include <ai/schema/json_schema.hpp>
#include <ai/tool/tool_set.hpp>
#include <ai/stream/async_generator.hpp>
#include <boost/json.hpp>
#include <memory>
#include <string>
#include <vector>

namespace ai::mcp {

/// Represents a tool exposed by an MCP server.
struct McpTool {
    std::string name;
    std::string description;
    schema::JsonSchema input_schema;
};

/// Configuration for connecting to an MCP server via stdio.
struct McpServerConfig {
    std::string name;
    std::string transport;  // "stdio" or "sse"
    std::string command;    // for stdio transport: the command to spawn
    std::vector<std::string> args;  // for stdio transport: command arguments
    std::string url;        // for sse transport: the server URL
    std::unordered_map<std::string, std::string> env;  // environment variables for subprocess
};

/// Configuration for connecting to an MCP server via SSE.
struct McpSseConfig {
    std::string url;  // SSE endpoint URL (e.g., "http://localhost:3000/sse")
    std::unordered_map<std::string, std::string> headers;
};

/// Client for communicating with an MCP (Model Context Protocol) server
/// using JSON-RPC 2.0 over stdio transport.
class McpClient {
public:
    explicit McpClient(McpServerConfig config);
    ~McpClient();

    // Non-copyable
    McpClient(const McpClient&) = delete;
    McpClient& operator=(const McpClient&) = delete;

    // Movable
    McpClient(McpClient&& other) noexcept;
    McpClient& operator=(McpClient&& other) noexcept;

    /// Connect to the MCP server (spawns subprocess, performs initialize handshake).
    Task<void> connect();

    /// Disconnect from the MCP server (terminates subprocess).
    void disconnect();

    /// List tools available on the MCP server.
    Task<std::vector<McpTool>> list_tools();

    /// Call a tool on the MCP server.
    Task<boost::json::value> call_tool(std::string name, boost::json::value input);

    /// Returns true if connected.
    bool is_connected() const;

    /// Get the server name.
    const std::string& server_name() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/// Convert MCP tools to an AI SDK ToolSet.
/// Each tool's execute function will call the MCP client to execute the tool.
ai::ToolSet mcp_tools_to_toolset(std::shared_ptr<McpClient> client, const std::vector<McpTool>& tools);

} // namespace ai::mcp
