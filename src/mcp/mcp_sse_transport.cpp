#include <ai/mcp/mcp_client.hpp>
#include <ai/http/client.hpp>
#include <ai/stream/sse_parser.hpp>
#include <boost/json.hpp>
#include <stdexcept>
#include <mutex>
#include <condition_variable>
#include <queue>

namespace ai::mcp {

namespace json = boost::json;

class McpSseTransport {
public:
    McpSseTransport(const McpSseConfig& config, http::HttpClient& client)
        : config_(config), client_(client) {}

    Task<void> connect() {
        // GET the SSE endpoint to establish the event stream
        http::Headers headers = config_.headers;
        headers["Accept"] = "text/event-stream";

        // Start streaming connection
        auto response = co_await client_.post_streaming(
            config_.url,
            json::value(json::object{}),
            headers
        );

        body_stream_ = std::move(response.body_stream);
        connected_ = true;
    }

    Task<json::value> send_request(json::value request) {
        if (messages_endpoint_.empty()) {
            throw std::runtime_error("MCP SSE: not connected, no messages endpoint");
        }

        http::Headers headers = config_.headers;
        headers["Content-Type"] = "application/json";

        auto response = co_await client_.post_json(
            messages_endpoint_,
            request,
            headers
        );

        co_return json::parse(response.body);
    }

    Task<std::optional<json::value>> receive_event() {
        if (!connected_) co_return std::nullopt;

        auto chunk = co_await body_stream_.next();
        if (!chunk) co_return std::nullopt;

        // Feed to SSE parser
        std::string data(chunk->begin(), chunk->end());
        sse_buffer_ += data;

        // Try to extract a complete SSE event
        auto newline_pos = sse_buffer_.find("\n\n");
        if (newline_pos == std::string::npos) co_return std::nullopt;

        auto event_text = sse_buffer_.substr(0, newline_pos);
        sse_buffer_ = sse_buffer_.substr(newline_pos + 2);

        // Parse event
        std::string event_type;
        std::string event_data;
        std::istringstream stream(event_text);
        std::string line;
        while (std::getline(stream, line)) {
            if (line.starts_with("event: ")) {
                event_type = line.substr(7);
            } else if (line.starts_with("data: ")) {
                event_data = line.substr(6);
            }
        }

        if (event_type == "endpoint") {
            // The server tells us where to POST messages
            // Could be relative or absolute URL
            if (event_data.starts_with("http")) {
                messages_endpoint_ = event_data;
            } else {
                // Relative to base URL
                auto base = config_.url;
                auto slash = base.rfind('/');
                if (slash != std::string::npos) {
                    messages_endpoint_ = base.substr(0, slash) + event_data;
                } else {
                    messages_endpoint_ = event_data;
                }
            }
            co_return std::nullopt; // Not a message event
        }

        if (event_type == "message" && !event_data.empty()) {
            co_return json::parse(event_data);
        }

        co_return std::nullopt;
    }

    bool is_connected() const { return connected_; }
    const std::string& messages_endpoint() const { return messages_endpoint_; }

private:
    McpSseConfig config_;
    http::HttpClient& client_;
    AsyncGenerator<std::vector<uint8_t>> body_stream_;
    bool connected_ = false;
    std::string messages_endpoint_;
    std::string sse_buffer_;
};

// Implementation of SSE-based MCP client connection
Task<void> connect_mcp_sse(McpClient& client, const McpSseConfig& config, http::HttpClient& http_client) {
    // This would be called to initialize an SSE-based MCP connection
    // The McpClient would store the transport and use it for send/receive
    co_return;
}

} // namespace ai::mcp
