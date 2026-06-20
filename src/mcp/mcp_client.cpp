#include <ai/mcp/mcp_client.hpp>
#include <ai/util/json.hpp>
#include <boost/json.hpp>

#include <atomic>
#include <mutex>
#include <stdexcept>
#include <string>

namespace ai::mcp {

namespace json = boost::json;

namespace {

std::string str_field(const json::object& obj, const char* key) {
    auto it = obj.find(key);
    if (it != obj.end() && it->value().is_string()) {
        return std::string(it->value().as_string());
    }
    return {};
}

// boost::json stores integer literals as int64; as_double() throws on those.
// Convert any JSON number to double.
double to_double(const json::value& v) {
    if (v.is_double()) return v.get_double();
    if (v.is_int64()) return static_cast<double>(v.get_int64());
    if (v.is_uint64()) return static_cast<double>(v.get_uint64());
    return 0.0;
}

} // namespace

struct McpClient::Impl {
    std::unique_ptr<Transport> transport;
    std::atomic<bool> connected{false};
    std::atomic<int> next_id{1};
    std::mutex io_mutex;
    std::string name;
    SamplingHandler sampling_handler;
    ProgressHandler progress_handler;

    void respond(const json::value& id, json::value result_or_error, bool is_error) {
        json::object response;
        response["jsonrpc"] = "2.0";
        response["id"] = id;
        if (is_error) response["error"] = std::move(result_or_error);
        else response["result"] = std::move(result_or_error);
        transport->send(json::serialize(response));
    }

    void handle_server_request(const json::object& req) {
        json::value id = req.contains("id") ? req.at("id") : json::value();
        std::string method = str_field(req, "method");

        if (method == "sampling/createMessage") {
            if (!sampling_handler) {
                json::object err{{"code", -32601}, {"message", "sampling not supported"}};
                respond(id, err, true);
                return;
            }
            json::value params = req.contains("params") ? req.at("params") : json::value();
            respond(id, sampling_handler(params), false);
            return;
        }
        json::object err{{"code", -32601}, {"message", "method not implemented: " + method}};
        respond(id, err, true);
    }

    void handle_notification(const json::object& note) {
        std::string method = str_field(note, "method");
        if (method == "notifications/progress" && progress_handler) {
            json::value token = nullptr;
            double progress = 0.0;
            std::optional<double> total;
            std::string message;
            if (note.contains("params") && note.at("params").is_object()) {
                auto& p = note.at("params").as_object();
                if (auto t = p.find("progressToken"); t != p.end()) token = t->value();
                if (auto pr = p.find("progress"); pr != p.end() && pr->value().is_number()) {
                    progress = to_double(pr->value());
                }
                if (auto to = p.find("total"); to != p.end() && to->value().is_number()) {
                    total = to_double(to->value());
                }
                message = str_field(p, "message");
            }
            progress_handler(token, progress, total, message);
        }
    }

    json::value send_request(const std::string& method, json::value params = json::value()) {
        if (!connected) throw std::runtime_error("MCP: not connected to server");

        int id = next_id.fetch_add(1);
        json::object request;
        request["jsonrpc"] = "2.0";
        request["id"] = id;
        request["method"] = method;
        if (!params.is_null()) request["params"] = params;

        std::string body = json::serialize(request);

        std::lock_guard lock(io_mutex);
        transport->send(body);

        while (true) {
            std::string resp_body = transport->receive();
            if (resp_body.empty()) {
                throw std::runtime_error("MCP: no response from server");
            }
            auto resp = ai::json::parse(resp_body);
            if (!resp.is_object()) throw std::runtime_error("MCP: invalid JSON-RPC response");
            auto& obj = resp.as_object();

            auto id_it = obj.find("id");
            bool has_id = id_it != obj.end() && !id_it->value().is_null();
            bool is_request = obj.contains("method");

            // Server -> client request (e.g. sampling/createMessage).
            if (has_id && is_request) {
                handle_server_request(obj);
                continue;
            }
            // Notification (e.g. progress).
            if (!has_id) {
                handle_notification(obj);
                continue;
            }
            // Response: match our id.
            if (id_it->value().is_int64() && id_it->value().as_int64() == id) {
                if (auto err = obj.find("error"); err != obj.end() && !err->value().is_null()) {
                    std::string msg = "MCP: server error";
                    if (err->value().is_object()) {
                        msg = "MCP: " + str_field(err->value().as_object(), "message");
                    }
                    throw std::runtime_error(msg);
                }
                if (auto r = obj.find("result"); r != obj.end()) return r->value();
                return json::value();
            }
            // Unrelated response: ignore and keep reading.
        }
    }

    void send_notification(const std::string& method, json::value params = json::value()) {
        if (!connected) return;
        json::object note;
        note["jsonrpc"] = "2.0";
        note["method"] = method;
        if (!params.is_null()) note["params"] = params;
        std::lock_guard lock(io_mutex);
        transport->send(json::serialize(note));
    }
};

// --- construction ---------------------------------------------------------

McpClient::McpClient(McpServerConfig config)
    : impl_(std::make_unique<Impl>()) {
    impl_->name = std::move(config.name);

    std::unique_ptr<Transport> t;
    if (config.transport == "http" || config.transport == "streamable-http") {
        t = std::make_unique<StreamableHttpTransport>(
            config.url, config.headers, config.api_key);
    } else if (config.transport == "stdio" || !config.command.empty()) {
        t = std::make_unique<StdioTransport>(config.command, config.args, config.env);
    } else {
        throw std::runtime_error("MCP: unsupported transport '" + config.transport + "'");
    }
    impl_->transport = std::move(t);
}

McpClient::McpClient(std::unique_ptr<Transport> transport, std::string name)
    : impl_(std::make_unique<Impl>()) {
    impl_->transport = std::move(transport);
    impl_->name = std::move(name);
}

McpClient::~McpClient() { disconnect(); }
McpClient::McpClient(McpClient&&) noexcept = default;
McpClient& McpClient::operator=(McpClient&&) noexcept = default;

// --- lifecycle -----------------------------------------------------------

Task<void> McpClient::connect() {
    impl_->transport->start();
    impl_->connected = true;

    json::object init_params;
    init_params["protocolVersion"] = "2024-11-05";
    init_params["clientInfo"] = json::object{{"name", "ai-sdk-cpp"}, {"version", "0.1.0"}};
    init_params["capabilities"] = json::object{};
    impl_->send_request("initialize", json::value(init_params));
    impl_->send_notification("notifications/initialized");
    co_return;
}

void McpClient::disconnect() {
    if (impl_ && impl_->connected) {
        impl_->connected = false;
        impl_->transport->stop();
    }
}

bool McpClient::is_connected() const { return impl_ && impl_->connected; }
const std::string& McpClient::server_name() const { return impl_->name; }

void McpClient::set_sampling_handler(SamplingHandler h) { impl_->sampling_handler = std::move(h); }
void McpClient::set_progress_handler(ProgressHandler h) { impl_->progress_handler = std::move(h); }

// --- tools ---------------------------------------------------------------

Task<std::vector<McpTool>> McpClient::list_tools() {
    auto result = impl_->send_request("tools/list");
    std::vector<McpTool> tools;
    if (!result.is_object()) co_return tools;

    auto tools_it = result.as_object().find("tools");
    if (tools_it == result.as_object().end() || !tools_it->value().is_array()) co_return tools;

    for (auto& tv : tools_it->value().as_array()) {
        if (!tv.is_object()) continue;
        auto& to = tv.as_object();
        McpTool tool;
        tool.name = str_field(to, "name");
        tool.description = str_field(to, "description");
        if (auto s = to.find("inputSchema"); s != to.end() && s->value().is_object()) {
            tool.input_schema = schema::JsonSchema(json::object(s->value().as_object()));
        }
        tools.push_back(std::move(tool));
    }
    co_return tools;
}

Task<json::value> McpClient::call_tool(std::string name, json::value input) {
    json::object params;
    params["name"] = name;
    params["arguments"] = input.is_object() ? input : json::object{};
    // Route progress notifications for this call back to the progress handler.
    params["_meta"] = json::object{{"progressToken", impl_->next_id.fetch_add(1)}};

    auto result = impl_->send_request("tools/call", json::value(params));
    if (!result.is_object()) co_return result;

    auto& obj = result.as_object();
    auto content_it = obj.find("content");
    if (content_it == obj.end() || !content_it->value().is_array()) co_return result;

    std::string text_result;
    for (auto& part : content_it->value().as_array()) {
        if (!part.is_object()) continue;
        auto& po = part.as_object();
        if (str_field(po, "type") == "text") {
            if (!text_result.empty()) text_result += "\n";
            text_result += str_field(po, "text");
        }
    }

    if (!text_result.empty()) {
        if (auto parsed = ai::json::safe_parse(text_result)) co_return *parsed;
        co_return json::value(text_result);
    }
    co_return result;
}

// --- resources -----------------------------------------------------------

Task<std::vector<McpResource>> McpClient::list_resources() {
    auto result = impl_->send_request("resources/list");
    std::vector<McpResource> out;
    if (!result.is_object()) co_return out;
    auto it = result.as_object().find("resources");
    if (it == result.as_object().end() || !it->value().is_array()) co_return out;
    for (auto& rv : it->value().as_array()) {
        if (!rv.is_object()) continue;
        auto& ro = rv.as_object();
        McpResource res;
        res.uri = str_field(ro, "uri");
        res.name = str_field(ro, "name");
        res.description = str_field(ro, "description");
        res.mime_type = str_field(ro, "mimeType");
        out.push_back(std::move(res));
    }
    co_return out;
}

Task<std::vector<McpResourceContent>> McpClient::read_resource(std::string uri) {
    json::object params;
    params["uri"] = uri;
    auto result = impl_->send_request("resources/read", json::value(params));
    std::vector<McpResourceContent> out;
    if (!result.is_object()) co_return out;
    auto it = result.as_object().find("contents");
    if (it == result.as_object().end() || !it->value().is_array()) co_return out;
    for (auto& cv : it->value().as_array()) {
        if (!cv.is_object()) continue;
        auto& co = cv.as_object();
        McpResourceContent content;
        content.uri = str_field(co, "uri");
        content.mime_type = str_field(co, "mimeType");
        content.text = str_field(co, "text");
        if (auto b = co.find("blob"); b != co.end() && b->value().is_string()) {
            content.blob_b64 = std::string(b->value().as_string());
        }
        out.push_back(std::move(content));
    }
    co_return out;
}

// --- prompts -------------------------------------------------------------

Task<std::vector<McpPrompt>> McpClient::list_prompts() {
    auto result = impl_->send_request("prompts/list");
    std::vector<McpPrompt> out;
    if (!result.is_object()) co_return out;
    auto it = result.as_object().find("prompts");
    if (it == result.as_object().end() || !it->value().is_array()) co_return out;
    for (auto& pv : it->value().as_array()) {
        if (!pv.is_object()) continue;
        auto& po = pv.as_object();
        McpPrompt prompt;
        prompt.name = str_field(po, "name");
        prompt.description = str_field(po, "description");
        if (auto a = po.find("arguments"); a != po.end() && a->value().is_array()) {
            for (auto& av : a->value().as_array()) {
                if (!av.is_object()) continue;
                auto& ao = av.as_object();
                McpPromptArgument arg;
                arg.name = str_field(ao, "name");
                arg.description = str_field(ao, "description");
                if (auto r = ao.find("required"); r != ao.end() && r->value().is_bool()) {
                    arg.required = r->value().as_bool();
                }
                prompt.arguments.push_back(std::move(arg));
            }
        }
        out.push_back(std::move(prompt));
    }
    co_return out;
}

Task<std::vector<McpPromptMessage>> McpClient::get_prompt(
    std::string name, json::object arguments
) {
    json::object params;
    params["name"] = name;
    if (!arguments.empty()) params["arguments"] = arguments;

    auto result = impl_->send_request("prompts/get", json::value(params));
    std::vector<McpPromptMessage> out;
    if (!result.is_object()) co_return out;
    auto it = result.as_object().find("messages");
    if (it == result.as_object().end() || !it->value().is_array()) co_return out;
    for (auto& mv : it->value().as_array()) {
        if (!mv.is_object()) continue;
        auto& mo = mv.as_object();
        McpPromptMessage msg;
        msg.role = str_field(mo, "role");
        // content may be a string or {type:text,text:...}
        auto c = mo.find("content");
        if (c != mo.end()) {
            if (c->value().is_string()) {
                msg.text = std::string(c->value().as_string());
            } else if (c->value().is_object()) {
                msg.text = str_field(c->value().as_object(), "text");
            }
        }
        out.push_back(std::move(msg));
    }
    co_return out;
}

// --- toolset adapter -----------------------------------------------------

ai::ToolSet mcp_tools_to_toolset(std::shared_ptr<McpClient> client,
                                 const std::vector<McpTool>& tools) {
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
                boost::json::value input, ToolExecutionContext
            ) -> Task<boost::json::value> {
                co_return co_await captured_client->call_tool(tool_name, std::move(input));
            },
        };
        toolset.add(std::move(def));
    }
    return toolset;
}

} // namespace ai::mcp
