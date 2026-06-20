#include <catch2/catch_test_macros.hpp>

#include <ai/mcp/mcp_client.hpp>
#include <ai/mcp/transport.hpp>

#include <boost/asio.hpp>
#include <boost/json.hpp>

#include <functional>
#include <memory>
#include <optional>
#include <string>

namespace json = boost::json;

namespace {

template <typename T>
T run(ai::Task<T> task, boost::asio::io_context& ioc) {
    task.start();
    int guard = 0;
    while (!task.done() && guard++ < 100'000) {
        ioc.run_one();
    }
    REQUIRE(task.done());
    return task.get();
}

void run_void(ai::Task<void> task, boost::asio::io_context& ioc) {
    task.start();
    int guard = 0;
    while (!task.done() && guard++ < 100'000) {
        ioc.run_one();
    }
    REQUIRE(task.done());
    task.get();
}

std::string method_of(const std::string& msg) {
    auto v = json::parse(msg);
    if (!v.is_object()) return {};
    auto& o = v.as_object();
    auto m = o.find("method");
    return (m != o.end() && m->value().is_string()) ? std::string(m->value().as_string()) : "";
}

json::value id_of(const std::string& msg) {
    auto v = json::parse(msg);
    if (!v.is_object()) return {};
    auto& o = v.as_object();
    auto i = o.find("id");
    return (i != o.end()) ? i->value() : json::value();
}

// Canned responses for the standard RPCs.
std::optional<std::string> standard_handler(const std::string& out) {
    auto id = id_of(out);
    if (id.is_null()) return std::nullopt;  // notification -> no response
    auto method = method_of(out);

    json::object resp{{"jsonrpc", "2.0"}, {"id", id}};
    json::object result;

    if (method == "initialize") {
        result["protocolVersion"] = "2024-11-05";
        result["capabilities"] = json::object{};
        result["serverInfo"] = json::object{{"name", "test-server"}, {"version", "1"}};
    } else if (method == "tools/list") {
        json::object tool;
        tool["name"] = "echo";
        tool["description"] = "echoes the input";
        tool["inputSchema"] = json::object{{"type", "object"}, {"properties", json::object{}}};
        result["tools"] = json::array{tool};
    } else if (method == "tools/call") {
        json::object part;
        part["type"] = "text";
        part["text"] = "echoed: hello";
        result["content"] = json::array{part};
    } else if (method == "resources/list") {
        json::object res;
        res["uri"] = "file:///x";
        res["name"] = "X";
        res["mimeType"] = "text/plain";
        result["resources"] = json::array{res};
    } else if (method == "resources/read") {
        json::object c;
        c["uri"] = "file:///x";
        c["text"] = "resource body";
        result["contents"] = json::array{c};
    } else if (method == "prompts/list") {
        json::object arg;
        arg["name"] = "who";
        arg["required"] = true;
        json::object prompt;
        prompt["name"] = "greet";
        prompt["description"] = "g";
        prompt["arguments"] = json::array{arg};
        result["prompts"] = json::array{prompt};
    } else if (method == "prompts/get") {
        json::object content;
        content["type"] = "text";
        content["text"] = "hi";
        json::object msg;
        msg["role"] = "user";
        msg["content"] = content;
        result["messages"] = json::array{msg};
    } else {
        resp["error"] = json::object{{"code", -32601}, {"message", "unknown method"}};
        return json::serialize(resp);
    }

    resp["result"] = std::move(result);
    return json::serialize(resp);
}

std::shared_ptr<ai::mcp::McpClient> make_client(
    std::function<std::optional<std::string>(const std::string&)> handler) {
    auto transport = std::make_unique<ai::mcp::InMemoryTransport>();
    transport->handler = std::move(handler);
    return std::make_shared<ai::mcp::McpClient>(std::move(transport), "test");
}

} // namespace

TEST_CASE("MCP client connects, lists tools, calls a tool", "[mcp]") {
    boost::asio::io_context ioc;
    auto client = make_client(standard_handler);
    run_void(client->connect(), ioc);
    REQUIRE(client->is_connected());

    auto tools = run(client->list_tools(), ioc);
    REQUIRE(tools.size() == 1);
    REQUIRE(tools[0].name == "echo");

    auto result = run(client->call_tool("echo", json::object{{"msg", "hello"}}), ioc);
    REQUIRE(result.is_string());
    REQUIRE(result.as_string() == "echoed: hello");
}

TEST_CASE("MCP resources list and read", "[mcp]") {
    boost::asio::io_context ioc;
    auto client = make_client(standard_handler);
    run_void(client->connect(), ioc);

    auto resources = run(client->list_resources(), ioc);
    REQUIRE(resources.size() == 1);
    REQUIRE(resources[0].uri == "file:///x");
    REQUIRE(resources[0].mime_type == "text/plain");

    auto content = run(client->read_resource("file:///x"), ioc);
    REQUIRE(content.size() == 1);
    REQUIRE(content[0].text == "resource body");
}

TEST_CASE("MCP prompts list and get", "[mcp]") {
    boost::asio::io_context ioc;
    auto client = make_client(standard_handler);
    run_void(client->connect(), ioc);

    auto prompts = run(client->list_prompts(), ioc);
    REQUIRE(prompts.size() == 1);
    REQUIRE(prompts[0].name == "greet");
    REQUIRE(prompts[0].arguments.size() == 1);
    REQUIRE(prompts[0].arguments[0].required);

    auto messages = run(client->get_prompt("greet", {{"who", "world"}}), ioc);
    REQUIRE(messages.size() == 1);
    REQUIRE(messages[0].role == "user");
    REQUIRE(messages[0].text == "hi");
}

TEST_CASE("MCP responds to server-initiated sampling", "[mcp]") {
    boost::asio::io_context ioc;
    auto transport = std::make_unique<ai::mcp::InMemoryTransport>();
    auto* tp = transport.get();

    transport->handler = [tp](const std::string& out) -> std::optional<std::string> {
        auto id = id_of(out);
        if (id.is_null()) return std::nullopt;
        auto method = method_of(out);
        json::object resp{{"jsonrpc", "2.0"}, {"id", id}};
        if (method == "initialize") {
            resp["result"] = json::object{{"capabilities", json::object{}},
                                          {"serverInfo", json::object{{"name", "s"}, {"version", "1"}}}};
        } else if (method == "tools/call") {
            // Server asks the client to sample, then answers the tool call.
            json::object sreq{{"jsonrpc", "2.0"}, {"id", 999},
                              {"method", "sampling/createMessage"},
                              {"params", json::object{{"messages", json::array{}}}}};
            tp->inject(json::serialize(sreq));
            resp["result"] = json::object{{"content", json::array{
                json::object{{"type", "text"}, {"text", "done"}}}}};
        } else {
            resp["error"] = json::object{{"code", -32601}, {"message", "no"}};
        }
        return json::serialize(resp);
    };

    bool sampling_called = false;
    auto client = std::make_shared<ai::mcp::McpClient>(std::move(transport), "test");
    client->set_sampling_handler([&](const json::value& /*params*/) -> json::value {
        sampling_called = true;
        return json::object{{"role", "assistant"},
                            {"content", json::object{{"type", "text"}, {"text", "sampled"}}},
                            {"model", "test-model"}};
    });

    run_void(client->connect(), ioc);
    auto result = run(client->call_tool("x", json::object{}), ioc);

    REQUIRE(sampling_called);
    REQUIRE(result.is_string());
    REQUIRE(result.as_string() == "done");
}

TEST_CASE("MCP progress notifications reach the handler", "[mcp]") {
    boost::asio::io_context ioc;
    auto transport = std::make_unique<ai::mcp::InMemoryTransport>();
    auto* tp = transport.get();

    transport->handler = [tp](const std::string& out) -> std::optional<std::string> {
        auto id = id_of(out);
        if (id.is_null()) return std::nullopt;
        auto method = method_of(out);
        json::object resp{{"jsonrpc", "2.0"}, {"id", id}};
        if (method == "initialize") {
            resp["result"] = json::object{{"capabilities", json::object{}},
                                          {"serverInfo", json::object{{"name", "s"}, {"version", "1"}}}};
        } else if (method == "tools/call") {
            json::object prog{{"jsonrpc", "2.0"}, {"method", "notifications/progress"},
                              {"params", json::object{{"progressToken", 1},
                                                      {"progress", 50}, {"total", 100},
                                                      {"message", "working"}}}};
            tp->inject(json::serialize(prog));
            resp["result"] = json::object{{"content", json::array{
                json::object{{"type", "text"}, {"text", "ok"}}}}};
        } else {
            resp["error"] = json::object{{"code", -32601}, {"message", "no"}};
        }
        return json::serialize(resp);
    };

    bool progress_called = false;
    double reported = 0;
    std::optional<double> total;
    std::string message;

    auto client = std::make_shared<ai::mcp::McpClient>(std::move(transport), "test");
    client->set_progress_handler([&](const json::value&, double p, std::optional<double> t,
                                     const std::string& msg) {
        progress_called = true;
        reported = p;
        total = t;
        message = msg;
    });

    run_void(client->connect(), ioc);
    run(client->call_tool("x", json::object{}), ioc);

    REQUIRE(progress_called);
    REQUIRE(reported == 50.0);
    REQUIRE(total.has_value());
    REQUIRE(*total == 100.0);
    REQUIRE(message == "working");
}
