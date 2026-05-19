#pragma once

#include <ai/stream/async_generator.hpp>
#include <ai/util/cancellation.hpp>
#include <boost/asio.hpp>
#include <boost/asio/ssl.hpp>
#include <boost/beast.hpp>
#include <boost/beast/websocket.hpp>
#include <boost/beast/websocket/ssl.hpp>
#include <boost/json.hpp>
#include <string>
#include <string_view>
#include <unordered_map>
#include <memory>
#include <functional>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>

namespace ai::http {

using WsHeaders = std::unordered_map<std::string, std::string>;

struct WebSocketConfig {
    std::chrono::seconds connect_timeout{30};
    std::chrono::seconds ping_interval{30};
    size_t max_message_size = 16 * 1024 * 1024;
};

class WebSocketClient {
public:
    using MessageHandler = std::function<void(const std::string&)>;
    using CloseHandler = std::function<void(int code, std::string_view reason)>;
    using ErrorHandler = std::function<void(std::exception_ptr)>;

    WebSocketClient(boost::asio::io_context& ioc, WebSocketConfig config = {});
    ~WebSocketClient();

    Task<void> connect(
        std::string_view url,
        WsHeaders headers = {},
        CancellationToken cancel = {}
    );

    Task<void> send(const boost::json::value& message);
    Task<void> send_text(std::string_view text);

    AsyncGenerator<std::string> messages();

    Task<void> close(int code = 1000, std::string_view reason = "");

    bool is_connected() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ai::http
