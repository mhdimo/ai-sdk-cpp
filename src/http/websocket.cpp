#include <ai/http/websocket.hpp>
#include <ai/http/client.hpp>
#include <ai/error/api_call_error.hpp>

namespace ai::http {

namespace beast = boost::beast;
namespace websocket = beast::websocket;
namespace net = boost::asio;
namespace ssl = net::ssl;
using tcp = net::ip::tcp;

using SslWebSocket = websocket::stream<ssl::stream<beast::tcp_stream>>;
using PlainWebSocket = websocket::stream<beast::tcp_stream>;

struct MessageChannel {
    std::mutex mutex;
    std::condition_variable cv;
    std::queue<std::string> messages;
    bool closed = false;
    std::exception_ptr error;

    void push(std::string msg) {
        std::lock_guard lock(mutex);
        messages.push(std::move(msg));
        cv.notify_one();
    }

    void close_channel() {
        std::lock_guard lock(mutex);
        closed = true;
        cv.notify_all();
    }

    void set_error(std::exception_ptr err) {
        std::lock_guard lock(mutex);
        error = err;
        closed = true;
        cv.notify_all();
    }

    std::optional<std::string> pop() {
        std::unique_lock lock(mutex);
        cv.wait(lock, [this] { return !messages.empty() || closed; });
        if (error) std::rethrow_exception(error);
        if (messages.empty()) return std::nullopt;
        auto msg = std::move(messages.front());
        messages.pop();
        return msg;
    }
};

struct WebSocketClient::Impl {
    net::io_context& ioc;
    WebSocketConfig config;
    ssl::context ssl_ctx{ssl::context::tlsv12_client};
    std::unique_ptr<SslWebSocket> ws;
    std::shared_ptr<MessageChannel> channel;
    std::thread read_thread;
    std::atomic<bool> connected{false};

    Impl(net::io_context& io, WebSocketConfig cfg)
        : ioc(io), config(cfg) {
        ssl_ctx.set_default_verify_paths();
        ssl_ctx.set_verify_mode(ssl::verify_peer);
    }

    ~Impl() {
        connected = false;
        if (channel) channel->close_channel();
        if (read_thread.joinable()) read_thread.join();
    }

    void do_connect(std::string_view url_str, const WsHeaders& headers) {
        auto url = Url::parse(url_str);
        std::string host = url.host;
        std::string port = url.port;
        std::string target = url.path;

        tcp::resolver resolver(ioc);
        auto results = resolver.resolve(host, port);

        ws = std::make_unique<SslWebSocket>(ioc, ssl_ctx);

        if (!SSL_set_tlsext_host_name(
                ws->next_layer().native_handle(), host.c_str())) {
            throw std::runtime_error("Failed to set SNI hostname");
        }

        auto& tcp_layer = beast::get_lowest_layer(*ws);
        tcp_layer.expires_after(config.connect_timeout);
        tcp_layer.connect(results);

        ws->next_layer().handshake(ssl::stream_base::client);

        tcp_layer.expires_never();
        ws->set_option(websocket::stream_base::timeout::suggested(
            beast::role_type::client));
        ws->set_option(websocket::stream_base::decorator(
            [&headers, &host](websocket::request_type& req) {
                for (auto& [k, v] : headers) {
                    req.set(k, v);
                }
                req.set(beast::http::field::host, host);
            }
        ));

        ws->handshake(host, target);
        ws->text(true);
        connected = true;

        channel = std::make_shared<MessageChannel>();
        auto ch = channel;
        auto ws_ptr = ws.get();

        read_thread = std::thread([ch, ws_ptr, this]() {
            beast::flat_buffer buffer;
            while (connected) {
                try {
                    buffer.clear();
                    ws_ptr->read(buffer);
                    std::string msg = beast::buffers_to_string(buffer.data());
                    ch->push(std::move(msg));
                } catch (const beast::system_error& se) {
                    if (se.code() == websocket::error::closed) {
                        break;
                    }
                    ch->set_error(std::current_exception());
                    break;
                } catch (...) {
                    ch->set_error(std::current_exception());
                    break;
                }
            }
            ch->close_channel();
        });
    }

    void do_send(std::string_view text) {
        if (!connected || !ws) {
            throw error::StreamError("WebSocket not connected");
        }
        ws->write(net::buffer(text.data(), text.size()));
    }

    void do_close(int code, std::string_view reason) {
        if (!connected || !ws) return;
        connected = false;
        try {
            ws->close(websocket::close_reason(
                static_cast<websocket::close_code>(code),
                std::string(reason)));
        } catch (...) {}
    }
};

WebSocketClient::WebSocketClient(net::io_context& ioc, WebSocketConfig config)
    : impl_(std::make_unique<Impl>(ioc, config)) {}

WebSocketClient::~WebSocketClient() = default;

Task<void> WebSocketClient::connect(
    std::string_view url,
    WsHeaders headers,
    CancellationToken cancel
) {
    cancel.throw_if_cancelled();
    impl_->do_connect(url, headers);
    co_return;
}

Task<void> WebSocketClient::send(const boost::json::value& message) {
    auto text = boost::json::serialize(message);
    impl_->do_send(text);
    co_return;
}

Task<void> WebSocketClient::send_text(std::string_view text) {
    impl_->do_send(text);
    co_return;
}

AsyncGenerator<std::string> WebSocketClient::messages() {
    auto channel = impl_->channel;
    if (!channel) co_return;

    while (true) {
        auto msg = channel->pop();
        if (!msg) break;
        co_yield std::move(*msg);
    }
}

Task<void> WebSocketClient::close(int code, std::string_view reason) {
    impl_->do_close(code, reason);
    co_return;
}

bool WebSocketClient::is_connected() const noexcept {
    return impl_->connected;
}

} // namespace ai::http
