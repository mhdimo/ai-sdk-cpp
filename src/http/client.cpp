#include <ai/http/client.hpp>
#include <ai/http/connection_pool.hpp>
#include <ai/http/response.hpp>
#include <ai/error/api_call_error.hpp>
#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>
#include <boost/asio/connect.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/asio/ssl/stream.hpp>
#include <boost/asio/thread_pool.hpp>
#include <boost/asio/post.hpp>
#include <boost/asio/strand.hpp>
#include <boost/optional.hpp>
#include <openssl/ssl.h>
#include <future>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <thread>

namespace ai::http {

namespace beast = boost::beast;
namespace http_ns = beast::http;
namespace net = boost::asio;
namespace ssl = net::ssl;
using tcp = net::ip::tcp;

Url Url::parse(std::string_view url_str) {
    Url result;
    std::string url(url_str);

    auto scheme_end = url.find("://");
    if (scheme_end != std::string::npos) {
        result.scheme = url.substr(0, scheme_end);
        url = url.substr(scheme_end + 3);
    } else {
        result.scheme = "https";
    }

    auto path_start = url.find('/');
    std::string authority;
    if (path_start != std::string::npos) {
        authority = url.substr(0, path_start);
        result.path = url.substr(path_start);
    } else {
        authority = url;
        result.path = "/";
    }

    auto port_start = authority.find(':');
    if (port_start != std::string::npos) {
        result.host = authority.substr(0, port_start);
        result.port = authority.substr(port_start + 1);
    } else {
        result.host = authority;
        result.port = result.is_tls() ? "443" : "80";
    }

    return result;
}

// ============================================================================
// Thread-safe channel for streaming chunks from worker thread to coroutine.
// The worker thread pushes chunks; the coroutine generator pops them.
// ============================================================================

struct StreamChannel {
    std::mutex mutex;
    std::condition_variable cv;
    std::queue<std::vector<uint8_t>> chunks;
    bool finished = false;
    std::exception_ptr error;

    void push(std::vector<uint8_t> chunk) {
        std::lock_guard lock(mutex);
        chunks.push(std::move(chunk));
        cv.notify_one();
    }

    void finish() {
        std::lock_guard lock(mutex);
        finished = true;
        cv.notify_one();
    }

    void set_error(std::exception_ptr err) {
        std::lock_guard lock(mutex);
        error = err;
        finished = true;
        cv.notify_one();
    }

    // Blocking pop - called from the generator coroutine context.
    // Returns nullopt when the stream is done.
    std::optional<std::vector<uint8_t>> pop() {
        std::unique_lock lock(mutex);
        cv.wait(lock, [this] { return !chunks.empty() || finished; });

        if (error) {
            std::rethrow_exception(error);
        }

        if (!chunks.empty()) {
            auto chunk = std::move(chunks.front());
            chunks.pop();
            return chunk;
        }

        return std::nullopt; // stream ended
    }
};

// ============================================================================
// Awaitable that bridges a std::future<T> into our Task<T> coroutine system.
// The coroutine suspends; a detached thread waits for the future and resumes
// the coroutine handle when the result is ready.
// ============================================================================

template <typename T>
struct FutureAwaitable {
    // We wrap the future in a shared_ptr so the awaitable is copyable
    // across the suspend/resume boundary while preserving move semantics
    // on the result.
    std::shared_ptr<std::future<T>> fut;

    explicit FutureAwaitable(std::future<T> f)
        : fut(std::make_shared<std::future<T>>(std::move(f))) {}

    bool await_ready() const noexcept {
        return fut->wait_for(std::chrono::seconds(0)) ==
               std::future_status::ready;
    }

    void await_suspend(std::coroutine_handle<> h) const {
        auto shared_fut = fut;
        std::thread([shared_fut, h]() {
            shared_fut->wait();
            h.resume();
        }).detach();
    }

    T await_resume() {
        return fut->get();
    }
};

// ============================================================================
// Implementation
// ============================================================================

struct HttpClient::Impl {
    net::io_context& ioc;
    HttpClientConfig config;
    ConnectionPool pool;
    net::thread_pool worker_pool;

    Impl(net::io_context& io, HttpClientConfig cfg)
        : ioc(io)
        , config(cfg)
        , pool(io, cfg.max_connections_per_host)
        , worker_pool(cfg.thread_pool_size) {}

    ~Impl() {
        worker_pool.join();
    }

    // Establish a connection (resolve + connect + TLS handshake).
    // Runs on a worker thread.
    void ensure_connected(ConnectionPool::Connection& conn) {
        if (conn.connected && conn.is_open()) {
            return; // reuse existing connection
        }

        tcp::resolver resolver(ioc);
        auto results = resolver.resolve(conn.host, conn.port);

        if (conn.is_tls) {
            auto& stream = *conn.ssl_stream;

            if (!SSL_set_tlsext_host_name(stream.native_handle(),
                                          conn.host.c_str())) {
                throw std::runtime_error("Failed to set SNI hostname");
            }

            auto& lowest = beast::get_lowest_layer(stream);
            lowest.expires_after(config.connect_timeout);
            lowest.connect(results);

            stream.handshake(ssl::stream_base::client);
            lowest.expires_after(config.read_timeout);
        } else {
            auto& stream = *conn.tcp_stream;
            stream.expires_after(config.connect_timeout);
            stream.connect(results);
            stream.expires_after(config.read_timeout);
        }

        conn.connected = true;
    }

    // Build an HTTP GET request.
    http_ns::request<http_ns::empty_body> build_get_request(
        const Url& url,
        const Headers& headers
    ) {
        http_ns::request<http_ns::empty_body> req{
            http_ns::verb::get, url.path, 11
        };
        req.set(http_ns::field::host, url.host);
        req.set(http_ns::field::connection, "keep-alive");
        for (auto& [key, value] : headers) {
            req.set(key, value);
        }
        req.prepare_payload();
        return req;
    }

    // Perform a full GET request on a worker thread.
    HttpResponse do_get_request(const Url& url,
                                const Headers& headers,
                                CancellationToken& cancel) {
        auto conn = pool.acquire(url.host, url.port, url.is_tls());

        try {
            ensure_connected(*conn);
            cancel.throw_if_cancelled();

            auto req = build_get_request(url, headers);

            beast::flat_buffer buffer;
            http_ns::response<http_ns::string_body> res;

            if (conn->is_tls) {
                http_ns::write(*conn->ssl_stream, req);
                cancel.throw_if_cancelled();
                http_ns::read(*conn->ssl_stream, buffer, res);
            } else {
                http_ns::write(*conn->tcp_stream, req);
                cancel.throw_if_cancelled();
                http_ns::read(*conn->tcp_stream, buffer, res);
            }

            HttpResponse response;
            response.status_code = static_cast<int>(res.result_int());
            for (auto& field : res) {
                response.headers[std::string(field.name_string())] =
                    std::string(field.value());
            }
            response.body = std::move(res.body());

            pool.release(std::move(conn));
            return response;
        } catch (...) {
            throw;
        }
    }

    // Build an HTTP POST request.
    http_ns::request<http_ns::string_body> build_request(
        const Url& url,
        const std::string& body_str,
        const std::string& content_type,
        const Headers& headers
    ) {
        http_ns::request<http_ns::string_body> req{
            http_ns::verb::post, url.path, 11
        };
        req.set(http_ns::field::host, url.host);
        req.set(http_ns::field::content_type, content_type);
        req.set(http_ns::field::connection, "keep-alive");
        for (auto& [key, value] : headers) {
            req.set(key, value);
        }
        req.body() = body_str;
        req.prepare_payload();
        return req;
    }

    // Perform a full (non-streaming) request on a worker thread.
    HttpResponse do_request(const Url& url, const std::string& body_str,
                            const std::string& content_type,
                            const Headers& headers,
                            CancellationToken& cancel) {
        auto conn = pool.acquire(url.host, url.port, url.is_tls());

        try {
            ensure_connected(*conn);
            cancel.throw_if_cancelled();

            auto req = build_request(url, body_str, content_type, headers);

            beast::flat_buffer buffer;
            http_ns::response<http_ns::string_body> res;

            if (conn->is_tls) {
                http_ns::write(*conn->ssl_stream, req);
                cancel.throw_if_cancelled();
                http_ns::read(*conn->ssl_stream, buffer, res);
            } else {
                http_ns::write(*conn->tcp_stream, req);
                cancel.throw_if_cancelled();
                http_ns::read(*conn->tcp_stream, buffer, res);
            }

            HttpResponse response;
            response.status_code = static_cast<int>(res.result_int());
            for (auto& field : res) {
                response.headers[std::string(field.name_string())] =
                    std::string(field.value());
            }
            response.body = std::move(res.body());

            // Return connection to pool for reuse (keep-alive).
            pool.release(std::move(conn));

            return response;
        } catch (...) {
            // Connection is likely broken; don't return to pool.
            throw;
        }
    }

    // Perform a streaming request: write the request, read headers, then
    // incrementally read body chunks into a channel.
    void do_streaming_request(
        const Url& url,
        const std::string& body_str,
        const Headers& headers,
        CancellationToken cancel,
        std::shared_ptr<StreamChannel> channel,
        std::promise<std::pair<int, Headers>> header_promise
    ) {
        try {
            auto conn = pool.acquire(url.host, url.port, url.is_tls());
            ensure_connected(*conn);
            cancel.throw_if_cancelled();

            auto req = build_request(url, body_str, "application/json",
                                     headers);

            if (conn->is_tls) {
                http_ns::write(*conn->ssl_stream, req);
            } else {
                http_ns::write(*conn->tcp_stream, req);
            }

            cancel.throw_if_cancelled();

            beast::flat_buffer buffer;

            if (conn->is_tls) {
                read_streaming_response(*conn->ssl_stream, buffer, cancel,
                                        channel, header_promise);
            } else {
                read_streaming_response(*conn->tcp_stream, buffer, cancel,
                                        channel, header_promise);
            }

            // Return connection to pool for keep-alive reuse.
            pool.release(std::move(conn));

        } catch (...) {
            try {
                header_promise.set_exception(std::current_exception());
            } catch (...) {
                // header_promise already satisfied; push error to channel.
                channel->set_error(std::current_exception());
            }
        }
    }

    // Read response incrementally from any Beast stream type.
    template <typename Stream>
    void read_streaming_response(
        Stream& stream,
        beast::flat_buffer& buffer,
        CancellationToken& cancel,
        std::shared_ptr<StreamChannel>& channel,
        std::promise<std::pair<int, Headers>>& header_promise
    ) {
        // Use response_parser with buffer_body for incremental reads.
        http_ns::response_parser<http_ns::buffer_body> parser;
        parser.body_limit(boost::none);

        // Read headers only.
        http_ns::read_header(stream, buffer, parser);

        int status_code = static_cast<int>(parser.get().result_int());
        Headers response_headers;
        for (auto& field : parser.get()) {
            response_headers[std::string(field.name_string())] =
                std::string(field.value());
        }

        // Deliver headers to the awaiting coroutine.
        header_promise.set_value({status_code, std::move(response_headers)});

        // If error status, read full body and signal error via channel.
        if (status_code >= 400) {
            std::string error_body;
            constexpr size_t read_buf_size = 8192;
            char read_buf[read_buf_size];

            while (!parser.is_done()) {
                parser.get().body().data = read_buf;
                parser.get().body().size = read_buf_size;

                beast::error_code ec;
                http_ns::read_some(stream, buffer, parser, ec);

                if (ec == http_ns::error::need_buffer) {
                    ec = {};
                }
                if (ec && ec != http_ns::error::end_of_stream) {
                    channel->set_error(
                        std::make_exception_ptr(std::runtime_error(
                            "Read error: " + ec.message())));
                    return;
                }

                size_t bytes_read =
                    read_buf_size - parser.get().body().size;
                error_body.append(read_buf, bytes_read);
            }

            bool retryable = (status_code == 408 || status_code == 429 ||
                              (status_code >= 500 && status_code < 600));
            channel->set_error(std::make_exception_ptr(
                error::ApiCallError(
                    "API call failed with status " +
                        std::to_string(status_code),
                    "", status_code, error_body, {},
                    retryable)));
            return;
        }

        // Read body in chunks and push to channel for SSE streaming.
        constexpr size_t chunk_buf_size = 4096;
        char chunk_buf[chunk_buf_size];

        while (!parser.is_done()) {
            cancel.throw_if_cancelled();

            parser.get().body().data = chunk_buf;
            parser.get().body().size = chunk_buf_size;

            beast::error_code ec;
            http_ns::read_some(stream, buffer, parser, ec);

            if (ec == http_ns::error::need_buffer) {
                ec = {};
            }

            if (ec && ec != http_ns::error::end_of_stream) {
                channel->set_error(
                    std::make_exception_ptr(std::runtime_error(
                        "Stream read error: " + ec.message())));
                return;
            }

            size_t bytes_read = chunk_buf_size - parser.get().body().size;
            if (bytes_read > 0) {
                std::vector<uint8_t> chunk(
                    reinterpret_cast<uint8_t*>(chunk_buf),
                    reinterpret_cast<uint8_t*>(chunk_buf) + bytes_read);
                channel->push(std::move(chunk));
            }

            if (ec == http_ns::error::end_of_stream) {
                break;
            }
        }

        channel->finish();
    }
};

HttpClient::HttpClient(net::io_context& ioc, HttpClientConfig config)
    : impl_(std::make_unique<Impl>(ioc, config)) {}

HttpClient::~HttpClient() = default;

boost::asio::io_context& HttpClient::get_io_context() const {
    return impl_->ioc;
}

Task<HttpResponse> HttpClient::get(
    std::string_view url_str,
    Headers headers,
    CancellationToken cancel
) {
    auto url = Url::parse(url_str);

    auto fut = std::async(std::launch::async,
        [this, url = std::move(url),
         headers = std::move(headers), cancel]() mutable {
            return impl_->do_get_request(url, headers, cancel);
        });

    auto response = co_await FutureAwaitable<HttpResponse>{std::move(fut)};

    if (response.status_code >= 400) {
        default_error_handler(response.status_code, response.body,
                              response.headers);
    }

    co_return response;
}

Task<HttpResponse> HttpClient::post_json(
    std::string_view url_str,
    const boost::json::value& body,
    Headers headers,
    CancellationToken cancel
) {
    auto url = Url::parse(url_str);
    auto body_str = boost::json::serialize(body);

    // Launch blocking I/O on a worker thread via std::async.
    auto fut = std::async(std::launch::async,
        [this, url = std::move(url), body_str = std::move(body_str),
         headers = std::move(headers), cancel]() mutable {
            return impl_->do_request(url, body_str, "application/json",
                                     headers, cancel);
        });

    // Await the future without blocking the coroutine.
    auto response = co_await FutureAwaitable<HttpResponse>{std::move(fut)};

    if (response.status_code >= 400) {
        default_error_handler(response.status_code, response.body,
                              response.headers);
    }

    co_return response;
}

Task<StreamingResponse> HttpClient::post_streaming(
    std::string_view url_str,
    const boost::json::value& body,
    Headers headers,
    CancellationToken cancel
) {
    auto url = Url::parse(url_str);
    auto body_str = boost::json::serialize(body);

    auto channel = std::make_shared<StreamChannel>();
    std::promise<std::pair<int, Headers>> header_promise;
    auto header_future = header_promise.get_future();

    // Launch streaming work on the thread pool.
    net::post(impl_->worker_pool,
        [this, url = std::move(url), body_str = std::move(body_str),
         headers = std::move(headers), cancel, channel,
         header_promise = std::move(header_promise)]() mutable {
            impl_->do_streaming_request(
                url, body_str, headers, cancel,
                channel, std::move(header_promise));
        });

    // Wait for headers from the worker thread.
    auto [status_code, response_headers] =
        co_await FutureAwaitable<std::pair<int, Headers>>{
            std::move(header_future)};

    StreamingResponse response;
    response.status_code = status_code;
    response.headers = std::move(response_headers);

    // Create a generator that yields chunks from the channel.
    // The channel's pop() blocks until data arrives from the worker thread.
    response.body_stream = [](std::shared_ptr<StreamChannel> ch)
        -> AsyncGenerator<std::vector<uint8_t>> {
        while (true) {
            auto chunk = ch->pop();
            if (!chunk.has_value()) {
                break; // stream finished
            }
            co_yield std::move(*chunk);
        }
    }(channel);

    co_return std::move(response);
}

Task<HttpResponse> HttpClient::post_multipart(
    std::string_view url_str,
    MultipartFormData form,
    Headers headers,
    CancellationToken cancel
) {
    auto url = Url::parse(url_str);
    auto content_type = form.content_type();
    auto body_str = form.body();

    auto fut = std::async(std::launch::async,
        [this, url = std::move(url), body_str = std::move(body_str),
         content_type = std::move(content_type),
         headers = std::move(headers), cancel]() mutable {
            return impl_->do_request(url, body_str, content_type,
                                     headers, cancel);
        });

    auto response = co_await FutureAwaitable<HttpResponse>{std::move(fut)};

    if (response.status_code >= 400) {
        default_error_handler(response.status_code, response.body,
                              response.headers);
    }

    co_return response;
}

} // namespace ai::http
