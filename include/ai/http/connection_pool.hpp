#pragma once

#include <boost/asio.hpp>
#include <boost/asio/ssl.hpp>
#include <boost/beast.hpp>
#include <string>
#include <string_view>
#include <memory>
#include <mutex>
#include <deque>
#include <unordered_map>
#include <chrono>

namespace ai::http {

class ConnectionPool {
public:
    ConnectionPool(boost::asio::io_context& ioc,
                   size_t max_per_host = 8,
                   std::chrono::seconds idle_timeout = std::chrono::seconds{60});
    ~ConnectionPool();

    using TcpStream = boost::beast::tcp_stream;
    using SslStream = boost::asio::ssl::stream<TcpStream>;

    struct Connection {
        std::unique_ptr<SslStream> ssl_stream;
        std::unique_ptr<TcpStream> tcp_stream;
        std::string host;
        std::string port;
        bool is_tls;
        bool connected = false;
        std::chrono::steady_clock::time_point last_used;

        bool is_open() const;
    };

    // Acquire a connection - may return a pooled (already connected) one.
    std::shared_ptr<Connection> acquire(std::string_view host,
                                        std::string_view port,
                                        bool tls);

    // Return a connection to the pool for reuse.
    void release(std::shared_ptr<Connection> conn);

    // Evict idle connections older than the timeout.
    void evict_expired();

    // Get the SSL context (for creating new connections).
    boost::asio::ssl::context& ssl_context() { return ssl_ctx_; }

    boost::asio::io_context& io_context() { return ioc_; }

private:
    boost::asio::io_context& ioc_;
    boost::asio::ssl::context ssl_ctx_;
    size_t max_per_host_;
    std::chrono::seconds idle_timeout_;
    std::mutex mutex_;

    struct HostKey {
        std::string host;
        std::string port;
        bool tls;
        bool operator==(const HostKey&) const = default;
    };

    struct HostKeyHash {
        size_t operator()(const HostKey& k) const {
            size_t h = std::hash<std::string>{}(k.host);
            h ^= std::hash<std::string>{}(k.port) << 1;
            h ^= std::hash<bool>{}(k.tls) << 2;
            return h;
        }
    };

    std::unordered_map<HostKey, std::deque<std::shared_ptr<Connection>>, HostKeyHash> pool_;
};

} // namespace ai::http
