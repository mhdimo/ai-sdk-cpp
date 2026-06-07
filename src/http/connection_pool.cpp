#include <ai/http/connection_pool.hpp>
#include <openssl/ssl.h>
#include <algorithm>

namespace ai::http {

namespace beast = boost::beast;
namespace net = boost::asio;
namespace ssl = net::ssl;

ConnectionPool::ConnectionPool(net::io_context& ioc, size_t max_per_host,
                               std::chrono::seconds idle_timeout)
    : ioc_(ioc)
    , ssl_ctx_(ssl::context::tlsv12_client)
    , max_per_host_(max_per_host)
    , idle_timeout_(idle_timeout) {
    ssl_ctx_.set_default_verify_paths();
    ssl_ctx_.set_verify_mode(ssl::verify_peer);
}

ConnectionPool::~ConnectionPool() = default;

bool ConnectionPool::Connection::is_open() const {
    if (is_tls) {
        if (!ssl_stream) return false;
        auto& lowest = beast::get_lowest_layer(*ssl_stream);
        return lowest.socket().is_open();
    } else {
        if (!tcp_stream) return false;
        return tcp_stream->socket().is_open();
    }
}

std::shared_ptr<ConnectionPool::Connection> ConnectionPool::acquire(
    std::string_view host, std::string_view port, bool tls
) {
    HostKey key{std::string(host), std::string(port), tls};

    {
        std::lock_guard lock(mutex_);
        auto it = pool_.find(key);
        if (it != pool_.end()) {
            while (!it->second.empty()) {
                auto conn = std::move(it->second.front());
                it->second.pop_front();

                // Check if the connection is still alive and not expired.
                auto now = std::chrono::steady_clock::now();
                auto idle_time = now - conn->last_used;
                if (idle_time > idle_timeout_) {
                    continue; // discard expired
                }
                if (!conn->is_open()) {
                    continue; // discard dead connections
                }

                return conn;
            }
        }
    }

    // No pooled connection available, create a new one.
    auto conn = std::make_shared<Connection>();
    conn->host = std::string(host);
    conn->port = std::string(port);
    conn->is_tls = tls;
    conn->connected = false;
    conn->last_used = std::chrono::steady_clock::now();

    if (tls) {
        conn->ssl_stream = std::make_unique<SslStream>(
            net::make_strand(ioc_), ssl_ctx_
        );
    } else {
        conn->tcp_stream = std::make_unique<TcpStream>(
            net::make_strand(ioc_)
        );
    }

    return conn;
}

void ConnectionPool::release(std::shared_ptr<Connection> conn) {
    if (!conn || !conn->is_open()) {
        return; // don't pool dead connections
    }

    conn->last_used = std::chrono::steady_clock::now();

    HostKey key{conn->host, conn->port, conn->is_tls};

    std::lock_guard lock(mutex_);
    auto& queue = pool_[key];
    if (queue.size() < max_per_host_) {
        queue.push_back(std::move(conn));
    }
    // else: discard (pool full for this host)
}

void ConnectionPool::evict_expired() {
    auto now = std::chrono::steady_clock::now();
    std::lock_guard lock(mutex_);

    for (auto it = pool_.begin(); it != pool_.end();) {
        auto& queue = it->second;
        std::erase_if(queue, [&](const std::shared_ptr<Connection>& conn) {
            return (now - conn->last_used) > idle_timeout_ || !conn->is_open();
        });
        if (queue.empty()) {
            it = pool_.erase(it);
        } else {
            ++it;
        }
    }
}

} // namespace ai::http
