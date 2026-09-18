#pragma once

// A real HTTP/1.1 server bound to 127.0.0.1, for tests that cannot inject a
// fake ai::http::IHttpClient.
//
// The provider unit tests all substitute IHttpClient, which is why src/http/*
// sits at a few percent coverage: nothing in this suite ever opens a socket.
// Anything reached through the C API is in that position permanently — the C
// binding constructs its own HttpClient and offers no seam — so testing the
// permission gate end-to-end means serving real bytes to a real client.
//
// Scripted, not smart: the Nth request gets the Nth body. The agent loop under
// test makes exactly two calls (one that asks for a tool, one that answers), so
// a list of two bodies is a complete conversation.

#include <boost/asio.hpp>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace ai::test {

class LocalHttpServer {
public:
    /// Serves `bodies` in order, one per request. `think_time` is how long to
    /// wait before each response, which is how a test reaches the state where
    /// the client is idle waiting on the network rather than on its own work.
    explicit LocalHttpServer(std::vector<std::string> bodies,
                             std::chrono::milliseconds think_time = std::chrono::milliseconds{0})
        : bodies_(std::move(bodies)), think_time_(think_time), acceptor_(ioc_) {
        using boost::asio::ip::tcp;
        tcp::endpoint endpoint(boost::asio::ip::make_address("127.0.0.1"), 0);
        acceptor_.open(endpoint.protocol());
        acceptor_.set_option(boost::asio::socket_base::reuse_address(true));
        acceptor_.bind(endpoint);
        acceptor_.listen();
        port_ = acceptor_.local_endpoint().port();
        thread_ = std::thread([this] { serve(); });
    }

    ~LocalHttpServer() {
        // The serving thread is parked in accept() or read_some(); closing from
        // here is what cancels that operation and lets the thread return.
        try {
            acceptor_.close();
        } catch (const std::exception&) {
            // Already closed, or never opened.
        }
        if (thread_.joinable()) {
            thread_.join();
        }
    }

    LocalHttpServer(const LocalHttpServer&) = delete;
    LocalHttpServer& operator=(const LocalHttpServer&) = delete;

    /// Point a provider's base_url at this. The path the client appends is
    /// ignored, so any suffix works.
    std::string base_url() const { return "http://127.0.0.1:" + std::to_string(port_); }

    int requests_served() const { return served_.load(); }

    /// The bodies of the requests received so far, in order. This is how a test
    /// sees what the client actually sent — the tool result handed back to the
    /// model, for instance, which is otherwise invisible from outside the loop.
    std::vector<std::string> request_bodies() const {
        std::lock_guard<std::mutex> lk(bodies_mutex_);
        return request_bodies_;
    }

    /// The nth request body (0-based), or "" if that request has not arrived.
    std::string request_body(size_t n) const {
        std::lock_guard<std::mutex> lk(bodies_mutex_);
        return n < request_bodies_.size() ? request_bodies_[n] : std::string{};
    }

private:
    static size_t content_length_of(std::string header_block) {
        for (char& c : header_block) {
            c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        }
        const std::string key = "content-length:";
        auto pos = header_block.find(key);
        if (pos == std::string::npos) {
            return 0;
        }
        pos += key.size();
        auto end = header_block.find('\r', pos);
        try {
            return static_cast<size_t>(std::stoul(header_block.substr(pos, end - pos)));
        } catch (const std::exception&) {
            return 0;
        }
    }

    void serve() {
        using boost::asio::ip::tcp;
        boost::system::error_code ec;
        while (true) {
            tcp::socket sock(ioc_);
            acceptor_.accept(sock, ec);
            if (ec) {
                return;  // acceptor closed -> shutdown
            }

            std::string pending;
            char chunk[4096];
            while (true) {
                auto header_end = pending.find("\r\n\r\n");
                while (header_end == std::string::npos) {
                    size_t n = sock.read_some(boost::asio::buffer(chunk), ec);
                    if (ec) {
                        break;  // peer closed -> back to accept
                    }
                    pending.append(chunk, n);
                    header_end = pending.find("\r\n\r\n");
                }
                if (ec) {
                    break;
                }

                size_t total = header_end + 4 + content_length_of(pending.substr(0, header_end));
                while (pending.size() < total) {
                    size_t n = sock.read_some(boost::asio::buffer(chunk), ec);
                    if (ec) {
                        break;
                    }
                    pending.append(chunk, n);
                }
                if (ec) {
                    break;
                }
                {
                    size_t body_len = total - (header_end + 4);
                    std::lock_guard<std::mutex> lk(bodies_mutex_);
                    request_bodies_.push_back(pending.substr(header_end + 4, body_len));
                }
                pending.erase(0, total);

                // Past the end of the script the last body repeats, so a test
                // that makes one call too many fails on an assertion rather
                // than on a dead socket.
                size_t index = std::min<size_t>(served_.load(), bodies_.size() - 1);
                const std::string& body = bodies_[index];
                served_.fetch_add(1);

                if (think_time_.count() > 0) {
                    std::this_thread::sleep_for(think_time_);
                }

                std::string response =
                    "HTTP/1.1 200 OK\r\n"
                    "Content-Type: application/json\r\n"
                    "Content-Length: " + std::to_string(body.size()) + "\r\n"
                    "Connection: keep-alive\r\n\r\n" + body;

                boost::asio::write(sock, boost::asio::buffer(response), ec);
                if (ec) {
                    break;
                }
            }
        }
    }

    std::vector<std::string> bodies_;
    std::chrono::milliseconds think_time_;
    boost::asio::io_context ioc_;
    boost::asio::ip::tcp::acceptor acceptor_;
    std::thread thread_;
    std::atomic<int> served_{0};
    // Written by the serving thread, read by the test thread.
    mutable std::mutex bodies_mutex_;
    std::vector<std::string> request_bodies_;
    unsigned short port_ = 0;
};

} // namespace ai::test
