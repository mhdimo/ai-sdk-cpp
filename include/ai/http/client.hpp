#pragma once

#include <ai/stream/async_generator.hpp>
#include <ai/util/cancellation.hpp>
#include <ai/http/multipart.hpp>
#include <boost/asio.hpp>
#include <boost/asio/ssl.hpp>
#include <boost/beast.hpp>
#include <boost/json.hpp>
#include <string>
#include <string_view>
#include <vector>
#include <unordered_map>
#include <memory>

namespace ai::http {

using Headers = std::unordered_map<std::string, std::string>;

struct HttpResponse {
    int status_code;
    Headers headers;
    std::string body;
};

struct StreamingResponse {
    int status_code;
    Headers headers;
    AsyncGenerator<std::vector<uint8_t>> body_stream;
};

struct Url {
    std::string scheme;
    std::string host;
    std::string port;
    std::string path;

    static Url parse(std::string_view url_str);
    bool is_tls() const { return scheme == "https"; }
};

struct HttpClientConfig {
    size_t thread_pool_size = 4;
    size_t max_connections_per_host = 8;
    std::chrono::seconds connect_timeout{30};
    std::chrono::seconds read_timeout{300};
};

class HttpClient {
public:
    explicit HttpClient(boost::asio::io_context& ioc,
                        HttpClientConfig config = {});
    ~HttpClient();

    Task<HttpResponse> get(
        std::string_view url,
        Headers headers,
        CancellationToken cancel = {}
    );

    Task<HttpResponse> post_json(
        std::string_view url,
        const boost::json::value& body,
        Headers headers,
        CancellationToken cancel = {}
    );

    Task<StreamingResponse> post_streaming(
        std::string_view url,
        const boost::json::value& body,
        Headers headers,
        CancellationToken cancel = {}
    );

    Task<HttpResponse> post_multipart(
        std::string_view url,
        MultipartFormData form,
        Headers headers,
        CancellationToken cancel = {}
    );

    boost::asio::io_context& get_io_context() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

inline Headers combine_headers(const Headers& a, const Headers& b) {
    Headers result = a;
    for (auto& [k, v] : b) {
        result[k] = v;
    }
    return result;
}

} // namespace ai::http
