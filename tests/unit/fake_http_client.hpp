#pragma once

// A shared in-memory HTTP client for provider unit tests (no network).

#include <ai/http/client.hpp>
#include <ai/http/multipart.hpp>
#include <ai/stream/async_generator.hpp>
#include <ai/util/cancellation.hpp>

#include <boost/asio.hpp>
#include <boost/json.hpp>

#include <stdexcept>
#include <string>
#include <vector>

namespace ai::test {

class FakeHttpClient : public ai::http::IHttpClient {
public:
    std::string json_body;    // returned by post_json
    std::string stream_body;  // returned by post_streaming (raw bytes)
    int post_json_calls = 0;
    int post_stream_calls = 0;
    // Most recent request body each call saw, so tests can assert what the
    // provider actually put on the wire (e.g. reasoning_effort, response_format).
    boost::json::value last_request_body;
    boost::json::value last_stream_request_body;
    boost::asio::io_context& ioc;

    explicit FakeHttpClient(boost::asio::io_context& ctx) : ioc(ctx) {}

    ai::Task<ai::http::HttpResponse> post_json(
        std::string_view, const boost::json::value& body, ai::http::Headers,
        ai::CancellationToken
    ) override {
        ++post_json_calls;
        last_request_body = body;
        co_return ai::http::HttpResponse{200, {}, json_body};
    }

    ai::Task<ai::http::StreamingResponse> post_streaming(
        std::string_view, const boost::json::value& body, ai::http::Headers,
        ai::CancellationToken
    ) override {
        ++post_stream_calls;
        last_stream_request_body = body;
        auto gen = [](std::string body) -> ai::AsyncGenerator<std::vector<uint8_t>> {
            std::vector<uint8_t> bytes(body.begin(), body.end());
            co_yield bytes;
        }(stream_body);
        co_return ai::http::StreamingResponse{200, {}, std::move(gen)};
    }

    ai::Task<ai::http::HttpResponse> get(
        std::string_view, ai::http::Headers, ai::CancellationToken
    ) override {
        throw std::runtime_error("FakeHttpClient::get not used");
    }

    ai::Task<ai::http::HttpResponse> post_multipart(
        std::string_view, ai::http::MultipartFormData, ai::http::Headers,
        ai::CancellationToken
    ) override {
        throw std::runtime_error("FakeHttpClient::post_multipart not used");
    }

    boost::asio::io_context& get_io_context() const override { return ioc; }
};

} // namespace ai::test
