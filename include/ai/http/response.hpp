#pragma once

#include <ai/http/client.hpp>
#include <ai/error/api_call_error.hpp>
#include <ai/util/json.hpp>
#include <boost/json.hpp>
#include <functional>
#include <string>
#include <optional>

namespace ai::http {

using ErrorHandler = std::function<void(int status_code, const std::string& body, const Headers& headers)>;

inline std::string extract_error_message(const std::string& body, int status_code) {
    std::string message = "API call failed with status " + std::to_string(status_code);
    auto parsed = ai::json::safe_parse(body);
    if (parsed && parsed->is_object()) {
        auto& obj = parsed->as_object();
        if (auto it = obj.find("error"); it != obj.end()) {
            if (it->value().is_object()) {
                auto& err = it->value().as_object();
                if (auto mit = err.find("message"); mit != err.end() && mit->value().is_string()) {
                    message = std::string(mit->value().as_string());
                }
            } else if (it->value().is_string()) {
                message = std::string(it->value().as_string());
            }
        }
        if (auto it = obj.find("message"); it != obj.end() && it->value().is_string()) {
            message = std::string(it->value().as_string());
        }
    }
    return message;
}

inline void default_error_handler(int status_code, const std::string& body, const Headers& headers) {
    std::string message = extract_error_message(body, status_code);
    std::string url;

    switch (status_code) {
    case 400:
        throw error::BadRequestError(std::move(message), url, body, headers);
    case 401:
        throw error::AuthenticationError(std::move(message), url, body, headers);
    case 403:
        throw error::PermissionDeniedError(std::move(message), url, body, headers);
    case 404:
        throw error::NotFoundError(std::move(message), url, body, headers);
    case 429:
        throw error::RateLimitError(std::move(message), url, body, headers);
    case 503:
        throw error::ServiceUnavailableError(std::move(message), url, body, headers);
    default: {
        bool retryable = (status_code == 408 || status_code == 429 ||
                          (status_code >= 500 && status_code < 600));
        throw error::ApiCallError(
            std::move(message), url, status_code,
            body, headers, retryable
        );
    }
    }
}

inline boost::json::value safe_parse_response(const std::string& body) {
    auto parsed = ai::json::safe_parse(body);
    if (!parsed) {
        throw error::InvalidResponseError(
            "Failed to parse API response as JSON", body);
    }
    return std::move(*parsed);
}

} // namespace ai::http
