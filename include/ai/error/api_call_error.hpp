#pragma once

#include <ai/error/ai_error.hpp>
#include <string>
#include <unordered_map>
#include <optional>
#include <chrono>

namespace ai::error {

using Headers = std::unordered_map<std::string, std::string>;

class ApiCallError : public AiError {
public:
    ApiCallError(
        std::string message,
        std::string url,
        int status_code,
        std::string response_body,
        Headers response_headers,
        bool retryable,
        std::exception_ptr cause = nullptr
    );

    std::string_view error_type() const noexcept override {
        return "AI_APICallError";
    }

    const std::string& url() const noexcept;
    int status_code() const noexcept;
    bool is_retryable() const noexcept;
    const std::string& response_body() const noexcept;
    const Headers& response_headers() const noexcept;

    std::optional<std::chrono::seconds> retry_after() const noexcept;

private:
    std::string url_;
    int status_code_;
    bool retryable_;
    std::string response_body_;
    Headers response_headers_;
};

class RateLimitError : public ApiCallError {
public:
    RateLimitError(
        std::string message,
        std::string url,
        std::string response_body,
        Headers response_headers,
        std::exception_ptr cause = nullptr
    );

    std::string_view error_type() const noexcept override {
        return "AI_RateLimitError";
    }
};

class AuthenticationError : public ApiCallError {
public:
    AuthenticationError(
        std::string message,
        std::string url,
        std::string response_body,
        Headers response_headers,
        std::exception_ptr cause = nullptr
    );

    std::string_view error_type() const noexcept override {
        return "AI_AuthenticationError";
    }
};

class PermissionDeniedError : public ApiCallError {
public:
    PermissionDeniedError(
        std::string message,
        std::string url,
        std::string response_body,
        Headers response_headers,
        std::exception_ptr cause = nullptr
    );

    std::string_view error_type() const noexcept override {
        return "AI_PermissionDeniedError";
    }
};

class NotFoundError : public ApiCallError {
public:
    NotFoundError(
        std::string message,
        std::string url,
        std::string response_body,
        Headers response_headers,
        std::exception_ptr cause = nullptr
    );

    std::string_view error_type() const noexcept override {
        return "AI_NotFoundError";
    }
};

class BadRequestError : public ApiCallError {
public:
    BadRequestError(
        std::string message,
        std::string url,
        std::string response_body,
        Headers response_headers,
        std::exception_ptr cause = nullptr
    );

    std::string_view error_type() const noexcept override {
        return "AI_BadRequestError";
    }
};

class ContentFilterError : public ApiCallError {
public:
    ContentFilterError(
        std::string message,
        std::string url,
        std::string response_body,
        Headers response_headers,
        std::exception_ptr cause = nullptr
    );

    std::string_view error_type() const noexcept override {
        return "AI_ContentFilterError";
    }
};

class ServiceUnavailableError : public ApiCallError {
public:
    ServiceUnavailableError(
        std::string message,
        std::string url,
        std::string response_body,
        Headers response_headers,
        std::exception_ptr cause = nullptr
    );

    std::string_view error_type() const noexcept override {
        return "AI_ServiceUnavailableError";
    }
};

class InvalidResponseError : public AiError {
public:
    InvalidResponseError(
        std::string message,
        std::string response_body,
        std::exception_ptr cause = nullptr
    );

    std::string_view error_type() const noexcept override {
        return "AI_InvalidResponseError";
    }

    const std::string& response_body() const noexcept;

private:
    std::string response_body_;
};

class StreamError : public AiError {
public:
    StreamError(std::string message, std::exception_ptr cause = nullptr);

    std::string_view error_type() const noexcept override {
        return "AI_StreamError";
    }
};

class TimeoutError : public AiError {
public:
    TimeoutError(std::string message, std::chrono::milliseconds elapsed);

    std::string_view error_type() const noexcept override {
        return "AI_TimeoutError";
    }

    std::chrono::milliseconds elapsed() const noexcept;

private:
    std::chrono::milliseconds elapsed_;
};

} // namespace ai::error
