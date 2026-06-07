#include <ai/error/api_call_error.hpp>
#include <charconv>

namespace ai::error {

ApiCallError::ApiCallError(
    std::string message,
    std::string url,
    int status_code,
    std::string response_body,
    Headers response_headers,
    bool retryable,
    std::exception_ptr cause
)
    : AiError("AI_APICallError", std::move(message), std::move(cause))
    , url_(std::move(url))
    , status_code_(status_code)
    , retryable_(retryable)
    , response_body_(std::move(response_body))
    , response_headers_(std::move(response_headers)) {}

const std::string& ApiCallError::url() const noexcept {
    return url_;
}

int ApiCallError::status_code() const noexcept {
    return status_code_;
}

bool ApiCallError::is_retryable() const noexcept {
    return retryable_;
}

const std::string& ApiCallError::response_body() const noexcept {
    return response_body_;
}

const Headers& ApiCallError::response_headers() const noexcept {
    return response_headers_;
}

std::optional<std::chrono::seconds> ApiCallError::retry_after() const noexcept {
    auto it = response_headers_.find("retry-after");
    if (it == response_headers_.end()) {
        it = response_headers_.find("Retry-After");
    }
    if (it == response_headers_.end()) return std::nullopt;

    int seconds = 0;
    auto [ptr, ec] = std::from_chars(
        it->second.data(),
        it->second.data() + it->second.size(),
        seconds);
    if (ec == std::errc()) {
        return std::chrono::seconds(seconds);
    }
    return std::nullopt;
}

// --- Derived error types ---

RateLimitError::RateLimitError(
    std::string message,
    std::string url,
    std::string response_body,
    Headers response_headers,
    std::exception_ptr cause
)
    : ApiCallError(std::move(message), std::move(url), 429,
                   std::move(response_body), std::move(response_headers),
                   true, std::move(cause)) {}

AuthenticationError::AuthenticationError(
    std::string message,
    std::string url,
    std::string response_body,
    Headers response_headers,
    std::exception_ptr cause
)
    : ApiCallError(std::move(message), std::move(url), 401,
                   std::move(response_body), std::move(response_headers),
                   false, std::move(cause)) {}

PermissionDeniedError::PermissionDeniedError(
    std::string message,
    std::string url,
    std::string response_body,
    Headers response_headers,
    std::exception_ptr cause
)
    : ApiCallError(std::move(message), std::move(url), 403,
                   std::move(response_body), std::move(response_headers),
                   false, std::move(cause)) {}

NotFoundError::NotFoundError(
    std::string message,
    std::string url,
    std::string response_body,
    Headers response_headers,
    std::exception_ptr cause
)
    : ApiCallError(std::move(message), std::move(url), 404,
                   std::move(response_body), std::move(response_headers),
                   false, std::move(cause)) {}

BadRequestError::BadRequestError(
    std::string message,
    std::string url,
    std::string response_body,
    Headers response_headers,
    std::exception_ptr cause
)
    : ApiCallError(std::move(message), std::move(url), 400,
                   std::move(response_body), std::move(response_headers),
                   false, std::move(cause)) {}

ContentFilterError::ContentFilterError(
    std::string message,
    std::string url,
    std::string response_body,
    Headers response_headers,
    std::exception_ptr cause
)
    : ApiCallError(std::move(message), std::move(url), 451,
                   std::move(response_body), std::move(response_headers),
                   false, std::move(cause)) {}

ServiceUnavailableError::ServiceUnavailableError(
    std::string message,
    std::string url,
    std::string response_body,
    Headers response_headers,
    std::exception_ptr cause
)
    : ApiCallError(std::move(message), std::move(url), 503,
                   std::move(response_body), std::move(response_headers),
                   true, std::move(cause)) {}

InvalidResponseError::InvalidResponseError(
    std::string message,
    std::string response_body,
    std::exception_ptr cause
)
    : AiError("AI_InvalidResponseError", std::move(message), std::move(cause))
    , response_body_(std::move(response_body)) {}

const std::string& InvalidResponseError::response_body() const noexcept {
    return response_body_;
}

StreamError::StreamError(std::string message, std::exception_ptr cause)
    : AiError("AI_StreamError", std::move(message), std::move(cause)) {}

TimeoutError::TimeoutError(std::string message, std::chrono::milliseconds elapsed)
    : AiError("AI_TimeoutError", std::move(message))
    , elapsed_(elapsed) {}

std::chrono::milliseconds TimeoutError::elapsed() const noexcept {
    return elapsed_;
}

} // namespace ai::error
