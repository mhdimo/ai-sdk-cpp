#include <ai/error/ai_error.hpp>
#include <boost/json.hpp>

namespace ai::error {

AiError::AiError(std::string name, std::string message)
    : name_(std::move(name))
    , message_(std::move(message))
    , what_(name_ + ": " + message_) {}

AiError::AiError(std::string name, std::string message, std::exception_ptr cause)
    : name_(std::move(name))
    , message_(std::move(message))
    , what_(name_ + ": " + message_)
    , cause_(std::move(cause)) {}

const char* AiError::what() const noexcept {
    return what_.c_str();
}

std::string_view AiError::name() const noexcept {
    return name_;
}

const std::string& AiError::message() const noexcept {
    return message_;
}

std::exception_ptr AiError::cause() const noexcept {
    return cause_;
}

NoOutputGeneratedError::NoOutputGeneratedError(std::string message)
    : AiError("AI_NoOutputGeneratedError", std::move(message)) {}

UnsupportedFunctionalityError::UnsupportedFunctionalityError(std::string functionality)
    : AiError("AI_UnsupportedFunctionalityError",
              "Unsupported functionality: " + functionality)
    , functionality_(std::move(functionality)) {}

std::string_view UnsupportedFunctionalityError::functionality() const noexcept {
    return functionality_;
}

TypeValidationError::TypeValidationError(std::string message, boost::json::value value)
    : AiError("AI_TypeValidationError", std::move(message))
    , value_(std::move(value)) {}

const boost::json::value& TypeValidationError::value() const noexcept {
    return value_;
}

} // namespace ai::error
