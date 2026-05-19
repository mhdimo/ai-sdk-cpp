#pragma once

#include <exception>
#include <string>
#include <string_view>
#include <memory>
#include <boost/json.hpp>

namespace ai::error {

class AiError : public std::exception {
public:
    AiError(std::string name, std::string message);
    AiError(std::string name, std::string message, std::exception_ptr cause);

    const char* what() const noexcept override;
    std::string_view name() const noexcept;
    const std::string& message() const noexcept;
    std::exception_ptr cause() const noexcept;

    virtual std::string_view error_type() const noexcept = 0;

    template <typename T>
    static bool is_instance(const std::exception& e) {
        return dynamic_cast<const T*>(&e) != nullptr;
    }

protected:
    std::string name_;
    std::string message_;
    std::string what_;
    std::exception_ptr cause_;
};

class NoOutputGeneratedError : public AiError {
public:
    explicit NoOutputGeneratedError(std::string message);

    std::string_view error_type() const noexcept override {
        return "AI_NoOutputGeneratedError";
    }
};

class UnsupportedFunctionalityError : public AiError {
public:
    UnsupportedFunctionalityError(std::string functionality);

    std::string_view error_type() const noexcept override {
        return "AI_UnsupportedFunctionalityError";
    }

    std::string_view functionality() const noexcept;

private:
    std::string functionality_;
};

class TypeValidationError : public AiError {
public:
    TypeValidationError(std::string message, boost::json::value value);

    std::string_view error_type() const noexcept override {
        return "AI_TypeValidationError";
    }

    const boost::json::value& value() const noexcept;

private:
    boost::json::value value_;
};

} // namespace ai::error
