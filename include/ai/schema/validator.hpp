#pragma once

#include <ai/schema/json_schema.hpp>
#include <boost/json.hpp>
#include <string>
#include <vector>
#include <optional>

namespace ai::schema {

struct ValidationResult {
    bool success;
    std::optional<boost::json::value> value;
    std::vector<std::string> errors;
};

class Validator {
public:
    explicit Validator(const JsonSchema& schema);

    ValidationResult validate(const boost::json::value& value) const;

private:
    JsonSchema schema_;

    bool validate_impl(
        const boost::json::value& value,
        const boost::json::object& schema,
        const std::string& path,
        std::vector<std::string>& errors
    ) const;

    bool validate_type(
        const boost::json::value& value,
        std::string_view type,
        const std::string& path,
        std::vector<std::string>& errors
    ) const;

    bool validate_object(
        const boost::json::object& obj,
        const boost::json::object& schema,
        const std::string& path,
        std::vector<std::string>& errors
    ) const;

    bool validate_array(
        const boost::json::array& arr,
        const boost::json::object& schema,
        const std::string& path,
        std::vector<std::string>& errors
    ) const;
};

} // namespace ai::schema
