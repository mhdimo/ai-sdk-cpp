#include <ai/schema/validator.hpp>
#include <ai/util/json.hpp>

namespace ai::schema {

Validator::Validator(const JsonSchema& schema) : schema_(schema) {}

ValidationResult Validator::validate(const boost::json::value& value) const {
    std::vector<std::string> errors;
    bool ok = validate_impl(value, schema_.raw(), "", errors);
    if (ok) {
        return {.success = true, .value = value, .errors = {}};
    }
    return {.success = false, .value = std::nullopt, .errors = std::move(errors)};
}

bool Validator::validate_impl(
    const boost::json::value& value,
    const boost::json::object& schema,
    const std::string& path,
    std::vector<std::string>& errors
) const {
    auto type_it = schema.find("type");
    if (type_it != schema.end() && type_it->value().is_string()) {
        auto type = type_it->value().as_string();
        if (!validate_type(value, type, path, errors)) {
            return false;
        }
    }

    auto enum_it = schema.find("enum");
    if (enum_it != schema.end() && enum_it->value().is_array()) {
        auto& arr = enum_it->value().as_array();
        bool found = false;
        for (auto& item : arr) {
            if (item == value) { found = true; break; }
        }
        if (!found) {
            errors.push_back(path + ": value not in enum");
            return false;
        }
    }

    if (value.is_object() && type_it != schema.end() &&
        type_it->value().as_string() == "object") {
        return validate_object(value.as_object(), schema, path, errors);
    }

    if (value.is_array() && type_it != schema.end() &&
        type_it->value().as_string() == "array") {
        return validate_array(value.as_array(), schema, path, errors);
    }

    return true;
}

bool Validator::validate_type(
    const boost::json::value& value,
    std::string_view type,
    const std::string& path,
    std::vector<std::string>& errors
) const {
    bool valid = false;
    if (type == "string") valid = value.is_string();
    else if (type == "number") valid = value.is_number();
    else if (type == "integer") valid = value.is_int64() || value.is_uint64();
    else if (type == "boolean") valid = value.is_bool();
    else if (type == "object") valid = value.is_object();
    else if (type == "array") valid = value.is_array();
    else if (type == "null") valid = value.is_null();
    else valid = true;

    if (!valid) {
        errors.push_back(path + ": expected type " + std::string(type));
        return false;
    }
    return true;
}

bool Validator::validate_object(
    const boost::json::object& obj,
    const boost::json::object& schema,
    const std::string& path,
    std::vector<std::string>& errors
) const {
    auto req_it = schema.find("required");
    if (req_it != schema.end() && req_it->value().is_array()) {
        for (auto& req : req_it->value().as_array()) {
            if (req.is_string()) {
                auto key = req.as_string();
                if (obj.find(key) == obj.end()) {
                    errors.push_back(path + ": missing required field '" +
                                     std::string(key) + "'");
                    return false;
                }
            }
        }
    }

    auto props_it = schema.find("properties");
    if (props_it != schema.end() && props_it->value().is_object()) {
        auto& props = props_it->value().as_object();
        for (auto& [key, prop_schema] : props) {
            auto it = obj.find(key);
            if (it != obj.end() && prop_schema.is_object()) {
                std::string child_path = path.empty() ? std::string(key) : path + "." + std::string(key);
                if (!validate_impl(it->value(), prop_schema.as_object(), child_path, errors)) {
                    return false;
                }
            }
        }
    }

    auto additional_it = schema.find("additionalProperties");
    if (additional_it != schema.end() && additional_it->value().is_bool() &&
        !additional_it->value().as_bool()) {
        if (props_it != schema.end() && props_it->value().is_object()) {
            auto& props = props_it->value().as_object();
            for (auto& [key, _] : obj) {
                if (props.find(key) == props.end()) {
                    errors.push_back(path + ": unexpected additional property '" +
                                     std::string(key) + "'");
                    return false;
                }
            }
        }
    }

    return true;
}

bool Validator::validate_array(
    const boost::json::array& arr,
    const boost::json::object& schema,
    const std::string& path,
    std::vector<std::string>& errors
) const {
    auto min_it = schema.find("minItems");
    if (min_it != schema.end() && min_it->value().is_int64()) {
        if (static_cast<int64_t>(arr.size()) < min_it->value().as_int64()) {
            errors.push_back(path + ": array has fewer items than minItems");
            return false;
        }
    }

    auto max_it = schema.find("maxItems");
    if (max_it != schema.end() && max_it->value().is_int64()) {
        if (static_cast<int64_t>(arr.size()) > max_it->value().as_int64()) {
            errors.push_back(path + ": array has more items than maxItems");
            return false;
        }
    }

    auto items_it = schema.find("items");
    if (items_it != schema.end() && items_it->value().is_object()) {
        auto& items_schema = items_it->value().as_object();
        for (size_t i = 0; i < arr.size(); ++i) {
            std::string child_path = path + "[" + std::to_string(i) + "]";
            if (!validate_impl(arr[i], items_schema, child_path, errors)) {
                return false;
            }
        }
    }

    return true;
}

} // namespace ai::schema
