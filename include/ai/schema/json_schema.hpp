#pragma once

#include <boost/json.hpp>
#include <string>
#include <string_view>
#include <vector>
#include <initializer_list>
#include <utility>

namespace ai::schema {

class JsonSchema {
public:
    JsonSchema() : schema_(boost::json::object{}) {}
    explicit JsonSchema(boost::json::object schema) : schema_(std::move(schema)) {}

    const boost::json::object& raw() const { return schema_; }
    std::string to_string() const { return boost::json::serialize(schema_); }

    static JsonSchema object(std::initializer_list<std::pair<std::string, JsonSchema>> properties) {
        boost::json::object props;
        for (auto& [name, schema] : properties) {
            props[name] = schema.schema_;
        }
        boost::json::object result;
        result["type"] = "object";
        result["properties"] = std::move(props);
        return JsonSchema{std::move(result)};
    }

    static JsonSchema string(std::string description = "") {
        boost::json::object result;
        result["type"] = "string";
        if (!description.empty()) {
            result["description"] = description;
        }
        return JsonSchema{std::move(result)};
    }

    static JsonSchema number(std::string description = "") {
        boost::json::object result;
        result["type"] = "number";
        if (!description.empty()) {
            result["description"] = description;
        }
        return JsonSchema{std::move(result)};
    }

    static JsonSchema integer(std::string description = "") {
        boost::json::object result;
        result["type"] = "integer";
        if (!description.empty()) {
            result["description"] = description;
        }
        return JsonSchema{std::move(result)};
    }

    static JsonSchema boolean(std::string description = "") {
        boost::json::object result;
        result["type"] = "boolean";
        if (!description.empty()) {
            result["description"] = description;
        }
        return JsonSchema{std::move(result)};
    }

    static JsonSchema array(JsonSchema items) {
        boost::json::object result;
        result["type"] = "array";
        result["items"] = items.schema_;
        return JsonSchema{std::move(result)};
    }

    static JsonSchema enum_of(std::vector<std::string> values) {
        boost::json::object result;
        result["type"] = "string";
        boost::json::array arr;
        for (auto& v : values) {
            arr.push_back(boost::json::value(v));
        }
        result["enum"] = std::move(arr);
        return JsonSchema{std::move(result)};
    }

    static JsonSchema null() {
        boost::json::object result;
        result["type"] = "null";
        return JsonSchema{std::move(result)};
    }

    JsonSchema& required(std::vector<std::string> fields) {
        boost::json::array arr;
        for (auto& f : fields) {
            arr.push_back(boost::json::value(f));
        }
        schema_["required"] = std::move(arr);
        return *this;
    }

    JsonSchema& with_description(std::string desc) {
        schema_["description"] = desc;
        return *this;
    }

    JsonSchema& additional_properties(bool allow) {
        schema_["additionalProperties"] = allow;
        return *this;
    }

    JsonSchema& min_items(int n) {
        schema_["minItems"] = n;
        return *this;
    }

    JsonSchema& max_items(int n) {
        schema_["maxItems"] = n;
        return *this;
    }

    JsonSchema& minimum(double n) {
        schema_["minimum"] = n;
        return *this;
    }

    JsonSchema& maximum(double n) {
        schema_["maximum"] = n;
        return *this;
    }

    JsonSchema& pattern(std::string p) {
        schema_["pattern"] = p;
        return *this;
    }

private:
    boost::json::object schema_;
};

} // namespace ai::schema
