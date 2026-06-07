#pragma once

#include <ai/schema/json_schema.hpp>
#include <string>
#include <vector>

namespace ai::schema::dsl {

class ObjectSchemaBuilder {
public:
    ObjectSchemaBuilder& property(std::string name, JsonSchema schema) {
        properties_.emplace_back(std::move(name), std::move(schema));
        return *this;
    }

    ObjectSchemaBuilder& required(std::string name) {
        required_.push_back(std::move(name));
        return *this;
    }

    ObjectSchemaBuilder& description(std::string desc) {
        description_ = std::move(desc);
        return *this;
    }

    ObjectSchemaBuilder& additional_properties(bool v) {
        additional_properties_ = v;
        return *this;
    }

    operator JsonSchema() const {
        auto schema = JsonSchema::object({});
        boost::json::object props;
        for (auto& [name, s] : properties_) {
            props[name] = s.raw();
        }
        auto& raw = const_cast<boost::json::object&>(schema.raw());
        raw["properties"] = std::move(props);
        if (!required_.empty()) {
            schema.required(required_);
        }
        if (description_) {
            schema.with_description(*description_);
        }
        if (additional_properties_) {
            schema.additional_properties(*additional_properties_);
        }
        return schema;
    }

private:
    std::vector<std::pair<std::string, JsonSchema>> properties_;
    std::vector<std::string> required_;
    std::optional<std::string> description_;
    std::optional<bool> additional_properties_;
};

inline ObjectSchemaBuilder object() { return {}; }

} // namespace ai::schema::dsl
