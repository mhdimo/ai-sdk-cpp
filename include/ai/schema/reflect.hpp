#pragma once

#include <ai/schema/json_schema.hpp>
#include <boost/json.hpp>
#include <string>
#include <vector>

namespace ai {

template<typename T>
schema::JsonSchema schema_for();

namespace detail {

struct FieldDef {
    std::string name;
    std::string description;
    schema::JsonSchema schema;
    bool required;
};

inline schema::JsonSchema build_schema(std::initializer_list<FieldDef> fields) {
    boost::json::object props;
    std::vector<std::string> required_fields;
    for (auto& f : fields) {
        props[f.name] = f.schema.raw();
        if (f.required) required_fields.push_back(f.name);
    }
    boost::json::object result;
    result["type"] = "object";
    result["properties"] = std::move(props);
    result["additionalProperties"] = false;
    if (!required_fields.empty()) {
        boost::json::array req_arr;
        for (auto& r : required_fields) req_arr.push_back(boost::json::value(r));
        result["required"] = std::move(req_arr);
    }
    return schema::JsonSchema{std::move(result)};
}

} // namespace detail

} // namespace ai

#define AI_FIELD(name, desc) \
    ai::detail::FieldDef{#name, desc, ai::schema::JsonSchema::string(desc), true}

#define AI_FIELD_NUMBER(name, desc) \
    ai::detail::FieldDef{#name, desc, ai::schema::JsonSchema::number(desc), true}

#define AI_FIELD_BOOL(name, desc) \
    ai::detail::FieldDef{#name, desc, ai::schema::JsonSchema::boolean(desc), true}

#define AI_FIELD_OPTIONAL(name, desc) \
    ai::detail::FieldDef{#name, desc, ai::schema::JsonSchema::string(desc), false}

#define AI_FIELD_OPTIONAL_NUMBER(name, desc) \
    ai::detail::FieldDef{#name, desc, ai::schema::JsonSchema::number(desc), false}

#define AI_FIELD_ENUM(name, desc, ...) \
    ai::detail::FieldDef{#name, desc, ai::schema::JsonSchema::enum_of({__VA_ARGS__}), true}

#define AI_FIELD_ARRAY(name, desc) \
    ai::detail::FieldDef{#name, desc, ai::schema::JsonSchema::array(ai::schema::JsonSchema::string()), true}

#define AI_SCHEMA(Type, ...) \
    template<> inline ai::schema::JsonSchema ai::schema_for<Type>() { \
        return ai::detail::build_schema({__VA_ARGS__}); \
    }
