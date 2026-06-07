#include <catch2/catch_test_macros.hpp>
#include <ai/schema/json_schema.hpp>

using namespace ai::schema;

TEST_CASE("JsonSchema string type", "[schema]") {
    auto schema = JsonSchema::string("A name");
    auto& raw = schema.raw();
    REQUIRE(raw.at("type").as_string() == "string");
    REQUIRE(raw.at("description").as_string() == "A name");
}

TEST_CASE("JsonSchema number type", "[schema]") {
    auto schema = JsonSchema::number();
    REQUIRE(schema.raw().at("type").as_string() == "number");
}

TEST_CASE("JsonSchema object type", "[schema]") {
    auto schema = JsonSchema::object({
        {"name", JsonSchema::string("User name")},
        {"age", JsonSchema::integer("User age")},
    }).required({"name", "age"}).additional_properties(false);

    auto& raw = schema.raw();
    REQUIRE(raw.at("type").as_string() == "object");
    REQUIRE(raw.at("additionalProperties").as_bool() == false);

    auto& props = raw.at("properties").as_object();
    REQUIRE(props.at("name").as_object().at("type").as_string() == "string");
    REQUIRE(props.at("age").as_object().at("type").as_string() == "integer");

    auto& req = raw.at("required").as_array();
    REQUIRE(req.size() == 2);
}

TEST_CASE("JsonSchema array type", "[schema]") {
    auto schema = JsonSchema::array(JsonSchema::string()).min_items(1).max_items(10);
    auto& raw = schema.raw();
    REQUIRE(raw.at("type").as_string() == "array");
    REQUIRE(raw.at("items").as_object().at("type").as_string() == "string");
    REQUIRE(raw.at("minItems").as_int64() == 1);
    REQUIRE(raw.at("maxItems").as_int64() == 10);
}

TEST_CASE("JsonSchema enum type", "[schema]") {
    auto schema = JsonSchema::enum_of({"red", "green", "blue"});
    auto& raw = schema.raw();
    REQUIRE(raw.at("enum").as_array().size() == 3);
}
