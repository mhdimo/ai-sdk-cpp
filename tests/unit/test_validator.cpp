#include <catch2/catch_test_macros.hpp>
#include <ai/schema/json_schema.hpp>
#include <ai/schema/validator.hpp>
#include <ai/util/json.hpp>

using namespace ai::schema;

TEST_CASE("Validator validates string type", "[validator]") {
    auto schema = JsonSchema::string();
    Validator v(schema);

    auto r1 = v.validate(boost::json::value("hello"));
    REQUIRE(r1.success);

    auto r2 = v.validate(boost::json::value(42));
    REQUIRE_FALSE(r2.success);
    REQUIRE_FALSE(r2.errors.empty());
}

TEST_CASE("Validator validates number type", "[validator]") {
    auto schema = JsonSchema::number();
    Validator v(schema);

    REQUIRE(v.validate(boost::json::value(3.14)).success);
    REQUIRE(v.validate(boost::json::value(42)).success);
    REQUIRE_FALSE(v.validate(boost::json::value("not a number")).success);
}

TEST_CASE("Validator validates object with required fields", "[validator]") {
    auto schema = JsonSchema::object({
        {"name", JsonSchema::string()},
        {"age", JsonSchema::number()},
    }).required({"name", "age"});

    Validator v(schema);

    auto valid = ai::json::parse(R"({"name": "Alice", "age": 30})");
    REQUIRE(v.validate(valid).success);

    auto missing = ai::json::parse(R"({"name": "Bob"})");
    auto result = v.validate(missing);
    REQUIRE_FALSE(result.success);
}

TEST_CASE("Validator validates nested objects", "[validator]") {
    auto schema = JsonSchema::object({
        {"user", JsonSchema::object({
            {"name", JsonSchema::string()},
        }).required({"name"})},
    }).required({"user"});

    Validator v(schema);

    auto valid = ai::json::parse(R"({"user": {"name": "Alice"}})");
    REQUIRE(v.validate(valid).success);

    auto invalid = ai::json::parse(R"({"user": {"name": 42}})");
    REQUIRE_FALSE(v.validate(invalid).success);
}

TEST_CASE("Validator rejects additional properties", "[validator]") {
    auto schema = JsonSchema::object({
        {"name", JsonSchema::string()},
    }).required({"name"}).additional_properties(false);

    Validator v(schema);

    auto valid = ai::json::parse(R"({"name": "Alice"})");
    REQUIRE(v.validate(valid).success);

    auto extra = ai::json::parse(R"({"name": "Alice", "extra": true})");
    REQUIRE_FALSE(v.validate(extra).success);
}

TEST_CASE("Validator validates arrays", "[validator]") {
    auto schema = JsonSchema::array(JsonSchema::string()).min_items(1).max_items(3);
    Validator v(schema);

    auto valid = ai::json::parse(R"(["a", "b"])");
    REQUIRE(v.validate(valid).success);

    auto empty = ai::json::parse(R"([])");
    REQUIRE_FALSE(v.validate(empty).success);

    auto too_many = ai::json::parse(R"(["a", "b", "c", "d"])");
    REQUIRE_FALSE(v.validate(too_many).success);
}

TEST_CASE("Validator validates enum values", "[validator]") {
    auto schema = JsonSchema::enum_of({"red", "green", "blue"});
    Validator v(schema);

    REQUIRE(v.validate(boost::json::value("red")).success);
    REQUIRE(v.validate(boost::json::value("green")).success);
    REQUIRE_FALSE(v.validate(boost::json::value("yellow")).success);
}
