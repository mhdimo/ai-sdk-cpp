#include <catch2/catch_test_macros.hpp>
#include <ai/util/cancellation.hpp>

using namespace ai;

TEST_CASE("CancellationToken starts uncancelled", "[cancel]") {
    CancellationSource source;
    auto token = source.token();
    REQUIRE_FALSE(token.is_cancelled());
}

TEST_CASE("CancellationToken reflects cancel", "[cancel]") {
    CancellationSource source;
    auto token = source.token();

    source.cancel();
    REQUIRE(token.is_cancelled());
}

TEST_CASE("throw_if_cancelled throws when cancelled", "[cancel]") {
    CancellationSource source;
    auto token = source.token();

    REQUIRE_NOTHROW(token.throw_if_cancelled());

    source.cancel();
    REQUIRE_THROWS_AS(token.throw_if_cancelled(), OperationCancelled);
}

TEST_CASE("on_cancel callback fires on cancel", "[cancel]") {
    CancellationSource source;
    auto token = source.token();

    bool called = false;
    token.on_cancel([&]() { called = true; });

    REQUIRE_FALSE(called);
    source.cancel();
    REQUIRE(called);
}

TEST_CASE("on_cancel callback fires immediately if already cancelled", "[cancel]") {
    CancellationSource source;
    auto token = source.token();
    source.cancel();

    bool called = false;
    token.on_cancel([&]() { called = true; });
    REQUIRE(called);
}

TEST_CASE("Multiple tokens from same source", "[cancel]") {
    CancellationSource source;
    auto t1 = source.token();
    auto t2 = source.token();

    REQUIRE_FALSE(t1.is_cancelled());
    REQUIRE_FALSE(t2.is_cancelled());

    source.cancel();
    REQUIRE(t1.is_cancelled());
    REQUIRE(t2.is_cancelled());
}

TEST_CASE("Default token is never cancelled", "[cancel]") {
    CancellationToken token;
    REQUIRE_FALSE(token.is_cancelled());
    REQUIRE_NOTHROW(token.throw_if_cancelled());
}
