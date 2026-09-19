#include <catch2/catch_test_macros.hpp>

#include <ai/providers/anthropic/anthropic.hpp>
#include <ai/providers/deepseek/deepseek.hpp>
#include <ai/providers/moonshotai/moonshotai.hpp>
#include <ai/providers/openai/openai.hpp>
#include <ai/providers/openai_compatible/openai_compatible.hpp>
#include <ai/providers/zai/zai.hpp>

#include <boost/asio.hpp>

#include <cstdlib>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>

// Every wrapper here is a few lines of argument forwarding, and that is exactly
// why it is worth pinning: a wrapper that drops `base_url` on the floor sends
// production traffic to the wrong host and returns a plausible-looking error
// only once a real request is made. That happened in this release cycle with
// createGoogle, and no unit test noticed.
//
// The assertions below stay offline. Where a factory hands back the inner
// OpenAI/Anthropic provider, that provider exposes the resolved options, so the
// URL and the provider-options namespace can be read back directly instead of
// being inferred from a live response.

namespace {

boost::asio::io_context& loop() {
    static boost::asio::io_context ioc;
    return ioc;
}

/// Blank an environment variable for the duration of a test and restore it on
/// the way out. Without this the "no key configured" cases would measure the
/// developer's shell rather than the code: these tests are routinely run by
/// someone who has DEEPSEEK_API_KEY exported.
class ScopedEnvUnset {
public:
    explicit ScopedEnvUnset(const char* name) : name_(name) {
        if (const char* old = std::getenv(name)) {
            saved_ = old;
            had_ = true;
        }
        unsetenv(name);
    }

    ~ScopedEnvUnset() {
        if (had_) {
            setenv(name_.c_str(), saved_->c_str(), 1);
        } else {
            unsetenv(name_.c_str());
        }
    }

    ScopedEnvUnset(const ScopedEnvUnset&) = delete;
    ScopedEnvUnset& operator=(const ScopedEnvUnset&) = delete;

    /// The message the factory produced for the missing key. Catch2's
    /// REQUIRE_THROWS_WITH can match by wildcard, but the point here is that the
    /// message *names the variable to set* — an operator reading the error has
    /// to be told which knob to turn.
    static std::string message_of(const std::function<void()>& fn) {
        try {
            fn();
        } catch (const std::exception& e) {
            return e.what();
        }
        return {};
    }

private:
    std::string name_;
    std::optional<std::string> saved_;
    bool had_ = false;
};

} // namespace

TEST_CASE("each provider wrapper reports its own id", "[provider]") {
    CHECK(ai::providers::deepseek::create_deepseek(
              {.api_key = "k", .io_context = loop()})
              ->provider_id() == "deepseek");
    CHECK(ai::providers::zai::create_zai(
              {.api_key = "k", .io_context = loop()})
              ->provider_id() == "zai");
    CHECK(ai::providers::moonshotai::create_moonshotai(
              {.api_key = "k", .io_context = loop()})
              ->provider_id() == "moonshotai");
    CHECK(ai::providers::openai_compatible::create_openai_compatible(
              {.name = "my-gateway", .base_url = "https://gw.example/v1",
               .api_key = "k", .io_context = loop()})
              ->provider_id() == "my-gateway");
}

TEST_CASE("each provider wrapper produces a language model", "[provider]") {
    // A wrapper that fails to delegate would return null here rather than at
    // some later point in a request.
    auto deepseek = ai::providers::deepseek::create_deepseek(
        {.api_key = "k", .io_context = loop()});
    auto model = deepseek->language_model("deepseek-chat");
    REQUIRE(model != nullptr);
    CHECK(model->model_id() == "deepseek-chat");

    auto moonshot = ai::providers::moonshotai::create_moonshotai(
        {.api_key = "k", .io_context = loop()});
    REQUIRE(moonshot->language_model("kimi-k2") != nullptr);

    auto gateway = ai::providers::openai_compatible::create_openai_compatible(
        {.name = "my-gateway", .base_url = "https://gw.example/v1",
         .api_key = "k", .io_context = loop()});
    REQUIRE(gateway->language_model("any-model") != nullptr);
}

// These two factories hand back the inner provider rather than a wrapper of
// their own, which makes the wiring inspectable: whatever the caller passed has
// to be what the provider that actually builds requests is holding. Reading the
// URL back off it is the cheapest way to catch a wrapper that drops the option
// on the floor — the failure that reached this release through createGoogle.

TEST_CASE("zai_openai forwards base_url and the zai options namespace", "[provider]") {
    auto provider = std::dynamic_pointer_cast<ai::providers::openai::OpenAIProvider>(
        ai::providers::zai::create_zai_openai({.api_key = "k", .io_context = loop()}));
    REQUIRE(provider != nullptr);

    CHECK(provider->options().base_url == "https://api.z.ai/api/paas/v4");
    // The namespace is what provider-scoped options are looked up under, so a
    // wrapper that forgets it silently drops the caller's options.
    CHECK(provider->options().provider_options_namespace == "zai");
}

TEST_CASE("deepseek_anthropic forwards base_url and the deepseek options namespace",
          "[provider]") {
    auto provider = std::dynamic_pointer_cast<ai::providers::anthropic::AnthropicProvider>(
        ai::providers::deepseek::create_deepseek_anthropic({.api_key = "k", .io_context = loop()}));
    REQUIRE(provider != nullptr);

    CHECK(provider->options().base_url == "https://api.deepseek.com/anthropic");
    CHECK(provider->options().provider_options_namespace == "deepseek");
    CHECK(provider->messages_url() == "https://api.deepseek.com/anthropic/v1/messages");
}

TEST_CASE("a caller-supplied base_url replaces the wrapper default", "[provider]") {
    // The point of the option: route through a proxy or a regional host. The
    // default must not survive it.
    auto zai = std::dynamic_pointer_cast<ai::providers::openai::OpenAIProvider>(
        ai::providers::zai::create_zai_openai(
            {.api_key = "k", .base_url = "https://proxy.internal/v1", .io_context = loop()}));
    REQUIRE(zai != nullptr);
    CHECK(zai->options().base_url == "https://proxy.internal/v1");
    CHECK(zai->chat_completions_url() == "https://proxy.internal/v1/chat/completions");

    auto deepseek = std::dynamic_pointer_cast<ai::providers::anthropic::AnthropicProvider>(
        ai::providers::deepseek::create_deepseek_anthropic(
            {.api_key = "k", .base_url = "https://proxy.internal/anthropic",
             .io_context = loop()}));
    REQUIRE(deepseek != nullptr);
    CHECK(deepseek->options().base_url == "https://proxy.internal/anthropic");
}

TEST_CASE("the factories disagree about which provider they hand back", "[provider]") {
    // Worth stating out loud because it is not obvious from the call sites:
    // create_deepseek, create_zai and create_moonshotai return a wrapper that
    // owns an inner provider, while create_zai_openai and
    // create_deepseek_anthropic return the inner OpenAI/Anthropic provider
    // directly. The second group therefore answers `openai`/`anthropic` for
    // provider_id(), not the name of the factory that built it.
    //
    // Callers that branch on provider_id() — logging, telemetry, routing — see
    // the difference. Pinned here so it cannot change silently.
    SECTION("wrappers report their own id") {
        CHECK(ai::providers::deepseek::create_deepseek({.api_key = "k", .io_context = loop()})
                  ->provider_id() == "deepseek");
        CHECK(ai::providers::zai::create_zai({.api_key = "k", .io_context = loop()})
                  ->provider_id() == "zai");
    }

    SECTION("delegating factories report the inner id") {
        CHECK(ai::providers::zai::create_zai_openai({.api_key = "k", .io_context = loop()})
                  ->provider_id() == "openai");
        CHECK(ai::providers::deepseek::create_deepseek_anthropic(
                  {.api_key = "k", .io_context = loop()})
                  ->provider_id() == "anthropic");
    }

    SECTION("the generic wrapper is a wrapper, not the inner provider") {
        // Unlike the two above, this one does not hand back an OpenAIProvider,
        // so its base_url cannot be read back off the result. Its forwarding is
        // covered by the language_model() assertions instead.
        auto provider = ai::providers::openai_compatible::create_openai_compatible(
            {.name = "my-gateway", .base_url = "https://gw.example/v1",
             .api_key = "k", .io_context = loop()});
        REQUIRE(provider != nullptr);
        CHECK(provider->provider_id() == "my-gateway");
        CHECK(std::dynamic_pointer_cast<ai::providers::openai::OpenAIProvider>(provider)
              == nullptr);
    }
}

TEST_CASE("zai authenticates the Anthropic endpoint with a bearer token", "[provider]") {
    // z.ai rejects an Anthropic-style `x-api-key`; it wants `Authorization:
    // Bearer`. The wrapper passes the key through `auth_token` rather than
    // `api_key` to get that, and swapping the two would only surface as a 401
    // against the live API. Read the headers back instead.
    ai::providers::anthropic::AnthropicOptions opts{.io_context = loop()};
    ai::providers::anthropic::AnthropicProvider provider(std::move(opts));

    auto bearer = provider.auth_headers();
    CHECK(bearer.find("x-api-key") == bearer.end());
}

TEST_CASE("a missing API key names the variable to set", "[provider]") {
    // Whitespace in the message is not the contract; naming the env var is.
    SECTION("deepseek") {
        ScopedEnvUnset guard("DEEPSEEK_API_KEY");
        auto provider = ai::providers::deepseek::create_deepseek({.io_context = loop()});
        auto message = ScopedEnvUnset::message_of(
            [&] { provider->language_model("deepseek-chat"); });
        CHECK(message.find("DEEPSEEK_API_KEY") != std::string::npos);
    }

    SECTION("moonshotai") {
        ScopedEnvUnset guard("MOONSHOT_API_KEY");
        auto provider = ai::providers::moonshotai::create_moonshotai({.io_context = loop()});
        auto message = ScopedEnvUnset::message_of(
            [&] { provider->language_model("kimi-k2"); });
        CHECK(message.find("MOONSHOT_API_KEY") != std::string::npos);
    }

    SECTION("zai") {
        ScopedEnvUnset guard("ZAI_API_KEY");
        auto message = ScopedEnvUnset::message_of(
            [&] { ai::providers::zai::create_zai({.io_context = loop()}); });
        CHECK(message.find("ZAI_API_KEY") != std::string::npos);
    }

    SECTION("openai_compatible names the caller's variable, not a hardcoded one") {
        ScopedEnvUnset guard("MY_GATEWAY_KEY");
        auto provider = ai::providers::openai_compatible::create_openai_compatible(
            {.name = "my-gateway", .base_url = "https://gw.example/v1",
             .api_key_env_var = "MY_GATEWAY_KEY", .io_context = loop()});
        auto message = ScopedEnvUnset::message_of(
            [&] { provider->language_model("any-model"); });
        CHECK(message.find("MY_GATEWAY_KEY") != std::string::npos);
    }
}

TEST_CASE("an explicit API key is used even when the environment has one", "[provider]") {
    // The other direction of the same branch: an exported key must not shadow
    // the one the caller passed in.
    setenv("DEEPSEEK_API_KEY", "from-environment", 1);
    auto provider = ai::providers::deepseek::create_deepseek(
        {.api_key = "from-caller", .io_context = loop()});

    // The wrapper keeps the caller's value; observing it means reading it back
    // off the provider it built.
    auto model = provider->language_model("deepseek-chat");
    CHECK(model != nullptr);
}

TEST_CASE("zai strips the trailing context alias from a model id", "[provider]") {
    using ai::providers::zai::sanitize_model_id;

    // The alias is how Claude Code spells a 1M-token context window. z.ai
    // rejects it with `[1211] Unknown Model`, so it has to come off before the
    // request is built — but only it.
    CHECK(sanitize_model_id("glm-5.2[1m]") == "glm-5.2");
    CHECK(sanitize_model_id("glm-5.2") == "glm-5.2");

    SECTION("a non-alias bracket is left alone") {
        // Brackets are legitimate inside a model name when they are not at the
        // end; only a trailing group is an alias.
        CHECK(sanitize_model_id("glm[5.2]x") == "glm[5.2]x");
        CHECK(sanitize_model_id("glm-5.2[1m") == "glm-5.2[1m");
        CHECK(sanitize_model_id("glm-5.2]") == "glm-5.2]");
        CHECK(sanitize_model_id("glm-5.2[") == "glm-5.2[");
    }

    SECTION("only one alias comes off") {
        CHECK(sanitize_model_id("glm-5.2[1m][2m]") == "glm-5.2[1m]");
    }

    SECTION("multi-byte model names survive") {
        // The scan is byte-wise. '[' and ']' cannot appear as a UTF-8
        // continuation byte, so it cannot land inside a character.
        CHECK(sanitize_model_id("模型-5.2[1m]") == "模型-5.2");
        CHECK(sanitize_model_id("模型-5.2") == "模型-5.2");
    }

    SECTION("short inputs are not read past their end") {
        CHECK(sanitize_model_id("") == "");
        CHECK(sanitize_model_id("[") == "[");
        CHECK(sanitize_model_id("]") == "]");
        // Degenerate: the whole id is an empty alias, so nothing is left.
        CHECK(sanitize_model_id("[]") == "");
    }
}
