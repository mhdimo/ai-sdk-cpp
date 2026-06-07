#include <ai/providers/bedrock/bedrock.hpp>
#include <ai/providers/bedrock/sigv4.hpp>
#include <cstdlib>
#include <stdexcept>

namespace ai::providers::bedrock {

namespace {

std::string env_or(const std::optional<std::string>& val, const char* env_var, const char* desc = nullptr) {
    if (val && !val->empty()) return *val;
    if (auto* e = std::getenv(env_var)) return e;
    if (desc) throw std::runtime_error(std::string("Missing ") + desc + ". Set " + env_var + " or pass it in options.");
    return "";
}

} // namespace

BedrockProvider::BedrockProvider(BedrockOptions options)
    : options_(std::move(options))
    , http_client_(options_.io_context) {
    resolved_bearer_token_ = env_or(options_.bearer_token, "AWS_BEARER_TOKEN_BEDROCK");
    use_bearer_token_ = !resolved_bearer_token_.empty();

    if (!use_bearer_token_) {
        resolved_region_ = env_or(options_.region, "AWS_REGION", "AWS region");
        resolved_access_key_id_ = env_or(options_.access_key_id, "AWS_ACCESS_KEY_ID", "AWS access key ID");
        resolved_secret_access_key_ = env_or(options_.secret_access_key, "AWS_SECRET_ACCESS_KEY", "AWS secret access key");
        resolved_session_token_ = env_or(options_.session_token, "AWS_SESSION_TOKEN");
    } else {
        resolved_region_ = env_or(options_.region, "AWS_REGION", "AWS region");
    }
}

std::string BedrockProvider::runtime_base_url() const {
    if (options_.base_url && !options_.base_url->empty()) return *options_.base_url;
    return "https://bedrock-runtime." + resolved_region_ + ".amazonaws.com";
}

http::Headers BedrockProvider::auth_headers(const std::string& url, const std::string& body) const {
    if (use_bearer_token_) {
        return {{"Authorization", "Bearer " + resolved_bearer_token_}};
    }

    AwsCredentials creds{
        .access_key_id = resolved_access_key_id_,
        .secret_access_key = resolved_secret_access_key_,
        .session_token = resolved_session_token_,
        .region = resolved_region_,
    };

    auto signed_req = sign_request(creds, "POST", url, {}, body, "bedrock");

    http::Headers result;
    for (auto& [k, v] : signed_req.headers) {
        result[k] = v;
    }
    return result;
}

// Forward declaration - implemented in bedrock_model.cpp
class BedrockLanguageModel;
LanguageModelPtr create_bedrock_model(std::string model_id, BedrockProvider& provider);

LanguageModelPtr BedrockProvider::language_model(std::string_view model_id) {
    return create_bedrock_model(std::string(model_id), *this);
}

ProviderPtr create_bedrock(BedrockOptions options) {
    return std::make_shared<BedrockProvider>(std::move(options));
}

} // namespace ai::providers::bedrock
