#pragma once

#include <string>
#include <map>

namespace ai::providers::bedrock {

struct AwsCredentials {
    std::string access_key_id;
    std::string secret_access_key;
    std::string session_token;
    std::string region;
};

struct SignedRequest {
    std::map<std::string, std::string> headers;
};

SignedRequest sign_request(
    const AwsCredentials& creds,
    const std::string& method,
    const std::string& url,
    const std::map<std::string, std::string>& headers,
    const std::string& body,
    const std::string& service = "bedrock"
);

} // namespace ai::providers::bedrock
