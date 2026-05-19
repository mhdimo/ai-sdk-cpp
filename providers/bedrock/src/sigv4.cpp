#include <ai/providers/bedrock/sigv4.hpp>
#include <openssl/hmac.h>
#include <openssl/evp.h>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <cctype>

namespace ai::providers::bedrock {

namespace {

std::string sha256_hex(const std::string& data) {
    unsigned char hash[EVP_MAX_MD_SIZE];
    unsigned int len = 0;
    EVP_MD_CTX* ctx = EVP_MD_CTX_new();
    EVP_DigestInit_ex(ctx, EVP_sha256(), nullptr);
    EVP_DigestUpdate(ctx, data.data(), data.size());
    EVP_DigestFinal_ex(ctx, hash, &len);
    EVP_MD_CTX_free(ctx);

    std::ostringstream ss;
    for (unsigned int i = 0; i < len; ++i)
        ss << std::hex << std::setfill('0') << std::setw(2) << (int)hash[i];
    return ss.str();
}

std::vector<unsigned char> hmac_sha256(const std::vector<unsigned char>& key, const std::string& data) {
    unsigned char result[EVP_MAX_MD_SIZE];
    unsigned int len = 0;
    HMAC(EVP_sha256(), key.data(), (int)key.size(),
         reinterpret_cast<const unsigned char*>(data.data()), data.size(),
         result, &len);
    return std::vector<unsigned char>(result, result + len);
}

std::vector<unsigned char> hmac_sha256(const std::string& key, const std::string& data) {
    std::vector<unsigned char> k(key.begin(), key.end());
    return hmac_sha256(k, data);
}

std::string hmac_sha256_hex(const std::vector<unsigned char>& key, const std::string& data) {
    auto raw = hmac_sha256(key, data);
    std::ostringstream ss;
    for (auto b : raw)
        ss << std::hex << std::setfill('0') << std::setw(2) << (int)b;
    return ss.str();
}

std::string uri_encode(const std::string& s, bool encode_slash = true) {
    std::ostringstream result;
    for (unsigned char c : s) {
        if (std::isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~')
            result << c;
        else if (c == '/' && !encode_slash)
            result << '/';
        else
            result << '%' << std::uppercase << std::hex << std::setfill('0') << std::setw(2) << (int)c;
    }
    return result.str();
}

struct ParsedUrl {
    std::string host;
    std::string path;
    std::string query;
};

ParsedUrl parse_url(const std::string& url) {
    ParsedUrl result;
    auto scheme_end = url.find("://");
    size_t host_start = (scheme_end != std::string::npos) ? scheme_end + 3 : 0;
    auto path_start = url.find('/', host_start);
    if (path_start == std::string::npos) {
        result.host = url.substr(host_start);
        result.path = "/";
    } else {
        result.host = url.substr(host_start, path_start - host_start);
        auto query_start = url.find('?', path_start);
        if (query_start == std::string::npos) {
            result.path = url.substr(path_start);
        } else {
            result.path = url.substr(path_start, query_start - path_start);
            result.query = url.substr(query_start + 1);
        }
    }
    return result;
}

std::string get_timestamp() {
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
    gmtime_r(&time, &tm);
    std::ostringstream ss;
    ss << std::put_time(&tm, "%Y%m%dT%H%M%SZ");
    return ss.str();
}

std::string get_date(const std::string& timestamp) {
    return timestamp.substr(0, 8);
}

} // namespace

SignedRequest sign_request(
    const AwsCredentials& creds,
    const std::string& method,
    const std::string& url,
    const std::map<std::string, std::string>& extra_headers,
    const std::string& body,
    const std::string& service
) {
    auto parsed = parse_url(url);
    auto timestamp = get_timestamp();
    auto date = get_date(timestamp);

    std::map<std::string, std::string> headers = extra_headers;
    headers["host"] = parsed.host;
    headers["x-amz-date"] = timestamp;
    headers["content-type"] = "application/json";
    if (!creds.session_token.empty()) {
        headers["x-amz-security-token"] = creds.session_token;
    }

    // Canonical headers (sorted, lowercase)
    std::string canonical_headers;
    std::string signed_headers;
    for (auto& [k, v] : headers) {
        std::string lower_key = k;
        std::transform(lower_key.begin(), lower_key.end(), lower_key.begin(), ::tolower);
        canonical_headers += lower_key + ":" + v + "\n";
        if (!signed_headers.empty()) signed_headers += ";";
        signed_headers += lower_key;
    }

    auto payload_hash = sha256_hex(body);

    // Canonical request
    std::string canonical_request =
        method + "\n" +
        uri_encode(parsed.path, false) + "\n" +
        parsed.query + "\n" +
        canonical_headers + "\n" +
        signed_headers + "\n" +
        payload_hash;

    // String to sign
    std::string credential_scope = date + "/" + creds.region + "/" + service + "/aws4_request";
    std::string string_to_sign =
        "AWS4-HMAC-SHA256\n" +
        timestamp + "\n" +
        credential_scope + "\n" +
        sha256_hex(canonical_request);

    // Signing key
    auto k_date = hmac_sha256("AWS4" + creds.secret_access_key, date);
    auto k_region = hmac_sha256(k_date, creds.region);
    auto k_service = hmac_sha256(k_region, service);
    auto k_signing = hmac_sha256(k_service, "aws4_request");

    auto signature = hmac_sha256_hex(k_signing, string_to_sign);

    std::string authorization =
        "AWS4-HMAC-SHA256 Credential=" + creds.access_key_id + "/" + credential_scope +
        ", SignedHeaders=" + signed_headers +
        ", Signature=" + signature;

    SignedRequest result;
    result.headers = headers;
    result.headers["authorization"] = authorization;
    return result;
}

} // namespace ai::providers::bedrock
