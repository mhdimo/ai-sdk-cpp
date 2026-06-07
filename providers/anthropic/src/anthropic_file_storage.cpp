#include <ai/providers/anthropic/anthropic_file_storage.hpp>
#include <ai/providers/anthropic/anthropic.hpp>
#include <ai/http/multipart.hpp>
#include <ai/error/api_call_error.hpp>
#include <boost/json.hpp>

namespace ai::providers::anthropic {

AnthropicFileStorage::AnthropicFileStorage(AnthropicProvider& provider)
    : provider_(provider) {}

Task<FileUploadResult> AnthropicFileStorage::upload(FileUploadOptions options) {
    namespace json = boost::json;

    http::MultipartFormData form;
    form.add_file("file", options.filename, options.content_type, std::move(options.data));

    if (options.purpose) {
        form.add_field("purpose", *options.purpose);
    }

    auto url = provider_.options().base_url + "/v1/files";
    auto headers = provider_.auth_headers();

    // Anthropic requires the beta header for file uploads
    headers["anthropic-beta"] = "files-api-2025-04-14";

    auto response = co_await provider_.http_client().post_multipart(
        url, std::move(form), headers, options.cancel
    );

    auto parsed = json::parse(response.body);
    auto& obj = parsed.as_object();

    FileUploadResult result;
    result.file_id = std::string(obj.at("id").as_string());
    result.filename = std::string(obj.at("filename").as_string());

    if (auto it = obj.find("size_bytes"); it != obj.end()) {
        result.bytes = static_cast<int>(it->value().as_int64());
    }

    if (auto it = obj.find("purpose"); it != obj.end()) {
        result.purpose = std::string(it->value().as_string());
    }

    if (auto it = obj.find("status"); it != obj.end()) {
        result.status = std::string(it->value().as_string());
    }

    co_return std::move(result);
}

} // namespace ai::providers::anthropic
