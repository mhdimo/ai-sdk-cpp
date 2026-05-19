#include <ai/providers/openai/openai_embedding.hpp>
#include <ai/providers/openai/openai.hpp>
#include <ai/http/client.hpp>
#include <ai/error/api_call_error.hpp>
#include <boost/json.hpp>

namespace ai::providers::openai {

OpenAIEmbeddingModel::OpenAIEmbeddingModel(std::string model_id, OpenAIProvider& provider)
    : model_id_(std::move(model_id)), provider_(provider) {}

Task<EmbedResult> OpenAIEmbeddingModel::do_embed(EmbedOptions options) {
    namespace json = boost::json;

    json::array input_array;
    for (const auto& value : options.values) {
        input_array.push_back(json::string(value));
    }

    json::object request_body;
    request_body["model"] = model_id_;
    request_body["input"] = std::move(input_array);
    request_body["encoding_format"] = "float";

    auto url = provider_.options().base_url + "/embeddings";
    auto headers = provider_.auth_headers();

    auto response = co_await provider_.http_client().post_json(
        url, json::value(std::move(request_body)), headers, options.cancel
    );

    auto parsed = json::parse(response.body);
    auto& obj = parsed.as_object();

    EmbedResult result;

    // Parse embeddings from data array
    auto& data = obj.at("data").as_array();
    result.embeddings.reserve(data.size());

    for (const auto& item : data) {
        auto& embedding_array = item.as_object().at("embedding").as_array();
        std::vector<float> embedding;
        embedding.reserve(embedding_array.size());

        for (const auto& val : embedding_array) {
            embedding.push_back(static_cast<float>(val.as_double()));
        }

        result.embeddings.push_back(std::move(embedding));
    }

    // Parse usage
    if (auto it = obj.find("usage"); it != obj.end()) {
        auto& usage_obj = it->value().as_object();
        if (auto pt = usage_obj.find("prompt_tokens"); pt != usage_obj.end()) {
            result.usage.input_tokens.total = static_cast<int>(pt->value().as_int64());
        }
        if (auto tt = usage_obj.find("total_tokens"); tt != usage_obj.end()) {
            auto total = static_cast<int>(tt->value().as_int64());
            auto prompt = result.usage.input_tokens.total.value_or(0);
            result.usage.output_tokens.total = total - prompt;
        }
    }

    co_return std::move(result);
}

// OpenAI file upload implementation
OpenAIFileStorage::OpenAIFileStorage(OpenAIProvider& provider)
    : provider_(provider) {}

Task<FileUploadResult> OpenAIFileStorage::upload(FileUploadOptions options) {
    namespace json = boost::json;

    http::MultipartFormData form;
    form.add_field("purpose", options.purpose.value_or("assistants"));
    form.add_file("file", options.filename, options.content_type, std::move(options.data));

    auto url = provider_.options().base_url + "/files";
    auto headers = provider_.auth_headers();

    auto response = co_await provider_.http_client().post_multipart(
        url, std::move(form), headers, options.cancel
    );

    auto parsed = json::parse(response.body);
    auto& obj = parsed.as_object();

    FileUploadResult result;
    result.file_id = std::string(obj.at("id").as_string());
    result.filename = std::string(obj.at("filename").as_string());
    result.bytes = static_cast<int>(obj.at("bytes").as_int64());

    if (auto it = obj.find("purpose"); it != obj.end()) {
        result.purpose = std::string(it->value().as_string());
    }
    if (auto it = obj.find("status"); it != obj.end()) {
        result.status = std::string(it->value().as_string());
    }

    co_return std::move(result);
}

} // namespace ai::providers::openai
