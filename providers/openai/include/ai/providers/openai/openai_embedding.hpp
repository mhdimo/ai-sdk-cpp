#pragma once

#include <ai/model/embedding_model.hpp>
#include <ai/model/file_storage.hpp>
#include <ai/http/client.hpp>
#include <string>

namespace ai::providers::openai {

class OpenAIProvider;

class OpenAIEmbeddingModel : public EmbeddingModel {
public:
    OpenAIEmbeddingModel(std::string model_id, OpenAIProvider& provider);

    std::string_view provider() const override { return "openai"; }
    std::string_view model_id() const override { return model_id_; }

    Task<EmbedResult> do_embed(EmbedOptions options) override;

private:
    std::string model_id_;
    OpenAIProvider& provider_;
};

class OpenAIFileStorage : public FileStorage {
public:
    explicit OpenAIFileStorage(OpenAIProvider& provider);

    std::string_view provider() const override { return "openai"; }
    Task<FileUploadResult> upload(FileUploadOptions options) override;

private:
    OpenAIProvider& provider_;
};

} // namespace ai::providers::openai
