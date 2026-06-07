#include <ai/core/embed.hpp>
#include <ai/model/embedding_model.hpp>
#include <stdexcept>

namespace ai {

Task<SingleEmbedResult> embed(SingleEmbedOptions options) {
    if (!options.model) {
        throw std::invalid_argument("embed: model is required");
    }

    if (options.value.empty()) {
        throw std::invalid_argument("embed: value must not be empty");
    }

    EmbedOptions embed_opts;
    embed_opts.values = {options.value};
    embed_opts.cancel = std::move(options.cancel);

    auto result = co_await options.model->do_embed(std::move(embed_opts));

    if (result.embeddings.empty()) {
        throw std::runtime_error("embed: model returned no embeddings");
    }

    SingleEmbedResult single_result;
    single_result.embedding = std::move(result.embeddings[0]);
    single_result.usage = std::move(result.usage);

    co_return std::move(single_result);
}

Task<EmbedManyResult> embed_many(EmbedManyOptions options) {
    if (!options.model) {
        throw std::invalid_argument("embed_many: model is required");
    }

    if (options.values.empty()) {
        throw std::invalid_argument("embed_many: values must not be empty");
    }

    EmbedOptions embed_opts;
    embed_opts.values = std::move(options.values);
    embed_opts.cancel = std::move(options.cancel);

    auto result = co_await options.model->do_embed(std::move(embed_opts));

    EmbedManyResult many_result;
    many_result.embeddings = std::move(result.embeddings);
    many_result.usage = std::move(result.usage);

    co_return std::move(many_result);
}

} // namespace ai
