#include <ai/model/wrap_language_model.hpp>
#include <ai/model/middleware.hpp>
#include <ai/model/language_model.hpp>
#include <vector>

namespace ai {

namespace {

class MiddlewareWrappedModel : public LanguageModel {
public:
    MiddlewareWrappedModel(LanguageModelPtr inner, std::vector<MiddlewarePtr> middleware)
        : inner_(std::move(inner))
        , middleware_(std::move(middleware)) {}

    std::string_view provider() const override { return inner_->provider(); }
    std::string_view model_id() const override { return inner_->model_id(); }

    Task<GenerateResult> do_generate(CallOptions options) override {
        // Build the chain from innermost to outermost.
        // The innermost function calls the actual model.
        auto inner = inner_;
        Middleware::GenerateFn base_generate = [inner, opts = options]() mutable -> Task<GenerateResult> {
            co_return co_await inner->do_generate(std::move(opts));
        };
        Middleware::StreamFn base_stream = [inner, opts = options]() mutable -> Task<StreamResult> {
            co_return co_await inner->do_stream(std::move(opts));
        };

        // Apply middleware in reverse order so the first middleware in the list
        // is the outermost wrapper.
        Middleware::GenerateFn current_generate = std::move(base_generate);
        Middleware::StreamFn current_stream = std::move(base_stream);

        for (int i = static_cast<int>(middleware_.size()) - 1; i >= 0; --i) {
            auto mw = middleware_[i];
            auto prev_generate = std::move(current_generate);
            auto prev_stream = std::move(current_stream);

            current_generate = [mw, prev_generate, prev_stream, &options]() mutable -> Task<GenerateResult> {
                co_return co_await mw->wrap_generate(prev_generate, prev_stream, options);
            };
            current_stream = [mw, prev_generate, prev_stream, &options]() mutable -> Task<StreamResult> {
                co_return co_await mw->wrap_stream(prev_generate, prev_stream, options);
            };
        }

        co_return co_await current_generate();
    }

    Task<StreamResult> do_stream(CallOptions options) override {
        auto inner = inner_;
        Middleware::GenerateFn base_generate = [inner, opts = options]() mutable -> Task<GenerateResult> {
            co_return co_await inner->do_generate(std::move(opts));
        };
        Middleware::StreamFn base_stream = [inner, opts = options]() mutable -> Task<StreamResult> {
            co_return co_await inner->do_stream(std::move(opts));
        };

        Middleware::GenerateFn current_generate = std::move(base_generate);
        Middleware::StreamFn current_stream = std::move(base_stream);

        for (int i = static_cast<int>(middleware_.size()) - 1; i >= 0; --i) {
            auto mw = middleware_[i];
            auto prev_generate = std::move(current_generate);
            auto prev_stream = std::move(current_stream);

            current_generate = [mw, prev_generate, prev_stream, &options]() mutable -> Task<GenerateResult> {
                co_return co_await mw->wrap_generate(prev_generate, prev_stream, options);
            };
            current_stream = [mw, prev_generate, prev_stream, &options]() mutable -> Task<StreamResult> {
                co_return co_await mw->wrap_stream(prev_generate, prev_stream, options);
            };
        }

        co_return co_await current_stream();
    }

private:
    LanguageModelPtr inner_;
    std::vector<MiddlewarePtr> middleware_;
};

} // namespace

LanguageModelPtr wrap_language_model(LanguageModelPtr model, std::vector<MiddlewarePtr> middleware) {
    if (middleware.empty()) {
        return model;
    }
    return std::make_shared<MiddlewareWrappedModel>(std::move(model), std::move(middleware));
}

} // namespace ai
