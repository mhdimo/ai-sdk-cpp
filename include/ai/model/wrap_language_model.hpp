#pragma once

#include <ai/model/language_model.hpp>
#include <ai/model/middleware.hpp>
#include <vector>

namespace ai {

/// Wraps a language model with a stack of middleware.
/// Middleware is applied in order: the first middleware in the vector is the
/// outermost wrapper (called first on request, last on response).
LanguageModelPtr wrap_language_model(LanguageModelPtr model, std::vector<MiddlewarePtr> middleware);

} // namespace ai
