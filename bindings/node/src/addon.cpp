#include <napi.h>
#include "ai_sdk.h"
#include <string>

class ContextWrapper : public Napi::ObjectWrap<ContextWrapper> {
public:
    static Napi::Object Init(Napi::Env env, Napi::Object exports);
    ContextWrapper(const Napi::CallbackInfo& info);
    ~ContextWrapper();
    ai_context_t handle() const { return ctx_; }

private:
    ai_context_t ctx_;
};

class ProviderWrapper : public Napi::ObjectWrap<ProviderWrapper> {
public:
    static Napi::Object Init(Napi::Env env, Napi::Object exports);
    ProviderWrapper(const Napi::CallbackInfo& info);
    ~ProviderWrapper();
    ai_provider_t handle() const { return provider_; }

private:
    ai_provider_t provider_;
};

class ModelWrapper : public Napi::ObjectWrap<ModelWrapper> {
public:
    static Napi::Object Init(Napi::Env env, Napi::Object exports);
    ModelWrapper(const Napi::CallbackInfo& info);
    ~ModelWrapper();
    ai_model_t handle() const { return model_; }

private:
    ai_model_t model_;
};

class ToolSetWrapper : public Napi::ObjectWrap<ToolSetWrapper> {
public:
    static Napi::Object Init(Napi::Env env, Napi::Object exports);
    ToolSetWrapper(const Napi::CallbackInfo& info);
    ~ToolSetWrapper();
    ai_tool_set_t handle() const { return tools_; }
    void AddTool(const Napi::CallbackInfo& info);

private:
    ai_tool_set_t tools_;
};

class AgentWrapper : public Napi::ObjectWrap<AgentWrapper> {
public:
    static Napi::Object Init(Napi::Env env, Napi::Object exports);
    AgentWrapper(const Napi::CallbackInfo& info);
    ~AgentWrapper();
    Napi::Value Call(const Napi::CallbackInfo& info);

private:
    ai_agent_t agent_;
};

// --- ContextWrapper ---

Napi::Object ContextWrapper::Init(Napi::Env env, Napi::Object exports) {
    Napi::Function func = DefineClass(env, "Context", {});
    exports.Set("Context", func);
    return exports;
}

ContextWrapper::ContextWrapper(const Napi::CallbackInfo& info)
    : Napi::ObjectWrap<ContextWrapper>(info) {
    ctx_ = ai_context_create();
    if (!ctx_) {
        Napi::Error::New(info.Env(), "Failed to create AI context").ThrowAsJavaScriptException();
    }
}

ContextWrapper::~ContextWrapper() {
    if (ctx_) ai_context_destroy(ctx_);
}

// --- ProviderWrapper ---

Napi::Object ProviderWrapper::Init(Napi::Env env, Napi::Object exports) {
    Napi::Function func = DefineClass(env, "Provider", {});
    exports.Set("Provider", func);
    return exports;
}

ProviderWrapper::ProviderWrapper(const Napi::CallbackInfo& info)
    : Napi::ObjectWrap<ProviderWrapper>(info) {
    Napi::Env env = info.Env();

    if (info.Length() < 2) {
        Napi::TypeError::New(env, "Expected (context, name, apiKey?, baseUrl?)").ThrowAsJavaScriptException();
        return;
    }

    auto ctx_obj = info[0].As<Napi::Object>();
    auto* ctx_wrap = Napi::ObjectWrap<ContextWrapper>::Unwrap(ctx_obj);
    std::string name = info[1].As<Napi::String>().Utf8Value();

    ai_provider_options_t opts = {};
    std::string api_key_str, base_url_str;

    if (info.Length() > 2 && !info[2].IsNull() && !info[2].IsUndefined()) {
        api_key_str = info[2].As<Napi::String>().Utf8Value();
        opts.api_key = api_key_str.c_str();
    }
    if (info.Length() > 3 && !info[3].IsNull() && !info[3].IsUndefined()) {
        base_url_str = info[3].As<Napi::String>().Utf8Value();
        opts.base_url = base_url_str.c_str();
    }

    provider_ = ai_provider_create(ctx_wrap->handle(), name.c_str(), opts);
    if (!provider_) {
        Napi::Error::New(env, "Failed to create provider").ThrowAsJavaScriptException();
    }
}

ProviderWrapper::~ProviderWrapper() {
    if (provider_) ai_provider_destroy(provider_);
}

// --- ModelWrapper ---

Napi::Object ModelWrapper::Init(Napi::Env env, Napi::Object exports) {
    Napi::Function func = DefineClass(env, "Model", {});
    exports.Set("Model", func);
    return exports;
}

ModelWrapper::ModelWrapper(const Napi::CallbackInfo& info)
    : Napi::ObjectWrap<ModelWrapper>(info) {
    Napi::Env env = info.Env();

    if (info.Length() < 2) {
        Napi::TypeError::New(env, "Expected (provider, modelId)").ThrowAsJavaScriptException();
        return;
    }

    auto prov_obj = info[0].As<Napi::Object>();
    auto* prov_wrap = Napi::ObjectWrap<ProviderWrapper>::Unwrap(prov_obj);
    std::string model_id = info[1].As<Napi::String>().Utf8Value();

    model_ = ai_model_create(prov_wrap->handle(), model_id.c_str());
    if (!model_) {
        Napi::Error::New(env, "Failed to create model").ThrowAsJavaScriptException();
    }
}

ModelWrapper::~ModelWrapper() {
    if (model_) ai_model_destroy(model_);
}

// --- ToolSetWrapper ---

struct ToolCallbackData {
    Napi::ThreadSafeFunction tsfn;
    Napi::FunctionReference callback;
};

Napi::Object ToolSetWrapper::Init(Napi::Env env, Napi::Object exports) {
    Napi::Function func = DefineClass(env, "ToolSet", {
        InstanceMethod("add", &ToolSetWrapper::AddTool),
    });
    exports.Set("ToolSet", func);
    return exports;
}

ToolSetWrapper::ToolSetWrapper(const Napi::CallbackInfo& info)
    : Napi::ObjectWrap<ToolSetWrapper>(info) {
    tools_ = ai_tool_set_create();
}

ToolSetWrapper::~ToolSetWrapper() {
    if (tools_) ai_tool_set_destroy(tools_);
}

void ToolSetWrapper::AddTool(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 4) {
        Napi::TypeError::New(env, "Expected (name, description, schemaJson, callback)").ThrowAsJavaScriptException();
        return;
    }

    std::string name = info[0].As<Napi::String>().Utf8Value();
    std::string description = info[1].As<Napi::String>().Utf8Value();
    std::string schema_json = info[2].As<Napi::String>().Utf8Value();
    Napi::Function callback = info[3].As<Napi::Function>();

    auto* cb_data = new ToolCallbackData();
    cb_data->callback = Napi::Persistent(callback);

    ai_tool_fn tool_fn = [](const char* tool_name, const char* input_json, void* user_data) -> ai_tool_result_t {
        auto* data = static_cast<ToolCallbackData*>(user_data);
        Napi::Env env = data->callback.Env();

        Napi::Value result = data->callback.Call({
            Napi::String::New(env, tool_name),
            Napi::String::New(env, input_json),
        });

        ai_tool_result_t res = {};
        if (result.IsObject()) {
            auto obj = result.As<Napi::Object>();
            std::string output = obj.Get("output").As<Napi::String>().Utf8Value();
            bool is_error = obj.Get("isError").As<Napi::Boolean>().Value();
            // Caller owns the string, we need to copy
            char* output_copy = new char[output.size() + 1];
            std::copy(output.begin(), output.end(), output_copy);
            output_copy[output.size()] = '\0';
            res.output_json = output_copy;
            res.is_error = is_error ? 1 : 0;
        }
        return res;
    };

    ai_tool_set_add(tools_, name.c_str(), description.c_str(), schema_json.c_str(), tool_fn, cb_data);
}

// --- AgentWrapper ---

Napi::Object AgentWrapper::Init(Napi::Env env, Napi::Object exports) {
    Napi::Function func = DefineClass(env, "Agent", {
        InstanceMethod("call", &AgentWrapper::Call),
    });
    exports.Set("Agent", func);
    return exports;
}

AgentWrapper::AgentWrapper(const Napi::CallbackInfo& info)
    : Napi::ObjectWrap<AgentWrapper>(info) {
    Napi::Env env = info.Env();

    if (info.Length() < 4) {
        Napi::TypeError::New(env, "Expected (model, toolSet, instructions, maxSteps)").ThrowAsJavaScriptException();
        return;
    }

    auto model_obj = info[0].As<Napi::Object>();
    auto* model_wrap = Napi::ObjectWrap<ModelWrapper>::Unwrap(model_obj);

    auto tools_obj = info[1].As<Napi::Object>();
    auto* tools_wrap = Napi::ObjectWrap<ToolSetWrapper>::Unwrap(tools_obj);

    std::string instructions = info[2].As<Napi::String>().Utf8Value();
    int max_steps = info[3].As<Napi::Number>().Int32Value();

    ai_agent_options_t opts = {};
    opts.model = model_wrap->handle();
    opts.tools = tools_wrap->handle();
    opts.instructions = instructions.c_str();
    opts.max_steps = max_steps;
    opts.on_event = nullptr;
    opts.user_data = nullptr;

    agent_ = ai_agent_create(opts);
    if (!agent_) {
        Napi::Error::New(env, "Failed to create agent").ThrowAsJavaScriptException();
    }
}

AgentWrapper::~AgentWrapper() {
    if (agent_) ai_agent_destroy(agent_);
}

Napi::Value AgentWrapper::Call(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 1) {
        Napi::TypeError::New(env, "Expected (prompt)").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    std::string prompt = info[0].As<Napi::String>().Utf8Value();
    ai_generate_result_t result = {};

    ai_status_t status = ai_agent_call(agent_, prompt.c_str(), &result);
    if (status != AI_OK) {
        std::string msg = result.text ? result.text : "Agent call failed";
        ai_generate_result_free(&result);
        Napi::Error::New(env, msg).ThrowAsJavaScriptException();
        return env.Undefined();
    }

    Napi::Object obj = Napi::Object::New(env);
    obj.Set("text", Napi::String::New(env, result.text ? result.text : ""));
    obj.Set("finishReason", Napi::String::New(env, result.finish_reason ? result.finish_reason : "stop"));
    obj.Set("inputTokens", Napi::Number::New(env, result.input_tokens));
    obj.Set("outputTokens", Napi::Number::New(env, result.output_tokens));
    obj.Set("steps", Napi::Number::New(env, result.steps));

    ai_generate_result_free(&result);
    return obj;
}

// --- Free functions ---

Napi::Value GenerateText(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 2) {
        Napi::TypeError::New(env, "Expected (model, options)").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    auto model_obj = info[0].As<Napi::Object>();
    auto* model_wrap = Napi::ObjectWrap<ModelWrapper>::Unwrap(model_obj);
    auto opts_obj = info[1].As<Napi::Object>();

    ai_generate_options_t opts = {};
    opts.model = model_wrap->handle();
    opts.temperature = -1;

    std::string prompt_str, system_str, messages_str;

    if (opts_obj.Has("prompt") && !opts_obj.Get("prompt").IsUndefined()) {
        prompt_str = opts_obj.Get("prompt").As<Napi::String>().Utf8Value();
        opts.prompt = prompt_str.c_str();
    }
    if (opts_obj.Has("system") && !opts_obj.Get("system").IsUndefined()) {
        system_str = opts_obj.Get("system").As<Napi::String>().Utf8Value();
        opts.system = system_str.c_str();
    }
    if (opts_obj.Has("messagesJson") && !opts_obj.Get("messagesJson").IsUndefined()) {
        messages_str = opts_obj.Get("messagesJson").As<Napi::String>().Utf8Value();
        opts.messages_json = messages_str.c_str();
    }
    if (opts_obj.Has("maxSteps") && !opts_obj.Get("maxSteps").IsUndefined()) {
        opts.max_steps = opts_obj.Get("maxSteps").As<Napi::Number>().Int32Value();
    }
    if (opts_obj.Has("maxOutputTokens") && !opts_obj.Get("maxOutputTokens").IsUndefined()) {
        opts.max_output_tokens = opts_obj.Get("maxOutputTokens").As<Napi::Number>().Int32Value();
    }
    if (opts_obj.Has("temperature") && !opts_obj.Get("temperature").IsUndefined()) {
        opts.temperature = opts_obj.Get("temperature").As<Napi::Number>().DoubleValue();
    }
    if (opts_obj.Has("toolSet") && !opts_obj.Get("toolSet").IsUndefined()) {
        auto ts_obj = opts_obj.Get("toolSet").As<Napi::Object>();
        auto* ts_wrap = Napi::ObjectWrap<ToolSetWrapper>::Unwrap(ts_obj);
        opts.tools = ts_wrap->handle();
    }

    ai_generate_result_t result = {};
    ai_status_t status = ai_generate_text(opts, &result);

    if (status != AI_OK) {
        std::string msg = result.text ? result.text : ai_status_message(status);
        ai_generate_result_free(&result);
        Napi::Error::New(env, msg).ThrowAsJavaScriptException();
        return env.Undefined();
    }

    Napi::Object obj = Napi::Object::New(env);
    obj.Set("text", Napi::String::New(env, result.text ? result.text : ""));
    obj.Set("finishReason", Napi::String::New(env, result.finish_reason ? result.finish_reason : "stop"));
    obj.Set("inputTokens", Napi::Number::New(env, result.input_tokens));
    obj.Set("outputTokens", Napi::Number::New(env, result.output_tokens));
    obj.Set("steps", Napi::Number::New(env, result.steps));

    ai_generate_result_free(&result);
    return obj;
}

Napi::Value StreamText(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 3) {
        Napi::TypeError::New(env, "Expected (model, options, callback)").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    auto model_obj = info[0].As<Napi::Object>();
    auto* model_wrap = Napi::ObjectWrap<ModelWrapper>::Unwrap(model_obj);
    auto opts_obj = info[1].As<Napi::Object>();
    Napi::Function callback = info[2].As<Napi::Function>();

    ai_generate_options_t opts = {};
    opts.model = model_wrap->handle();
    opts.temperature = -1;

    std::string prompt_str, system_str, messages_str;

    if (opts_obj.Has("prompt") && !opts_obj.Get("prompt").IsUndefined()) {
        prompt_str = opts_obj.Get("prompt").As<Napi::String>().Utf8Value();
        opts.prompt = prompt_str.c_str();
    }
    if (opts_obj.Has("system") && !opts_obj.Get("system").IsUndefined()) {
        system_str = opts_obj.Get("system").As<Napi::String>().Utf8Value();
        opts.system = system_str.c_str();
    }
    if (opts_obj.Has("maxOutputTokens") && !opts_obj.Get("maxOutputTokens").IsUndefined()) {
        opts.max_output_tokens = opts_obj.Get("maxOutputTokens").As<Napi::Number>().Int32Value();
    }
    if (opts_obj.Has("temperature") && !opts_obj.Get("temperature").IsUndefined()) {
        opts.temperature = opts_obj.Get("temperature").As<Napi::Number>().DoubleValue();
    }

    struct StreamCallbackData {
        Napi::FunctionReference fn;
    };
    auto* cb_data = new StreamCallbackData();
    cb_data->fn = Napi::Persistent(callback);

    ai_stream_callback_fn stream_cb = [](ai_stream_event_t event, void* user_data) {
        auto* data = static_cast<StreamCallbackData*>(user_data);
        Napi::Env env = data->fn.Env();

        const char* type_str = "unknown";
        switch (event.type) {
            case AI_STREAM_TEXT_DELTA: type_str = "text_delta"; break;
            case AI_STREAM_TOOL_CALL_START: type_str = "tool_call_start"; break;
            case AI_STREAM_TOOL_CALL_DELTA: type_str = "tool_call_delta"; break;
            case AI_STREAM_TOOL_CALL_END: type_str = "tool_call_end"; break;
            case AI_STREAM_FINISH: type_str = "finish"; break;
            case AI_STREAM_ERROR: type_str = "error"; break;
            case AI_STREAM_STEP_FINISH: type_str = "step_finish"; break;
        }

        data->fn.Call({
            Napi::String::New(env, type_str),
            event.text ? Napi::String::New(env, event.text) : env.Null(),
            event.tool_name ? Napi::String::New(env, event.tool_name) : env.Null(),
            event.tool_call_id ? Napi::String::New(env, event.tool_call_id) : env.Null(),
        });
    };

    ai_status_t status = ai_stream_text(opts, stream_cb, cb_data);
    delete cb_data;

    if (status != AI_OK) {
        Napi::Error::New(env, ai_status_message(status)).ThrowAsJavaScriptException();
    }

    return env.Undefined();
}

Napi::Value GetVersion(const Napi::CallbackInfo& info) {
    return Napi::String::New(info.Env(), ai_sdk_version());
}

// --- Module initialization ---

Napi::Object Init(Napi::Env env, Napi::Object exports) {
    ContextWrapper::Init(env, exports);
    ProviderWrapper::Init(env, exports);
    ModelWrapper::Init(env, exports);
    ToolSetWrapper::Init(env, exports);
    AgentWrapper::Init(env, exports);

    exports.Set("generateText", Napi::Function::New(env, GenerateText));
    exports.Set("streamText", Napi::Function::New(env, StreamText));
    exports.Set("version", Napi::Function::New(env, GetVersion));

    return exports;
}

NODE_API_MODULE(ai_sdk_native, Init)
