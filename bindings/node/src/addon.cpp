#include <napi.h>
#include "ai_sdk.h"
#include <string>
#include <vector>
#include <thread>
#include <atomic>
#include <mutex>
#include <map>
#include <future>

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

struct ToolCallbackData;  // defined below; ToolSetWrapper owns + releases these.

class ToolSetWrapper : public Napi::ObjectWrap<ToolSetWrapper> {
public:
    static Napi::Object Init(Napi::Env env, Napi::Object exports);
    ToolSetWrapper(const Napi::CallbackInfo& info);
    ~ToolSetWrapper();
    ai_tool_set_t handle() const { return tools_; }
    void AddTool(const Napi::CallbackInfo& info);
    void SetTools(ai_tool_set_t h) { if (tools_) ai_tool_set_destroy(tools_); tools_ = h; }

private:
    ai_tool_set_t tools_;
    std::vector<ToolCallbackData*> tool_callbacks_;
};

class AgentWrapper : public Napi::ObjectWrap<AgentWrapper> {
public:
    static Napi::Object Init(Napi::Env env, Napi::Object exports);
    AgentWrapper(const Napi::CallbackInfo& info);
    ~AgentWrapper();
    Napi::Value Call(const Napi::CallbackInfo& info);
    ai_agent_t handle() const { return agent_; }

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

// --- Async tool execution ---
// When the agent loop runs on a worker thread (async streaming), tool callbacks
// fire on that thread. They invoke the (possibly async — e.g. interactive
// permission prompts) JS tool via a ThreadSafeFunction and await the result
// through a shared promise/future. Sync entry points (main thread) still call
// the JS tool directly.

static std::thread::id g_main_thread_id;
static std::mutex g_pending_mutex;
static std::map<uint64_t, std::shared_ptr<std::promise<std::pair<std::string, bool>>>> g_pending;
static std::atomic<uint64_t> g_call_id{0};
static Napi::FunctionReference g_resolve_ref;

struct ToolCallPayload {
    uint64_t id;
    std::string name;
    std::string input;
};

static ai_tool_result_t make_tool_result(const std::string& output, bool is_error) {
    ai_tool_result_t res = {};
    char* copy = new char[output.size() + 1];
    std::copy(output.begin(), output.end(), copy);
    copy[output.size()] = '\0';
    res.output_json = copy;
    res.is_error = is_error ? 1 : 0;
    return res;
}

// Main-thread TSFN receiver: calls the JS tool, then fulfills the pending
// promise via __resolveToolCall (handles both Promise-returning and sync tools).
static void ToolCallReceiver(Napi::Env env, Napi::Function jsToolFn, ToolCallPayload* p) {
    uint64_t id = p->id;
    Napi::Value ret = jsToolFn.Call({ Napi::String::New(env, p->name), Napi::String::New(env, p->input) });
    delete p;

    auto extract = [](const Napi::Object& o) -> std::pair<std::string, bool> {
        std::string out = (o.Has("output") && o.Get("output").IsString())
            ? o.Get("output").As<Napi::String>().Utf8Value() : "";
        bool err = o.Has("isError") && o.Get("isError").IsBoolean() && o.Get("isError").As<Napi::Boolean>().Value();
        return { out, err };
    };

    if (ret.IsObject() && ret.As<Napi::Object>().Has("then")) {
        Napi::Object promiseObj = ret.As<Napi::Object>();
        Napi::Function then = promiseObj.Get("then").As<Napi::Function>();
        Napi::Function onFulfill = Napi::Function::New(env, [id](const Napi::CallbackInfo& info) {
            std::string out; bool err = false;
            if (info.Length() > 0 && info[0].IsObject()) std::tie(out, err) = std::make_pair(
                (info[0].As<Napi::Object>().Has("output") && info[0].As<Napi::Object>().Get("output").IsString())
                    ? info[0].As<Napi::Object>().Get("output").As<Napi::String>().Utf8Value() : "",
                info[0].As<Napi::Object>().Has("isError") && info[0].As<Napi::Object>().Get("isError").As<Napi::Boolean>().Value());
            g_resolve_ref.Call({ Napi::Number::New(info.Env(), (double)id), Napi::String::New(info.Env(), out), Napi::Boolean::New(info.Env(), err) });
            return info.Env().Undefined();
        });
        Napi::Function onReject = Napi::Function::New(env, [id](const Napi::CallbackInfo& info) {
            std::string emsg = (info.Length() > 0 && info[0].IsObject() && info[0].As<Napi::Object>().Has("message"))
                ? info[0].As<Napi::Object>().Get("message").As<Napi::String>().Utf8Value() : std::string("tool threw");
            g_resolve_ref.Call({ Napi::Number::New(info.Env(), (double)id), Napi::String::New(info.Env(), emsg), Napi::Boolean::New(info.Env(), true) });
            return info.Env().Undefined();
        });
        then.Call(promiseObj, { onFulfill, onReject });
    } else if (ret.IsObject()) {
        auto [out, err] = extract(ret.As<Napi::Object>());
        g_resolve_ref.Call({ Napi::Number::New(env, (double)id), Napi::String::New(env, out), Napi::Boolean::New(env, err) });
    } else {
        g_resolve_ref.Call({ Napi::Number::New(env, (double)id), Napi::String::New(env, ""), Napi::Boolean::New(env, true) });
    }
}

// __resolveToolCall(id, output, isError) — invoked from JS (.then/.catch) to
// fulfill the promise the worker thread is blocked on.
static Napi::Value ResolveToolCall(const Napi::CallbackInfo& info) {
    uint64_t id = (uint64_t)info[0].As<Napi::Number>().DoubleValue();
    std::string output = info[1].As<Napi::String>().Utf8Value();
    bool is_error = info[2].As<Napi::Boolean>().Value();
    std::shared_ptr<std::promise<std::pair<std::string, bool>>> prom;
    {
        std::lock_guard<std::mutex> lk(g_pending_mutex);
        auto it = g_pending.find(id);
        if (it != g_pending.end()) prom = it->second;
    }
    if (prom) prom->set_value({ output, is_error });
    return info.Env().Undefined();
}

// --- ToolSetWrapper ---

struct ToolCallbackData {
    Napi::ThreadSafeFunction tsfn;
    Napi::FunctionReference callback;
};

static Napi::FunctionReference g_toolset_constructor;

Napi::Object ToolSetWrapper::Init(Napi::Env env, Napi::Object exports) {
    Napi::Function func = DefineClass(env, "ToolSet", {
        InstanceMethod("add", &ToolSetWrapper::AddTool),
    });
    g_toolset_constructor = Napi::Persistent(func);
    exports.Set("ToolSet", func);
    return exports;
}

ToolSetWrapper::ToolSetWrapper(const Napi::CallbackInfo& info)
    : Napi::ObjectWrap<ToolSetWrapper>(info) {
    tools_ = ai_tool_set_create();
}

ToolSetWrapper::~ToolSetWrapper() {
    if (tools_) ai_tool_set_destroy(tools_);
    // NOTE: ToolCallbackData (TSFN + callback) is intentionally NOT freed here.
    // The agent's copy of the ToolSet references the same ToolCallbackData via
    // the execute function's user_data pointer. Freeing it would cause a
    // use-after-free (SIGTRAP) when the agent calls a tool after this wrapper
    // is GC'd. The data lives for the process lifetime (bounded: one per tool
    // per session creation — acceptable for a long-running agent).
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
    cb_data->tsfn = Napi::ThreadSafeFunction::New(env, callback, "ai-tool", 0, 1);
    // Don't let the per-tool TSFN hold the event loop alive between turns (the
    // stream TSFN already keeps it alive while a turn is in flight).
    cb_data->tsfn.Unref(env);

    ai_tool_fn tool_fn = [](const char* tool_name, const char* input_json, void* user_data) -> ai_tool_result_t {
        auto* data = static_cast<ToolCallbackData*>(user_data);

        // Sync entry points (generateText/agent.call/session.send) run the
        // io_context on the main thread — call the JS tool directly.
        if (std::this_thread::get_id() == g_main_thread_id) {
            Napi::Env env = data->callback.Env();
            Napi::Value result = data->callback.Call({
                Napi::String::New(env, tool_name ? tool_name : ""),
                Napi::String::New(env, input_json ? input_json : ""),
            });
            if (result.IsObject()) {
                auto obj = result.As<Napi::Object>();
                std::string output = (obj.Has("output") && obj.Get("output").IsString())
                    ? obj.Get("output").As<Napi::String>().Utf8Value() : "";
                bool is_error = obj.Has("isError") && obj.Get("isError").As<Napi::Boolean>().Value();
                return make_tool_result(output, is_error);
            }
            return make_tool_result("", true);
        }

        // Async streaming runs the loop on a worker thread — invoke the (possibly
        // async) JS tool via TSFN and block until JS resolves. The main thread
        // stays free to run the tool (incl. interactive permission prompts).
        auto promise = std::make_shared<std::promise<std::pair<std::string, bool>>>();
        auto fut = promise->get_future();
        uint64_t id = g_call_id.fetch_add(1);
        {
            std::lock_guard<std::mutex> lk(g_pending_mutex);
            g_pending[id] = promise;
        }
        auto* p = new ToolCallPayload{ id, tool_name ? tool_name : "", input_json ? input_json : "" };
        data->tsfn.NonBlockingCall(p, ToolCallReceiver);
        auto result_pair = fut.get();
        {
            std::lock_guard<std::mutex> lk(g_pending_mutex);
            g_pending.erase(id);
        }
        return make_tool_result(result_pair.first, result_pair.second);
    };

    ai_tool_set_add(tools_, name.c_str(), description.c_str(), schema_json.c_str(), tool_fn, cb_data);
    tool_callbacks_.push_back(cb_data);
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

// --- Async streaming: background thread + ThreadSafeFunction ---
//
// The C stream calls (ai_stream_text, ai_session_send_stream) block while they
// drain the io_context on the calling thread. To keep the JS event loop alive
// (required for Ink/React UIs), each stream runs on a detached std::thread and
// events are relayed to the JS callback on the main thread via a
// Napi::ThreadSafeFunction. The C API stays a simple blocking function.

struct StreamEventPayload {
    std::string type;
    std::string text;
    std::string toolName;
    std::string toolCallId;
    int inputTokens = 0;
    int outputTokens = 0;
};

struct StreamSession {
    Napi::ThreadSafeFunction tsfn;
    std::atomic<bool> terminal{false};   // set when a finish/error event fires
};

// Relay one C stream event to JS via the TSFN. Called on the worker thread.
static void emit_stream_event(StreamSession& s, const ai_stream_event_t& event) {
    auto* p = new StreamEventPayload();
    switch (event.type) {
        case AI_STREAM_TEXT_DELTA: p->type = "text_delta"; break;
        case AI_STREAM_TOOL_CALL_START: p->type = "tool_call_start"; break;
        case AI_STREAM_TOOL_CALL_DELTA: p->type = "tool_call_delta"; break;
        case AI_STREAM_TOOL_CALL_END: p->type = "tool_call_end"; break;
        case AI_STREAM_FINISH: p->type = "finish"; s.terminal = true;
            p->inputTokens = event.input_tokens; p->outputTokens = event.output_tokens; break;
        case AI_STREAM_ERROR: p->type = "error"; s.terminal = true; break;
        case AI_STREAM_STEP_FINISH: p->type = "step_finish"; break;
        case AI_STREAM_REASONING_START: p->type = "reasoning_start"; break;
        case AI_STREAM_REASONING_DELTA: p->type = "reasoning_delta"; break;
        case AI_STREAM_REASONING_END: p->type = "reasoning_end"; break;
        case AI_STREAM_TOOL_RESULT: p->type = "tool_result"; break;
        default: p->type = "unknown"; break;
    }
    if (event.text) p->text = event.text;
    if (event.tool_name) p->toolName = event.tool_name;
    if (event.tool_call_id) p->toolCallId = event.tool_call_id;
    s.tsfn.NonBlockingCall(p, [](Napi::Env env, Napi::Function jsCallback, StreamEventPayload* p) {
        Napi::Value usage = env.Null();
        if (p->type == "finish") {
            auto u = Napi::Object::New(env);
            u.Set("inputTokens", Napi::Number::New(env, p->inputTokens));
            u.Set("outputTokens", Napi::Number::New(env, p->outputTokens));
            usage = u;
        }
        jsCallback.Call({
            Napi::String::New(env, p->type),
            p->text.empty() ? env.Null() : Napi::String::New(env, p->text),
            p->toolName.empty() ? env.Null() : Napi::String::New(env, p->toolName),
            p->toolCallId.empty() ? env.Null() : Napi::String::New(env, p->toolCallId),
            usage,
        });
        delete p;
    });
}

// C stream callback shared by both streaming entry points.
static ai_stream_callback_fn stream_callback = [](ai_stream_event_t event, void* user_data) {
    emit_stream_event(*static_cast<StreamSession*>(user_data), event);
};

// Run a blocking C stream call on a worker thread; return immediately. `cCall`
// is invoked as cCall(StreamSession* s, ai_stream_callback_fn cb) and must drive
// the stream to completion. After it returns, the TSFN is released.
template <typename F>
static void RunStreamAsync(Napi::Env env, Napi::Function callback, F cCall) {
    auto* s = new StreamSession();
    s->tsfn = Napi::ThreadSafeFunction::New(env, callback, "ai-stream", 0, 1);
    std::thread([s, cCall = std::move(cCall)]() mutable {
        cCall(s, stream_callback);
        if (!s->terminal) {
            // The C call returned without a finish/error event (e.g. it failed
            // before streaming) — synthesize one so the JS consumer terminates.
            ai_stream_event_t ev = {};
            ev.type = AI_STREAM_FINISH;
            emit_stream_event(*s, ev);
        }
        s->tsfn.Release();
        delete s;
    }).detach();
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

    // Option strings are captured by value — they must outlive this call since
    // the stream runs on a worker thread after we return.
    auto model = model_wrap->handle();
    std::string prompt_str, system_str, messages_str;
    int max_output_tokens = 0, max_steps = 0;
    double temperature = -1;
    ai_tool_set_t tools = nullptr;

    if (opts_obj.Has("prompt") && !opts_obj.Get("prompt").IsUndefined()) {
        prompt_str = opts_obj.Get("prompt").As<Napi::String>().Utf8Value();
    }
    if (opts_obj.Has("system") && !opts_obj.Get("system").IsUndefined()) {
        system_str = opts_obj.Get("system").As<Napi::String>().Utf8Value();
    }
    if (opts_obj.Has("messagesJson") && !opts_obj.Get("messagesJson").IsUndefined()) {
        messages_str = opts_obj.Get("messagesJson").As<Napi::String>().Utf8Value();
    }
    if (opts_obj.Has("maxSteps") && !opts_obj.Get("maxSteps").IsUndefined()) {
        max_steps = opts_obj.Get("maxSteps").As<Napi::Number>().Int32Value();
    }
    if (opts_obj.Has("maxOutputTokens") && !opts_obj.Get("maxOutputTokens").IsUndefined()) {
        max_output_tokens = opts_obj.Get("maxOutputTokens").As<Napi::Number>().Int32Value();
    }
    if (opts_obj.Has("temperature") && !opts_obj.Get("temperature").IsUndefined()) {
        temperature = opts_obj.Get("temperature").As<Napi::Number>().DoubleValue();
    }
    if (opts_obj.Has("toolSet") && !opts_obj.Get("toolSet").IsUndefined()) {
        auto ts_obj = opts_obj.Get("toolSet").As<Napi::Object>();
        auto* ts_wrap = Napi::ObjectWrap<ToolSetWrapper>::Unwrap(ts_obj);
        tools = ts_wrap->handle();
    }

    RunStreamAsync(env, callback,
        [model, prompt_str, system_str, messages_str, max_output_tokens, max_steps, temperature, tools]
        (StreamSession* s, ai_stream_callback_fn cb) {
            ai_generate_options_t opts = {};
            opts.model = model;
            opts.temperature = temperature;
            opts.max_output_tokens = max_output_tokens;
            opts.max_steps = max_steps;
            opts.tools = tools;
            opts.prompt = prompt_str.empty() ? nullptr : prompt_str.c_str();
            opts.system = system_str.empty() ? nullptr : system_str.c_str();
            opts.messages_json = messages_str.empty() ? nullptr : messages_str.c_str();
            ai_stream_text(opts, cb, s);
        });

    return env.Undefined();
}

Napi::Value GetVersion(const Napi::CallbackInfo& info) {
    return Napi::String::New(info.Env(), ai_sdk_version());
}

// --- SessionWrapper ---

class SessionWrapper : public Napi::ObjectWrap<SessionWrapper> {
public:
    static Napi::Object Init(Napi::Env env, Napi::Object exports);
    SessionWrapper(const Napi::CallbackInfo& info);
    ~SessionWrapper();
    Napi::Value Send(const Napi::CallbackInfo& info);
    Napi::Value SendStream(const Napi::CallbackInfo& info);

private:
    ai_session_t session_;
};

Napi::Object SessionWrapper::Init(Napi::Env env, Napi::Object exports) {
    Napi::Function func = DefineClass(env, "Session", {
        InstanceMethod("send", &SessionWrapper::Send),
        InstanceMethod("sendStream", &SessionWrapper::SendStream),
    });
    exports.Set("Session", func);
    return exports;
}

SessionWrapper::SessionWrapper(const Napi::CallbackInfo& info)
    : Napi::ObjectWrap<SessionWrapper>(info) {
    Napi::Env env = info.Env();
    if (info.Length() < 1) {
        Napi::TypeError::New(env, "Expected (agent[, memoryDir, maxContextTokens])").ThrowAsJavaScriptException();
        return;
    }
    auto agent_obj = info[0].As<Napi::Object>();
    auto* agent_wrap = Napi::ObjectWrap<AgentWrapper>::Unwrap(agent_obj);
    // Optional memory session: (agent, memoryDir, maxContextTokens) ->
    // MemoryContextStrategy (auto-inject + sliding-window auto-compact).
    if (info.Length() >= 2 && info[1].IsString()) {
        std::string dir = info[1].As<Napi::String>().Utf8Value();
        int max_tokens = (info.Length() >= 3 && info[2].IsNumber())
            ? info[2].As<Napi::Number>().Int32Value() : 0;
        session_ = ai_session_create_with_memory(agent_wrap->handle(), dir.c_str(), max_tokens);
    } else {
        session_ = ai_session_create(agent_wrap->handle());
    }
    if (!session_) {
        Napi::Error::New(env, "Failed to create session").ThrowAsJavaScriptException();
    }
}

SessionWrapper::~SessionWrapper() {
    if (session_) ai_session_destroy(session_);
}

Napi::Value SessionWrapper::Send(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 1) {
        Napi::TypeError::New(env, "Expected (prompt)").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    std::string prompt = info[0].As<Napi::String>().Utf8Value();
    ai_generate_result_t result = {};
    ai_status_t status = ai_session_send(session_, prompt.c_str(), &result);
    if (status != AI_OK) {
        std::string msg = result.text ? result.text : "Session send failed";
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

Napi::Value SessionWrapper::SendStream(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 2) {
        Napi::TypeError::New(env, "Expected (prompt, callback)").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    std::string prompt = info[0].As<Napi::String>().Utf8Value();
    Napi::Function callback = info[1].As<Napi::Function>();
    ai_session_t session = session_;

    RunStreamAsync(env, callback,
        [session, prompt](StreamSession* s, ai_stream_callback_fn cb) {
            ai_session_send_stream(session, prompt.c_str(), cb, s);
        });

    return env.Undefined();
}

// --- Standard toolkit + permissions ---

Napi::Value StandardToolkit(const Napi::CallbackInfo& info) {
    Napi::Object obj = g_toolset_constructor.New({});
    auto* wrap = Napi::ObjectWrap<ToolSetWrapper>::Unwrap(obj);
    wrap->SetTools(ai_standard_toolkit_create());
    return obj;
}

Napi::Value WithPermissions(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 2) {
        Napi::TypeError::New(env, "Expected (toolSet, policy)").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    auto tools_obj = info[0].As<Napi::Object>();
    auto* tools_wrap = Napi::ObjectWrap<ToolSetWrapper>::Unwrap(tools_obj);
    Napi::Function policy_fn = info[1].As<Napi::Function>();
    auto* fn_ref = new Napi::FunctionReference();
    *fn_ref = Napi::Persistent(policy_fn);

    ai_permission_policy_fn c_fn = [](const char* tool, const char* input_json,
                                      void* ud) -> int {
        auto* ref = static_cast<Napi::FunctionReference*>(ud);
        Napi::Env e = ref->Env();
        Napi::Value result = ref->Call({Napi::String::New(e, tool),
                                        Napi::String::New(e, input_json)});
        return result.IsNumber() ? result.As<Napi::Number>().Int32Value()
                                 : AI_PERMISSION_DENY;
    };

    ai_tool_set_t gated = ai_with_permissions(tools_wrap->handle(), c_fn, fn_ref);
    Napi::Object obj = g_toolset_constructor.New({});
    auto* out_wrap = Napi::ObjectWrap<ToolSetWrapper>::Unwrap(obj);
    out_wrap->SetTools(gated);
    return obj;
}

// --- MCP ---

Napi::Value MergeToolSets(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 2) {
        Napi::TypeError::New(env, "Expected (destToolSet, srcToolSet)").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    auto* dest = Napi::ObjectWrap<ToolSetWrapper>::Unwrap(info[0].As<Napi::Object>());
    auto* src = Napi::ObjectWrap<ToolSetWrapper>::Unwrap(info[1].As<Napi::Object>());
    ai_tool_set_merge(dest->handle(), src->handle());
    return env.Undefined();
}

Napi::Value McpToolsetFromServer(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 2) {
        Napi::TypeError::New(env, "Expected (context, configJson)").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    auto* ctx_wrap = Napi::ObjectWrap<ContextWrapper>::Unwrap(info[0].As<Napi::Object>());
    std::string config = info[1].As<Napi::String>().Utf8Value();
    ai_tool_set_t ts = ai_mcp_toolset_from_server(ctx_wrap->handle(), config.c_str());
    if (!ts) {
        Napi::Error::New(env, ai_last_error(ctx_wrap->handle())).ThrowAsJavaScriptException();
        return env.Undefined();
    }
    Napi::Object obj = g_toolset_constructor.New({});
    Napi::ObjectWrap<ToolSetWrapper>::Unwrap(obj)->SetTools(ts);
    return obj;
}

// --- MemoryStore ---

class MemoryStoreWrapper : public Napi::ObjectWrap<MemoryStoreWrapper> {
public:
    static Napi::Object Init(Napi::Env env, Napi::Object exports);
    MemoryStoreWrapper(const Napi::CallbackInfo& info);
    ~MemoryStoreWrapper();
    Napi::Value Save(const Napi::CallbackInfo& info);
private:
    ai_memory_store_t store_;
};

Napi::Object MemoryStoreWrapper::Init(Napi::Env env, Napi::Object exports) {
    Napi::Function func = DefineClass(env, "MemoryStore", {
        InstanceMethod("save", &MemoryStoreWrapper::Save),
    });
    exports.Set("MemoryStore", func);
    return exports;
}

MemoryStoreWrapper::MemoryStoreWrapper(const Napi::CallbackInfo& info)
    : Napi::ObjectWrap<MemoryStoreWrapper>(info), store_(nullptr) {
    Napi::Env env = info.Env();
    if (info.Length() < 1) {
        Napi::TypeError::New(env, "Expected (dir)").ThrowAsJavaScriptException();
        return;
    }
    std::string dir = info[0].As<Napi::String>().Utf8Value();
    store_ = ai_memory_store_create(dir.c_str());
    if (!store_) {
        Napi::Error::New(env, "Failed to create memory store").ThrowAsJavaScriptException();
    }
}

MemoryStoreWrapper::~MemoryStoreWrapper() {
    if (store_) ai_memory_store_destroy(store_);
}

Napi::Value MemoryStoreWrapper::Save(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 3) {
        Napi::TypeError::New(env, "Expected (scope, key, content)").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    std::string scope = info[0].As<Napi::String>().Utf8Value();
    std::string key = info[1].As<Napi::String>().Utf8Value();
    std::string content = info[2].As<Napi::String>().Utf8Value();
    ai_status_t status = ai_memory_save(store_, scope.c_str(), key.c_str(), content.c_str());
    if (status != AI_OK) {
        Napi::Error::New(env, ai_status_message(status)).ThrowAsJavaScriptException();
    }
    return env.Undefined();
}

// --- Batch ---

class BatchWrapper : public Napi::ObjectWrap<BatchWrapper> {
public:
    static Napi::Object Init(Napi::Env env, Napi::Object exports);
    BatchWrapper(const Napi::CallbackInfo& info);
    ~BatchWrapper();
    Napi::Value Run(const Napi::CallbackInfo& info);
private:
    ai_batch_t batch_;
};

Napi::Object BatchWrapper::Init(Napi::Env env, Napi::Object exports) {
    Napi::Function func = DefineClass(env, "Batch", {
        InstanceMethod("run", &BatchWrapper::Run),
    });
    exports.Set("Batch", func);
    return exports;
}

BatchWrapper::BatchWrapper(const Napi::CallbackInfo& info)
    : Napi::ObjectWrap<BatchWrapper>(info), batch_(nullptr) {
    Napi::Env env = info.Env();
    if (info.Length() < 2) {
        Napi::TypeError::New(env, "Expected (provider, modelId)").ThrowAsJavaScriptException();
        return;
    }
    auto prov_obj = info[0].As<Napi::Object>();
    auto* prov_wrap = Napi::ObjectWrap<ProviderWrapper>::Unwrap(prov_obj);
    std::string model_id = info[1].As<Napi::String>().Utf8Value();
    batch_ = ai_batch_create(prov_wrap->handle(), model_id.c_str());
    if (!batch_) {
        Napi::Error::New(env, "Provider does not support batching").ThrowAsJavaScriptException();
    }
}

BatchWrapper::~BatchWrapper() {
    if (batch_) ai_batch_destroy(batch_);
}

Napi::Value BatchWrapper::Run(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 1) {
        Napi::TypeError::New(env, "Expected (requests[, pollIntervalMs])").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    Napi::Array reqs = info[0].As<Napi::Array>();
    int count = reqs.Length();
    int poll_ms = (info.Length() > 1 && info[1].IsNumber()) ? info[1].As<Napi::Number>().Int32Value() : 5000;

    // String storage must outlive the call.
    std::vector<std::string> custom_ids, prompts, systems;
    std::vector<ai_batch_request_t> creqs(count);
    for (int i = 0; i < count; ++i) {
        auto r = reqs.Get(i).As<Napi::Object>();
        custom_ids.push_back(r.Has("customId") ? r.Get("customId").As<Napi::String>().Utf8Value()
                                               : (std::string("req-") + std::to_string(i)));
        prompts.push_back(r.Has("prompt") ? r.Get("prompt").As<Napi::String>().Utf8Value() : std::string());
        systems.push_back(r.Has("system") ? r.Get("system").As<Napi::String>().Utf8Value() : std::string());
        creqs[i].custom_id = custom_ids.back().c_str();
        creqs[i].prompt = prompts.back().empty() ? nullptr : prompts.back().c_str();
        creqs[i].system = systems.back().empty() ? nullptr : systems.back().c_str();
        creqs[i].max_output_tokens = (r.Has("maxOutputTokens") && r.Get("maxOutputTokens").IsNumber())
            ? r.Get("maxOutputTokens").As<Napi::Number>().Int32Value() : 0;
        creqs[i].temperature = (r.Has("temperature") && r.Get("temperature").IsNumber())
            ? r.Get("temperature").As<Napi::Number>().DoubleValue() : -1.0;
    }

    ai_batch_result_t result = {};
    ai_status_t status = ai_batch_run(batch_, creqs.data(), count, poll_ms, &result);
    if (status != AI_OK) {
        ai_batch_result_free(&result);
        Napi::Error::New(env, ai_status_message(status)).ThrowAsJavaScriptException();
        return env.Undefined();
    }

    Napi::Object out = Napi::Object::New(env);
    out.Set("batchId", Napi::String::New(env, result.batch_id ? result.batch_id : ""));
    out.Set("status", Napi::String::New(env, result.status ? result.status : "unknown"));
    Napi::Array items = Napi::Array::New(env, result.count);
    for (int i = 0; i < result.count; ++i) {
        Napi::Object it = Napi::Object::New(env);
        it.Set("customId", Napi::String::New(env, result.items[i].custom_id ? result.items[i].custom_id : ""));
        it.Set("result", result.items[i].result_json ? Napi::String::New(env, result.items[i].result_json) : env.Null());
        it.Set("error", result.items[i].error ? Napi::String::New(env, result.items[i].error) : env.Null());
        items[(uint32_t)i] = it;
    }
    out.Set("items", items);
    ai_batch_result_free(&result);
    return out;
}

// --- Module initialization ---

Napi::Object Init(Napi::Env env, Napi::Object exports) {
    g_main_thread_id = std::this_thread::get_id();
    g_resolve_ref = Napi::Persistent(Napi::Function::New(env, ResolveToolCall, "__resolveToolCall"));

    ContextWrapper::Init(env, exports);
    ProviderWrapper::Init(env, exports);
    ModelWrapper::Init(env, exports);
    ToolSetWrapper::Init(env, exports);
    AgentWrapper::Init(env, exports);
    SessionWrapper::Init(env, exports);
    MemoryStoreWrapper::Init(env, exports);
    BatchWrapper::Init(env, exports);

    exports.Set("generateText", Napi::Function::New(env, GenerateText));
    exports.Set("streamText", Napi::Function::New(env, StreamText));
    exports.Set("standardToolkit", Napi::Function::New(env, StandardToolkit));
    exports.Set("withPermissions", Napi::Function::New(env, WithPermissions));
    exports.Set("mergeToolSets", Napi::Function::New(env, MergeToolSets));
    exports.Set("mcpToolsetFromServer", Napi::Function::New(env, McpToolsetFromServer));
    exports.Set("version", Napi::Function::New(env, GetVersion));

    return exports;
}

NODE_API_MODULE(ai_sdk_native, Init)
