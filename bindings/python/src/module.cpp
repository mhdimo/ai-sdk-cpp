#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include "ai_sdk.h"
#include <string>
#include <stdexcept>
#include <vector>

namespace py = pybind11;

static void check_status(ai_status_t status, ai_context_t ctx = nullptr) {
    if (status == AI_OK) return;
    const char* msg = ctx ? ai_last_error(ctx) : ai_status_message(status);
    switch (status) {
        case AI_ERROR_RATE_LIMIT:
            throw std::runtime_error(std::string("RateLimitError: ") + msg);
        case AI_ERROR_AUTHENTICATION:
            throw std::runtime_error(std::string("AuthenticationError: ") + msg);
        case AI_ERROR_TIMEOUT:
            throw std::runtime_error(std::string("TimeoutError: ") + msg);
        default:
            throw std::runtime_error(msg);
    }
}

class PyAiContext {
public:
    PyAiContext() : ctx_(ai_context_create()) {
        if (!ctx_) throw std::runtime_error("Failed to create AI context");
    }
    ~PyAiContext() { if (ctx_) ai_context_destroy(ctx_); }
    ai_context_t handle() const { return ctx_; }

    PyAiContext(const PyAiContext&) = delete;
    PyAiContext& operator=(const PyAiContext&) = delete;

private:
    ai_context_t ctx_;
};

class PyProvider {
public:
    PyProvider(PyAiContext& ctx, const std::string& name,
              const std::string& api_key = "", const std::string& base_url = "")
        : ctx_(&ctx) {
        ai_provider_options_t opts = {};
        if (!api_key.empty()) opts.api_key = api_key.c_str();
        if (!base_url.empty()) opts.base_url = base_url.c_str();
        provider_ = ai_provider_create(ctx.handle(), name.c_str(), opts);
        if (!provider_) throw std::runtime_error("Failed to create provider: " + name);
    }
    ~PyProvider() { if (provider_) ai_provider_destroy(provider_); }
    ai_provider_t handle() const { return provider_; }

    PyProvider(const PyProvider&) = delete;
    PyProvider& operator=(const PyProvider&) = delete;

private:
    PyAiContext* ctx_;
    ai_provider_t provider_;
};

class PyModel {
public:
    PyModel(PyProvider& provider, const std::string& model_id) {
        model_ = ai_model_create(provider.handle(), model_id.c_str());
        if (!model_) throw std::runtime_error("Failed to create model: " + model_id);
    }
    ~PyModel() { if (model_) ai_model_destroy(model_); }
    ai_model_t handle() const { return model_; }

    PyModel(const PyModel&) = delete;
    PyModel& operator=(const PyModel&) = delete;

private:
    ai_model_t model_;
};

struct PyToolEntry {
    std::string name;
    std::string description;
    std::string schema_json;
    py::function callback;
};

class PyToolSet {
public:
    PyToolSet() : tools_(ai_tool_set_create()) {
        if (!tools_) throw std::runtime_error("Failed to create tool set");
    }
    ~PyToolSet() { if (tools_) ai_tool_set_destroy(tools_); }
    ai_tool_set_t handle() const { return tools_; }

    /// Wrap an existing C toolset handle (e.g. from ai_standard_toolkit_create),
    /// taking ownership.
    static std::unique_ptr<PyToolSet> wrap(ai_tool_set_t h) {
        auto ts = std::make_unique<PyToolSet>();  // constructs an empty handle
        ai_tool_set_destroy(ts->tools_);
        ts->tools_ = h;
        return ts;
    }

    void add(const std::string& name, const py::dict& schema,
             const std::string& description, py::function fn) {
        py::module_ json = py::module_::import("json");
        std::string schema_json = json.attr("dumps")(schema).cast<std::string>();

        entries_.push_back({name, description, schema_json, fn});
        auto* entry = &entries_.back();

        ai_tool_fn c_fn = [](const char* tool_name, const char* input_json, void* user_data) -> ai_tool_result_t {
            auto* e = static_cast<PyToolEntry*>(user_data);
            ai_tool_result_t res = {};

            py::gil_scoped_acquire acquire;
            try {
                py::module_ json = py::module_::import("json");
                py::object input = json.attr("loads")(input_json);
                py::object result = e->callback(input);

                std::string output;
                if (py::isinstance<py::str>(result)) {
                    output = result.cast<std::string>();
                } else {
                    output = json.attr("dumps")(result).cast<std::string>();
                }

                char* out_copy = new char[output.size() + 1];
                std::copy(output.begin(), output.end(), out_copy);
                out_copy[output.size()] = '\0';
                res.output_json = out_copy;
                res.is_error = 0;
            } catch (const py::error_already_set& e_py) {
                std::string err = e_py.what();
                char* err_copy = new char[err.size() + 1];
                std::copy(err.begin(), err.end(), err_copy);
                err_copy[err.size()] = '\0';
                res.output_json = err_copy;
                res.is_error = 1;
            }
            return res;
        };

        ai_tool_set_add(tools_, name.c_str(), description.c_str(), schema_json.c_str(), c_fn, entry);
    }

    PyToolSet(const PyToolSet&) = delete;
    PyToolSet& operator=(const PyToolSet&) = delete;

private:
    ai_tool_set_t tools_;
    std::vector<PyToolEntry> entries_;
};

struct PyGenerateResult {
    std::string text;
    std::string finish_reason;
    int input_tokens;
    int output_tokens;
    int steps;
};

struct PyStreamEvent {
    std::string type;
    std::string text;
    std::string tool_name;
    std::string tool_call_id;
};

class PyAgent {
public:
    PyAgent(PyModel& model, PyToolSet& tools,
            const std::string& instructions = "", int max_steps = 50) {
        ai_agent_options_t opts = {};
        opts.model = model.handle();
        opts.tools = tools.handle();
        opts.instructions = instructions.c_str();
        opts.max_steps = max_steps;
        opts.on_event = nullptr;
        opts.user_data = nullptr;

        agent_ = ai_agent_create(opts);
        if (!agent_) throw std::runtime_error("Failed to create agent");
    }

    ~PyAgent() { if (agent_) ai_agent_destroy(agent_); }
    ai_agent_t handle() const { return agent_; }

    PyGenerateResult call(const std::string& prompt) {
        ai_generate_result_t result = {};
        ai_status_t status;

        {
            py::gil_scoped_release release;
            status = ai_agent_call(agent_, prompt.c_str(), &result);
        }

        if (status != AI_OK) {
            std::string msg = result.text ? result.text : "Agent call failed";
            ai_generate_result_free(&result);
            throw std::runtime_error(msg);
        }

        PyGenerateResult res;
        res.text = result.text ? result.text : "";
        res.finish_reason = result.finish_reason ? result.finish_reason : "stop";
        res.input_tokens = result.input_tokens;
        res.output_tokens = result.output_tokens;
        res.steps = result.steps;

        ai_generate_result_free(&result);
        return res;
    }

    std::vector<PyStreamEvent> call_stream(const std::string& prompt) {
        std::vector<PyStreamEvent> events;

        ai_stream_callback_fn cb = [](ai_stream_event_t event, void* user_data) {
            auto* evts = static_cast<std::vector<PyStreamEvent>*>(user_data);
            PyStreamEvent e;
            switch (event.type) {
                case AI_STREAM_TEXT_DELTA: e.type = "text_delta"; break;
                case AI_STREAM_TOOL_CALL_START: e.type = "tool_call_start"; break;
                case AI_STREAM_TOOL_CALL_DELTA: e.type = "tool_call_delta"; break;
                case AI_STREAM_TOOL_CALL_END: e.type = "tool_call_end"; break;
                case AI_STREAM_FINISH: e.type = "finish"; break;
                case AI_STREAM_ERROR: e.type = "error"; break;
                case AI_STREAM_STEP_FINISH: e.type = "step_finish"; break;
            }
            if (event.text) e.text = event.text;
            if (event.tool_name) e.tool_name = event.tool_name;
            if (event.tool_call_id) e.tool_call_id = event.tool_call_id;
            evts->push_back(std::move(e));
        };

        ai_status_t status;
        {
            py::gil_scoped_release release;
            status = ai_agent_call_stream(agent_, prompt.c_str(), cb, &events);
        }

        if (status != AI_OK) {
            throw std::runtime_error("Agent streaming call failed");
        }
        return events;
    }

    PyAgent(const PyAgent&) = delete;
    PyAgent& operator=(const PyAgent&) = delete;

private:
    ai_agent_t agent_;
};

static PyGenerateResult py_generate_text(
    PyModel& model,
    const std::string& prompt = "",
    const std::string& system = "",
    const std::string& messages_json = "",
    PyToolSet* tools = nullptr,
    int max_steps = 1,
    int max_output_tokens = 0,
    double temperature = -1.0
) {
    ai_generate_options_t opts = {};
    opts.model = model.handle();
    opts.temperature = temperature;
    opts.max_steps = max_steps;
    opts.max_output_tokens = max_output_tokens;

    if (!prompt.empty()) opts.prompt = prompt.c_str();
    if (!system.empty()) opts.system = system.c_str();
    if (!messages_json.empty()) opts.messages_json = messages_json.c_str();
    if (tools) opts.tools = tools->handle();

    ai_generate_result_t result = {};
    ai_status_t status;

    {
        py::gil_scoped_release release;
        status = ai_generate_text(opts, &result);
    }

    if (status != AI_OK) {
        std::string msg = result.text ? result.text : ai_status_message(status);
        ai_generate_result_free(&result);
        throw std::runtime_error(msg);
    }

    PyGenerateResult res;
    res.text = result.text ? result.text : "";
    res.finish_reason = result.finish_reason ? result.finish_reason : "stop";
    res.input_tokens = result.input_tokens;
    res.output_tokens = result.output_tokens;
    res.steps = result.steps;

    ai_generate_result_free(&result);
    return res;
}

static std::vector<PyStreamEvent> py_stream_text(
    PyModel& model,
    const std::string& prompt = "",
    const std::string& system = "",
    int max_output_tokens = 0,
    double temperature = -1.0
) {
    ai_generate_options_t opts = {};
    opts.model = model.handle();
    opts.temperature = temperature;
    opts.max_output_tokens = max_output_tokens;

    if (!prompt.empty()) opts.prompt = prompt.c_str();
    if (!system.empty()) opts.system = system.c_str();

    std::vector<PyStreamEvent> events;

    ai_stream_callback_fn cb = [](ai_stream_event_t event, void* user_data) {
        auto* evts = static_cast<std::vector<PyStreamEvent>*>(user_data);
        PyStreamEvent e;
        switch (event.type) {
            case AI_STREAM_TEXT_DELTA: e.type = "text_delta"; break;
            case AI_STREAM_TOOL_CALL_START: e.type = "tool_call_start"; break;
            case AI_STREAM_TOOL_CALL_DELTA: e.type = "tool_call_delta"; break;
            case AI_STREAM_TOOL_CALL_END: e.type = "tool_call_end"; break;
            case AI_STREAM_FINISH: e.type = "finish"; break;
            case AI_STREAM_ERROR: e.type = "error"; break;
            case AI_STREAM_STEP_FINISH: e.type = "step_finish"; break;
        }
        if (event.text) e.text = event.text;
        if (event.tool_name) e.tool_name = event.tool_name;
        if (event.tool_call_id) e.tool_call_id = event.tool_call_id;
        evts->push_back(std::move(e));
    };

    ai_status_t status;
    {
        py::gil_scoped_release release;
        status = ai_stream_text(opts, cb, &events);
    }

    if (status != AI_OK) {
        throw std::runtime_error("Stream text failed");
    }
    return events;
}

class PySession {
public:
    PySession(PyAgent& agent) {
        session_ = ai_session_create(agent.handle());
        if (!session_) throw std::runtime_error("Failed to create session");
    }
    ~PySession() { if (session_) ai_session_destroy(session_); }

    PyGenerateResult send(const std::string& prompt) {
        ai_generate_result_t result = {};
        ai_status_t status;
        {
            py::gil_scoped_release release;
            status = ai_session_send(session_, prompt.c_str(), &result);
        }
        if (status != AI_OK) {
            std::string msg = result.text ? result.text : ai_status_message(status);
            ai_generate_result_free(&result);
            throw std::runtime_error(msg);
        }
        PyGenerateResult res;
        res.text = result.text ? result.text : "";
        res.finish_reason = result.finish_reason ? result.finish_reason : "stop";
        res.input_tokens = result.input_tokens;
        res.output_tokens = result.output_tokens;
        res.steps = result.steps;
        ai_generate_result_free(&result);
        return res;
    }

    void send_stream(const std::string& prompt, py::function callback) {
        struct CallbackData {
            py::function callback;
        };
        CallbackData data{callback};

        ai_stream_callback_fn cb = [](ai_stream_event_t event, void* user_data) {
            auto* d = static_cast<CallbackData*>(user_data);
            py::gil_scoped_acquire acquire;
            try {
                PyStreamEvent e;
                switch (event.type) {
                    case AI_STREAM_TEXT_DELTA: e.type = "text_delta"; break;
                    case AI_STREAM_TOOL_CALL_START: e.type = "tool_call_start"; break;
                    case AI_STREAM_TOOL_CALL_DELTA: e.type = "tool_call_delta"; break;
                    case AI_STREAM_TOOL_CALL_END: e.type = "tool_call_end"; break;
                    case AI_STREAM_FINISH: e.type = "finish"; break;
                    case AI_STREAM_ERROR: e.type = "error"; break;
                    case AI_STREAM_STEP_FINISH: e.type = "step_finish"; break;
                    case AI_STREAM_REASONING_START: e.type = "reasoning_start"; break;
                    case AI_STREAM_REASONING_DELTA: e.type = "reasoning_delta"; break;
                    case AI_STREAM_REASONING_END: e.type = "reasoning_end"; break;
                    case AI_STREAM_TOOL_RESULT: e.type = "tool_result"; break;
                }
                if (event.text) e.text = event.text;
                if (event.tool_name) e.tool_name = event.tool_name;
                if (event.tool_call_id) e.tool_call_id = event.tool_call_id;
                d->callback(e);
            } catch (...) {
                // Ignore exceptions
            }
        };

        ai_status_t status;
        {
            py::gil_scoped_release release;
            status = ai_session_send_stream(session_, prompt.c_str(), cb, &data);
        }

        if (status != AI_OK) {
            throw std::runtime_error("Session streaming send failed");
        }
    }

    PySession(const PySession&) = delete;
    PySession& operator=(const PySession&) = delete;

private:
    ai_session_t session_;
};

static std::unique_ptr<PyToolSet> py_standard_toolkit() {
    return PyToolSet::wrap(ai_standard_toolkit_create());
}

static std::unique_ptr<PyToolSet> py_with_permissions(PyToolSet& tools, py::function policy_fn) {
    // policy_fn(tool: str, input_json: str) -> int  (0=allow, 1=deny, 2=ask)
    // NOTE: the function is heap-allocated and kept alive for the toolset's
    // lifetime (one allocation per call; not GC'd).
    auto* fn = new py::function(std::move(policy_fn));
    ai_permission_policy_fn c_fn = [](const char* tool, const char* input_json, void* ud) -> int {
        auto* f = static_cast<py::function*>(ud);
        py::gil_scoped_acquire acquire;
        try {
            return f->operator()(tool, input_json).cast<int>();
        } catch (...) {
            return AI_PERMISSION_DENY;  // fail closed
        }
    };
    return PyToolSet::wrap(ai_with_permissions(tools.handle(), c_fn, fn));
}

PYBIND11_MODULE(_native, m) {
    m.doc() = "AI SDK C++ native bindings for Python";

    py::class_<PyAiContext>(m, "Context")
        .def(py::init<>());

    py::class_<PyProvider>(m, "Provider")
        .def(py::init<PyAiContext&, const std::string&, const std::string&, const std::string&>(),
             py::arg("ctx"), py::arg("name"),
             py::arg("api_key") = "", py::arg("base_url") = "")
        .def("model", [](PyProvider& self, const std::string& model_id) {
            return std::make_unique<PyModel>(self, model_id);
        }, py::return_value_policy::move)
        .def("__call__", [](PyProvider& self, const std::string& model_id) {
            return std::make_unique<PyModel>(self, model_id);
        }, py::return_value_policy::move);

    py::class_<PyModel>(m, "Model")
        .def(py::init<PyProvider&, const std::string&>());

    py::class_<PyToolSet>(m, "ToolSet")
        .def(py::init<>())
        .def("add", &PyToolSet::add,
             py::arg("name"), py::arg("schema"),
             py::arg("description"), py::arg("fn"));

    py::class_<PyGenerateResult>(m, "GenerateResult")
        .def_readonly("text", &PyGenerateResult::text)
        .def_readonly("finish_reason", &PyGenerateResult::finish_reason)
        .def_readonly("input_tokens", &PyGenerateResult::input_tokens)
        .def_readonly("output_tokens", &PyGenerateResult::output_tokens)
        .def_readonly("steps", &PyGenerateResult::steps);

    py::class_<PyStreamEvent>(m, "StreamEvent")
        .def_readonly("type", &PyStreamEvent::type)
        .def_readonly("text", &PyStreamEvent::text)
        .def_readonly("tool_name", &PyStreamEvent::tool_name)
        .def_readonly("tool_call_id", &PyStreamEvent::tool_call_id);

    py::class_<PyAgent>(m, "Agent")
        .def(py::init<PyModel&, PyToolSet&, const std::string&, int>(),
             py::arg("model"), py::arg("tools"),
             py::arg("instructions") = "", py::arg("max_steps") = 50)
        .def("call", &PyAgent::call)
        .def("call_stream", &PyAgent::call_stream);

    py::class_<PySession>(m, "Session")
        .def(py::init<PyAgent&>(), py::arg("agent"))
        .def("send", &PySession::send, py::arg("prompt"))
        .def("send_stream", &PySession::send_stream, py::arg("prompt"), py::arg("callback"));

    m.def("standard_toolkit", &py_standard_toolkit,
          "Returns a ToolSet with read_file/write_file/edit_file/glob/grep/bash.");
    m.def("with_permissions", &py_with_permissions,
          py::arg("tools"), py::arg("policy"),
          "Wrap a ToolSet with a permission policy(policy)(tool, input_json) -> int (0=allow,1=deny,2=ask).");

    m.def("generate_text", &py_generate_text,
          py::arg("model"),
          py::arg("prompt") = "",
          py::arg("system") = "",
          py::arg("messages_json") = "",
          py::arg("tools") = nullptr,
          py::arg("max_steps") = 1,
          py::arg("max_output_tokens") = 0,
          py::arg("temperature") = -1.0);

    m.def("stream_text", &py_stream_text,
          py::arg("model"),
          py::arg("prompt") = "",
          py::arg("system") = "",
          py::arg("max_output_tokens") = 0,
          py::arg("temperature") = -1.0);

    m.def("version", []() { return std::string(ai_sdk_version()); });
}
