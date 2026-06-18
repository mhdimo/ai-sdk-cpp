#ifndef AI_SDK_H
#define AI_SDK_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stddef.h>

/* --------------------------------------------------------------------------
 * Opaque handles — users never see internals
 * -------------------------------------------------------------------------- */

typedef struct ai_context* ai_context_t;
typedef struct ai_provider* ai_provider_t;
typedef struct ai_model* ai_model_t;
typedef struct ai_agent* ai_agent_t;
typedef struct ai_tool_set* ai_tool_set_t;
typedef struct ai_stream* ai_stream_t;
typedef struct ai_result* ai_result_t;

/* --------------------------------------------------------------------------
 * Error handling
 * -------------------------------------------------------------------------- */

typedef enum {
    AI_OK = 0,
    AI_ERROR_INVALID_ARGUMENT = 1,
    AI_ERROR_API_CALL = 2,
    AI_ERROR_RATE_LIMIT = 3,
    AI_ERROR_AUTHENTICATION = 4,
    AI_ERROR_TIMEOUT = 5,
    AI_ERROR_STREAM = 6,
    AI_ERROR_CANCELLED = 7,
    AI_ERROR_INVALID_RESPONSE = 8,
    AI_ERROR_INTERNAL = 99,
} ai_status_t;

const char* ai_status_message(ai_status_t status);
const char* ai_last_error(ai_context_t ctx);

/* --------------------------------------------------------------------------
 * Context — owns the event loop, must be created first
 * -------------------------------------------------------------------------- */

ai_context_t ai_context_create(void);
void ai_context_destroy(ai_context_t ctx);

/* --------------------------------------------------------------------------
 * Providers
 * -------------------------------------------------------------------------- */

typedef struct {
    const char* api_key;       /* NULL = use env var */
    const char* base_url;      /* NULL = use default */
} ai_provider_options_t;

ai_provider_t ai_provider_create(ai_context_t ctx, const char* provider_name, ai_provider_options_t opts);
void ai_provider_destroy(ai_provider_t provider);

/* --------------------------------------------------------------------------
 * Models
 * -------------------------------------------------------------------------- */

ai_model_t ai_model_create(ai_provider_t provider, const char* model_id);
void ai_model_destroy(ai_model_t model);

/* --------------------------------------------------------------------------
 * Tool system
 * -------------------------------------------------------------------------- */

typedef struct {
    const char* output_json;   /* JSON string of the tool output */
    int is_error;              /* 0 = success, 1 = error */
} ai_tool_result_t;

typedef ai_tool_result_t (*ai_tool_fn)(const char* tool_name, const char* input_json, void* user_data);

ai_tool_set_t ai_tool_set_create(void);
void ai_tool_set_destroy(ai_tool_set_t tools);

ai_status_t ai_tool_set_add(
    ai_tool_set_t tools,
    const char* name,
    const char* description,
    const char* input_schema_json,
    ai_tool_fn callback,
    void* user_data
);

/* --------------------------------------------------------------------------
 * generate_text — synchronous (blocks until complete)
 * -------------------------------------------------------------------------- */

typedef struct {
    ai_model_t model;
    const char* prompt;            /* simple string prompt */
    const char* system;            /* system message, NULL for none */
    const char* messages_json;     /* full messages array as JSON, NULL to use prompt */
    ai_tool_set_t tools;           /* NULL for no tools */
    int max_steps;                 /* 0 = default (1) */
    int max_output_tokens;         /* 0 = model default */
    double temperature;            /* -1 = model default */
} ai_generate_options_t;

typedef struct {
    const char* text;
    const char* finish_reason;     /* "stop", "length", "tool_calls" */
    int input_tokens;
    int output_tokens;
    int steps;
} ai_generate_result_t;

ai_status_t ai_generate_text(ai_generate_options_t opts, ai_generate_result_t* result);
void ai_generate_result_free(ai_generate_result_t* result);

/* --------------------------------------------------------------------------
 * stream_text — streaming with callback
 * -------------------------------------------------------------------------- */

typedef enum {
    AI_STREAM_TEXT_DELTA = 0,
    AI_STREAM_TOOL_CALL_START = 1,
    AI_STREAM_TOOL_CALL_DELTA = 2,
    AI_STREAM_TOOL_CALL_END = 3,
    AI_STREAM_FINISH = 4,
    AI_STREAM_ERROR = 5,
    AI_STREAM_STEP_FINISH = 6,
} ai_stream_event_type_t;

typedef struct {
    ai_stream_event_type_t type;
    const char* text;           /* text delta or error message */
    const char* tool_name;      /* for tool events */
    const char* tool_call_id;   /* for tool events */
    const char* finish_reason;  /* for finish events */
} ai_stream_event_t;

typedef void (*ai_stream_callback_fn)(ai_stream_event_t event, void* user_data);

ai_status_t ai_stream_text(ai_generate_options_t opts, ai_stream_callback_fn callback, void* user_data);

/* --------------------------------------------------------------------------
 * Agent — high-level tool-loop agent
 * -------------------------------------------------------------------------- */

typedef struct {
    ai_model_t model;
    ai_tool_set_t tools;
    const char* instructions;
    int max_steps;
    ai_stream_callback_fn on_event;  /* NULL for no streaming */
    void* user_data;
} ai_agent_options_t;

ai_agent_t ai_agent_create(ai_agent_options_t opts);
void ai_agent_destroy(ai_agent_t agent);

ai_status_t ai_agent_call(ai_agent_t agent, const char* prompt, ai_generate_result_t* result);
ai_status_t ai_agent_call_stream(ai_agent_t agent, const char* prompt, ai_stream_callback_fn callback, void* user_data);

/* --------------------------------------------------------------------------
 * Batch — submit many requests, poll to completion, fetch results.
 * Only providers that support batching (e.g. Anthropic, OpenAI) can create a
 * batch handle; ai_batch_create returns NULL for unsupported providers.
 * ai_batch_run blocks until the batch reaches a terminal state.
 * -------------------------------------------------------------------------- */

typedef struct ai_batch* ai_batch_t;

typedef struct {
    const char* custom_id;     /* user-defined id to match the response */
    const char* prompt;        /* simple string prompt; NULL for none */
    const char* system;        /* optional system message; NULL for none */
    int max_output_tokens;     /* 0 = model default */
    double temperature;        /* -1 = model default */
} ai_batch_request_t;

typedef struct {
    const char* custom_id;
    const char* result_json;   /* generated text/JSON on success, NULL if error */
    const char* error;         /* error message, NULL on success */
} ai_batch_item_t;

typedef struct {
    const char* batch_id;
    ai_batch_item_t* items;    /* array of `count` items */
    int count;
    const char* status;        /* "completed", "failed", "cancelled", "expired", "other" */
    void* _storage;            /* internal; freed by ai_batch_result_free */
} ai_batch_result_t;

ai_batch_t ai_batch_create(ai_provider_t provider, const char* model_id);
void ai_batch_destroy(ai_batch_t batch);

ai_status_t ai_batch_run(
    ai_batch_t batch,
    const ai_batch_request_t* requests,
    int count,
    int poll_interval_ms,
    ai_batch_result_t* result
);
void ai_batch_result_free(ai_batch_result_t* result);

/* --------------------------------------------------------------------------
 * Utility
 * -------------------------------------------------------------------------- */

const char* ai_sdk_version(void);

#ifdef __cplusplus
}
#endif

#endif /* AI_SDK_H */
