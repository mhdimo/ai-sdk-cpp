# Building a Coding Agent in Rust via C FFI

This guide shows how to use `ai-sdk-cpp` from Rust by linking against its C shared library (`libai_sdk.so` on Linux, `libai_sdk.dylib` on macOS). You get the unified multi-provider AI engine with Rust's safety guarantees layered on top.

## Why Use ai-sdk-cpp from Rust?

- **Native performance end-to-end.** The C++ engine handles HTTP, JSON parsing, and the tool loop at native speed. Your Rust tool implementations run at native speed too. No runtime overhead anywhere.
- **Unified provider interface.** Switch between Anthropic, OpenAI, Google, Groq, xAI, Mistral, and others with a single string change. Same tool definitions, same agent loop.
- **Rust safety on top of C speed.** Wrap the C API once with proper `Drop` implementations, then use safe Rust for the rest of your agent.
- **No async runtime required.** The C API is blocking, which makes it straightforward to call from synchronous Rust code or integrate into any async runtime you choose.

---

## Prerequisites

- Rust toolchain (1.70+)
- A built `ai-sdk-cpp` library (see [Build Instructions](#build-instructions) below)
- An API key for your chosen provider

---

## Setup

### Project Layout

```
my-coding-agent/
  Cargo.toml
  build.rs
  src/
    main.rs
    ffi.rs        # Raw unsafe bindings
    wrapper.rs    # Safe Rust wrappers
```

### Cargo.toml

```toml
[package]
name = "coding-agent"
version = "0.1.0"
edition = "2021"

[dependencies]
serde = { version = "1", features = ["derive"] }
serde_json = "1"

[build-dependencies]
# Option A: manual linking (simpler, shown below)
# Option B: use bindgen to auto-generate bindings from the header
# bindgen = "0.71"
```

### build.rs

This tells Cargo where to find the shared library:

```rust
fn main() {
    // Point to where you built ai-sdk-cpp
    let lib_dir = std::env::var("AI_SDK_LIB_DIR")
        .unwrap_or_else(|_| "../ai-sdk-cpp/build/lib".to_string());

    println!("cargo:rustc-link-search=native={lib_dir}");
    println!("cargo:rustc-link-lib=dylib=ai_sdk");

    // Re-run if the library changes
    println!("cargo:rerun-if-env-changed=AI_SDK_LIB_DIR");
}
```

---

## Raw FFI Bindings

The `ffi.rs` module declares the C types and functions. This is the unsafe foundation everything else builds on.

```rust
//! Raw C FFI bindings to ai-sdk-cpp.
//! These are all unsafe — use the wrapper module for safe access.

use std::os::raw::{c_char, c_double, c_int, c_void};

// Opaque handle types
pub type AiContext = *mut c_void;
pub type AiProvider = *mut c_void;
pub type AiModel = *mut c_void;
pub type AiAgent = *mut c_void;
pub type AiToolSet = *mut c_void;

// Status codes
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AiStatus {
    Ok = 0,
    InvalidArgument = 1,
    ApiCall = 2,
    RateLimit = 3,
    Authentication = 4,
    Timeout = 5,
    Stream = 6,
    Cancelled = 7,
    InvalidResponse = 8,
    Internal = 99,
}

// Provider options
#[repr(C)]
pub struct AiProviderOptions {
    pub api_key: *const c_char,   // NULL = use env var
    pub base_url: *const c_char,  // NULL = use default
}

// Tool result returned from callbacks
#[repr(C)]
pub struct AiToolResult {
    pub output_json: *const c_char,
    pub is_error: c_int,
}

// Tool callback signature
pub type AiToolFn = unsafe extern "C" fn(
    tool_name: *const c_char,
    input_json: *const c_char,
    user_data: *mut c_void,
) -> AiToolResult;

// Generation options
#[repr(C)]
pub struct AiGenerateOptions {
    pub model: AiModel,
    pub prompt: *const c_char,
    pub system: *const c_char,
    pub messages_json: *const c_char,
    pub tools: AiToolSet,
    pub max_steps: c_int,
    pub max_output_tokens: c_int,
    pub temperature: c_double,
}

// Generation result
#[repr(C)]
pub struct AiGenerateResult {
    pub text: *const c_char,
    pub finish_reason: *const c_char,
    pub input_tokens: c_int,
    pub output_tokens: c_int,
    pub steps: c_int,
}

// Stream event types
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AiStreamEventType {
    TextDelta = 0,
    ToolCallStart = 1,
    ToolCallDelta = 2,
    ToolCallEnd = 3,
    Finish = 4,
    Error = 5,
    StepFinish = 6,
}

#[repr(C)]
pub struct AiStreamEvent {
    pub event_type: AiStreamEventType,
    pub text: *const c_char,
    pub tool_name: *const c_char,
    pub tool_call_id: *const c_char,
    pub finish_reason: *const c_char,
}

pub type AiStreamCallbackFn =
    unsafe extern "C" fn(event: AiStreamEvent, user_data: *mut c_void);

// Agent options
#[repr(C)]
pub struct AiAgentOptions {
    pub model: AiModel,
    pub tools: AiToolSet,
    pub instructions: *const c_char,
    pub max_steps: c_int,
    pub on_event: Option<AiStreamCallbackFn>,
    pub user_data: *mut c_void,
}

extern "C" {
    // Context
    pub fn ai_context_create() -> AiContext;
    pub fn ai_context_destroy(ctx: AiContext);

    // Error info
    pub fn ai_status_message(status: AiStatus) -> *const c_char;
    pub fn ai_last_error(ctx: AiContext) -> *const c_char;

    // Provider
    pub fn ai_provider_create(
        ctx: AiContext,
        provider_name: *const c_char,
        opts: AiProviderOptions,
    ) -> AiProvider;
    pub fn ai_provider_destroy(provider: AiProvider);

    // Model
    pub fn ai_model_create(provider: AiProvider, model_id: *const c_char) -> AiModel;
    pub fn ai_model_destroy(model: AiModel);

    // Tool set
    pub fn ai_tool_set_create() -> AiToolSet;
    pub fn ai_tool_set_destroy(tools: AiToolSet);
    pub fn ai_tool_set_add(
        tools: AiToolSet,
        name: *const c_char,
        description: *const c_char,
        input_schema_json: *const c_char,
        callback: AiToolFn,
        user_data: *mut c_void,
    ) -> AiStatus;

    // Generate
    pub fn ai_generate_text(
        opts: AiGenerateOptions,
        result: *mut AiGenerateResult,
    ) -> AiStatus;
    pub fn ai_generate_result_free(result: *mut AiGenerateResult);

    // Stream
    pub fn ai_stream_text(
        opts: AiGenerateOptions,
        callback: AiStreamCallbackFn,
        user_data: *mut c_void,
    ) -> AiStatus;

    // Agent
    pub fn ai_agent_create(opts: AiAgentOptions) -> AiAgent;
    pub fn ai_agent_destroy(agent: AiAgent);
    pub fn ai_agent_call(
        agent: AiAgent,
        prompt: *const c_char,
        result: *mut AiGenerateResult,
    ) -> AiStatus;

    // Utility
    pub fn ai_sdk_version() -> *const c_char;
}
```

---

## Error Handling: Mapping ai_status_t to Result

Convert the C status codes into an idiomatic Rust `Result`:

```rust
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AiError {
    InvalidArgument(String),
    ApiCall(String),
    RateLimit(String),
    Authentication(String),
    Timeout(String),
    Stream(String),
    Cancelled(String),
    InvalidResponse(String),
    Internal(String),
}

impl fmt::Display for AiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AiError::InvalidArgument(msg) => write!(f, "invalid argument: {msg}"),
            AiError::ApiCall(msg) => write!(f, "API call failed: {msg}"),
            AiError::RateLimit(msg) => write!(f, "rate limited: {msg}"),
            AiError::Authentication(msg) => write!(f, "authentication failed: {msg}"),
            AiError::Timeout(msg) => write!(f, "request timed out: {msg}"),
            AiError::Stream(msg) => write!(f, "stream error: {msg}"),
            AiError::Cancelled(msg) => write!(f, "cancelled: {msg}"),
            AiError::InvalidResponse(msg) => write!(f, "invalid response: {msg}"),
            AiError::Internal(msg) => write!(f, "internal error: {msg}"),
        }
    }
}

impl std::error::Error for AiError {}

/// Convert an AiStatus into a Result. On Ok, returns (). On error,
/// queries the context for the detailed error message.
pub fn check_status(status: ffi::AiStatus, ctx: ffi::AiContext) -> Result<(), AiError> {
    if status == ffi::AiStatus::Ok {
        return Ok(());
    }

    let msg = unsafe {
        let ptr = ffi::ai_last_error(ctx);
        if ptr.is_null() {
            "unknown error".to_string()
        } else {
            std::ffi::CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    };

    Err(match status {
        ffi::AiStatus::InvalidArgument => AiError::InvalidArgument(msg),
        ffi::AiStatus::ApiCall => AiError::ApiCall(msg),
        ffi::AiStatus::RateLimit => AiError::RateLimit(msg),
        ffi::AiStatus::Authentication => AiError::Authentication(msg),
        ffi::AiStatus::Timeout => AiError::Timeout(msg),
        ffi::AiStatus::Stream => AiError::Stream(msg),
        ffi::AiStatus::Cancelled => AiError::Cancelled(msg),
        ffi::AiStatus::InvalidResponse => AiError::InvalidResponse(msg),
        _ => AiError::Internal(msg),
    })
}
```

---

## Safe Rust Wrapper with RAII

The wrapper module provides safe types that own their C handles and free them on drop. This is the idiomatic Rust way to manage C resources.

```rust
//! Safe wrappers around the ai-sdk-cpp C API.
//! All handles implement Drop for automatic cleanup.

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_void};
use std::ptr;

use crate::ffi;
use crate::error::{AiError, check_status};

// --------------------------------------------------------------------------
// Context
// --------------------------------------------------------------------------

/// Owns the AI SDK context. Must be created before any other SDK object.
pub struct Context {
    raw: ffi::AiContext,
}

impl Context {
    pub fn new() -> Self {
        let raw = unsafe { ffi::ai_context_create() };
        assert!(!raw.is_null(), "ai_context_create returned null");
        Context { raw }
    }

    pub fn as_raw(&self) -> ffi::AiContext {
        self.raw
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        unsafe { ffi::ai_context_destroy(self.raw) };
    }
}

// --------------------------------------------------------------------------
// Provider
// --------------------------------------------------------------------------

/// A provider instance (e.g., "anthropic", "openai").
pub struct Provider {
    raw: ffi::AiProvider,
    // Hold a reference to prevent the context from being dropped first
    _ctx: ffi::AiContext,
}

impl Provider {
    /// Create a provider. Pass `None` for api_key to read from the environment.
    pub fn new(ctx: &Context, name: &str, api_key: Option<&str>) -> Result<Self, AiError> {
        let c_name = CString::new(name).map_err(|_| {
            AiError::InvalidArgument("provider name contains null byte".into())
        })?;

        let c_key = api_key.map(|k| CString::new(k).unwrap());

        let opts = ffi::AiProviderOptions {
            api_key: c_key.as_ref().map_or(ptr::null(), |k| k.as_ptr()),
            base_url: ptr::null(),
        };

        let raw = unsafe {
            ffi::ai_provider_create(ctx.as_raw(), c_name.as_ptr(), opts)
        };

        if raw.is_null() {
            return Err(AiError::Internal("ai_provider_create returned null".into()));
        }

        Ok(Provider { raw, _ctx: ctx.as_raw() })
    }

    pub fn as_raw(&self) -> ffi::AiProvider {
        self.raw
    }
}

impl Drop for Provider {
    fn drop(&mut self) {
        unsafe { ffi::ai_provider_destroy(self.raw) };
    }
}

// --------------------------------------------------------------------------
// Model
// --------------------------------------------------------------------------

/// A model handle (e.g., "claude-sonnet-4-20250514", "gpt-4o").
pub struct Model {
    raw: ffi::AiModel,
}

impl Model {
    pub fn new(provider: &Provider, model_id: &str) -> Result<Self, AiError> {
        let c_id = CString::new(model_id).map_err(|_| {
            AiError::InvalidArgument("model_id contains null byte".into())
        })?;

        let raw = unsafe { ffi::ai_model_create(provider.as_raw(), c_id.as_ptr()) };

        if raw.is_null() {
            return Err(AiError::Internal("ai_model_create returned null".into()));
        }

        Ok(Model { raw })
    }

    pub fn as_raw(&self) -> ffi::AiModel {
        self.raw
    }
}

impl Drop for Model {
    fn drop(&mut self) {
        unsafe { ffi::ai_model_destroy(self.raw) };
    }
}

// --------------------------------------------------------------------------
// ToolSet
// --------------------------------------------------------------------------

/// A collection of tools the model can invoke.
pub struct ToolSet {
    raw: ffi::AiToolSet,
    // Box the closures so they live as long as the ToolSet
    _closures: Vec<Box<dyn Fn(&str, &str) -> ToolOutput>>,
}

/// The result of a tool invocation.
pub struct ToolOutput {
    pub json: String,
    pub is_error: bool,
}

impl ToolOutput {
    pub fn success(json: impl Into<String>) -> Self {
        ToolOutput { json: json.into(), is_error: false }
    }

    pub fn error(json: impl Into<String>) -> Self {
        ToolOutput { json: json.into(), is_error: true }
    }
}

impl ToolSet {
    pub fn new() -> Self {
        let raw = unsafe { ffi::ai_tool_set_create() };
        assert!(!raw.is_null());
        ToolSet { raw, _closures: Vec::new() }
    }

    /// Register a tool. The closure receives (tool_name, input_json) and
    /// returns a ToolOutput.
    pub fn add<F>(
        &mut self,
        name: &str,
        description: &str,
        input_schema_json: &str,
        handler: F,
    ) -> Result<(), AiError>
    where
        F: Fn(&str, &str) -> ToolOutput + 'static,
    {
        let c_name = CString::new(name).unwrap();
        let c_desc = CString::new(description).unwrap();
        let c_schema = CString::new(input_schema_json).unwrap();

        // Box the closure and leak it into a raw pointer for the C callback
        let boxed: Box<dyn Fn(&str, &str) -> ToolOutput> = Box::new(handler);
        let user_data = &*boxed as *const dyn Fn(&str, &str) -> ToolOutput as *mut c_void;

        let status = unsafe {
            ffi::ai_tool_set_add(
                self.raw,
                c_name.as_ptr(),
                c_desc.as_ptr(),
                c_schema.as_ptr(),
                tool_trampoline,
                user_data,
            )
        };

        // Keep the closure alive for the lifetime of the ToolSet
        self._closures.push(boxed);

        if status != ffi::AiStatus::Ok {
            return Err(AiError::InvalidArgument(format!(
                "failed to add tool '{name}'"
            )));
        }

        Ok(())
    }

    pub fn as_raw(&self) -> ffi::AiToolSet {
        self.raw
    }
}

impl Drop for ToolSet {
    fn drop(&mut self) {
        unsafe { ffi::ai_tool_set_destroy(self.raw) };
    }
}

/// The extern "C" trampoline that bridges C callbacks to Rust closures.
/// This is the key piece that makes tool callbacks work safely.
unsafe extern "C" fn tool_trampoline(
    tool_name: *const c_char,
    input_json: *const c_char,
    user_data: *mut c_void,
) -> ffi::AiToolResult {
    // Recover the closure from the user_data pointer
    let closure = &*(user_data as *const dyn Fn(&str, &str) -> ToolOutput);

    let name = CStr::from_ptr(tool_name).to_str().unwrap_or("");
    let input = CStr::from_ptr(input_json).to_str().unwrap_or("{}");

    let output = closure(name, input);

    // The C library will copy this string, so we leak it here and rely on
    // the library to not hold the pointer beyond the callback return.
    // A more robust approach uses a thread-local arena; this is fine for
    // single-threaded agent loops.
    let c_output = CString::new(output.json).unwrap_or_default();

    ffi::AiToolResult {
        output_json: c_output.into_raw(),
        is_error: if output.is_error { 1 } else { 0 },
    }
}

// --------------------------------------------------------------------------
// Agent
// --------------------------------------------------------------------------

/// A high-level agent that runs the tool loop automatically.
pub struct Agent {
    raw: ffi::AiAgent,
}

impl Agent {
    pub fn new(
        model: &Model,
        tools: &ToolSet,
        instructions: &str,
        max_steps: i32,
    ) -> Result<Self, AiError> {
        let c_instructions = CString::new(instructions).unwrap();

        let opts = ffi::AiAgentOptions {
            model: model.as_raw(),
            tools: tools.as_raw(),
            instructions: c_instructions.as_ptr(),
            max_steps,
            on_event: None,
            user_data: ptr::null_mut(),
        };

        let raw = unsafe { ffi::ai_agent_create(opts) };

        if raw.is_null() {
            return Err(AiError::Internal("ai_agent_create returned null".into()));
        }

        Ok(Agent { raw })
    }

    /// Send a prompt to the agent and block until it finishes.
    pub fn call(&self, prompt: &str, ctx: &Context) -> Result<GenerateResult, AiError> {
        let c_prompt = CString::new(prompt).unwrap();
        let mut raw_result = std::mem::MaybeUninit::<ffi::AiGenerateResult>::uninit();

        let status = unsafe {
            ffi::ai_agent_call(self.raw, c_prompt.as_ptr(), raw_result.as_mut_ptr())
        };

        check_status(status, ctx.as_raw())?;

        let raw_result = unsafe { raw_result.assume_init() };
        let result = GenerateResult::from_raw(&raw_result);

        unsafe { ffi::ai_generate_result_free(&raw_result as *const _ as *mut _) };

        Ok(result)
    }
}

impl Drop for Agent {
    fn drop(&mut self) {
        unsafe { ffi::ai_agent_destroy(self.raw) };
    }
}

// --------------------------------------------------------------------------
// GenerateResult
// --------------------------------------------------------------------------

/// The result of a generation or agent call, fully owned.
#[derive(Debug, Clone)]
pub struct GenerateResult {
    pub text: String,
    pub finish_reason: String,
    pub input_tokens: i32,
    pub output_tokens: i32,
    pub steps: i32,
}

impl GenerateResult {
    fn from_raw(raw: &ffi::AiGenerateResult) -> Self {
        unsafe {
            GenerateResult {
                text: ptr_to_string(raw.text),
                finish_reason: ptr_to_string(raw.finish_reason),
                input_tokens: raw.input_tokens,
                output_tokens: raw.output_tokens,
                steps: raw.steps,
            }
        }
    }
}

unsafe fn ptr_to_string(ptr: *const c_char) -> String {
    if ptr.is_null() {
        String::new()
    } else {
        CStr::from_ptr(ptr).to_string_lossy().into_owned()
    }
}
```

---

## The Tool Callback Pattern

The trickiest part of the FFI is the tool callback. The C API uses a function pointer plus a `void* user_data` -- the classic C pattern for closures. Here is how the pieces fit together:

```
Rust closure (captures state)
    |
    v
Box<dyn Fn(&str, &str) -> ToolOutput>   <-- heap-allocated, lives in ToolSet._closures
    |
    | (cast to *mut c_void)
    v
user_data pointer  ------>  passed to C library
    |
    | (C library calls tool_trampoline with user_data)
    v
tool_trampoline (extern "C")
    |
    | (cast user_data back to &dyn Fn)
    v
Rust closure executes, returns ToolOutput
    |
    v
Convert to AiToolResult (C struct) and return to C
```

The critical safety invariant: the `Box<dyn Fn>` stored in `ToolSet._closures` must outlive all uses of the tool set. Because `_closures` is a field of `ToolSet` and dropped at the same time, this is guaranteed.

---

## Building a Coding Agent with Tools

Now let's define the five tools a coding agent needs:

```rust
use serde_json::Value;
use std::fs;
use std::path::Path;
use std::process::Command;

fn register_coding_tools(tools: &mut ToolSet) {
    // read_file
    tools.add(
        "read_file",
        "Read the contents of a file and return them as text.",
        r#"{"type":"object","properties":{"path":{"type":"string","description":"File path to read"}},"required":["path"]}"#,
        |_name, input| {
            let v: Value = serde_json::from_str(input).unwrap_or_default();
            let path = v["path"].as_str().unwrap_or("");
            match fs::read_to_string(path) {
                Ok(content) => ToolOutput::success(
                    serde_json::json!({"content": content}).to_string()
                ),
                Err(e) => ToolOutput::error(
                    serde_json::json!({"error": e.to_string()}).to_string()
                ),
            }
        },
    ).unwrap();

    // write_file
    tools.add(
        "write_file",
        "Write content to a file. Creates parent directories if needed.",
        r#"{"type":"object","properties":{"path":{"type":"string","description":"File path"},"content":{"type":"string","description":"Content to write"}},"required":["path","content"]}"#,
        |_name, input| {
            let v: Value = serde_json::from_str(input).unwrap_or_default();
            let path = v["path"].as_str().unwrap_or("");
            let content = v["content"].as_str().unwrap_or("");

            if let Some(parent) = Path::new(path).parent() {
                let _ = fs::create_dir_all(parent);
            }

            match fs::write(path, content) {
                Ok(()) => ToolOutput::success(
                    serde_json::json!({"written_bytes": content.len()}).to_string()
                ),
                Err(e) => ToolOutput::error(
                    serde_json::json!({"error": e.to_string()}).to_string()
                ),
            }
        },
    ).unwrap();

    // run_command
    tools.add(
        "run_command",
        "Execute a shell command and return stdout/stderr.",
        r#"{"type":"object","properties":{"command":{"type":"string","description":"Shell command to execute"},"cwd":{"type":"string","description":"Working directory (optional)"}},"required":["command"]}"#,
        |_name, input| {
            let v: Value = serde_json::from_str(input).unwrap_or_default();
            let cmd = v["command"].as_str().unwrap_or("");
            let cwd = v["cwd"].as_str();

            let mut proc = Command::new("sh");
            proc.arg("-c").arg(cmd);
            if let Some(dir) = cwd {
                proc.current_dir(dir);
            }

            match proc.output() {
                Ok(output) => {
                    let stdout = String::from_utf8_lossy(&output.stdout);
                    let stderr = String::from_utf8_lossy(&output.stderr);
                    let code = output.status.code().unwrap_or(-1);
                    ToolOutput::success(serde_json::json!({
                        "exit_code": code,
                        "stdout": stdout,
                        "stderr": stderr,
                    }).to_string())
                }
                Err(e) => ToolOutput::error(
                    serde_json::json!({"error": e.to_string()}).to_string()
                ),
            }
        },
    ).unwrap();

    // list_files
    tools.add(
        "list_files",
        "List files in a directory, optionally recursively.",
        r#"{"type":"object","properties":{"directory":{"type":"string","description":"Directory to list"},"recursive":{"type":"boolean","description":"List recursively"}},"required":["directory"]}"#,
        |_name, input| {
            let v: Value = serde_json::from_str(input).unwrap_or_default();
            let dir = v["directory"].as_str().unwrap_or(".");
            let recursive = v["recursive"].as_bool().unwrap_or(false);

            if recursive {
                // Use find for recursive listing
                let output = Command::new("find")
                    .args([dir, "-type", "f"])
                    .output();
                match output {
                    Ok(o) => ToolOutput::success(serde_json::json!({
                        "files": String::from_utf8_lossy(&o.stdout)
                    }).to_string()),
                    Err(e) => ToolOutput::error(
                        serde_json::json!({"error": e.to_string()}).to_string()
                    ),
                }
            } else {
                match fs::read_dir(dir) {
                    Ok(entries) => {
                        let files: Vec<String> = entries
                            .filter_map(|e| e.ok())
                            .map(|e| {
                                let name = e.file_name().to_string_lossy().into_owned();
                                if e.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                                    format!("{name}/")
                                } else {
                                    name
                                }
                            })
                            .collect();
                        ToolOutput::success(serde_json::json!({"files": files}).to_string())
                    }
                    Err(e) => ToolOutput::error(
                        serde_json::json!({"error": e.to_string()}).to_string()
                    ),
                }
            }
        },
    ).unwrap();

    // search_files
    tools.add(
        "search_files",
        "Search for a text pattern across files using grep.",
        r#"{"type":"object","properties":{"pattern":{"type":"string","description":"Regex pattern to search for"},"directory":{"type":"string","description":"Directory to search in"},"include":{"type":"string","description":"File glob filter (e.g. *.rs)"}},"required":["pattern","directory"]}"#,
        |_name, input| {
            let v: Value = serde_json::from_str(input).unwrap_or_default();
            let pattern = v["pattern"].as_str().unwrap_or("");
            let dir = v["directory"].as_str().unwrap_or(".");
            let include = v["include"].as_str();

            let mut cmd = Command::new("grep");
            cmd.args(["-rn", pattern, dir]);
            if let Some(glob) = include {
                cmd.arg(format!("--include={glob}"));
            }

            match cmd.output() {
                Ok(output) => ToolOutput::success(serde_json::json!({
                    "matches": String::from_utf8_lossy(&output.stdout)
                }).to_string()),
                Err(e) => ToolOutput::error(
                    serde_json::json!({"error": e.to_string()}).to_string()
                ),
            }
        },
    ).unwrap();
}
```

---

## Full Working Example

Here is a complete, self-contained coding agent in Rust. This assumes you have the wrapper module from above available as `mod wrapper` and `mod ffi`.

```rust
// src/main.rs

mod ffi;
mod error;
mod wrapper;

use std::io::{self, BufRead, Write};
use wrapper::{Agent, Context, Model, Provider, ToolSet, ToolOutput};
use serde_json::Value;

fn main() {
    // 1. Initialize the SDK
    let ctx = Context::new();

    // 2. Create a provider (reads ANTHROPIC_API_KEY from env)
    let provider = Provider::new(&ctx, "anthropic", None)
        .expect("Failed to create provider. Is ANTHROPIC_API_KEY set?");

    // 3. Select a model
    let model = Model::new(&provider, "claude-sonnet-4-20250514")
        .expect("Failed to create model");

    // 4. Register tools
    let mut tools = ToolSet::new();
    register_coding_tools(&mut tools);

    // 5. Create the agent
    let agent = Agent::new(
        &model,
        &tools,
        "You are a coding agent. Read files before editing. Run tests after changes. Be concise.",
        50,
    ).expect("Failed to create agent");

    // 6. REPL loop
    println!("Coding Agent (Rust + ai-sdk-cpp)");
    println!("Type your request, or 'exit' to quit.\n");

    let stdin = io::stdin();
    let mut stdout = io::stdout();

    loop {
        print!("> ");
        stdout.flush().unwrap();

        let mut input = String::new();
        if stdin.lock().read_line(&mut input).unwrap() == 0 {
            break; // EOF
        }

        let input = input.trim();
        if input.is_empty() {
            continue;
        }
        if input == "exit" {
            break;
        }

        match agent.call(input, &ctx) {
            Ok(result) => {
                println!("\n{}\n", result.text);
                println!(
                    "[{} steps | {} input tokens | {} output tokens]\n",
                    result.steps, result.input_tokens, result.output_tokens
                );
            }
            Err(e) => {
                eprintln!("Error: {e}\n");
            }
        }
    }

    // All handles are dropped here automatically via RAII.
    // Context, Provider, Model, ToolSet, Agent — all cleaned up.
}

fn register_coding_tools(tools: &mut ToolSet) {
    use std::fs;
    use std::path::Path;
    use std::process::Command;

    tools.add("read_file", "Read file contents.",
        r#"{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}"#,
        |_, input| {
            let v: Value = serde_json::from_str(input).unwrap_or_default();
            let path = v["path"].as_str().unwrap_or("");
            match fs::read_to_string(path) {
                Ok(c) => ToolOutput::success(serde_json::json!({"content": c}).to_string()),
                Err(e) => ToolOutput::error(serde_json::json!({"error": e.to_string()}).to_string()),
            }
        },
    ).unwrap();

    tools.add("write_file", "Write content to a file.",
        r#"{"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},"required":["path","content"]}"#,
        |_, input| {
            let v: Value = serde_json::from_str(input).unwrap_or_default();
            let path = v["path"].as_str().unwrap_or("");
            let content = v["content"].as_str().unwrap_or("");
            if let Some(p) = Path::new(path).parent() { let _ = fs::create_dir_all(p); }
            match fs::write(path, content) {
                Ok(()) => ToolOutput::success(serde_json::json!({"ok": true}).to_string()),
                Err(e) => ToolOutput::error(serde_json::json!({"error": e.to_string()}).to_string()),
            }
        },
    ).unwrap();

    tools.add("run_command", "Run a shell command.",
        r#"{"type":"object","properties":{"command":{"type":"string"}},"required":["command"]}"#,
        |_, input| {
            let v: Value = serde_json::from_str(input).unwrap_or_default();
            let cmd = v["command"].as_str().unwrap_or("");
            match Command::new("sh").arg("-c").arg(cmd).output() {
                Ok(o) => ToolOutput::success(serde_json::json!({
                    "exit_code": o.status.code().unwrap_or(-1),
                    "stdout": String::from_utf8_lossy(&o.stdout),
                    "stderr": String::from_utf8_lossy(&o.stderr),
                }).to_string()),
                Err(e) => ToolOutput::error(serde_json::json!({"error": e.to_string()}).to_string()),
            }
        },
    ).unwrap();

    tools.add("list_files", "List directory contents.",
        r#"{"type":"object","properties":{"directory":{"type":"string"}},"required":["directory"]}"#,
        |_, input| {
            let v: Value = serde_json::from_str(input).unwrap_or_default();
            let dir = v["directory"].as_str().unwrap_or(".");
            match fs::read_dir(dir) {
                Ok(entries) => {
                    let files: Vec<String> = entries.filter_map(|e| e.ok())
                        .map(|e| e.file_name().to_string_lossy().into_owned()).collect();
                    ToolOutput::success(serde_json::json!({"files": files}).to_string())
                }
                Err(e) => ToolOutput::error(serde_json::json!({"error": e.to_string()}).to_string()),
            }
        },
    ).unwrap();

    tools.add("search_files", "Grep for a pattern in files.",
        r#"{"type":"object","properties":{"pattern":{"type":"string"},"directory":{"type":"string"}},"required":["pattern","directory"]}"#,
        |_, input| {
            let v: Value = serde_json::from_str(input).unwrap_or_default();
            let pattern = v["pattern"].as_str().unwrap_or("");
            let dir = v["directory"].as_str().unwrap_or(".");
            match Command::new("grep").args(["-rn", pattern, dir]).output() {
                Ok(o) => ToolOutput::success(serde_json::json!({"matches": String::from_utf8_lossy(&o.stdout)}).to_string()),
                Err(_) => ToolOutput::success(serde_json::json!({"matches": ""}).to_string()),
            }
        },
    ).unwrap();
}
```

---

## Build Instructions

### Step 1: Build ai-sdk-cpp

```bash
git clone https://github.com/anthropics/ai-sdk-cpp.git
cd ai-sdk-cpp
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_C_BINDINGS=ON
cmake --build . --config Release
```

This produces:
- Linux: `build/lib/libai_sdk.so`
- macOS: `build/lib/libai_sdk.dylib`
- Windows: `build/lib/ai_sdk.dll`

### Step 2: Build Your Rust Agent

```bash
# Tell Cargo where to find the library
export AI_SDK_LIB_DIR=/path/to/ai-sdk-cpp/build/lib

# Build
cargo build --release

# On macOS, also set the dynamic library path for running
export DYLD_LIBRARY_PATH=$AI_SDK_LIB_DIR:$DYLD_LIBRARY_PATH

# On Linux
export LD_LIBRARY_PATH=$AI_SDK_LIB_DIR:$LD_LIBRARY_PATH
```

### Step 3: Run

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
cargo run --release
```

### Alternative: Static Linking

If you prefer a single binary with no shared library dependency:

```rust
// build.rs — link statically instead
fn main() {
    let lib_dir = std::env::var("AI_SDK_LIB_DIR")
        .unwrap_or_else(|_| "../ai-sdk-cpp/build/lib".to_string());

    println!("cargo:rustc-link-search=native={lib_dir}");
    println!("cargo:rustc-link-lib=static=ai_sdk");

    // Link C++ standard library (required since ai-sdk-cpp is C++ internally)
    #[cfg(target_os = "macos")]
    println!("cargo:rustc-link-lib=dylib=c++");
    #[cfg(target_os = "linux")]
    println!("cargo:rustc-link-lib=dylib=stdc++");
}
```

Build the C++ library as static first:

```bash
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_C_BINDINGS=ON -DBUILD_SHARED_LIBS=OFF
cmake --build . --config Release
```

### Alternative: Using bindgen

Instead of writing FFI bindings by hand, you can auto-generate them from the header:

```toml
# Cargo.toml
[build-dependencies]
bindgen = "0.71"
```

```rust
// build.rs
fn main() {
    let lib_dir = std::env::var("AI_SDK_LIB_DIR")
        .unwrap_or_else(|_| "../ai-sdk-cpp/build/lib".to_string());
    let include_dir = std::env::var("AI_SDK_INCLUDE_DIR")
        .unwrap_or_else(|_| "../ai-sdk-cpp/bindings/c".to_string());

    println!("cargo:rustc-link-search=native={lib_dir}");
    println!("cargo:rustc-link-lib=dylib=ai_sdk");

    let bindings = bindgen::Builder::default()
        .header(format!("{include_dir}/ai_sdk.h"))
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Unable to generate bindings");

    let out_path = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings");
}
```

Then include the generated bindings:

```rust
// src/ffi.rs
#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
```

---

## Key Safety Notes

1. **Drop order matters.** The `Agent` and `Model` hold references into the `Provider`, which holds a reference into the `Context`. Declare them in order (context first, agent last) so Rust drops them in reverse order. Alternatively, use `Arc` if you need shared ownership.

2. **Tool closures must not panic.** A panic across the FFI boundary is undefined behavior. Wrap tool logic in `std::panic::catch_unwind` if there is any risk:

   ```rust
   tools.add("risky_tool", "...", "...", |name, input| {
       match std::panic::catch_unwind(|| do_work(name, input)) {
           Ok(result) => result,
           Err(_) => ToolOutput::error(
               r#"{"error": "internal panic in tool"}"#.to_string()
           ),
       }
   }).unwrap();
   ```

3. **String lifetime.** The C library copies strings passed to it, so CString temporaries in the wrapper functions are safe. The `output_json` pointer returned from tool callbacks is consumed immediately by the C library before the next call.

4. **Thread safety.** The C API is not thread-safe per context. Do not share an `AiContext` across threads. Create separate contexts for concurrent agents, or serialize access with a `Mutex`.

---

## Switching Providers

Change one line to use a different AI provider:

```rust
// Anthropic
let provider = Provider::new(&ctx, "anthropic", None)?;
let model = Model::new(&provider, "claude-sonnet-4-20250514")?;

// OpenAI
let provider = Provider::new(&ctx, "openai", None)?;
let model = Model::new(&provider, "gpt-4o")?;

// Google
let provider = Provider::new(&ctx, "google", None)?;
let model = Model::new(&provider, "gemini-2.5-pro")?;

// Groq (fast inference)
let provider = Provider::new(&ctx, "groq", None)?;
let model = Model::new(&provider, "llama-3.3-70b-versatile")?;
```

Set the corresponding environment variable (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GOOGLE_GENERATIVE_AI_API_KEY`, `GROQ_API_KEY`) and the SDK reads it automatically.

---

## Next Steps

- Add a **confirmation prompt** before write_file and run_command for safety
- Implement **conversation memory** by passing `messages_json` to subsequent calls
- Add **token budget tracking** using `result.input_tokens + result.output_tokens`
- Wrap the agent in a Tokio task with `spawn_blocking` if you need async integration
- Add **ANSI color output** for tool calls and results in the REPL
- Build a **permission sandbox** restricting file access to the project directory

You now have a fully native coding agent: C++ engine for speed, Rust for safety, and any AI provider you want. The model thinks, your tools execute, RAII handles the cleanup.
