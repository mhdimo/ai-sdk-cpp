//! Rust FFI bindings for ai-sdk-cpp.
//!
//! Link against `libai_sdk` (the C shared library). See the build instructions
//! in the repository README.

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int, c_void};

// --- Opaque handle types ---
pub type AiContext = *mut c_void;
pub type AiProvider = *mut c_void;
pub type AiModel = *mut c_void;
pub type AiAgent = *mut c_void;
pub type AiSession = *mut c_void;
pub type AiToolSet = *mut c_void;

// --- Status ---
pub const AI_OK: c_int = 0;

// --- FFI declarations ---
extern "C" {
    fn ai_context_create() -> AiContext;
    fn ai_context_destroy(ctx: AiContext);
    fn ai_provider_create(
        ctx: AiContext, name: *const c_char, opts: AiProviderOptions,
    ) -> AiProvider;
    fn ai_provider_destroy(p: AiProvider);
    fn ai_model_create(p: AiProvider, model_id: *const c_char) -> AiModel;
    fn ai_model_destroy(m: AiModel);
    fn ai_generate_text(opts: AiGenerateOptions, result: *mut AiGenerateResult) -> c_int;
    fn ai_generate_result_free(result: *mut AiGenerateResult);
    fn ai_session_create(agent: AiAgent) -> AiSession;
    fn ai_session_destroy(s: AiSession);
    fn ai_session_send(s: AiSession, prompt: *const c_char, result: *mut AiGenerateResult) -> c_int;
    fn ai_standard_toolkit_create() -> AiToolSet;
    fn ai_tool_set_destroy(ts: AiToolSet);
    fn ai_sdk_version() -> *const c_char;
}

#[repr(C)]
pub struct AiProviderOptions {
    pub api_key: *const c_char,
    pub base_url: *const c_char,
}

#[repr(C)]
pub struct AiGenerateOptions {
    pub model: AiModel,
    pub prompt: *const c_char,
    pub system: *const c_char,
    pub messages_json: *const c_char,
    pub tools: AiToolSet,
    pub max_steps: c_int,
    pub max_output_tokens: c_int,
    pub temperature: f64,
}

#[repr(C)]
pub struct AiGenerateResult {
    pub text: *const c_char,
    pub finish_reason: *const c_char,
    pub input_tokens: c_int,
    pub output_tokens: c_int,
    pub steps: c_int,
}

// --- Safe wrappers ---

pub struct Context(AiContext);
impl Context {
    pub fn new() -> Self { unsafe { Self(ai_context_create()) } }
    pub fn as_ptr(&self) -> AiContext { self.0 }
}
impl Drop for Context {
    fn drop(&mut self) { unsafe { ai_context_destroy(self.0) } }
}

pub struct GenerateResult {
    pub text: String,
    pub finish_reason: String,
    pub input_tokens: i32,
    pub output_tokens: i32,
    pub steps: i32,
}

pub fn generate_text(model: AiModel, prompt: &str) -> Result<GenerateResult, String> {
    let c_prompt = CString::new(prompt).map_err(|e| e.to_string())?;
    let opts = AiGenerateOptions {
        model,
        prompt: c_prompt.as_ptr(),
        system: std::ptr::null(),
        messages_json: std::ptr::null(),
        tools: std::ptr::null_mut(),
        max_steps: 1,
        max_output_tokens: 0,
        temperature: -1.0,
    };
    let mut result = AiGenerateResult {
        text: std::ptr::null(),
        finish_reason: std::ptr::null(),
        input_tokens: 0,
        output_tokens: 0,
        steps: 0,
    };
    let status = unsafe { ai_generate_text(opts, &mut result) };
    if status != AI_OK {
        unsafe { ai_generate_result_free(&mut result) };
        return Err(format!("generate failed (status {})", status));
    }
    let r = unsafe {
        GenerateResult {
            text: CStr::from_ptr(result.text).to_string_lossy().into_owned(),
            finish_reason: CStr::from_ptr(result.finish_reason).to_string_lossy().into_owned(),
            input_tokens: result.input_tokens,
            output_tokens: result.output_tokens,
            steps: result.steps,
        }
    };
    unsafe { ai_generate_result_free(&mut result) };
    Ok(r)
}

pub fn version() -> &'static str {
    unsafe { CStr::from_ptr(ai_sdk_version()).to_string_lossy().into_owned().leak() }
}
