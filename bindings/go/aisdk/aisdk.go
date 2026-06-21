// Package aisdk provides Go bindings for ai-sdk-cpp via cgo.
// Build: see bindings/go/README.md (requires libai_sdk built).
package aisdk

/*
#cgo CFLAGS: -I${SRCDIR}/../../bindings/c
#cgo LDFLAGS: -L${SRCDIR}/../../build/lib -lai_sdk -lstdc++ -lm
#include <stdlib.h>
#include "ai_sdk.h"
*/
import "C"
import (
	"fmt"
	"unsafe"
)

// Context owns the event loop. Create one per application.
type Context struct{ handle C.ai_context_t }

func NewContext() *Context { return &Context{handle: C.ai_context_create()} }
func (c *Context) Close()  { C.ai_context_destroy(c.handle) }

// Provider connects to an LLM provider.
type Provider struct {
	handle *C.struct_ai_provider
	ctx    *Context
}

func NewProvider(ctx *Context, name, apiKey, baseURL string) (*Provider, error) {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	opts := C.ai_provider_options_t{}
	if apiKey != "" {
		cKey := C.CString(apiKey)
		defer C.free(unsafe.Pointer(cKey))
		opts.api_key = cKey
	}
	if baseURL != "" {
		cURL := C.CString(baseURL)
		defer C.free(unsafe.Pointer(cURL))
		opts.base_url = cURL
	}
	h := C.ai_provider_create(ctx.handle, cName, opts)
	if h == nil {
		msg := C.GoString(C.ai_last_error(ctx.handle))
		return nil, fmt.Errorf("failed to create provider: %s", msg)
	}
	return &Provider{handle: h, ctx: ctx}, nil
}
func (p *Provider) Close() { C.ai_provider_destroy(p.handle) }

// Model is a language model handle.
type Model struct{ handle *C.struct_ai_model }

func (p *Provider) Model(modelID string) (*Model, error) {
	cID := C.CString(modelID)
	defer C.free(unsafe.Pointer(cID))
	h := C.ai_model_create(p.handle, cID)
	if h == nil {
		return nil, fmt.Errorf("failed to create model: %s", modelID)
	}
	return &Model{handle: h}, nil
}
func (m *Model) Close() { C.ai_model_destroy(m.handle) }

// GenerateResult holds a text generation result.
type GenerateResult struct {
	Text          string
	FinishReason  string
	InputTokens   int
	OutputTokens  int
	Steps         int
}

// GenerateText runs a one-shot text generation.
func GenerateText(model *Model, prompt string) (*GenerateResult, error) {
	cPrompt := C.CString(prompt)
	defer C.free(unsafe.Pointer(cPrompt))
	opts := C.ai_generate_options_t{model: model.handle, prompt: cPrompt, temperature: -1}
	var result C.ai_generate_result_t
	status := C.ai_generate_text(opts, &result)
	if status != C.AI_OK {
		return nil, fmt.Errorf("generate failed: %s", C.GoString(C.ai_status_message(status)))
	}
	r := &GenerateResult{
		Text:         C.GoString(result.text),
		FinishReason: C.GoString(result.finish_reason),
		InputTokens:  int(result.input_tokens),
		OutputTokens: int(result.output_tokens),
		Steps:        int(result.steps),
	}
	C.ai_generate_result_free(&result)
	return r, nil
}

// Session provides context-managed multi-turn conversation.
type Session struct{ handle C.ai_session_t }

func NewSession(agent *Agent) *Session {
	return &Session{handle: C.ai_session_create(agent.handle)}
}
func (s *Session) Close() { C.ai_session_destroy(s.handle) }
func (s *Session) Send(prompt string) (*GenerateResult, error) {
	cPrompt := C.CString(prompt)
	defer C.free(unsafe.Pointer(cPrompt))
	var result C.ai_generate_result_t
	status := C.ai_session_send(s.handle, cPrompt, &result)
	if status != C.AI_OK {
		return nil, fmt.Errorf("session send failed: %s", C.GoString(C.ai_status_message(status)))
	}
	r := &GenerateResult{
		Text:         C.GoString(result.text),
		FinishReason: C.GoString(result.finish_reason),
		InputTokens:  int(result.input_tokens),
		OutputTokens: int(result.output_tokens),
		Steps:        int(result.steps),
	}
	C.ai_generate_result_free(&result)
	return r, nil
}

// Agent wraps a tool-loop agent.
type Agent struct{ handle C.ai_agent_t }

// StandardToolkit returns a ToolSet with read_file/write_file/edit_file/glob/grep/bash.
func StandardToolkit() *ToolSet {
	return &ToolSet{handle: C.ai_standard_toolkit_create()}
}

// ToolSet wraps a set of tools.
type ToolSet struct{ handle C.ai_tool_set_t }
func (t *ToolSet) Close() { C.ai_tool_set_destroy(t.handle) }

func Version() string { return C.GoString(C.ai_sdk_version()) }
