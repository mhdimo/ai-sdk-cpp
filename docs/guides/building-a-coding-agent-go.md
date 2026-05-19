# Building a Coding Agent in Go

This guide walks you through using `ai-sdk-cpp` from Go via cgo. You will link against `libai_sdk`, build an idiomatic Go wrapper, and assemble a fully functional coding agent with tools for reading files, writing files, running commands, listing directories, and searching code.

---

## Why Go + ai-sdk-cpp?

Go is excellent for CLI tools and long-running processes. By linking against `libai_sdk` through cgo, you get:

- **Native C++ engine performance** -- network I/O, JSON parsing, and tool dispatch happen in compiled C++ code, not interpreted runtimes.
- **Unified provider interface** -- Anthropic, OpenAI, Google, Groq, xAI, Mistral, and others through a single API.
- **Static binary deployment** -- compile your agent into a single binary with the shared library alongside it.
- **Go's concurrency model** -- run multiple agent calls concurrently if needed.

---

## Setup

### Prerequisites

- Go 1.21 or later
- `libai_sdk` built from the `ai-sdk-cpp` repository (produces `libai_sdk.so` on Linux, `libai_sdk.dylib` on macOS)
- The C header at `bindings/c/ai_sdk.h`

### Project Layout

```
my-coding-agent/
  go.mod
  main.go
  pkg/
    aisdk/
      aisdk.go       # Go wrapper package
      callback.go    # cgo export + dispatch table
```

### go.mod

```
module github.com/yourname/coding-agent

go 1.21
```

### cgo Directives

At the top of your Go wrapper package, specify where to find the header and library:

```go
/*
#cgo CFLAGS: -I${SRCDIR}/../../../ai-sdk-cpp/bindings/c
#cgo LDFLAGS: -L${SRCDIR}/../../../ai-sdk-cpp/build/lib -lai_sdk
#include "ai_sdk.h"
#include <stdlib.h>
*/
import "C"
```

Adjust the paths to match your build layout. You can also use environment variables at build time:

```bash
export CGO_CFLAGS="-I/path/to/ai-sdk-cpp/bindings/c"
export CGO_LDFLAGS="-L/path/to/ai-sdk-cpp/build/lib -lai_sdk"
go build ./...
```

---

## Go Wrapper Package

The wrapper translates C handles into Go types with explicit `Close()` methods. Finalizers are unreliable for resources with ordering constraints, so this package uses the explicit cleanup pattern that Go developers expect.

### aisdk.go

```go
package aisdk

/*
#cgo CFLAGS: -I${SRCDIR}/../../../ai-sdk-cpp/bindings/c
#cgo LDFLAGS: -L${SRCDIR}/../../../ai-sdk-cpp/build/lib -lai_sdk
#include "ai_sdk.h"
#include <stdlib.h>

// Forward declaration for the exported Go callback.
extern ai_tool_result_t goToolCallback(const char* tool_name, const char* input_json, void* user_data);
*/
import "C"
import (
	"fmt"
	"unsafe"
)

// Status codes mapped from ai_status_t.
type Status int

const (
	OK                 Status = C.AI_OK
	ErrInvalidArgument Status = C.AI_ERROR_INVALID_ARGUMENT
	ErrAPICall         Status = C.AI_ERROR_API_CALL
	ErrRateLimit       Status = C.AI_ERROR_RATE_LIMIT
	ErrAuthentication  Status = C.AI_ERROR_AUTHENTICATION
	ErrTimeout         Status = C.AI_ERROR_TIMEOUT
	ErrStream          Status = C.AI_ERROR_STREAM
	ErrCancelled       Status = C.AI_ERROR_CANCELLED
	ErrInvalidResponse Status = C.AI_ERROR_INVALID_RESPONSE
	ErrInternal        Status = C.AI_ERROR_INTERNAL
)

// SDKError wraps an ai_status_t with a descriptive message.
type SDKError struct {
	Code    Status
	Message string
}

func (e *SDKError) Error() string {
	return fmt.Sprintf("ai-sdk error %d: %s", e.Code, e.Message)
}

func statusToError(s C.ai_status_t, ctx C.ai_context_t) error {
	if s == C.AI_OK {
		return nil
	}
	msg := C.GoString(C.ai_status_message(s))
	if ctx != nil {
		if detail := C.ai_last_error(ctx); detail != nil {
			msg = C.GoString(detail)
		}
	}
	return &SDKError{Code: Status(s), Message: msg}
}

// Context owns the event loop. Create one per application.
type Context struct {
	handle C.ai_context_t
}

func NewContext() *Context {
	return &Context{handle: C.ai_context_create()}
}

func (c *Context) Close() {
	if c.handle != nil {
		C.ai_context_destroy(c.handle)
		c.handle = nil
	}
}

// Provider represents a configured AI provider.
type Provider struct {
	handle C.ai_provider_t
}

type ProviderOptions struct {
	APIKey  string // empty = use environment variable
	BaseURL string // empty = use default
}

func NewProvider(ctx *Context, name string, opts ProviderOptions) (*Provider, error) {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))

	var cOpts C.ai_provider_options_t
	if opts.APIKey != "" {
		cKey := C.CString(opts.APIKey)
		defer C.free(unsafe.Pointer(cKey))
		cOpts.api_key = cKey
	}
	if opts.BaseURL != "" {
		cURL := C.CString(opts.BaseURL)
		defer C.free(unsafe.Pointer(cURL))
		cOpts.base_url = cURL
	}

	handle := C.ai_provider_create(ctx.handle, cName, cOpts)
	if handle == nil {
		return nil, &SDKError{Code: ErrInternal, Message: "failed to create provider"}
	}
	return &Provider{handle: handle}, nil
}

func (p *Provider) Close() {
	if p.handle != nil {
		C.ai_provider_destroy(p.handle)
		p.handle = nil
	}
}

// Model represents a specific model from a provider.
type Model struct {
	handle C.ai_model_t
}

func NewModel(provider *Provider, modelID string) *Model {
	cID := C.CString(modelID)
	defer C.free(unsafe.Pointer(cID))
	return &Model{handle: C.ai_model_create(provider.handle, cID)}
}

func (m *Model) Close() {
	if m.handle != nil {
		C.ai_model_destroy(m.handle)
		m.handle = nil
	}
}

// GenerateResult holds the output of a text generation call.
type GenerateResult struct {
	Text         string
	FinishReason string
	InputTokens  int
	OutputTokens int
	Steps        int
}

// GenerateOptions configures a generateText call.
type GenerateOptions struct {
	Model          *Model
	Prompt         string
	System         string
	Tools          *ToolSet
	MaxSteps       int
	MaxOutputTokens int
	Temperature    float64 // use -1 for model default
}

// GenerateText performs a synchronous text generation, including tool loops.
func GenerateText(ctx *Context, opts GenerateOptions) (*GenerateResult, error) {
	var cOpts C.ai_generate_options_t
	cOpts.model = opts.Model.handle
	cOpts.temperature = C.double(opts.Temperature)
	cOpts.max_steps = C.int(opts.MaxSteps)
	cOpts.max_output_tokens = C.int(opts.MaxOutputTokens)

	var cPrompt, cSystem *C.char
	if opts.Prompt != "" {
		cPrompt = C.CString(opts.Prompt)
		defer C.free(unsafe.Pointer(cPrompt))
		cOpts.prompt = cPrompt
	}
	if opts.System != "" {
		cSystem = C.CString(opts.System)
		defer C.free(unsafe.Pointer(cSystem))
		cOpts.system = cSystem
	}
	if opts.Tools != nil {
		cOpts.tools = opts.Tools.handle
	}

	var cResult C.ai_generate_result_t
	status := C.ai_generate_text(cOpts, &cResult)
	if err := statusToError(status, ctx.handle); err != nil {
		return nil, err
	}
	defer C.ai_generate_result_free(&cResult)

	return &GenerateResult{
		Text:         C.GoString(cResult.text),
		FinishReason: C.GoString(cResult.finish_reason),
		InputTokens:  int(cResult.input_tokens),
		OutputTokens: int(cResult.output_tokens),
		Steps:        int(cResult.steps),
	}, nil
}

// Agent wraps the high-level agent API.
type Agent struct {
	handle C.ai_agent_t
	ctx    *Context
}

type AgentOptions struct {
	Model        *Model
	Tools        *ToolSet
	Instructions string
	MaxSteps     int
}

func NewAgent(ctx *Context, opts AgentOptions) *Agent {
	var cOpts C.ai_agent_options_t
	cOpts.model = opts.Model.handle
	cOpts.max_steps = C.int(opts.MaxSteps)

	if opts.Tools != nil {
		cOpts.tools = opts.Tools.handle
	}

	var cInstructions *C.char
	if opts.Instructions != "" {
		cInstructions = C.CString(opts.Instructions)
		defer C.free(unsafe.Pointer(cInstructions))
		cOpts.instructions = cInstructions
	}

	return &Agent{
		handle: C.ai_agent_create(cOpts),
		ctx:    ctx,
	}
}

func (a *Agent) Call(prompt string) (*GenerateResult, error) {
	cPrompt := C.CString(prompt)
	defer C.free(unsafe.Pointer(cPrompt))

	var cResult C.ai_generate_result_t
	status := C.ai_agent_call(a.handle, cPrompt, &cResult)
	if err := statusToError(status, a.ctx.handle); err != nil {
		return nil, err
	}
	defer C.ai_generate_result_free(&cResult)

	return &GenerateResult{
		Text:         C.GoString(cResult.text),
		FinishReason: C.GoString(cResult.finish_reason),
		InputTokens:  int(cResult.input_tokens),
		OutputTokens: int(cResult.output_tokens),
		Steps:        int(cResult.steps),
	}, nil
}

func (a *Agent) Close() {
	if a.handle != nil {
		C.ai_agent_destroy(a.handle)
		a.handle = nil
	}
}

// Version returns the SDK version string.
func Version() string {
	return C.GoString(C.ai_sdk_version())
}
```

---

## The Tool Callback: Registry Pattern

Go cannot pass Go function pointers directly to C. The standard cgo pattern is:

1. Export a single C-callable function from Go.
2. Maintain a registry (dispatch table) mapping integer IDs to Go functions.
3. Pass the ID as the `void* user_data` to C.
4. When C calls the exported function, look up the ID and dispatch to the correct Go function.

### callback.go

```go
package aisdk

/*
#include "ai_sdk.h"
#include <stdlib.h>
*/
import "C"
import (
	"sync"
	"unsafe"
)

// ToolFunc is the Go signature for a tool implementation.
// It receives the tool name and input JSON, and returns output JSON and an error flag.
type ToolFunc func(toolName string, inputJSON string) (outputJSON string, isError bool)

// --- Registry ---

var (
	registryMu   sync.Mutex
	registry     = map[uintptr]ToolFunc{}
	nextID       uintptr = 1
)

func registerTool(fn ToolFunc) uintptr {
	registryMu.Lock()
	defer registryMu.Unlock()
	id := nextID
	nextID++
	registry[id] = fn
	return id
}

func unregisterTool(id uintptr) {
	registryMu.Lock()
	defer registryMu.Unlock()
	delete(registry, id)
}

func lookupTool(id uintptr) ToolFunc {
	registryMu.Lock()
	defer registryMu.Unlock()
	return registry[id]
}

// goToolCallback is exported to C. It dispatches to the registered Go function.
//
//export goToolCallback
func goToolCallback(toolName *C.char, inputJSON *C.char, userData unsafe.Pointer) C.ai_tool_result_t {
	id := uintptr(userData)
	fn := lookupTool(id)

	var result C.ai_tool_result_t
	if fn == nil {
		errMsg := C.CString(`{"error": "tool not found in registry"}`)
		result.output_json = errMsg
		result.is_error = 1
		return result
	}

	goName := C.GoString(toolName)
	goInput := C.GoString(inputJSON)
	output, isErr := fn(goName, goInput)

	result.output_json = C.CString(output)
	if isErr {
		result.is_error = 1
	}
	return result
}

// ToolSet manages a set of tools and their registrations.
type ToolSet struct {
	handle C.ai_tool_set_t
	ids    []uintptr // track registered IDs for cleanup
}

func NewToolSet() *ToolSet {
	return &ToolSet{
		handle: C.ai_tool_set_create(),
	}
}

// Add registers a tool with the SDK. The schemaJSON must be a valid JSON Schema
// describing the tool's input parameters.
func (ts *ToolSet) Add(name, description, schemaJSON string, fn ToolFunc) error {
	id := registerTool(fn)
	ts.ids = append(ts.ids, id)

	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	cDesc := C.CString(description)
	defer C.free(unsafe.Pointer(cDesc))
	cSchema := C.CString(schemaJSON)
	defer C.free(unsafe.Pointer(cSchema))

	status := C.ai_tool_set_add(
		ts.handle,
		cName,
		cDesc,
		cSchema,
		C.ai_tool_fn(C.goToolCallback),
		unsafe.Pointer(id),
	)

	if status != C.AI_OK {
		unregisterTool(id)
		return &SDKError{Code: Status(status), Message: "failed to add tool"}
	}
	return nil
}

func (ts *ToolSet) Close() {
	if ts.handle != nil {
		C.ai_tool_set_destroy(ts.handle)
		ts.handle = nil
	}
	for _, id := range ts.ids {
		unregisterTool(id)
	}
	ts.ids = nil
}
```

### How the Dispatch Works

```
C library calls tool  -->  goToolCallback (exported to C)
                                |
                                v
                      registry[user_data] --> your Go function
                                |
                                v
                      returns ai_tool_result_t to C
```

The `user_data` pointer carries the registry ID. No Go pointers cross the cgo boundary -- only an integer cast to `unsafe.Pointer`. This satisfies the cgo pointer-passing rules.

---

## Error Handling

The wrapper maps every `ai_status_t` to a typed Go error:

```go
var ErrRateLimit = &SDKError{Code: Status(C.AI_ERROR_RATE_LIMIT)}

func IsRateLimit(err error) bool {
	var sdkErr *SDKError
	if errors.As(err, &sdkErr) {
		return sdkErr.Code == ErrRateLimit
	}
	return false
}

func IsAuthentication(err error) bool {
	var sdkErr *SDKError
	if errors.As(err, &sdkErr) {
		return sdkErr.Code == ErrAuthentication
	}
	return false
}
```

Usage in calling code:

```go
result, err := agent.Call("Fix the bug in main.go")
if err != nil {
	var sdkErr *aisdk.SDKError
	if errors.As(err, &sdkErr) {
		switch sdkErr.Code {
		case aisdk.ErrRateLimit:
			log.Println("Rate limited, backing off...")
			time.Sleep(30 * time.Second)
		case aisdk.ErrAuthentication:
			log.Fatal("Invalid API key. Set ANTHROPIC_API_KEY.")
		case aisdk.ErrTimeout:
			log.Println("Request timed out, retrying...")
		default:
			log.Fatalf("SDK error: %v", err)
		}
	}
}
```

The full error code mapping:

| ai_status_t                  | Go constant          | Meaning                            |
|------------------------------|----------------------|------------------------------------|
| `AI_OK`                      | `OK`                 | Success                            |
| `AI_ERROR_INVALID_ARGUMENT`  | `ErrInvalidArgument` | Bad parameter passed to the SDK    |
| `AI_ERROR_API_CALL`          | `ErrAPICall`         | Provider API returned an error     |
| `AI_ERROR_RATE_LIMIT`        | `ErrRateLimit`       | Rate limit exceeded                |
| `AI_ERROR_AUTHENTICATION`    | `ErrAuthentication`  | Invalid or missing API key         |
| `AI_ERROR_TIMEOUT`           | `ErrTimeout`         | Request timed out                  |
| `AI_ERROR_STREAM`            | `ErrStream`          | Streaming connection error         |
| `AI_ERROR_CANCELLED`         | `ErrCancelled`       | Request was cancelled              |
| `AI_ERROR_INVALID_RESPONSE`  | `ErrInvalidResponse` | Could not parse provider response  |
| `AI_ERROR_INTERNAL`          | `ErrInternal`        | Internal SDK error                 |

---

## Full Working Example: A Coding Agent

Save this as `main.go`. It implements five tools and wires them into an agent that can navigate and modify a codebase.

```go
package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/yourname/coding-agent/pkg/aisdk"
)

func main() {
	ctx := aisdk.NewContext()
	defer ctx.Close()

	provider, err := aisdk.NewProvider(ctx, "anthropic", aisdk.ProviderOptions{})
	if err != nil {
		fmt.Fprintf(os.Stderr, "provider error: %v\n", err)
		os.Exit(1)
	}
	defer provider.Close()

	model := aisdk.NewModel(provider, "claude-sonnet-4-20250514")
	defer model.Close()

	tools := aisdk.NewToolSet()
	defer tools.Close()

	// --- Register tools ---
	tools.Add("read_file", "Read a file and return its contents.",
		`{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}`,
		func(_ string, input string) (string, bool) {
			var p struct{ Path string }
			json.Unmarshal([]byte(input), &p)
			data, err := os.ReadFile(p.Path)
			if err != nil {
				return fmt.Sprintf(`{"error":"%s"}`, err.Error()), true
			}
			out, _ := json.Marshal(map[string]string{"content": string(data)})
			return string(out), false
		})

	tools.Add("write_file", "Write content to a file, creating directories as needed.",
		`{"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},"required":["path","content"]}`,
		func(_ string, input string) (string, bool) {
			var p struct{ Path, Content string }
			json.Unmarshal([]byte(input), &p)
			os.MkdirAll(filepath.Dir(p.Path), 0o755)
			if err := os.WriteFile(p.Path, []byte(p.Content), 0o644); err != nil {
				return fmt.Sprintf(`{"error":"%s"}`, err.Error()), true
			}
			return fmt.Sprintf(`{"written":"%s","bytes":%d}`, p.Path, len(p.Content)), false
		})

	tools.Add("run_command", "Execute a shell command and return stdout/stderr.",
		`{"type":"object","properties":{"command":{"type":"string"},"cwd":{"type":"string"}},"required":["command"]}`,
		func(_ string, input string) (string, bool) {
			var p struct{ Command, Cwd string }
			json.Unmarshal([]byte(input), &p)
			cmd := exec.Command("sh", "-c", p.Command)
			if p.Cwd != "" {
				cmd.Dir = p.Cwd
			}
			out, err := cmd.CombinedOutput()
			if err != nil {
				return fmt.Sprintf(`{"output":%q,"error":"%s"}`, string(out), err.Error()), true
			}
			return fmt.Sprintf(`{"output":%q}`, string(out)), false
		})

	tools.Add("list_files", "List files in a directory.",
		`{"type":"object","properties":{"directory":{"type":"string"},"recursive":{"type":"boolean"}},"required":["directory"]}`,
		func(_ string, input string) (string, bool) {
			var p struct {
				Directory string
				Recursive bool
			}
			json.Unmarshal([]byte(input), &p)
			var files []string
			if p.Recursive {
				filepath.Walk(p.Directory, func(path string, _ os.FileInfo, _ error) error {
					files = append(files, path)
					if len(files) > 200 {
						return filepath.SkipAll
					}
					return nil
				})
			} else {
				entries, _ := os.ReadDir(p.Directory)
				for _, e := range entries {
					files = append(files, e.Name())
				}
			}
			out, _ := json.Marshal(files)
			return string(out), false
		})

	tools.Add("search_files", "Search for a pattern in files using grep.",
		`{"type":"object","properties":{"pattern":{"type":"string"},"directory":{"type":"string"},"include":{"type":"string"}},"required":["pattern","directory"]}`,
		func(_ string, input string) (string, bool) {
			var p struct{ Pattern, Directory, Include string }
			json.Unmarshal([]byte(input), &p)
			args := []string{"-rn"}
			if p.Include != "" {
				args = append(args, "--include="+p.Include)
			}
			args = append(args, p.Pattern, p.Directory)
			out, _ := exec.Command("grep", args...).Output()
			lines := strings.Split(string(out), "\n")
			if len(lines) > 50 {
				lines = lines[:50]
			}
			return strings.Join(lines, "\n"), false
		})

	// --- Create agent ---
	agent := aisdk.NewAgent(ctx, aisdk.AgentOptions{
		Model: model,
		Tools: tools,
		Instructions: `You are a coding agent. Rules:
- Always read a file before editing it.
- Use run_command to execute tests after making changes.
- Be concise in your explanations.`,
		MaxSteps: 30,
	})
	defer agent.Close()

	// --- REPL ---
	fmt.Printf("Coding Agent (ai-sdk-cpp %s) - type 'exit' to quit\n\n", aisdk.Version())
	scanner := bufio.NewScanner(os.Stdin)
	for {
		fmt.Print("> ")
		if !scanner.Scan() {
			break
		}
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		if line == "exit" {
			break
		}

		result, err := agent.Call(line)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n\n", err)
			continue
		}
		fmt.Printf("\n%s\n", result.Text)
		fmt.Printf("[%d steps | %d input tokens | %d output tokens]\n\n",
			result.Steps, result.InputTokens, result.OutputTokens)
	}
}
```

---

## Build Instructions

### 1. Build libai_sdk

From the `ai-sdk-cpp` repository root:

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

This produces `build/lib/libai_sdk.so` (Linux) or `build/lib/libai_sdk.dylib` (macOS).

### 2. Build the Go binary

```bash
# Point cgo at the SDK headers and library
export CGO_CFLAGS="-I/path/to/ai-sdk-cpp/bindings/c"
export CGO_LDFLAGS="-L/path/to/ai-sdk-cpp/build/lib -lai_sdk"

# Build
go build -o coding-agent ./...
```

### 3. Set the library path at runtime

The dynamic linker needs to find `libai_sdk` at runtime.

**Linux:**

```bash
export LD_LIBRARY_PATH=/path/to/ai-sdk-cpp/build/lib:$LD_LIBRARY_PATH
./coding-agent
```

**macOS:**

```bash
export DYLD_LIBRARY_PATH=/path/to/ai-sdk-cpp/build/lib:$DYLD_LIBRARY_PATH
./coding-agent
```

**Alternative: install the library system-wide:**

```bash
sudo cp build/lib/libai_sdk.* /usr/local/lib/
sudo ldconfig  # Linux only
```

### 4. Set your API key

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
./coding-agent
```

---

## Static Linking (Optional)

If you prefer a fully self-contained binary, build `libai_sdk` as a static library:

```bash
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF
make -j$(nproc)
```

Then link with:

```bash
export CGO_LDFLAGS="-L/path/to/ai-sdk-cpp/build/lib -lai_sdk -lstdc++ -lm -lpthread"
go build -o coding-agent ./...
```

The resulting binary has no runtime dependency on `libai_sdk`.

---

## Switching Providers

Change one line to use a different provider. The tools and agent logic stay identical:

```go
// OpenAI
provider, _ := aisdk.NewProvider(ctx, "openai", aisdk.ProviderOptions{})
model := aisdk.NewModel(provider, "gpt-4o")

// Google
provider, _ := aisdk.NewProvider(ctx, "google", aisdk.ProviderOptions{})
model := aisdk.NewModel(provider, "gemini-2.5-pro")

// Groq
provider, _ := aisdk.NewProvider(ctx, "groq", aisdk.ProviderOptions{})
model := aisdk.NewModel(provider, "llama-3.3-70b-versatile")
```

---

## Key Design Decisions

**Explicit Close() over finalizers.** Go finalizers run at GC time in arbitrary order. Since the SDK has object ownership (context owns providers, providers own models), explicit cleanup gives you deterministic teardown. Use `defer` at the call site.

**Registry pattern for callbacks.** Go's cgo rules forbid passing Go pointers to C when they might be stored. The registry pattern uses an integer ID (cast to `unsafe.Pointer`) as a stable reference. The Go function lives in a global map, safe from the garbage collector.

**JSON in, JSON out for tools.** The C API uses JSON strings for tool input and output. This maps naturally to Go's `encoding/json` package. Parse the input with `json.Unmarshal`, build your response, and return a JSON string.

**Synchronous tools.** The C callback is synchronous -- it blocks the SDK's tool loop until you return. For most coding agent tools (file I/O, short commands), this is fine. For long-running commands, consider a timeout in your tool implementation.

---

## Next Steps

- Add a confirmation prompt before destructive operations (writes, command execution)
- Implement conversation history by accumulating messages and using `messages_json`
- Add signal handling (SIGINT) to cancel in-flight requests gracefully
- Build rate-limit retry logic using the `ErrRateLimit` error code
- Restrict file operations to a project directory for safety
