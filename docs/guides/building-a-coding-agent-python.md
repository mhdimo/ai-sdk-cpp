# Building a Coding Agent with ai-sdk-cpp Python Bindings

Build your own coding agent -- like Claude Code, Codex, or Kimi Code -- in an afternoon. This guide walks you through the Python bindings for the ai-sdk-cpp engine, from first API call to a fully functional CLI agent that can read, write, search, and execute code on your behalf.

## Why the C++ Engine from Python?

The ai-sdk-cpp engine handles inference orchestration (tool loops, retries, streaming, token management) in native C++, then exposes it to Python via zero-copy bindings. You get:

- **10x faster orchestration** -- tool loop dispatch, JSON schema validation, and streaming happen in compiled code, not interpreted Python
- **Native system access** -- file I/O, subprocess execution, and memory management run at C-level speed
- **Familiar API** -- if you have used the Vercel AI SDK in TypeScript, the Python API mirrors it closely: `generate_text`, `Agent`, `tool`, `ToolSet`
- **Multi-provider** -- one unified interface for Anthropic, OpenAI, Google, and more

The Python layer is thin: you define tools, configure providers, and interact with results. The engine does the heavy lifting.

---

## Installation

### From Source (development)

```bash
cd bindings/python
pip install -e .
```

This builds the C++ extension in-place. Requires CMake 3.20+ and a C++20 compiler.

### From a Prebuilt Wheel

```bash
pip install ai-sdk-cpp
```

Wheels are available for Linux (x86_64, aarch64), macOS (arm64, x86_64), and Windows (x64).

### Verify Installation

```bash
python -c "from ai_sdk import generate_text, create_anthropic; print('Ready')"
```

---

## Quick Start: 5 Lines to Generate Text

```python
from ai_sdk import create_anthropic, generate_text

model = create_anthropic()("claude-sonnet-4-20250514")
result = generate_text(model, prompt="Explain what a coding agent is in two sentences.")
print(result.text)
```

Set your API key beforehand:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

That is all it takes to call a model. Now let's build something real.

---

## Building a Coding Agent: Step by Step

A coding agent is a loop: the model reasons about what to do, calls tools to interact with the filesystem and shell, observes results, and continues until the task is done. Our job is to define the tools and let the engine handle the loop.

### Step 1: Choose Your Model

```python
from ai_sdk import create_anthropic

anthropic = create_anthropic()  # reads ANTHROPIC_API_KEY from env
model = anthropic("claude-sonnet-4-20250514")
```

### Step 2: Define Tools

Tools are functions the model can call. Each tool has a name, a JSON Schema for its parameters, a description, and an implementation.

### Step 3: Create an Agent

```python
from ai_sdk import Agent, ToolSet

agent = Agent(
    model,
    tools,
    instructions="You are a senior software engineer...",
    max_steps=50,
)
```

### Step 4: Run It

```python
result = agent.call("Find and fix the bug in src/auth.py")
print(result.text)
```

Let's flesh out each piece.

---

## Defining Tools

### The `@tool` Decorator

The cleanest way to register tools. The decorator takes a name, a JSON Schema, and a description:

```python
from ai_sdk import tool

@tool(
    "read_file",
    {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Absolute or relative file path"}
        },
        "required": ["path"],
    },
    "Read the contents of a file and return it as text",
)
def read_file(input):
    with open(input["path"]) as f:
        return f.read()
```

The function receives a dictionary matching the schema. Return a string -- that becomes the tool result the model sees.

### Using ToolSet for Programmatic Registration

When you build tools dynamically or want them grouped:

```python
from ai_sdk import ToolSet

tools = ToolSet()

def read_file_fn(input):
    with open(input["path"]) as f:
        return f.read()

tools.add(
    "read_file",
    {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
    "Read a file",
    read_file_fn,
)
```

Both approaches produce identical tool definitions for the engine.

---

## Essential Coding Agent Tools

Here is the full set of tools a capable coding agent needs:

### read_file

```python
@tool(
    "read_file",
    {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the file to read"},
            "offset": {"type": "integer", "description": "Line number to start from (0-indexed)"},
            "limit": {"type": "integer", "description": "Max lines to return"},
        },
        "required": ["path"],
    },
    "Read file contents. Use offset/limit for large files.",
)
def read_file(input):
    path = input["path"]
    offset = input.get("offset", 0)
    limit = input.get("limit")

    with open(path) as f:
        lines = f.readlines()

    selected = lines[offset:] if limit is None else lines[offset : offset + limit]
    numbered = [f"{i + offset + 1:4d} | {line}" for i, line in enumerate(selected)]
    return "".join(numbered)
```

### write_file

```python
@tool(
    "write_file",
    {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to write to"},
            "content": {"type": "string", "description": "Full file content"},
        },
        "required": ["path", "content"],
    },
    "Write content to a file. Creates parent directories if needed.",
)
def write_file(input):
    import os

    path = input["path"]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write(input["content"])
    return f"Wrote {len(input['content'])} bytes to {path}"
```

### edit_file (Patch-Based)

This is the most important tool for a coding agent. Instead of rewriting entire files, it applies targeted edits:

```python
@tool(
    "edit_file",
    {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the file to edit"},
            "old_string": {"type": "string", "description": "Exact text to find and replace"},
            "new_string": {"type": "string", "description": "Replacement text"},
        },
        "required": ["path", "old_string", "new_string"],
    },
    "Replace an exact string in a file. The old_string must match exactly one location.",
)
def edit_file(input):
    path = input["path"]
    old = input["old_string"]
    new = input["new_string"]

    with open(path) as f:
        content = f.read()

    occurrences = content.count(old)
    if occurrences == 0:
        return f"Error: old_string not found in {path}"
    if occurrences > 1:
        return f"Error: old_string matches {occurrences} locations. Be more specific."

    content = content.replace(old, new, 1)
    with open(path, "w") as f:
        f.write(content)

    return f"Applied edit to {path}"
```

### run_command

```python
import subprocess

@tool(
    "run_command",
    {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "Shell command to execute"},
            "timeout": {"type": "integer", "description": "Timeout in seconds (default 30)"},
        },
        "required": ["command"],
    },
    "Execute a shell command and return stdout+stderr.",
)
def run_command(input):
    timeout = input.get("timeout", 30)
    try:
        result = subprocess.run(
            input["command"],
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = result.stdout + result.stderr
        return f"Exit code: {result.returncode}\n{output[:10000]}"
    except subprocess.TimeoutExpired:
        return f"Error: Command timed out after {timeout}s"
```

### list_files

```python
import os

@tool(
    "list_files",
    {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Directory path"},
            "recursive": {"type": "boolean", "description": "List recursively (default false)"},
        },
        "required": ["path"],
    },
    "List files in a directory.",
)
def list_files(input):
    path = input["path"]
    recursive = input.get("recursive", False)

    entries = []
    if recursive:
        for root, dirs, files in os.walk(path):
            # Skip hidden and common ignore dirs
            dirs[:] = [d for d in dirs if not d.startswith(".") and d != "node_modules"]
            for f in files:
                entries.append(os.path.relpath(os.path.join(root, f), path))
    else:
        entries = os.listdir(path)

    return "\n".join(sorted(entries)[:500])
```

### search_files

```python
import subprocess

@tool(
    "search_files",
    {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Search pattern (regex supported)"},
            "path": {"type": "string", "description": "Directory to search in"},
            "file_glob": {"type": "string", "description": "File pattern filter (e.g. '*.py')"},
        },
        "required": ["pattern", "path"],
    },
    "Search for a pattern across files using ripgrep.",
)
def search_files(input):
    cmd = ["rg", "--line-number", "--no-heading", "--max-count", "50"]
    if "file_glob" in input:
        cmd.extend(["--glob", input["file_glob"]])
    cmd.extend([input["pattern"], input["path"]])

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return result.stdout[:10000] or "No matches found."
    except FileNotFoundError:
        # Fallback to grep if rg not available
        cmd = ["grep", "-rn", input["pattern"], input["path"]]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return result.stdout[:10000] or "No matches found."
```

---

## The @tool Decorator Pattern

The decorator pattern keeps tool definitions close to their implementations. Here is the anatomy:

```python
@tool(name, schema, description)
def implementation(input: dict) -> str:
    ...
```

**Rules:**
- `name` -- unique identifier the model uses to call the tool (use snake_case)
- `schema` -- JSON Schema object describing the input parameters
- `description` -- tells the model when and how to use the tool (be specific)
- `input` -- always a dictionary matching the schema
- Return value -- always a string (the tool result shown to the model)

You can collect decorated tools into a list:

```python
from ai_sdk import tool

all_tools = []

@tool("read_file", {...}, "Read a file")
def read_file(input):
    ...
all_tools.append(read_file)

@tool("write_file", {...}, "Write a file")
def write_file(input):
    ...
all_tools.append(write_file)
```

Or use the ToolSet approach for runtime assembly:

```python
tools = ToolSet()
for name, schema, desc, fn in discovered_tools:
    tools.add(name, schema, desc, fn)
```

---

## Streaming Responses

For a CLI agent, streaming is essential -- users need to see the model thinking in real time:

```python
for event in agent.call_stream("Refactor the database connection pool"):
    if event.type == "text_delta":
        print(event.text, end="", flush=True)
    elif event.type == "tool_call":
        print(f"\n--- Calling: {event.tool_name} ---")
    elif event.type == "tool_result":
        print(f"--- Result ({len(event.text)} chars) ---\n")

print()  # Final newline
```

Event types:

| Event Type     | Fields           | Description                          |
|----------------|------------------|--------------------------------------|
| `text_delta`   | `text`           | Incremental text from the model      |
| `tool_call`    | `tool_name`, `input` | Model is calling a tool          |
| `tool_result`  | `text`           | Result returned from tool execution  |
| `step_end`     | `step_number`    | A reasoning/action step completed    |
| `done`         | `result`         | Final result object                  |

---

## Error Handling

The engine surfaces provider errors as typed exceptions:

```python
from ai_sdk import (
    Agent,
    RateLimitError,
    AuthenticationError,
    InvalidRequestError,
    NetworkError,
)

def run_agent(agent, prompt, retries=3):
    for attempt in range(retries):
        try:
            return agent.call(prompt)
        except RateLimitError as e:
            wait = e.retry_after or (2 ** attempt)
            print(f"Rate limited. Waiting {wait}s...")
            import time
            time.sleep(wait)
        except AuthenticationError:
            print("Invalid API key. Check your ANTHROPIC_API_KEY.")
            raise
        except InvalidRequestError as e:
            print(f"Bad request: {e.message}")
            raise
        except NetworkError:
            print(f"Network error, retrying ({attempt + 1}/{retries})...")
            continue

    raise RuntimeError("Max retries exceeded")
```

For tool errors, return error messages as strings rather than raising exceptions -- this lets the model self-correct:

```python
@tool("read_file", {...}, "Read a file")
def read_file(input):
    try:
        with open(input["path"]) as f:
            return f.read()
    except FileNotFoundError:
        return f"Error: File not found: {input['path']}"
    except PermissionError:
        return f"Error: Permission denied: {input['path']}"
```

---

## Multi-Provider Setup

Use different models for different tasks -- a fast model for simple operations, a powerful model for complex reasoning:

```python
from ai_sdk import create_anthropic, create_openai, create_google, Agent

# Provider setup
anthropic = create_anthropic()
openai = create_openai()       # reads OPENAI_API_KEY
google = create_google()       # reads GOOGLE_API_KEY

# Models for different purposes
reasoning_model = anthropic("claude-sonnet-4-20250514")  # complex tasks
fast_model = openai("gpt-4o-mini")                       # simple lookups
embedding_model = google("text-embedding-004")           # search

# Use the reasoning model for the main agent
agent = Agent(reasoning_model, tools, instructions="...", max_steps=50)

# Use the fast model for quick classifications
from ai_sdk import generate_text

def classify_task(prompt):
    result = generate_text(
        fast_model,
        prompt=f"Classify this task as 'bug_fix', 'feature', or 'refactor': {prompt}",
        max_output_tokens=10,
    )
    return result.text.strip()
```

---

## Configuration

### Environment Variables

The providers read standard env vars:

| Provider   | Environment Variable     |
|------------|--------------------------|
| Anthropic  | `ANTHROPIC_API_KEY`      |
| OpenAI     | `OPENAI_API_KEY`         |
| Google     | `GOOGLE_API_KEY`         |

### .env Files

Load a `.env` file before creating providers:

```python
from dotenv import load_dotenv
load_dotenv()  # loads .env from current directory

from ai_sdk import create_anthropic
model = create_anthropic()("claude-sonnet-4-20250514")
```

### Explicit Keys

Pass keys directly when env vars are not suitable:

```python
anthropic = create_anthropic(api_key="sk-ant-...")
openai = create_openai(api_key="sk-...", base_url="https://my-proxy.example.com/v1")
```

### Agent Configuration

```python
agent = Agent(
    model,
    tools,
    instructions="...",    # System prompt
    max_steps=50,          # Max tool-use iterations before stopping
)
```

---

## Full Working Example: Coding Agent CLI

Here is a complete, working coding agent in ~80 lines. Save this as `agent.py` and run it:

```python
#!/usr/bin/env python3
"""A minimal coding agent powered by ai-sdk-cpp."""

import os
import subprocess
import sys

from ai_sdk import Agent, ToolSet, create_anthropic

# --- Provider Setup ---
model = create_anthropic()("claude-sonnet-4-20250514")

# --- Tool Definitions ---
tools = ToolSet()

tools.add(
    "read_file",
    {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
    "Read a file and return its contents with line numbers.",
    lambda input: (
        "".join(f"{i+1:4d} | {l}" for i, l in enumerate(open(input["path"]).readlines()))
        if os.path.exists(input["path"])
        else f"Error: {input['path']} not found"
    ),
)

tools.add(
    "write_file",
    {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]},
    "Write content to a file, creating directories as needed.",
    lambda input: (
        os.makedirs(os.path.dirname(input["path"]) or ".", exist_ok=True)
        or open(input["path"], "w").write(input["content"])
        and f"Wrote {input['path']}"
    ),
)

tools.add(
    "edit_file",
    {"type": "object", "properties": {"path": {"type": "string"}, "old_string": {"type": "string"}, "new_string": {"type": "string"}}, "required": ["path", "old_string", "new_string"]},
    "Replace exact text in a file. old_string must match exactly once.",
    lambda input: (
        (lambda c: "Not found" if input["old_string"] not in c else (
            open(input["path"], "w").write(c.replace(input["old_string"], input["new_string"], 1)) and "Applied"
        ))(open(input["path"]).read())
    ),
)

tools.add(
    "run_command",
    {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]},
    "Execute a shell command. Returns stdout, stderr, and exit code.",
    lambda input: (
        (lambda r: f"[exit {r.returncode}]\n{r.stdout}{r.stderr}")(
            subprocess.run(input["command"], shell=True, capture_output=True, text=True, timeout=30)
        )
    ),
)

tools.add(
    "list_files",
    {"type": "object", "properties": {"path": {"type": "string"}, "recursive": {"type": "boolean"}}, "required": ["path"]},
    "List files in a directory.",
    lambda input: "\n".join(sorted(
        (os.path.relpath(os.path.join(r, f), input["path"])
         for r, _, fs in os.walk(input["path"]) for f in fs)
        if input.get("recursive") else os.listdir(input["path"])
    )[:200]),
)

tools.add(
    "search_files",
    {"type": "object", "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}}, "required": ["pattern", "path"]},
    "Search for a regex pattern across files.",
    lambda input: subprocess.run(
        ["grep", "-rn", input["pattern"], input["path"]],
        capture_output=True, text=True, timeout=10,
    ).stdout[:8000] or "No matches.",
)

# --- Agent ---
agent = Agent(
    model,
    tools,
    instructions="""You are a senior software engineer working as a coding agent.
You have direct access to the filesystem and shell. Use your tools to understand
the codebase, make changes, and verify your work. Always read before editing.
Run tests after making changes. Be precise with edits.""",
    max_steps=50,
)

# --- Main Loop ---
def main():
    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])
    else:
        prompt = input("What would you like me to do? > ")

    print(f"\nWorking on: {prompt}\n{'─' * 60}")

    for event in agent.call_stream(prompt):
        if event.type == "text_delta":
            print(event.text, end="", flush=True)
        elif event.type == "tool_call":
            print(f"\n  ▶ {event.tool_name}")
        elif event.type == "tool_result":
            preview = event.text[:100].replace("\n", " ")
            print(f"  ◀ {preview}{'...' if len(event.text) > 100 else ''}")

    print(f"\n{'─' * 60}\nDone.")

if __name__ == "__main__":
    main()
```

### Running It

```bash
# Set your API key
export ANTHROPIC_API_KEY="sk-ant-..."

# Run with a prompt
python agent.py "Add input validation to the create_user function in app/models.py"

# Or run interactively
python agent.py
```

---

## Tips for Production Agents

**Sandboxing:** Wrap `run_command` with restrictions. Disallow `rm -rf /`, limit network access, or run in a container.

**Token budgets:** Monitor `result.input_tokens + result.output_tokens`. Set `max_steps` to prevent runaway loops.

**Context management:** For large codebases, use `search_files` before `read_file`. Don't dump entire files into context when a grep will do.

**Tool descriptions matter:** The model decides which tool to call based on descriptions. Be specific: "Search for a regex pattern across all files in a directory tree" is better than "Search files."

**Iterative refinement:** Start with a small toolset (read, write, run). Add tools like `edit_file` and `search_files` once the basics work.

---

## Next Steps

- Add a conversation history loop for multi-turn interactions
- Integrate with git (add `git_diff`, `git_commit` tools)
- Add a permission system (confirm before writes/executions)
- Build a TUI with `rich` or `textual` for a polished experience
- Add `embed` and a vector store for semantic code search

You now have everything you need to build a coding agent. The C++ engine handles the hard parts -- tool loops, streaming, retries, multi-provider routing. Your job is defining great tools and writing clear instructions. Ship it.
