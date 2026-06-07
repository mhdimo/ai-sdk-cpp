# Building a Coding Agent in TypeScript

You can build your own Claude Code clone in about an hour. This guide walks you through it step by step using `ai-sdk-cpp` -- the TypeScript bindings for the native C++ AI SDK engine.

## Why ai-sdk-cpp?

The `ai-sdk-cpp` package gives you:

- **Near-native speed.** The core engine is C++. Network I/O, JSON parsing, token counting, and tool dispatch all happen at native speed. Your agent spends its time waiting on the model, not on your runtime.
- **Unified provider interface.** Swap between Anthropic, OpenAI, Google, and others with a single line change. Same tools, same agent loop, same result shape.
- **Familiar API shape.** If you have used the Vercel AI SDK, you already know this API. `generateText`, `streamText`, `tool()`, `Agent` -- same concepts, backed by compiled native code.
- **Built for CLI tools and long-running agents.** No framework overhead. No bundler. Just TypeScript, a terminal, and a model.

---

## Installation

```bash
npm install ai-sdk-cpp
```

Set your provider API key as an environment variable:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
# or
export OPENAI_API_KEY="sk-..."
```

---

## Quick Start

Four lines to generate text:

```typescript
import { createAnthropic, generateText } from 'ai-sdk-cpp';

const anthropic = createAnthropic();
const model = anthropic('claude-sonnet-4-20250514');

const result = await generateText({ model, prompt: 'Explain recursion in one sentence.' });
console.log(result.text);
```

That is the foundation. Everything else builds on top of it.

---

## Building a Coding Agent

A coding agent is a loop: the model thinks, calls tools, observes results, and thinks again until the task is done. The ingredients are:

1. A set of **tools** the model can call (read files, write files, run commands)
2. A **system prompt** that tells the model how to behave
3. A **multi-step loop** that keeps going until the model says it is finished

Let's build each piece.

---

## Defining Tools

The `tool()` helper gives you a clean way to define tools with a JSON Schema for input validation, a description for the model, and an execution function.

### Signature

```typescript
tool(name, inputSchema, description, execute)
```

- `name` -- tool identifier the model uses to call it
- `inputSchema` -- JSON Schema object describing the input
- `description` -- human-readable explanation for the model
- `execute` -- function that receives the parsed input and returns a string

### read_file

```typescript
import { tool } from 'ai-sdk-cpp';
import * as fs from 'node:fs';

const readFile = tool('read_file', {
  type: 'object',
  properties: {
    path: { type: 'string', description: 'Absolute or relative file path to read' },
  },
  required: ['path'],
}, 'Read the contents of a file and return them as text.', (input) => {
  try {
    return fs.readFileSync(input.path, 'utf-8');
  } catch (err: any) {
    return `Error reading file: ${err.message}`;
  }
});
```

### write_file

```typescript
const writeFile = tool('write_file', {
  type: 'object',
  properties: {
    path: { type: 'string', description: 'File path to write to' },
    content: { type: 'string', description: 'Content to write' },
  },
  required: ['path', 'content'],
}, 'Write content to a file. Creates the file if it does not exist.', (input) => {
  try {
    const dir = path.dirname(input.path);
    fs.mkdirSync(dir, { recursive: true });
    fs.writeFileSync(input.path, input.content, 'utf-8');
    return `Successfully wrote ${input.content.length} bytes to ${input.path}`;
  } catch (err: any) {
    return `Error writing file: ${err.message}`;
  }
});
```

### run_command

```typescript
import { execSync } from 'node:child_process';

const runCommand = tool('run_command', {
  type: 'object',
  properties: {
    command: { type: 'string', description: 'Shell command to execute' },
    cwd: { type: 'string', description: 'Working directory (optional)' },
  },
  required: ['command'],
}, 'Execute a shell command and return its stdout/stderr.', (input) => {
  try {
    const output = execSync(input.command, {
      cwd: input.cwd || process.cwd(),
      encoding: 'utf-8',
      timeout: 30_000,
      maxBuffer: 1024 * 1024,
    });
    return output || '(no output)';
  } catch (err: any) {
    return `Exit code ${err.status ?? 1}\nstdout: ${err.stdout ?? ''}\nstderr: ${err.stderr ?? ''}`;
  }
});
```

### list_files

```typescript
const listFiles = tool('list_files', {
  type: 'object',
  properties: {
    directory: { type: 'string', description: 'Directory to list' },
    recursive: { type: 'boolean', description: 'List recursively (default false)' },
  },
  required: ['directory'],
}, 'List files in a directory.', (input) => {
  try {
    if (input.recursive) {
      const output = execSync(`find "${input.directory}" -type f | head -200`, {
        encoding: 'utf-8',
      });
      return output || '(empty directory)';
    }
    const entries = fs.readdirSync(input.directory, { withFileTypes: true });
    return entries
      .map((e) => `${e.isDirectory() ? '[dir] ' : '      '}${e.name}`)
      .join('\n');
  } catch (err: any) {
    return `Error listing files: ${err.message}`;
  }
});
```

### search_files

```typescript
const searchFiles = tool('search_files', {
  type: 'object',
  properties: {
    pattern: { type: 'string', description: 'Grep pattern to search for' },
    directory: { type: 'string', description: 'Directory to search in' },
    include: { type: 'string', description: 'File glob to include (e.g. "*.ts")' },
  },
  required: ['pattern', 'directory'],
}, 'Search for a text pattern across files using grep.', (input) => {
  try {
    const includeFlag = input.include ? `--include="${input.include}"` : '';
    const cmd = `grep -rn ${includeFlag} "${input.pattern}" "${input.directory}" | head -50`;
    const output = execSync(cmd, { encoding: 'utf-8' });
    return output || 'No matches found.';
  } catch {
    return 'No matches found.';
  }
});
```

### edit_file

For surgical edits, a string-replacement tool is more efficient than rewriting entire files:

```typescript
const editFile = tool('edit_file', {
  type: 'object',
  properties: {
    path: { type: 'string', description: 'File to edit' },
    old_string: { type: 'string', description: 'Exact text to find and replace' },
    new_string: { type: 'string', description: 'Replacement text' },
  },
  required: ['path', 'old_string', 'new_string'],
}, 'Replace an exact string in a file. Fails if the string is not found or not unique.', (input) => {
  try {
    const content = fs.readFileSync(input.path, 'utf-8');
    const occurrences = content.split(input.old_string).length - 1;

    if (occurrences === 0) {
      return `Error: old_string not found in ${input.path}`;
    }
    if (occurrences > 1) {
      return `Error: old_string found ${occurrences} times. Provide more context to make it unique.`;
    }

    const updated = content.replace(input.old_string, input.new_string);
    fs.writeFileSync(input.path, updated, 'utf-8');
    return `Successfully edited ${input.path}`;
  } catch (err: any) {
    return `Error editing file: ${err.message}`;
  }
});
```

---

## Using generateText with Tools

Pass tools to `generateText` and set `maxSteps` to allow the model to call tools repeatedly:

```typescript
import { createAnthropic, generateText } from 'ai-sdk-cpp';

const anthropic = createAnthropic();
const model = anthropic('claude-sonnet-4-20250514');

const result = await generateText({
  model,
  prompt: 'Read the file src/index.ts and add a comment at the top.',
  tools: [readFile, writeFile, editFile],
  maxSteps: 10,
});

console.log(result.text);
console.log(`Completed in ${result.steps} steps, used ${result.usage.inputTokens} input tokens.`);
```

The engine handles the tool loop internally. Each step, it sends the model's tool calls to your execute functions, feeds the results back, and continues until the model produces a final text response or `maxSteps` is reached.

---

## Streaming

For interactive CLI agents, streaming gives immediate feedback. The `streamText` function returns an async iterable of events:

```typescript
import { createAnthropic, streamText } from 'ai-sdk-cpp';

const anthropic = createAnthropic();
const model = anthropic('claude-sonnet-4-20250514');

for await (const event of streamText({ model, prompt: 'Write a haiku about coding' })) {
  switch (event.type) {
    case 'text_delta':
      process.stdout.write(event.text!);
      break;
    case 'tool_call':
      console.log(`\n[calling ${event.toolName}...]`);
      break;
    case 'tool_result':
      console.log(`[${event.toolName} returned ${event.result!.length} chars]`);
      break;
    case 'finish':
      console.log(`\n\nDone. Reason: ${event.finishReason}`);
      break;
  }
}
```

---

## The Agent Class

For multi-step agents with a persistent instruction set, the `Agent` class wraps `generateText` with a system prompt and higher step limit:

```typescript
import { createAnthropic, Agent } from 'ai-sdk-cpp';

const anthropic = createAnthropic();

const agent = new Agent({
  model: anthropic('claude-sonnet-4-20250514'),
  tools: [readFile, writeFile, editFile, runCommand, listFiles, searchFiles],
  instructions: `You are a senior software engineer working in the user's codebase.

Rules:
- Always read files before editing them.
- Use edit_file for surgical changes. Only use write_file for new files.
- Run tests after making changes.
- Explain what you did after completing the task.`,
  maxSteps: 50,
});

const result = await agent.call('Find and fix the bug where users can sign up with an empty email.');
console.log(result.text);
```

The `Agent` class handles:
- Prepending system instructions to every call
- Executing tools in a loop up to `maxSteps`
- Returning a `GenerateResult` with the final text, usage, and step count

---

## Error Handling

Tools should catch their own errors and return error messages as strings -- the model can then decide how to recover. For the outer agent call, wrap in try/catch:

```typescript
try {
  const result = await agent.call(userInput);
  console.log(result.text);
} catch (err: any) {
  if (err.code === 'MAX_STEPS_REACHED') {
    console.error('Agent hit the step limit. Try a more specific prompt or increase maxSteps.');
  } else if (err.code === 'PROVIDER_ERROR') {
    console.error(`API error: ${err.message}`);
  } else {
    console.error(`Unexpected error: ${err.message}`);
  }
}
```

Common error codes:
- `MAX_STEPS_REACHED` -- the agent used all its steps without finishing
- `PROVIDER_ERROR` -- the upstream API returned an error (rate limit, auth, etc.)
- `TOOL_EXECUTION_ERROR` -- a tool threw an unhandled exception
- `INVALID_API_KEY` -- the API key env var is missing or invalid

---

## Full Working Example: A Coding Agent CLI

Here is a complete, working coding agent in about 70 lines. Save it as `agent.ts` and run with `npx tsx agent.ts`:

```typescript
import * as fs from 'node:fs';
import * as path from 'node:path';
import * as readline from 'node:readline';
import { execSync } from 'node:child_process';
import { createAnthropic, Agent, tool } from 'ai-sdk-cpp';

// --- Tools ---

const readFile = tool('read_file', {
  type: 'object',
  properties: { path: { type: 'string' } },
  required: ['path'],
}, 'Read a file.', (input) => {
  try { return fs.readFileSync(input.path, 'utf-8'); }
  catch (e: any) { return `Error: ${e.message}`; }
});

const writeFile = tool('write_file', {
  type: 'object',
  properties: { path: { type: 'string' }, content: { type: 'string' } },
  required: ['path', 'content'],
}, 'Write a file.', (input) => {
  fs.mkdirSync(path.dirname(input.path), { recursive: true });
  fs.writeFileSync(input.path, input.content);
  return `Wrote ${input.path}`;
});

const editFile = tool('edit_file', {
  type: 'object',
  properties: { path: { type: 'string' }, old_string: { type: 'string' }, new_string: { type: 'string' } },
  required: ['path', 'old_string', 'new_string'],
}, 'Replace exact text in a file.', (input) => {
  const content = fs.readFileSync(input.path, 'utf-8');
  if (!content.includes(input.old_string)) return 'Error: string not found';
  fs.writeFileSync(input.path, content.replace(input.old_string, input.new_string));
  return `Edited ${input.path}`;
});

const runCommand = tool('run_command', {
  type: 'object',
  properties: { command: { type: 'string' } },
  required: ['command'],
}, 'Run a shell command.', (input) => {
  try { return execSync(input.command, { encoding: 'utf-8', timeout: 30_000 }); }
  catch (e: any) { return `Exit ${e.status}: ${e.stderr || e.stdout || e.message}`; }
});

const listFiles = tool('list_files', {
  type: 'object',
  properties: { directory: { type: 'string' } },
  required: ['directory'],
}, 'List files in a directory.', (input) => {
  try { return execSync(`find "${input.directory}" -type f | head -100`, { encoding: 'utf-8' }); }
  catch (e: any) { return `Error: ${e.message}`; }
});

const searchFiles = tool('search_files', {
  type: 'object',
  properties: { pattern: { type: 'string' }, directory: { type: 'string' } },
  required: ['pattern', 'directory'],
}, 'Grep for a pattern.', (input) => {
  try { return execSync(`grep -rn "${input.pattern}" "${input.directory}" | head -30`, { encoding: 'utf-8' }); }
  catch { return 'No matches.'; }
});

// --- Agent ---

const agent = new Agent({
  model: createAnthropic()('claude-sonnet-4-20250514'),
  tools: [readFile, writeFile, editFile, runCommand, listFiles, searchFiles],
  instructions: 'You are a coding agent. Read before editing. Run tests after changes. Be concise.',
  maxSteps: 50,
});

// --- REPL ---

const rl = readline.createInterface({ input: process.stdin, output: process.stdout });

console.log('Coding Agent (type "exit" to quit)\n');

const ask = () => {
  rl.question('> ', async (input) => {
    if (input.trim() === 'exit') { rl.close(); return; }
    try {
      const result = await agent.call(input);
      console.log(`\n${result.text}\n`);
      console.log(`[${result.steps} steps | ${result.usage.inputTokens + result.usage.outputTokens} tokens]\n`);
    } catch (err: any) {
      console.error(`Error: ${err.message}\n`);
    }
    ask();
  });
};

ask();
```

Run it:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
npx tsx agent.ts
```

Then interact:

```
> Fix the type error in src/utils.ts
```

The agent will read the file, identify the issue, make the edit, run the type checker, and report back.

---

## Switching Providers

The entire agent works with any supported provider. Swap one line:

```typescript
import { createOpenAI, Agent } from 'ai-sdk-cpp';

const openai = createOpenAI();
const agent = new Agent({
  model: openai('gpt-4o'),
  // ...everything else stays the same
});
```

---

## When to Use ai-sdk-cpp vs the Vercel AI SDK

| | ai-sdk-cpp | Vercel AI SDK (TypeScript) |
|---|---|---|
| **Runtime** | Native C++ engine with TS bindings | Pure TypeScript |
| **Best for** | CLI tools, native apps, long-running agents, performance-sensitive workloads | Web apps, serverless, Next.js, React Server Components |
| **Startup time** | Fast (precompiled binary) | Depends on bundler/runtime |
| **Streaming** | Async iterables | Async iterables + React hooks |
| **Framework integrations** | None (intentional) | React, Vue, Svelte, Angular, RSC |
| **Tool execution** | Synchronous by default, in-process | Async, supports server actions |
| **Package size** | Larger (includes native binary) | Smaller (pure JS) |
| **Platform** | macOS, Linux, Windows (x64/arm64) | Anywhere Node/Deno/Bun/Edge runs |

**Use ai-sdk-cpp when:**
- You are building a CLI tool (like your own Claude Code)
- You need minimal latency in the tool loop (native JSON parsing, native HTTP)
- You are embedding an agent in a native desktop app (Electron, Tauri)
- You want the fastest possible local execution for autonomous agents

**Use the Vercel AI SDK when:**
- You are building a web application with React/Next.js
- You need server-side streaming to the browser
- You want framework hooks like `useChat` and `useCompletion`
- You are deploying to serverless/edge environments

---

## Next Steps

- Add a **confirmation step** for destructive operations (file writes, command execution)
- Implement **conversation history** so the agent remembers previous interactions
- Add **token budget tracking** to warn before hitting context limits
- Build a **permission system** that sandboxes file access to the project directory
- Integrate **git operations** so the agent can commit its own changes

You now have everything you need to build a production-grade coding agent. The model does the thinking. The tools do the work. The native engine makes it fast. Go build something.
