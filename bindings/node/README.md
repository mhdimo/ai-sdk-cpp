# ai-sdk-cpp

Node.js bindings for [ai-sdk-cpp](https://github.com/mhdimo/ai-sdk-cpp), a C++20
AI agent framework. A N-API addon over the framework's C ABI — the orchestration
layer between your application and LLM providers, as native code rather than a
JS reimplementation.

```bash
npm install ai-sdk-cpp
```

Requires Node 18 or newer. Prebuilt binaries ship for `darwin-arm64`,
`darwin-x64`, `linux-arm64` and `linux-x64`, so installing does not need a
compiler, a C++ toolchain, or a network fetch of one. On any other platform the
install fails loudly and names the platforms that are supported, rather than
leaving you with a package that breaks at `require()`.

## Quick start

```js
const { createAnthropic, generateText } = require('ai-sdk-cpp');

const anthropic = createAnthropic();              // reads ANTHROPIC_API_KEY
const model = anthropic('claude-sonnet-4-20250514');

const result = await generateText({
  model,
  prompt: 'Summarise what a coroutine is in two sentences.',
});

console.log(result.text);
console.log(result.usage);        // { inputTokens, outputTokens }
```

Pass credentials explicitly when you would rather not depend on the ambient
environment:

```js
const openai = createOpenAI({ apiKey: 'sk-…', baseUrl: 'https://api.openai.com/v1' });
```

Available provider factories: `createAnthropic`, `createOpenAI`, `createGoogle`,
`createDeepSeek`, `createZai`, `createDeepSeekAnthropic`, `createZaiOpenAI`.
Each reads its own environment variable (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`,
`GOOGLE_GENERATIVE_AI_API_KEY`, `DEEPSEEK_API_KEY`, `ZAI_API_KEY`) and honours a
`<PROVIDER>_BASE_URL` override where the provider supports one.

## Streaming

```js
const { streamText } = require('ai-sdk-cpp');

for await (const event of streamText({ model, prompt: 'Write a haiku.' })) {
  if (event.type === 'text_delta') process.stdout.write(event.text);
  if (event.type === 'error') console.error('stream failed:', event.text);
}
```

Event types: `text_delta`, `tool_call_start`, `tool_call_delta`,
`tool_call_end`, `reasoning_start`, `reasoning_delta`, `reasoning_end`,
`tool_result`, `step_finish`, `finish`, `error`. Token usage arrives on the
`finish` event.

**Streaming reports failures as a terminal `error` event; it does not throw.**
Always check for it — a stream that ends without one is a stream that succeeded.

## Tools and agents

```js
const { tool, Agent } = require('ai-sdk-cpp');

const add = tool(
  'add',
  { type: 'object', properties: { a: { type: 'number' }, b: { type: 'number' } },
    required: ['a', 'b'], additionalProperties: false },
  'Add two numbers.',
  ({ a, b }) => a + b
);

const agent = new Agent({ model, tools: [add], instructions: 'Be concise.' });
const result = await agent.call('What is 17 + 25?');
```

The agent runs the tool loop itself: it calls the model, executes the tools,
feeds the results back, and repeats until the model finishes or `maxSteps` is
reached. Independent tool calls within a single step run concurrently.

## Sessions

A `Session` holds conversation history across turns and applies a context
strategy, so long conversations keep working without you managing the window.

```js
const { Session } = require('ai-sdk-cpp');

const session = new Session(agent, { memoryDir: './.memory', maxContextTokens: 100_000 });
await session.send('My name is Liang.');
const answer = await session.send('What did I just tell you?');

// Streaming turns deliver events on the event loop, so a UI stays responsive.
for await (const event of session.sendStream('And now a long answer…')) {
  if (event.type === 'text_delta') process.stdout.write(event.text);
}
```

With `memoryDir` set, relevant persisted memory is injected before each turn and
history auto-compacts near the token budget. `enableCheckpoint` (default `true`)
writes a summary into memory every five turns; that costs an extra model call,
so set it to `false` if you do not want it.

## Other entry points

- `standardToolkit()` — `read_file`, `write_file`, `edit_file`, `glob`, `grep`, `bash`.
- `withPermissions(tools, policy, approver?)` — wrap a tool set in an approval
  policy, with a per-session allow-always cache and a fail-closed default.
- `mcpToolsetFromServer(configJson)` — tools from an MCP server.
- `mergeToolSets(dest, src)` — combine tool sets.
- `describeToolSet(tools)` — introspect a set as `{ name, description, inputSchema }`.
- `MemoryStore`, `Batch`, `version()`.

## Behaviour worth knowing before you file a bug

- **An ambient `ANTHROPIC_AUTH_TOKEN` outranks an explicitly passed `apiKey`.**
  The Anthropic provider offers two credentials and prefers the Bearer token, so
  if your environment carries `ANTHROPIC_AUTH_TOKEN` — as it does for anyone
  running Claude Code against an Anthropic-compatible gateway —
  `createAnthropic({ apiKey })` authenticates with the ambient token instead.
  Unset the variable, or pass `authToken` deliberately.
- **Passing the wrong kind of object to an entry point can abort the process
  rather than throw.** Several wrapper types unwrap their arguments without
  checking the unwrap succeeded. `describeToolSet()` is guarded; the rest are
  not, as of 1.0.0.
- **`Session.send()` and `Agent.call()` are async.** They return a promise and
  do not block the event loop for the turn. A turn that calls a tool has to hand
  control back to the event loop that the tool's own JS runs on.

## License

MIT — see [LICENSE](LICENSE).
