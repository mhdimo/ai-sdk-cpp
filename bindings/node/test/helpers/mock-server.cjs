// Mock provider server. Canned Anthropic/OpenAI-shaped responses, no network.
//
// MUST run in its own process: the binding's synchronous entry points
// (generateText / Agent.call / Session.send) block the Node event loop while
// they drive the C++ io_context. An in-process server could never answer a
// request whose response the caller is blocking on — it would deadlock.
//
// Scenarios are selected by the `model` field in the request body, so tests
// just ask for a model named after the behaviour they want. Requests are
// routed by path: /chat/completions (OpenAI-compatible) vs /v1/messages
// (Anthropic). Note the Anthropic provider ignores the SSE `event:` name and
// reads only the JSON `type` field inside `data:`.

const http = require('node:http');

// --- OpenAI-compatible (/chat/completions) -------------------------------

const OPENAI_STOP = {
  id: 'chatcmpl-mock',
  object: 'chat.completion',
  created: 1700000000,
  model: 'mock',
  choices: [
    {
      index: 0,
      message: { role: 'assistant', content: 'pong' },
      logprobs: null,
      finish_reason: 'stop',
    },
  ],
  usage: { prompt_tokens: 12, completion_tokens: 2, total_tokens: 14 },
};

function openaiToolCall(name, args, id = 'call_mock_1') {
  return {
    id: 'chatcmpl-mock',
    object: 'chat.completion',
    created: 1700000000,
    model: 'mock',
    choices: [
      {
        index: 0,
        message: {
          role: 'assistant',
          content: 'I will check that for you.',
          tool_calls: [
            {
              index: 0,
              id,
              type: 'function',
              function: { name, arguments: JSON.stringify(args) },
            },
          ],
        },
        logprobs: null,
        finish_reason: 'tool_calls',
      },
    ],
    usage: { prompt_tokens: 30, completion_tokens: 9, total_tokens: 39 },
  };
}

const OPENAI_LENGTH = {
  ...OPENAI_STOP,
  choices: [{ ...OPENAI_STOP.choices[0], finish_reason: 'length' }],
};

/** Build an SSE body the way DeepSeek/OpenAI actually send it. */
function openaiSse(chunks) {
  return chunks.map((c) => `data: ${JSON.stringify(c)}\n\n`).join('');
}

function openaiChunk(delta, finishReason = null, usage = null) {
  return {
    id: 'chatcmpl-mock',
    object: 'chat.completion.chunk',
    created: 1700000000,
    model: 'mock',
    choices: [{ index: 0, delta, logprobs: null, finish_reason: finishReason }],
    usage,
  };
}

const USAGE_CHUNK = openaiChunk(
  { content: '' },
  'stop',
  { prompt_tokens: 7, completion_tokens: 6, total_tokens: 13 }
);

/** Text stream including EMPTY deltas, exactly as the live API produces them. */
const OPENAI_STREAM_EMPTY_DELTAS = openaiSse([
  openaiChunk({ role: 'assistant', content: '' }),
  openaiChunk({ content: 'po' }),
  openaiChunk({ content: '' }),
  openaiChunk({ content: 'ng' }),
  USAGE_CHUNK,
]) + 'data: [DONE]\n\n';

/** A stream that emits content then stops for length. */
const OPENAI_STREAM_LENGTH = openaiSse([
  openaiChunk({ role: 'assistant', content: '' }),
  openaiChunk({ content: 'trunc' }),
  openaiChunk({ content: '' }, 'length', { prompt_tokens: 5, completion_tokens: 3, total_tokens: 8 }),
]) + 'data: [DONE]\n\n';

/** Tool-call stream: argument fragments arrive split across deltas. */
function toolCallSse(name) {
  return (
    openaiSse([
      openaiChunk({ role: 'assistant', content: '' }),
      openaiChunk({
        tool_calls: [
          { index: 0, id: 'call_mock_1', type: 'function', function: { name, arguments: '' } },
        ],
      }),
      openaiChunk({ tool_calls: [{ index: 0, function: { arguments: '{"city":' } }] }),
      openaiChunk({ tool_calls: [{ index: 0, function: { arguments: '"Paris"}' } }] }),
    ]) +
    openaiSse([
      openaiChunk({ content: '' }, 'tool_calls', {
        prompt_tokens: 20,
        completion_tokens: 7,
        total_tokens: 27,
      }),
    ]) +
    'data: [DONE]\n\n'
  );
}

/** Reasoning deltas (reasoning models emit these before content). */
const OPENAI_STREAM_REASONING = openaiSse([
  openaiChunk({ role: 'assistant', content: '' }),
  { ...openaiChunk({ reasoning_content: 'thinking...' }) },
  openaiChunk({ content: '4' }),
  USAGE_CHUNK,
]) + 'data: [DONE]\n\n';

// --- Google (/models/<id>:generateContent) -------------------------------

function googleText(text) {
  return {
    candidates: [
      {
        content: { role: 'model', parts: [{ text }] },
        finishReason: 'STOP',
      },
    ],
    usageMetadata: { promptTokenCount: 5, candidatesTokenCount: 1 },
  };
}

// --- Anthropic (/v1/messages) --------------------------------------------

function anthropicText(text, stopReason = 'end_turn', content = null) {
  return {
    id: 'msg_mock',
    type: 'message',
    role: 'assistant',
    model: 'mock',
    content: content ?? [{ type: 'text', text }],
    stop_reason: stopReason,
    usage: { input_tokens: 10, output_tokens: 2 },
  };
}

const ANTHROPIC_TOOL_USE = anthropicText(null, 'tool_use', [
  {
    type: 'tool_use',
    id: 'toolu_mock_1',
    name: 'get_weather',
    input: { city: 'Paris' },
  },
]);

function anthropicSse(events) {
  // The provider ignores the `event:` name, but real servers send it — include
  // it so the fixture stays honest about the wire format.
  return events
    .map(([name, data]) => `event: ${name}\ndata: ${JSON.stringify(data)}\n\n`)
    .join('');
}

function anthropicStream(text, stopReason = 'end_turn') {
  return anthropicStreamDeltas([text], stopReason);
}

/** Anthropic also emits empty text deltas between real tokens. */
function anthropicStreamDeltas(deltas, stopReason = 'end_turn') {
  return anthropicSse([
    [
      'message_start',
      {
        type: 'message_start',
        message: {
          id: 'msg_mock',
          type: 'message',
          role: 'assistant',
          model: 'mock',
          usage: { input_tokens: 10, output_tokens: 0 },
        },
      },
    ],
    ['content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }],
    ...deltas.map((text) => [
      'content_block_delta',
      { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text } },
    ]),
    ['content_block_stop', { type: 'content_block_stop', index: 0 }],
    ['message_delta', { type: 'message_delta', delta: { stop_reason: stopReason }, usage: { output_tokens: 2 } }],
    ['message_stop', { type: 'message_stop' }],
  ]);
}

function anthropicToolStream(name) {
  return anthropicSse([
    [
      'message_start',
      {
        type: 'message_start',
        message: {
          id: 'msg_mock',
          type: 'message',
          role: 'assistant',
          model: 'mock',
          usage: { input_tokens: 20, output_tokens: 0 },
        },
      },
    ],
    [
      'content_block_start',
      {
        type: 'content_block_start',
        index: 0,
        content_block: { type: 'tool_use', id: 'toolu_mock_1', name },
      },
    ],
    ['content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'input_json_delta', partial_json: '{"city":' } }],
    ['content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'input_json_delta', partial_json: '"Paris"}' } }],
    ['content_block_stop', { type: 'content_block_stop', index: 0 }],
    ['message_delta', { type: 'message_delta', delta: { stop_reason: 'tool_use' }, usage: { output_tokens: 7 } }],
    ['message_stop', { type: 'message_stop' }],
  ]);
}

// --- Scenario selection ---------------------------------------------------

function hasToolResult(messages) {
  return Array.isArray(messages) && messages.some((m) => m && m.role === 'tool');
}

function countToolResults(messages) {
  return Array.isArray(messages) ? messages.filter((m) => m && m.role === 'tool').length : 0;
}

/** Anthropic sends tool results as a *user* message with tool_result blocks. */
function anthropicHasToolResult(messages) {
  if (!Array.isArray(messages)) return false;
  return messages.some(
    (m) =>
      m &&
      Array.isArray(m.content) &&
      m.content.some((b) => b && b.type === 'tool_result')
  );
}

/** Text of the most recent user message, whatever shape `content` takes. */
function lastUserText(messages) {
  if (!Array.isArray(messages)) return '';
  for (let i = messages.length - 1; i >= 0; i--) {
    const m = messages[i];
    if (!m || m.role !== 'user') continue;
    if (typeof m.content === 'string') return m.content;
    if (Array.isArray(m.content)) {
      return m.content.filter((b) => b && b.type === 'text').map((b) => b.text).join('');
    }
  }
  return '';
}

function pickOpenAiScenario(model, body) {
  const messages = body.messages ?? [];
  switch (model) {
    case 'mock-tool-call':
      // First turn asks for the tool; second turn (with the tool result
      // present) returns prose, exercising the loop-back to the model.
      return hasToolResult(messages)
        ? { kind: 'json', value: OPENAI_STOP }
        : { kind: 'json', value: openaiToolCall('get_weather', { city: 'Paris' }) };
    case 'mock-read-file':
      // The prompt IS the path to read, so a test can point this at a temp
      // file without the mock needing to know where that file lives.
      return hasToolResult(messages)
        ? { kind: 'json', value: OPENAI_STOP }
        : { kind: 'json', value: openaiToolCall('read_file', { path: lastUserText(messages) }) };
    case 'mock-read-file-twice':
      // Asks for the same tool twice in one turn. That is the shape a
      // "always allow" needs: the first call is what the approver decides,
      // the second is the same tool asked for again after its result came
      // back, and must not reach the approver.
      return countToolResults(messages) < 2
        ? { kind: 'json', value: openaiToolCall('read_file', { path: lastUserText(messages) }) }
        : { kind: 'json', value: OPENAI_STOP };
    case 'mock-add-numbers':
      // Asks the model to use the MCP server's `add_numbers` tool. MCP tools are
      // offered under their server-qualified name, so that is the name the model
      // must use — the bare `add_numbers` is deliberately not exposed.
      return hasToolResult(messages)
        ? { kind: 'json', value: OPENAI_STOP }
        : { kind: 'json', value: openaiToolCall('mcp__numbers__add_numbers', { a: 2, b: 3 }) };
    case 'mock-length':
      return { kind: 'json', value: OPENAI_LENGTH };
    case 'mock-stream':
    case 'mock-stream-empty-deltas':
      return { kind: 'sse', value: OPENAI_STREAM_EMPTY_DELTAS };
    case 'mock-stream-length':
      return { kind: 'sse', value: OPENAI_STREAM_LENGTH };
    case 'mock-stream-reasoning':
      return { kind: 'sse', value: OPENAI_STREAM_REASONING };
    case 'mock-stream-tool':
      return hasToolResult(messages)
        ? { kind: 'sse', value: OPENAI_STREAM_EMPTY_DELTAS }
        : { kind: 'sse', value: toolCallSse('get_weather') };
    case 'mock-401':
      return {
        kind: 'json',
        status: 401,
        value: { error: { message: 'Invalid API key', type: 'authentication_error' } },
      };
    case 'mock-429':
      return {
        kind: 'json',
        status: 429,
        headers: { 'retry-after': '3' },
        value: { error: { message: 'Rate limit exceeded', type: 'rate_limit_error' } },
      };
    case 'mock-500':
      return { kind: 'json', status: 500, value: { error: { message: 'Server exploded' } } };
    case 'mock-400':
      return { kind: 'json', status: 400, value: { error: { message: 'Bad request' } } };
    case 'mock-malformed': {
      // Top-level body is not a JSON object -> provider's as_object() throws.
      return { kind: 'raw', value: '"just a string"' };
    }
    case 'mock-not-json':
      return { kind: 'raw', value: 'this is not json at all' };
    case 'mock-truncated-sse':
      // Stream ends without a finish chunk or [DONE].
      return {
        kind: 'sse',
        value: `data: ${JSON.stringify(openaiChunk({ role: 'assistant', content: 'partial' }))}\n\n`,
      };
    default:
      return { kind: 'json', value: OPENAI_STOP };
  }
}

function pickAnthropicScenario(model, body) {
  const messages = body.messages ?? [];
  switch (model) {
    case 'mock-tool-call':
      return anthropicHasToolResult(messages)
        ? { kind: 'json', value: anthropicText('pong') }
        : { kind: 'json', value: ANTHROPIC_TOOL_USE };
    case 'mock-stream':
      return { kind: 'sse', value: anthropicStream('pong') };
    case 'mock-stream-empty-deltas':
      return { kind: 'sse', value: anthropicStreamDeltas(['', 'po', '', 'ng', '']) };
    case 'mock-stream-tool':
      return anthropicHasToolResult(messages)
        ? { kind: 'sse', value: anthropicStream('done') }
        : { kind: 'sse', value: anthropicToolStream('get_weather') };
    case 'mock-401':
      return {
        kind: 'json',
        status: 401,
        value: { type: 'error', error: { type: 'authentication_error', message: 'Invalid API key' } },
      };
    case 'mock-429':
      return {
        kind: 'json',
        status: 429,
        headers: { 'retry-after': '3' },
        value: { type: 'error', error: { type: 'rate_limit_error', message: 'Rate limited' } },
      };
    default:
      return { kind: 'json', value: anthropicText('pong') };
  }
}

/**
 * Route by URL path, then pick the scenario by model name.
 *
 * Path, not body, distinguishes the providers: Google puts the model in the
 * URL (`/models/<id>:generateContent`) and never sends a `model` field.
 */
function pickScenario(path, model, body) {
  if (path.includes(':generateContent')) {
    return { kind: 'json', value: googleText('pong') };
  }
  if (path.startsWith('/v1/messages')) {
    return pickAnthropicScenario(model, body);
  }
  return pickOpenAiScenario(model, body);
}

// --- Anthropic batch (/v1/messages/batches) ------------------------------

// A batch is submit -> poll until terminal -> fetch JSONL results. The mock
// reports `in_progress` on the first poll and `ended` on the next, so the
// caller's wait loop runs exactly twice.
let batchRequests = [];
let batchPolls = 0;

/** First text block of the first user message, for echoing back. */
function promptFromParams(params) {
  const messages = params && params.messages;
  if (!Array.isArray(messages)) return '';
  const text = lastUserText(messages);
  return text || '(no prompt)';
}

function batchResultsJsonl() {
  return (
    batchRequests
      .map((r) =>
        JSON.stringify({
          custom_id: r.custom_id,
          result: {
            type: 'succeeded',
            message: {
              id: 'msg_mock_batch',
              type: 'message',
              role: 'assistant',
              model: 'mock',
              content: [{ type: 'text', text: `echo:${promptFromParams(r.params)}` }],
              stop_reason: 'end_turn',
              usage: { input_tokens: 3, output_tokens: 1 },
            },
          },
        })
      )
      .join('\n') + '\n'
  );
}

function handleBatch(req, res, body, path) {
  const send = (obj) => {
    res.writeHead(200, { 'content-type': 'application/json' });
    res.end(JSON.stringify(obj));
  };

  if (req.method === 'POST' && path === '/v1/messages/batches') {
    batchRequests = Array.isArray(body.requests) ? body.requests : [];
    batchPolls = 0;
    send({
      id: 'msgbatch_mock',
      processing_status: 'in_progress',
      request_counts: { processing: batchRequests.length },
    });
    return;
  }

  if (req.method === 'GET' && path === '/v1/messages/batches/msgbatch_mock') {
    const done = batchPolls++ > 0;
    send({
      id: 'msgbatch_mock',
      processing_status: done ? 'ended' : 'in_progress',
      created_at: '2024-01-01T00:00:00Z',
      ended_at: done ? '2024-01-01T00:01:00Z' : undefined,
      request_counts: done
        ? { succeeded: batchRequests.length }
        : { processing: batchRequests.length },
    });
    return;
  }

  if (req.method === 'GET' && path === '/v1/messages/batches/msgbatch_mock/results') {
    res.writeHead(200, { 'content-type': 'application/x-jsonl' });
    res.end(batchResultsJsonl());
    return;
  }

  res.writeHead(404, { 'content-type': 'application/json' });
  res.end(JSON.stringify({ error: { message: `no batch route for ${req.method} ${path}` } }));
}

// --- Server ---------------------------------------------------------------

let requestLog = [];

const server = http.createServer((req, res) => {
  // Control endpoint: lets tests assert what the binding actually put on the
  // wire (request path, body, provider options) without touching the mock.
  if (req.method === 'GET' && req.url === '/__requests') {
    res.writeHead(200, { 'content-type': 'application/json' });
    res.end(JSON.stringify(requestLog));
    return;
  }
  if (req.method === 'POST' && req.url === '/__reset') {
    requestLog = [];
    res.writeHead(200);
    res.end('ok');
    return;
  }

  const chunks = [];
  req.on('data', (c) => chunks.push(c));
  req.on('end', () => {
    const raw = Buffer.concat(chunks).toString();
    let body = {};
    try {
      body = JSON.parse(raw);
    } catch {
      /* leave body empty; scenario selection falls back to the default */
    }

    const path = req.url.split('?')[0];
    requestLog.push({ url: req.url, headers: req.headers, body, raw });

    // Batch routes share the /v1/messages prefix, so they come first.
    if (path.startsWith('/v1/messages/batches')) {
      handleBatch(req, res, body, path);
      return;
    }

    const model = typeof body.model === 'string' ? body.model : '';
    const scenario = pickScenario(path, model, body);
    const status = scenario.status ?? 200;
    const headers = { 'content-type': scenario.kind === 'sse' ? 'text/event-stream' : 'application/json', ...(scenario.headers ?? {}) };

    if (scenario.kind === 'sse') {
      res.writeHead(status, headers);
      res.write(scenario.value);
      res.end();
      return;
    }
    res.writeHead(status, headers);
    res.end(scenario.kind === 'raw' ? scenario.value : JSON.stringify(scenario.value));
  });
});

server.listen(Number(process.argv[2] || 0), '127.0.0.1', () => {
  // The parent reads this line to learn the ephemeral port.
  console.log(`MOCK_PORT=${server.address().port}`);
});

module.exports = { server };
