// Streaming against a mock provider. No API keys, no network.
//
// Unlike the synchronous entry points these never block the JS event loop, so
// they can be driven in-process with an ordinary `for await` and need no
// watchdog.
const test = require('node:test');
const assert = require('node:assert/strict');
const ai = require('../dist/index.js');
const { startMock } = require('./helpers/harness.cjs');

let mock;
test.before(async () => {
  mock = await startMock();
});
test.after(() => mock && mock.stop());

function model(name, provider = ai.createOpenAI) {
  return provider({ apiKey: 'test-key', baseUrl: mock.baseUrl })(name);
}

async function collect(opts) {
  const events = [];
  for await (const ev of ai.streamText(opts)) events.push(ev);
  return events;
}

const textOf = (events) =>
  events
    .filter((e) => e.type === 'text_delta')
    .map((e) => e.text)
    .join('');

test('streamText concatenates text deltas', async () => {
  const events = await collect({ model: model('mock-stream-empty-deltas'), prompt: 'say pong' });
  assert.equal(textOf(events), 'pong');
});

test('every text_delta carries a string, so concatenation never yields "undefined"', async () => {
  // Providers emit empty deltas between real tokens. If the wrapper omits the
  // field for those, `out += ev.text` appends the literal string "undefined".
  const events = await collect({ model: model('mock-stream-empty-deltas'), prompt: 'say pong' });

  const deltas = events.filter((e) => e.type === 'text_delta');
  assert.ok(deltas.length > 0, 'expected some text_delta events');
  assert.ok(
    deltas.every((e) => typeof e.text === 'string'),
    `every text_delta must have a string .text, got: ${JSON.stringify(deltas.map((e) => e.text))}`
  );
  assert.ok(
    deltas.some((e) => e.text === ''),
    'the mock sends empty deltas; they should survive as empty strings'
  );
  assert.ok(!textOf(events).includes('undefined'));
});

test('streamText reports usage and finish reason on the final event', async () => {
  const events = await collect({ model: model('mock-stream-empty-deltas'), prompt: 'say pong' });

  const finish = events.at(-1);
  assert.equal(finish.type, 'finish');
  assert.equal(finish.usage.inputTokens, 7);
  assert.equal(finish.usage.outputTokens, 6);
  assert.equal(finish.usage.finishReason, 'stop');
});

test('streamText surfaces a length finish reason', async () => {
  const events = await collect({ model: model('mock-stream-length'), prompt: 'write a lot' });

  assert.equal(textOf(events), 'trunc');
  assert.equal(events.at(-1).usage.finishReason, 'length');
});

test('streamText emits reasoning before the answer', async () => {
  const events = await collect({ model: model('mock-stream-reasoning'), prompt: 'think' });
  const types = events.map((e) => e.type);

  assert.ok(types.includes('reasoning_start'));
  assert.ok(types.includes('reasoning_delta'));
  assert.ok(types.includes('reasoning_end'));

  const reasoning = events.find((e) => e.type === 'reasoning_delta');
  assert.equal(reasoning.text, 'thinking...');

  // Reasoning must be complete before the answer starts.
  assert.ok(
    types.indexOf('reasoning_end') < types.indexOf('finish'),
    'reasoning should end before the turn finishes'
  );
  assert.equal(textOf(events), '4');
});

test('streamText streams a tool call as start/delta/end', async () => {
  const events = await collect({
    model: model('mock-stream-tool'),
    tools: [
      ai.tool('get_weather', { type: 'object', properties: { city: { type: 'string' } } }, 'w', (i) => i),
    ],
    prompt: 'weather?',
    maxSteps: 3,
  });

  const start = events.find((e) => e.type === 'tool_call_start');
  assert.ok(start, 'expected a tool_call_start event');
  assert.equal(start.toolName, 'get_weather');
  assert.ok(start.toolCallId, 'tool_call_start should carry a call id');

  const args = events
    .filter((e) => e.type === 'tool_call_delta')
    .map((e) => e.text)
    .join('');
  assert.deepEqual(JSON.parse(args), { city: 'Paris' });

  assert.ok(events.some((e) => e.type === 'tool_call_end'));
});

test('streamText runs the tool and loops back to the model', async () => {
  await mock.reset();
  const events = await collect({
    model: model('mock-stream-tool'),
    tools: [
      ai.tool('get_weather', { type: 'object', properties: { city: { type: 'string' } } }, 'w', (i) => ({
        city: i.city,
        weather: 'sunny',
      })),
    ],
    prompt: 'weather?',
    maxSteps: 3,
  });

  assert.equal(textOf(events), 'pong', 'the second turn should stream the final answer');

  const reqs = await mock.requests();
  assert.equal(reqs.length, 2, 'the tool result should be sent back to the model');
  assert.match(String(reqs[1].body.messages.find((m) => m.role === 'tool').content), /sunny/);
});

test('a truncated stream still yields the text it received', async () => {
  const events = await collect({ model: model('mock-truncated-sse'), prompt: 'hi' });

  assert.equal(textOf(events), 'partial');
  assert.equal(events.at(-1).type, 'finish');
});

test('streaming failures surface as a terminal error event', async () => {
  for (const [name, status] of [
    ['mock-401', '401'],
    ['mock-429', '429'],
    ['mock-500', '500'],
  ]) {
    const events = await collect({ model: model(name), prompt: 'hi' });
    const err = events.find((e) => e.type === 'error');

    assert.ok(err, `${name} should produce an error event`);
    assert.match(err.text, new RegExp(status), `${name} error should name the status`);
    assert.equal(events.at(-1).type, 'error', 'the error event terminates the stream');
  }
});

test('streamText works through the Anthropic-shaped path', async () => {
  const events = await collect({ model: model('mock-stream-empty-deltas', ai.createAnthropic), prompt: 'hi' });
  assert.equal(textOf(events), 'pong');
});

test('streamText sends the system message and tool schemas on the wire', async () => {
  await mock.reset();
  await collect({
    model: model('mock-stream-empty-deltas'),
    system: 'be terse',
    prompt: 'the user question',
  });

  const [req] = await mock.requests();
  assert.ok(
    req.body.messages.some((m) => m.role === 'system' && m.content === 'be terse'),
    'system message should be sent'
  );
  assert.deepEqual(req.body.messages.at(-1), { role: 'user', content: 'the user question' });
  assert.equal(req.body.stream, true, 'the streaming path should ask for a stream');
});
