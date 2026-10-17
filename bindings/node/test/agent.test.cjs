// Agent (streaming and error handling) against a mock provider.
// No API keys, no network.
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

async function collect(iterable) {
  const events = [];
  for await (const ev of iterable) events.push(ev);
  return events;
}

const textOf = (events) =>
  events
    .filter((e) => e.type === 'text_delta')
    .map((e) => e.text)
    .join('');

test('Agent.callStream streams a plain answer', async () => {
  const agent = new ai.Agent({ model: model('mock-stream-empty-deltas'), tools: [] });
  const events = await collect(agent.callStream('say pong'));

  assert.equal(textOf(events), 'pong');
  const finish = events.at(-1);
  assert.equal(finish.type, 'finish');
  assert.equal(finish.usage.finishReason, 'stop');
  assert.equal(finish.usage.inputTokens, 7);
  assert.equal(finish.usage.outputTokens, 6);
});

test('Agent.callStream runs the tool loop and streams the final step', async () => {
  await mock.reset();
  const agent = new ai.Agent({
    model: model('mock-stream-tool'),
    tools: [
      ai.tool(
        'get_weather',
        { type: 'object', properties: { city: { type: 'string' } } },
        'w',
        (i) => ({ city: i.city, weather: 'sunny' }),
      ),
    ],
    maxSteps: 3,
  });

  const events = await collect(agent.callStream('weather?'));

  const start = events.find((e) => e.type === 'tool_call_start');
  assert.ok(start, 'expected a tool_call_start event');
  assert.equal(start.toolName, 'get_weather');

  assert.equal(textOf(events), 'pong', 'the second step should stream the final answer');

  // One finish per model call: the tool-call step, then the final answer.
  const finishes = events.filter((e) => e.type === 'finish');
  assert.equal(finishes[0].usage.finishReason, 'tool_calls');
  assert.equal(finishes.at(-1).usage.finishReason, 'stop');

  const reqs = await mock.requests();
  assert.equal(reqs.length, 2, 'the tool result should be sent back to the model');
  assert.match(String(reqs[1].body.messages.find((m) => m.role === 'tool').content), /sunny/);
});

test('Agent.callStream surfaces a stream failure as a terminal error event', async () => {
  const agent = new ai.Agent({ model: model('mock-401'), tools: [] });
  const events = await collect(agent.callStream('hi'));

  const err = events.find((e) => e.type === 'error');
  assert.ok(err, 'a 401 should produce an error event');
  assert.match(err.text, /401/);
  assert.equal(events.at(-1).type, 'error', 'the error event terminates the stream');
});

test('promise-returning calls carry a stable error code', async () => {
  const cases = [
    ['mock-401', 'AI_ERROR_AUTHENTICATION'],
    ['mock-429', 'AI_ERROR_RATE_LIMIT'],
    ['mock-500', 'AI_ERROR_API_CALL'],
  ];

  for (const [name, code] of cases) {
    await assert.rejects(
      () => ai.generateText({ model: model(name), prompt: 'hi' }),
      (err) => {
        assert.equal(err.code, code, `${name} should map to ${code}`);
        return true;
      },
    );
  }
});

test('Agent.call rejections carry the same code', async () => {
  const agent = new ai.Agent({ model: model('mock-401'), tools: [] });
  await assert.rejects(
    () => agent.call('hi'),
    (err) => err.code === 'AI_ERROR_AUTHENTICATION',
  );
});
