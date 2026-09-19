// generateText (non-streaming) against a mock provider. No API keys, no network.
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

test('generateText returns text, finish reason and usage', async () => {
  const r = await ai.generateText({ model: model('mock-stop'), prompt: 'say pong' });

  assert.equal(r.text, 'pong');
  assert.equal(r.finishReason, 'stop');
  assert.equal(r.usage.inputTokens, 12);
  assert.equal(r.usage.outputTokens, 2);
  assert.equal(r.steps, 1);
});

test('generateText surfaces a length finish reason', async () => {
  const r = await ai.generateText({ model: model('mock-length'), prompt: 'write a lot' });
  assert.equal(r.finishReason, 'length');
});

test('generateText sends the prompt and system message on the wire', async () => {
  await mock.reset();
  await ai.generateText({
    model: model('mock-stop'),
    prompt: 'the user question',
    system: 'be terse',
  });

  const [req] = await mock.requests();
  assert.equal(req.url, '/chat/completions');
  assert.equal(req.body.model, 'mock-stop');
  assert.deepEqual(req.body.messages.at(-1), { role: 'user', content: 'the user question' });
  assert.ok(
    req.body.messages.some((m) => m.role === 'system' && m.content === 'be terse'),
    'system message should be sent'
  );
});

test('generateText forwards maxOutputTokens and temperature', async () => {
  await mock.reset();
  await ai.generateText({
    model: model('mock-stop'),
    prompt: 'hi',
    maxOutputTokens: 42,
    temperature: 0.25,
  });

  const [req] = await mock.requests();
  assert.equal(req.body.max_tokens, 42);
  assert.equal(req.body.temperature, 0.25);
});

test('generateText accepts a messages array', async () => {
  await mock.reset();
  await ai.generateText({
    model: model('mock-stop'),
    messages: [
      { role: 'user', content: 'first' },
      { role: 'assistant', content: 'second' },
      { role: 'user', content: 'third' },
    ],
  });

  const [req] = await mock.requests();
  assert.deepEqual(
    req.body.messages.map((m) => m.content),
    ['first', 'second', 'third']
  );
});

test('generateText maps a 401 to an authentication error', async () => {
  await assert.rejects(
    () => ai.generateText({ model: model('mock-401'), prompt: 'hi' }),
    /auth/i
  );
});

test('generateText maps a 429 to an error mentioning the rate limit', async () => {
  await assert.rejects(
    () => ai.generateText({ model: model('mock-429'), prompt: 'hi' }),
    /rate limit/i
  );
});

test('generateText reports a 500 without hanging', async () => {
  await assert.rejects(() => ai.generateText({ model: model('mock-500'), prompt: 'hi' }));
});

test('generateText reports a non-JSON body instead of crashing', async () => {
  await assert.rejects(() => ai.generateText({ model: model('mock-not-json'), prompt: 'hi' }));
});

test('generateText works through the Anthropic-shaped path', async () => {
  const m = model('mock-stop', ai.createAnthropic);
  const r = await ai.generateText({ model: m, prompt: 'hi' });

  assert.equal(r.text, 'pong');
});

test('generateText sends Anthropic requests to /v1/messages', async () => {
  await mock.reset();
  await ai.generateText({ model: model('mock-stop', ai.createAnthropic), prompt: 'hi' });

  const [req] = await mock.requests();
  assert.equal(req.url, '/v1/messages');
  assert.ok(req.headers['anthropic-version'], 'anthropic-version header should be set');
});
