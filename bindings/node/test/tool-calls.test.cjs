// Tool calls through the synchronous entry points.
//
// Every case here runs in its own process under a wall-clock budget. That is
// not incidental: when one of these entry points hangs it does so at 100% CPU
// with the JS event loop blocked, so an in-process `setTimeout` would never
// fire and the run would wedge instead of failing. The parent kills the child
// from outside.
const test = require('node:test');
const assert = require('node:assert/strict');
const { startMock, runScenario } = require('./helpers/harness.cjs');

let mock;
test.before(async () => {
  mock = await startMock();
});
test.after(() => mock && mock.stop());

/** Run one scenario in a child process, under a wall-clock budget. */
const run = (scenario, opts = {}) =>
  runScenario(scenario, { env: { MOCK_BASE_URL: mock.baseUrl }, ...opts });

test('generateText runs a tool call and loops back to the model', async () => {
  await mock.reset();
  const r = await run('generateText-tool');

  assert.equal(r.steps, 2, 'should take two steps: tool call, then the answer');
  assert.equal(r.text, 'pong');

  // The tool result must actually reach the model on the follow-up request.
  const reqs = await mock.requests();
  assert.equal(reqs.length, 2, 'expected the tool result to be sent back to the model');

  const followUp = reqs[1].body.messages;
  const toolMsg = followUp.find((m) => m.role === 'tool');
  assert.ok(toolMsg, 'follow-up request should carry a tool result message');
  assert.match(String(toolMsg.content), /sunny/, 'tool result should contain what the tool returned');
});

test('Agent.call runs a tool call and loops back to the model', async () => {
  await mock.reset();
  const r = await run('agent-call-tool');

  assert.equal(r.steps, 2);
  assert.equal(r.text, 'pong');

  const reqs = await mock.requests();
  assert.equal(reqs.length, 2);
  assert.ok(
    reqs[1].body.messages.some((m) => m.role === 'tool'),
    'follow-up request should carry a tool result message'
  );
});

test('Session.send runs a tool call', async () => {
  await mock.reset();
  const r = await run('session-send-tool');

  assert.equal(r.text, 'pong');
});

test('a tool that awaits real async work still completes', async () => {
  // The tool's timer can only fire if the JS event loop is free while the
  // native call is in flight. This is the case a blocked main thread loses.
  await mock.reset();
  const r = await run('generateText-async-tool');

  assert.equal(r.steps, 2);
  assert.equal(r.text, 'pong');

  const reqs = await mock.requests();
  assert.match(String(reqs[1].body.messages.find((m) => m.role === 'tool').content), /sunny/);
});

test('a throwing tool is reported to the model instead of killing the turn', async () => {
  await mock.reset();
  const r = await run('generateText-throwing-tool');

  assert.equal(r.text, 'pong', 'the loop should continue after a tool throws');

  const reqs = await mock.requests();
  const toolMsg = reqs[1].body.messages.find((m) => m.role === 'tool');
  assert.match(String(toolMsg.content), /weather service is down/, 'the error should reach the model');
});

test('a session survives more than one tool-calling turn', async () => {
  await mock.reset();
  const r = await run('session-send-twice');

  assert.equal(r.first, 'pong');
  assert.equal(r.second, 'pong');
});

test('Session.sendStream runs a tool call', async () => {
  await mock.reset();
  const r = await run('session-stream-tool');

  assert.ok(
    r.events.includes('tool_call_start'),
    `expected a tool_call_start event, got: ${r.events.join(', ')}`
  );
  assert.ok(
    r.events.includes('tool_call_end'),
    `expected a tool_call_end event, got: ${r.events.join(', ')}`
  );

  const reqs = await mock.requests();
  assert.equal(reqs.length, 2, 'expected the streamed tool result to be sent back to the model');
});
