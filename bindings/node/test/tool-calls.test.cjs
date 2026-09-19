// Tool calls through the synchronous entry points.
//
// Every case here runs in its own process under a wall-clock budget. That is
// not incidental: when one of these entry points hangs it does so at 100% CPU
// with the JS event loop blocked, so an in-process `setTimeout` would never
// fire and the run would wedge instead of failing. The parent kills the child
// from outside.
const test = require('node:test');
const assert = require('node:assert/strict');
const net = require('node:net');
const { startMock, runScenario } = require('./helpers/harness.cjs');

let mock;
test.before(async () => {
  mock = await startMock();
});
test.after(() => mock && mock.stop());

/** Run one scenario in a child process, under a wall-clock budget. */
const run = (scenario, opts = {}) =>
  runScenario(scenario, { env: { MOCK_BASE_URL: mock.baseUrl }, ...opts });

/**
 * A port with nothing listening on it. Binding to 0 and immediately closing
 * hands back a port the kernel just proved was free, so connecting to it is
 * refused at once instead of hanging on a firewall.
 */
const deadPort = () =>
  new Promise((resolve, reject) => {
    const srv = net.createServer();
    srv.on('error', reject);
    srv.listen(0, '127.0.0.1', () => {
      const { port } = srv.address();
      srv.close(() => resolve(port));
    });
  });

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

// The counterpart to the tool-call case above: the provider rejects the request
// itself. Unlike streamText, which has always surfaced this as an error event,
// the session entry point returned the failure status without emitting
// anything, and the binding then synthesized a *finish* event for the missing
// terminal event. The caller saw an empty turn that had succeeded — the one
// outcome a consumer cannot detect, because checking for an error is exactly
// what finds nothing.
test('Session.sendStream surfaces a request failure as an error event', async () => {
  for (const [name, status] of [
    ['mock-401', '401'],
    ['mock-429', '429'],
    ['mock-500', '500'],
  ]) {
    await mock.reset();
    // `run` spreads opts after its own default env, so the whole env has to be
    // passed here -- a partial one would drop MOCK_BASE_URL.
    const r = await run('session-stream-error', {
      env: { MOCK_BASE_URL: mock.baseUrl, MOCK_ERROR_MODEL: name },
    });

    assert.ok(
      r.events.includes('error'),
      `${name}: expected an error event, got: ${r.events.join(', ') || '(none)'}`
    );
    assert.match(
      String(r.errorText),
      new RegExp(status),
      `${name}: the error event should name the status, not be a placeholder`
    );
    assert.equal(
      r.events.at(-1),
      'error',
      `${name}: the error has to terminate the stream, or the caller keeps waiting`
    );
    assert.ok(
      !r.events.includes('finish'),
      `${name}: a failed turn must not also report a clean finish`
    );
  }
});

// The transport case, which the status-code cases above structurally cannot
// reach: the request never gets a response at all. That covers a refused
// connection, DNS failure, TLS failure and timeouts — the failures a machine
// on a flaky network actually sees. Nothing in the stream ever errors, so the
// failure has to be reported by the call itself, and before this was fixed the
// caller got a `finish` carrying zero tokens and an empty finishReason: a
// successful-looking empty turn, which is the one outcome a consumer cannot
// detect, because looking for the error is exactly what finds nothing.
test('Session.sendStream reports a connection failure instead of an empty success', async () => {
  const port = await deadPort();
  await mock.reset();

  // Both agent shapes, because they take different routes through the C entry
  // point: with tools the failure surfaces from inside the stream, without them
  // it lands in the outer catch. A tool set is the *less* common shape for a
  // plain chat session, so covering only the tool-bearing case misses the bug.
  for (const noTools of ['1', '0']) {
    const what = noTools === '1' ? 'without tools' : 'with tools';
    const r = await run('session-stream-error', {
      env: { MOCK_BASE_URL: `http://127.0.0.1:${port}`, SESSION_NO_TOOLS: noTools },
    });

    assert.ok(
      r.events.includes('error'),
      `${what}: expected an error event, got: ${r.events.join(', ') || '(none)'}`
    );
    assert.equal(
      r.events.at(-1),
      'error',
      `${what}: the error has to terminate the stream, or the caller keeps waiting`
    );
    assert.ok(
      !r.events.includes('finish'),
      `${what}: a turn whose request never reached the provider must not report a clean finish`
    );
    assert.ok(
      r.errorText && r.errorText.length > 0,
      `${what}: the error event needs a message, not a placeholder: ${JSON.stringify(r.errorText)}`
    );
  }
});

// What the tool-result path retains, rather than what it returns.
//
// The failure this is for is invisible to every other test here: a binding that
// allocates a buffer per tool call and frees none of them returns exactly the
// right values and leaks the entire output of every call for the life of the
// process. Nothing fails, nothing logs, and the process simply grows -- which
// is why it is worth a test that measures memory instead of assertions.
test('tool results do not accumulate in the process', async () => {
  await mock.reset();
  // A long budget on purpose: this runs 150 round trips through the mock, and
  // the failure it looks for is a slow growth rather than a hang.
  const r = await run('tool-result-memory', { timeoutMs: 180000 });

  // First, that the loop did the work. A scenario that quietly stopped calling
  // the tool would leak nothing and pass the assertion below for the wrong
  // reason -- and the leak would go on being unmeasured.
  assert.equal(
    r.steps,
    (r.calls + 1) * 2,
    `expected every one of the ${r.calls} calls to run the tool and loop back`
  );

  // 150 calls of 1 MiB with a buffer per call is 150 MiB of growth. The fixed
  // version reuses one buffer per thread, so it stays flat. The bar sits at a
  // fifth of the leak: far above allocator and GC noise, so it reports the
  // defect rather than the machine, and far below the defect, so it cannot be
  // passed by accident.
  const limit = (r.bytes * r.calls) / 5;
  const mib = (n) => `${(n / 1048576).toFixed(1)} MiB`;
  assert.ok(
    r.growth < limit,
    `RSS grew ${mib(r.growth)} over ${r.calls} tool calls of ${mib(r.bytes)} each ` +
      `(limit ${mib(limit)}). The tool-result buffer is being allocated per call ` +
      `and never freed; see the ownership note on ai_tool_result_t in ` +
      `bindings/c/ai_sdk.h.`
  );
});
