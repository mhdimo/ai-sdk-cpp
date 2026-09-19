// Live tier: the same binding paths as the hermetic suite, against a real
// provider. A mock can only prove the binding is self-consistent — it cannot
// prove that a real wire format parses into the shape the binding promises.
//
// Opt-in. Without DEEPSEEK_API_KEY every case skips, so `npm run test:live` is
// safe to run anywhere. The key is read from the environment only; it is never
// written to a file or reported in an assertion message.
//
// Assertions here are about *shape and causality*, never wording: a live model
// may phrase anything any way it likes, but a token it could not have known
// without calling the tool must appear in the answer, and a session must still
// remember what it was told a turn ago.
//
//   DEEPSEEK_API_KEY=... npm run test:live
//
// Each case runs in its own process under a wall-clock budget — the same
// reasoning as the hermetic suite: a hang in a native call blocks the JS event
// loop, so only a timer out here can fail it.
const test = require('node:test');
const assert = require('node:assert/strict');
const path = require('node:path');
const { runIsolated } = require('../helpers/harness.cjs');

const RUNNER = path.join(__dirname, 'live-runner.cjs');

const KEY = process.env.DEEPSEEK_API_KEY;
const MODEL = process.env.LIVE_MODEL || 'deepseek-chat';

// Real models are slow, and the tool cases take two round trips. Generous.
const TIMEOUT_MS = 90000;

/** Run one live scenario out-of-process and return its reported facts. */
async function live(scenario) {
  const r = await runIsolated(RUNNER, {
    env: { LIVE_SCENARIO: scenario },
    timeoutMs: TIMEOUT_MS,
  });

  if (!r.completed) {
    throw new Error(
      `live scenario "${scenario}" did not finish within ${TIMEOUT_MS}ms ` +
        `(killed=${r.killed} signal=${r.signal}) — it hung.\n` +
        `stdout: ${r.stdout.trim() || '<empty>'}\n` +
        `stderr: ${r.stderr.trim() || '<empty>'}`
    );
  }

  const line = r.stdout.split('\n').find((l) => l.startsWith('OK ') || l.startsWith('ERR '));
  if (!line) {
    throw new Error(`live scenario "${scenario}" produced no result\nstdout: ${r.stdout}\nstderr: ${r.stderr}`);
  }
  if (line.startsWith('ERR ')) {
    throw new Error(`live scenario "${scenario}" failed: ${line.slice(4)}`);
  }
  return JSON.parse(line.slice(3));
}

// Everything in this file needs a key; without one, skip the lot with a reason
// that says how to opt in.
const skip = KEY ? false : `set DEEPSEEK_API_KEY to run the live tier (model: ${MODEL})`;

test('live: generateText returns text and token usage', { skip }, async () => {
  const r = await live('generate');

  assert.ok(typeof r.text === 'string' && r.text.trim().length > 0, `expected text, got: ${JSON.stringify(r.text)}`);
  assert.ok(r.inputTokens > 0, `expected a positive input token count, got ${r.inputTokens}`);
  assert.ok(r.outputTokens > 0, `expected a positive output token count, got ${r.outputTokens}`);
  assert.ok(r.steps >= 1, `expected at least one step, got ${r.steps}`);
  assert.ok(
    typeof r.finishReason === 'string' && r.finishReason.length > 0,
    `expected a finish reason, got: ${JSON.stringify(r.finishReason)}`
  );
});

test('live: streamText delivers string deltas and a finish with usage', { skip }, async () => {
  const r = await live('stream');

  assert.ok(r.deltaCount > 0, `expected text deltas, got events: ${r.types.join(', ')}`);
  assert.equal(r.nonStringDeltas, 0, 'every text delta should be a string (an undefined renders as "undefined")');
  assert.ok(r.text.trim().length > 0, `the deltas should concatenate to text, got: ${JSON.stringify(r.text)}`);
  assert.ok(r.types.includes('finish'), `expected a finish event, got: ${r.types.join(', ')}`);
  assert.ok(
    r.finishUsage?.inputTokens > 0,
    `expected usage on the finish event, got: ${JSON.stringify(r.finishUsage)}`
  );
});

test('live: a tool call round-trips through the model', { skip }, async () => {
  const r = await live('tool-call');

  // The token is generated locally and only ever crosses the wire inside the
  // tool result. If it comes back in the answer, the whole loop ran: the call
  // was parsed from the response, executed in JS, and fed back.
  assert.equal(
    r.containsSecret,
    true,
    `the model should not have been able to answer without calling the tool; it said: ${JSON.stringify(r.text)}`
  );
  assert.equal(r.steps, 2, 'a tool call and the answer are two steps');
});

test('live: a streamed tool call round-trips through the model', { skip }, async () => {
  const r = await live('stream-tool');

  assert.ok(r.types.includes('tool_call_start'), `expected a tool_call_start event, got: ${r.types.join(', ')}`);
  assert.ok(r.types.includes('tool_call_end'), `expected a tool_call_end event, got: ${r.types.join(', ')}`);
  assert.equal(
    r.containsSecret,
    true,
    `the streamed tool result should reach the model; it said: ${JSON.stringify(r.text)}`
  );
});

test('live: a session remembers the previous turn', { skip }, async () => {
  const r = await live('session-two-turns');

  assert.ok(r.firstText.trim().length > 0, 'the first turn should answer');
  // A codeword invented locally cannot be recalled unless the first turn's
  // messages were carried into the second request.
  assert.equal(
    r.recalled,
    true,
    `the session should have carried history; the second turn said: ${JSON.stringify(r.secondText)}`
  );
});
