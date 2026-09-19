// The rest of the public surface: version, every provider factory, the
// standard toolkit, permissions, memory, batch, and MCP.
//
// Two safety models, chosen by whether the call can deadlock:
//   - nothing that calls back into JS: plain in-process tests
//   - anything that runs a JS tool callback: the isolated scenario runner,
//     because a deadlock there blocks the JS event loop and an in-process
//     timer could never fire to fail the test
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const ai = require('../dist/index.js');
const { startMock, runScenario } = require('./helpers/harness.cjs');

const pkg = require('../package.json');
const MCP_SERVER = path.join(__dirname, 'helpers', 'mcp-server.cjs');

let mock;
test.before(async () => {
  mock = await startMock();
});
test.after(() => mock && mock.stop());

function tmpdir(prefix) {
  return fs.mkdtempSync(path.join(os.tmpdir(), prefix));
}

/** Every file under `dir`, recursively. */
function walk(dir) {
  const out = [];
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) out.push(...walk(full));
    else out.push(full);
  }
  return out;
}

/** The tool-result message from the model's follow-up request. */
async function toolResultContent() {
  const reqs = await mock.requests();
  assert.equal(reqs.length, 2, 'expected a follow-up request carrying the tool result');
  const msg = reqs[1].body.messages.find((m) => m.role === 'tool');
  assert.ok(msg, 'follow-up request should carry a tool result message');
  return String(msg.content);
}

// --- version --------------------------------------------------------------

test('version() agrees with the published package version', () => {
  // A release that bumps one and not the other ships an SDK that lies about
  // which version it is.
  assert.equal(ai.version(), pkg.version);
});

// --- providers ------------------------------------------------------------

// Each provider must actually reach its own endpoint. Path, not body,
// identifies Google: it puts the model in the URL and sends no `model` field.
const PROVIDERS = [
  ['createAnthropic', (u) => u.startsWith('/v1/messages')],
  ['createDeepSeekAnthropic', (u) => u.startsWith('/v1/messages')],
  ['createZai', (u) => u.startsWith('/v1/messages')],
  ['createOpenAI', (u) => u === '/chat/completions'],
  ['createDeepSeek', (u) => u === '/chat/completions'],
  ['createZaiOpenAI', (u) => u === '/chat/completions'],
  ['createGoogle', (u) => u.includes(':generateContent')],
];

for (const [factory, matchesUrl] of PROVIDERS) {
  test(`${factory}() generates text against its own endpoint`, async (t) => {
    await mock.reset();
    assert.equal(typeof ai[factory], 'function', `${factory} should be exported`);

    let result;
    try {
      const provider = ai[factory]({ apiKey: 'test-key', baseUrl: mock.baseUrl });
      result = await ai.generateText({ model: provider('mock-plain'), prompt: 'hi' });
    } catch (e) {
      // A provider that is compiled out of the build cannot be constructed.
      // That is a build-configuration choice, not a defect — but say so
      // loudly rather than quietly passing.
      if (/Failed to create provider/.test(e.message)) {
        t.skip(`${factory} is not compiled into this build (AI_SDK_PROVIDER_* off)`);
        return;
      }
      throw e;
    }

    assert.equal(result.text, 'pong', `${factory} should return the model's text`);

    const reqs = await mock.requests();
    assert.equal(reqs.length, 1, `${factory} should make exactly one request`);
    assert.ok(matchesUrl(reqs[0].url), `${factory} hit an unexpected path: ${reqs[0].url}`);
  });
}

test('provider factories expose both call and .model() forms', () => {
  const provider = ai.createOpenAI({ apiKey: 'test-key', baseUrl: mock.baseUrl });

  const called = provider('my-model');
  const viaMethod = provider.model('my-model');

  assert.equal(called.modelId, 'my-model');
  assert.equal(viaMethod.modelId, 'my-model');
  assert.equal(called.provider, 'openai');
});

test('an API error rejects the promise instead of hanging', async () => {
  const model = ai.createOpenAI({ apiKey: 'test-key', baseUrl: mock.baseUrl })('mock-401');

  await assert.rejects(
    () => ai.generateText({ model, prompt: 'hi' }),
    /401|Invalid API key|Authentication failed/i,
    'a 401 should surface as a rejection the caller can catch'
  );
});

// --- standard toolkit -----------------------------------------------------

test('standardToolkit() reads a real file and hands it to the model', async () => {
  const file = path.join(tmpdir('ai-sdk-toolkit-'), 'notes.txt');
  fs.writeFileSync(file, 'the secret handshake is pineapple');

  await mock.reset();
  const r = await runScenario('standard-toolkit-read-file', {
    env: { MOCK_BASE_URL: mock.baseUrl, TARGET_FILE: file },
  });

  assert.equal(r.text, 'pong');
  assert.equal(r.steps, 2, 'the tool call and the answer are two steps');
  assert.match(await toolResultContent(), /pineapple/, 'the file contents should reach the model');
});

// --- permissions ----------------------------------------------------------

test('withPermissions() runs a tool the policy allows', async () => {
  const file = path.join(tmpdir('ai-sdk-perm-'), 'notes.txt');
  fs.writeFileSync(file, 'the secret handshake is pineapple');

  await mock.reset();
  const r = await runScenario('permissions-allow', {
    env: { MOCK_BASE_URL: mock.baseUrl, TARGET_FILE: file },
  });

  assert.equal(r.text, 'pong');
  assert.deepEqual(r.policyCalls, ['read_file'], 'the policy should be consulted for the tool');
  assert.match(await toolResultContent(), /pineapple/);
});

test('withPermissions() blocks a denied tool and tells the model why', async () => {
  const file = path.join(tmpdir('ai-sdk-perm-'), 'notes.txt');
  fs.writeFileSync(file, 'the secret handshake is pineapple');

  await mock.reset();
  const r = await runScenario('permissions-deny', {
    env: { MOCK_BASE_URL: mock.baseUrl, TARGET_FILE: file },
  });

  assert.deepEqual(r.policyCalls, ['read_file']);

  const content = await toolResultContent();
  assert.match(content, /permission_denied/, 'the model should learn the call was denied');
  assert.doesNotMatch(content, /pineapple/, 'a denied tool must not disclose file contents');
});

// --- the interactive approver --------------------------------------------
//
// `Ask` is the decision a policy gives when it does not want to answer — the
// case every tool no settings rule mentions lands in. Without an approver that
// is a Deny, which is what these tests would find if the approver never made
// it across the binding.

/** A file whose contents only ever reach the model if the read really ran. */
function secretFile() {
  const file = path.join(tmpdir('ai-sdk-ask-'), 'notes.txt');
  fs.writeFileSync(file, 'the secret handshake is pineapple');
  return file;
}

const ask = (scenario, file) =>
  runScenario(scenario, { env: { MOCK_BASE_URL: mock.baseUrl, TARGET_FILE: file } });

test('an approver can allow what the policy left undecided', async () => {
  const file = secretFile();
  await mock.reset();

  const r = await ask('permissions-ask-allow', file);

  assert.equal(r.text, 'pong');
  assert.equal(r.steps, 2, 'the call should have run: tool call, then the answer');
  assert.deepEqual(r.policyCalls, ['read_file'], 'the policy is consulted first');
  assert.equal(r.approverCalls.length, 1, 'and the ask reached the approver exactly once');
  assert.equal(r.approverCalls[0].tool, 'read_file');
  assert.match(r.approverCalls[0].inputJson, /notes\.txt/, 'the approver sees the input');
  assert.ok(r.rationale, 'the engine supplies a rationale to show the user');
  assert.match(await toolResultContent(), /pineapple/, 'an approved tool actually ran');
});

test('an approver can refuse', async () => {
  const file = secretFile();
  await mock.reset();

  const r = await ask('permissions-ask-deny', file);

  assert.equal(r.approverCalls.length, 1, 'the refusal came from the approver, not a missing one');
  const content = await toolResultContent();
  assert.match(content, /permission_denied/);
  assert.doesNotMatch(content, /pineapple/, 'a refused tool must not disclose file contents');
});

test('a policy can hand the model a reason for its refusal', async () => {
  const file = secretFile();
  await mock.reset();

  const r = await runScenario('permissions-deny-reason', {
    env: { MOCK_BASE_URL: mock.baseUrl, TARGET_FILE: file },
  });

  assert.deepEqual(r.policyCalls, ['read_file']);
  assert.equal(r.steps, 2, 'a refusal ends the loop, it does not abort it');

  const content = await toolResultContent();
  assert.match(content, /permission_denied/);
  assert.match(
    content,
    /that path is outside the project root/,
    'the reason has to survive the trip through the C API into the tool result',
  );
  assert.doesNotMatch(content, /pineapple/);
});

test("an approver's reason overrides the policy's", async () => {
  const file = secretFile();
  await mock.reset();

  const r = await ask('permissions-ask-deny-reason', file);

  assert.equal(r.approverCalls.length, 1);
  const content = await toolResultContent();
  assert.match(content, /permission_denied/);
  assert.match(
    content,
    /the user declined this particular call/,
    'the approver answered second, so its reason is the one worth showing',
  );
  assert.doesNotMatch(content, /pineapple/);
});

test('with no approver an undecided call still fails closed', async () => {
  const file = secretFile();
  await mock.reset();

  const r = await ask('permissions-ask-no-approver', file);

  assert.deepEqual(r.approverCalls, [], 'there was no approver to call');
  const content = await toolResultContent();
  assert.match(content, /permission_denied/, 'Ask without an approver is a Deny');
  assert.doesNotMatch(content, /pineapple/);
});

test('a verdict that is not a decision refuses the call', async () => {
  const file = secretFile();
  await mock.reset();

  const r = await ask('permissions-ask-junk', file);

  assert.equal(r.approverCalls.length, 1, 'the approver was the thing that answered');
  const content = await toolResultContent();
  assert.match(content, /permission_denied/, 'a gate reads an unknown verdict as a refusal');
  assert.doesNotMatch(content, /pineapple/);
});

test('a policy that decides on its own never reaches the approver', async () => {
  const file = secretFile();
  await mock.reset();

  const r = await ask('permissions-allow-skips-approver', file);

  assert.deepEqual(r.approverCalls, [], 'only Ask escalates');
  assert.match(await toolResultContent(), /pineapple/);
});

test('always-allow stops asking for that tool for the rest of the session', async () => {
  const file = secretFile();
  await mock.reset();

  const r = await ask('permissions-ask-always', file);

  assert.equal(r.text, 'pong');
  assert.equal(
    r.approverCalls.length,
    1,
    'the second call of the same tool must not ask again'
  );

  // …and the second call still ran: both results carry the file's contents.
  const reqs = await mock.requests();
  assert.equal(reqs.length, 3, 'tool call, tool call again, then the answer');
  const results = reqs[2].body.messages.filter((m) => m.role === 'tool');
  assert.equal(results.length, 2, 'both calls produced a result');
  for (const msg of results) {
    assert.match(String(msg.content), /pineapple/, 'an allowed call ran, not just the first');
  }
});

// --- merging tool sets ----------------------------------------------------

test('mergeToolSets() keeps the merged definitions usable', async () => {
  const file = path.join(tmpdir('ai-sdk-merge-'), 'notes.txt');
  fs.writeFileSync(file, 'the secret handshake is pineapple');

  await mock.reset();
  const r = await runScenario('merge-toolsets', {
    env: { MOCK_BASE_URL: mock.baseUrl, TARGET_FILE: file },
  });

  assert.equal(r.text, 'pong');
  assert.match(await toolResultContent(), /pineapple/);
});

// --- describing tool sets -------------------------------------------------

test('describeToolSet() reports what a set exposes, in name order', () => {
  const described = ai.describeToolSet(ai.standardToolkit());

  assert.ok(Array.isArray(described));
  assert.ok(described.length >= 5, 'the standard toolkit has several tools');

  const names = described.map((t) => t.name);
  assert.deepEqual(
    names,
    [...names].sort(),
    'the order is stable, or two descriptions cannot be compared'
  );
  assert.ok(names.includes('read_file'));

  const readFile = described.find((t) => t.name === 'read_file');
  assert.equal(typeof readFile.description, 'string');
  assert.equal(typeof readFile.inputSchema, 'object');
  assert.equal(readFile.inputSchema.type, 'object');
});

test('describeToolSet() throws on something that is not a tool set', () => {
  // The assertion is only that it throws. Unwrapping a plain {} yields a null
  // handle, so without the guard this is a segfault that takes the host process
  // down rather than an exception the caller can catch -- and the message
  // belongs to node-addon-api, not to us.
  assert.throws(() => ai.describeToolSet({}));
});

test('describeToolSet() reads through a wrapped tool set', () => {
  // ToolSet has no public constructor, so the interesting case is a set the
  // public API derives rather than builds: if this only worked on a raw
  // toolkit it would be useless for the MCP and permissions sets, which are
  // exactly the ones you cannot otherwise see inside.
  const gated = ai.withPermissions(ai.standardToolkit(), () => ai.PermissionDecision.Allow);

  const names = ai.describeToolSet(gated).map((t) => t.name);
  assert.ok(names.includes('read_file'), 'the wrapper does not hide the tools it wraps');
  assert.deepEqual(names, [...names].sort(), 'order stays stable through a wrapper');
});

// --- sessions with memory -------------------------------------------------

test('a session with memoryDir runs the turn and creates its memory directory', async () => {
  const dir = tmpdir('ai-sdk-session-mem-');
  const file = path.join(tmpdir('ai-sdk-session-file-'), 'notes.txt');
  fs.writeFileSync(file, 'the secret handshake is pineapple');

  await mock.reset();
  const r = await runScenario('session-with-memory', {
    env: { MOCK_BASE_URL: mock.baseUrl, TARGET_FILE: file, MEMORY_DIR: dir },
  });

  assert.equal(r.text, 'pong');
  assert.ok(fs.existsSync(dir), 'the session should have created its memory directory');
});

// --- MemoryStore ----------------------------------------------------------

test('MemoryStore writes a record to disk as markdown', () => {
  const dir = tmpdir('ai-sdk-memory-');
  const store = new ai.MemoryStore(dir);
  store.save('project', 'build-command', 'cmake --build build -j8');

  const files = walk(dir);
  assert.ok(files.length > 0, `expected a file under ${dir}`);
  assert.ok(
    files.some((f) => f.endsWith('.md')),
    `expected a markdown record, got: ${files.join(', ')}`
  );

  const contents = files.map((f) => fs.readFileSync(f, 'utf8')).join('\n');
  assert.match(contents, /cmake --build build -j8/);
  assert.match(contents, /build-command/);
});

// --- Batch ----------------------------------------------------------------

test('Batch submits, polls, and returns one result per request', async () => {
  await mock.reset();
  const provider = ai.createAnthropic({ apiKey: 'test-key', baseUrl: mock.baseUrl });
  const batch = new ai.Batch(provider, 'claude-sonnet-4-20250514');

  const r = batch.run(
    [
      { customId: 'first', prompt: 'one' },
      { customId: 'second', prompt: 'two' },
    ],
    1 // poll immediately; the mock ends the batch on the second poll
  );

  assert.equal(r.batchId, 'msgbatch_mock');
  assert.equal(r.status, 'completed');
  assert.deepEqual(
    r.items.map((i) => i.customId),
    ['first', 'second'],
    'results should line up with the requests'
  );
  assert.equal(r.items[0].result, 'echo:one', 'each result should carry its own prompt');
  assert.equal(r.items[1].result, 'echo:two');
  assert.equal(r.items[0].error, null);
});

test('Batch rejects a provider that cannot batch', () => {
  const provider = ai.createDeepSeek({ apiKey: 'test-key', baseUrl: mock.baseUrl });
  assert.throws(
    () => new ai.Batch(provider, 'deepseek-chat'),
    /does not support batching/,
    'an unsupported provider should fail at construction, not at run'
  );
});

// --- MCP ------------------------------------------------------------------

test('mcpToolsetFromServer() connects over stdio and runs the server tool', async () => {
  await mock.reset();
  const r = await runScenario('mcp-add-numbers', {
    env: { MOCK_BASE_URL: mock.baseUrl, MCP_SERVER },
    timeoutMs: 20000, // spawning a child process costs more than the default
  });

  assert.equal(r.text, 'pong');
  assert.match(
    await toolResultContent(),
    /sum=5/,
    'the MCP server result should reach the model'
  );
});

test('mcpToolsetFromServer() rejects an unusable config', () => {
  assert.throws(() => ai.mcpToolsetFromServer('{ not json'), /MCP config/i);
});

test('mcpToolsetFromServer() fails fast when the server cannot be spawned', async () => {
  const r = await runScenario('mcp-bad-command', {
    env: { MOCK_BASE_URL: mock.baseUrl },
    timeoutMs: 20000,
  });

  assert.equal(r.threw, true, 'an unspawnable server should raise, not wait forever');
  // Assert *why* it threw, not just that it did. Without this the test still
  // passes when the config is rejected earlier for an unrelated reason — which
  // is exactly what happened when `name` became required and this config did
  // not have one yet.
  assert.match(
    r.message,
    /spawn/i,
    'the failure should come from spawning the process, not from validating the config'
  );
});
