// Runs ONE binding operation against a real provider, in its own process, so
// the parent can apply a wall-clock budget. The hermetic suite covers behaviour
// with a mock; this covers the part a mock cannot — that a real provider's
// wire format is parsed into the shape the binding promises.
//
// Reads LIVE_SCENARIO, DEEPSEEK_API_KEY (and optionally LIVE_MODEL /
// LIVE_BASE_URL) from the environment. The key is only ever read here and
// handed to the provider — never written anywhere.
//
// Prints a single line to stdout: `OK <json>` or `ERR <message>`.
// Exits 0 on success, 1 on error.
//
// Scenarios report *facts* (counts, flags, the strings involved); the parent
// decides what those facts have to satisfy. Assertions about model wording
// would be flaky, so there are none.

const crypto = require('node:crypto');
const ai = require('../../dist/index.js');

const SCENARIO = process.env.LIVE_SCENARIO;
const API_KEY = process.env.DEEPSEEK_API_KEY;

if (!API_KEY) {
  process.stdout.write('ERR DEEPSEEK_API_KEY is not set\n');
  process.exit(1);
}

const MODEL = process.env.LIVE_MODEL || 'deepseek-chat';
const provider = ai.createDeepSeek(
  process.env.LIVE_BASE_URL ? { apiKey: API_KEY, baseUrl: process.env.LIVE_BASE_URL } : { apiKey: API_KEY }
);

const token = () => 'TOKEN-' + crypto.randomBytes(4).toString('hex');

/** A tool whose result the model cannot guess: it has to actually call it. */
function lookupTool(secret) {
  return ai.tool(
    'lookup_release_token',
    {
      type: 'object',
      properties: { key: { type: 'string', description: 'The key to look up.' } },
      required: ['key'],
      additionalProperties: false,
    },
    'Look up the release token for a key.',
    (input) => ({ key: input.key, token: secret })
  );
}

async function run() {
  switch (SCENARIO) {
    case 'generate': {
      const r = await ai.generateText({ model: provider(MODEL), prompt: 'Say hello in exactly three words.' });
      return {
        text: r.text,
        finishReason: r.finishReason,
        inputTokens: r.usage.inputTokens,
        outputTokens: r.usage.outputTokens,
        steps: r.steps,
      };
    }

    case 'stream': {
      const deltas = [];
      const types = [];
      let finishUsage = null;
      let finishReason = null;

      for await (const ev of ai.streamText({ model: provider(MODEL), prompt: 'Say hello in exactly five words.' })) {
        types.push(ev.type);
        if (ev.type === 'text_delta') deltas.push(ev.text);
        if (ev.type === 'finish') {
          finishUsage = ev.usage || null;
          // The reason rides along inside `usage` rather than as a top-level
          // field — see the finish handling in index.ts.
          finishReason = (ev.usage && ev.usage.finishReason) || null;
        }
      }

      return {
        types,
        deltaCount: deltas.length,
        // Every delta must be a string — a `undefined` here is the bug that
        // used to render as the literal word "undefined" downstream.
        nonStringDeltas: deltas.filter((d) => typeof d !== 'string').length,
        text: deltas.join(''),
        finishUsage,
        finishReason,
      };
    }

    case 'tool-call': {
      const secret = token();
      const r = await ai.generateText({
        model: provider(MODEL),
        tools: [lookupTool(secret)],
        prompt:
          "Call the lookup_release_token tool with key 'release'. " +
          'Then reply with the token value it returned, exactly as given, and nothing else.',
        maxSteps: 3,
      });

      return { text: r.text, steps: r.steps, secret, containsSecret: r.text.includes(secret) };
    }

    case 'stream-tool': {
      const secret = token();
      const types = [];
      const deltas = [];

      for await (const ev of ai.streamText({
        model: provider(MODEL),
        tools: [lookupTool(secret)],
        prompt:
          "Call the lookup_release_token tool with key 'release'. " +
          'Then reply with the token value it returned, exactly as given, and nothing else.',
        maxSteps: 3,
      })) {
        types.push(ev.type);
        if (ev.type === 'text_delta') deltas.push(ev.text);
      }

      const text = deltas.join('');
      return { types, text, secret, containsSecret: text.includes(secret) };
    }

    case 'session-two-turns': {
      const codeword = 'CODENAME-' + crypto.randomBytes(3).toString('hex');
      const agent = new ai.Agent({
        model: provider(MODEL),
        tools: [],
        instructions: 'Answer in as few words as possible.',
        maxSteps: 3,
      });
      const session = new ai.Session(agent);

      const first = await session.send(`Remember this codeword: ${codeword}. Reply with just OK.`);
      const second = await session.send('What was the codeword? Reply with just the codeword.');

      return {
        firstText: first.text,
        secondText: second.text,
        codeword,
        // Proves the session carried history into the second request.
        recalled: second.text.includes(codeword),
      };
    }

    default:
      throw new Error(`unknown scenario: ${SCENARIO}`);
  }
}

run()
  .then((result) => {
    process.stdout.write(`OK ${JSON.stringify(result)}\n`);
    process.exit(0);
  })
  .catch((err) => {
    process.stdout.write(`ERR ${err && err.message ? err.message : String(err)}\n`);
    process.exit(1);
  });
