// Runs ONE binding operation in its own process so the parent can apply a
// wall-clock budget. Required for the synchronous entry points: when they hang
// they block the JS event loop, so no in-process timer can ever fire.
//
// Reads SCENARIO and MOCK_BASE_URL from the environment.
// Prints a single line to stdout: `OK <json>` or `ERR <message>`.
// Exits 0 on success, 1 on error.

const ai = require('../../dist/index.js');

const SCENARIO = process.env.SCENARIO;
const BASE = process.env.MOCK_BASE_URL;

const provider = ai.createOpenAI({ apiKey: 'test-key', baseUrl: BASE });
const model = (name) => provider(name);

const SCHEMA = {
  type: 'object',
  properties: { city: { type: 'string' } },
  required: ['city'],
};

/** A tool that returns synchronously. */
const getWeather = () =>
  ai.tool('get_weather', SCHEMA, 'Get the current weather for a city.', (input) => ({
    city: input.city,
    weather: 'sunny',
  }));

/**
 * A tool that does real async work (a timer stands in for any I/O). This only
 * completes if the JS event loop is still turning while the native call is in
 * flight — which is the whole point of running the call on a worker thread.
 */
const getWeatherSlowly = () =>
  ai.tool('get_weather', SCHEMA, 'Get the current weather for a city.', async (input) => {
    await new Promise((r) => setTimeout(r, 50));
    return { city: input.city, weather: 'sunny' };
  });

/** A tool that rejects; the loop should record the failure rather than die. */
const getWeatherBadly = () =>
  ai.tool('get_weather', SCHEMA, 'Get the current weather for a city.', () => {
    throw new Error('weather service is down');
  });

async function run() {
  switch (SCENARIO) {
    case 'generateText-tool': {
      const r = await ai.generateText({
        model: model('mock-tool-call'),
        tools: [getWeather()],
        prompt: 'What is the weather in Paris?',
        maxSteps: 3,
      });
      return { text: r.text, steps: r.steps };
    }

    case 'agent-call-tool': {
      const agent = new ai.Agent({
        model: model('mock-tool-call'),
        tools: [getWeather()],
        instructions: 'Use the tool.',
        maxSteps: 3,
      });
      const r = await agent.call('What is the weather in Paris?');
      return { text: r.text, steps: r.steps };
    }

    case 'session-send-tool': {
      const agent = new ai.Agent({
        model: model('mock-tool-call'),
        tools: [getWeather()],
        instructions: 'Use the tool.',
        maxSteps: 3,
      });
      const session = new ai.Session(agent);
      const r = await session.send('What is the weather in Paris?');
      return { text: r.text, steps: r.steps };
    }

    case 'session-stream-tool': {
      const agent = new ai.Agent({
        model: model('mock-stream-tool'),
        tools: [getWeather()],
        instructions: 'Use the tool.',
        maxSteps: 3,
      });
      const session = new ai.Session(agent);
      const types = [];
      for await (const ev of session.sendStream('What is the weather in Paris?')) {
        types.push(ev.type);
      }
      return { events: types };
    }

    // The same turn, but the provider rejects at request time. A 401 lands
    // before the first stream part, which is the case the session entry point
    // handles differently from streamText. It has to arrive as a terminal
    // error event; anything else means the caller is told the turn succeeded.
    //
    // SESSION_NO_TOOLS drops the tool set. The two shapes take different routes
    // through the C entry point -- with tools the failure surfaces from inside
    // the stream, without them it reaches the outer catch -- so a test that
    // only ever uses tools cannot see the second one.
    case 'session-stream-error': {
      const useTools = process.env.SESSION_NO_TOOLS !== '1';
      const agent = new ai.Agent({
        model: model(process.env.MOCK_ERROR_MODEL || 'mock-401'),
        tools: useTools ? [getWeather()] : [],
        instructions: useTools ? 'Use the tool.' : 'Answer directly.',
        maxSteps: 3,
      });
      const session = new ai.Session(agent);
      const types = [];
      let errorText = null;
      for await (const ev of session.sendStream('What is the weather in Paris?')) {
        types.push(ev.type);
        if (ev.type === 'error') errorText = ev.text;
      }
      return { events: types, errorText };
    }

    case 'generateText-async-tool': {
      const r = await ai.generateText({
        model: model('mock-tool-call'),
        tools: [getWeatherSlowly()],
        prompt: 'What is the weather in Paris?',
        maxSteps: 3,
      });
      return { text: r.text, steps: r.steps };
    }

    case 'generateText-throwing-tool': {
      const r = await ai.generateText({
        model: model('mock-tool-call'),
        tools: [getWeatherBadly()],
        prompt: 'What is the weather in Paris?',
        maxSteps: 3,
      });
      return { text: r.text, steps: r.steps };
    }

    case 'session-send-twice': {
      const agent = new ai.Agent({
        model: model('mock-tool-call'),
        tools: [getWeather()],
        instructions: 'Use the tool.',
        maxSteps: 3,
      });
      const session = new ai.Session(agent);
      const first = await session.send('What is the weather in Paris?');
      const second = await session.send('And again?');
      return { first: first.text, second: second.text, turns: 2 };
    }

    // --- Standard toolkit + permissions ---------------------------------
    //
    // The prompt is a path, and `mock-read-file` asks the model to read it
    // back — so the assertion is that the file's *contents* reach the model.

    case 'standard-toolkit-read-file': {
      const agent = new ai.Agent({
        model: model('mock-read-file'),
        tools: [],
        extraToolSets: [ai.standardToolkit()],
        maxSteps: 3,
      });
      const r = await agent.call(process.env.TARGET_FILE);
      return { text: r.text, steps: r.steps };
    }

    case 'permissions-allow': {
      const seen = [];
      const gated = ai.withPermissions(ai.standardToolkit(), (tool, inputJson) => {
        seen.push(tool);
        return 0; // AI_PERMISSION_ALLOW
      });
      const agent = new ai.Agent({
        model: model('mock-read-file'),
        tools: [],
        extraToolSets: [gated],
        maxSteps: 3,
      });
      const r = await agent.call(process.env.TARGET_FILE);
      return { text: r.text, steps: r.steps, policyCalls: seen };
    }

    case 'permissions-deny-reason': {
      // The object form: a refusal plus the sentence the model gets to read.
      const seen = [];
      const gated = ai.withPermissions(ai.standardToolkit(), (tool) => {
        seen.push(tool);
        return {
          decision: ai.PermissionDecision.Deny,
          reason: 'that path is outside the project root',
        };
      });
      const agent = new ai.Agent({
        model: model('mock-read-file'),
        tools: [],
        extraToolSets: [gated],
        maxSteps: 3,
      });
      const r = await agent.call(process.env.TARGET_FILE);
      return { text: r.text, steps: r.steps, policyCalls: seen };
    }

    case 'permissions-deny': {
      const seen = [];
      const gated = ai.withPermissions(ai.standardToolkit(), (tool) => {
        seen.push(tool);
        return 1; // AI_PERMISSION_DENY
      });
      const agent = new ai.Agent({
        model: model('mock-read-file'),
        tools: [],
        extraToolSets: [gated],
        maxSteps: 3,
      });
      const r = await agent.call(process.env.TARGET_FILE);
      return { text: r.text, steps: r.steps, policyCalls: seen };
    }

    // The interactive approver (third argument to withPermissions). Ask is
    // what an unlisted tool gets, and the approver is where a UI prompt would
    // go — so it is async here on purpose, and only an approver that is
    // *awaited* can answer.
    case 'permissions-ask-allow':
    case 'permissions-ask-deny':
    case 'permissions-ask-deny-reason':
    case 'permissions-ask-always':
    case 'permissions-ask-junk':
    case 'permissions-ask-no-approver':
    case 'permissions-allow-skips-approver': {
      const decides = SCENARIO !== 'permissions-allow-skips-approver';
      const verdict = {
        'permissions-ask-allow': ai.PermissionDecision.Allow,
        'permissions-ask-deny': ai.PermissionDecision.Deny,
        // The object form on the approver side too: a user who clicks "no"
        // often has a reason, and it is worth more to the model than the
        // bare fact of the refusal.
        'permissions-ask-deny-reason': {
          decision: ai.PermissionDecision.Deny,
          reason: 'the user declined this particular call',
        },
        'permissions-ask-always': ai.PermissionDecision.AllowAlways,
        // Not a decision at all: a gate has to read this as a refusal.
        'permissions-ask-junk': 42,
        'permissions-ask-no-approver': ai.PermissionDecision.Allow, // never reached
        'permissions-allow-skips-approver': ai.PermissionDecision.Deny, // never reached
      }[SCENARIO];

      const policyCalls = [];
      const approverCalls = [];
      const gated = ai.withPermissions(
        ai.standardToolkit(),
        (tool) => {
          policyCalls.push(tool);
          return decides ? ai.PermissionDecision.Ask : ai.PermissionDecision.Allow;
        },
        SCENARIO === 'permissions-ask-no-approver'
          ? undefined
          : async (tool, inputJson, rationale) => {
              approverCalls.push({ tool, inputJson, rationale });
              // A prompt is not instant, and the worker thread has to still be
              // parked when this resolves.
              await new Promise((r) => setTimeout(r, 20));
              return verdict;
            }
      );

      const agent = new ai.Agent({
        model: model(SCENARIO === 'permissions-ask-always' ? 'mock-read-file-twice' : 'mock-read-file'),
        tools: [],
        extraToolSets: [gated],
        maxSteps: 5,
      });
      const r = await agent.call(process.env.TARGET_FILE);
      return {
        text: r.text,
        steps: r.steps,
        policyCalls,
        approverCalls,
        rationale: approverCalls.length ? approverCalls[0].rationale : null,
      };
    }

    case 'merge-toolsets': {
      // Merging the standard toolkit into itself must not lose definitions:
      // the read still has to work through the merged set.
      const dest = ai.standardToolkit();
      ai.mergeToolSets(dest, ai.standardToolkit());
      const agent = new ai.Agent({
        model: model('mock-read-file'),
        tools: [],
        extraToolSets: [dest],
        maxSteps: 3,
      });
      const r = await agent.call(process.env.TARGET_FILE);
      return { text: r.text, steps: r.steps };
    }

    case 'session-with-memory': {
      const agent = new ai.Agent({
        model: model('mock-read-file'),
        tools: [],
        extraToolSets: [ai.standardToolkit()],
        maxSteps: 3,
      });
      const session = new ai.Session(agent, {
        memoryDir: process.env.MEMORY_DIR,
        enableCheckpoint: false, // no extra summarizer call to mock
      });
      const r = await session.send(process.env.TARGET_FILE);
      return { text: r.text, steps: r.steps };
    }

    case 'mcp-add-numbers': {
      // `name` is required and qualifies every tool the server exposes as
      // mcp__<name>__<tool>; it is what permission rules key on. The mock model
      // asks for the qualified name for the same reason.
      const toolset = ai.mcpToolsetFromServer(
        JSON.stringify({
          name: 'numbers',
          transport: 'stdio',
          command: process.execPath,
          args: [process.env.MCP_SERVER],
        })
      );
      const agent = new ai.Agent({
        model: model('mock-add-numbers'),
        tools: [],
        extraToolSets: [toolset],
        maxSteps: 3,
      });
      const r = await agent.call('What is 2 + 3?');
      return { text: r.text, steps: r.steps };
    }

    case 'mcp-bad-command': {
      // A server that cannot be spawned must fail fast and loudly — not spin
      // waiting on a pipe that will never open.
      let message = null;
      try {
        ai.mcpToolsetFromServer(
          JSON.stringify({
            name: 'unspawnable',
            transport: 'stdio',
            command: '/nonexistent/definitely-not-a-real-binary-xyz',
          })
        );
      } catch (e) {
        message = e && e.message ? e.message : String(e);
      }
      return { threw: message !== null, message };
    }

    case 'generateText-plain': {
      const r = await ai.generateText({ model: model('mock-stop'), prompt: 'hi' });
      return { text: r.text };
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
