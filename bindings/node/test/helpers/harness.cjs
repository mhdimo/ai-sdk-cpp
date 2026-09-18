// Parent-side test harness: boots the mock provider server in a child process
// and gives tests a baseUrl to point a provider at.
//
// The mock MUST be out-of-process. The binding's synchronous entry points
// (generateText / Agent.call / Session.send) run the C++ io_context on the JS
// main thread, so an in-process server could never answer a request whose
// response the caller is blocking on.

const { spawn } = require('node:child_process');
const path = require('node:path');

const SERVER = path.join(__dirname, 'mock-server.cjs');
const DYLIB_DIR = path.join(__dirname, '..', '..', '..', 'build', 'bindings', 'c');

/**
 * Start the mock server. Resolves once it reports its port.
 * @returns {Promise<{baseUrl: string, requests: Function, reset: Function, stop: Function}>}
 */
function startMock() {
  return new Promise((resolve, reject) => {
    const child = spawn(process.execPath, [SERVER, '0'], {
      stdio: ['ignore', 'pipe', 'pipe'],
      // The addon links @rpath/libai_sdk.dylib; make sure the mock process can
      // resolve it too (harmless on Linux, where rpath already covers it).
      env: { ...process.env, DYLD_LIBRARY_PATH: [DYLIB_DIR, process.env.DYLD_LIBRARY_PATH].filter(Boolean).join(':') },
    });

    let settled = false;
    let stdout = '';
    let stderr = '';

    const timer = setTimeout(() => {
      if (settled) return;
      settled = true;
      child.kill('SIGKILL');
      reject(new Error(`mock server did not report a port in time\nstderr: ${stderr}`));
    }, 10000);

    child.stdout.on('data', (buf) => {
      stdout += buf.toString();
      const m = stdout.match(/MOCK_PORT=(\d+)/);
      if (m && !settled) {
        settled = true;
        clearTimeout(timer);
        const port = Number(m[1]);
        const baseUrl = `http://127.0.0.1:${port}`;

        const call = async (method, route) => {
          const res = await fetch(`${baseUrl}${route}`, { method });
          if (!res.ok) throw new Error(`${route} -> ${res.status}`);
          return res;
        };

        resolve({
          baseUrl,
          child,
          requests: async () => (await call('GET', '/__requests')).json(),
          reset: async () => {
            await call('POST', '/__reset');
          },
          stop: () => {
            child.kill('SIGKILL');
          },
        });
      }
    });

    child.stderr.on('data', (b) => {
      stderr += b.toString();
    });

    child.on('error', (err) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      reject(err);
    });

    child.on('exit', (code) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      reject(new Error(`mock server exited early with code ${code}\nstderr: ${stderr}`));
    });
  });
}

/**
 * Run a script in a separate process with a hard wall-clock budget.
 *
 * Used for scenarios that block the JS event loop: a timer inside the test
 * process would never fire, so the only way to fail fast instead of wedging
 * the whole run is to isolate the call and kill it from out here.
 *
 * @returns {Promise<{completed: boolean, code: number|null, signal: string|null, stdout: string, stderr: string}>}
 */
function runIsolated(scriptPath, { env = {}, timeoutMs = 20000 } = {}) {
  return new Promise((resolve) => {
    const child = spawn(process.execPath, [scriptPath], {
      stdio: ['ignore', 'pipe', 'pipe'],
      env: { ...process.env, ...env },
    });

    let stdout = '';
    let stderr = '';
    let completed = false;
    let killed = false;

    child.stdout.on('data', (b) => (stdout += b.toString()));
    child.stderr.on('data', (b) => (stderr += b.toString()));

    const timer = setTimeout(() => {
      killed = true;
      child.kill('SIGKILL');
    }, timeoutMs);

    child.on('exit', (code, signal) => {
      clearTimeout(timer);
      completed = !killed;
      resolve({ completed, killed, code, signal, stdout, stderr });
    });
  });
}

const RUNNER = path.join(__dirname, 'scenario-runner.cjs');

/**
 * Run one scenario in a child process and return its `OK {...}` payload.
 *
 * Throws a descriptive failure if the child had to be killed: that is the
 * signature of a hang, and the whole point of isolating these calls.
 *
 * @param {string} scenario  name understood by scenario-runner.cjs
 * @param {{env?: object, timeoutMs?: number}} opts
 */
async function runScenario(scenario, { env = {}, timeoutMs = 15000 } = {}) {
  const r = await runIsolated(RUNNER, { env: { SCENARIO: scenario, ...env }, timeoutMs });

  if (!r.completed) {
    throw new Error(
      `scenario "${scenario}" did not finish within ${timeoutMs}ms ` +
        `(killed=${r.killed} signal=${r.signal}) — it hung.\n` +
        `stdout: ${r.stdout.trim() || '<empty>'}\n` +
        `stderr: ${r.stderr.trim() || '<empty>'}`
    );
  }

  const line = r.stdout.split('\n').find((l) => l.startsWith('OK ') || l.startsWith('ERR '));
  if (!line) {
    throw new Error(
      `scenario "${scenario}" produced no result line\nstdout: ${r.stdout}\nstderr: ${r.stderr}`
    );
  }
  if (line.startsWith('ERR ')) {
    throw new Error(`scenario "${scenario}" threw: ${line.slice(4)}\nstderr: ${r.stderr}`);
  }

  return JSON.parse(line.slice(3));
}

module.exports = { startMock, runIsolated, runScenario, DYLIB_DIR, RUNNER };
