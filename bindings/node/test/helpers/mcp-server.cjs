// A minimal MCP server over stdio, so the binding's MCP path can be exercised
// end-to-end without a real server binary or a network.
//
// Framing is newline-delimited JSON-RPC (see src/mcp/transport.cpp) — not the
// Content-Length framing some MCP servers use.
//
// Exposes one tool, `add_numbers`, whose result embeds the sum so a test can
// assert that the value actually made it back to the model.

let buffer = '';

function send(message) {
  process.stdout.write(JSON.stringify(message) + '\n');
}

function handle(req) {
  switch (req.method) {
    case 'initialize':
      return send({
        jsonrpc: '2.0',
        id: req.id,
        result: {
          protocolVersion: '2024-11-05',
          capabilities: { tools: {} },
          serverInfo: { name: 'mock-mcp', version: '1.0.0' },
        },
      });

    // Notifications carry no id and must not be answered.
    case 'notifications/initialized':
      return undefined;

    case 'tools/list':
      return send({
        jsonrpc: '2.0',
        id: req.id,
        result: {
          tools: [
            {
              name: 'add_numbers',
              description: 'Add two numbers.',
              inputSchema: {
                type: 'object',
                properties: { a: { type: 'number' }, b: { type: 'number' } },
                required: ['a', 'b'],
              },
            },
          ],
        },
      });

    case 'tools/call': {
      const args = (req.params && req.params.arguments) || {};
      const sum = (args.a || 0) + (args.b || 0);
      return send({
        jsonrpc: '2.0',
        id: req.id,
        result: { content: [{ type: 'text', text: `sum=${sum}` }], isError: false },
      });
    }

    default:
      return send({
        jsonrpc: '2.0',
        id: req.id,
        error: { code: -32601, message: `unknown method ${req.method}` },
      });
  }
}

process.stdin.on('data', (chunk) => {
  buffer += chunk.toString();
  let idx;
  while ((idx = buffer.indexOf('\n')) >= 0) {
    const line = buffer.slice(0, idx).trim();
    buffer = buffer.slice(idx + 1);
    if (!line) continue;
    try {
      handle(JSON.parse(line));
    } catch (e) {
      // A malformed line is the server's problem to report, not to crash on.
      send({ jsonrpc: '2.0', id: null, error: { code: -32700, message: e.message } });
    }
  }
});
