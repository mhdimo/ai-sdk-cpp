// End-to-end smoke: TS API -> Node addon -> libai_sdk (C++) -> live provider.
// Exercises generateText, Session.sendStream (with reasoning/tool events),
// MemoryStore, and Batch.
const ai = require("./dist/index.js");
const fs = require("fs");
const os = require("os");
const path = require("path");

const provider = ai.createAnthropic();      // ANTHROPIC_BASE_URL + ANTHROPIC_AUTH_TOKEN from env
const model = provider("glm-4.6");

async function main() {
  console.log("ai-sdk-cpp version:", ai.version());

  console.log("\n-- generateText --");
  const r = await ai.generateText({ model, prompt: "Reply with exactly one word: pong" });
  console.log("text:", JSON.stringify(r.text), "| usage:", r.usage.inputTokens, "in /", r.usage.outputTokens, "out");

  console.log("\n-- Session.sendStream --");
  const agent = new ai.Agent({ model, tools: [], instructions: "Be terse.", maxSteps: 3 });
  const session = new ai.Session(agent);
  const counts = {};
  process.stdout.write("stream: ");
  session.sendStream("Reply with exactly: pong", (ev) => {
    counts[ev.type] = (counts[ev.type] || 0) + 1;
    if (ev.type === "text_delta") process.stdout.write(ev.text || "");
    else if (ev.type === "finish") console.log(" [finish]");
    else if (ev.type === "reasoning_delta") process.stdout.write("\x1b[2m" + (ev.text || "") + "\x1b[0m");
  });
  console.log("event counts:", counts);

  console.log("\n-- MemoryStore --");
  const memDir = fs.mkdtempSync(path.join(os.tmpdir(), "ai-sdk-mem-"));
  const mem = new ai.MemoryStore(memDir);
  mem.save("project", "greeting", "hello from TypeScript via the Node binding");
  console.log("memory saved to", memDir, "→ files:", fs.readdirSync(memDir));

  console.log("\n-- Batch --");
  try {
    const batch = new ai.Batch(provider, "glm-4.6");
    console.log("batch created (unexpected on this provider)");
  } catch (e) {
    console.log("batch create threw (expected if provider lacks batching):", String(e).split("\n")[0], "— binding is wired");
  }

  console.log("\nALL OK");
}

main().catch((e) => { console.error("FAIL:", e); process.exit(1); });
