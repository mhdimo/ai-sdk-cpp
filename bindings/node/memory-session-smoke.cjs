// Proves C++-native memory auto-inject (Path B): seed a memory record, then ask
// the agent through a memory-enabled Session. The MemoryContextStrategy injects
// the record into the turn — the agent knows the answer without it being in the
// prompt. Run: DEEPSEEK_API_KEY=... node memory-session-smoke.cjs
const ai = require("./dist/index.js");
const fs = require("fs");

const MEM_DIR = "/tmp/ai-mem-smoke";
fs.rmSync(MEM_DIR, { recursive: true, force: true });

async function main() {
  // 1. Seed memory (file-backed; the session's store reads the same dir).
  const seed = new ai.MemoryStore(MEM_DIR);
  seed.save("project", "project-name", "The project's name is Zephyr. Always use this name.");

  // 2. Agent + memory-enabled session (C++ owns history + auto-inject + compact).
  const provider = ai.createDeepSeek({ apiKey: process.env.DEEPSEEK_API_KEY });
  const model = provider(process.env.DEEPSEEK_MODEL || "deepseek-v4-flash");
  const agent = new ai.Agent({ model, tools: [], instructions: "Answer briefly using any provided project memory.", maxSteps: 3 });
  const session = new ai.Session(agent, { memoryDir: MEM_DIR, maxContextTokens: 8000 });

  // 3. Ask — the project name is NOT in this prompt; it must come from memory.
  process.stdout.write("answer: ");
  for await (const ev of session.sendStream("What is the project name? Reply with just the name.")) {
    if (ev.type === "text_delta") process.stdout.write(ev.text || "");
    else if (ev.type === "finish") console.log("\n[finish] usage:", ev.usage);
    else if (ev.type === "error") { console.error("\n[error]", ev.text); process.exit(1); }
  }
  console.log("memory auto-inject smoke done");
}
main().catch((e) => { console.error("FAIL:", String(e).split("\n")[0]); process.exit(1); });
