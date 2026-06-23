// Verifies reasoning events flow through the Node stream callback, using
// DeepSeek (which returns reasoning_content). Run:
//   node reasoning.cjs
const ai = require("./dist/index.js");

async function main() {
  const provider = ai.createDeepSeek({ apiKey: process.env.DEEPSEEK_API_KEY });
  const model = provider(process.env.DEEPSEEK_MODEL || "deepseek-v4-flash");
  const counts = {};
  process.stdout.write("reasoning: ");
  for await (const ev of ai.streamText({ model, prompt: "What is 7 * 13? Think briefly." })) {
    counts[ev.type] = (counts[ev.type] || 0) + 1;
    if (ev.type === "reasoning_delta") process.stdout.write("\x1b[2m" + (ev.text || "") + "\x1b[0m");
    else if (ev.type === "text_delta") process.stdout.write(ev.text || "");
    else if (ev.type === "finish") process.stdout.write(" [finish]");
  }
  console.log("\nevent counts:", counts);
  console.log(counts.reasoning_delta ? "REASONING EVENTS FLOWED ✓" : "no reasoning events");
}
main().catch((e) => { console.error("FAIL:", String(e).split("\n")[0]); process.exit(1); });
