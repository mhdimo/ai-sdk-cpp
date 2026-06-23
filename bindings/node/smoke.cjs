// End-to-end smoke: TS API -> Node addon -> libai_sdk (C++) -> live provider.
// Run: DYLD_LIBRARY_PATH=../../build/bindings/c node smoke.cjs
const ai = require("./dist/index.js");

async function main() {
  console.log("ai-sdk-cpp version:", ai.version());
  // No opts: the C binding now reads ANTHROPIC_BASE_URL + ANTHROPIC_AUTH_TOKEN
  // from the env (e.g. z.ai's Anthropic-compatible endpoint).
  const provider = ai.createAnthropic();
  const model = provider("glm-4.6");

  console.log("\n-- generateText --");
  const r = await ai.generateText({
    model,
    prompt: "Reply with exactly one word: pong",
  });
  console.log("text   :", JSON.stringify(r.text));
  console.log("reason :", r.finishReason);
  console.log("usage  :", r.usage.inputTokens, "in /", r.usage.outputTokens, "out");
  console.log("steps  :", r.steps);
}

main().catch((e) => { console.error("FAIL:", e); process.exit(1); });
