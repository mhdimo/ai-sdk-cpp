// Proves streaming is NON-BLOCKING: a setInterval ticker keeps firing while the
// stream runs. If ticks accrue *during* the stream, the JS event loop was not
// frozen. Run: DEEPSEEK_API_KEY=... node async-smoke.cjs
const ai = require("./dist/index.js");

async function main() {
  const provider = ai.createDeepSeek({ apiKey: process.env.DEEPSEEK_API_KEY });
  const model = provider(process.env.DEEPSEEK_MODEL || "deepseek-v4-flash");

  let ticks = 0;
  const ticker = setInterval(() => { ticks++; }, 100);

  const events = { reasoning_delta: 0, text_delta: 0, finish: 0 };
  const start = Date.now();
  console.log("streaming (reasoning + answer) while a 100ms ticker runs...");
  for await (const ev of ai.streamText({ model, prompt: "What is 7 * 13? Think briefly." })) {
    if (ev.type === "reasoning_delta") events.reasoning_delta++;
    else if (ev.type === "text_delta") events.text_delta++;
    else if (ev.type === "finish") events.finish++;
  }
  clearInterval(ticker);
  const ms = Date.now() - start;

  console.log(`stream took ${ms}ms; ticker fired ${ticks} times during it`);
  console.log("events:", events);
  if (ticks >= 3) {
    console.log("NON-BLOCKING ✓ — event loop stayed alive during the stream");
  } else {
    console.log("BLOCKING ✗ — event loop was frozen during the stream");
    process.exit(1);
  }
}
main().catch((e) => { console.error("FAIL:", String(e).split("\n")[0]); process.exit(1); });
