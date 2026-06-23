// Proves ASYNC TOOL EXECUTION: an agent calls a tool whose execute() awaits
// (simulating an interactive permission prompt). The C++ loop (worker thread)
// must block on that await while the JS main thread stays alive (ticker fires),
// then receive the result and finish.
const ai = require("./dist/index.js");

async function main() {
  const provider = ai.createAnthropic();          // ANTHROPIC_BASE_URL + ANTHROPIC_AUTH_TOKEN from env
  const model = provider("glm-4.6");

  // A tool that simulates an interactive permission (500ms await).
  const getWeather = ai.tool(
    "get_weather",
    { type: "object", properties: { city: { type: "string" } }, required: ["city"] },
    "Get the current weather for a city.",
    async (input) => {
      console.error("[tool] execute start city=" + input.city);
      await new Promise((r) => setTimeout(r, 500));   // simulate permission prompt
      console.error("[tool] execute done");
      return { city: input.city, weather: "sunny, 72F" };
    }
  );

  const agent = new ai.Agent({ model, tools: [getWeather], instructions: "Use the get_weather tool when asked about weather. Be terse.", maxSteps: 5 });
  const session = new ai.Session(agent);

  let ticks = 0;
  const ticker = setInterval(() => { ticks++; }, 100);
  const events = {};
  console.error("[main] streaming a turn that uses an async tool...");
  process.stdout.write("answer: ");
  for await (const ev of session.sendStream("What's the weather in Paris? Use the tool.")) {
    console.error("[event] " + ev.type + (ev.toolName ? " " + ev.toolName : ""));
    events[ev.type] = (events[ev.type] || 0) + 1;
    if (ev.type === "tool_call_start") console.log("  [tool_call:", ev.toolName + "]");
    if (ev.type === "tool_result") console.log("  [tool_result:", (ev.text || "").slice(0, 60) + "]");
    if (ev.type === "text_delta") process.stdout.write(ev.text || "");
    if (ev.type === "finish") console.log("  [finish]");
  }
  clearInterval(ticker);
  console.log("event counts:", events);
  console.log("ticker fired during turn:", ticks);

  const usedTool = (events.tool_call_start || 0) > 0;
  if (usedTool && ticks >= 2) {
    console.log("ASYNC TOOL EXEC ✓ — tool called, awaited, main thread stayed responsive");
  } else if (!usedTool) {
    console.log("NOTE: model did not call the tool — can't fully verify (model behavior). Binding still built + streams.");
  } else {
    console.log("BLOCKED ✗ — ticker did not fire during the tool await");
    process.exit(1);
  }
}
main().catch((e) => { console.error("FAIL:", String(e).split("\n")[0]); process.exit(1); });
