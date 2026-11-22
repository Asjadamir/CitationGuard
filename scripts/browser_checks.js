// Part 1: run only after separate open/snapshot calls establish browser health.
// CHROME_DEVTOOLS_AXI_SESSION=<owned-session> chrome-devtools-axi run < scripts/browser_checks.js
function assert(value, message) {
  if (!value) throw new Error(message);
  console.log(`PASS ${message}`);
}
await page.wait("#verify:not([disabled])");
assert(await page.eval(() => document.getElementById("model-status").textContent.includes("5,183")), "real model ready with 5,183 papers");
// Oracle: official SciFact dev claim 53 SUPPORT, document 45638119.
await page.fill("#claim", "ALDH1 expression is associated with poorer prognosis in breast cancer.");
await page.click("#verify");
await page.wait("#results:not([hidden])");
assert(await page.eval(() => document.getElementById("result-heading").textContent === "Potential support"), "official support example renders a tentative verdict");
assert(await page.eval(() => document.querySelector(".evidence-card").textContent.includes("45638119")), "real source document provenance visible");
assert(await page.eval(() => document.querySelectorAll(".evidence-card").length === 5 && document.querySelectorAll("#evidence-list .evidence-card").length === 2), "two used sources shown; three context sources retained");
assert(await page.eval(() => !document.getElementById("additional-evidence").open && !document.getElementById("scores").open), "context and technical details start collapsed");
assert(await page.eval(() => document.activeElement.id === "results"), "result receives keyboard focus");
await page.click("#additional-evidence summary");
assert(await page.eval(() => document.getElementById("additional-evidence").open && document.querySelectorAll("#context-list .evidence-card").length === 3), "additional context is accessible on demand");
await page.click("#scores summary");
assert(await page.eval(() => document.getElementById("score-values").textContent.includes("threshold: 0.76") && document.getElementById("score-values").textContent.includes("Source 5 retrieval score")), "all scores remain available in collapsed details");
