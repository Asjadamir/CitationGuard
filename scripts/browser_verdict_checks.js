// Part 2: same healthy, task-owned browser session as browser_checks.js.
function assert(value, message) {
  if (!value) throw new Error(message);
  console.log(`PASS ${message}`);
}
async function checkClaim(claim) {
  await page.wait("#verify:not([disabled])");
  await page.fill("#claim", claim);
  await page.click("#verify");
  await page.wait("#results:not([hidden])");
}
// Oracle: official SciFact dev claim 249 CONTRADICT, document 1568684.
await checkClaim("Chenodeosycholic acid treatment reduces whole-body energy expenditure.");
assert(await page.eval(() => document.getElementById("result-heading").textContent === "Potential contradiction"), "official contradiction example renders");
// Oracle: official dev claim 1 has no gold evidence.
await checkClaim("0-dimensional biomaterials show inductive properties.");
assert(await page.eval(() => document.getElementById("result-heading").textContent === "Insufficient evidence" && document.getElementById("empty-results").hidden && document.querySelectorAll(".evidence-card").length > 0), "related evidence below threshold is not called verified");
await checkClaim("Qzxvplmn qzzxrrt.");
assert(await page.eval(() => document.getElementById("result-heading").textContent === "Insufficient evidence" && !document.getElementById("empty-results").hidden && document.querySelectorAll(".evidence-card").length === 0), "unknown vocabulary abstains with no invented evidence");
