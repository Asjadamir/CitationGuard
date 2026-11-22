// Part 3: same healthy, task-owned browser session as browser_checks.js.
function assert(value, message) {
  if (!value) throw new Error(message);
  console.log(`PASS ${message}`);
}
await page.click(".scope summary");
await page.fill("#doc-id", "9999999999");
await page.fill("#claim", "Aspirin inhibits platelet aggregation.");
await page.click("#verify");
await page.wait("#form-error:not([hidden])");
assert(await page.eval(() => document.getElementById("form-error").textContent.includes("not in this corpus") && document.getElementById("results").hidden), "unknown document error replaces stale results");
await page.wait("#verify:not([disabled])");
await page.fill("#doc-id", "4983");
await page.fill("#claim", "Microstructural development of newborn cerebral white matter.");
await page.click("#verify");
await page.wait("#results:not([hidden])");
assert(await page.eval(() => document.querySelectorAll(".source-meta").length > 0 && [...document.querySelectorAll(".source-meta")].every(element => element.textContent.includes("document 4983"))), "document-restricted UI shows only the selected source");
await page.wait("#verify:not([disabled])");
await page.click("#clear");
assert(await page.eval(() => document.getElementById("claim").value === "" && document.getElementById("doc-id").value === "" && document.getElementById("results").hidden && document.activeElement.id === "claim"), "clear resets inputs, results and keyboard focus");
await page.click("[data-example]");
assert(await page.eval(() => document.getElementById("claim").value === "Vitamin D deficiency is associated with cardiovascular disease." && document.getElementById("results").hidden), "example fills a claim without inventing a verification");
