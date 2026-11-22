"use strict";

const byId = (id) => document.getElementById(id);
let modelReady = false;
let checking = false;
let latestReport = null;
let pollTimer = null;

function updateControls() {
  byId("verify").disabled = !modelReady || checking;
  byId("verify").textContent = checking ? "Checking evidence…" : "Check evidence →";
  for (const control of document.querySelectorAll("#claim, #doc-id, #clear, [data-example]")) {
    control.disabled = checking;
  }
  byId("claim-form").setAttribute("aria-busy", String(checking));
}

function clearResult() {
  latestReport = null;
  byId("results").hidden = true;
  byId("form-error").hidden = true;
  byId("progress").textContent = "";
  byId("count").textContent = `${byId("claim").value.length} / 2000`;
}

async function readResponse(response) {
  let body;
  try {
    body = await response.json();
  } catch {
    throw new Error("The local server returned an unreadable response. Check the server and retry.");
  }
  if (!response.ok) throw new Error(body.error || "The request failed. Please retry.");
  return body;
}

async function checkStatus() {
  clearTimeout(pollTimer);
  try {
    const status = await readResponse(await fetch("/api/status", { signal: AbortSignal.timeout(10000) }));
    modelReady = status.state === "ready";
    byId("setup-error").hidden = status.state !== "error";
    if (status.state === "error") byId("setup-message").textContent = status.error;
    byId("model-status").textContent = modelReady
      ? `Ready · ${status.model.documents.toLocaleString()} papers`
      : status.state === "error" ? "Setup needs attention" : "Preparing model…";
    if (!modelReady && status.state !== "error") {
      byId("progress").textContent = "Preparing the local model. This may take a minute.";
      pollTimer = setTimeout(checkStatus, 2000);
    } else if (!checking) {
      byId("progress").textContent = "";
    }
  } catch {
    modelReady = false;
    byId("model-status").textContent = "Server unavailable";
    byId("setup-error").hidden = false;
    byId("setup-message").textContent = "Cannot reach the local server. Make sure python src/app.py is running, then Retry setup.";
  }
  updateControls();
}

function addText(parent, tag, text, className = "") {
  const element = document.createElement(tag);
  element.textContent = text;
  if (className) element.className = className;
  parent.append(element);
  return element;
}

function renderReport(report) {
  latestReport = report;
  const labels = { SUPPORT: "Potential support", CONTRADICT: "Potential contradiction", NO_EVIDENCE: "Insufficient evidence" };
  byId("result-heading").textContent = labels[report.label];
  byId("checked-claim").textContent = report.claim;
  byId("result-explanation").textContent = report.reason === "no_matches"
    ? "No matching sentences found. The model did not issue a verdict."
    : report.reason === "below_threshold"
      ? "Related text found, but insufficient evidence for a verdict."
      : "Tentative judgment on the first two sentences together. Review the sources.";
  byId("searched-scope").textContent = report.scope.doc_id === null
    ? "SciFact abstracts · all papers"
    : `SciFact abstracts · document ${report.scope.doc_id} only`;
  const scores = byId("score-values");
  scores.replaceChildren();
  if (report.probabilities) {
    for (const [label, value] of Object.entries(report.probabilities)) {
      addText(scores, "p", `${label}: ${(value * 100).toFixed(1)}%`);
    }
    addText(scores, "p", `Evidence score: ${report.evidence_score.toFixed(3)} · threshold: ${report.threshold.toFixed(2)}`);
  } else {
    addText(scores, "p", "Not scored: no matching evidence.");
  }
  const evidenceList = byId("evidence-list");
  const contextList = byId("context-list");
  evidenceList.replaceChildren();
  contextList.replaceChildren();
  byId("additional-evidence").open = false;
  byId("additional-evidence").hidden = !report.evidence.some((item) => !item.used_by_verifier);
  for (const item of report.evidence) {
    const card = document.createElement("article");
    card.className = "panel evidence-card";
    addText(card, "p", `${item.rank} · ${item.used_by_verifier ? "Used for judgment" : "Context only"}`, "evidence-role");
    addText(card, "h3", item.title);
    addText(card, "p", `SciFact document ${item.doc_id} · sentence ${item.sentence_id} (zero-based)`, "source-meta");
    addText(card, "blockquote", item.text);
    (item.used_by_verifier ? evidenceList : contextList).append(card);
    addText(scores, "p", `Source ${item.rank} retrieval score: ${item.retrieval_score.toFixed(3)}`);
  }
  byId("empty-results").hidden = report.evidence.length !== 0;
  byId("model-details").textContent = `${report.model.name}\n${report.model.version}\nChecked: ${report.checked_at}\nCorpus SHA-256: ${report.model.corpus_sha256}\nTraining SHA-256: ${report.model.training_sha256}\nTrain rows: ${report.model.training_rows}; scikit-learn: ${report.model.sklearn_version}`;
  byId("limitations").replaceChildren();
  for (const limitation of report.limitations) addText(byId("limitations"), "li", limitation);
  byId("results").hidden = false;
  byId("results").focus();
}

byId("claim-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!modelReady || checking) return;
  clearResult();
  checking = true;
  updateControls();
  byId("progress").textContent = "Retrieving sentences and checking the evidence…";
  try {
    const response = await fetch("/api/verify", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ claim: byId("claim").value, doc_id: byId("doc-id").value.trim() }),
      signal: AbortSignal.timeout(60000),
    });
    renderReport(await readResponse(response));
    byId("progress").textContent = "";
  } catch (error) {
    byId("form-error").textContent = error.name === "TimeoutError"
      ? "The check timed out. No verdict was received. Wait a moment and try again."
      : error.name === "TypeError" ? "Cannot reach the local server. Check that it is running and retry." : error.message;
    byId("form-error").hidden = false;
    byId("progress").textContent = "";
  } finally {
    checking = false;
    updateControls();
  }
});

for (const control of document.querySelectorAll("#claim, #doc-id")) control.addEventListener("input", clearResult);
for (const button of document.querySelectorAll("[data-example]")) {
  button.addEventListener("click", () => {
    byId("claim").value = button.dataset.example;
    byId("doc-id").value = "";
    clearResult();
    byId("claim").focus();
  });
}
byId("clear").addEventListener("click", () => {
  byId("claim-form").reset();
  clearResult();
  byId("claim").focus();
});
byId("retry").addEventListener("click", async () => {
  byId("retry").disabled = true;
  try {
    await readResponse(await fetch("/api/retry", { method: "POST", signal: AbortSignal.timeout(10000) }));
    await checkStatus();
  } catch {
    byId("setup-message").textContent = "Cannot retry setup. Start the local server and try again.";
  } finally {
    byId("retry").disabled = false;
  }
});
byId("download").addEventListener("click", () => {
  if (!latestReport) return;
  const url = URL.createObjectURL(new Blob([JSON.stringify(latestReport, null, 2)], { type: "application/json" }));
  const link = document.createElement("a");
  link.href = url;
  link.download = "citationguard-report.json";
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});
checkStatus();
