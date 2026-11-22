# Product scope and acceptance record

## Discovery (2026-11-21)

Read the complete original README, `F26-04.pdf` (4 pages),
`reports/phase4_report.pdf` (4 pages), the Phase 4 Markdown report, notebook
objectives/download cells, and all `src/` implementations. Both PDFs were
successfully extracted in full using `pdftotext -layout <file> -`.
No repository agent instructions or tests existed at task start.

The Phase 3 PDF describes the older independent finder/checker and its known
shortcuts. The README and Phase 4 report supersede that implementation with
length-controlled negatives, comparison features, hybrid BM25 retrieval,
top-two joined sentences, and strict abstention. Neither source promises an
LLM, external paper lookup, neural weights, or a deployment-ready medical
verification service. Phase 4 reports only 0.387 strict accuracy, 0.301 macro
F1 and 22% verdict coverage at the strict operating point; judgments must not
be presented as facts.

Existing dependencies: Python/pandas/NumPy/scikit-learn/SciPy/matplotlib.
Processed CSVs are supplied; original sentence boundaries and training code
require raw SciFact, which is absent from the checkout. The notebook's public
SciFact S3 download returned HTTP 200 during discovery. No credentials or paid
APIs are needed for the existing models. There are no saved trained weights.
The supplied results select BM25 k1=0.9, b=0.5, hybrid lambda=2.0, k=2,
strict threshold=0.76. These are retained, not retuned on dev.

## Bounded implementation checklist

- [ ] Reproducible, integrity-checked raw SciFact setup without notebook execution;
      explicit offline/missing/corrupt data failures, no synthetic data fallback.
- [ ] Reuse the actual train-only comparison-feature verifier and hybrid BM25
      retriever in an interactive local application (not precomputed predictions).
- [ ] Claim -> ranked real sentences -> top-two evidence -> three-way model
      scores -> strict abstention. Zero lexical matches must not produce a verdict.
- [ ] Optional restriction to a SciFact document ID for checking a cited paper;
      no invented DOI, URL, bibliographic metadata, or unsupported external lookup.
- [ ] Display exact sentence text, original zero-based sentence ID, SciFact
      document ID, paper title, retrieval score and which sentences the verifier
      used. Related text is not automatically supporting evidence.
- [ ] Usable responsive keyboard-accessible UI: instructions, examples, input
      validation, startup/loading/empty/error/retry states; clear model limitations
      and no implication that NO_EVIDENCE means a claim is false.
- [ ] Behavioral tests with independent oracles for retrieval/provenance,
      thresholding, API validation, real training/inference and failure paths.
- [ ] Automated checks and bounded real browser checks via chrome-devtools-axi;
      record observed results separately from old research metrics.
- [ ] Local setup/configuration/test documentation, capabilities/limitations and
      future hosting prerequisites. No deployment or infrastructure changes.

Non-goals: internet-wide science search, PDF uploads/OCR, neural model research,
new accuracy claims, accounts, billing, production hosting. This is a complete
local workflow built on a limited research model, not a scientifically reliable
final verification authority. No material product decision is required to
connect the documented existing workflow; unavailable data or credentials would
be escalated rather than replaced with mocks.

## Acceptance evidence

Validation so far (Python 3.12.16; no mocked classifier in integration tests):

- `python scripts/setup_data.py --archive .tools/scifact_data.tar.gz` succeeded
  with archive SHA-256 `11c621288d41ac144d29b13b0f8503b3820b7d6e8b1f6ff24dff335c196d76be`.
- `MPLBACKEND=Agg MPLCONFIGDIR="$PWD/.cache/matplotlib" OPENBLAS_NUM_THREADS=1
  OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q`: **42 passed in 5.52s**,
  no skips. Four real-model test cases cover official dev claims 1, 53, 248,
  249, absent vocabulary and restricted retrieval. Fourteen dependency
  deprecation warnings originate in matplotlib 3.10.3 / pyparsing 3.3.3.
- `.venv/bin/python -m ruff check .`: **All checks passed!**
- `node --check src/static/app.js`, `node --check scripts/browser_checks.js`,
  `.venv/bin/python -m compileall -q src scripts tests` and `git diff --check`
  succeeded.
- Real local server on loopback port 8765 returned `/api/status` state `ready`,
  5,183 documents, 45,952 sentences, 1,312 training rows; scikit-learn 1.6.1.
  Corpus digest `b8d6c89624cb2ed74dee8938effc4f5d8bd2086887880af8110d64be4ceade62`;
  training digest `646847fe52d60a93b1e7e09df020b6471c6c07a41288e84700951ebd444d0679`.

**Browser verification incomplete / tooling blocker:** a named isolated
`chrome-devtools-axi` session opened `about:blank` using the installed Brave
binary. Running `scripts/browser_checks.js` then failed with MCP request timeout;
following the tool's diagnostic suggestion reported `BRIDGE_NOT_READY` and a
CDP target that had gone away. No UI-flow success is claimed. Browser/session
cleanup was requested. Browser behavior, responsive layout, accessibility and
export remain to be verified before completion. Historical Phase 4 scores are
context, not a new accuracy evaluation. No deployment or PR has been made.
