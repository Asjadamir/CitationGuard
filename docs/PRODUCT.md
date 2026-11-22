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

- [x] Reproducible, integrity-checked raw SciFact setup without notebook execution;
      explicit offline/missing/corrupt data failures, no synthetic data fallback.
- [x] Reuse the actual train-only comparison-feature verifier and hybrid BM25
      retriever in an interactive local application (not precomputed predictions).
- [x] Claim -> ranked real sentences -> top-two evidence -> three-way model
      scores -> strict abstention. Zero lexical matches must not produce a verdict.
- [x] Optional restriction to a SciFact document ID for checking a cited paper;
      no invented DOI, URL, bibliographic metadata, or unsupported external lookup.
- [x] Display exact sentence text, original zero-based sentence ID, SciFact
      document ID, paper title, retrieval score and which sentences the verifier
      used. Related text is not automatically supporting evidence.
- [x] Usable responsive keyboard-accessible UI: instructions, examples, input
      validation, startup/loading/empty/error/retry states; clear model limitations
      and no implication that NO_EVIDENCE means a claim is false.
- [x] Behavioral tests with independent oracles for retrieval/provenance,
      thresholding, API validation, real training/inference and failure paths.
- [x] Automated checks and bounded real browser checks (Playwright, scripts/browser_e2e.py);
      record observed results separately from old research metrics.
- [x] Local setup/configuration/test documentation, capabilities/limitations and
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
  training digest `646847fe52d60a93b1e7e09df020b6471c6c07a41288e84700951ebd444d0679`
  (line endings normalized since Phase 5, so Windows checkouts report the same value).

**Earlier browser blocker (superseded):** the first attempt with
`chrome-devtools-axi` timed out (`BRIDGE_NOT_READY`), so the `scripts/browser_*.js`
checks never ran. Phase 5 ports all of them, plus export, keyboard and mobile
checks, to `scripts/browser_e2e.py` (Playwright).

## Phase 5 verification (2026-11-22, Windows 11, CPython 3.12.11, CPU)

- `python scripts/setup_data.py` downloaded and verified the pinned archive.
- `python -m pytest -q`: **44 passed**, 0 skipped (adds line-ending-independent
  training digest tests). `ruff check .` and `node --check src/static/app.js`: clean.
- `python scripts/browser_e2e.py --url http://127.0.0.1:8765 --channel chrome`:
  **18/18 checks passed** in headless Chrome against the real model. Covered:
  official dev claims 53 (support, gold paper 45638119 cited), 249 (contradiction)
  and 1 (abstains); unknown vocabulary (no invented evidence); unknown document
  ID error clears stale results; document-restricted search; JSON download equals
  the displayed result; Clear/example buttons; keyboard-only submission; result
  focus; no horizontal scroll at 375 px. Screenshots: `figures/phase5_app_*.png`.
- `python src/phase5_analysis.py`: the product inference path reproduces the
  Phase 4 strict pipeline on all 300 dev claims (identical labels 300/300;
  macro F1 0.301, strict accuracy 0.387, hallucinated 0.134, coverage 0.22).
  Mean check latency 41 ms (p95 72 ms); startup ~21 s.
- Analysis and report: `reports/phase5_report.{md,pdf}`. No deployment was made.
