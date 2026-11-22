# CitationGuard

**Evidence before certainty.** A local scientific-claim checker with real
retrieved evidence, an experimental model judgment and traceable sources.

Enter a claim → search SciFact abstracts → join the top two sentences → run
the trained verifier → show **potential support**, **potential contradiction**
or **insufficient evidence** with the exact sources. Optionally restrict the
search to a cited SciFact document ID. Download the complete review as JSON.

> This is a functional local research application, **not a reliable scientific
> or medical verification authority**. Its simple lexical model can be wrong,
> even with high scores. It does not search the web, read full papers, generate
> citations, or use an LLM. “Insufficient evidence” does not mean “false”.

## Run locally

Use **Python 3.12** (the tested version), a CPU and a modern browser. No GPU,
API keys, accounts or paid services are required. A first installation needs
internet access for Python packages and the public 3 MB SciFact archive.

```bash
git clone https://github.com/Asjadamir/CitationGuard.git
cd CitationGuard
python3.12 -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python scripts/setup_data.py
python src/app.py
```

Open **http://127.0.0.1:8000**. Startup builds the BM25 search indexes and trains
the existing logistic-regression verifier on the supplied **train** CSVs only.
The page shows preparation progress; checks are disabled until ready. Allow a
minute on CPU (speed depends on hardware). Stop with Ctrl+C. The app binds only
to loopback, runs without debug mode, and must not be exposed as a public server.

The downloader checks the release's SHA-256 and copies only named JSONL files.
It does not execute the notebook or extract arbitrary archive paths. If S3 is
unavailable, obtain the [official SciFact release](https://github.com/allenai/scifact)
and transfer its unchanged `data.tar.gz`, then run:

```bash
python scripts/setup_data.py --archive /path/to/data.tar.gz
```

The expected archive SHA-256 is
`11c621288d41ac144d29b13b0f8503b3820b7d6e8b1f6ff24dff335c196d76be`.
A changed upstream release fails closed; inspect it before updating this pin.
The corpus is integrity-checked again at startup. Never substitute generated
or re-split sentences: original sentence IDs are part of the evidence contract.

### Configuration (no secrets)

- `python src/app.py --port 8001`: choose another local port.
- `CITATIONGUARD_DATA_DIR=/absolute/path/to/data python src/app.py`: directory
  containing the verified `corpus.jsonl`. Defaults to
  `data/raw/scifact/data/`, relative to the repository, not the working directory.
- `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`: useful on shared or small machines
  to avoid thread oversubscription.
- `MPLBACKEND=Agg MPLCONFIGDIR=.cache/matplotlib`: optional headless plotting
  configuration for research/check commands.

There are no service credentials to configure. After initial setup the checker
works offline. Claims/reports are not persisted by the server and no model API
is called. Downloaded reports are saved by your browser and contain your claim;
handle them accordingly. Server diagnostics go to the terminal, not the UI.

### Troubleshooting

- **Missing or corrupt SciFact:** the page reports a setup error, not a verdict.
  Run the data setup command, then **Retry setup**; no restart is required.
- **Model setup failed:** inspect the terminal and reinstall pinned requirements;
  ensure supplied `data/processed/*_train.csv` files are present. Retry setup.
- **Unknown document:** enter a numeric SciFact ID, not a DOI, URL or PubMed ID,
  or leave the field blank to search all papers.
- **No matches:** there are no positive lexical matches in the selected scope;
  the classifier is not called and no citation/confidence is fabricated.
- **Busy/disconnected:** only one claim is processed at a time. Retry shortly
  or restart your own local app. Request and input sizes are bounded.

## Reading a result

- **Potential support / contradiction** is a tentative judgment on the first two
  retrieved sentences **together**, not a label for each paper or sentence.
- The fixed Phase 4 strict rule requires `1 - P(NO_EVIDENCE) >= 0.76`, then
  chooses between SUPPORT and CONTRADICT. Otherwise it abstains. No-match
  searches always abstain, even if a classifier would have guessed a label.
- Evidence includes the untruncated sentence, paper title, **SciFact** document
  ID, original **zero-based** sentence ID, retrieval score and whether it was
  used. SciFact IDs are dataset identifiers, not verified links to full papers.
- BM25 relevance is not scientific support. Model scores are **uncalibrated**,
  not the probability the claim is true. Top sentences may come from different
  papers. Document-restricted inference is useful for inspection but was not
  separately evaluated when the corpus-wide threshold was selected.
- The fixed corpus contains 5,183 abstracts / 45,952 sentences, not current
  literature. The model struggles with paraphrases, negation, long claims,
  populations/comparators and statistical reasoning. Clinical/public-health
  claims are particularly risky. Always consult original research.

Historical Phase 4 corpus-wide strict results: macro F1 **0.301**, strict
accuracy **0.387**, hallucinated verdict rate **0.134**, coverage **0.22**.
These are supplied research results, **not a new product accuracy claim**.
See [the original evaluation report](reports/phase4_report.md) for definitions,
confidence intervals, baselines and known biases.

## Tests and checks

```bash
python -m pip install -r requirements-dev.txt
python scripts/setup_data.py           # required for the real-model tests
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q
python -m ruff check .
node --check src/static/app.js
```

Unit/API tests use tiny explicit source records and a classifier boundary double
for deterministic failure/threshold tests. Integration tests **train and run the
actual model** against SciFact; expected verdicts come from official dev
annotations for four named smoke examples, not from the implementation's
predictions. These examples are not a fresh accuracy benchmark. Without raw
SciFact the integration tests explicitly skip; that is not full validation.
CI downloads the checksum-pinned data and runs both suites. Browser evidence
and the bounded requirements checklist are recorded in [docs/PRODUCT.md](docs/PRODUCT.md).

## JSON API

```bash
curl http://127.0.0.1:8000/api/status
curl -H 'Content-Type: application/json' \
  -d '{"claim":"ALDH1 expression is associated with poorer prognosis in breast cancer.","doc_id":""}' \
  http://127.0.0.1:8000/api/verify
```

`GET /api/status`: loading/ready/error and model provenance.
`POST /api/retry`: restart failed model setup (never duplicate a running job).
`POST /api/verify`: `claim` (3–2000 characters), optional numeric-string `doc_id`.
Success returns the label/reason, exact evidence, scores (null if no matches),
threshold, model/data digests, timestamp and limitations. Invalid inputs return
400, oversized requests 413, unavailable/busy 503, unexpected inference failure
500 with no verdict. No API route fetches user-supplied URLs.

## Research context and structure

The project goal is to investigate whether scientific claims are actually
supported by cited evidence. The original README's current phase was
**Phase 4 — Evaluation and analysis**. The next phase adds an interactive local
product around that working CPU stack, preserving the notebook and reports.
The older Phase 3 PDF predates the fixes below; Phase 4 and its code supersede it.

- `src/product.py`, `src/app.py`, `src/templates/`, `src/static/`: inference/UI.
- `scripts/setup_data.py`: public dataset installation, no notebook required.
- `src/phase3_pipeline.py`: original retrieval/verifier and training helpers.
- `src/phase4_evaluation.py`: BM25/hybrid classes and research evaluation.
- `F26-04.ipynb`: data exploration, Phase 3 models and Phase 4 evaluation.
- `F26-04.pdf`, `reports/phase4_report.{md,pdf}`: supplied project reports.
- `data/processed/`: supplied clean training/dev CSVs; raw data is untracked.
- `results/`, `figures/`: original research metrics, predictions and plots.
- `tests/`: behavior, HTTP, failure handling and real-model integration tests.

Phase 3 fixes retained: negatives use two claim-similar sentences rather than
whole abstracts; separate claim/evidence/comparison features retain negation;
balanced class weights; stable retrieval tie ordering. The product retains
Phase 4's train-selected BM25 `k1=0.9, b=0.5`, hybrid weight `2.0`, joined top-two
aggregation and strict threshold `0.76`. It does not retune on dev.

To reproduce the old experiments (writes `results/` and `figures/`):

```bash
python -m pip install -r requirements-research.txt  # only needed for notebook tools
python src/phase3_pipeline.py
python src/phase4_evaluation.py
```

## Hosting later — not deployed

No deployment, purchased services or production-infrastructure changes are
included. Before any public hosting: obtain an explicit product/reliability
operating-point decision, review SciFact's upstream data terms, evaluate on the
intended domain, choose CPU/RAM/concurrency budgets with measurements, and add
a production WSGI server, TLS/reverse proxy, access/rate limits, privacy/logging
policy and operational monitoring. Training runs at each process startup and
is not a background hosted service. Neural NLI/dense retrieval would be a
separate evaluated model change, not an API key silently assumed to exist.

## Team and credit

- Muhammad Asjad (24L-2570) — Team Lead
- Hamza Sheikh (24L-2500)
- Haider Ali (24L-2522)

Dataset: [SciFact](https://github.com/allenai/scifact), Wadden et al.,
“Fact or Fiction: Verifying Scientific Claims”, EMNLP 2020. Models use
scikit-learn, NumPy, SciPy and pandas; the local UI uses Flask and plain HTML/CSS/JS.
