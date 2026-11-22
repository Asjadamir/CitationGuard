# CitationGuard — Phase 5: Final Output and Analysis

**Team F26-04:** Muhammad Asjad (24L-2570, Lead) · Hamza Sheikh (24L-2500) · Haider Ali (24L-2522)
**Dataset:** SciFact (5,183 abstracts, 45,952 sentences; 300 dev claims) · **Code:** `src/product.py`, `src/app.py`, `src/phase5_analysis.py`, `scripts/browser_e2e.py`

## 1. Final system

CitationGuard checks whether a scientific claim is supported by real published evidence. The final output is a **working local web application** built on the evaluated Phase 3/4 models, plus this analysis of how well it works and why it fails.

1. **Retrieve:** hybrid BM25 (sentence score + λ·abstract score; k1 = 0.9, b = 0.5, λ = 2.0, all chosen on train) ranks all 45,952 SciFact sentences. The user can optionally restrict the search to one cited SciFact paper.
2. **Verify:** the top-2 sentences are joined and scored by the comparison-feature logistic regression (separate claim/evidence TF-IDF + negation, direction and number-mismatch features), trained on **train only** at startup.
3. **Abstain:** a verdict (SUPPORT / CONTRADICT) is issued only if the evidence score 1 − P(NO_EVIDENCE) ≥ **τ = 0.76**, the threshold fixed on train in Phase 4 to keep hallucinated verdicts ≤ 10%. If no sentence shares a word with the claim, the classifier is not called.
4. **Show provenance:** every result lists the exact sentences, paper titles, SciFact document and sentence IDs, retrieval scores, class scores, and model/data SHA-256 digests. It can be downloaded as a JSON report.

The web UI is shown in Figure 1. It runs on a CPU, needs no API key, and works offline after a one-time, checksum-verified data download.

## 2. Experimental results

### 2.1 The product reproduces the research pipeline exactly

`src/phase5_analysis.py` runs the **product's own inference path** (`ClaimVerifier.verify`, the function the web app calls) on all 300 dev claims and compares it with Phase 4's research pipeline.

| Metric (dev, 300 claims, strict τ = 0.76) | Phase 4 research script | Shipped product |
|---|---|---|
| Accuracy | 0.407 | 0.407 |
| Macro F1 | 0.301 | 0.301 |
| Strict (grounded) accuracy | 0.387 | 0.387 |
| Hallucinated-verdict rate ↓ | 0.134 | 0.134 |
| Coverage (claims given a verdict) | 0.22 | 0.22 |
| Per-claim label agreement | — | **300 / 300** |

So the evaluation reported in Phase 4 is also the evaluation of the delivered product. Section 5.1 reports what happens when the Phase 4 *selection* procedure is re-run in a different environment.

### 2.2 New experiment: "cited-paper" mode

In the app, a user can restrict the search to the paper a claim cites. Phase 4 never evaluated this mode. Here, each dev claim is searched only within its first cited SciFact document.

| Search scope | Top-2 has gold rationale | Top-5 has gold | Strict acc. | Hallucinated ↓ | Coverage |
|---|---|---|---|---|---|
| All 5,183 papers | 0.585 | 0.723 | 0.387 | 0.134 | 0.22 |
| Cited paper only | **0.713** | **0.872** | 0.397 | 0.125 | 0.20 |

Knowing the cited paper raises top-2 evidence recall by 13 points. However, strict accuracy barely moves (+0.010; 9 vs 6 discordant claims, McNemar p = 0.61), because the verifier, not retrieval, then becomes the bottleneck. 30 of 300 labels change between the two modes.

### 2.3 Speed

On a laptop CPU (single BLAS thread), startup (indexing + training) takes **21–28 s**. A check takes **41–71 ms** on average (p95 ≤ 105 ms) across two full dev runs, so the system responds interactively. `results/phase5_analysis.json` stores the latest timings.

## 3. Error analysis

Figure 2 (left) traces every dev claim to its outcome under the strict threshold.

| Outcome | Claims | Meaning |
|---|---|---|
| NEI claim, correct abstention | 97 | No evidence exists, and the system said so. |
| NEI claim, verdict issued ("hallucinated") | 15 | See *annotation gap* below. |
| Gold evidence in top-2 → correct verdict | 19 | Correct and grounded. |
| Gold evidence in top-2 → **abstained** | **74** | Largest loss: right evidence found, but the verifier's score stays below τ. |
| Gold evidence in top-2 → wrong direction | 17 | SUPPORT ↔ CONTRADICT confusion. |
| Gold evidence not retrieved → abstained / wrong / correct-but-ungrounded | 63 / 9 / 6 | Retrieval misses. |

**Main findings**

- **Over-abstention is the biggest loss, larger than retrieval misses** (74 vs 63 claims). Many correct rationales paraphrase the claim, so lexical overlap stays low and the evidence score stays under τ = 0.76. Example: dev 49, *"ADAR1 binds to Dicer to cleave pre-miRNA"*, scores 0.31 even with its gold sentence retrieved.
- **Issued verdicts are often wrong.** Only 19 of 66 verdicts (29%) are correct *and* cite gold evidence. A verdict from this model must be read as "worth checking", which is how the UI words it ("Potential support").
- **Direction errors come from lexical matching, not reasoning.** Dev 51, *"ALDH1 expression is associated with **better** breast cancer outcomes"* (gold CONTRADICT), gets SUPPORT with a *higher* score (0.97) than its true counterpart dev 53 (*"…poorer prognosis…"*, 0.85). The second retrieved sentence is about a *different* gene (*"PD-L1 expression correlates with better clinical outcomes in breast cancer"*). It shares the words *better, outcomes, breast cancer* with the claim, and the joined top-2 input lets this unrelated paper outweigh the gold sentence that says *poor prognosis*. Similarly, *"Birth-weight is positively associated with breast cancer"* (dev 179, SUPPORT) is called CONTRADICT, and *"Activation of PPM1D suppresses p53"* (dev 70) is called CONTRADICT: the model cannot track which entity a comparative or causal verb applies to.
- **Annotation gap: some "hallucinations" are correct.** The highest-scoring "hallucinated" verdicts are *"Statins decrease blood cholesterol"* → SUPPORT and *"Statins increase blood cholesterol"* → CONTRADICT. Both cite *"Statins are potent inhibitors of cholesterol biosynthesis."* SciFact labels these claims NEI because no rationale was annotated for them, not because the corpus lacks evidence. The 13.4% hallucination rate is therefore an **upper bound**.
- **Long claims:** strict accuracy is 0.395 for claims of ≤ 15 words and 0.358 for longer ones, while top-2 retrieval recall is identical (0.585). The drop is in verification: multi-condition claims (population + intervention + outcome) cannot be checked part by part with bag-of-words features.

## 4. Theoretical insights

**(1) Retrieval sets a hard ceiling.** A claim can only be strictly correct if it is NEI or its gold rationale is among the 2 verified sentences:
strict accuracy ≤ P(NEI) + P(has evidence) · Recall@2 = 0.373 + 0.627 × 0.585 = **0.740**.
The system reaches 0.387, about half of this ceiling. Errors compound multiplicatively across stages: perfecting either stage alone cannot exceed what the other allows. Cited-paper mode (§2.2) raises the ceiling to 0.82 but not the result, which confirms that the verifier is now the binding constraint.

**(2) Abstention controls hallucination, not precision.** Figure 2 (middle) is the selective-prediction risk–coverage curve (risk = share of verdicts that are not correct and grounded). The strict threshold cuts hallucinated verdicts from 24% (argmax, no abstention) to 13%, but risk stays at about 0.7–0.75 at every coverage above 15% (AURC = 0.76). The evidence score ranks *"some relevant evidence was used"* reasonably well, but it carries no information about whether the *direction* (support vs contradict) is right. Abstention is therefore the right tool for the safety goal (don't assert without evidence), but it cannot fix direction errors. Those need a better verifier.

**(3) The scores are not probabilities.** Figure 2 (right) is a reliability diagram of the evidence score against the observed rate of gold evidence being used. Expected calibration error is **0.15**. The model is under-confident at low scores and over-confident above 0.8 (0.85–0.94 → only 52–59% gold evidence). This is why the UI labels scores "uncalibrated" and does not present them as the chance that a claim is true. Temperature/isotonic calibration on train out-of-fold predictions is a cheap future fix.

**(4) Why a linear lexical model cannot do direction well.** SUPPORT and CONTRADICT rationales share almost the same words. They differ in polarity, comparators and *which* entity a verb attaches to ("ALDH1 → poor prognosis" vs "PD-L1 → better outcomes"). A bag-of-words classifier is invariant to that binding, so it sees "better … outcomes … breast cancer" in both cases. Our hand-built mismatch features (negation, increase/decrease, numbers) recover part of this (Phase 4: CONTRADICT F1 0.10 → 0.29), but compositional meaning requires a model that encodes structure, such as a cross-encoder NLI model. This matches the Phase 4 bias finding that epidemiological claims, which depend on effect directions and confidence intervals, are the weakest topic (accuracy 0.31).

## 5. Demonstrating functionality and reproducibility

| Check | Command | Result (Windows 11, Python 3.12.11, CPU) |
|---|---|---|
| Data install (SHA-256 pinned) | `python scripts/setup_data.py` | Verified `11c62128…6d76be`; corpus digest re-checked at startup |
| Unit, API and real-model tests | `python -m pytest -q` | **44 passed**, 0 skipped |
| Lint / JS syntax | `ruff check .` · `node --check src/static/app.js` | Clean |
| Real-browser end-to-end | `python scripts/browser_e2e.py --url http://127.0.0.1:8765 --channel chrome` | **18 / 18 checks passed** |
| Product vs research | `python src/phase5_analysis.py` | Identical dev labels (300/300), ~1 min |
| Phase 4 re-run | `python src/phase4_evaluation.py` | Final model and dev predictions identical; CV-selected τ differs (see 5.1), 7.5 min |
| CI | GitHub Actions `checks.yml` | Installs data, runs tests + lint on every push |

The browser test drives the real app and model in headless Chrome. Its expected labels come from official SciFact dev annotations, not from the model's own output:

- dev 53 → *Potential support*, citing gold paper 45638119;
- dev 249 → *Potential contradiction*;
- dev 1 (no evidence) → *Insufficient evidence*;
- nonsense words → abstention with no invented sources.

It also checks:

- the error message and hidden stale results for an unknown document ID;
- that a document-restricted search shows only that paper;
- that the JSON export matches the screen;
- the Clear and example buttons;
- that a claim can be submitted using only the keyboard;
- that there is no horizontal scrolling at phone width (375 px).

The screenshots in Figure 1 come from this run. All screenshots are saved in `figures/phase5_app_*.png`.

### 5.1 Reproducibility finding: the threshold-selection procedure depends on the numpy version

Re-running Phase 4 in the pinned Phase 5 environment (numpy 2.2.6, scikit-learn 1.6.1) gives these results:

- **The final model is reproduced exactly:** dev scores match to within 1e-11, and hybrid-retrieval MRR (0.578) is identical.
- **The train-CV choices are not.** `GroupKFold` orders groups of equal size with numpy's unstable `argsort`, so numpy 1.26 and numpy 2 build different cross-validation folds. Only numpy 1.26.4 reproduces the committed majority-class CV score (0.21255; numpy 2 gives 0.21275). The different folds change the out-of-fold predictions, and the strict rule ("lowest τ with train hallucination ≤ 10%") then picks **τ = 0.74 instead of 0.76**. The train hallucination rate at 0.74–0.76 sits at 0.09–0.12, right next to the 10% target.
- **Dev results barely change** at τ = 0.74: strict accuracy 0.390 (vs 0.387), hallucinated verdicts 0.143 (vs 0.134), coverage 0.26 (vs 0.22). Topic clusters (KMeans) and the TF-IDF baseline (MRR 0.447 vs 0.456) also shift. Phase 4 already noted the TF-IDF difference.

The product therefore **fixes τ = 0.76 as a constant** instead of re-tuning it at startup. The committed Phase 4 numbers are kept as the reference, and a future change should sort groups with a stable sort before cross-validation. The general lesson is that an operating point chosen on a flat, noisy region of a validation curve should be reported with its sensitivity, not as a single exact value.

## 6. Limitations and future work

- **Domain and scope:** the system searches only a fixed set of 5,183 biomedical abstracts. It does not search the web, read full papers or use current literature. Clinical claims are the least reliable area.
- **Model:** lexical, uncalibrated, and conservative (22% coverage; 29% of issued verdicts are correct and grounded). It is a triage aid, not a fact-checking authority.
- **Evaluation:** the dev set is small (300 claims, wide CIs), NEI labels are incomplete (§3), and cited-paper mode uses the same τ as all-paper search.

**Next steps, in order of expected gain:**

1. A cross-encoder NLI verifier (e.g. trained on SciFact + MultiNLI) to fix direction errors and over-abstention.
2. Per-sentence rather than joined verification, so one off-topic sentence cannot flip the verdict.
3. Calibrating the evidence score.
4. Dense retrieval to lift the Recall@2 ceiling.
5. Before any public hosting: a production server, rate limits and a privacy policy (see README).
