# CitationGuard — Phase 4: Evaluation and Analysis

**Team F26-04:** Muhammad Asjad (24L-2570, Lead) · Hamza Sheikh (24L-2500) · Haider Ali (24L-2522)
**Dataset:** SciFact (5,183 abstracts; 809 train / 300 dev claims) · **Code:** `src/phase4_evaluation.py`, notebook section "Phase 4"

## 1. Evaluation setup

CitationGuard is a retrieval-augmented claim checker. It retrieves evidence sentences for a scientific claim from the SciFact corpus, then labels the claim **SUPPORT**, **CONTRADICT** or **NO_EVIDENCE**. We evaluate three levels:

| Level | Metrics |
|---|---|
| Retrieval (sentence & abstract) | Recall@k, Hit@k, MRR |
| Verifier (given gold evidence) | Accuracy, macro F1, per-class F1, confusion matrix |
| End-to-end pipeline (claim level) | Macro F1, *strict (grounded) accuracy*, *hallucinated-verdict rate*, coverage |

**Protocol.** All hyperparameters and thresholds are chosen on **train** with 5-fold cross-validation grouped by claim, so no claim appears in both folds. The official **dev** split is used only for reporting. The 95% confidence intervals come from 1,000 bootstrap resamples over claims. Paired comparisons use McNemar's exact test. Seed = 42.

## 2. Problems found in Phase 3 and how we fixed them

Evaluating Phase 3 properly exposed four real defects. We fixed each with a targeted change and kept the overall design.

| Problem | Evidence | Fix |
|---|---|---|
| **Length shortcut.** NO_EVIDENCE inputs were whole abstracts (~204 words); SUPPORT/CONTRADICT inputs were 1–2 sentences (~45 words). | Text length alone separates NO_EVIDENCE with **98.5%** dev accuracy. The reported NO_EVIDENCE F1 of 0.90 was inflated. | NO_EVIDENCE = the 2 most claim-similar sentences of the non-evidence abstract (same size as a rationale). |
| **CONTRADICT not learned** (F1 0.105; 11/122 found). Claim and evidence were merged into one bag of words, and the English stop-word list deleted *not, no, never, without, less, more*. | Restoring stop words or adding class weights alone left macro F1 ≤ 0.44. | Separate claim / evidence TF-IDF features, plus "claim words missing from the evidence", plus 8 comparison features (overlap, negation mismatch, increase-vs-decrease mismatch, missing numbers). |
| **Class imbalance** (CONTRADICT = 26% of pairs). | CONTRADICT recall 0.09. | `class_weight="balanced"`; C = 2.0 chosen by train CV. |
| **Non-reproducible retrieval.** `argsort` tie order depends on the numpy version. | Dev MRR 0.447 (numpy 2) vs 0.456 (numpy 1.26). | Stable sort. |

## 3. Results

### 3.1 Retrieval (dev, 188 claims with evidence, 46k candidate sentences)

| Retriever | MRR [95% CI] | R@1 | R@5 | R@10 | Hit@5 |
|---|---|---|---|---|---|
| TF-IDF (Phase 3) | 0.456 [0.393, 0.515] | 0.248 | 0.476 | 0.562 | 0.601 |
| BM25 (k1=0.9, b=0.5, tuned on train) | 0.536 [0.480, 0.595] | 0.285 | 0.538 | 0.644 | 0.676 |
| **Hybrid: BM25 sentence + abstract score** | **0.578 [0.517, 0.634]** | **0.331** | **0.604** | **0.736** | **0.723** |

The hybrid's per-claim gain over Phase 3 is +0.122 reciprocal rank (CI [0.077, 0.174]), so it is significant. At abstract level, BM25 finds a correct paper at rank 1 for 71% of claims and in the top 10 for 94% (TF-IDF: 55% / 86%).

**Representation size.** We swept the TF-IDF vocabulary (vector dimension) from 2k to 123k. MRR rises from 0.26 (2k) to 0.456 (50k), then falls slightly (0.444 at the full 123k). The Phase 3 choice of 50k is close to optimal: smaller vectors lose rare biomedical terms, and larger ones add noisy bigrams.

### 3.2 Verifier comparison (gold evidence, dev = 468 pairs)

| Model | A: original data (macro F1) | **B: length-controlled** macro F1 [95% CI] | B: CONTRADICT F1 | Train CV macro F1 |
|---|---|---|---|---|
| Majority class | 0.211 | 0.211 | 0.00 | 0.213 |
| Claim only (no evidence) | 0.398 | 0.398 [0.345, 0.458] | 0.14 | 0.415 |
| Evidence only | 0.521 | 0.415 | 0.08 | 0.445 |
| Phase 3 original: TF-IDF + LR | 0.522 | 0.414 [0.364, 0.459] | 0.10 | 0.446 |
| TF-IDF + LR (balanced) | 0.500 | 0.405 | 0.20 | 0.461 |
| TF-IDF + LinearSVC | 0.479 | 0.406 | 0.15 | 0.461 |
| TF-IDF + Complement NB | 0.400 | 0.403 | 0.19 | 0.445 |
| **Phase 3 fixed: comparison features + LR** | 0.641 | **0.540 [0.486, 0.597]** | **0.29** | **0.615 ± 0.013** |

Fair data (protocol B) changes the picture:
- The original Phase 3 verifier drops from 0.52 to 0.41 macro F1, the same as a model that **never sees the evidence**.
- Every bag-of-words model sits around 0.40–0.42.
- Only the comparison-feature model separates from the rest, and it is also the best in train CV (it was selected there).
- McNemar on dev against the original model: 83 vs 34 discordant pairs, **p = 7 × 10⁻⁶**.

Confusion matrix of the fixed model (rows = true S/C/NE): `[[107, 84, 25], [80, 36, 6], [13, 6, 111]]`. NO_EVIDENCE recall is 0.85. SUPPORT↔CONTRADICT remains the main confusion.

### 3.3 End-to-end pipeline vs. no retrieval (RAG baseline comparison; 300 dev claims)

Pipeline: hybrid retriever → top-2 sentences → verifier. We chose top-2 joined over max/mean of top-5 or top-1 by train out-of-fold F1 (0.451 vs 0.294 / 0.361 / 0.396). The "no retrieval" baseline labels the claim from its wording alone. This is our analogue of generation without retrieval.

| System | Macro F1 | Strict accuracy | Hallucinated verdicts ↓ | Coverage |
|---|---|---|---|---|
| No retrieval (claim only) | **0.410** | 0.220 | 0.411 | 0.66 |
| Retrieval, no abstention | 0.386 | **0.387** | 0.241 | 0.41 |
| Retrieval + abstention, F1-tuned (τ = 0.44) | 0.400 | 0.357 | 0.420 | 0.57 |
| **Retrieval + strict abstention (τ = 0.76)** | 0.301 | **0.387** | **0.134** | 0.22 |
| Oracle evidence (upper bound) | 0.468 | 0.523 | 0.170 | 0.45 |

*Hallucinated verdict rate* = share of claims with no evidence in the corpus that still receive SUPPORT/CONTRADICT. *Strict accuracy* = correct label **and** a gold rationale sentence among the cited sentences (FEVER-style).

Retrieval barely changes plain macro F1, but it changes **whether answers are grounded**. Without retrieval, 41% of evidence-free claims get a verdict and none cites evidence. With retrieval, strict accuracy rises from 0.22 to 0.39 and hallucinated verdicts fall to 24%. The claim-only score of 0.41 is not real verification: it exploits wording patterns in SciFact claims.

**Anti-hallucination control.** Our counterpart of a strict system prompt is an abstention rule: answer NO_EVIDENCE unless P(evidence) ≥ τ. We fix τ on train as the lowest threshold that keeps train hallucinations ≤ 10%. On dev this gives 13.4% hallucinated verdicts, below the oracle's 17%, with no loss in strict accuracy. The cost is coverage (22%) and macro F1, as the trade-off curve in `figures/phase4_pipeline_comparison.png` shows. Tuning τ for F1 instead would *raise* hallucination to 42%. So the operating point should be chosen for the application, not for F1.

The gap to the oracle (0.39 vs 0.52 strict accuracy) comes mainly from retrieval: the top-2 sentences contain a gold rationale for only 59% of claims that have evidence.

## 4. Error analysis

Fixed verifier, dev:

- **Claim length:** macro F1 is 0.57 for claims of ≤ 15 words and 0.37 for claims of > 20 words. Long claims combine several conditions (population + intervention + outcome), and bag-of-words features cannot check each one.
- **Lexical overlap:** F1 is 0.70 when the claim and evidence share > 70% of their words, and about 0.52 below that. Paraphrased evidence is hard.
- **Numbers:** accuracy is 0.62 for claims with numbers vs 0.51 without, because a missing or mismatched number is an easy signal. Statistical evidence is a top error source, though. Example: *"ADHD medications do not increase cardiovascular risk"* (SUPPORT) with evidence *"RR 0.83 (95% CI 0.72–0.96)"* was predicted CONTRADICT with 0.98 confidence. Reading a CI needs numeric reasoning the model lacks.
- **Comparatives and direction:** *"Incidence of heart failure decreased by 10% in women"* and *"DOACs have lower mortality than warfarin"* were both predicted CONTRADICT. The evidence states higher/lower for a *different* comparison group.
- **Interpretability:** the strongest CONTRADICT features are claim words absent from the evidence: *decreases, worsens, ineffective, negatively, lower, accelerates*. This shows the model learned direction mismatches, as intended. One residual issue: evidence length is still a top NO_EVIDENCE feature, because gold rationales vary in length (1–5 sentences) while NO_EVIDENCE is fixed at 2.

## 5. Bias / fairness check

SciFact has no demographic attributes, so we checked for **topic bias** and **shortcut bias**:

- **Topic subgroups** (6 KMeans clusters on LSA-reduced claim TF-IDF, fit on train): accuracy ranges from **0.31** for *"associated, risk, cancer"* (n = 120) to 0.55–0.75 for the molecular-biology clusters, a 0.44 gap. Epidemiological claims hinge on effect sizes and confidence intervals, which our features do not model. A deployed system would therefore be least reliable in clinical / public-health claims, where errors are most costly.
- **Negation shortcut:** in SciFact, claims containing a negation are CONTRADICT 40% of the time (26% overall). The model predicts CONTRADICT for 43% of them, which tracks the true rate rather than over-relying on the word. Accuracy on negated claims (0.51) is close to the rest (0.55).
- **Claim-wording artefacts:** a claim-only model reaches 0.40–0.41 macro F1, well above majority (0.21). Part of any score on SciFact therefore comes from how annotators wrote claims, and we report this baseline next to every result.

## 6. Key insights

1. **Fair evaluation matters more than a new model.** One data-construction artefact made the Phase 3 verifier look better than a claim-only model; it was not.
2. Explicit claim-vs-evidence comparison is what lets a linear model detect contradiction. Representation choices (stop words, separate fields) matter more than the classifier: LR, SVM and NB all score about 0.40.
3. Retrieval is the bottleneck of the full system. Better retrieval (BM25 + abstract score) gives +27% MRR, and the oracle gap is still large.
4. Abstention gives a controllable trade-off between hallucination and coverage. It must be tuned for hallucination, not F1.

## 7. Challenges

- **Detecting our own leakage and shortcuts** required diagnostic baselines (length-only, claim-only), not just higher scores.
- **Avoiding dev overfitting:** every choice (BM25 k1/b, hybrid λ, C, aggregation, τ) needed grouped CV on train, and out-of-fold pipeline runs to tune τ.
- **Small dev set** (122 CONTRADICT pairs) gives wide CIs (±0.05 macro F1), so we report intervals and significance tests rather than point estimates.
- **Reproducibility:** tie-breaking in ranking changed results across library versions.
- **Compute:** no GPU, so neural NLI / dense retrievers were out of scope. They are the natural next step for numeric and comparative reasoning.
