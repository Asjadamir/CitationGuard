# CitationGuard

CitationGuard is an AI research project for evidence-based
scientific claim verification and hallucination detection.

## Team

- Muhammad Asjad (24L-2570) — Team Lead
- Hamza Sheikh (24L-2500)
- Haider Ali (24L-2522)

## Current Phase

Phase 4 — Evaluation and analysis (retrieval / verifier comparison,
end-to-end pipeline vs. no retrieval, error analysis, subgroup checks).

Run: `python src/phase3_pipeline.py` then `python src/phase4_evaluation.py`
(from the repo root, with SciFact extracted into `data/raw/scifact/`).

## Phase 3 fixes

Found while evaluating Phase 3; fixed in `src/phase3_pipeline.py`:

1. NO_EVIDENCE inputs were whole abstracts, so text length alone revealed
   the label (98.5% accuracy). They are now the 2 most claim-similar sentences.
2. The verifier merged claim + evidence into one bag of words and removed
   negation words, so CONTRADICT was almost never found (F1 0.10). It now
   uses separate claim/evidence features plus claim–evidence comparison
   features (CONTRADICT F1 0.29, macro F1 0.41 → 0.54 on fair data).
3. No class weighting despite imbalance → `class_weight="balanced"`.
4. Retrieval ranking ties depended on the numpy version → stable sort.

## Dataset

SciFact — Scientific Claim Verification Dataset

Official repository:
https://github.com/allenai/scifact

## Repository Structure

- `F26-04.ipynb` — Phase 2 EDA + Phase 3 models + Phase 4 evaluation
- `F26-04.pdf` — phase report
- `data/processed/` — clean model-ready datasets
- `figures/` — EDA, Phase 3 and Phase 4 plots
- `results/` — Phase 3/4 metrics and predictions
- `src/phase3_pipeline.py` — retrieval + verifier pipeline
- `src/phase4_evaluation.py` — Phase 4 evaluation and analysis
- `reports/phase4_report.md` — Phase 4 report text

## Project Goal

The project investigates whether AI-generated scientific claims are
actually supported by cited scientific evidence.
