# Project agent memory

This file is the project's committed home for project-intrinsic agent knowledge: build, test, release, architecture, and sharp-edge notes that should travel with the code.

- Local setup, API contract, checks and hosting exclusions: `README.md`. Accepted scope and actual validation evidence: `docs/PRODUCT.md`.
- Run the app with `python src/app.py`; test imports use `src/` on the Python path (`pyproject.toml`). Research scripts retain their direct-file entry points.
- Raw SciFact is intentionally untracked. Install with `python scripts/setup_data.py`; preserve original sentence boundaries/zero-based IDs. Missing data is an explicit error, never a synthetic fallback.
- Reuse the Phase 3 comparison-feature verifier and Phase 4 hybrid retrieval. Product training must use train CSVs only; supplied historical metrics are not proof of new accuracy. Threshold/model limitations are explained in `README.md`.
- Keep τ = 0.76 a constant: re-running Phase 4's train-CV selection under numpy 2 yields 0.74 (unstable `GroupKFold` tie order; see `reports/phase5_report.md` §5.1). Do not overwrite committed Phase 4 results.
- Browser checks: `scripts/browser_e2e.py` (Playwright) against a running app; it rewrites `figures/phase5_app_*.png`.

## Maintaining this file

Keep this file for knowledge useful to almost every future agent session in this project.
Do not repeat what the codebase already shows; point to the authoritative file or command instead.
Prefer rewriting or pruning existing entries over appending new ones.
When updating this file, preserve this bar for all agents and keep entries concise.
