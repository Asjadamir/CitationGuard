"""CitationGuard Phase 5 — analysis of the shipped product.

Runs the *product* inference path (product.load_verifier -> ClaimVerifier.verify,
exactly what the web app calls) on all 300 official SciFact dev claims and
reports:

1) Consistency: product metrics vs. the Phase 4 research pipeline.
2) A new experiment: "cited-paper mode" (search restricted to the claim's
   first cited SciFact document), which Phase 4 never evaluated.
3) Error decomposition: retrieval hit/miss x verifier outcome.
4) Selective prediction: risk-coverage curve, AURC, calibration (ECE) of the
   evidence score, and the retrieval ceiling on strict accuracy.
5) Inference latency on CPU.

The model is trained on train only; dev is used only for reporting and no
threshold is retuned. Run from the repo root: python src/phase5_analysis.py
"""

from __future__ import annotations

import json
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from phase3_pipeline import FIGURES, RAW_CLAIMS, RAW_CORPUS, RESULTS, gold_evidence_keys, load_jsonl
from phase4_evaluation import SEED, claim_gold_label, mcnemar, pipeline_metrics
from product import THRESHOLD, load_verifier

PHASE4_STRICT = "Retrieval top-2 + strict abstention (tau=0.76)"


def run(verifier, claims: list[dict], restrict: bool) -> pd.DataFrame:
    rows = []
    for claim in claims:
        doc_id = int(claim["cited_doc_ids"][0]) if restrict else None
        start = time.perf_counter()
        report = verifier.verify(claim["claim"], doc_id)
        elapsed = time.perf_counter() - start
        gold = gold_evidence_keys(claim)
        used = [(e["doc_id"], e["sentence_id"]) for e in report["evidence"] if e["used_by_verifier"]]
        probs = report["probabilities"] or {}
        rows.append({
            "claim_id": claim["id"],
            "claim": claim["claim"],
            "gold": claim_gold_label(claim),
            "pred": report["label"],
            "reason": report["reason"],
            "evidence_conf": report["evidence_score"] if report["evidence_score"] is not None else 0.0,
            "verdict": max(("SUPPORT", "CONTRADICT"), key=lambda k: probs.get(k, 0.0)),
            "cited_is_gold": any(key in gold for key in used),
            "top5_has_gold": any((e["doc_id"], e["sentence_id"]) in gold for e in report["evidence"]),
            "n_words": len(claim["claim"].split()),
            "latency_s": elapsed,
        })
    return pd.DataFrame(rows)


def decompose(df: pd.DataFrame) -> dict:
    """Where each dev claim ends up under the strict operating point."""
    ne = df["gold"] == "NO_EVIDENCE"
    out = {
        "NEI: correct abstention": int((ne & (df["pred"] == "NO_EVIDENCE")).sum()),
        "NEI: hallucinated verdict": int((ne & (df["pred"] != "NO_EVIDENCE")).sum()),
    }
    ev = df[~ne]
    for hit, name in ((True, "gold in top-2"), (False, "gold not retrieved")):
        part = ev[ev["cited_is_gold"] == hit]
        out[f"{name}: correct"] = int((part["pred"] == part["gold"]).sum())
        out[f"{name}: abstained"] = int((part["pred"] == "NO_EVIDENCE").sum())
        out[f"{name}: wrong direction"] = int(((part["pred"] != part["gold"]) & (part["pred"] != "NO_EVIDENCE")).sum())
    return out


def risk_coverage(df: pd.DataFrame) -> tuple[list[dict], float]:
    """Risk = share of issued verdicts that are not correct-and-grounded."""
    good = ((df["verdict"] == df["gold"]) & df["cited_is_gold"]).to_numpy()
    order = np.argsort(-df["evidence_conf"].to_numpy(), kind="stable")
    cum_good = np.cumsum(good[order])
    n = np.arange(1, len(df) + 1)
    risk = 1 - cum_good / n
    curve = [{"coverage": float(c), "risk": float(r)} for c, r in zip(n / len(df), risk)]
    return curve, float(risk.mean())


def calibration(df: pd.DataFrame, bins: int = 10) -> tuple[list[dict], float]:
    """Is evidence_conf a probability that usable (gold) evidence was used?"""
    conf = df["evidence_conf"].to_numpy()
    hit = (df["cited_is_gold"] & (df["gold"] != "NO_EVIDENCE")).to_numpy().astype(float)
    edges = np.linspace(0, 1, bins + 1)
    rows, ece = [], 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf >= lo) & ((conf < hi) | (hi == 1.0))
        if mask.any():
            rows.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": int(mask.sum()), "mean_conf": float(conf[mask].mean()),
                         "evidence_rate": float(hit[mask].mean())})
            ece += mask.mean() * abs(conf[mask].mean() - hit[mask].mean())
    return rows, float(ece)


def plot(decomp: dict, curve: list[dict], calib: list[dict], strict_cov: float, path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), gridspec_kw={"width_ratios": [1.35, 1, 1]})
    names = list(decomp)
    colors = ["#5E9E6E", "#C0504D", "#5E9E6E", "#9AA5B1", "#C0504D", "#8FB996", "#9AA5B1", "#E08A86"]
    axes[0].barh(names[::-1], [decomp[k] for k in names][::-1], color=colors[::-1])
    for i, v in enumerate([decomp[k] for k in names][::-1]):
        axes[0].text(v + 1, i, str(v), va="center", fontsize=8)
    axes[0].set_title("Where the 300 dev claims end up (strict τ = 0.76)", fontsize=10)
    axes[0].tick_params(axis="y", labelsize=8)
    axes[0].set_xlabel("claims")

    axes[1].plot([c["coverage"] for c in curve], [c["risk"] for c in curve], color="#2F6F8F")
    axes[1].axvline(strict_cov, ls="--", color="#C0504D")
    axes[1].text(strict_cov + 0.02, 0.05, f"strict τ\ncoverage {strict_cov:.2f}", fontsize=8, color="#C0504D")
    axes[1].set(xlabel="coverage (share of claims given a verdict)", ylabel="risk (verdict not correct + grounded)",
                ylim=(0, 1), xlim=(0, 1))
    axes[1].set_title("Risk–coverage curve (product, dev)", fontsize=10)

    xs = [c["mean_conf"] for c in calib]
    axes[2].plot([0, 1], [0, 1], ls=":", color="#9AA5B1", label="perfect calibration")
    axes[2].plot(xs, [c["evidence_rate"] for c in calib], marker="o", color="#2F6F8F", label="observed")
    for c in calib:
        axes[2].annotate(str(c["n"]), (c["mean_conf"], c["evidence_rate"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    axes[2].set(xlabel="evidence score 1 − P(NO_EVIDENCE)", ylabel="share with gold evidence used", xlim=(0, 1), ylim=(0, 1))
    axes[2].set_title("Evidence-score calibration (n per bin)", fontsize=10)
    axes[2].legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    rng = np.random.default_rng(SEED)
    t0 = time.perf_counter()
    verifier = load_verifier(RAW_CORPUS.parent)
    startup = time.perf_counter() - t0
    dev = load_jsonl(RAW_CLAIMS["dev"])

    corpus_df = run(verifier, dev, restrict=False)
    cited_df = run(verifier, dev, restrict=True)

    systems = {}
    for name, df in (("Product, all papers (strict)", corpus_df), ("Product, cited paper only (strict)", cited_df)):
        m = pipeline_metrics(df, df["pred"].to_numpy(), rng)
        m["no_match_abstentions"] = int((df["reason"] == "no_matches").sum())
        m["top2_contains_gold"] = float(df.loc[df["gold"] != "NO_EVIDENCE", "cited_is_gold"].mean())
        m["top5_contains_gold"] = float(df.loc[df["gold"] != "NO_EVIDENCE", "top5_has_gold"].mean())
        systems[name] = m

    def strict_correct(df):
        return ((df["pred"] == df["gold"]) & ((df["gold"] == "NO_EVIDENCE") | df["cited_is_gold"])).to_numpy()

    cited_vs_corpus = mcnemar(strict_correct(cited_df), strict_correct(corpus_df))

    phase4 = json.loads((RESULTS / "phase4_evaluation.json").read_text(encoding="utf-8"))["pipeline"]["systems_dev"][PHASE4_STRICT]
    keys = ("accuracy", "macro_f1", "strict_accuracy", "hallucinated_verdict_rate", "coverage")
    consistency = {k: {"phase4_research": phase4[k], "product": systems["Product, all papers (strict)"][k]} for k in keys}
    p4_preds = pd.read_csv(RESULTS / "phase4_pipeline_dev_predictions.csv")
    agree = corpus_df.merge(p4_preds[["claim_id", "pred_final"]], on="claim_id")
    consistency["per_claim_label_agreement"] = float((agree["pred"] == agree["pred_final"]).mean())

    decomp = decompose(corpus_df)
    curve, aurc = risk_coverage(corpus_df)
    calib, ece = calibration(corpus_df)
    ev = corpus_df["gold"] != "NO_EVIDENCE"
    ceiling = float((~ev).mean() + (ev & corpus_df["cited_is_gold"]).mean())

    long_mask = corpus_df["n_words"] > 15
    slices = {}
    for name, mask in (("claims <= 15 words", ~long_mask), ("claims > 15 words", long_mask)):
        part = corpus_df[mask]
        slices[name] = {"n": int(mask.sum()), "strict_accuracy": pipeline_metrics(part, part["pred"].to_numpy(), rng)["strict_accuracy"],
                        "top2_contains_gold": float(part.loc[part["gold"] != "NO_EVIDENCE", "cited_is_gold"].mean())}

    lat = corpus_df["latency_s"].to_numpy()
    out = {
        "phase": 5,
        "seed": SEED,
        "threshold": THRESHOLD,
        "model": verifier.metadata,
        "systems_dev": systems,
        "consistency_with_phase4": consistency,
        "cited_vs_all_papers_mcnemar_strict": cited_vs_corpus,
        "error_decomposition_dev": decomp,
        "strict_accuracy_retrieval_ceiling": ceiling,
        "aurc": aurc,
        "risk_at_strict_tau": float(1 - systems["Product, all papers (strict)"]["strict_accuracy"]),
        "calibration_bins": calib,
        "evidence_score_ece": ece,
        "length_slices": slices,
        "latency": {"startup_s": startup, "mean_s": float(lat.mean()), "p95_s": float(np.percentile(lat, 95)), "max_s": float(lat.max())},
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "phase5_analysis.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    corpus_df.assign(cited_pred=cited_df["pred"].values).drop(columns=["latency_s"]).to_csv(
        RESULTS / "phase5_product_dev_predictions.csv", index=False)
    plot(decomp, curve, calib, systems["Product, all papers (strict)"]["coverage"], FIGURES / "phase5_error_and_selective.png")

    print(pd.DataFrame(systems).T[["accuracy", "macro_f1", "strict_accuracy", "hallucinated_verdict_rate", "coverage",
                                   "top2_contains_gold", "top5_contains_gold"]].round(3).to_string())
    print("\nConsistency with Phase 4:", json.dumps(consistency, indent=1))
    print("Cited-paper vs all-papers strict correctness (McNemar):", cited_vs_corpus)
    print("Error decomposition:", json.dumps(decomp, indent=1))
    print(f"Retrieval ceiling on strict accuracy: {ceiling:.3f}  AURC: {aurc:.3f}  ECE: {ece:.3f}")
    print("Length slices:", json.dumps(slices, indent=1))
    print("Latency:", json.dumps(out["latency"], indent=1))


if __name__ == "__main__":
    main()
