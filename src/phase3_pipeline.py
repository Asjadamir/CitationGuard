"""CitationGuard Phase 3 — model pipeline.

Two complementary models on Phase 2 / SciFact data:

1) Evidence retrieval (TF-IDF): claim → ranked abstract sentences
2) Claim–evidence verification (TF-IDF + LogisticRegression):
   SUPPORT / CONTRADICT / NO_EVIDENCE

Run: python src/phase3_pipeline.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
RAW_CORPUS = ROOT / "data" / "raw" / "scifact" / "data" / "corpus.jsonl"
RAW_CLAIMS = {
    "train": ROOT / "data" / "raw" / "scifact" / "data" / "claims_train.jsonl",
    "dev": ROOT / "data" / "raw" / "scifact" / "data" / "claims_dev.jsonl",
}
PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"

VERIFIER_LABELS = ["SUPPORT", "CONTRADICT", "NO_EVIDENCE"]
RECALL_KS = (1, 3, 5, 10)


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_sentence_index(corpus_rows: list[dict]) -> pd.DataFrame:
    records = []
    for doc in corpus_rows:
        doc_id = int(doc["doc_id"])
        title = doc.get("title") or ""
        abstract = doc.get("abstract") or []
        for sent_id, sentence in enumerate(abstract):
            text = str(sentence).strip()
            if not text:
                continue
            records.append(
                {
                    "doc_id": doc_id,
                    "sent_id": sent_id,
                    "title": title,
                    "sentence": text,
                    "indexed_text": f"{title}. {text}",
                }
            )
    return pd.DataFrame(records)


def gold_evidence_keys(claim: dict) -> set[tuple[int, int]]:
    keys: set[tuple[int, int]] = set()
    evidence = claim.get("evidence") or {}
    for doc_id, rationales in evidence.items():
        doc_id_int = int(doc_id)
        for rationale in rationales:
            for sent_id in rationale.get("sentences") or []:
                keys.add((doc_id_int, int(sent_id)))
    return keys


def claims_with_evidence(claims: list[dict]) -> list[dict]:
    return [c for c in claims if gold_evidence_keys(c)]


def evaluate_retrieval(
    claims: list[dict],
    sentence_df: pd.DataFrame,
    vectorizer: TfidfVectorizer,
    sentence_matrix,
    ks: tuple[int, ...] = RECALL_KS,
) -> dict:
    """Rank all corpus sentences for each claim; score gold-sentence recall."""
    recalls = {k: [] for k in ks}
    mrr_scores = []
    n_gold = []

    sent_keys = list(zip(sentence_df["doc_id"].tolist(), sentence_df["sent_id"].tolist()))

    for claim in claims:
        gold = gold_evidence_keys(claim)
        if not gold:
            continue
        n_gold.append(len(gold))
        q = vectorizer.transform([claim["claim"]])
        sims = cosine_similarity(q, sentence_matrix).ravel()
        order = np.argsort(-sims)

        rank_of_first = None
        for rank_idx, sent_idx in enumerate(order, start=1):
            if sent_keys[sent_idx] in gold:
                rank_of_first = rank_idx
                break

        for k in ks:
            top_keys = {sent_keys[i] for i in order[:k]}
            recalls[k].append(len(gold & top_keys) / len(gold))

        mrr_scores.append(0.0 if rank_of_first is None else 1.0 / rank_of_first)

    metrics = {
        "n_claims_with_evidence": len(mrr_scores),
        "mean_gold_sentences": float(np.mean(n_gold)) if n_gold else 0.0,
        "mrr": float(np.mean(mrr_scores)) if mrr_scores else 0.0,
    }
    for k in ks:
        metrics[f"recall@{k}"] = float(np.mean(recalls[k])) if recalls[k] else 0.0
        metrics[f"hit@{k}"] = float(np.mean([1 if r > 0 else 0 for r in recalls[k]]))
    return metrics


def run_retrieval() -> dict:
    print("\n" + "=" * 64)
    print("Phase 3A — TF-IDF Evidence Sentence Retrieval")
    print("=" * 64)

    corpus_rows = load_jsonl(RAW_CORPUS)
    sentence_df = build_sentence_index(corpus_rows)
    print(f"Corpus docs: {len(corpus_rows)}")
    print(f"Indexed sentences: {len(sentence_df)}")

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_features=50000,
        sublinear_tf=True,
    )
    sentence_matrix = vectorizer.fit_transform(sentence_df["indexed_text"])

    results = {}
    for split, path in RAW_CLAIMS.items():
        claims = claims_with_evidence(load_jsonl(path))
        metrics = evaluate_retrieval(claims, sentence_df, vectorizer, sentence_matrix)
        results[split] = metrics
        print(f"\n[{split}] claims with gold evidence: {metrics['n_claims_with_evidence']}")
        print(f"  MRR:      {metrics['mrr']:.4f}")
        for k in RECALL_KS:
            print(
                f"  Recall@{k:<2} {metrics[f'recall@{k}']:.4f} | "
                f"Hit@{k:<2} {metrics[f'hit@{k}']:.4f}"
            )

    # Demo: top-5 retrieved sentences for first dev claim with evidence
    demo_claims = claims_with_evidence(load_jsonl(RAW_CLAIMS["dev"]))
    demo = demo_claims[0]
    q = vectorizer.transform([demo["claim"]])
    sims = cosine_similarity(q, sentence_matrix).ravel()
    top = np.argsort(-sims)[:5]
    gold = gold_evidence_keys(demo)
    demo_rows = []
    for rank, idx in enumerate(top, start=1):
        row = sentence_df.iloc[idx]
        key = (int(row.doc_id), int(row.sent_id))
        demo_rows.append(
            {
                "rank": rank,
                "score": float(sims[idx]),
                "doc_id": int(row.doc_id),
                "sent_id": int(row.sent_id),
                "is_gold": key in gold,
                "sentence": row.sentence[:180],
            }
        )
    demo_df = pd.DataFrame(demo_rows)
    print("\nDemo retrieval — claim:")
    print(f"  {demo['claim']}")
    print("Top-5 sentences:")
    print(demo_df.to_string(index=False))

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = {
        "model": {
            "type": "tfidf_sentence_retrieval",
            "vectorizer": "TfidfVectorizer(word 1-2grams, max_features=50000)",
            "similarity": "cosine",
            "candidate_pool": "all abstract sentences in SciFact corpus",
        },
        "metrics": results,
        "demo_claim_id": demo.get("id"),
        "demo_top5": demo_rows,
    }
    path = RESULTS / "phase3_retrieval_metrics.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    demo_df.to_csv(RESULTS / "phase3_retrieval_demo.csv", index=False)
    print(f"\nSaved {path}")
    return out


def prepare_verifier_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_v = pd.read_csv(PROCESSED / "verification_train.csv")
    dev_v = pd.read_csv(PROCESSED / "verification_dev.csv")
    train_ne = pd.read_csv(PROCESSED / "no_evidence_train.csv")
    dev_ne = pd.read_csv(PROCESSED / "no_evidence_dev.csv")

    train_v = train_v.assign(
        text=train_v["claim"].fillna("") + " [SEP] " + train_v["evidence_text"].fillna("")
    )
    dev_v = dev_v.assign(
        text=dev_v["claim"].fillna("") + " [SEP] " + dev_v["evidence_text"].fillna("")
    )
    train_ne = train_ne.assign(
        text=train_ne["claim"].fillna("")
        + " [SEP] "
        + train_ne["abstract_text"].fillna("")
    )
    dev_ne = dev_ne.assign(
        text=dev_ne["claim"].fillna("") + " [SEP] " + dev_ne["abstract_text"].fillna("")
    )

    train = pd.concat(
        [train_v[["claim_id", "claim", "text", "label"]], train_ne[["claim_id", "claim", "text", "label"]]],
        ignore_index=True,
    )
    dev = pd.concat(
        [dev_v[["claim_id", "claim", "text", "label"]], dev_ne[["claim_id", "claim", "text", "label"]]],
        ignore_index=True,
    )
    return train, dev


def plot_confusion(cm: np.ndarray, labels: list[str], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Phase 3B — Verifier Confusion Matrix (Dev)")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_verifier() -> dict:
    print("\n" + "=" * 64)
    print("Phase 3B — TF-IDF + LogisticRegression Verifier (3-class)")
    print("=" * 64)

    train, dev = prepare_verifier_frames()
    print("Train label counts:\n", train["label"].value_counts().to_string())
    print("Dev label counts:\n", dev["label"].value_counts().to_string())

    model = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    ngram_range=(1, 2),
                    max_features=40000,
                    min_df=2,
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=4000,
                    solver="lbfgs",
                    C=1.0,
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(train["text"], train["label"])
    pred = model.predict(dev["text"])
    proba = model.predict_proba(dev["text"])
    class_index = {c: i for i, c in enumerate(model.classes_)}

    metrics = {
        "accuracy": float(accuracy_score(dev["label"], pred)),
        "macro_f1": float(f1_score(dev["label"], pred, average="macro", labels=VERIFIER_LABELS)),
        "weighted_f1": float(
            f1_score(dev["label"], pred, average="weighted", labels=VERIFIER_LABELS)
        ),
        "majority_baseline_accuracy": float(dev["label"].value_counts(normalize=True).iloc[0]),
        "per_class": classification_report(
            dev["label"],
            pred,
            labels=VERIFIER_LABELS,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            dev["label"], pred, labels=VERIFIER_LABELS
        ).tolist(),
        "train_size": int(len(train)),
        "dev_size": int(len(dev)),
        "model": {
            "vectorizer": "TfidfVectorizer(1-2grams, max_features=40000)",
            "classifier": "LogisticRegression(solver=lbfgs, C=1.0)",
            "labels": VERIFIER_LABELS,
        },
    }

    print(f"\nAccuracy:            {metrics['accuracy']:.4f}")
    print(f"Majority baseline:   {metrics['majority_baseline_accuracy']:.4f}")
    print(f"Macro F1:            {metrics['macro_f1']:.4f}")
    print(f"Weighted F1:         {metrics['weighted_f1']:.4f}")
    print("\nClassification report (dev):")
    print(
        classification_report(
            dev["label"], pred, labels=VERIFIER_LABELS, digits=4, zero_division=0
        )
    )

    pred_df = dev.copy()
    pred_df["predicted_label"] = pred
    for label in VERIFIER_LABELS:
        pred_df[f"prob_{label}"] = proba[:, class_index[label]]
    pred_df["correct"] = pred_df["label"] == pred_df["predicted_label"]

    FIGURES.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    cm_path = FIGURES / "phase3_confusion_matrix.png"
    plot_confusion(np.array(metrics["confusion_matrix"]), VERIFIER_LABELS, cm_path)

    pred_path = RESULTS / "phase3_verifier_dev_predictions.csv"
    metrics_path = RESULTS / "phase3_verifier_metrics.json"
    pred_df.to_csv(pred_path, index=False)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("Sample predictions:")
    cols = ["claim_id", "label", "predicted_label", "correct", "claim"]
    print(pred_df[cols].head(10).to_string(index=False))
    print(f"\nSaved {metrics_path}")
    print(f"Saved {pred_path}")
    print(f"Saved {cm_path}")
    return metrics


def plot_retrieval_bars(retrieval_out: dict, out_path: Path) -> None:
    split = "dev"
    metrics = retrieval_out["metrics"][split]
    labels = [f"R@{k}" for k in RECALL_KS] + ["MRR"]
    values = [metrics[f"recall@{k}"] for k in RECALL_KS] + [metrics["mrr"]]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, values, color="#2F6F8F")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Phase 3A — TF-IDF Retrieval on Dev (open corpus)")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.2f}", ha="center")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    retrieval_out = run_retrieval()
    plot_retrieval_bars(retrieval_out, FIGURES / "phase3_retrieval_metrics.png")
    verifier_out = run_verifier()

    summary = {
        "phase": 3,
        "project": "CitationGuard",
        "retrieval_dev_mrr": retrieval_out["metrics"]["dev"]["mrr"],
        "retrieval_dev_recall@5": retrieval_out["metrics"]["dev"]["recall@5"],
        "retrieval_dev_hit@5": retrieval_out["metrics"]["dev"]["hit@5"],
        "verifier_dev_accuracy": verifier_out["accuracy"],
        "verifier_dev_macro_f1": verifier_out["macro_f1"],
        "verifier_majority_baseline": verifier_out["majority_baseline_accuracy"],
    }
    summary_path = RESULTS / "phase3_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 64)
    print("PHASE 3 SUMMARY")
    print("=" * 64)
    print(json.dumps(summary, indent=2))
    print(f"\nWrote {summary_path}")


if __name__ == "__main__":
    main()
