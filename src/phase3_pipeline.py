"""CitationGuard Phase 3 — model pipeline.

Two complementary models on Phase 2 / SciFact data:

1) Evidence retrieval (TF-IDF): claim → ranked abstract sentences
2) Claim–evidence verification (TF-IDF + LogisticRegression):
   SUPPORT / CONTRADICT / NO_EVIDENCE

Fixes applied after the Phase 3 review (see README, "Phase 3 fixes"):
- NO_EVIDENCE examples are the 2 most claim-similar sentences of a cited
  non-evidence abstract, not the whole abstract (removes a length shortcut).
- The verifier encodes claim and evidence separately, keeps negation words
  and adds claim–evidence comparison features, with balanced class weights.
- Retrieval ranking uses a stable sort, so ties no longer depend on the
  numpy version.

Run: python src/phase3_pipeline.py
"""

from __future__ import annotations

import json
import re
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
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

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
NO_EVIDENCE_SENTENCES = 2  # matches the ~2 gold rationale sentences per pair

# Small hand-built lexicons for claim–evidence comparison features.
NEGATION = {
    "not", "no", "never", "none", "neither", "nor", "without", "cannot",
    "fail", "fails", "failed", "lack", "lacks", "absent", "unaffected",
    "independent", "unrelated", "unchanged", "nonsignificant", "insignificant",
}
UP_WORDS = {
    "increase", "increases", "increased", "increasing", "higher", "high",
    "more", "greater", "enhance", "enhances", "enhanced", "promote",
    "promotes", "elevated", "raise", "raises", "improve", "improves",
    "improved", "upregulates", "upregulated", "activates", "induces",
}
DOWN_WORDS = {
    "decrease", "decreases", "decreased", "decreasing", "lower", "low",
    "less", "fewer", "reduce", "reduces", "reduced", "inhibit", "inhibits",
    "inhibited", "suppress", "suppresses", "suppressed", "impair", "impairs",
    "impaired", "downregulates", "downregulated", "prevents", "blocks",
}
TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(str(text).lower())


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
        order = np.argsort(-sims, kind="stable")

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
    top = np.argsort(-sims, kind="stable")[:5]
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


def top_sentences(abstract: list[str], claim: str, vectorizer: TfidfVectorizer, k: int) -> str:
    """The k abstract sentences most similar to the claim, in abstract order."""
    sents = [str(x).strip() for x in abstract if str(x).strip()]
    sims = (vectorizer.transform([claim]) @ vectorizer.transform(sents).T).toarray().ravel()
    keep = sorted(np.argsort(-sims, kind="stable")[:k])
    return " ".join(sents[i] for i in keep)


def prepare_verifier_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    """claim / evidence / label frames for train and dev.

    SUPPORT/CONTRADICT evidence = gold rationale sentences (~2 sentences).
    NO_EVIDENCE evidence = the 2 most claim-similar sentences of a cited
    non-evidence abstract. Phase 3 originally used the whole abstract, which
    let the model detect NO_EVIDENCE from text length alone.
    """
    corpus = {int(d["doc_id"]): d["abstract"] for d in load_jsonl(RAW_CORPUS)}
    sim_vec = TfidfVectorizer(lowercase=True, stop_words="english", sublinear_tf=True)
    sim_vec.fit([sent for abstract in corpus.values() for sent in abstract])

    frames = []
    for split in ("train", "dev"):
        v = pd.read_csv(PROCESSED / f"verification_{split}.csv")
        ne = pd.read_csv(PROCESSED / f"no_evidence_{split}.csv")
        v = v.assign(evidence=v["evidence_text"].fillna(""))
        ne = ne.assign(
            evidence=[
                top_sentences(corpus[int(d)], c, sim_vec, NO_EVIDENCE_SENTENCES)
                for c, d in zip(ne["claim"], ne["doc_id"])
            ]
        )
        cols = ["claim_id", "doc_id", "claim", "evidence", "label"]
        df = pd.concat([v[cols], ne[cols]], ignore_index=True)
        frames.append(add_pair_columns(df))
    return frames[0], frames[1]


def add_pair_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["claim"] = df["claim"].fillna("")
    df["evidence"] = df["evidence"].fillna("")
    df["text"] = df["claim"] + " [SEP] " + df["evidence"]
    # Claim words that the evidence does not contain (e.g. an added "not").
    df["claim_only_tokens"] = [
        " ".join(t for t in tokens(c) if t not in set(tokens(e)))
        for c, e in zip(df["claim"], df["evidence"])
    ]
    return df


PAIR_FEATURE_NAMES = [
    "word_overlap", "log_evidence_len", "claim_negations", "evidence_negations",
    "negation_mismatch", "claim_up_evidence_down", "claim_down_evidence_up",
    "claim_numbers_missing",
]


def pair_numeric_features(df: pd.DataFrame) -> np.ndarray:
    """Claim–evidence comparison: overlap, negation and direction mismatch."""
    feats = []
    for claim, evid in zip(df["claim"], df["evidence"]):
        cs, es = set(tokens(claim)), set(tokens(evid))
        c_neg, e_neg = len(cs & NEGATION), len(es & NEGATION)
        c_up, c_down = len(cs & UP_WORDS), len(cs & DOWN_WORDS)
        e_up, e_down = len(es & UP_WORDS), len(es & DOWN_WORDS)
        c_nums = {t for t in cs if t.isdigit()}
        feats.append([
            len(cs & es) / max(len(cs), 1),
            np.log1p(len(tokens(evid))),
            c_neg, e_neg, float((c_neg > 0) != (e_neg > 0)),
            float(c_up > 0 and e_down > 0), float(c_down > 0 and e_up > 0),
            len(c_nums - es) / max(len(c_nums), 1),
        ])
    return np.array(feats, dtype=float)


def _tfidf(**kw) -> TfidfVectorizer:
    params = dict(lowercase=True, ngram_range=(1, 2), max_features=40000, min_df=2, sublinear_tf=True)
    params.update(kw)
    return TfidfVectorizer(**params)


def build_verifier() -> Pipeline:
    """Claim and evidence get separate TF-IDF blocks (stop words kept, since
    "not"/"no"/"without" carry the label) plus comparison features."""
    features = ColumnTransformer(
        [
            ("claim", _tfidf(), "claim"),
            ("evidence", _tfidf(), "evidence"),
            ("claim_only", _tfidf(ngram_range=(1, 1)), "claim_only_tokens"),
            (
                "pair",
                Pipeline([
                    ("f", FunctionTransformer(
                        pair_numeric_features,
                        feature_names_out=lambda *_: np.array(PAIR_FEATURE_NAMES),
                    )),
                    ("s", StandardScaler()),
                ]),
                ["claim", "evidence"],
            ),
        ]
    )
    # C=2.0 chosen by 5-fold grouped CV on train (C=0.5/1/2/4 -> macro F1 .598/.607/.613/.607)
    clf = LogisticRegression(max_iter=4000, solver="lbfgs", C=2.0, class_weight="balanced", random_state=42)
    return Pipeline(steps=[("x", features), ("clf", clf)])


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

    model = build_verifier()
    model.fit(train, train["label"])
    pred = model.predict(dev)
    proba = model.predict_proba(dev)
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
            "features": "TF-IDF(claim) + TF-IDF(evidence) + TF-IDF(claim-only words) + 8 comparison features",
            "classifier": "LogisticRegression(solver=lbfgs, C=2.0, class_weight=balanced)",
            "no_evidence_input": f"{NO_EVIDENCE_SENTENCES} most claim-similar sentences of a non-evidence abstract",
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

    pred_df = dev.drop(columns=["claim_only_tokens"]).copy()
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
