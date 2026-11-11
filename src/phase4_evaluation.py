"""CitationGuard Phase 4 — evaluation and analysis.

Builds on the Phase 3 models and evaluates them properly:

1) Retrieval comparison: TF-IDF (Phase 3) vs BM25 vs a doc+sentence hybrid,
   plus a vocabulary-size ("embedding size") sweep, with bootstrap 95% CIs.
2) Verifier comparison under two protocols:
     A. original Phase 3 data (NO_EVIDENCE = whole abstract)
     B. fixed Phase 3 data (NO_EVIDENCE = 2 most claim-similar sentences)
   Baselines include majority class, a claim-only model (no evidence) and
   the original Phase 3 verifier; "Phase 3 fixed" is build_verifier().
3) End-to-end pipeline (retrieve -> verify) at claim level, compared with
   no retrieval and oracle evidence, with an abstention threshold that
   limits hallucinated (unsupported) verdicts.
4) Error analysis and subgroup / bias checks on the best verifier.

Hyperparameters and thresholds are chosen on train (grouped cross-validation
by claim); the official dev split is used only for the reported numbers.

Run: python src/phase4_evaluation.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.naive_bayes import ComplementNB
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.svm import LinearSVC

from phase3_pipeline import (
    FIGURES,
    PROCESSED,
    RAW_CLAIMS,
    RAW_CORPUS,
    RESULTS,
    _tfidf,
    add_pair_columns,
    build_sentence_index,
    build_verifier,
    gold_evidence_keys,
    load_jsonl,
    pair_numeric_features,
    prepare_verifier_frames,
    tokens,
    top_sentences,
)

SEED = 42
N_BOOT = 1000
KS = (1, 3, 5, 10)
CLAIM_LABELS = ["SUPPORT", "CONTRADICT", "NO_EVIDENCE"]

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, n_boot: int = N_BOOT) -> list[float]:
    """95% percentile CI of the mean of per-item values."""
    values = np.asarray(values, dtype=float)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def cluster_bootstrap_f1(
    y_true: np.ndarray, y_pred: np.ndarray, groups: np.ndarray, rng: np.random.Generator
) -> dict:
    """Bootstrap over claims (rows of the same claim are resampled together)."""
    uniq = np.unique(groups)
    rows_by_group = {g: np.flatnonzero(groups == g) for g in uniq}
    accs, f1s = [], []
    for _ in range(N_BOOT):
        sample = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([rows_by_group[g] for g in sample])
        accs.append(accuracy_score(y_true[idx], y_pred[idx]))
        f1s.append(f1_score(y_true[idx], y_pred[idx], average="macro", labels=CLAIM_LABELS, zero_division=0))
    return {
        "accuracy_ci": [float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))],
        "macro_f1_ci": [float(np.percentile(f1s, 2.5)), float(np.percentile(f1s, 97.5))],
    }


def mcnemar(correct_a: np.ndarray, correct_b: np.ndarray) -> dict:
    """Exact McNemar test on paired correctness vectors."""
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    p = 1.0 if b + c == 0 else float(binomtest(b, b + c, 0.5).pvalue)
    return {"a_only_correct": b, "b_only_correct": c, "p_value": p}


def classification_metrics(y_true, y_pred) -> dict:
    f1s = f1_score(y_true, y_pred, average=None, labels=CLAIM_LABELS, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=CLAIM_LABELS, zero_division=0)),
        **{f"f1_{lab}": float(v) for lab, v in zip(CLAIM_LABELS, f1s)},
    }


# ---------------------------------------------------------------------------
# 1) Retrieval
# ---------------------------------------------------------------------------


class Corpus:
    """Sentence index plus lookups shared by retrieval and the pipeline."""

    def __init__(self) -> None:
        self.docs = load_jsonl(RAW_CORPUS)
        self.sent_df = build_sentence_index(self.docs)
        self.keys = list(zip(self.sent_df["doc_id"], self.sent_df["sent_id"]))
        self.key_to_idx = {k: i for i, k in enumerate(self.keys)}
        self.doc_ids = [int(d["doc_id"]) for d in self.docs]
        self.doc_pos = {d: i for i, d in enumerate(self.doc_ids)}
        self.sent_doc_pos = np.array([self.doc_pos[d] for d in self.sent_df["doc_id"]])
        self.abstract = {int(d["doc_id"]): [s.strip() for s in d["abstract"] if s.strip()] for d in self.docs}
        self.doc_text = [f"{d.get('title') or ''}. {' '.join(d['abstract'])}" for d in self.docs]


class TfidfRetriever:
    name = "TF-IDF (Phase 3)"

    def __init__(self, texts: list[str], max_features: int | None = 50000, ngram_range=(1, 2)):
        self.vec = TfidfVectorizer(
            lowercase=True, stop_words="english", ngram_range=ngram_range,
            min_df=2, max_features=max_features, sublinear_tf=True,
        )
        self.matrix = self.vec.fit_transform(texts)

    def score(self, queries: list[str]) -> np.ndarray:
        return (self.vec.transform(queries) @ self.matrix.T).toarray()


class BM25Retriever:
    """Okapi BM25 with sparse matrices (no extra dependency)."""

    name = "BM25"

    def __init__(self, texts: list[str], k1: float = 1.2, b: float = 0.75):
        self.vec = CountVectorizer(lowercase=True, stop_words="english", min_df=1)
        tf = self.vec.fit_transform(texts).tocsr().astype(np.float32)
        n_docs = tf.shape[0]
        df = np.bincount(tf.indices, minlength=tf.shape[1])
        idf = np.log(1 + (n_docs - df + 0.5) / (df + 0.5))
        lengths = np.asarray(tf.sum(axis=1)).ravel()
        norm = k1 * (1 - b + b * lengths / lengths.mean())
        row_norm = np.repeat(norm, np.diff(tf.indptr))
        tf.data = tf.data * (k1 + 1) / (tf.data + row_norm)
        self.weights = (tf @ _diag(idf)).T.tocsr()  # vocab x docs

    def score(self, queries: list[str]) -> np.ndarray:
        q = self.vec.transform(queries)
        q.data[:] = 1.0
        return (q @ self.weights).toarray()


def _diag(values: np.ndarray):
    from scipy.sparse import diags

    return diags(values.astype(np.float32))


class HybridRetriever:
    """Sentence score + lambda * score of the sentence's whole abstract."""

    name = "Hybrid (sentence + abstract)"

    def __init__(self, corpus: Corpus, sent_retriever, doc_retriever, lam: float):
        self.corpus, self.sent, self.doc, self.lam = corpus, sent_retriever, doc_retriever, lam

    def score(self, queries: list[str]) -> np.ndarray:
        s = self.sent.score(queries)
        d = self.doc.score(queries)
        return _minmax(s) + self.lam * _minmax(d)[:, self.corpus.sent_doc_pos]


def _minmax(m: np.ndarray) -> np.ndarray:
    lo = m.min(axis=1, keepdims=True)
    hi = m.max(axis=1, keepdims=True)
    return (m - lo) / np.maximum(hi - lo, 1e-9)


def per_claim_retrieval(retriever, claims: list[dict], corpus: Corpus, batch: int = 100) -> pd.DataFrame:
    """Rank of gold sentences per claim → recall@k, hit@k, reciprocal rank."""
    rows = []
    for start in range(0, len(claims), batch):
        chunk = claims[start : start + batch]
        scores = retriever.score([c["claim"] for c in chunk])
        for claim, s in zip(chunk, scores):
            gold_idx = [corpus.key_to_idx[k] for k in gold_evidence_keys(claim) if k in corpus.key_to_idx]
            ranks = np.array([(s > s[g]).sum() + 1 for g in gold_idx])
            row = {"claim_id": claim["id"], "rr": 1.0 / ranks.min(), "best_rank": int(ranks.min())}
            for k in KS:
                row[f"recall@{k}"] = float((ranks <= k).mean())
                row[f"hit@{k}"] = float((ranks <= k).any())
            rows.append(row)
    return pd.DataFrame(rows)


def per_claim_doc_retrieval(retriever, claims: list[dict], corpus: Corpus) -> pd.DataFrame:
    rows = []
    scores = retriever.score([c["claim"] for c in claims])
    for claim, s in zip(claims, scores):
        gold = [corpus.doc_pos[int(d)] for d in claim["evidence"]]
        best = min((s > s[g]).sum() + 1 for g in gold)
        rows.append({"claim_id": claim["id"], **{f"hit@{k}": float(best <= k) for k in KS}})
    return pd.DataFrame(rows)


def summarize_retrieval(df: pd.DataFrame, rng) -> dict:
    out = {"mrr": float(df["rr"].mean()), "mrr_ci": bootstrap_ci(df["rr"], rng)}
    for k in KS:
        out[f"recall@{k}"] = float(df[f"recall@{k}"].mean())
        out[f"hit@{k}"] = float(df[f"hit@{k}"].mean())
    out["recall@5_ci"] = bootstrap_ci(df["recall@5"], rng)
    return out


def run_retrieval_eval(corpus: Corpus, train_claims, dev_claims, rng) -> dict:
    print("\n" + "=" * 64)
    print("Phase 4A — Retrieval comparison")
    print("=" * 64)
    texts = corpus.sent_df["indexed_text"].tolist()

    tfidf = TfidfRetriever(texts)

    # BM25: choose k1, b on TRAIN claims (dev untouched)
    grid = []
    for k1 in (0.9, 1.2, 1.6):
        for b in (0.5, 0.75, 0.9):
            m = per_claim_retrieval(BM25Retriever(texts, k1, b), train_claims, corpus)["rr"].mean()
            grid.append({"k1": k1, "b": b, "train_mrr": float(m)})
    best_bm = max(grid, key=lambda g: g["train_mrr"])
    bm25 = BM25Retriever(texts, best_bm["k1"], best_bm["b"])
    print(f"BM25 tuned on train: k1={best_bm['k1']} b={best_bm['b']}")

    # Abstract-level retriever + hybrid lambda tuned on train
    doc_tfidf = TfidfRetriever(corpus.doc_text)
    doc_tfidf.name = "TF-IDF abstract"
    doc_bm25 = BM25Retriever(corpus.doc_text, best_bm["k1"], best_bm["b"])
    doc_bm25.name = "BM25 abstract"
    lam_grid = []
    for lam in (0.0, 0.25, 0.5, 1.0, 2.0):
        m = per_claim_retrieval(HybridRetriever(corpus, bm25, doc_bm25, lam), train_claims, corpus)["rr"].mean()
        lam_grid.append({"lambda": lam, "train_mrr": float(m)})
    best_lam = max(lam_grid, key=lambda g: g["train_mrr"])["lambda"]
    hybrid = HybridRetriever(corpus, bm25, doc_bm25, best_lam)
    print(f"Hybrid lambda tuned on train: {best_lam}")

    per_claim = {}
    results = {}
    for r in (tfidf, bm25, hybrid):
        per_claim[r.name] = per_claim_retrieval(r, dev_claims, corpus)
        results[r.name] = summarize_retrieval(per_claim[r.name], rng)
        m = results[r.name]
        print(f"{r.name:<30} MRR {m['mrr']:.3f} {m['mrr_ci']}  R@5 {m['recall@5']:.3f}  Hit@5 {m['hit@5']:.3f}")

    doc_results = {}
    for r in (doc_tfidf, doc_bm25):
        d = per_claim_doc_retrieval(r, dev_claims, corpus)
        doc_results[r.name] = {f"hit@{k}": float(d[f"hit@{k}"].mean()) for k in KS}
        print(f"{r.name:<30} abstract Hit@1 {doc_results[r.name]['hit@1']:.3f}  Hit@10 {doc_results[r.name]['hit@10']:.3f}")

    # Paired comparison: does the best retriever beat Phase 3 per claim?
    base, best_name = per_claim[tfidf.name], max(results, key=lambda n: results[n]["mrr"])
    diff = per_claim[best_name]["rr"].values - base["rr"].values
    paired = {"best": best_name, "mean_rr_gain": float(diff.mean()), "gain_ci": bootstrap_ci(diff, rng)}

    # Vocabulary-size ("embedding size") sweep for the TF-IDF retriever
    sweep = []
    for mf in (2000, 5000, 10000, 20000, 50000, 100000, None):
        r = TfidfRetriever(texts, max_features=mf)
        d = per_claim_retrieval(r, dev_claims, corpus)
        sweep.append({"max_features": mf or len(r.vec.vocabulary_), "mrr": float(d["rr"].mean()),
                      "recall@5": float(d["recall@5"].mean())})
    print("Vocabulary sweep:", [(s["max_features"], round(s["mrr"], 3)) for s in sweep])

    plot_retrieval_comparison(results, sweep, FIGURES / "phase4_retrieval_comparison.png")
    best_df = per_claim[best_name].merge(base[["claim_id", "best_rank"]], on="claim_id", suffixes=("", "_tfidf"))
    best_df.to_csv(RESULTS / "phase4_retrieval_per_claim.csv", index=False)

    return {
        "best_retriever": best_name,
        "bm25_grid_train": grid,
        "hybrid_lambda_grid_train": lam_grid,
        "sentence_retrieval_dev": results,
        "abstract_retrieval_dev": doc_results,
        "paired_vs_phase3": paired,
        "vocab_size_sweep_dev": sweep,
        "_retrievers": {"tfidf": tfidf, "bm25": bm25, "hybrid": hybrid},
    }


def plot_retrieval_comparison(results: dict, sweep: list[dict], out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    names = list(results)
    metrics = ["recall@1", "recall@5", "recall@10", "mrr"]
    width = 0.8 / len(names)
    colors = ["#9AA5B1", "#2F6F8F", "#D08C34"]
    for i, n in enumerate(names):
        vals = [results[n][m] for m in metrics]
        axes[0].bar(np.arange(len(metrics)) + i * width, vals, width, label=n, color=colors[i % 3])
    axes[0].set_xticks(np.arange(len(metrics)) + width * (len(names) - 1) / 2)
    axes[0].set_xticklabels(["R@1", "R@5", "R@10", "MRR"])
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Sentence retrieval on dev (open corpus)")
    axes[0].legend(fontsize=8, frameon=False)
    axes[1].plot([s["max_features"] for s in sweep], [s["mrr"] for s in sweep], marker="o", color="#2F6F8F", label="MRR")
    axes[1].plot([s["max_features"] for s in sweep], [s["recall@5"] for s in sweep], marker="s", color="#D08C34", label="Recall@5")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("TF-IDF vocabulary size (vector dimension)")
    axes[1].set_title("Effect of representation size (dev)")
    axes[1].set_ylim(0, 0.7)
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2) Verifier comparison
# ---------------------------------------------------------------------------


def verifier_models() -> dict:
    """Name -> factory. Column-based so every model sees the same DataFrame."""

    def on(col, vec=None):
        return ColumnTransformer([(col, vec or _tfidf(stop_words="english"), col)])

    return {
        "Majority class": None,
        "Claim only (no evidence)": lambda: Pipeline([("x", on("claim")), ("clf", LogisticRegression(max_iter=4000))]),
        "Evidence only": lambda: Pipeline([("x", on("evidence")), ("clf", LogisticRegression(max_iter=4000))]),
        "Phase 3 original: TF-IDF + LR": lambda: Pipeline([("x", on("text")), ("clf", LogisticRegression(max_iter=4000, C=1.0))]),
        "TF-IDF + LR (balanced)": lambda: Pipeline([("x", on("text")), ("clf", LogisticRegression(max_iter=4000, class_weight="balanced"))]),
        "TF-IDF + LinearSVC": lambda: Pipeline([("x", on("text")), ("clf", LinearSVC(C=0.5, class_weight="balanced"))]),
        "TF-IDF + ComplementNB": lambda: Pipeline([("x", on("text")), ("clf", ComplementNB())]),
        "Phase 3 fixed: pair features + LR": build_verifier,
    }


def build_frames() -> dict:
    """B = fixed Phase 3 frames; A = same rows with NO_EVIDENCE shown as the
    whole abstract, i.e. the original Phase 3 data with the length shortcut."""
    train_b, dev_b = prepare_verifier_frames()
    frames = {}
    for split, df in (("train", train_b), ("dev", dev_b)):
        ne = pd.read_csv(PROCESSED / f"no_evidence_{split}.csv")
        whole = {(c, d): a for c, d, a in zip(ne["claim_id"], ne["doc_id"], ne["abstract_text"])}
        a = df.copy()
        mask = a["label"] == "NO_EVIDENCE"
        a.loc[mask, "evidence"] = [whole[(c, d)] for c, d in zip(a.loc[mask, "claim_id"], a.loc[mask, "doc_id"])]
        frames.setdefault("A", []).append(add_pair_columns(a))
        frames.setdefault("B", []).append(df)
    return {k: tuple(v) for k, v in frames.items()}


def fit_predict(factory, train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    if factory is None:
        return np.full(len(test), train["label"].value_counts().idxmax(), dtype=object)
    model = factory()
    model.fit(train, train["label"])
    return model.predict(test)


def run_verifier_eval(rng) -> dict:
    print("\n" + "=" * 64)
    print("Phase 4B — Verifier comparison (oracle evidence)")
    print("=" * 64)
    frames = build_frames()

    models = verifier_models()
    out = {"protocols": {}, "cv_train": {}}
    preds_b = {}
    for protocol, (train, dev) in frames.items():
        rows = {}
        for name, factory in models.items():
            pred = fit_predict(factory, train, dev)
            m = classification_metrics(dev["label"].values, pred)
            m.update(cluster_bootstrap_f1(dev["label"].values, pred, dev["claim_id"].values, rng))
            rows[name] = m
            if protocol == "B":
                preds_b[name] = pred
        out["protocols"][protocol] = rows
        print(f"\nProtocol {protocol} ({'whole abstract' if protocol == 'A' else 'length-controlled'} NO_EVIDENCE)")
        print(pd.DataFrame(rows).T[["accuracy", "macro_f1", "f1_SUPPORT", "f1_CONTRADICT", "f1_NO_EVIDENCE"]].round(3).to_string())

    # 5-fold grouped CV on train (protocol B) — model selection without dev
    train_b, dev_b = frames["B"]
    for name, factory in models.items():
        scores = []
        for tr, te in GroupKFold(n_splits=5).split(train_b, groups=train_b["claim_id"]):
            pred = fit_predict(factory, train_b.iloc[tr], train_b.iloc[te])
            scores.append(f1_score(train_b["label"].iloc[te], pred, average="macro", labels=CLAIM_LABELS, zero_division=0))
        out["cv_train"][name] = {"macro_f1_mean": float(np.mean(scores)), "macro_f1_std": float(np.std(scores))}
    selected = max((n for n in models if models[n] is not None), key=lambda n: out["cv_train"][n]["macro_f1_mean"])
    out["selected_by_cv"] = selected
    print(f"\nSelected by 5-fold grouped CV on train: {selected}")

    y = dev_b["label"].values
    out["mcnemar_selected_vs_phase3_original"] = mcnemar(preds_b[selected] == y, preds_b["Phase 3 original: TF-IDF + LR"] == y)
    out["confusion_selected_B"] = confusion_matrix(y, preds_b[selected], labels=CLAIM_LABELS).tolist()
    out["confusion_phase3_B"] = confusion_matrix(y, preds_b["Phase 3 original: TF-IDF + LR"], labels=CLAIM_LABELS).tolist()

    plot_verifier_comparison(out, FIGURES / "phase4_verifier_comparison.png")
    plot_confusions(out, FIGURES / "phase4_confusion_matrices.png")

    pred_df = dev_b[["claim_id", "doc_id", "claim", "evidence", "label"]].copy()
    pred_df["pred_phase3_original"] = preds_b["Phase 3 original: TF-IDF + LR"]
    pred_df["pred_selected"] = preds_b[selected]
    pred_df["pred_claim_only"] = preds_b["Claim only (no evidence)"]
    pred_df.to_csv(RESULTS / "phase4_verifier_dev_predictions.csv", index=False)
    out["_frames_B"] = frames["B"]
    out["_factory"] = models[selected]
    return out


def plot_verifier_comparison(out: dict, path: Path) -> None:
    names = [n for n in out["protocols"]["A"]]
    a = [out["protocols"]["A"][n]["macro_f1"] for n in names]
    b = [out["protocols"]["B"][n]["macro_f1"] for n in names]
    err = np.array([[out["protocols"]["B"][n]["macro_f1"] - out["protocols"]["B"][n]["macro_f1_ci"][0],
                     out["protocols"]["B"][n]["macro_f1_ci"][1] - out["protocols"]["B"][n]["macro_f1"]] for n in names]).T
    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.barh(y + 0.2, a, 0.4, color="#9AA5B1", label="A: NO_EVIDENCE = whole abstract")
    ax.barh(y - 0.2, b, 0.4, xerr=err, color="#2F6F8F", label="B: length-controlled (95% CI)", error_kw={"lw": 1})
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Macro F1 on dev")
    ax.set_title("Verifier comparison (gold evidence)")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusions(out: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, key, title in (
        (axes[0], "confusion_phase3_B", "Phase 3 original"),
        (axes[1], "confusion_selected_B", out["selected_by_cv"]),
    ):
        cm = np.array(out[key])
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        ax.set_xticklabels(["SUP", "CON", "NE"])
        ax.set_yticklabels(["SUP", "CON", "NE"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{title}\n(dev, protocol B)", fontsize=10)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3) End-to-end pipeline: retrieval-augmented vs no retrieval vs oracle
# ---------------------------------------------------------------------------


def claim_gold_label(claim: dict) -> str:
    labels = {r["label"] for rs in claim["evidence"].values() for r in rs}
    return labels.pop() if labels else "NO_EVIDENCE"


def pipeline_predict(model, claims: list[dict], scores: np.ndarray, corpus: Corpus, k: int):
    """Verify the claim against its top-k retrieved sentences joined together
    (same shape as the 2-sentence training evidence). Chosen over
    max-over-top-5 / mean-over-top-5 / top-1 by out-of-fold macro F1 on train
    (.451 vs .294 / .361 / .396)."""
    rows = []
    classes = list(model.classes_)
    for claim, s in zip(claims, scores):
        top = np.argsort(-s, kind="stable")[:k]
        frame = add_pair_columns(pd.DataFrame({
            "claim": [claim["claim"]],
            "evidence": [" ".join(corpus.sent_df["sentence"].values[top])],
        }))
        proba = model.predict_proba(frame)[0]
        sc = {lab: proba[classes.index(lab)] for lab in ("SUPPORT", "CONTRADICT")}
        gold = gold_evidence_keys(claim)
        rows.append({
            "claim_id": claim["id"],
            "gold": claim_gold_label(claim),
            "argmax": classes[int(np.argmax(proba))],
            "verdict": max(sc, key=sc.get),
            "evidence_conf": float(1 - proba[classes.index("NO_EVIDENCE")]),
            "retrieval_score": float(s[top[0]]),
            "cited_is_gold": any(corpus.keys[i] in gold for i in top),
        })
    return pd.DataFrame(rows)


def apply_threshold(df: pd.DataFrame, tau: float) -> np.ndarray:
    return np.where(df["evidence_conf"] >= tau, df["verdict"], "NO_EVIDENCE")


def pipeline_metrics(df: pd.DataFrame, pred: np.ndarray, rng) -> dict:
    y = df["gold"].values
    m = classification_metrics(y, pred)
    gold_ne = y == "NO_EVIDENCE"
    asserted = pred != "NO_EVIDENCE"
    m["hallucinated_verdict_rate"] = float(asserted[gold_ne].mean())  # verdict on a claim with no evidence
    m["ungrounded_verdict_rate"] = float((~df["cited_is_gold"].values[asserted]).mean()) if asserted.any() else 0.0
    # FEVER-style strict accuracy: correct label AND (abstained correctly or cited a gold sentence)
    strict = (pred == y) & (gold_ne | df["cited_is_gold"].values)
    m["strict_accuracy"] = float(strict.mean())
    m["coverage"] = float(asserted.mean())
    m["accuracy_ci"] = bootstrap_ci(pred == y, rng)
    return m


def run_pipeline_eval(corpus: Corpus, retrievers: dict, best_name: str, ver: dict, train_claims_all, dev_claims_all, rng) -> dict:
    print("\n" + "=" * 64)
    print("Phase 4C — End-to-end pipeline (claim level)")
    print("=" * 64)
    retriever = {r.name: r for r in retrievers.values()}[best_name]
    train_b, dev_b = ver["_frames_B"]
    factory = ver["_factory"]
    k = 2

    # Abstention threshold tuned on TRAIN with out-of-fold verifiers (no dev peeking).
    train_scores = retriever.score([c["claim"] for c in train_claims_all])
    oof = []
    claim_ids = np.array([c["id"] for c in train_claims_all])
    for fold, (tr, te) in enumerate(GroupKFold(n_splits=5).split(claim_ids, groups=claim_ids)):
        held = set(claim_ids[te])
        model = factory().fit(train_b[~train_b["claim_id"].isin(held)], train_b.loc[~train_b["claim_id"].isin(held), "label"])
        oof.append(pipeline_predict(model, [train_claims_all[i] for i in te], train_scores[te], corpus, k))
    oof = pd.concat(oof, ignore_index=True)
    taus = np.round(np.arange(0.30, 0.991, 0.01), 2)
    tau_grid = []
    for t in taus:
        m = pipeline_metrics(oof, apply_threshold(oof, t), rng)
        tau_grid.append({"tau": float(t), "train_macro_f1": m["macro_f1"], "train_hallucinated": m["hallucinated_verdict_rate"]})
    tau = max(tau_grid, key=lambda g: g["train_macro_f1"])["tau"]
    # Strict operating point: lowest tau whose train hallucinated-verdict rate is <= 10%.
    tau_strict = min((g["tau"] for g in tau_grid if g["train_hallucinated"] <= 0.10), default=float(taus[-1]))
    print(f"Thresholds tuned on train (OOF): tau_f1={tau}  tau_strict={tau_strict}")

    model = factory().fit(train_b, train_b["label"])
    dev_scores = retriever.score([c["claim"] for c in dev_claims_all])
    dev_df = pipeline_predict(model, dev_claims_all, dev_scores, corpus, k)

    # Baseline without retrieval: classify the claim text alone at claim level.
    claim_only = Pipeline([("x", ColumnTransformer([("c", _tfidf(stop_words="english"), "claim")])),
                           ("clf", LogisticRegression(max_iter=4000, class_weight="balanced"))])
    tr_df = pd.DataFrame({"claim": [c["claim"] for c in train_claims_all], "label": [claim_gold_label(c) for c in train_claims_all]})
    claim_only.fit(tr_df, tr_df["label"])
    no_retrieval = claim_only.predict(pd.DataFrame({"claim": [c["claim"] for c in dev_claims_all]}))

    # Oracle: verifier sees the gold rationale (or the cited abstract for NEI claims).
    oracle = []
    for c in dev_claims_all:
        if c["evidence"]:
            ev = " ".join(corpus.sent_df["sentence"].values[corpus.key_to_idx[key]] for key in sorted(gold_evidence_keys(c)))
        else:
            ev = top_sentences(corpus.abstract[int(c["cited_doc_ids"][0])], c["claim"], retrievers["tfidf"].vec, k=2)
        oracle.append(model.predict(add_pair_columns(pd.DataFrame({"claim": [c["claim"]], "evidence": [ev]})))[0])
    oracle = np.array(oracle)

    systems = {
        "No retrieval (claim only)": no_retrieval,
        f"Retrieval top-{k}, no abstention": dev_df["argmax"].values,
        f"Retrieval top-{k} + abstention, F1-tuned (tau={tau})": apply_threshold(dev_df, tau),
        f"Retrieval top-{k} + strict abstention (tau={tau_strict})": apply_threshold(dev_df, tau_strict),
        "Oracle evidence (upper bound)": oracle,
    }
    results = {}
    for name, pred in systems.items():
        df = dev_df if "Oracle" not in name else dev_df.assign(cited_is_gold=True)
        if "No retrieval" in name:
            df = dev_df.assign(cited_is_gold=False)
        results[name] = pipeline_metrics(df, pred, rng)
    print(pd.DataFrame(results).T[["accuracy", "macro_f1", "strict_accuracy", "hallucinated_verdict_rate", "ungrounded_verdict_rate", "coverage"]].round(3).to_string())

    # Selective-prediction curve on dev (reporting only; tau was fixed on train)
    curve = []
    for t in np.round(np.arange(0.2, 0.991, 0.01), 2):
        p = apply_threshold(dev_df, t)
        m = pipeline_metrics(dev_df, p, rng)
        curve.append({"tau": float(t), **{key: m[key] for key in ("macro_f1", "hallucinated_verdict_rate", "coverage", "strict_accuracy")}})
    plot_pipeline(results, curve, tau_strict, FIGURES / "phase4_pipeline_comparison.png")

    dev_df["pred_final"] = apply_threshold(dev_df, tau_strict)
    dev_df["pred_no_retrieval"] = no_retrieval
    dev_df["pred_oracle"] = oracle
    dev_df.to_csv(RESULTS / "phase4_pipeline_dev_predictions.csv", index=False)
    return {"k": k, "tau_f1": tau, "tau_strict": tau_strict, "tau_grid_train": tau_grid, "systems_dev": results, "threshold_curve_dev": curve,
            "retrieval_topk_contains_gold": float(dev_df.loc[dev_df["gold"] != "NO_EVIDENCE", "cited_is_gold"].mean())}


def plot_pipeline(results: dict, curve: list[dict], tau: float, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw={"width_ratios": [1.3, 1]})
    names = list(results)
    short = ["No\nretrieval", "Retrieval", "Retrieval +\nF1 abstention", "Retrieval +\nstrict abstention", "Oracle\nevidence"]
    metrics = [("macro_f1", "Macro F1", "#2F6F8F"), ("strict_accuracy", "Strict (grounded) acc.", "#5E9E6E"),
               ("hallucinated_verdict_rate", "Hallucinated verdicts", "#C0504D")]
    x = np.arange(len(names))
    for i, (key, label, color) in enumerate(metrics):
        axes[0].bar(x + (i - 1) * 0.26, [results[n][key] for n in names], 0.26, label=label, color=color)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(short, fontsize=9)
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Claim-level pipeline on dev (300 claims)")
    axes[0].legend(frameon=False, fontsize=8)
    t = [c["tau"] for c in curve]
    axes[1].plot(t, [c["macro_f1"] for c in curve], marker="o", ms=2, color="#2F6F8F", label="Macro F1")
    axes[1].plot(t, [c["hallucinated_verdict_rate"] for c in curve], marker="o", ms=2, color="#C0504D", label="Hallucinated verdict rate")
    axes[1].plot(t, [c["coverage"] for c in curve], ls="--", color="#9AA5B1", label="Coverage (verdict given)")
    axes[1].axvline(tau, color="black", lw=0.8, ls=":")
    axes[1].text(tau - 0.01, 0.95, "strict tau\n(chosen on train)", fontsize=8, va="top", ha="right")
    axes[1].set_xlabel("Abstention threshold tau on P(evidence)")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Abstention trade-off (dev)")
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4) Error analysis and subgroup / bias checks
# ---------------------------------------------------------------------------


def run_error_analysis(ver: dict, rng) -> dict:
    print("\n" + "=" * 64)
    print("Phase 4D — Error analysis and subgroup checks")
    print("=" * 64)
    train_b, dev_b = ver["_frames_B"]
    model = ver["_factory"]().fit(train_b, train_b["label"])
    df = dev_b.copy()
    df["pred"] = model.predict(df)
    df["correct"] = df["pred"] == df["label"]
    df["claim_words"] = df["claim"].map(lambda c: len(tokens(c)))
    feats = pair_numeric_features(df)
    df["overlap"] = feats[:, 0]
    df["claim_negated"] = feats[:, 2] > 0
    df["evidence_negated"] = feats[:, 3] > 0
    df["has_number"] = df["claim"].str.contains(r"\d")

    def by(col, bins=None, labels=None):
        key = pd.cut(df[col], bins=bins, labels=labels) if bins is not None else df[col]
        g = df.groupby(key, observed=True)
        return [{"group": str(k), "n": int(len(v)), "accuracy": float(v["correct"].mean()),
                 "macro_f1": float(f1_score(v["label"], v["pred"], average="macro", labels=CLAIM_LABELS, zero_division=0))}
                for k, v in g]

    slices = {
        "claim_length": by("claim_words", [0, 10, 15, 20, 100], ["<=10", "11-15", "16-20", ">20"]),
        "lexical_overlap": by("overlap", [-0.01, 0.3, 0.5, 0.7, 1.0], ["<=0.3", "0.3-0.5", "0.5-0.7", ">0.7"]),
        "claim_negated": by("claim_negated"),
        "claim_has_number": by("has_number"),
    }

    # Shortcut check: how is CONTRADICT predicted when the claim contains a negation word?
    neg = df[df["claim_negated"]]
    shortcut = {
        "claims_with_negation": int(len(neg)),
        "share_gold_contradict": float((neg["label"] == "CONTRADICT").mean()) if len(neg) else 0.0,
        "share_pred_contradict": float((neg["pred"] == "CONTRADICT").mean()) if len(neg) else 0.0,
        "share_gold_contradict_overall": float((df["label"] == "CONTRADICT").mean()),
    }

    # Topic subgroups: KMeans on LSA-reduced claim TF-IDF (fit on train claims),
    # named by the top terms of each centroid. Raw sparse TF-IDF gave one giant cluster.
    claims_tr = train_b["claim"].drop_duplicates()
    lsa = make_pipeline(TfidfVectorizer(stop_words="english", min_df=2), TruncatedSVD(50, random_state=SEED), Normalizer())
    km = KMeans(n_clusters=6, random_state=SEED, n_init=10).fit(lsa.fit_transform(claims_tr))
    terms = np.array(lsa[0].get_feature_names_out())
    centers = lsa[1].inverse_transform(km.cluster_centers_)
    names = {i: ", ".join(terms[np.argsort(-centers[i])[:3]]) for i in range(6)}
    df["topic"] = [names[i] for i in km.predict(lsa.transform(df["claim"]))]
    topics = by("topic")
    accs = [t["accuracy"] for t in topics if t["n"] >= 20]
    topic_gap = float(max(accs) - min(accs)) if accs else 0.0

    # Most confident mistakes for the report
    proba = model.predict_proba(df)
    df["confidence"] = proba.max(axis=1)
    worst = df[~df["correct"]].sort_values("confidence", ascending=False).head(8)
    examples = worst[["claim_id", "label", "pred", "confidence", "claim", "evidence"]].assign(
        evidence=lambda d: d["evidence"].str.slice(0, 220)
    ).to_dict(orient="records")

    # Strongest features of the logistic regression per class (interpretability)
    clf = model.named_steps["clf"]
    fnames = model.named_steps["x"].get_feature_names_out()
    top_features = {}
    for i, lab in enumerate(clf.classes_):
        top_features[lab] = [str(fnames[j]) for j in np.argsort(-clf.coef_[i])[:12]]

    for name, rows in {**slices, "topic": topics}.items():
        print(f"\n{name}:")
        print(pd.DataFrame(rows).round(3).to_string(index=False))
    print("\nNegation shortcut check:", json.dumps(shortcut, indent=1))
    print("Topic accuracy gap (groups n>=20):", round(topic_gap, 3))

    plot_slices(slices, topics, FIGURES / "phase4_error_slices.png")
    df.drop(columns=["text", "claim_only_tokens"]).to_csv(RESULTS / "phase4_error_analysis_dev.csv", index=False)
    return {"slices": slices, "topic_subgroups": topics, "topic_accuracy_gap": topic_gap,
            "negation_shortcut": shortcut, "confident_errors": examples, "top_features": top_features}


def plot_slices(slices: dict, topics: list[dict], path: Path) -> None:
    panels = [("claim_length", "Claim length (words)"), ("lexical_overlap", "Claim–evidence word overlap"),
              ("claim_negated", "Claim contains negation")]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6), gridspec_kw={"width_ratios": [1, 1, 0.6, 1.6]})
    for ax, (key, title) in zip(axes, panels):
        rows = slices[key]
        ax.bar([r["group"] for r in rows], [r["macro_f1"] for r in rows], color="#2F6F8F")
        for i, r in enumerate(rows):
            ax.text(i, r["macro_f1"] + 0.02, f"n={r['n']}", ha="center", fontsize=7)
        ax.set_ylim(0, 1)
        ax.set_title(title, fontsize=10)
    axes[0].set_ylabel("Macro F1 (dev)")
    axes[3].barh([t["group"] for t in topics], [t["accuracy"] for t in topics], color="#D08C34")
    for i, t in enumerate(topics):
        axes[3].text(t["accuracy"] + 0.01, i, f"n={t['n']}", va="center", fontsize=7)
    axes[3].set_xlim(0, 1)
    axes[3].set_title("Accuracy by claim topic cluster", fontsize=10)
    axes[3].tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------


def _public(d: dict) -> dict:
    return {k: v for k, v in d.items() if not k.startswith("_")}


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    corpus = Corpus()
    train_all = load_jsonl(RAW_CLAIMS["train"])
    dev_all = load_jsonl(RAW_CLAIMS["dev"])
    train_ev = [c for c in train_all if gold_evidence_keys(c)]
    dev_ev = [c for c in dev_all if gold_evidence_keys(c)]

    retrieval = run_retrieval_eval(corpus, train_ev, dev_ev, rng)
    verifier = run_verifier_eval(rng)
    pipeline = run_pipeline_eval(corpus, retrieval["_retrievers"], retrieval["best_retriever"], verifier, train_all, dev_all, rng)
    errors = run_error_analysis(verifier, rng)

    report = {
        "phase": 4,
        "seed": SEED,
        "bootstrap_samples": N_BOOT,
        "retrieval": _public(retrieval),
        "verifier": _public(verifier),
        "pipeline": pipeline,
        "error_analysis": errors,
    }
    path = RESULTS / "phase4_evaluation.json"
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
