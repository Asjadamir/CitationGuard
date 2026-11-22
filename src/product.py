"""Interactive inference using the existing Phase 3/4 CPU models."""

from datetime import datetime, timezone
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

from phase3_pipeline import ROOT, add_pair_columns, build_verifier, prepare_verifier_frames
from phase4_evaluation import BM25Retriever, Corpus, HybridRetriever

THRESHOLD = 0.76
CORPUS_SHA256 = "b8d6c89624cb2ed74dee8938effc4f5d8bd2086887880af8110d64be4ceade62"
MAX_CLAIM_LENGTH = 2000


class InputError(ValueError):
    """An actionable invalid claim or document selection."""


def validate_input(payload: dict) -> tuple[str, int | None]:
    if not isinstance(payload, dict):
        raise InputError("Send a JSON object with a claim.")
    claim = payload.get("claim")
    if not isinstance(claim, str) or not 3 <= len(claim.strip()) <= MAX_CLAIM_LENGTH:
        raise InputError(f"Enter a claim between 3 and {MAX_CLAIM_LENGTH} characters.")
    if not any(character.isalnum() for character in claim):
        raise InputError("The claim must contain words or numbers.")
    document = payload.get("doc_id")
    if document is None or document == "":
        return claim.strip(), None
    if not isinstance(document, str) or not document.isascii() or not document.isdigit() or len(document) > 10:
        raise InputError("Use a numeric SciFact document ID (up to 10 digits), not a URL or DOI.")
    return claim.strip(), int(document)


def strict_verdict(probabilities: dict[str, float]) -> tuple[str, str]:
    """Use the Phase 4 train-selected threshold, not the largest class score."""
    if 1 - probabilities["NO_EVIDENCE"] < THRESHOLD:
        return "NO_EVIDENCE", "below_threshold"
    label = max(("SUPPORT", "CONTRADICT"), key=probabilities.get)
    return label, "model_verdict"


class ClaimVerifier:
    def __init__(self, corpus: Corpus, retriever: HybridRetriever, model, metadata: dict):
        self.corpus = corpus
        self.retriever = retriever
        self.model = model
        self.metadata = metadata

    def verify(self, claim: str, doc_id: int | None = None) -> dict:
        if doc_id is not None and doc_id not in self.corpus.doc_pos:
            raise InputError(f"SciFact document {doc_id} is not in this corpus. Check the ID or search all papers.")
        scores = self.retriever.score([claim])[0]
        candidates = np.arange(len(scores))
        if doc_id is not None:
            candidates = candidates[self.corpus.sent_df["doc_id"].to_numpy() == doc_id]
        # Never send arbitrary zero-score sentences to the verifier.
        candidates = candidates[scores[candidates] > 0]
        order = candidates[np.argsort(-scores[candidates], kind="stable")][:5]
        evidence = []
        for rank, index in enumerate(order, start=1):
            row = self.corpus.sent_df.iloc[index]
            evidence.append({
                "rank": rank,
                "doc_id": int(row.doc_id),
                "sentence_id": int(row.sent_id),
                "title": row.title,
                "text": row.sentence,
                "retrieval_score": float(scores[index]),
                "used_by_verifier": rank <= 2,
            })
        probabilities = None
        if evidence:
            frame = add_pair_columns(pd.DataFrame({
                "claim": [claim],
                "evidence": [" ".join(item["text"] for item in evidence[:2])],
            }))
            values = self.model.predict_proba(frame)[0]
            probabilities = {str(label): float(value) for label, value in zip(self.model.classes_, values)}
            label, reason = strict_verdict(probabilities)
        else:
            label, reason = "NO_EVIDENCE", "no_matches"
        return {
            "claim": claim,
            "scope": {"dataset": "SciFact", "doc_id": doc_id},
            "label": label,
            "reason": reason,
            "probabilities": probabilities,
            "evidence_score": None if probabilities is None else 1 - probabilities["NO_EVIDENCE"],
            "threshold": THRESHOLD,
            "evidence": evidence,
            "model": self.metadata,
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "limitations": [
                "Experimental lexical classifier; the verdict can be wrong. This is not a fact or medical advice.",
                "NO_EVIDENCE means insufficient evidence for a verdict, not that the claim is false.",
                "Only the fixed SciFact abstracts are searched, not the internet or full papers.",
                "Scores are uncalibrated model outputs, not probabilities that a claim is true.",
                "The verifier judges the top two sentences jointly, possibly from different papers; it does not label each citation.",
                "The strict threshold was evaluated on corpus-wide retrieval, not document-restricted searches.",
            ],
        }


def text_digest(paths) -> str:
    """SHA-256 of text files with CRLF normalized to LF, so the provenance
    digest is identical on Windows (git autocrlf) and Linux/macOS checkouts."""
    digest = hashlib.sha256()
    for path in paths:
        digest.update(Path(path).read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def load_verifier(data_dir: Path) -> ClaimVerifier:
    raw_corpus = data_dir / "corpus.jsonl"
    try:
        digest = hashlib.sha256(raw_corpus.read_bytes()).hexdigest()
    except OSError as error:
        raise RuntimeError("SciFact corpus is missing or unreadable. Run: python scripts/setup_data.py, then Retry setup.") from error
    if digest != CORPUS_SHA256:
        raise RuntimeError("SciFact corpus failed its integrity check. Reinstall with python scripts/setup_data.py, then Retry setup.")
    corpus = Corpus(raw_corpus)
    retriever = HybridRetriever(
        corpus,
        BM25Retriever(corpus.sent_df["indexed_text"].tolist(), k1=0.9, b=0.5),
        BM25Retriever(corpus.doc_text, k1=0.9, b=0.5),
        lam=2.0,
    )
    # Never load development labels to train the interactive model.
    (train,) = prepare_verifier_frames(splits=("train",), raw_corpus=raw_corpus)
    model = build_verifier().fit(train, train["label"])
    training_digest = text_digest(ROOT / "data/processed" / name for name in ("verification_train.csv", "no_evidence_train.csv"))
    metadata = {
        "name": "Phase 3 comparison-feature logistic regression + Phase 4 hybrid BM25",
        "version": "citationguard-local-v1",
        "corpus_sha256": digest,
        "training_sha256": training_digest,
        "training_rows": len(train),
        "documents": len(corpus.docs),
        "sentences": len(corpus.sent_df),
        "sklearn_version": sklearn.__version__,
        "retrieval": {"k1": 0.9, "b": 0.5, "lambda": 2.0, "verifier_sentences": 2},
    }
    return ClaimVerifier(corpus, retriever, model, metadata)
