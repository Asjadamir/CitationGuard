import json

import pytest

from phase4_evaluation import BM25Retriever, Corpus, HybridRetriever
from product import ClaimVerifier


class FixedClassifier:
    """Boundary double with explicit class order and test scores."""

    classes_ = ["CONTRADICT", "NO_EVIDENCE", "SUPPORT"]

    def __init__(self, probabilities=(0.10, 0.05, 0.85)):
        self.probabilities = probabilities
        self.frames = []

    def predict_proba(self, frame):
        self.frames.append(frame.copy())
        return [self.probabilities]


@pytest.fixture
def tiny_corpus(tmp_path):
    # Independent source records: sentence 0 is blank to catch renumbering bugs.
    docs = [
        {"doc_id": 17, "title": "Copper study", "abstract": ["", "Copper reduces fever.", "Copper harms fish."]},
        {"doc_id": 29, "title": "Lunar observations", "abstract": ["The moon has craters."]},
    ]
    path = tmp_path / "corpus.jsonl"
    path.write_text("\n".join(json.dumps(doc) for doc in docs))
    return Corpus(path)


@pytest.fixture
def tiny_verifier(tiny_corpus):
    retriever = HybridRetriever(
        tiny_corpus,
        BM25Retriever(tiny_corpus.sent_df["indexed_text"].tolist(), k1=0.9, b=0.5),
        BM25Retriever(tiny_corpus.doc_text, k1=0.9, b=0.5),
        lam=2.0,
    )
    return ClaimVerifier(tiny_corpus, retriever, FixedClassifier(), {"documents": 2})
