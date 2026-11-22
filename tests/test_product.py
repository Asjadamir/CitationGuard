"""Behavioral oracles: Phase 4 operating rule and literal source records.

These reject argmax-only decisions, zero-match citations, document-filter leaks,
renumbered sentence IDs, truncated evidence and fabricated model scores.
"""

import numpy as np
import pytest

from phase4_evaluation import BM25Retriever
from phase3_pipeline import ROOT
from product import InputError, load_verifier, strict_verdict, text_digest, validate_input

# SHA-256 of the committed (LF) training CSV blobs, from `git cat-file blob`.
COMMITTED_TRAINING_SHA256 = "646847fe52d60a93b1e7e09df020b6471c6c07a41288e84700951ebd444d0679"


def test_text_digest_ignores_line_endings(tmp_path):
    (tmp_path / "lf.csv").write_bytes(b"a,b\n1,2\n")
    (tmp_path / "crlf.csv").write_bytes(b"a,b\r\n1,2\r\n")
    assert text_digest([tmp_path / "lf.csv"]) == text_digest([tmp_path / "crlf.csv"])
    assert text_digest([tmp_path / "lf.csv"]) != text_digest([tmp_path / "lf.csv", tmp_path / "lf.csv"])


def test_training_digest_matches_committed_data_on_any_checkout():
    files = [ROOT / "data/processed" / name for name in ("verification_train.csv", "no_evidence_train.csv")]
    assert text_digest(files) == COMMITTED_TRAINING_SHA256


@pytest.mark.parametrize("payload", [None, [], {}, {"claim": " "}, {"claim": 42}, {"claim": "xx"}, {"claim": "x" * 2001}, {"claim": "!!!"}])
def test_invalid_claims(payload):
    with pytest.raises(InputError):
        validate_input(payload)


@pytest.mark.parametrize("doc_id", [42, True, [], "https://example.org", "-1", "١٧", "1" * 11])
def test_invalid_document_ids(doc_id):
    with pytest.raises(InputError):
        validate_input({"claim": "Copper reduces fever.", "doc_id": doc_id})


def test_input_normalization_and_boundary():
    assert validate_input({"claim": "  Copper reduces fever.  ", "doc_id": "17"}) == ("Copper reduces fever.", 17)
    assert validate_input({"claim": "x" * 2000, "doc_id": ""}) == ("x" * 2000, None)


@pytest.mark.parametrize("probabilities,expected", [
    ({"SUPPORT": .60, "CONTRADICT": .10, "NO_EVIDENCE": .30}, ("NO_EVIDENCE", "below_threshold")),
    ({"SUPPORT": .55, "CONTRADICT": .21, "NO_EVIDENCE": .24}, ("SUPPORT", "model_verdict")),
    ({"SUPPORT": .10, "CONTRADICT": .85, "NO_EVIDENCE": .05}, ("CONTRADICT", "model_verdict")),
])
def test_strict_threshold_not_argmax(probabilities, expected):
    # Report defines P(S)+P(C) >= .76; below threshold abstains even if S wins.
    assert strict_verdict(probabilities) == expected


def test_exact_source_provenance_and_joined_verifier_input(tiny_verifier):
    report = tiny_verifier.verify("Copper reduces fever.", 17)
    assert report["label"] == "SUPPORT"
    assert report["probabilities"] == {"CONTRADICT": .10, "NO_EVIDENCE": .05, "SUPPORT": .85}
    assert report["evidence"][0]["doc_id"] == 17
    assert report["evidence"][0]["sentence_id"] == 1
    assert report["evidence"][0]["text"] == "Copper reduces fever."
    assert report["evidence"][1]["sentence_id"] == 2
    assert all(item["title"] == "Copper study" for item in report["evidence"])
    assert all(item["used_by_verifier"] for item in report["evidence"])
    assert tiny_verifier.model.frames[0].iloc[0]["evidence"] == "Copper reduces fever. Copper harms fish."


def test_no_matches_abstains_without_calling_classifier(tiny_verifier):
    report = tiny_verifier.verify("Zyzzyva quuxxyz.")
    assert report["label"] == "NO_EVIDENCE"
    assert report["reason"] == "no_matches"
    assert report["evidence"] == []
    assert report["probabilities"] is None
    assert report["evidence_score"] is None
    assert tiny_verifier.model.frames == []


def test_document_filter_does_not_leak_other_papers(tiny_verifier):
    report = tiny_verifier.verify("Copper reduces fever.", 29)
    assert report["scope"] == {"dataset": "SciFact", "doc_id": 29}
    assert report["reason"] == "no_matches"
    assert report["evidence"] == []
    with pytest.raises(InputError, match="not in this corpus"):
        tiny_verifier.verify("Copper reduces fever.", 999)


def test_equal_scores_have_stable_source_order(tiny_verifier):
    tiny_verifier.retriever.score = lambda queries: np.array([[1., 1., 1.]])
    report = tiny_verifier.verify("a query")
    assert [(item["doc_id"], item["sentence_id"]) for item in report["evidence"]] == [(17, 1), (17, 2), (29, 0)]
    assert [item["used_by_verifier"] for item in report["evidence"]] == [True, True, False]


def test_bm25_matches_literal_document_not_unrelated_text():
    # Exact topical match must outrank a document with no query words.
    scores = BM25Retriever(["copper fever", "lunar craters"]).score(["copper fever"])[0]
    assert scores[0] > 0
    assert scores[1] == 0


def test_missing_and_tampered_corpus_fail_honestly(tmp_path):
    with pytest.raises(RuntimeError, match="missing or unreadable"):
        load_verifier(tmp_path)
    (tmp_path / "corpus.jsonl").write_text('{}\n')
    with pytest.raises(RuntimeError, match="integrity check"):
        load_verifier(tmp_path)
