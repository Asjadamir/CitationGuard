"""Real SciFact CPU model, no classifier or retrieval mocks.

Oracles: official SciFact dev annotations for claims 1, 53, 248 and 249. These are
smoke/regression examples, not an independent accuracy evaluation of the model.
They catch broken training, swapped class order and disconnected retrieval.
"""

import math

import pytest

from app import create_app
from phase3_pipeline import RAW_CORPUS
from product import load_verifier


@pytest.fixture(scope="module")
def real_verifier():
    if not RAW_CORPUS.exists():
        pytest.skip("Install public SciFact first: python scripts/setup_data.py")
    return load_verifier(RAW_CORPUS.parent)


@pytest.mark.integration
@pytest.mark.parametrize("claim,expected,document", [
    ("ALDH1 expression is associated with poorer prognosis in breast cancer.", "SUPPORT", 45638119),
    ("Chenodeosycholic acid treatment increases whole-body energy expenditure.", "SUPPORT", 1568684),
    ("Chenodeosycholic acid treatment reduces whole-body energy expenditure.", "CONTRADICT", 1568684),
])
def test_real_claim_to_grounded_verdict(real_verifier, claim, expected, document):
    # Use the public HTTP boundary with the real fitted model and index.
    app = create_app(start_model=False)
    state = app.extensions["model_state"]
    state.verifier, state.state = real_verifier, "ready"
    response = app.test_client().post("/api/verify", json={"claim": claim})
    assert response.status_code == 200
    report = response.json
    assert report["label"] == expected
    assert any(item["doc_id"] == document and item["used_by_verifier"] for item in report["evidence"])
    assert math.isclose(sum(report["probabilities"].values()), 1, abs_tol=1e-8)
    assert all(0 <= value <= 1 for value in report["probabilities"].values())
    assert report["model"]["training_rows"] == 1312
    assert report["model"]["documents"] == 5183
    assert report["model"]["sentences"] == 45952


@pytest.mark.integration
def test_real_unknown_vocabulary_and_document_scope(real_verifier):
    # Official dev claim 1 has no gold evidence; relevant retrieval is not proof.
    report = real_verifier.verify("0-dimensional biomaterials show inductive properties.")
    assert report["label"] == "NO_EVIDENCE"
    assert report["reason"] == "below_threshold"
    assert report["evidence"]
    assert report["probabilities"] is not None
    report = real_verifier.verify("Qzxvplmn qzzxrrt.")
    assert report["label"] == "NO_EVIDENCE"
    assert report["evidence"] == []
    assert report["probabilities"] is None
    report = real_verifier.verify("Microstructural development of newborn cerebral white matter.", 4983)
    assert report["evidence"]
    assert {item["doc_id"] for item in report["evidence"]} == {4983}
    assert report["evidence"][0]["title"] == "Microstructural development of human newborn cerebral white matter assessed in vivo by diffusion tensor magnetic resonance imaging."
