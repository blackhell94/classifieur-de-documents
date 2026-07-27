"""
Tests API FastAPI
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_health():
    """Test endpoint health"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data


def test_predict_text():
    """Test prédiction avec texte.

    Retourne 200 si des modèles entraînés sont présents,
    404 sinon (installation fraîche sans artefacts).
    """
    response = client.post(
        "/predict",
        data={"text": "This is an invoice for payment of $100."}
    )
    assert response.status_code in (200, 404)
    if response.status_code == 200:
        data = response.json()
        assert "predicted_class" in data
        assert "probabilities" in data


def test_predict_no_input():
    """Test erreur sans input"""
    response = client.post("/predict", data={})
    assert response.status_code == 400
