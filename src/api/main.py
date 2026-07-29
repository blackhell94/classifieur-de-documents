"""
API FastAPI - Prédiction de documents + SHAP local
"""

import io
import json
import logging
import os
import sys
import time
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# Ajouter src au path (permet `python src/api/main.py` depuis la racine)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import PipelineConfig
from constants import ID2LABEL, NUM_CLASSES
from interpretability.shap_explainer import explain_single_document

# Configuration logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("API")

# Charger la configuration (chemin résolu par rapport à la racine du projet
# si la variable d'environnement n'est pas définie)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = os.getenv("PIPELINE_CONFIG", str(_PROJECT_ROOT / "config.yaml"))
config = PipelineConfig.from_yaml(CONFIG_PATH)

app = FastAPI(
    title="Document Classification API",
    description="Classification de documents en 16 catégories avec interprétabilité SHAP",
    version="1.0.0",
)


class PredictionResponse(BaseModel):
    predicted_class: str
    predicted_label_id: int
    probabilities: Dict[str, float]
    model_used: str
    shap_explanation: Optional[Dict[str, Any]] = None
    ocr_text: Optional[str] = None
    processing_time_ms: float


class HealthResponse(BaseModel):
    status: str
    models_available: List[str]
    best_model: Optional[str]
    timestamp: str


class BatchRequest(BaseModel):
    texts: List[str]
    model_name: str = "best"


# ---------------------------------------------------------------------------
# Chargement des modèles (avec cache pour éviter un rechargement par requête)
# ---------------------------------------------------------------------------

def _get_best_model() -> str:
    """Récupère le meilleur modèle depuis le fichier JSON."""
    best_path = f"{config.output.models_dir}/best_model.json"
    if os.path.exists(best_path):
        with open(best_path, "r", encoding="utf-8") as f:
            info = json.load(f)
        return info.get("best_model", "logistic_regression")
    return "logistic_regression"


@lru_cache(maxsize=8)
def _load_sklearn_model(model_name: str):
    """Charge un modèle sklearn (mis en cache)."""
    path = f"{config.output.sklearn_dir}/{model_name}.joblib"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Modèle sklearn non trouvé: {path}")
    return joblib.load(path)


@lru_cache(maxsize=4)
def _load_transformer_model(model_name: str):
    """Charge un modèle transformer (mis en cache)."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    path = f"{config.output.transformers_dir}/{model_name}/final"
    if not os.path.exists(path):
        # Compatibilité avec un modèle sauvegardé directement dans le dossier
        path = f"{config.output.transformers_dir}/{model_name}"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Modèle transformer non trouvé: {path}")
    model = AutoModelForSequenceClassification.from_pretrained(path)
    tokenizer = AutoTokenizer.from_pretrained(path)
    model.eval()
    return model, tokenizer


@lru_cache(maxsize=1)
def _load_vectorizer():
    """Charge le vectorizer TF-IDF (mis en cache)."""
    vectorizer_path = os.path.join(config.data.intermediate, "tfidf_vectorizer.joblib")
    if not os.path.exists(vectorizer_path):
        raise FileNotFoundError(
            "Vectorizer TF-IDF non trouvé. Exécutez l'étape text_mining d'abord."
        )
    return joblib.load(vectorizer_path)


# ---------------------------------------------------------------------------
# Prédiction
# ---------------------------------------------------------------------------

def _predict_sklearn(model, text: str) -> Dict[str, Any]:
    """Prédiction avec modèle sklearn."""
    vectorizer = _load_vectorizer()
    X = vectorizer.transform([text])

    pred_id = int(model.predict(X)[0])

    # Probabilités (si disponible)
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[0]
    else:
        # Pour SVM sans proba, retourner one-hot
        probs = np.zeros(NUM_CLASSES)
        probs[pred_id] = 1.0

    probabilities = {ID2LABEL[i]: float(probs[i]) for i in range(NUM_CLASSES)}

    return {
        "predicted_class": ID2LABEL.get(pred_id, "unknown"),
        "predicted_label_id": pred_id,
        "probabilities": probabilities,
    }


def _predict_transformers(model_tokenizer, text: str) -> Dict[str, Any]:
    """Prédiction avec modèle transformer."""
    import torch
    import torch.nn.functional as F

    model, tokenizer = model_tokenizer

    inputs = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=config.transformers.models[0].max_length,
        return_tensors="pt",
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)[0].cpu().numpy()

    pred_id = int(np.argmax(probs))
    probabilities = {ID2LABEL.get(i, f"class_{i}"): float(probs[i]) for i in range(len(probs))}

    return {
        "predicted_class": ID2LABEL.get(pred_id, "unknown"),
        "predicted_label_id": pred_id,
        "probabilities": probabilities,
    }


def _run_prediction(input_text: str, model_name: str, explain: bool,
                    ocr_text: Optional[str], start_time: float) -> PredictionResponse:
    """Logique de prédiction partagée par /predict et /predict/batch."""
    if not input_text or len(input_text.strip()) < 10:
        raise HTTPException(400, "Texte trop court ou vide après OCR")

    # Déterminer le modèle
    if model_name == "best":
        model_name = _get_best_model()

    # Prédiction
    try:
        # Essayer sklearn d'abord
        try:
            model = _load_sklearn_model(model_name)
            result = _predict_sklearn(model, input_text)
        except FileNotFoundError:
            # Essayer transformers
            model_tokenizer = _load_transformer_model(model_name)
            result = _predict_transformers(model_tokenizer, input_text)
    except HTTPException:
        raise
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        logger.error(f"Erreur prédiction: {e}", exc_info=True)
        raise HTTPException(500, f"Erreur prédiction: {str(e)}")

    # SHAP local si demandé
    shap_exp = None
    if explain and config.api.enable_shap_local:
        try:
            shap_exp = explain_single_document(config, model_name, input_text)
        except Exception as e:
            logger.warning(f"SHAP local échoué: {e}")
            shap_exp = {"error": str(e)}

    elapsed = (time.time() - start_time) * 1000

    return PredictionResponse(
        predicted_class=result["predicted_class"],
        predicted_label_id=result["predicted_label_id"],
        probabilities=result["probabilities"],
        model_used=model_name,
        shap_explanation=shap_exp,
        ocr_text=ocr_text,
        processing_time_ms=elapsed,
    )


def _ocr_image(image_bytes: bytes) -> str:
    """OCR sur une image avec Tesseract."""
    import cv2
    import pytesseract
    from PIL import Image

    image = Image.open(io.BytesIO(image_bytes))

    # Prétraitement si configuré
    if config.ocr.preprocess:
        img_array = np.array(image)
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        denoised = cv2.fastNlMeansDenoising(binary, None, 10, 7, 21)
        image = Image.fromarray(denoised)

    if config.ocr.tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = config.ocr.tesseract_cmd

    text = pytesseract.image_to_string(image, lang=config.ocr.language)
    return text.strip()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def root():
    """Page d'accueil de l'API."""
    return """
    <html>
        <head><title>Document Classification API</title></head>
        <body>
            <h1>Document Classification API</h1>
            <p>Classification de documents en 16 catégories</p>
            <ul>
                <li><a href="/docs">Documentation API (Swagger)</a></li>
                <li><a href="/health">État de santé</a></li>
            </ul>
        </body>
    </html>
    """


def _available_models() -> List[str]:
    """Liste les modèles entraînés présents sur disque."""
    models = []

    for m in config.sklearn.models:
        path = f"{config.output.sklearn_dir}/{m.name}.joblib"
        if os.path.exists(path):
            models.append(m.name)

    for m in config.transformers.models:
        path = f"{config.output.transformers_dir}/{m.name}"
        if os.path.exists(path):
            models.append(m.name)

    return models


@app.get("/health", response_model=HealthResponse)
async def health():
    """Vérifie l'état de l'API et les modèles disponibles."""
    models = _available_models()
    best = _get_best_model() if models else None

    return HealthResponse(
        status="healthy" if models else "no_models",
        models_available=models,
        best_model=best,
        timestamp=datetime.now().isoformat(),
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None),
    model_name: str = Form("best"),
    explain: bool = Form(False),
):
    """
    Prédit la classe d'un document.

    - **file**: Image du document (TIF, JPG, PNG)
    - **text**: Texte déjà OCRisé (alternative au file)
    - **model_name**: Nom du modèle à utiliser ("best" pour auto)
    - **explain**: Activer SHAP local
    """
    start_time = time.time()

    # Vérifier entrée
    if file is None and text is None:
        raise HTTPException(400, "Fournissez 'file' (image) ou 'text' (texte OCRisé)")

    # OCR si image fournie
    ocr_text = None
    if file is not None:
        content = await file.read()
        if len(content) > config.api.max_file_size:
            raise HTTPException(413, f"Fichier trop grand (> {config.api.max_file_size} bytes)")
        try:
            ocr_text = _ocr_image(content)
        except Exception as e:
            raise HTTPException(422, f"Erreur OCR: {e}")
        input_text = ocr_text
    else:
        input_text = text

    return _run_prediction(input_text, model_name, explain, ocr_text, start_time)


@app.post("/predict/batch")
async def predict_batch(request: BatchRequest):
    """Prédiction batch sur plusieurs textes."""
    results = []
    for txt in request.texts:
        try:
            result = _run_prediction(
                input_text=txt,
                model_name=request.model_name,
                explain=False,
                ocr_text=None,
                start_time=time.time(),
            )
            results.append(result)
        except HTTPException as e:
            results.append({"error": e.detail, "text": txt[:100]})
        except Exception as e:
            results.append({"error": str(e), "text": txt[:100]})

    return {"predictions": results}


@app.get("/models")
async def list_models():
    """Liste tous les modèles disponibles."""
    models = _available_models()
    return {
        "models": models,
        "best_model": _get_best_model() if models else None,
    }


if __name__ == "__main__":
    uvicorn.run(
        app,
        host=config.api.host,
        port=config.api.port,
        log_level="info",
    )
