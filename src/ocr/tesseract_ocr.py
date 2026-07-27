"""
OCR avec Tesseract - Extraction de texte depuis images TIF
Produit un CSV avec les colonnes attendues par l'étape text_mining:
path, label, split, text (+ métadonnées OCR)
"""

import logging
import os
from typing import Any, Dict

import chardet
import cv2
import numpy as np
import pandas as pd
import pytesseract
from PIL import Image

from config.settings import PipelineConfig

logger = logging.getLogger("Pipeline")


def detect_encoding(file_path: str) -> str:
    """Détecte l'encodage d'un fichier CSV (sur les premiers 100 Ko)."""
    with open(file_path, "rb") as f:
        result = chardet.detect(f.read(100_000))
    return result["encoding"] or "utf-8"


def preprocess_image(image: Image.Image) -> Image.Image:
    """Prétraite l'image pour améliorer l'OCR."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    img_array = np.array(image)

    # Grayscale
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    # Binarisation adaptative
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )

    # Débruitage
    denoised = cv2.fastNlMeansDenoising(binary, None, 10, 7, 21)

    return Image.fromarray(denoised)


def ocr_single_image(image_path: str, config: PipelineConfig) -> Dict[str, Any]:
    """OCR sur une seule image."""
    try:
        if not os.path.exists(image_path):
            return {
                "text": "",
                "confidence": 0,
                "error": f"Fichier non trouvé: {image_path}",
                "word_count": 0,
            }

        image = Image.open(image_path)

        if image.mode != "RGB":
            image = image.convert("RGB")

        # Prétraitement si activé
        if config.ocr.preprocess:
            image = preprocess_image(image)

        # Configuration Tesseract
        if config.ocr.tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = config.ocr.tesseract_cmd

        # OCR
        config_tesseract = "--psm 6"
        text = pytesseract.image_to_string(
            image,
            lang=config.ocr.language,
            config=config_tesseract,
        )

        # Données de confiance
        data = pytesseract.image_to_data(
            image,
            lang=config.ocr.language,
            output_type=pytesseract.Output.DICT,
        )

        confidences = [c for c in data["conf"] if c > 0]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        word_count = len([w for w in data["text"] if w.strip()])

        return {
            "text": text,
            "confidence": round(avg_confidence, 2),
            "error": None,
            "word_count": word_count,
        }

    except Exception as e:
        return {
            "text": "",
            "confidence": 0,
            "error": str(e),
            "word_count": 0,
        }


def run_ocr(config: PipelineConfig) -> Dict[str, Any]:
    """Exécute l'OCR sur toutes les images listées dans le CSV d'entrée."""
    logger.info("=" * 70)
    logger.info("OCR - TESSERACT")
    logger.info("=" * 70)

    input_csv = config.data.raw_csv
    output_csv = config.data.ocr_csv

    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"CSV d'entrée non trouvé: {input_csv}")

    # Détecter encodage et lire CSV
    logger.info(f"Lecture: {input_csv}")
    detected_encoding = detect_encoding(input_csv)
    logger.info(f"Encodage détecté: {detected_encoding}")

    encodings_to_try = [detected_encoding, "utf-8", "latin-1", "cp1252", "iso-8859-1"]

    df = None
    for encoding in encodings_to_try:
        try:
            df = pd.read_csv(input_csv, sep=None, engine="python", encoding=encoding)
            logger.info(f"Fichier lu avec succès: {encoding}")
            break
        except UnicodeDecodeError:
            logger.warning(f"Échec encodage: {encoding}")
            continue

    if df is None:
        raise ValueError("Impossible de lire le fichier avec les encodages testés")

    logger.info(f"Documents à traiter: {len(df)}")

    # Déterminer la colonne des chemins
    path_column = "path" if "path" in df.columns else df.columns[0]
    if path_column != "path":
        df = df.rename(columns={path_column: "path"})

    # Vérifier les colonnes attendues en aval
    for col, default in (("label", -1), ("split", "unknown")):
        if col not in df.columns:
            logger.warning(f"Colonne '{col}' absente du CSV d'entrée, valeur par défaut: {default}")
            df[col] = default

    # Traiter chaque image
    texts, confidences, word_counts, errors = [], [], [], []
    n_success = 0

    for idx, row in df.iterrows():
        image_path = row["path"]
        logger.info(f"[{idx + 1}/{len(df)}] {image_path}")

        if pd.isna(image_path) or str(image_path).strip() == "":
            ocr_result = {"text": "", "confidence": 0, "word_count": 0, "error": "Chemin vide"}
        else:
            ocr_result = ocr_single_image(str(image_path), config)

        if ocr_result["error"]:
            logger.warning(f"  Erreur: {ocr_result['error']}")
        else:
            n_success += 1
            logger.info(
                f"  Confiance: {ocr_result['confidence']:.2f}% | Mots: {ocr_result['word_count']}"
            )

        texts.append(ocr_result["text"])
        confidences.append(ocr_result["confidence"])
        word_counts.append(ocr_result["word_count"])
        errors.append(ocr_result["error"])

    # Colonne 'text' = attendue par text_mining ; métadonnées préfixées ocr_
    df["text"] = texts
    df["ocr_confidence"] = confidences
    df["ocr_word_count"] = word_counts
    df["ocr_error"] = errors

    # Sauvegarder
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    logger.info(f"\n{'=' * 50}")
    logger.info("OCR terminé!")
    logger.info(f"Résultats: {output_csv}")
    logger.info(f"Total: {len(df)} | Succès: {n_success}")

    return {"output_csv": output_csv, "n_documents": len(df), "n_success": n_success}
