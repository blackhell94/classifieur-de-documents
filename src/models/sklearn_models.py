"""
Entraînement des modèles sklearn - NB, LR, RF, SVM
Même train/test que les transformers
"""

import json
import logging
import os
import time
from typing import Any, Dict

import joblib
import numpy as np
import scipy.sparse as sp
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from config.settings import PipelineConfig, SklearnModelConfig
from constants import LABEL_NAMES

logger = logging.getLogger("Pipeline")


def get_model_class(class_name: str):
    """Retourne la classe de modèle sklearn."""
    models = {
        "MultinomialNB": MultinomialNB,
        "LogisticRegression": LogisticRegression,
        "RandomForestClassifier": RandomForestClassifier,
        "LinearSVC": LinearSVC,
    }
    return models.get(class_name)


def train_single_model(
    model_config: SklearnModelConfig,
    X_train, X_test,
    y_train, y_test,
    config: PipelineConfig,
) -> Dict[str, Any]:
    """Entraîne un seul modèle sklearn et retourne les métriques."""

    logger.info(f"\n{'=' * 55}")
    logger.info(f"  {model_config.name}")
    logger.info(f"{'=' * 55}")

    # Instancier le modèle
    model_class = get_model_class(model_config.class_name)
    if model_class is None:
        raise ValueError(f"Classe de modèle inconnue: {model_config.class_name}")

    model = model_class(**model_config.params)

    # Note: TF-IDF produit déjà des valeurs non négatives, aucun scaling
    # n'est nécessaire pour MultinomialNB. On entraîne tous les modèles sur
    # exactement la même matrice pour garantir la cohérence train/inférence.

    # Entraînement
    t0 = time.time()
    model.fit(X_train, y_train)
    duration = time.time() - t0

    # Prédictions
    y_pred_train = model.predict(X_train)
    y_pred = model.predict(X_test)

    # Métriques
    metrics = {
        "accuracy_train": accuracy_score(y_train, y_pred_train),
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "duration_s": round(duration, 1),
    }

    logger.info(f"  Durée: {duration:.1f}s")
    logger.info(f"  Accuracy train: {metrics['accuracy_train']:.4f} | test: {metrics['accuracy']:.4f}")
    logger.info(f"  F1-macro: {metrics['f1_macro']:.4f}")
    logger.info(f"  F1-weighted: {metrics['f1_weighted']:.4f}")
    logger.info("\n  Classification report:")
    logger.info(classification_report(y_test, y_pred, target_names=LABEL_NAMES, zero_division=0))

    # Sauvegarder le modèle
    output_path = f"{config.output.sklearn_dir}/{model_config.name}.joblib"
    joblib.dump(model, output_path)
    logger.info(f"  Modèle sauvegardé: {output_path}")

    # Sauvegarder prédictions si demandé
    if config.evaluation.save_predictions:
        pred_path = f"{config.reports.comparison_dir}/pred_{model_config.name}.npy"
        np.save(pred_path, y_pred)

    return {
        "model": model,
        "metrics": metrics,
        "y_pred": y_pred,
    }


def train_sklearn_models(config: PipelineConfig) -> Dict[str, Any]:
    """Entraîne tous les modèles sklearn."""
    logger.info("=" * 70)
    logger.info("MODÈLES SKLEARN")
    logger.info("=" * 70)

    # Charger données
    X_train = sp.load_npz(config.data.X_train_tfidf)
    X_test = sp.load_npz(config.data.X_test_tfidf)
    y_train = np.load(config.data.y_train)
    y_test = np.load(config.data.y_test)

    logger.info(f"X_train: {X_train.shape} | X_test: {X_test.shape}")

    # Entraîner chaque modèle
    results = {}
    new_metrics = {}

    for model_config in config.sklearn.models:
        try:
            result = train_single_model(
                model_config, X_train, X_test, y_train, y_test, config
            )
            results[model_config.name] = result
            new_metrics[model_config.name] = result["metrics"]
        except Exception as e:
            logger.error(f"Erreur entraînement {model_config.name}: {e}", exc_info=True)

    # Fusionner avec les métriques existantes (transformers).
    # Les métriques fraîchement calculées ÉCRASENT les anciennes valeurs
    # pour les mêmes modèles (et non l'inverse).
    metrics_path = f"{config.reports.comparison_dir}/metrics.json"
    all_metrics = {}
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            all_metrics = json.load(f)
    all_metrics.update(new_metrics)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)

    logger.info("Modèles sklearn entraînés")
    return results
