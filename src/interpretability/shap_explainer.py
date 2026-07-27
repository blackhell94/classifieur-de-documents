"""
SHAP Explainer - Interprétabilité globale et locale
Supporte: sklearn (TF-IDF) et transformers (BERT, XLM-RoBERTa)
"""

import logging
import os
import warnings
from typing import Any, Dict, Optional

import joblib
import matplotlib

matplotlib.use("Agg")  # Backend non interactif (serveur / pipeline)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
import shap

from config.settings import PipelineConfig

logger = logging.getLogger("Pipeline")

# Réduire le bruit des warnings SHAP sans monkey-patcher de module privé
warnings.filterwarnings("ignore", module="shap")


class ShapExplainer:
    """Gère les explications SHAP pour tous les types de modèles."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.explainers = {}
        self.shap_values_cache = {}

    def _load_model(self, model_name: str):
        """Charge un modèle entraîné."""
        # Essayer sklearn d'abord
        sklearn_path = f"{self.config.output.sklearn_dir}/{model_name}.joblib"
        if os.path.exists(sklearn_path):
            return joblib.load(sklearn_path), "sklearn"

        # Essayer transformers (modèle final, avec repli sur le dossier racine)
        for candidate in (
            f"{self.config.output.transformers_dir}/{model_name}/final",
            f"{self.config.output.transformers_dir}/{model_name}",
        ):
            if os.path.exists(os.path.join(candidate, "config.json")):
                from transformers import AutoModelForSequenceClassification, AutoTokenizer

                model = AutoModelForSequenceClassification.from_pretrained(candidate)
                tokenizer = AutoTokenizer.from_pretrained(candidate)
                return (model, tokenizer), "transformers"

        raise FileNotFoundError(f"Modèle '{model_name}' non trouvé")

    def _load_vectorizer(self):
        """Charge le vectorizer TF-IDF sauvegardé par text_mining."""
        vectorizer_path = os.path.join(self.config.data.intermediate, "tfidf_vectorizer.joblib")
        if os.path.exists(vectorizer_path):
            return joblib.load(vectorizer_path)

        # Recréer le vectorizer si absent (repli)
        logger.warning("Vectorizer non trouvé, re-fit sur le train set...")
        from sklearn.feature_extraction.text import TfidfVectorizer

        vectorizer = TfidfVectorizer(
            max_features=self.config.text_mining.max_features,
            ngram_range=tuple(self.config.text_mining.ngram_range),
            min_df=self.config.text_mining.min_df,
            max_df=self.config.text_mining.max_df,
            sublinear_tf=self.config.text_mining.sublinear_tf,
        )
        train_df = pd.read_csv(self.config.data.train_df)
        vectorizer.fit(train_df["text_cleaned"])
        return vectorizer

    def _load_data(self, sample_size: Optional[int] = None):
        """Charge les données pour SHAP."""
        X_test = sp.load_npz(self.config.data.X_test_tfidf)
        y_test = np.load(self.config.data.y_test)

        # Échantillonner si demandé (reproductible)
        if sample_size and sample_size < X_test.shape[0]:
            rng = np.random.default_rng(self.config.data.random_state)
            indices = rng.choice(X_test.shape[0], sample_size, replace=False)
            X_test = X_test[indices]
            y_test = y_test[indices]

        return X_test, y_test

    def _make_sklearn_explainer(self, model, background: np.ndarray):
        """Choisit l'explainer SHAP le plus adapté au type de modèle sklearn.

        TreeExplainer pour les forêts, LinearExplainer pour les modèles
        linéaires : nettement plus rapides que l'explainer générique sur
        des milliers de features TF-IDF.
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import LinearSVC

        if isinstance(model, RandomForestClassifier):
            return shap.TreeExplainer(model)
        if isinstance(model, (LogisticRegression, LinearSVC)):
            return shap.LinearExplainer(model, background)
        return shap.Explainer(model, background)

    def explain_sklearn_global(self, model_name: str, sample_size: int = 500) -> Dict[str, Any]:
        """Explication SHAP globale pour modèle sklearn sur TF-IDF."""
        logger.info(f"SHAP global pour {model_name} (sklearn, {sample_size} docs)...")

        model, _ = self._load_model(model_name)
        X_test, _ = self._load_data(sample_size)

        # Convertir en dense pour SHAP
        X_test_dense = X_test.toarray()
        vectorizer = self._load_vectorizer()
        feature_names = vectorizer.get_feature_names_out()

        # Créer l'explainer adapté et calculer les valeurs SHAP
        explainer = self._make_sklearn_explainer(model, X_test_dense)
        shap_values = explainer(X_test_dense)

        # Sauvegarder visualisations
        output_dir = self.config.reports.comparison_dir
        os.makedirs(output_dir, exist_ok=True)
        plots = []

        # Summary plot (beeswarm) - global
        # NB: shap.summary_plot retourne None, on dessine puis on sauvegarde
        # via matplotlib.
        shap.summary_plot(
            shap_values.values if hasattr(shap_values, "values") else shap_values,
            X_test_dense,
            feature_names=feature_names,
            max_display=self.config.shap.max_display,
            show=False,
        )
        path = f"{output_dir}/shap_global_{model_name}_summary.png"
        plt.savefig(path, dpi=200, bbox_inches="tight")
        plt.close("all")
        plots.append(os.path.basename(path))
        logger.info(f"  Summary plot: {path}")

        # Bar plot (feature importance)
        shap.summary_plot(
            shap_values.values if hasattr(shap_values, "values") else shap_values,
            X_test_dense,
            feature_names=feature_names,
            plot_type="bar",
            max_display=self.config.shap.max_display,
            show=False,
        )
        path = f"{output_dir}/shap_global_{model_name}_bar.png"
        plt.savefig(path, dpi=200, bbox_inches="tight")
        plt.close("all")
        plots.append(os.path.basename(path))
        logger.info(f"  Bar plot: {path}")

        return {
            "model": model_name,
            "type": "sklearn_global",
            "sample_size": sample_size,
            "plots": plots,
        }

    def explain_transformers_global(self, model_name: str, sample_size: int = 100) -> Dict[str, Any]:
        """Explication SHAP globale pour transformers (BERT, XLM-RoBERTa).

        Plus lent, donc échantillon réduit (max 50 documents).
        """
        logger.info(f"SHAP global pour {model_name} (transformers, {sample_size} docs)...")

        (model, tokenizer), _ = self._load_model(model_name)

        # Charger quelques textes
        test_df = pd.read_csv(self.config.data.test_df)
        texts = test_df["text_cleaned"].astype(str).head(sample_size).tolist()
        texts = texts[: min(50, len(texts))]  # Limite pour la performance

        # Créer pipeline transformers
        import transformers

        pred = transformers.pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            top_k=None,  # remplace return_all_scores=True (déprécié)
            truncation=True,
            max_length=self.config.transformers.models[0].max_length,
        )

        # Explainer SHAP pour transformers
        explainer = shap.Explainer(pred)
        shap_values = explainer(texts)

        # Générer et ÉCRIRE le HTML sur disque
        output_dir = self.config.reports.comparison_dir
        os.makedirs(output_dir, exist_ok=True)
        html_body = shap.plots.text(shap_values, display=False)

        html_path = f"{output_dir}/shap_global_{model_name}.html"
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(f"<html><head><meta charset='utf-8'></head><body>{html_body}</body></html>")
        logger.info(f"  HTML sauvegardé: {html_path}")

        return {
            "model": model_name,
            "type": "transformers_global",
            "sample_size": len(texts),
            "html_path": html_path,
        }

    def explain_local(self, model_name: str, text: str, top_k: int = 5) -> Dict[str, Any]:
        """Explication SHAP locale pour un document spécifique."""
        logger.info(f"SHAP local pour {model_name}...")

        model, model_type = self._load_model(model_name)

        if model_type == "sklearn":
            return self._explain_local_sklearn(model, text, top_k)
        elif model_type == "transformers":
            return self._explain_local_transformers(model, text, top_k)
        else:
            raise ValueError(f"Type de modèle inconnu: {model_type}")

    def _explain_local_sklearn(self, model, text: str, top_k: int) -> Dict[str, Any]:
        """Explication locale pour sklearn (TF-IDF features)."""
        vectorizer = self._load_vectorizer()

        # Vectoriser le texte
        X = vectorizer.transform([text])
        X_dense = X.toarray()

        # Créer explainer adapté
        explainer = self._make_sklearn_explainer(model, X_dense)
        shap_values = explainer(X_dense)

        # Extraire top features
        feature_names = vectorizer.get_feature_names_out()
        values = shap_values.values[0]

        # Pour multiclass, prendre la classe prédite
        pred_class = int(model.predict(X)[0])
        if values.ndim > 1:
            values = values[:, pred_class]

        top_indices = np.argsort(np.abs(values))[-top_k:][::-1]

        return {
            "predicted_class": pred_class,
            "top_features": [
                {
                    "feature": str(feature_names[i]),
                    "shap_value": float(values[i]),
                    "impact": "positive" if values[i] > 0 else "negative",
                }
                for i in top_indices
            ],
        }

    def _explain_local_transformers(self, model_tokenizer, text: str, top_k: int) -> Dict[str, Any]:
        """Explication locale pour transformers (tokens)."""
        model, tokenizer = model_tokenizer

        import transformers

        pred = transformers.pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            top_k=None,
            truncation=True,
            max_length=self.config.transformers.models[0].max_length,
        )

        explainer = shap.Explainer(pred)
        shap_values = explainer([text])

        # Prédiction (un seul appel, réutilisé partout)
        pred_output = pred(text)
        scores = pred_output[0] if pred_output else []
        if scores:
            best = max(scores, key=lambda s: s["score"])
            predicted_label = best["label"]
            pred_class = max(range(len(scores)), key=lambda i: scores[i]["score"])
        else:
            predicted_label = "unknown"
            pred_class = 0

        # Extraire les tokens importants
        tokens = shap_values[0].data
        values = shap_values[0].values

        # values shape: (tokens, classes) pour multiclass
        if values.ndim > 1:
            class_values = values[:, pred_class]
        else:
            class_values = values

        n = min(len(tokens), len(class_values))
        top_indices = np.argsort(np.abs(class_values[:n]))[-top_k:][::-1]

        return {
            "predicted_class": predicted_label,
            "top_tokens": [
                {
                    "token": str(tokens[i]),
                    "shap_value": float(class_values[i]),
                    "impact": "positive" if class_values[i] > 0 else "negative",
                }
                for i in top_indices
            ],
        }


def run_shap_global(config: PipelineConfig) -> Dict[str, Any]:
    """Exécute SHAP global pour tous les modèles activés."""
    logger.info("=" * 70)
    logger.info("SHAP - EXPLICATIONS GLOBALES")
    logger.info("=" * 70)

    explainer = ShapExplainer(config)
    results = {}

    # SHAP pour modèles sklearn
    if config.steps.train_sklearn:
        for model_config in config.sklearn.models:
            try:
                result = explainer.explain_sklearn_global(
                    model_config.name,
                    sample_size=config.shap.global_sample_size,
                )
                results[model_config.name] = result
            except Exception as e:
                logger.error(f"Erreur SHAP pour {model_config.name}: {e}", exc_info=True)

    # SHAP pour transformers (échantillon plus petit)
    if config.steps.train_transformers:
        for model_config in config.transformers.models:
            try:
                result = explainer.explain_transformers_global(
                    model_config.name,
                    sample_size=min(100, config.shap.global_sample_size // 5),
                )
                results[model_config.name] = result
            except Exception as e:
                logger.error(f"Erreur SHAP pour {model_config.name}: {e}", exc_info=True)

    # Fichier sentinelle attendu par le cache de l'orchestrateur
    sentinel = f"{config.reports.comparison_dir}/shap_global.html"
    links = "".join(
        f"<li><a href='{os.path.basename(r['html_path'])}'>{name}</a></li>"
        for name, r in results.items()
        if isinstance(r, dict) and "html_path" in r
    )
    imgs = "".join(
        f"<h3>{name}</h3>" + "".join(f"<img src='{p}' style='max-width:100%'/>" for p in r["plots"])
        for name, r in results.items()
        if isinstance(r, dict) and "plots" in r
    )
    with open(sentinel, "w", encoding="utf-8") as f:
        f.write(
            "<html><head><meta charset='utf-8'><title>SHAP Global</title></head><body>"
            "<h1>Explications SHAP globales</h1>"
            f"<h2>Transformers</h2><ul>{links}</ul>"
            f"<h2>Sklearn</h2>{imgs}"
            "</body></html>"
        )
    logger.info(f"Index SHAP: {sentinel}")

    logger.info("SHAP global terminé")
    return results


def explain_single_document(config: PipelineConfig, model_name: str, text: str) -> Dict[str, Any]:
    """Explication locale pour un document (utilisé par l'API)."""
    explainer = ShapExplainer(config)
    return explainer.explain_local(model_name, text, top_k=config.shap.max_display)
