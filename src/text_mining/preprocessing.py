"""
Text Mining - Prétraitement, vectorisation et échantillonnage stratifié
Split fixe par indices, avec validation de cohérence des indices
"""

import json
import logging
import os
import re
from collections import Counter
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split

from config.settings import PipelineConfig
from constants import ID2LABEL, NUM_CLASSES

logger = logging.getLogger("Pipeline")


class TextPreprocessor:
    """Préprocesseur de texte OCRisé."""

    def __init__(self, language="english"):
        self.language = language
        self.stop_words = set(stopwords.words(language))
        self.stop_words.update({"ocr", "scan", "scanned", "page", "document", "file"})
        self.lemmatizer = WordNetLemmatizer()
        self.tokenizer = RegexpTokenizer(r"[a-zA-Z]{3,}")

    def clean_text(self, text):
        """Nettoie le texte OCRisé."""
        if pd.isna(text):
            return ""
        text = str(text).lower()
        text = re.sub(r"[^a-zA-Z\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def preprocess(self, text):
        """Pipeline complet: clean + tokenize + filter + lemmatize."""
        cleaned = self.clean_text(text)
        tokens = self.tokenizer.tokenize(cleaned)
        tokens = [t for t in tokens if t not in self.stop_words]
        tokens = [self.lemmatizer.lemmatize(t) for t in tokens]
        return " ".join(tokens)


def load_and_sample_data(config: PipelineConfig) -> pd.DataFrame:
    """Charge les données OCR et applique l'échantillonnage stratifié."""
    logger.info("Chargement et échantillonnage des données...")

    # Charger OCR (écrit en utf-8-sig par l'étape OCR)
    df = pd.read_csv(config.data.ocr_csv, encoding="utf-8-sig", sep=None, engine="python")

    required = ["path", "label", "split", "text"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Colonnes manquantes dans {config.data.ocr_csv}: {missing}. "
            f"Colonnes disponibles: {list(df.columns)}"
        )

    df = df[required].copy()
    df["text"] = df["text"].astype(str).str.strip()
    df["text_len"] = df["text"].str.len()

    logger.info(f"  Documents chargés: {len(df):,}")
    logger.info(f"  Distribution initiale: {df['label'].nunique()} classes")

    # Filtrer documents vides
    df = df[df["text_len"] > config.data.min_text_length].reset_index(drop=True)
    logger.info(f"  Après filtrage (>{config.data.min_text_length} chars): {len(df):,}")

    # Échantillonnage si demandé
    if 0 < config.data.sample_size < len(df):
        n_per_class = config.data.sample_size // NUM_CLASSES
        frames = []
        for lbl in range(NUM_CLASSES):
            sub = df[df["label"] == lbl]
            n = min(len(sub), n_per_class)
            if n > 0:
                frames.append(sub.sample(n, random_state=config.data.random_state))
        df = pd.concat(frames).reset_index(drop=True)
        logger.info(f"  Échantillon: {len(df):,} documents ({n_per_class}/classe)")
    else:
        logger.info(f"  Utilisation de tous les documents: {len(df):,}")

    return df


def preprocess_texts(df: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """Prétraite les textes."""
    logger.info("Prétraitement des textes...")

    preprocessor = TextPreprocessor()
    df["text_cleaned"] = df["text"].apply(preprocessor.preprocess)
    df = df[df["text_cleaned"].str.len() > 0].reset_index(drop=True)

    logger.info(f"  Documents après nettoyage: {len(df):,}")
    logger.info(f"  Exemple: {df['text_cleaned'].iloc[0][:200]}...")

    return df


def create_split_indices(df: pd.DataFrame, config: PipelineConfig) -> Tuple[np.ndarray, np.ndarray]:
    """Crée ou charge les indices de split fixe.

    Les indices sauvegardés ne sont réutilisés que si le DataFrame a la
    même taille que lors de leur création (fichier de métadonnées),
    sinon un nouveau split est généré pour éviter tout décalage.
    """
    train_idx_path = config.data.train_indices
    test_idx_path = config.data.test_indices
    meta_path = os.path.join(config.data.intermediate, "split_meta.json")

    # Réutiliser les indices existants s'ils sont cohérents avec le df actuel
    if os.path.exists(train_idx_path) and os.path.exists(test_idx_path) and os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if (
            meta.get("n_documents") == len(df)
            and meta.get("random_state") == config.data.random_state
            and meta.get("test_size") == config.data.test_size
        ):
            logger.info("Chargement des indices de split existants...")
            train_idx = np.load(train_idx_path)
            test_idx = np.load(test_idx_path)
            return train_idx, test_idx
        logger.warning(
            "Indices de split existants incompatibles avec les données actuelles "
            f"(attendu {meta.get('n_documents')} docs, trouvé {len(df)}). Régénération du split."
        )

    # Créer nouveau split
    logger.info("Création du split train/test...")
    train_idx, test_idx = train_test_split(
        np.arange(len(df)),
        test_size=config.data.test_size,
        stratify=df["label"],
        random_state=config.data.random_state,
    )

    # Sauvegarder indices + métadonnées de validation
    np.save(train_idx_path, train_idx)
    np.save(test_idx_path, test_idx)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "n_documents": len(df),
                "random_state": config.data.random_state,
                "test_size": config.data.test_size,
            },
            f,
            indent=2,
        )
    logger.info(f"  Train: {len(train_idx)} | Test: {len(test_idx)}")

    return train_idx, test_idx


def vectorize_texts(df_train: pd.DataFrame, df_test: pd.DataFrame, config: PipelineConfig) -> Tuple[Any, Any, Any]:
    """Vectorise les textes avec TF-IDF ou CountVectorizer."""
    logger.info(f"Vectorisation ({config.text_mining.vectorizer})...")

    if config.text_mining.vectorizer == "count":
        vectorizer = CountVectorizer(
            max_features=config.text_mining.max_features,
            ngram_range=tuple(config.text_mining.ngram_range),
            min_df=config.text_mining.min_df,
            max_df=config.text_mining.max_df,
        )
    else:
        vectorizer = TfidfVectorizer(
            max_features=config.text_mining.max_features,
            ngram_range=tuple(config.text_mining.ngram_range),
            min_df=config.text_mining.min_df,
            max_df=config.text_mining.max_df,
            sublinear_tf=config.text_mining.sublinear_tf,
        )

    X_train = vectorizer.fit_transform(df_train["text_cleaned"])
    X_test = vectorizer.transform(df_test["text_cleaned"])

    logger.info(f"  X_train: {X_train.shape}")
    logger.info(f"  X_test: {X_test.shape}")
    logger.info(f"  Features exemple: {list(vectorizer.get_feature_names_out()[:10])}")

    # Sauvegarder vectorizer
    os.makedirs(config.data.intermediate, exist_ok=True)
    vectorizer_path = os.path.join(config.data.intermediate, "tfidf_vectorizer.joblib")
    joblib.dump(vectorizer, vectorizer_path)
    logger.info(f"  Vectorizer sauvegardé: {vectorizer_path}")

    return X_train, X_test, vectorizer


def run_text_mining(config: PipelineConfig) -> Dict[str, Any]:
    """Exécute l'étape complète de text mining."""
    logger.info("=" * 70)
    logger.info("TEXT MINING")
    logger.info("=" * 70)

    # 1. Charger et échantillonner
    df = load_and_sample_data(config)

    # 2. Prétraiter
    df = preprocess_texts(df, config)

    # 3. Split fixe
    train_idx, test_idx = create_split_indices(df, config)

    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_test = df.iloc[test_idx].reset_index(drop=True)

    # 4. Vectoriser
    X_train, X_test, vectorizer = vectorize_texts(df_train, df_test, config)

    # 5. Sauvegarder
    sp.save_npz(config.data.X_train_tfidf, X_train)
    sp.save_npz(config.data.X_test_tfidf, X_test)
    np.save(config.data.y_train, df_train["label"].values)
    np.save(config.data.y_test, df_test["label"].values)
    df_train.to_csv(config.data.train_df, index=False)
    df_test.to_csv(config.data.test_df, index=False)

    # 6. Top mots par classe
    logger.info("\nTop 10 mots par classe:")
    for lbl in range(NUM_CLASSES):
        texts = df[df["label"] == lbl]["text_cleaned"]
        if len(texts) > 0:
            all_words = " ".join(texts).split()
            top10 = Counter(all_words).most_common(10)
            logger.info(f"  {ID2LABEL[lbl]:25s}: {[w for w, _ in top10]}")

    logger.info("Text mining terminé")

    return {
        "n_train": len(df_train),
        "n_test": len(df_test),
        "n_features": X_train.shape[1],
        "vectorizer": vectorizer,
    }
