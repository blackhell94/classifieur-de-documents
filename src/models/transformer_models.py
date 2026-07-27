"""
Entraînement des modèles Transformers - BERT + XLM-RoBERTa
Même train/test que sklearn (indices fixes)
"""

import json
import logging
import os
import time
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from config.settings import PipelineConfig, TransformerModelConfig
from constants import ID2LABEL, LABEL2ID, LABEL_NAMES, NUM_CLASSES

logger = logging.getLogger("Pipeline")


def compute_metrics(eval_pred):
    """Fonction de métriques pour HuggingFace Trainer."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1_macro": f1_score(labels, predictions, average="macro", zero_division=0),
        "f1_weighted": f1_score(labels, predictions, average="weighted", zero_division=0),
    }


def train_single_transformer(
    model_config: TransformerModelConfig,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: PipelineConfig,
) -> Dict[str, Any]:
    """Entraîne un seul modèle transformer."""

    logger.info(f"\n{'=' * 55}")
    logger.info(f"  {model_config.name.upper()} - {model_config.model_name}")
    logger.info(f"{'=' * 55}")

    # Vérifier GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"  Device: {device.upper()}")

    if device == "cuda":
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        torch.cuda.empty_cache()

    # Tokenizer
    logger.info("  Chargement tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name)

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=model_config.max_length,
        )

    # Créer datasets HuggingFace
    train_ds = Dataset.from_pandas(
        train_df[["text", "label"]].rename(columns={"label": "labels"}),
        preserve_index=False,
    )
    test_ds = Dataset.from_pandas(
        test_df[["text", "label"]].rename(columns={"label": "labels"}),
        preserve_index=False,
    )

    logger.info("  Tokenisation...")
    train_ds = train_ds.map(tokenize_function, batched=True)
    test_ds = test_ds.map(tokenize_function, batched=True)

    train_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    test_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    # Modèle
    logger.info("  Chargement modèle...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_config.model_name,
        num_labels=NUM_CLASSES,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )
    model.to(device)

    # Training arguments
    output_dir = f"{config.output.transformers_dir}/{model_config.name}"
    has_eval = len(test_ds) > 0

    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=model_config.learning_rate,
        per_device_train_batch_size=model_config.batch_size,
        per_device_eval_batch_size=model_config.batch_size * 2,
        gradient_accumulation_steps=model_config.gradient_accumulation,
        num_train_epochs=model_config.epochs,
        weight_decay=model_config.weight_decay,
        warmup_ratio=model_config.warmup_ratio,
        seed=config.data.random_state,
        fp16=model_config.fp16 and (device == "cuda"),
        report_to="none",
        save_strategy="epoch",
        eval_strategy="epoch" if has_eval else "no",
        load_best_model_at_end=has_eval,
        metric_for_best_model="f1_weighted",
        logging_dir=f"{output_dir}/logs",
        logging_steps=50,
        dataloader_num_workers=0,  # Windows compatibility
        optim="adamw_torch",
    )

    # Trainer
    # NB: depuis transformers 4.46, l'argument `tokenizer` est remplacé
    # par `processing_class`.
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds if has_eval else None,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    # Entraînement
    logger.info(f"  Entraînement ({model_config.epochs} epochs)...")
    t0 = time.time()
    trainer.train()
    duration = time.time() - t0

    # Évaluation
    logger.info("  Évaluation...")
    predictions = trainer.predict(test_ds)
    y_pred = np.argmax(predictions.predictions, axis=-1)
    y_true = predictions.label_ids

    # Métriques
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "duration_s": round(duration, 1),
    }

    logger.info(f"  Durée: {duration:.1f}s")
    logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"  F1-macro: {metrics['f1_macro']:.4f}")
    logger.info(f"  F1-weighted: {metrics['f1_weighted']:.4f}")
    logger.info(classification_report(y_true, y_pred, target_names=LABEL_NAMES, digits=3, zero_division=0))

    # Sauvegarder
    logger.info("  Sauvegarde modèle...")
    trainer.save_model(f"{output_dir}/final")
    tokenizer.save_pretrained(f"{output_dir}/final")

    # Sauvegarder prédictions
    if config.evaluation.save_predictions:
        pred_path = f"{config.reports.comparison_dir}/pred_{model_config.name}.npy"
        np.save(pred_path, y_pred)

    return {
        "model": model,
        "metrics": metrics,
        "y_pred": y_pred,
        "trainer": trainer,
    }


def train_transformer_models(config: PipelineConfig) -> Dict[str, Any]:
    """Entraîne tous les modèles transformers."""
    logger.info("=" * 70)
    logger.info("MODÈLES TRANSFORMERS")
    logger.info("=" * 70)

    # Charger données textuelles
    train_df = pd.read_csv(config.data.train_df)
    test_df = pd.read_csv(config.data.test_df)

    # Nettoyer texte pour transformers (le texte brut suffit, pas de TF-IDF)
    def clean_for_transformers(text):
        if pd.isna(text):
            return ""
        text = str(text).replace("\n", " ").replace("\r", " ")
        return " ".join(text.split()).strip()

    train_df["text"] = train_df["text"].apply(clean_for_transformers)
    test_df["text"] = test_df["text"].apply(clean_for_transformers)

    # Filtrer textes trop courts
    train_df = train_df[train_df["text"].str.len() > 10].reset_index(drop=True)
    test_df = test_df[test_df["text"].str.len() > 10].reset_index(drop=True)

    logger.info(f"Train: {len(train_df)} | Test: {len(test_df)}")

    # Charger métriques existantes (sklearn) si présentes
    metrics_path = f"{config.reports.comparison_dir}/metrics.json"
    all_metrics = {}
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            all_metrics = json.load(f)

    # Entraîner chaque modèle
    results = {}
    for model_config in config.transformers.models:
        try:
            result = train_single_transformer(model_config, train_df, test_df, config)
            results[model_config.name] = result
            all_metrics[model_config.name] = result["metrics"]
        except Exception as e:
            logger.error(f"Erreur entraînement {model_config.name}: {e}", exc_info=True)

    # Sauvegarder toutes les métriques
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)

    logger.info("Modèles transformers entraînés")
    return results
