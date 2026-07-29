"""
Évaluation des modèles - Matrices de confusion, métriques, visualisations
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, List

from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, f1_score, precision_score, recall_score
)

from config.settings import PipelineConfig
from constants import LABEL_NAMES

logger = logging.getLogger("Pipeline")
SHORT_NAMES = [name[:8] for name in LABEL_NAMES]


def plot_confusion_matrix(y_true, y_pred, model_name: str, output_dir: str) -> List[str]:
    """Génère les matrices de confusion (counts + normalisée)."""
    logger.info(f"  Matrices de confusion pour {model_name}...")

    generated = []

    # 1. Counts
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=SHORT_NAMES, yticklabels=SHORT_NAMES,
                annot_kws={'size': 8})
    plt.xlabel('Prédit', fontsize=12)
    plt.ylabel('Réel', fontsize=12)
    plt.title(f'Matrice de Confusion - Counts\n{model_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()

    path = os.path.join(output_dir, f'cm_{model_name}_counts.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    generated.append(path)

    # 2. Normalisée
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # éviter division par zéro si classe absente
    cm_norm = cm.astype(float) / row_sums
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='YlOrRd',
                xticklabels=SHORT_NAMES, yticklabels=SHORT_NAMES,
                annot_kws={'size': 8})
    plt.xlabel('Prédit', fontsize=12)
    plt.ylabel('Réel', fontsize=12)
    plt.title(f'Matrice de Confusion - Normalisée\n{model_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()

    path = os.path.join(output_dir, f'cm_{model_name}_normalized.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    generated.append(path)

    return generated


def plot_model_comparison_bar(metrics_dict: Dict[str, Dict], output_dir: str) -> str:
    """Graphique comparatif des modèles (bar chart)."""
    logger.info("  Graphique comparatif...")

    models = list(metrics_dict.keys())
    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 7))

    # Récupérer métriques
    accuracy = [metrics_dict[m].get('accuracy', 0) for m in models]
    f1_macro = [metrics_dict[m].get('f1_macro', 0) for m in models]
    f1_weighted = [metrics_dict[m].get('f1_weighted', 0) for m in models]

    bars1 = ax.bar(x - width, accuracy, width, label='Accuracy', color='skyblue')
    bars2 = ax.bar(x, f1_macro, width, label='F1 Macro', color='lightgreen')
    bars3 = ax.bar(x + width, f1_weighted, width, label='F1 Weighted', color='salmon')

    ax.set_xlabel('Modèles', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Comparaison des Performances', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)

    # Valeurs sur les barres
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    path = os.path.join(output_dir, 'model_comparison.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()

    return path


def evaluate_all_models(config: PipelineConfig) -> Dict[str, Any]:
    """Évalue tous les modèles et génère les visualisations."""
    logger.info("=" * 70)
    logger.info("ÉVALUATION DES MODÈLES")
    logger.info("=" * 70)

    output_dir = config.reports.comparison_dir
    os.makedirs(output_dir, exist_ok=True)

    # Charger métriques
    metrics_path = os.path.join(output_dir, 'metrics.json')
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"Métriques non trouvées: {metrics_path}")

    with open(metrics_path, 'r') as f:
        all_metrics = json.load(f)

    # Générer matrices de confusion pour chaque modèle
    y_test = np.load(config.data.y_test)

    for model_name in all_metrics.keys():
        pred_path = os.path.join(output_dir, f'pred_{model_name}.npy')
        if not os.path.exists(pred_path):
            logger.warning(f"  Prédictions non trouvées pour {model_name}")
            continue
        y_pred = np.load(pred_path)
        if len(y_pred) != len(y_test):
            logger.warning(
                f"  Prédictions de {model_name} ignorées: {len(y_pred)} échantillons "
                f"au lieu de {len(y_test)}. Régénérez-les sur le test set complet "
                f"(scripts/regenerate_predictions.py)."
            )
            continue
        plot_confusion_matrix(y_test, y_pred, model_name, output_dir)

    # Graphique comparatif
    comparison_plot = plot_model_comparison_bar(all_metrics, output_dir)

    logger.info("Évaluation terminée")

    return {
        "metrics": all_metrics,
        "comparison_plot": comparison_plot
    }
