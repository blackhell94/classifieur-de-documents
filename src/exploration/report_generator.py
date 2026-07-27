"""
Exploration des données - Fusion image + texte en rapport Markdown
"""

import os
import logging
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, List
from collections import Counter
import re

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

from config.settings import PipelineConfig
from constants import ID2LABEL

logger = logging.getLogger("Pipeline")

# Configuration plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def explore_images(df_images: pd.DataFrame, output_dir: str) -> List[str]:
    """Explore les métriques des images (boxplots)."""
    logger.info("Exploration des images...")

    metrics = ["aspect_ratio", "median_intensity", "skewness", "kurtosis", "entropy"]
    generated_plots = []

    for metric in metrics:
        if metric not in df_images.columns:
            logger.warning(f"  Métrique '{metric}' non trouvée, sautée")
            continue

        plt.figure(figsize=(14, 6))
        sns.boxplot(data=df_images, x="label", y=metric, palette="coolwarm")
        plt.title(f"Distribution de {metric} par label", fontsize=14, fontweight="bold")
        plt.xlabel("Label", fontsize=12)
        plt.ylabel(metric, fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()

        filename = os.path.join(output_dir, f"boxplot_{metric}.png")
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        generated_plots.append(filename)
        logger.info(f"  Sauvegardé: {filename}")

    return generated_plots


def explore_texts(df_text: pd.DataFrame, output_dir: str) -> List[str]:
    """Explore les textes OCRisés (longueurs + mots fréquents)."""
    logger.info("Exploration des textes...")

    # 1. Longueur des textes
    df_text['char_count'] = df_text['text'].fillna('').astype(str).apply(len)

    plt.figure(figsize=(14, 6))
    sns.boxplot(data=df_text, x='label', y='char_count', palette='coolwarm')
    plt.title('Distribution des longueurs de texte par label', fontsize=14, fontweight='bold')
    plt.xlabel('Label', fontsize=12)
    plt.ylabel('Nombre de caractères', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()

    filename = os.path.join(output_dir, 'boxplot_text_length.png')
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  Sauvegardé: {filename}")

    # 2. Mots fréquents par classe
    stop_words = set(stopwords.words('english'))
    tokenizer = RegexpTokenizer(r'[a-zA-Z]{3,}')

    def clean_and_tokenize(text):
        text = str(text).lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        words = text.split()
        return [w for w in words if w not in stop_words and len(w) > 2]

    labels = sorted(df_text['label'].unique())
    word_plots = []

    for label in labels:
        label_texts = df_text[df_text['label'] == label]['text']

        all_words = []
        for text in label_texts:
            all_words.extend(clean_and_tokenize(text))

        word_freq = Counter(all_words)
        top_15 = word_freq.most_common(15)

        if top_15:
            words, counts = zip(*top_15)

            plt.figure(figsize=(10, 6))
            bars = plt.barh(range(len(words)), counts, color=sns.color_palette("viridis", len(words)))
            plt.yticks(range(len(words)), words)
            plt.gca().invert_yaxis()
            plt.xlabel('Fréquence')
            plt.title(f'{ID2LABEL.get(label, f"Label {label}")} (n={len(label_texts)})', fontsize=13, fontweight='bold')

            for i, (bar, count) in enumerate(zip(bars, counts)):
                plt.text(count + max(counts)*0.01, i, str(count), va='center')

            plt.tight_layout()
            filename = os.path.join(output_dir, f'words_frequency_{label}.png')
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            plt.close()
            word_plots.append(filename)

    return [filename] + word_plots


def generate_exploration_report(config: PipelineConfig) -> Dict[str, Any]:
    """Génère le rapport d'exploration complet (image + texte)."""
    logger.info("=" * 70)
    logger.info("EXPLORATION DES DONNÉES")
    logger.info("=" * 70)

    output_dir = config.reports.exploration_dir
    os.makedirs(output_dir, exist_ok=True)

    # Charger données OCR
    if not os.path.exists(config.data.ocr_csv):
        raise FileNotFoundError(f"Fichier OCR non trouvé: {config.data.ocr_csv}")

    df = pd.read_csv(config.data.ocr_csv, encoding='latin-1', sep=None, engine='python')

    # Statistiques globales
    n_docs = len(df)
    n_classes = df['label'].nunique() if 'label' in df.columns else 0

    logger.info(f"Documents: {n_docs:,} | Classes: {n_classes}")

    # Exploration images (si métriques disponibles)
    image_metrics = ['aspect_ratio', 'median_intensity', 'skewness', 'kurtosis', 'entropy']
    has_image_metrics = any(m in df.columns for m in image_metrics)

    image_plots = []
    if has_image_metrics:
        image_plots = explore_images(df, output_dir)
    else:
        logger.info("Pas de métriques image trouvées, saut de l'exploration image")

    # Exploration textes
    text_plots = explore_texts(df, output_dir)

    # Générer rapport Markdown
    report_path = os.path.join(output_dir, 'report.md')

    md_content = f"""# Rapport d'Exploration des Données

## Métadonnées

- **Nombre de documents:** {n_docs:,}
- **Nombre de classes:** {n_classes}
- **Date de génération:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## Distribution des Classes

"""

    if 'label' in df.columns:
        dist = df['label'].value_counts().sort_index()
        md_content += "| Label | Nom | Count |\n"
        md_content += "|-------|-----|-------|\n"
        for label, count in dist.items():
            name = ID2LABEL.get(label, f"Class {label}")
            md_content += f"| {label} | {name} | {count} |\n"

    md_content += """

## Exploration des Images

"""

    if image_plots:
        for plot in image_plots:
            basename = os.path.basename(plot)
            md_content += f"### {basename.replace('boxplot_', '').replace('.png', '').replace('_', ' ').title()}\n\n"
            md_content += f"![{basename}]({basename})\n\n"
    else:
        md_content += "*Métriques image non disponibles*\n\n"

    md_content += """
## Exploration des Textes

### Longueur des Textes

![Longueur textes](boxplot_text_length.png)

### Mots Fréquents par Classe

"""

    for label in sorted(df['label'].unique()):
        name = ID2LABEL.get(label, f"Label {label}")
        md_content += f"#### {name}\n\n"
        md_content += f"![Mots fréquents {label}](words_frequency_{label}.png)\n\n"

    md_content += """
---

*Généré automatiquement par le pipeline*
"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    logger.info(f"Rapport sauvegardé: {report_path}")

    return {
        "report_path": report_path,
        "image_plots": image_plots,
        "text_plots": text_plots
    }
