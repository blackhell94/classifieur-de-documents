"""
Génération du tableau comparatif automatique et choix du meilleur modèle
"""

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict

import pandas as pd

from config.settings import PipelineConfig

logger = logging.getLogger("Pipeline")


def load_metrics(config: PipelineConfig) -> pd.DataFrame:
    """Charge les métriques de tous les modèles."""
    metrics_file = f"{config.reports.comparison_dir}/metrics.json"

    if not os.path.exists(metrics_file):
        raise FileNotFoundError(f"Fichier métriques non trouvé: {metrics_file}")

    with open(metrics_file, "r", encoding="utf-8") as f:
        all_metrics = json.load(f)

    if not all_metrics:
        raise ValueError(f"Aucune métrique dans {metrics_file}")

    # Convertir en DataFrame
    rows = []
    for model_name, metrics in all_metrics.items():
        row = {"Modèle": model_name}
        row.update(metrics)
        rows.append(row)

    df = pd.DataFrame(rows)

    # Trier par la métrique principale décroissante
    primary = config.evaluation.primary_metric
    if primary in df.columns:
        df = df.sort_values(primary, ascending=False).reset_index(drop=True)
    else:
        logger.warning(f"Métrique principale '{primary}' absente des résultats")

    return df


def _fmt(value) -> str:
    """Formate une métrique numérique, ou 'N/A' si absente/non numérique."""
    try:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return "N/A"
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "N/A"


def generate_comparison_table(config: PipelineConfig) -> Dict[str, Any]:
    """Génère le tableau comparatif Markdown et JSON."""
    logger.info("Génération du tableau comparatif...")

    df = load_metrics(config)
    primary = config.evaluation.primary_metric

    # Sauvegarder en JSON
    json_path = f"{config.reports.comparison_dir}/comparison_table.json"
    df.to_json(json_path, orient="records", indent=2, force_ascii=False)
    logger.info(f"  JSON sauvegardé: {json_path}")

    # Générer Markdown
    md_path = f"{config.reports.comparison_dir}/comparison_table.md"

    best_row = df.iloc[0]
    best_model = best_row["Modèle"]
    best_score = best_row.get(primary)

    md_content = (
        "# Tableau Comparatif des Modèles\n\n"
        f"**Généré le:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n"
        f"**Taille échantillon:** {config.data.sample_size:,} documents  \n"
        f"**Métrique principale:** {primary}\n\n"
        "---\n\n"
        "## Classement des Modèles\n\n"
        f"{df.to_markdown(index=False, floatfmt='.4f')}\n\n"
        "---\n\n"
        "## Meilleur Modèle\n\n"
        f"**{best_model}** avec un **{primary}** de **{_fmt(best_score)}**\n\n"
        "### Top 3:\n"
    )

    for i in range(min(3, len(df))):
        row = df.iloc[i]
        md_content += (
            f"\n{i + 1}. **{row['Modèle']}** — "
            f"F1-macro: {_fmt(row.get('f1_macro'))} | "
            f"F1-weighted: {_fmt(row.get('f1_weighted'))} | "
            f"Accuracy: {_fmt(row.get('accuracy'))}"
        )

    md_content += "\n\n---\n\n## Visualisations\n\n"
    md_content += "![Comparaison des modèles](model_comparison.png)\n\n"

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    logger.info(f"  Markdown sauvegardé: {md_path}")

    # Sauvegarder le meilleur modèle dans un fichier
    best_info = {
        "best_model": best_model,
        "best_f1_macro": float(best_score) if best_score is not None else None,
        "timestamp": datetime.now().isoformat(),
        "primary_metric": primary,
    }

    best_path = f"{config.output.models_dir}/best_model.json"
    with open(best_path, "w", encoding="utf-8") as f:
        json.dump(best_info, f, indent=2, ensure_ascii=False)

    logger.info(f"Meilleur modèle: {best_model} ({primary}: {_fmt(best_score)})")

    return {
        "comparison_table": df,
        "best_model": best_model,
        "best_f1_macro": best_score,
        "md_path": md_path,
        "json_path": json_path,
    }


def generate_final_report(config: PipelineConfig) -> Dict[str, Any]:
    """Génère le rapport final complet."""
    logger.info("Génération du rapport final...")

    comparison_md = f"{config.reports.comparison_dir}/comparison_table.md"
    exploration_md = f"{config.reports.exploration_dir}/report.md"
    report_path = f"{config.reports.comparison_dir}/final_report.md"

    report = (
        "# Rapport Final - Classification de Documents\n\n"
        "**Projet:** Classification Multi-Classe (16 Catégories)  \n"
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n"
        "**Pipeline Version:** 1.0\n\n"
        "---\n\n"
        "## Résumé Exécutif\n\n"
        "Ce rapport présente les résultats complets du pipeline de classification de documents.\n\n"
        "### Configuration\n"
        f"- **Nombre de documents:** {config.data.sample_size:,}\n"
        "- **Nombre de classes:** 16\n"
        f"- **Split:** {int((1 - config.data.test_size) * 100)}% train / {int(config.data.test_size * 100)}% test\n"
        f"- **Métrique principale:** {config.evaluation.primary_metric}\n\n"
        "---\n\n"
        "## Exploration des Données\n\n"
    )

    # Inclure exploration si disponible
    if os.path.exists(exploration_md):
        with open(exploration_md, "r", encoding="utf-8") as f:
            report += f.read()
    else:
        report += "*Exploration non disponible*\n"

    report += "\n---\n\n## Résultats des Modèles\n\n"

    # Inclure comparaison
    if os.path.exists(comparison_md):
        with open(comparison_md, "r", encoding="utf-8") as f:
            report += f.read()
    else:
        report += "*Comparaison non disponible*\n"

    report += (
        "\n\n---\n\n"
        "## Interprétabilité (SHAP)\n\n"
        "Les explications SHAP globales sont disponibles dans:\n"
        "- `reports/comparison/shap_global.html`\n\n"
        "---\n\n"
        "## API de Prédiction\n\n"
        "L'API FastAPI est disponible via:\n"
        "```bash\n"
        "python -m src.api.main\n"
        "```\n\n"
        "Endpoint: `http://localhost:8000/predict`\n\n"
        "---\n\n"
        "## Structure des Fichiers\n\n"
        "```\n"
        "models/\n"
        "├── sklearn/\n"
        "│   ├── naive_bayes.joblib\n"
        "│   ├── logistic_regression.joblib\n"
        "│   ├── random_forest.joblib\n"
        "│   └── linear_svm.joblib\n"
        "├── transformers/\n"
        "│   ├── bert/\n"
        "│   └── xlmroberta/\n"
        "└── best_model.json\n"
        "\n"
        "reports/\n"
        "├── exploration/\n"
        "│   └── report.md\n"
        "└── comparison/\n"
        "    ├── comparison_table.md\n"
        "    ├── comparison_table.json\n"
        "    ├── final_report.md\n"
        "    └── shap_global.html\n"
        "```\n\n"
        "---\n\n"
        "*Généré automatiquement par le pipeline de classification*\n"
    )

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    logger.info(f"  Rapport final: {report_path}")

    return {"report_path": report_path}
