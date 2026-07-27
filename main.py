#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
MAIN PIPELINE - Classification de Documents (16 Catégories)
================================================================================
Orchestrateur principal qui active/désactive chaque étape du pipeline.
Vérifie les dépendances si une étape est désactivée.

Usage:
    python main.py --config config.yaml
    python main.py --config config.yaml --steps ocr,text_mining,train_sklearn
    python main.py --config config.yaml --skip exploration,shap_global
================================================================================
"""

import argparse
import sys
import os
import yaml
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

# Ajouter src au path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config.settings import PipelineConfig
from pipeline_orchestrator import PipelineOrchestrator


def setup_logging(log_dir: str = "logs") -> logging.Logger:
    """Configure le logging avec fichier et console."""
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"pipeline_{timestamp}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger("Pipeline")


def parse_args() -> argparse.Namespace:
    """Parse les arguments en ligne de commande."""
    parser = argparse.ArgumentParser(
        description="Pipeline de classification de documents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  # Pipeline complet
  python main.py --config config.yaml

  # Uniquement certaines étapes
  python main.py --config config.yaml --steps ocr,text_mining,train_sklearn

  # Sauter certaines étapes
  python main.py --config config.yaml --skip exploration,shap_global

  # Forcer le réentraînement (ignore les fichiers existants)
  python main.py --config config.yaml --force
        """
    )

    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config.yaml",
        help="Chemin vers le fichier de configuration (default: config.yaml)"
    )

    parser.add_argument(
        "--steps",
        type=str,
        default=None,
        help="Étapes à exécuter (séparées par virgule). Ex: ocr,text_mining,train_sklearn"
    )

    parser.add_argument(
        "--skip",
        type=str,
        default=None,
        help="Étapes à sauter (séparées par virgule). Ex: exploration,shap_global"
    )

    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Forcer la réexécution même si les fichiers de sortie existent"
    )

    parser.add_argument(
        "--dry-run", "-d",
        action="store_true",
        help="Afficher le plan d'exécution sans rien exécuter"
    )

    return parser.parse_args()


def main():
    """Point d'entrée principal."""
    args = parse_args()

    # Setup logging
    logger = setup_logging()
    logger.info("=" * 80)
    logger.info("PIPELINE DE CLASSIFICATION DE DOCUMENTS - 16 CATÉGORIES")
    logger.info("=" * 80)
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Charger la configuration
    try:
        config = PipelineConfig.from_yaml(args.config)
        logger.info(f"Configuration chargée: {config.data.sample_size:,} documents")
    except Exception as e:
        logger.error(f"Erreur chargement config: {e}")
        sys.exit(1)

    # Override les étapes si spécifiées en CLI
    if args.steps:
        steps_to_run = [s.strip() for s in args.steps.split(",")]
        config.override_steps(steps_to_run)
        logger.info(f"Étapes forcées: {steps_to_run}")

    if args.skip:
        steps_to_skip = [s.strip() for s in args.skip.split(",")]
        config.skip_steps(steps_to_skip)
        logger.info(f"Étapes sautées: {steps_to_skip}")

    # Initialiser l'orchestrateur
    orchestrator = PipelineOrchestrator(config, force=args.force)

    # Mode dry-run
    if args.dry_run:
        logger.info("MODE DRY-RUN - Plan d'exécution:")
        orchestrator.show_plan()
        return

    # Exécuter le pipeline
    try:
        logger.info("Démarrage du pipeline...")
        results = orchestrator.run()

        logger.info("=" * 80)
        logger.info("PIPELINE TERMINÉ AVEC SUCCÈS")
        logger.info("=" * 80)

        # Résumé
        if "comparison" in results:
            best_model = results["comparison"].get("best_model", "N/A")
            best_f1 = results["comparison"].get("best_f1_macro", 0)
            logger.info(f"Meilleur modèle: {best_model} (F1-macro: {best_f1:.4f})")

        logger.info(f"Rapports: {config.reports.comparison_dir}")
        logger.info(f"Modèles: {config.output.models_dir}")

    except Exception as e:
        logger.error(f"ERREUR PIPELINE: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
