"""
Orchestrateur du Pipeline - Gère l'activation/désactivation et les dépendances
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

from config.settings import PipelineConfig


class PipelineOrchestrator:
    """Orchestre l'exécution du pipeline avec vérification des dépendances."""

    def __init__(self, config: PipelineConfig, force: bool = False):
        self.config = config
        self.force = force
        self.logger = logging.getLogger("Pipeline")
        self.results = {}

        # Créer les dossiers nécessaires
        self._create_directories()

    def _create_directories(self):
        """Crée tous les dossiers de sortie."""
        dirs = [
            os.path.dirname(self.config.data.raw_csv),
            os.path.dirname(self.config.data.ocr_csv),
            self.config.data.intermediate,
            self.config.output.models_dir,
            self.config.output.sklearn_dir,
            self.config.output.transformers_dir,
            self.config.reports.exploration_dir,
            self.config.reports.comparison_dir,
            self.config.output.logs_dir,
        ]
        for d in dirs:
            if d:
                os.makedirs(d, exist_ok=True)

    def _check_dependencies(self, step: str) -> bool:
        """Vérifie que les dépendances d'une étape sont satisfaites."""
        if not self.config.check_dependencies:
            return True

        dependencies = self.config.get_step_dependencies()
        required_steps = dependencies.get(step, [])

        # Vérifier les étapes parentes
        for req_step in required_steps:
            if not getattr(self.config.steps, req_step):
                # Si l'étape parente est désactivée, vérifier les fichiers de sortie
                outputs = self.config.get_step_outputs()
                expected_files = outputs.get(req_step, [])

                for f in expected_files:
                    if not os.path.exists(f):
                        self.logger.error(
                            f"Étape '{step}' requiert '{req_step}' mais "
                            f"fichier manquant: {f}"
                        )
                        return False

                self.logger.warning(
                    f"Étape '{req_step}' désactivée mais fichiers existants trouvés. "
                    f"Étape '{step}' peut continuer."
                )

        return True

    def _check_outputs_exist(self, step: str) -> bool:
        """Vérifie si les sorties d'une étape existent déjà."""
        if self.force:
            return False

        outputs = self.config.get_step_outputs()
        expected_files = outputs.get(step, [])

        return all(os.path.exists(f) for f in expected_files)

    def show_plan(self):
        """Affiche le plan d'exécution sans exécuter."""
        self.logger.info("PLAN D'EXÉCUTION:")
        self.logger.info("-" * 60)

        for step in self._get_execution_order():
            active = getattr(self.config.steps, step)
            status = "ACTIVÉ" if active else "DÉSACTIVÉ"

            deps = self.config.get_step_dependencies().get(step, [])
            deps_str = ", ".join(deps) if deps else "Aucune"

            outputs = self.config.get_step_outputs().get(step, [])
            outputs_exist = all(os.path.exists(f) for f in outputs)
            cache_status = "EN CACHE" if outputs_exist else "À GÉNÉRER"

            self.logger.info(f"  {step:20s} | {status:12s} | Déps: {deps_str:20s} | {cache_status}")

    def _get_execution_order(self) -> List[str]:
        """Retourne l'ordre d'exécution topologique des étapes."""
        # Ordre linéaire pour simplifier (DAG déjà résolu dans config)
        return [
            "ocr",
            "exploration", 
            "text_mining",
            "train_sklearn",
            "train_transformers",
            "evaluate",
            "shap_global",
            "comparison",
            "report"
        ]

    def run(self) -> Dict[str, Any]:
        """Exécute le pipeline complet."""
        execution_order = self._get_execution_order()

        for step in execution_order:
            if not getattr(self.config.steps, step):
                self.logger.info(f" Étape '{step}' désactivée - sautée")
                continue

            # Vérifier dépendances
            if not self._check_dependencies(step):
                raise RuntimeError(f"Dépendances non satisfaites pour '{step}'")

            # Vérifier cache
            if self._check_outputs_exist(step):
                self.logger.info(f"Étape '{step}' déjà en cache - sautée (utilisez --force pour réexécuter)")
                continue

            # Exécuter l'étape
            self.logger.info(f"Exécution de '{step}'...")
            step_result = self._run_step(step)
            self.results[step] = step_result
            self.logger.info(f"Étape '{step}' terminée")

        return self.results

    def _run_step(self, step: str) -> Any:
        """Exécute une étape spécifique."""

        if step == "ocr":
            from ocr.tesseract_ocr import run_ocr
            return run_ocr(self.config)

        elif step == "exploration":
            from exploration.report_generator import generate_exploration_report
            return generate_exploration_report(self.config)

        elif step == "text_mining":
            from text_mining.preprocessing import run_text_mining
            return run_text_mining(self.config)

        elif step == "train_sklearn":
            from models.sklearn_models import train_sklearn_models
            return train_sklearn_models(self.config)

        elif step == "train_transformers":
            from models.transformer_models import train_transformer_models
            return train_transformer_models(self.config)

        elif step == "evaluate":
            from evaluation.metrics import evaluate_all_models
            return evaluate_all_models(self.config)

        elif step == "shap_global":
            from interpretability.shap_explainer import run_shap_global
            return run_shap_global(self.config)

        elif step == "comparison":
            from evaluation.comparison import generate_comparison_table
            return generate_comparison_table(self.config)

        elif step == "report":
            from evaluation.comparison import generate_final_report
            return generate_final_report(self.config)

        else:
            raise ValueError(f"Étape inconnue: {step}")
