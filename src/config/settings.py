"""
Configuration du pipeline - Charge .env + config.yaml
"""

import os
import yaml
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path


@dataclass
class DataConfig:
    raw_csv: str
    ocr_csv: str
    intermediate: str
    sample_size: int
    random_state: int
    test_size: float
    min_text_length: int
    train_indices: str
    test_indices: str
    train_df: str
    test_df: str
    X_train_tfidf: str
    X_test_tfidf: str
    y_train: str
    y_test: str


@dataclass
class OCRConfig:
    tesseract_cmd: Optional[str]
    language: str
    preprocess: bool
    confidence_threshold: int


@dataclass
class TextMiningConfig:
    vectorizer: str
    max_features: int
    ngram_range: List[int]
    min_df: int
    max_df: float
    sublinear_tf: bool
    use_stemming: bool
    use_lemmatization: bool


@dataclass
class SklearnModelConfig:
    name: str
    class_name: str
    params: Dict[str, Any]


@dataclass
class SklearnConfig:
    models: List[SklearnModelConfig]


@dataclass
class TransformerModelConfig:
    name: str
    model_name: str
    max_length: int
    batch_size: int
    gradient_accumulation: int
    epochs: int
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    fp16: bool


@dataclass
class TransformersConfig:
    models: List[TransformerModelConfig]


@dataclass
class ShapConfig:
    global_sample_size: int
    max_display: int
    local_enabled: bool


@dataclass
class EvaluationConfig:
    metrics: List[str]
    primary_metric: str
    save_predictions: bool
    save_confusion_matrices: bool


@dataclass
class APIConfig:
    host: str
    port: int
    model_name: str
    enable_shap_local: bool
    max_file_size: int


@dataclass
class ReportsConfig:
    format: str
    exploration_dir: str
    comparison_dir: str
    include_image_plots: bool
    include_text_plots: bool


@dataclass
class OutputConfig:
    models_dir: str
    sklearn_dir: str
    transformers_dir: str
    logs_dir: str


@dataclass
class PipelineSteps:
    ocr: bool
    exploration: bool
    text_mining: bool
    train_sklearn: bool
    train_transformers: bool
    evaluate: bool
    shap_global: bool
    comparison: bool
    report: bool


@dataclass
class PipelineConfig:
    steps: PipelineSteps
    check_dependencies: bool
    data: DataConfig
    ocr: OCRConfig
    text_mining: TextMiningConfig
    sklearn: SklearnConfig
    transformers: TransformersConfig
    shap: ShapConfig
    evaluation: EvaluationConfig
    api: APIConfig
    reports: ReportsConfig
    output: OutputConfig

    @classmethod
    def from_yaml(cls, path: str) -> "PipelineConfig":
        """Charge la configuration depuis un fichier YAML."""
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        # Convertir les modèles sklearn
        sklearn_models = [
            SklearnModelConfig(
                name=m["name"],
                class_name=m["class"],
                params=m.get("params", {})
            )
            for m in config["sklearn"]["models"]
        ]

        # Convertir les modèles transformers
        transformer_models = [
            TransformerModelConfig(
                name=m["name"],
                model_name=m["model_name"],
                max_length=m["max_length"],
                batch_size=m["batch_size"],
                gradient_accumulation=m["gradient_accumulation"],
                epochs=m["epochs"],
                learning_rate=m["learning_rate"],
                weight_decay=m["weight_decay"],
                warmup_ratio=m["warmup_ratio"],
                fp16=m["fp16"]
            )
            for m in config["transformers"]["models"]
        ]

        return cls(
            steps=PipelineSteps(**config["pipeline"]["steps"]),
            check_dependencies=config["pipeline"]["check_dependencies"],
            data=DataConfig(**config["data"]),
            ocr=OCRConfig(**config["ocr"]),
            text_mining=TextMiningConfig(**config["text_mining"]),
            sklearn=SklearnConfig(models=sklearn_models),
            transformers=TransformersConfig(models=transformer_models),
            shap=ShapConfig(**config["shap"]),
            evaluation=EvaluationConfig(**config["evaluation"]),
            api=APIConfig(**config["api"]),
            reports=ReportsConfig(**config["reports"]),
            output=OutputConfig(**config["output"])
        )

    def override_steps(self, steps_to_run: List[str]):
        """Force l'activation des étapes spécifiées."""
        for step in self.steps.__dict__:
            setattr(self.steps, step, step in steps_to_run)

    def skip_steps(self, steps_to_skip: List[str]):
        """Désactive les étapes spécifiées."""
        for step in steps_to_skip:
            if hasattr(self.steps, step):
                setattr(self.steps, step, False)

    def get_step_dependencies(self) -> Dict[str, List[str]]:
        """Retourne le DAG des dépendances entre étapes."""
        return {
            "ocr": [],
            "exploration": ["ocr"],
            "text_mining": ["ocr"],
            "train_sklearn": ["text_mining"],
            "train_transformers": ["text_mining"],
            "evaluate": ["train_sklearn", "train_transformers"],
            "shap_global": ["train_sklearn", "train_transformers"],
            "comparison": ["evaluate"],
            "report": ["exploration", "comparison"]
        }

    def get_step_outputs(self) -> Dict[str, List[str]]:
        """Retourne les fichiers de sortie attendus par étape."""
        return {
            "ocr": [self.data.ocr_csv],
            "exploration": [
                f"{self.reports.exploration_dir}/report.md"
            ],
            "text_mining": [
                self.data.train_indices,
                self.data.test_indices,
                self.data.X_train_tfidf,
                self.data.X_test_tfidf,
                self.data.y_train,
                self.data.y_test,
                self.data.train_df,
                self.data.test_df
            ],
            "train_sklearn": [
                f"{self.output.sklearn_dir}/{m.name}.joblib"
                for m in self.sklearn.models
            ],
            "train_transformers": [
                # Le dossier est créé dès le début de l'entraînement par TrainingArguments :
                # on vérifie le modèle final sauvegardé, pas le simple dossier.
                f"{self.output.transformers_dir}/{m.name}/final/config.json"
                for m in self.transformers.models
            ],
            "evaluate": [
                f"{self.reports.comparison_dir}/metrics.json"
            ],
            "shap_global": [
                f"{self.reports.comparison_dir}/shap_global.html"
            ],
            "comparison": [
                f"{self.reports.comparison_dir}/comparison_table.md"
            ],
            "report": [
                f"{self.reports.comparison_dir}/final_report.md"
            ]
        }
