"""
Tests configuration
"""

import pytest
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config.settings import PipelineConfig


def test_config_load():
    """Test chargement config.yaml"""
    config = PipelineConfig.from_yaml("config.yaml")
    assert config is not None
    assert config.data.sample_size > 0
    assert config.data.intermediate  # champ requis par vectorizer/split
    assert len(config.sklearn.models) == 4
    assert len(config.transformers.models) == 2


def test_step_dependencies():
    """Test dépendances entre étapes"""
    config = PipelineConfig.from_yaml("config.yaml")
    deps = config.get_step_dependencies()

    assert "text_mining" in deps["train_sklearn"]
    assert "train_sklearn" in deps["evaluate"]
    assert "evaluate" in deps["comparison"]


def test_step_outputs():
    """Test fichiers de sortie attendus"""
    config = PipelineConfig.from_yaml("config.yaml")
    outputs = config.get_step_outputs()

    assert len(outputs["text_mining"]) > 0
    assert "data/intermediate/X_train_tfidf.npz" in outputs["text_mining"]
