"""
Tests prétraitement texte
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from text_mining.preprocessing import TextPreprocessor


def test_text_preprocessor():
    """Test préprocesseur de texte"""
    preprocessor = TextPreprocessor()

    text = "Hello World! This is a TEST document."
    result = preprocessor.preprocess(text)

    assert isinstance(result, str)
    assert len(result) > 0
    assert "hello" in result  # minuscules
    assert "test" in result


def test_clean_text():
    """Test nettoyage texte"""
    preprocessor = TextPreprocessor()

    text = "Hello!!! World??? 123"
    cleaned = preprocessor.clean_text(text)

    assert "!" not in cleaned
    assert "?" not in cleaned
    assert "1" not in cleaned
