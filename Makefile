# =============================================================================
# Makefile - Document Classification Pipeline
# =============================================================================

.PHONY: help install data pipeline api test clean

# Variables
PYTHON = python
CONFIG = config.yaml

# =============================================================================
# COMMANDS
# =============================================================================

help: ## Affiche l'aide
	@echo "Commandes disponibles:"
	@echo "  make install     - Installe les dépendances"
	@echo "  make data        - Prépare les données (OCR + text mining)"
	@echo "  make train       - Entraîne tous les modèles"
	@echo "  make evaluate    - Évalue et compare les modèles"
	@echo "  make api         - Lance l'API FastAPI"
	@echo "  make report      - Génère le rapport final"
	@echo "  make full        - Pipeline complet"
	@echo "  make test        - Lance les tests"
	@echo "  make clean       - Nettoie les fichiers générés"

install: ## Installe les dépendances
	pip install -r requirements.txt
	$(PYTHON) -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger')"

data: ## Prépare les données (OCR + text mining)
	$(PYTHON) main.py --steps ocr,text_mining

train: ## Entraîne tous les modèles
	$(PYTHON) main.py --steps train_sklearn,train_transformers

evaluate: ## Évalue et compare
	$(PYTHON) main.py --steps evaluate,shap_global,comparison

api: ## Lance l'API
	$(PYTHON) -m src.api.main

report: ## Génère le rapport final
	$(PYTHON) main.py --steps report

full: ## Pipeline complet
	$(PYTHON) main.py

test: ## Lance les tests
	pytest tests/ -v --cov=src --cov-report=html

clean: ## Nettoie les fichiers générés
	rm -rf logs/*.log
	rm -rf reports/exploration/*.png
	rm -rf reports/comparison/*.png
	rm -rf reports/comparison/*.json
	rm -rf reports/comparison/*.md
	rm -rf models/sklearn/*.joblib
	rm -rf models/transformers/*/
	rm -rf data/intermediate/*.npz
	rm -rf data/intermediate/*.npy
	rm -rf data/intermediate/*.csv
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
