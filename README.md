# Classification de Documents - Pipeline Complet

Pipeline de classification multi-classe (16 catégories) pour documents scannés avec OCR, ML, Deep Learning, SHAP et API FastAPI.

## Architecture

```
├── main.py                    # Orchestrateur principal
├── config.yaml                # Configuration centralisée
├── src/
│   ├── config/               # Gestion configuration
│   ├── ocr/                  # Tesseract OCR
│   ├── exploration/          # Analyse exploratoire
│   ├── text_mining/          # Prétraitement + TF-IDF
│   ├── models/               # ML classique + Transformers
│   ├── evaluation/           # Métriques + comparaison
│   ├── interpretability/     # SHAP global + local
│   └── api/                  # FastAPI
```

## Démarrage Rapide

### 1. Installation

```bash
# Cloner le repo
git clone <repo-url>
cd projet-document-classification

# Créer l'environnement
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou venv\Scripts\activate  # Windows

# Installer les dépendances
pip install -r requirements.txt

# Télécharger les données NLTK
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

### 2. Configuration

```bash
cp .env.example .env
# Éditer .env avec vos chemins
```

### 3. Exécution du Pipeline

```bash
# Pipeline complet
python main.py

# Uniquement certaines étapes
python main.py --steps ocr,text_mining,train_sklearn

# Sauter l'exploration
python main.py --skip exploration

# Forcer la réexécution
python main.py --force

# Mode dry-run (plan uniquement)
python main.py --dry-run
```

### 4. API de Prédiction

```bash
# Démarrer l'API
python -m src.api.main

# Tester
 curl -X POST "http://localhost:8000/predict" \
  -F "file=@document.tif" \
  -F "model_name=best" \
  -F "explain=true"
```

## Les 16 Catégories

| ID | Catégorie | ID | Catégorie |
|----|-----------|----|-----------|
| 0 | letter | 8 | file folder |
| 1 | form | 9 | news article |
| 2 | email | 10 | budget |
| 3 | handwritten | 11 | invoice |
| 4 | advertisement | 12 | presentation |
| 5 | scientific report | 13 | questionnaire |
| 6 | scientific publication | 14 | resume |
| 7 | specification | 15 | memo |

## Modèles

### ML Classique (sklearn)
- Naive Bayes
- Logistic Regression
- Random Forest
- Linear SVM

### Deep Learning (Transformers)
- BERT-base-uncased
- XLM-RoBERTa-base

## Métriques

- **F1-macro** (métrique principale)
- F1-weighted
- Accuracy
- Precision/Recall macro

## Interprétabilité SHAP

- **Global**: Summary plots, bar plots (500 documents)
- **Local**: Explication par token/feature pour chaque prédiction

## Rapports Auto-générés

- `reports/exploration/report.md` - Analyse exploratoire
- `reports/comparison/comparison_table.md` - Tableau comparatif
- `reports/comparison/final_report.md` - Rapport final complet

## Docker (optionnel)

```bash
docker-compose up --build
```

## Documentation API

Accédez à `http://localhost:8000/docs` pour la documentation Swagger.

## Configuration

Modifier `config.yaml` pour:
- Activer/désactiver les étapes
- Changer la taille d'échantillonnage
- Configurer les hyperparamètres
- Ajuster les paramètres SHAP

## Licence

MIT
