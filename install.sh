#!/bin/bash
# =============================================================================
# INSTALLATION RAPIDE - Document Classification Pipeline
# =============================================================================

echo "Installation du pipeline de classification de documents..."

# 1. Vérifier Python
echo "Vérification de Python..."
python --version || python3 --version

# 2. Créer environnement virtuel
echo "Création de l'environnement virtuel..."
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# 3. Installer dépendances
echo "Installation des dépendances..."
pip install --upgrade pip
pip install -r requirements.txt

# 4. Télécharger NLTK
echo "Téléchargement des données NLTK..."
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger')"

# 5. Créer .env
echo "Configuration..."
if [ ! -f .env ]; then
    cp .env.example .env
    echo "Fichier .env créé. Éditez-le avec vos chemins."
fi

# 6. Créer dossiers
echo "Création des dossiers..."
mkdir -p data/raw data/intermediate data/processed
mkdir -p models/sklearn models/transformers
mkdir -p reports/exploration reports/comparison
mkdir -p logs

echo ""
echo "Installation terminée!"
echo ""
echo "Prochaines étapes:"
echo "  1. Éditer .env avec vos chemins"
echo "  2. Placer vos données dans data/raw/"
echo "  3. Lancer: python main.py --dry-run"
echo ""
echo "Commandes utiles:"
echo "  make help        # Aide"
echo "  make full        # Pipeline complet"
echo "  make api         # Lancer l'API"
echo ""
