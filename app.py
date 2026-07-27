"""
Application Streamlit - Classification de Documents (16 Catégories)
Soutenance Data Science - Datascientest Octobre 2025
"""

import streamlit as st
import requests
import json
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import io

# Configuration de la page
st.set_page_config(
    page_title="Classification de Documents",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# CONSTANTES
# =============================================================================

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from constants import ID2LABEL, LABEL2ID  # noqa: E402

CATEGORY_COLORS = {
    "letter": "#FF6B6B", "form": "#4ECDC4", "email": "#45B7D1",
    "handwritten": "#96CEB4", "advertisement": "#FFEAA7",
    "scientific report": "#DDA0DD", "scientific publication": "#98D8C8",
    "specification": "#F7DC6F", "file folder": "#BB8FCE",
    "news article": "#85C1E9", "budget": "#F8C471",
    "invoice": "#82E0AA", "presentation": "#F1948A",
    "questionnaire": "#85C1E9", "resume": "#D7BDE2",
    "memo": "#A9DFBF"
}

# =============================================================================
# FONCTIONS
# =============================================================================

def call_api_predict(file=None, text=None, model_name="best", explain=False):
    """Appelle l'API de prédiction."""
    try:
        url = "http://localhost:8000/predict"
        data = {"model_name": model_name, "explain": str(explain).lower()}

        if file is not None:
            files = {"file": file}
            response = requests.post(url, files=files, data=data, timeout=60)
        elif text is not None:
            data["text"] = text
            response = requests.post(url, data=data, timeout=30)
        else:
            return None

        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Erreur API: {response.status_code} - {response.text}")
            return None
    except requests.exceptions.ConnectionError:
        st.error("Impossible de se connecter a l'API. Verifiez qu'elle est demarree sur http://localhost:8000")
        return None
    except Exception as e:
        st.error(f"Erreur: {str(e)}")
        return None


def call_api_health():
    """Verifie l'etat de l'API."""
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        return response.json() if response.status_code == 200 else None
    except:
        return None


def create_probability_chart(probabilities):
    """Cree un graphique des probabilites."""
    df = pd.DataFrame([
        {"Classe": k, "Probabilite": v}
        for k, v in probabilities.items()
    ])
    df = df.sort_values("Probabilite", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = [CATEGORY_COLORS.get(c, "gray") for c in df["Classe"]]
    bars = ax.barh(df["Classe"], df["Probabilite"], color=colors, edgecolor='white')

    for bar, val in zip(bars, df["Probabilite"]):
        ax.text(val + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{val:.1%}', va='center', fontsize=9)

    ax.set_xlim(0, 1)
    ax.set_xlabel("Probabilite", fontsize=12)
    ax.set_title("Probabilites par classe", fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    return fig


def create_shap_chart(shap_data):
    """Cree un graphique SHAP."""
    if not shap_data or "top_features" not in shap_data:
        return None

    features = shap_data["top_features"][:10]
    df = pd.DataFrame(features)
    df = df.sort_values("shap_value", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['#e74c3c' if v < 0 else '#27ae60' for v in df["shap_value"]]
    bars = ax.barh(df["feature"], df["shap_value"], color=colors, edgecolor='white')

    for bar, val in zip(bars, df["shap_value"]):
        ax.text(val + (0.02 if val >= 0 else -0.02), 
                bar.get_y() + bar.get_height()/2,
                f'{val:+.3f}', va='center', ha='left' if val >= 0 else 'right',
                fontsize=9)

    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.set_xlabel("Valeur SHAP", fontsize=12)
    ax.set_title("Features les plus influentes (SHAP)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.markdown("## DocuClassif")
    st.markdown("*Classification automatique de documents*")
    st.markdown("---")

    health = call_api_health()
    if health:
        st.success("API connectee")
        st.info(f"Modeles disponibles: {len(health.get('models_available', []))}")
        st.info(f"Meilleur modele: {health.get('best_model', 'N/A')}")
    else:
        st.error("API non disponible")
        st.info("Demarrez l'API avec: python -m src.api.main")

    st.markdown("---")
    st.markdown("**Equipe:**")
    st.markdown("- Ahmed BOUZAZI")
    st.markdown("- Agathe FANOST")
    st.markdown("- Sanae RAOUI")
    st.markdown("**Formation:** Data Science Oct 2025")

# =============================================================================
# HEADER
# =============================================================================

st.markdown('<div class="main-header">Classification de Documents</div>', 
            unsafe_allow_html=True)
st.markdown('<div class="sub-header">16 categories | OCR + NLP + Machine Learning | SHAP</div>', 
            unsafe_allow_html=True)

# =============================================================================
# TABS
# =============================================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Prediction", 
    "Resultats", 
    "Interpretabilite",
    "Performance", 
    "ℹA propos"
])

# =============================================================================
# TAB 1: PREDICTION
# =============================================================================

with tab1:
    st.header("Predire la categorie d'un document")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Par image (OCR automatique)")
        uploaded_file = st.file_uploader(
            "Choisir un document (TIF, PNG, JPG)",
            type=["tif", "tiff", "png", "jpg", "jpeg"],
            key="file_uploader"
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Document charge", use_column_width=True)

            if st.button("Predire la classe", key="predict_file"):
                with st.spinner("Analyse en cours... (OCR + Classification)"):
                    uploaded_file.seek(0)
                    result = call_api_predict(
                        file=uploaded_file,
                        model_name="best",
                        explain=True
                    )

                    if result:
                        st.session_state["last_result"] = result
                        st.session_state["prediction_type"] = "image"
                        st.success("Prediction reussie !")
                        st.rerun()

    with col2:
        st.subheader("Par texte (deja OCRise)")
        text_input = st.text_area(
            "Collez le texte du document",
            height=200,
            placeholder="Ex: Invoice #12345\nDate: 2024-03-15\nTotal: $1,250.00...",
            key="text_input"
        )

        if st.button("Predire la classe", key="predict_text"):
            if text_input.strip():
                with st.spinner("Analyse en cours..."):
                    result = call_api_predict(
                        text=text_input,
                        model_name="best",
                        explain=True
                    )

                    if result:
                        st.session_state["last_result"] = result
                        st.session_state["prediction_type"] = "text"
                        st.success("Prediction reussie !")
                        st.rerun()
            else:
                st.warning("Veuillez entrer du texte")

# =============================================================================
# TAB 2: RESULTATS
# =============================================================================

with tab2:
    st.header("Resultats de la derniere prediction")

    if "last_result" not in st.session_state:
        st.info("Faites une prediction dans l'onglet 'Prediction' pour voir les resultats")
    else:
        result = st.session_state["last_result"]

        pred_class = result["predicted_class"]
        pred_id = result["predicted_label_id"]
        confidence = result["probabilities"][pred_class]

        st.markdown(
            f'<div class="prediction-box">'
            f'<h2>{pred_class.upper()}</h2>'
            f'<p style="font-size: 1.5rem;">Confiance: <strong>{confidence:.1%}</strong></p>'
            f'<p>ID: {pred_id} | Modele: {result["model_used"]}</p>'
            f'<p>Temps: {result["processing_time_ms"]:.0f} ms</p>'
            f'</div>',
            unsafe_allow_html=True
        )

        st.subheader("Probabilites par classe")
        fig = create_probability_chart(result["probabilities"])
        st.pyplot(fig)

        st.subheader("Top 3 classes les plus probables")
        top3 = sorted(result["probabilities"].items(), key=lambda x: x[1], reverse=True)[:3]
        cols = st.columns(3)
        for i, (cls, prob) in enumerate(top3):
            with cols[i]:
                st.metric(label=f"#{i+1} {cls}", value=f"{prob:.1%}")

        if result.get("ocr_text"):
            with st.expander("Texte OCRise (extrait)"):
                st.text(result["ocr_text"][:2000])

# =============================================================================
# TAB 3: INTERPRETABILITE
# =============================================================================

with tab3:
    st.header("Interpretabilite SHAP")

    if "last_result" not in st.session_state:
        st.info("Faites une prediction avec 'explain=true' pour voir l'interpretabilite")
    else:
        result = st.session_state["last_result"]
        shap_data = result.get("shap_explanation")

        if shap_data and "top_features" in shap_data:
            st.subheader("Features les plus influentes")
            fig = create_shap_chart(shap_data)
            if fig:
                st.pyplot(fig)

            st.subheader("Detail des features SHAP")
            df_shap = pd.DataFrame(shap_data["top_features"])
            df_shap["impact"] = df_shap["shap_value"].apply(
                lambda x: "Positif" if x > 0 else "Negatif"
            )
            st.dataframe(df_shap, use_container_width=True)

            st.subheader("Interpretation")
            positive = [f for f in shap_data["top_features"] if f["shap_value"] > 0]
            negative = [f for f in shap_data["top_features"] if f["shap_value"] < 0]

            if positive:
                st.write("**Mots poussant vers la classe predite :**")
                for feat in positive[:5]:
                    st.write(f"- `{feat['feature']}` : +{feat['shap_value']:.3f}")

            if negative:
                st.write("**Mots eloignant de la classe predite :**")
                for feat in negative[:5]:
                    st.write(f"- `{feat['feature']}` : {feat['shap_value']:.3f}")
        else:
            st.warning("Pas d'explication SHAP disponible.")

# =============================================================================
# TAB 4: PERFORMANCE
# =============================================================================

with tab4:
    st.header("Performance des modeles")

    performance_data = {
        "Modele": ["Logistic Regression", "Linear SVM", "Random Forest", 
                   "Naive Bayes", "BERT", "XLM-RoBERTa"],
        "F1-macro": [0.6741, 0.6605, 0.6511, 0.6050, 0.73, 0.43],
        "Accuracy": [0.6620, 0.6560, 0.6425, 0.5970, 0.75, 0.45],
        "F1-weighted": [0.6741, 0.6605, 0.6511, 0.6050, 0.74, 0.43],
        "Duree (s)": [15.3, 12.1, 45.2, 2.1, 1800, 1500]
    }
    df_perf = pd.DataFrame(performance_data)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Comparaison F1-macro")
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ['#27ae60' if f > 0.65 else '#f39c12' if f > 0.60 else '#e74c3c' 
                  for f in df_perf["F1-macro"]]
        bars = ax.barh(df_perf["Modele"], df_perf["F1-macro"], color=colors, edgecolor='white')
        ax.set_xlim(0, 1)
        ax.set_xlabel("F1-macro")
        ax.set_title("Performance des modeles (F1-macro)")
        for bar, val in zip(bars, df_perf["F1-macro"]):
            ax.text(val + 0.02, bar.get_y() + bar.get_height()/2, 
                    f'{val:.3f}', va='center')
        plt.tight_layout()
        st.pyplot(fig)

    with col2:
        st.subheader("Efficacite (F1 / temps)")
        df_perf["Efficacite"] = df_perf["F1-macro"] / df_perf["Duree (s)"]
        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.barh(df_perf["Modele"], df_perf["Efficacite"], 
                       color='#3498db', edgecolor='white')
        ax.set_xlabel("F1-macro / seconde")
        ax.set_title("Efficacite computationnelle")
        plt.tight_layout()
        st.pyplot(fig)

    st.subheader("Tableau comparatif complet")
    st.dataframe(df_perf.style.highlight_max(subset=["F1-macro", "Accuracy", "F1-weighted"], 
                                              color='green')
                           .highlight_min(subset=["Duree (s)"], color='lightblue'),
                 use_container_width=True)

    st.subheader("Matrice de confusion (Logistic Regression)")
    conf_matrix = np.array([
        [75, 5, 3, 8, 2, 1, 2, 1, 2, 1, 0, 0, 0, 0, 0, 0],
        [4, 82, 2, 1, 3, 1, 1, 2, 3, 0, 0, 0, 0, 1, 0, 0],
        [5, 2, 78, 2, 3, 1, 1, 1, 2, 2, 0, 0, 0, 0, 0, 3],
        [15, 1, 2, 35, 5, 2, 2, 1, 8, 3, 1, 1, 2, 1, 1, 20],
        [3, 2, 3, 2, 40, 5, 8, 2, 3, 15, 2, 3, 5, 2, 3, 2],
        [1, 1, 1, 1, 4, 70, 15, 3, 1, 2, 0, 0, 0, 0, 0, 1],
        [1, 0, 1, 1, 5, 18, 68, 2, 1, 2, 0, 0, 0, 0, 0, 1],
        [1, 2, 1, 1, 2, 3, 2, 80, 1, 1, 1, 2, 2, 1, 0, 0],
        [5, 8, 3, 10, 2, 1, 1, 2, 30, 1, 2, 3, 2, 5, 3, 22],
        [2, 1, 2, 2, 12, 3, 4, 2, 1, 65, 1, 2, 2, 1, 1, 1],
        [0, 1, 0, 1, 2, 1, 1, 2, 2, 1, 78, 8, 1, 1, 0, 1],
        [0, 0, 0, 1, 3, 1, 1, 2, 2, 1, 5, 82, 1, 0, 0, 1],
        [1, 2, 1, 2, 5, 1, 1, 3, 2, 2, 1, 2, 75, 1, 0, 1],
        [2, 5, 2, 1, 3, 1, 1, 2, 4, 1, 1, 1, 1, 75, 0, 1],
        [2, 1, 1, 2, 3, 1, 2, 1, 2, 2, 1, 1, 1, 2, 78, 1],
        [8, 3, 5, 5, 3, 2, 2, 2, 8, 2, 1, 2, 2, 3, 2, 52],
    ])

    fig, ax = plt.subplots(figsize=(12, 10))
    short_names = [ID2LABEL[i][:8] for i in range(16)]
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=short_names, yticklabels=short_names,
                ax=ax, cbar_kws={'label': 'Nombre de documents'})
    ax.set_xlabel("Predi", fontsize=12)
    ax.set_ylabel("Reel", fontsize=12)
    ax.set_title("Matrice de confusion - Logistic Regression", fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    st.pyplot(fig)

# =============================================================================
# TAB 5: A PROPOS
# =============================================================================

with tab5:
    st.header("ℹA propos du projet")

    st.markdown("""
    ### Classification et Extraction d'Information sur un Document

    Ce projet vise a classifier automatiquement des documents scannes en 16 categories 
    a l'aide de techniques d'OCR, de text mining et de machine learning.

    ### Architecture du pipeline

    ```
    Document (Image TIF/PNG)
         ↓
    [OCR Tesseract] → Texte brut
         ↓
    [Pretraitement] → Clean, tokenize, lemmatize
         ↓
    [Vectorisation TF-IDF] → Vecteurs numeriques
         ↓
    [Modele ML/DL] → Classification
         ↓
    [SHAP] → Interpretabilite
    ```

    ### Modeles utilises

    | Type | Modeles |
    |------|---------|
    | **ML Classique** | Naive Bayes, Logistic Regression, Random Forest, Linear SVM |
    | **Deep Learning** | BERT (bert-base-uncased), XLM-RoBERTa (xlm-roberta-base) |
    | **Interpretabilite** | SHAP (global + local) |

    ### Dataset

    - **Source :** RVL-CDIP (Ryerson Vision Lab)
    - **Taille :** 400 000 documents
    - **Classes :** 16 categories equilibrees
    - **Langue :** Anglais

    ### Equipe

    - **Ahmed BOUZAZI**
    - **Agathe FANOST**  
    - **Sanae RAOUI**

    **Formation :** Data Science — Datascientest (Octobre 2025)
    """)

    st.subheader("Les 16 categories de documents")
    cols = st.columns(4)
    for i, (cat_id, cat_name) in enumerate(ID2LABEL.items()):
        with cols[i % 4]:
            color = CATEGORY_COLORS.get(cat_name, "gray")
            text_color = "white" if cat_name in ['handwritten', 'advertisement', 'file folder', 
                                                   'scientific report', 'scientific publication', 'resume'] else 'black'
            st.markdown(
                f'<div style="background-color: {color}; padding: 10px; border-radius: 10px; '
                f'margin: 5px; text-align: center; color: {text_color};">'
                f'<strong>{cat_id}</strong><br>{cat_name}</div>',
                unsafe_allow_html=True
            )

# =============================================================================
# FOOTER
# =============================================================================

st.markdown("---")
st.markdown("<p style='text-align: center; color: #666;'>© 2026 - Projet Data Science Datascientest</p>", 
            unsafe_allow_html=True)
