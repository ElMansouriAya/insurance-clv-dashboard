# 🛡️ CLV Risk Intelligence Dashboard

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org)
[![Framework](https://img.shields.io/badge/framework-Dash/Plotly-orange.svg)](https://dash.plotly.com/)
[![UI](https://img.shields.io/badge/UI-Tailwind%20CSS-38B2AC.svg)](https://tailwindcss.com/)

> Une plateforme interactive de prédiction de la **Customer Lifetime Value (CLV)** intégrant une gestion avancée de l'incertitude par **Bootstrap non-paramétrique**.

---

## 📋 Présentation du Projet
Ce projet, réalisé dans le cadre d'un module de mathématiques et machine learning, vise à fournir aux assureurs un outil de scoring client ultra-précis. Contrairement aux modèles classiques qui ne donnent qu'une estimation ponctuelle, notre outil fournit un **intervalle de confiance à 95%** pour chaque prédiction, permettant une meilleure évaluation du risque financier.



## 🧠 Architecture Technique
Le système repose sur trois piliers majeurs :

1.  **Modèle Prédictif** : Random Forest Regressor (entraîné sur le logarithme de la CLV pour stabiliser la variance).
2.  **Moteur d'Incertitude** : Méthode Bootstrap basée sur la distribution des résidus de calibration.
3.  **Interface Interactive** : Dashboard moderne utilisant **Dash** pour la logique Python et **Tailwind CSS** pour le design.

## 📁 Structure du Dépôt
```text
├── app.py                      # Application Dash principale
├── real_data.json              # Statistiques et résultats de tests exportés
├── requirements.txt            # Dépendances du projet
├── models/                     # Artefacts du modèle (Pickle & Numpy)
│   ├── model.pkl               # Modèle Random Forest
│   ├── scaler.pkl              # Normalisation des données
│   └── residus_log_reference.npy # Base des résidus pour Bootstrap
├── reports/                    # Preuves de validation scientifique
│   ├── intervals_prediction_test.csv
│   └── intervals_prediction_calibration.csv
└── notebook/                   # Recherche et entraînement
    └── MathsProject_Final.ipynb
