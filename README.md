# 🛡️ CLV Risk Intelligence Dashboard

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org)
[![Framework](https://img.shields.io/badge/framework-Dash/Plotly-orange.svg)](https://dash.plotly.com/)
[![UI](https://img.shields.io/badge/UI-Tailwind%20CSS-38B2AC.svg)](https://tailwindcss.com/)
[![Machine Learning](https://img.shields.io/badge/ML-Random%20Forest-green.svg)](https://scikit-learn.org/)
[![Statistics](https://img.shields.io/badge/Stats-Bootstrap%20%26%20Quantiles-red.svg)](https://en.wikipedia.org/wiki/Bootstrapping_(statistics))

> Une solution décisionnelle complète pour la prédiction de la **Customer Lifetime Value (CLV)** intégrant une quantification du risque par méthodes statistiques robustes.

---

## 📋 Présentation du Projet
Ce projet propose une approche intégrée pour l'estimation de la valeur client dans le secteur de l'assurance. L'objectif est de transformer des données historiques en un outil d'aide à la décision capable de prédire la CLV tout en fournissant une mesure rigoureuse de l'incertitude via un **intervalle de confiance à 95%**.

## 🛠️ Méthodologie et Étapes Clés

### 1. Analyse et Modélisation (Pipeline ML)
Le travail débute par une phase d'exploration et de préparation des données documentée dans le dossier `notebook/` :
* **Prétraitement** : Nettoyage des données, gestion des valeurs aberrantes et transformation logarithmique pour stabiliser la variance de la CLV.
* **Apprentissage** : Entraînement d'un modèle **Random Forest Regressor** capable de capturer les relations complexes entre les variables socio-démographiques et la valeur client.

### 2. Gestion de l'Incertitude et du Risque
Pour sécuriser les prévisions, le projet déploie une double approche statistique :
* **Estimation par Bootstrapping** : Utilisation du rééchantillonnage non-paramétrique sur les résidus de calibration (1500 simulations). Cette méthode garantit des Intervalles de Prédiction (IP) robustes, même en l'absence de normalité des erreurs.
* **Méthode des Quantiles** : Extraction des bornes de l'intervalle à partir de la distribution empirique simulée. Les quantiles **2.5%** et **97.5%** définissent la fourchette de sécurité à 95% pour chaque prédiction.



### 3. Audit et Validation Scientifique
La fiabilité du système est vérifiée par un protocole d'audit strict :
* **Test de Shapiro-Wilk** : Analyse de la distribution des résidus pour justifier l'usage de méthodes non-paramétriques.
* **Z-Test de Couverture** : Validation du taux de couverture réel. Le modèle atteint un score de **94.8%**, confirmant l'exactitude statistique des intervalles générés.

### 4. Interface Décisionnelle (SaaS Dashboard)
Le déploiement est réalisé via une application **Dash** interactive stylisée avec **Tailwind CSS** :
* **Simulation Dynamique** : Saisie des profils clients et calcul instantané de la CLV.
* **Jauge de Risque** : Traduction visuelle de l'intervalle de confiance pour une lecture métier immédiate.
* **Monitoring de Performance** : Visualisation des métriques (R², MAE, RMSE) et des graphiques d'audit (Réel vs Prédit).



## 📁 Structure du Dépôt
```text
├── app.py                      # Application Dash principale
├── real_data.json              # Statistiques et résultats de tests exportés
├── requirements.txt            # Dépendances du projet
├── models/                     # Artefacts du modèle (Pickle & Numpy)
│   ├── model.pkl               # Modèle Random Forest
│   ├── feature_columns.pkl              # Structure exacte des données
│   └── residus_log_reference.npy # Base des résidus pour Bootstrap
├── reports/                    # Preuves de validation scientifique
│   ├── intervals_prediction_test.csv
│   └── intervals_prediction_calibration.csv
└── notebook/                   # Recherche et entraînement
    └── MathsProject.ipynb
```

## 🚀 Installation et Utilisation

### 1. Cloner le projet

```bash
git clone https://github.com/ElMansouriAya/insurance-clv-dashboard.git
cd insurance-clv-dashboard
```
### 2. Créer et activer un environnement virtuel
Il est fortement recommandé d'isoler les dépendances dans un environnement virtuel :

Sur Windows :

```bash

python -m venv venv
.\venv\Scripts\activate
```
Sur Mac / Linux :

```bash

python3 -m venv venv
source venv/bin/activate
```
### 3. Installer les dépendances
Une fois l'environnement activé, installez les bibliothèques nécessaires :

```bash

pip install -r requirements.txt
```
### 4. Lancer l'application
Exécutez le script principal pour démarrer le serveur Dash :

```bash

python app.py
```
L'interface sera accessible sur votre navigateur à l'adresse : http://127.0.0.1:8057
