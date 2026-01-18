

import dash
from dash import dcc, html, callback, Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from scipy import stats
import pickle
import os
from pathlib import Path

# ============================================================================
# CHARGEMENT DU MODÈLE RANDOM FOREST
# ============================================================================

MODEL_PATH = Path('models/model.pkl')
RESIDUALS_PATH = Path('models/residus_log_reference.npy')
ENCODER_PATH = Path("models/encoder.pkl")
SCALER_PATH = Path("models/scaler.pkl")
FEATURE_COLUMNS_PATH = Path("models/feature_columns.pkl")


rf_model = None
residus_log = None
encoder = None
scaler = None
scaler_numeric_cols = None
FEATURE_COLUMNS = None
MODEL_LOADED = False

def load_model():
    """Charger le modèle Random Forest et les artefacts associés"""
    global rf_model, residus_log, encoder, scaler, scaler_numeric_cols, FEATURE_COLUMNS, MODEL_LOADED

    try:
        if MODEL_PATH.exists():
            with open(MODEL_PATH, 'rb') as f:
                rf_model = pickle.load(f)
            print(f" Modèle Random Forest chargé depuis {MODEL_PATH}")
        else:
            print(f" Fichier modèle non trouvé: {MODEL_PATH}")
            return False

        if RESIDUALS_PATH.exists():
            residus_log = np.load(RESIDUALS_PATH)
            print(f" Résidus chargés: {len(residus_log)} observations")
        else:
            residus_log = np.random.normal(0, 0.3, 1000)
            print("   → Résidus synthétiques générés")

        if ENCODER_PATH.exists():
            with open(ENCODER_PATH, 'rb') as f:
                encoder = pickle.load(f)
            print(f" Encoder chargé depuis {ENCODER_PATH}")

        if SCALER_PATH.exists():
            with open(SCALER_PATH, 'rb') as f:
                scaler_data = pickle.load(f)
                # Support ancien format (scaler seul) et nouveau format (dict)
                if isinstance(scaler_data, dict):
                    scaler = scaler_data['scaler']
                    scaler_numeric_cols = scaler_data.get('numeric_cols', None)
                else:
                    scaler = scaler_data
                    scaler_numeric_cols = None
            print(f" Scaler chargé depuis {SCALER_PATH}")
            if scaler_numeric_cols:
                print(f"   → {len(scaler_numeric_cols)} colonnes numériques à scaler")

        #  Colonnes d'entraînement (indispensable pour éviter les erreurs sklearn: feature names mismatch)
        if FEATURE_COLUMNS_PATH.exists():
            with open(FEATURE_COLUMNS_PATH, 'rb') as f:
                FEATURE_COLUMNS = pickle.load(f)
            print(f" Feature columns chargées: {len(FEATURE_COLUMNS)} colonnes")
        elif rf_model is not None and hasattr(rf_model, 'feature_names_in_'):
            FEATURE_COLUMNS = list(rf_model.feature_names_in_)
            print(f" Feature columns détectées depuis le modèle: {len(FEATURE_COLUMNS)} colonnes")
        else:
            FEATURE_COLUMNS = None
            print(" feature_columns.pkl introuvable (alignement des colonnes limité)")

        MODEL_LOADED = True
        return True

    except Exception as e:
        print(f" Erreur lors du chargement du modèle: {e}")
        MODEL_LOADED = False
        return False

print("\n" + "="*60)
load_model()
print("="*60 + "\n")

# ============================================================================
# CHARGEMENT DES DONNÉES RÉELLES DEPUIS real_data.json
# ============================================================================

import json

REAL_DATA_PATH = Path("real_data.json")
DATA_LOADED = False

def load_real_data():
    """Charger les données réelles depuis real_data.json"""
    global PORTFOLIO_STATS, FEATURE_IMPORTANCE, CLV_BY_COVERAGE, CLV_BY_VEHICLE
    global STATISTICAL_TESTS, SAMPLE_PREDICTIONS, DATA_LOADED
    
    if REAL_DATA_PATH.exists():
        try:
            with open(REAL_DATA_PATH, 'r') as f:
                data = json.load(f)
            
            # Merge avec les valeurs par défaut pour éviter les KeyError
            default_portfolio = {
                "total_customers": 9134,
                "mean_clv": 8004.94,
                "median_clv": 5780.18,
                "total_revenue": 73133193.12,
                "r2_score": 0.5846,
                "mae": 2847.32,
                "rmse": 4521.67,
                "coverage_empirical": 94.26,
                "coverage_calibration": 94.93,
                "coverage_target": 95,
                "mean_interval_width": 9748,
            }
            loaded_portfolio = data.get("portfolio_stats", {}) or {}

            # Compat: certains exports utilisent du camelCase
            if "coverageTarget" in loaded_portfolio and "coverage_target" not in loaded_portfolio:
                loaded_portfolio["coverage_target"] = loaded_portfolio["coverageTarget"]
            if "coverageEmpirical" in loaded_portfolio and "coverage_empirical" not in loaded_portfolio:
                loaded_portfolio["coverage_empirical"] = loaded_portfolio["coverageEmpirical"]
            if "coverageCalibration" in loaded_portfolio and "coverage_calibration" not in loaded_portfolio:
                loaded_portfolio["coverage_calibration"] = loaded_portfolio["coverageCalibration"]

            PORTFOLIO_STATS = {**default_portfolio, **loaded_portfolio}

            # Feature importance
            fi_data = data.get("feature_importance", [])
            FEATURE_IMPORTANCE = pd.DataFrame(fi_data)
            if not FEATURE_IMPORTANCE.empty:
                FEATURE_IMPORTANCE.columns = ['feature', 'importance']
            
            # CLV by coverage
            cov_data = data.get("clv_by_coverage", [])
            CLV_BY_COVERAGE = pd.DataFrame(cov_data)
            
            # CLV by vehicle
            veh_data = data.get("clv_by_vehicle", [])
            CLV_BY_VEHICLE = pd.DataFrame(veh_data)
            
            # Statistical tests (merge + normalisation de clés pour éviter les KeyError)
            default_tests = {
                "shapiro_wilk": {"statistic": 0.7055, "p_value": 1.72e-48, "passed": False},
                "kolmogorov_smirnov": {"statistic": 0.3072, "p_value": 8.1e-151, "passed": False},
                "z_test_coverage": {"statistic": -1.4350, "p_value": 0.098, "passed": True},
            }
            loaded_tests = data.get("statistical_tests", {}) or {}

            # Compat: certains exports utilisent du camelCase
            if "kolmogorovSmirnov" in loaded_tests and "kolmogorov_smirnov" not in loaded_tests:
                loaded_tests["kolmogorov_smirnov"] = loaded_tests["kolmogorovSmirnov"]
            if "shapiroWilk" in loaded_tests and "shapiro_wilk" not in loaded_tests:
                loaded_tests["shapiro_wilk"] = loaded_tests["shapiroWilk"]
            if "zTestCoverage" in loaded_tests and "z_test_coverage" not in loaded_tests:
                loaded_tests["z_test_coverage"] = loaded_tests["zTestCoverage"]

            STATISTICAL_TESTS = {**default_tests, **loaded_tests}

            # Sample predictions
            pred_data = data.get("sample_predictions", [])
            SAMPLE_PREDICTIONS = pd.DataFrame(pred_data)
            
            DATA_LOADED = True
            print(f" Données réelles chargées depuis {REAL_DATA_PATH}")
            return True
            
        except Exception as e:
            print(f" Erreur lors du chargement des données: {e}")
            return False
    else:
        print(f" Fichier {REAL_DATA_PATH} non trouvé, utilisation des données par défaut")
        return False

# Valeurs par défaut (utilisées si real_data.json n'existe pas)
PORTFOLIO_STATS = {
    "total_customers": 9134,
    "mean_clv": 8004.94,
    "median_clv": 5780.18,
    "total_revenue": 73133193.12,
    "r2_score": 0.5846,
    "mae": 2847.32,
    "rmse": 4521.67,
    "coverage_empirical": 94.26,
    "coverage_calibration": 94.93,
    "coverage_target": 95,
    "mean_interval_width": 9748,
}

FEATURE_IMPORTANCE = pd.DataFrame({
    "feature": ["monthly_premium_auto", "total_claim_amount", "number_of_policies"],
    "importance": [0.42, 0.18, 0.08]
})

CLV_BY_COVERAGE = pd.DataFrame({
    "coverage": ["Basic", "Extended", "Premium"],
    "mean_clv": [5200, 7800, 12400],
    "count": [4012, 3245, 1877]
})

CLV_BY_VEHICLE = pd.DataFrame({
    "vehicle": ["Two-Door Car", "Four-Door Car", "SUV", "Sports Car", "Luxury Car", "Luxury SUV"],
    "mean_clv": [5100, 5800, 7200, 8500, 14200, 16800]
})

STATISTICAL_TESTS = {
    "shapiro_wilk": {"statistic": 0.7055, "p_value": 1.72e-48, "passed": False},
    "kolmogorov_smirnov": {"statistic": 0.3072, "p_value": 8.1e-151, "passed": False},
    "z_test_coverage": {"statistic": -1.4350, "p_value": 0.098, "passed": True},
}

np.random.seed(42)
SAMPLE_PREDICTIONS = pd.DataFrame({
    "index": range(100),
    "real": np.sort(2000 + np.random.rand(100) * 18000),
})
SAMPLE_PREDICTIONS["predicted"] = SAMPLE_PREDICTIONS["real"] * (0.85 + np.random.rand(100) * 0.3)
width = SAMPLE_PREDICTIONS["predicted"] * (0.3 + np.random.rand(100) * 0.4)
SAMPLE_PREDICTIONS["lower"] = SAMPLE_PREDICTIONS["predicted"] - width / 2
SAMPLE_PREDICTIONS["upper"] = SAMPLE_PREDICTIONS["predicted"] + width / 2

# Charger les données réelles
print("\n" + "="*60)
load_real_data()
print("="*60 + "\n")

# ============================================================================
# FONCTIONS DE PRÉDICTION
# ============================================================================

def prepare_features(coverage, vehicle_class, employment, education, gender, 
                     marital_status, monthly_premium, total_claims, num_policies, income):
    """Préparer les features pour le modèle Random Forest."""
    data = {
        'income': [income],
        'monthly_premium_auto': [monthly_premium],
        'months_since_last_claim': [12],
        'months_since_policy_inception': [24],
        'number_of_open_complaints': [0],
        'number_of_policies': [num_policies],
        'total_claim_amount': [total_claims],
        'policy_month': [6],
        'state': ['California'],
        'response': ['No'],
        'coverage': [coverage],
        'education': [education],
        'employmentstatus': [employment],
        'gender': [gender],
        'location_code': ['Suburban'],
        'marital_status': [marital_status],
        'policy_type': ['Personal Auto'],
        'policy': ['Personal L1'],
        'renew_offer_type': ['Offer1'],
        'sales_channel': ['Agent'],
        'vehicle_class': [vehicle_class],
        'vehicle_size': ['Medsize'],
    }
    return pd.DataFrame(data)


def predict_with_model(coverage, vehicle_class, employment, education, gender,
                       marital_status, monthly_premium, total_claims, num_policies, income):
    """Faire une prédiction avec le modèle Random Forest réel."""
    global rf_model, residus_log, encoder, scaler, scaler_numeric_cols, MODEL_LOADED
    
    if not MODEL_LOADED or rf_model is None:
        return simulate_prediction(coverage, vehicle_class, employment, 
                                   monthly_premium, total_claims, num_policies, income)
    
    try:
        df = prepare_features(
            coverage, vehicle_class, employment, education, gender,
            marital_status, monthly_premium, total_claims, num_policies, income
        )
        
        if encoder is not None:
            X_encoded = encoder.transform(df)
            if hasattr(X_encoded, 'toarray'):
                X_encoded = X_encoded.toarray()
            X_encoded = pd.DataFrame(X_encoded)
        else:
            X_encoded = pd.get_dummies(df, drop_first=False)

            expected_cols = None
            if FEATURE_COLUMNS is not None:
                expected_cols = FEATURE_COLUMNS
            elif rf_model is not None and hasattr(rf_model, 'feature_names_in_'):
                expected_cols = list(rf_model.feature_names_in_)

            # Aligner (ajouter les colonnes manquantes à 0 + respecter l'ordre)
            if expected_cols is not None:
                X_encoded = X_encoded.reindex(columns=expected_cols, fill_value=0)
        
        #  Appliquer le StandardScaler sur les colonnes numériques
        if scaler is not None:
            if scaler_numeric_cols is not None:
                # Nouveau format: on connaît les colonnes exactes
                cols_to_scale = [c for c in scaler_numeric_cols if c in X_encoded.columns]
                if cols_to_scale:
                    X_encoded[cols_to_scale] = scaler.transform(X_encoded[cols_to_scale])
            else:
                # Ancien format: scaler toutes les colonnes numériques
                numeric_cols = X_encoded.select_dtypes(include=[np.number]).columns.tolist()
                if numeric_cols:
                    X_encoded[numeric_cols] = scaler.transform(X_encoded[numeric_cols])
        
        pred_log = rf_model.predict(X_encoded)[0]
        prediction = np.expm1(pred_log)
        
        n_bootstrap = 1500
        boot_preds = np.zeros(n_bootstrap)
        
        for i in range(n_bootstrap):
            boot_residual = np.random.choice(residus_log)
            boot_preds[i] = np.expm1(pred_log + boot_residual)
        
        lower_bound = np.percentile(boot_preds, 2.5)
        upper_bound = np.percentile(boot_preds, 97.5)
        
        return float(prediction), float(lower_bound), float(upper_bound)
        
    except Exception as e:
        print(f" Erreur de prédiction: {e}")
        return simulate_prediction(coverage, vehicle_class, employment,
                                   monthly_premium, total_claims, num_policies, income)


def simulate_prediction(coverage, vehicle_class, employment, monthly_premium, 
                        total_claims, num_policies, income):
    """Simulation de prédiction (utilisée si le modèle n'est pas chargé)."""
    base = 3000
    base += monthly_premium * 35
    base += total_claims * 2
    base += num_policies * 500
    
    if coverage == "Premium":
        base *= 1.6
    elif coverage == "Extended":
        base *= 1.2
    
    if "Luxury" in vehicle_class:
        base *= 1.8
    elif vehicle_class == "Sports Car":
        base *= 1.3
    elif vehicle_class == "SUV":
        base *= 1.15
    
    if employment == "Unemployed":
        base *= 0.85
    elif employment == "Retired":
        base *= 1.1
    
    base += income * 0.02
    
    noise = 0.9 + np.random.rand() * 0.2
    prediction = base * noise
    
    interval_ratio = 0.25 + np.random.rand() * 0.3
    half_width = prediction * interval_ratio
    
    return prediction, prediction - half_width, prediction + half_width


# ============================================================================
# CONFIGURATION DASH AVEC TAILWIND
# ============================================================================

app = dash.Dash(
    __name__,
    suppress_callback_exceptions=True,
    title="CLV Prediction Dashboard"
)

# Custom index string avec Tailwind CDN
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <script src="https://cdn.tailwindcss.com"></script>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&family=Montserrat:wght@400;600;700;800;900&display=swap" rel="stylesheet">
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">
        <script>
            tailwind.config = {
                theme: {
                    extend: {
                        colors: {
                            background: '#0a0f1a',
                            card: '#111827',
                            'card-hover': '#1f2937',
                            border: '#1e293b',
                            primary: '#22c55e',
                            'primary-glow': '#4ade80',
                            secondary: '#6366f1',
                            success: '#22c55e',
                            warning: '#f59e0b',
                            danger: '#ef4444',
                            info: '#06b6d4',
                            muted: '#64748b',
                            foreground: '#f1f5f9',
                            'muted-foreground': '#94a3b8',
                        },
                        fontFamily: {
                            sans: ['Inter', 'system-ui', 'sans-serif'],
                            mono: ['JetBrains Mono', 'monospace'],
                        }
                    }
                }
            }
        </script>
        <style>
            body { 
                background-color: #0a0f1a; 
                font-family: 'Inter', system-ui, sans-serif;
            }
            /* Fix Plotly graph sizing issues */
            .plotly-graph-div { 
                background: transparent !important; 
            }
            .js-plotly-plot, .plotly {
                width: 100% !important;
            }
            .dash-graph {
                height: auto !important;
                overflow: visible !important;
            }
            /* Prevent infinite resize loop */
            .js-plotly-plot .plotly .main-svg {
                overflow: visible;
            }
            /* Neon glow effects for dark theme */
            .glow-primary {
                box-shadow: 0 0 30px rgba(34, 197, 94, 0.25);
            }
            .glow-success {
                box-shadow: 0 0 30px rgba(34, 197, 94, 0.25);
            }
            .glow-warning {
                box-shadow: 0 0 30px rgba(245, 158, 11, 0.25);
            }
            .glow-danger {
                box-shadow: 0 0 30px rgba(239, 68, 68, 0.25);
            }
            .glow-info {
                box-shadow: 0 0 30px rgba(6, 182, 212, 0.25);
            }
            /* Dash dropdown styles - Dark theme */
            .Select-control { background-color: #111827 !important; border-color: #1e293b !important; }
            .Select-menu-outer { background-color: #111827 !important; border-color: #1e293b !important; }
            .Select-option { background-color: #111827 !important; color: #f1f5f9 !important; }
            .Select-option:hover { background-color: #1f2937 !important; }
            .Select-value-label { color: #f1f5f9 !important; }
            .Select-placeholder { color: #94a3b8 !important; }
            .Select-input input { color: #f1f5f9 !important; }
        </style>
    </head>
    <body class="bg-background text-foreground min-h-screen antialiased">
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# ============================================================================
# GRAPHIQUES PLOTLY
# ============================================================================

def create_feature_importance_chart():
    """Graphique d'importance des variables"""
    df = FEATURE_IMPORTANCE.sort_values('importance', ascending=True)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df['importance'],
        y=df['feature'],
        orientation='h',
        marker=dict(
            color=df['importance'],
            colorscale=[[0, '#6366f1'], [0.5, '#3b82f6'], [1, '#06b6d4']],
            line=dict(width=0)
        ),
        hovertemplate='%{y}: %{x:.1%}<extra></extra>'
    ))
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(17,24,39,0)',
        plot_bgcolor='rgba(17,24,39,0)',
        font=dict(family='Inter', color='#94a3b8'),
        xaxis=dict(title='Importance', gridcolor='rgba(30, 41, 59, 0.8)', tickformat='.0%'),
        yaxis=dict(gridcolor='rgba(30, 41, 59, 0.8)'),
        margin=dict(l=180, r=40, t=20, b=40),
        height=350,
        hoverlabel=dict(bgcolor='#1f2937', font_size=12, bordercolor='#1e293b')
    )
    return fig


def create_clv_distribution_chart():
    """Distribution CLV"""
    np.random.seed(42)
    clv_values = np.concatenate([
        np.random.exponential(4000, 5000),
        np.random.normal(8000, 2000, 3000),
        np.random.exponential(15000, 1134)
    ])
    clv_values = clv_values[clv_values > 0][:9134]
    
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=clv_values,
        nbinsx=50,
        marker=dict(
            color='rgba(59, 130, 246, 0.7)',
            line=dict(color='rgba(96, 165, 250, 1)', width=1)
        ),
        hovertemplate='CLV: $%{x:,.0f}<br>Count: %{y}<extra></extra>'
    ))
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(17,24,39,0)',
        plot_bgcolor='rgba(17,24,39,0)',
        font=dict(family='Inter', color='#94a3b8'),
        xaxis=dict(title='Customer Lifetime Value ($)', gridcolor='rgba(30, 41, 59, 0.8)', tickformat='$,.0f'),
        yaxis=dict(title='Nombre de clients', gridcolor='rgba(30, 41, 59, 0.8)'),
        margin=dict(l=60, r=40, t=20, b=60),
        height=350,
        hoverlabel=dict(bgcolor='#1f2937', font_size=12, bordercolor='#1e293b')
    )
    return fig


def create_prediction_chart():
    """Graphique des prédictions vs réel"""
    df = SAMPLE_PREDICTIONS
    
    fig = go.Figure()
    
    # Intervalle de confiance
    fig.add_trace(go.Scatter(
        x=list(df['index']) + list(df['index'][::-1]),
        y=list(df['upper']) + list(df['lower'][::-1]),
        fill='toself',
        fillcolor='rgba(59, 130, 246, 0.2)',
        line=dict(color='rgba(0,0,0,0)'),
        name='IC 95%',
        hoverinfo='skip'
    ))
    
    # Valeurs réelles
    fig.add_trace(go.Scatter(
        x=df['index'],
        y=df['real'],
        mode='markers',
        marker=dict(color='#ef4444', size=6),
        name='CLV Réel',
        hovertemplate='Réel: $%{y:,.0f}<extra></extra>'
    ))
    
    # Prédictions
    fig.add_trace(go.Scatter(
        x=df['index'],
        y=df['predicted'],
        mode='lines',
        line=dict(color='#3b82f6', width=2),
        name='Prédiction',
        hovertemplate='Prédit: $%{y:,.0f}<extra></extra>'
    ))
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(17,24,39,0)',
        plot_bgcolor='rgba(17,24,39,0)',
        font=dict(family='Inter', color='#94a3b8'),
        xaxis=dict(title='Index Client', gridcolor='rgba(30, 41, 59, 0.8)'),
        yaxis=dict(title='CLV ($)', gridcolor='rgba(30, 41, 59, 0.8)', tickformat='$,.0f'),
        margin=dict(l=80, r=40, t=20, b=60),
        height=350,
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
        hoverlabel=dict(bgcolor='#1f2937', font_size=12, bordercolor='#1e293b')
    )
    return fig


def create_real_vs_predicted_scatter():
    """Nuage de points: Valeurs Réelles vs Valeurs Prédites + enveloppe d'incertitude."""
    df = SAMPLE_PREDICTIONS.copy()
    if df is None or df.empty:
        df = pd.DataFrame({"real": [], "predicted": [], "lower": [], "upper": []})

    # Tri pour dessiner une enveloppe propre
    if "real" in df.columns:
        df = df.sort_values("real")

    fig = go.Figure()

    # Enveloppe d'incertitude (IC95%) autour des prédictions
    if all(c in df.columns for c in ["real", "lower", "upper"]):
        fig.add_trace(go.Scatter(
            x=list(df["real"]) + list(df["real"][::-1]),
            y=list(df["upper"]) + list(df["lower"][::-1]),
            fill='toself',
            fillcolor='rgba(59, 130, 246, 0.18)',
            line=dict(color='rgba(0,0,0,0)'),
            name='Enveloppe IC 95%',
            hoverinfo='skip'
        ))

    # Points (réel vs prédit)
    if all(c in df.columns for c in ["real", "predicted"]):
        fig.add_trace(go.Scatter(
            x=df["real"],
            y=df["predicted"],
            mode='markers',
            marker=dict(color='rgba(239, 68, 68, 0.85)', size=6),
            name='Clients',
            hovertemplate='Réel: $%{x:,.0f}<br>Prédit: $%{y:,.0f}<extra></extra>'
        ))

    # Diagonale parfaite (y=x)
    if "real" in df.columns and len(df) > 0:
        min_v = float(np.nanmin([df["real"].min(), df.get("predicted", df["real"]).min()]))
        max_v = float(np.nanmax([df["real"].max(), df.get("predicted", df["real"]).max()]))
        fig.add_trace(go.Scatter(
            x=[min_v, max_v],
            y=[min_v, max_v],
            mode='lines',
            line=dict(color='rgba(245, 158, 11, 0.9)', dash='dash', width=2),
            name='y = x (parfait)'
        ))

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(17,24,39,0)',
        plot_bgcolor='rgba(17,24,39,0)',
        font=dict(family='Inter', color='#94a3b8'),
        xaxis=dict(title='Valeur Réelle (CLV $)', gridcolor='rgba(30, 41, 59, 0.8)', tickformat='$,.0f'),
        yaxis=dict(title='Valeur Prédite (CLV $)', gridcolor='rgba(30, 41, 59, 0.8)', tickformat='$,.0f'),
        margin=dict(l=70, r=30, t=20, b=60),
        height=350,
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
        hoverlabel=dict(bgcolor='#1f2937', font_size=12, bordercolor='#1e293b')
    )
    return fig


def create_qq_plot():
    """QQ-Plot des résidus (pour visualiser la non-normalité et justifier le Bootstrap)."""
    global residus_log

    # Utiliser les résidus réels si disponibles; sinon une distribution heavy-tail (Student) pour illustrer.
    if residus_log is not None and len(residus_log) > 20:
        residuals = np.asarray(residus_log).astype(float)
    else:
        np.random.seed(42)
        residuals = stats.t(df=3).rvs(1200)

    residuals = residuals[np.isfinite(residuals)]
    n = len(residuals)
    if n < 10:
        residuals = np.random.normal(0, 1, 200)
        n = len(residuals)

    probs = np.linspace(0.01, 0.99, n)
    qq_theoretical = np.sort(stats.norm.ppf(probs))
    qq_sample = np.sort(residuals)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=qq_theoretical,
        y=qq_sample,
        mode='markers',
        marker=dict(color='rgba(59, 130, 246, 0.75)', size=4),
        name='Résidus (observés)'
    ))

    # Droite de référence (ajustée sur Q1/Q3 pour être robuste)
    q1_t, q3_t = np.percentile(qq_theoretical, [25, 75])
    q1_s, q3_s = np.percentile(qq_sample, [25, 75])
    slope = (q3_s - q1_s) / (q3_t - q1_t) if (q3_t - q1_t) != 0 else 1
    intercept = q1_s - slope * q1_t
    x_line = np.array([qq_theoretical.min(), qq_theoretical.max()])
    y_line = intercept + slope * x_line

    fig.add_trace(go.Scatter(
        x=x_line,
        y=y_line,
        mode='lines',
        line=dict(color='rgba(239, 68, 68, 0.9)', dash='dash', width=2),
        name='Référence normale'
    ))

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(17,24,39,0)',
        plot_bgcolor='rgba(17,24,39,0)',
        font=dict(family='Inter', color='#94a3b8'),
        xaxis=dict(title='Quantiles théoriques (Normal)', gridcolor='rgba(30, 41, 59, 0.8)'),
        yaxis=dict(title='Quantiles observés (résidus)', gridcolor='rgba(30, 41, 59, 0.8)'),
        margin=dict(l=60, r=40, t=20, b=60),
        height=300,
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.02)
    )
    return fig


def create_residuals_chart():
    """Distribution des résidus (observée) + superposition normale (théorique)."""
    global residus_log

    if residus_log is not None and len(residus_log) > 20:
        residuals = np.asarray(residus_log).astype(float)
    else:
        np.random.seed(42)
        residuals = stats.t(df=3).rvs(1200)

    residuals = residuals[np.isfinite(residuals)]
    if len(residuals) < 10:
        residuals = np.random.normal(0, 1, 200)

    mu = float(np.mean(residuals))
    sigma = float(np.std(residuals) + 1e-9)

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=residuals,
        nbinsx=40,
        marker=dict(color='rgba(99, 102, 241, 0.7)', line=dict(color='rgba(129, 140, 248, 1)', width=1)),
        name='Résidus observés'
    ))

    x_range = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 200)
    # normalisation approx. pour être sur la même échelle que l'histogramme
    bin_width = (x_range.max() - x_range.min()) / 40
    y_normal = stats.norm.pdf(x_range, mu, sigma) * len(residuals) * bin_width
    fig.add_trace(go.Scatter(
        x=x_range,
        y=y_normal,
        mode='lines',
        line=dict(color='rgba(245, 158, 11, 0.95)', width=2),
        name='Normale théorique'
    ))

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(17,24,39,0)',
        plot_bgcolor='rgba(17,24,39,0)',
        font=dict(family='Inter', color='#94a3b8'),
        xaxis=dict(title='Résidus (log-scale)', gridcolor='rgba(30, 41, 59, 0.8)'),
        yaxis=dict(title='Fréquence', gridcolor='rgba(30, 41, 59, 0.8)'),
        margin=dict(l=60, r=40, t=20, b=60),
        height=300,
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.02)
    )
    return fig


# ============================================================================
# COMPOSANTS TAILWIND
# ============================================================================

def create_sidebar():
    """Barre latérale de navigation"""
    nav_items = [
        {"label": "Portfolio Insights", "icon": "fa-solid fa-table-columns", "href": "/", "description": "Vue d'ensemble du portefeuille"},
        {"label": "Smart Simulator", "icon": "fa-solid fa-calculator", "href": "/simulator", "description": "Prédiction CLV & Risque"},
        {"label": "Audit Scientifique", "icon": "fa-solid fa-shield-halved", "href": "/audit", "description": "Validation statistique"},
    ]
    
    mode_class = "bg-success/20 text-success" if MODEL_LOADED else "bg-warning/20 text-warning"
    mode_text = "ML Réel" if MODEL_LOADED else "Simulation"
    mode_icon = "fa-solid fa-microchip" if MODEL_LOADED else "fa-solid fa-cog"
    
    return html.Div(
        className="fixed left-0 top-0 h-full w-64 bg-card border-r border-border flex flex-col z-50",
        children=[
            html.Div(
                className="h-20 flex items-center gap-3 px-4 border-b border-border",
                children=[
                    html.Div(
                        className="flex flex-col leading-none",
                        style={"fontFamily": "'Montserrat', sans-serif"},
                        children=[
                            html.Div(
                                className="flex items-baseline",
                                children=[
                                    html.Span("Customer", className="text-2xl font-black tracking-tight text-foreground"),
                                    html.Span("Path", className="text-2xl font-black tracking-tight bg-gradient-to-r from-primary to-emerald-400 bg-clip-text", style={"WebkitBackgroundClip": "text", "WebkitTextFillColor": "transparent"})
                                ]
                            ),
                            html.Div(
                                className="flex items-center gap-1.5 mt-1",
                                children=[
                                    html.Div(className="h-0.5 w-8 bg-gradient-to-r from-primary to-emerald-400 rounded-full"),
                                    html.Span("Prédiction CLV", className="text-[9px] uppercase tracking-[0.25em] text-muted-foreground/50 font-semibold")
                                ]
                            )
                        ]
                    )
                ]
            ),
            
            # Navigation avec descriptions
            html.Nav(
                className="flex-1 p-3 space-y-1",
                children=[
                    dcc.Link(
                        className="flex items-center gap-3 px-3 py-3 rounded-lg text-muted-foreground hover:bg-card-hover hover:text-foreground transition-all duration-200",
                        href=item["href"],
                        children=[
                            html.I(className=f"{item['icon']} h-5 w-5 flex-shrink-0"),
                            html.Div(
                                className="flex-1 min-w-0",
                                children=[
                                    html.P(item["label"], className="text-sm font-medium truncate"),
                                    html.P(item["description"], className="text-[10px] text-muted-foreground/50 truncate")
                                ]
                            )
                        ]
                    ) for item in nav_items
                ]
            ),
            
            # Mode indicator
            html.Div(
                className="p-4 border-t border-border",
                children=[
                    html.Div(
                        className=f"flex items-center gap-2 px-3 py-2 rounded-lg {mode_class}",
                        children=[
                            html.I(className=f"{mode_icon} text-sm"),
                            html.Span(mode_text, className="text-sm font-medium")
                        ]
                    )
                ]
            )
        ]
    )


def create_kpi_card(title, value, subtitle=None, icon="fa-solid fa-chart-bar", color="primary"):
    """Carte KPI avec Tailwind et bordures fluorescentes"""
    color_map = {
        "primary": "text-primary",
        "success": "text-success",
        "warning": "text-warning",
        "info": "text-info",
        "danger": "text-danger"
    }
    border_glow_map = {
        "primary": "border-primary/50 glow-primary",
        "success": "border-success/50 glow-success",
        "warning": "border-warning/50 glow-warning",
        "info": "border-info/50 glow-info",
        "danger": "border-danger/50 glow-danger"
    }
    icon_bg_map = {
        "primary": "bg-primary/20",
        "success": "bg-success/20",
        "warning": "bg-warning/20",
        "info": "bg-info/20",
        "danger": "bg-danger/20"
    }
    text_color = color_map.get(color, "text-primary")
    border_glow = border_glow_map.get(color, "border-primary/50 glow-primary")
    icon_bg = icon_bg_map.get(color, "bg-primary/20")
    
    return html.Div(
        className=f"bg-card border rounded-xl p-6 hover:bg-card-hover transition-all duration-300 {border_glow}",
        children=[
            html.Div(
                className="flex items-start justify-between",
                children=[
                    html.Div(
                        className="space-y-1",
                        children=[
                            html.P(title, className="text-sm font-medium text-muted-foreground"),
                            html.P(value, className=f"text-3xl font-bold tracking-tight {text_color}"),
                            html.P(subtitle, className="text-xs text-muted-foreground") if subtitle else None
                        ]
                    ),
                    html.Div(
                        className=f"p-3 rounded-lg {icon_bg}",
                        children=[
                            html.I(className=f"{icon} text-lg {text_color}")
                        ]
                    )
                ]
            )
        ]
    )


def create_stat_badge(label, passed, stat_value=None, p_value=None, interpretation=None):
    """Badge de test statistique avec icônes Font Awesome"""
    if passed:
        badge_class = "bg-success/20 text-success border-success/30"
        icon_class = "fa-solid fa-circle-check"
        status = "PASS"
        glow = "glow-success"
    else:
        badge_class = "bg-warning/20 text-warning border-warning/30"
        icon_class = "fa-solid fa-triangle-exclamation"
        status = "ATTENTION"
        glow = "glow-warning"
    
    return html.Div(
        className=f"bg-card border rounded-xl p-6 {glow}",
        style={"borderColor": "rgba(16, 185, 129, 0.3)" if passed else "rgba(245, 158, 11, 0.3)"},
        children=[
            html.Div(
                className="flex items-start gap-4",
                children=[
                    html.Div(
                        className=f"p-3 rounded-full {badge_class}",
                        children=[
                            html.I(className=f"{icon_class} text-xl")
                        ]
                    ),
                    html.Div(
                        className="flex-1",
                        children=[
                            html.Div(
                                className="flex items-center justify-between mb-2",
                                children=[
                                    html.Span(label, className="font-semibold text-foreground"),
                                    html.Span(
                                        status,
                                        className=f"px-3 py-1 rounded-full text-xs font-bold border {badge_class}"
                                    )
                                ]
                            ),
                            html.Div(
                                className="flex gap-6 mb-3",
                                children=[
                                    html.Div([
                                        html.Span("Statistique", className="text-xs text-muted-foreground block"),
                                        html.Span(f"{stat_value:.4f}" if stat_value else "N/A", className="font-mono font-bold text-foreground")
                                    ]) if stat_value else None,
                                    html.Div([
                                        html.Span("P-Value", className="text-xs text-muted-foreground block"),
                                        html.Span(
                                            f"{p_value:.2e}" if p_value and p_value < 0.0001 else f"{p_value:.4f}" if p_value else "N/A",
                                            className=f"font-mono font-bold {'text-success' if passed else 'text-warning'}"
                                        )
                                    ]) if p_value is not None else None
                                ]
                            ),
                            html.P(
                                interpretation,
                                className="text-sm text-muted-foreground border-t border-border pt-3"
                            ) if interpretation else None
                        ]
                    )
                ]
            )
        ]
    )


def create_risk_management_gauge(prediction, lower, upper, confidence=95):
    """
    Jauge de Gestion du Risque Professionnelle avec Plotly
    - Affiche la prédiction centrale avec aiguille
    - Zone de confiance Bootstrap colorée [Inf, Sup]
    - Interprétation dynamique pour les assureurs
    """
    range_val = upper - lower
    risk_ratio = range_val / prediction if prediction > 0 else 0.5
    
    # Déterminer le niveau de risque
    if risk_ratio < 0.4:
        risk_level = "low"
        risk_text = "Faible"
        risk_color = "success"
        gauge_color = "#22c55e"  # Green-500
        zone_color = "rgba(34, 197, 94, 0.3)"
    elif risk_ratio < 0.7:
        risk_level = "medium"
        risk_text = "Modéré"
        risk_color = "warning"
        gauge_color = "#f59e0b"  # Amber-500
        zone_color = "rgba(245, 158, 11, 0.3)"
    else:
        risk_level = "high"
        risk_text = "Élevé"
        risk_color = "danger"
        gauge_color = "#ef4444"  # Red-500
        zone_color = "rgba(239, 68, 68, 0.3)"
    
    color_classes = {
        "success": {"bg": "bg-success", "text": "text-success", "glow": "glow-success", "border": "border-success/30"},
        "warning": {"bg": "bg-warning", "text": "text-warning", "glow": "glow-warning", "border": "border-warning/30"},
        "danger": {"bg": "bg-danger", "text": "text-danger", "glow": "glow-danger", "border": "border-danger/30"}
    }
    colors = color_classes[risk_color]
    
    # Calcul des bornes pour la jauge (avec marge de 10% de chaque côté)
    margin = range_val * 0.15
    gauge_min = max(0, lower - margin)
    gauge_max = upper + margin
    
    # Créer la jauge Plotly professionnelle
    fig = go.Figure()
    
    # Jauge principale avec zone de confiance
    fig.add_trace(go.Indicator(
        mode="gauge+number+delta",
        value=prediction,
        number={
            'prefix': "$",
            'font': {'size': 48, 'color': '#f1f5f9', 'family': 'Inter'},
            'valueformat': ",.0f"
        },
        delta={
            'reference': (lower + upper) / 2,
            'relative': False,
            'increasing': {'color': '#22c55e'},
            'decreasing': {'color': '#ef4444'},
            'font': {'size': 16}
        },
        gauge={
            'axis': {
                'range': [gauge_min, gauge_max],
                'tickwidth': 2,
                'tickcolor': "#475569",
                'tickformat': "$,.0f",
                'tickfont': {'color': '#94a3b8', 'size': 11, 'family': 'Inter'}
            },
            'bar': {'color': gauge_color, 'thickness': 0.3},
            'bgcolor': "rgba(30, 41, 59, 0.8)",
            'borderwidth': 2,
            'bordercolor': "rgba(51, 65, 85, 0.8)",
            'steps': [
                # Zone avant l'intervalle de confiance (gris)
                {'range': [gauge_min, lower], 'color': 'rgba(51, 65, 85, 0.4)'},
                # Zone de confiance Bootstrap (colorée)
                {'range': [lower, upper], 'color': zone_color},
                # Zone après l'intervalle de confiance (gris)
                {'range': [upper, gauge_max], 'color': 'rgba(51, 65, 85, 0.4)'}
            ],
            'threshold': {
                'line': {'color': '#f1f5f9', 'width': 4},
                'thickness': 0.8,
                'value': prediction
            }
        },
        domain={'x': [0, 1], 'y': [0, 1]}
    ))
    
    # Annotations pour les bornes
    fig.add_annotation(
        x=0.12, y=0.25,
        text=f"<b>Borne Inf.</b><br>${lower:,.0f}",
        showarrow=False,
        font=dict(size=12, color='#ef4444', family='Inter'),
        align='center'
    )
    
    fig.add_annotation(
        x=0.88, y=0.25,
        text=f"<b>Borne Sup.</b><br>${upper:,.0f}",
        showarrow=False,
        font=dict(size=12, color='#22c55e', family='Inter'),
        align='center'
    )
    
    # Annotation centrale pour l'intervalle
    fig.add_annotation(
        x=0.5, y=-0.05,
        text=f"Zone de Confiance Bootstrap {confidence}%",
        showarrow=False,
        font=dict(size=11, color='#06b6d4', family='Inter'),
        align='center'
    )
    
    fig.update_layout(
        paper_bgcolor='rgba(17,24,39,0)',
        plot_bgcolor='rgba(17,24,39,0)',
        font=dict(family='Inter', color='#94a3b8'),
        margin=dict(l=30, r=30, t=50, b=30),
        height=280
    )
    
    return html.Div(
        className=f"bg-card border {colors['border']} rounded-xl p-6 {colors['glow']}",
        children=[
            # Header avec titre
            html.Div(
                className="flex items-center justify-between mb-4",
                children=[
                    html.Div(
                        className="flex items-center gap-2",
                        children=[
                            html.I(className="fa-solid fa-gauge-high text-primary text-xl"),
                            html.H3("Jauge de Gestion du Risque", className="text-lg font-semibold text-foreground")
                        ]
                    ),
                    html.Div(
                        className=f"px-3 py-1 rounded-full text-xs font-bold {colors['bg']}/20 {colors['text']} border {colors['border']}",
                        children=[f"Incertitude {risk_text}"]
                    )
                ]
            ),
            
            # Jauge Plotly
            dcc.Graph(
                figure=fig,
                config={'displayModeBar': False, 'responsive': True, 'staticPlot': False},
                style={'height': '280px'}
            ),
            
            # Stats Row détaillées
            html.Div(
                className="grid grid-cols-3 gap-4 pt-4 border-t border-border/50",
                children=[
                    html.Div(
                        className="bg-background/50 rounded-lg p-3 text-center",
                        children=[
                            html.P("Borne Inférieure", className="text-xs text-muted-foreground mb-1"),
                            html.P(f"${lower:,.0f}", className="text-xl font-bold font-mono text-danger")
                        ]
                    ),
                    html.Div(
                        className="bg-background/50 rounded-lg p-3 text-center",
                        children=[
                            html.P(f"Marge d'Erreur", className="text-xs text-muted-foreground mb-1"),
                            html.P(f"±${range_val/2:,.0f}", className="text-xl font-bold font-mono text-warning")
                        ]
                    ),
                    html.Div(
                        className="bg-background/50 rounded-lg p-3 text-center",
                        children=[
                            html.P("Borne Supérieure", className="text-xs text-muted-foreground mb-1"),
                            html.P(f"${upper:,.0f}", className="text-xl font-bold font-mono text-success")
                        ]
                    )
                ]
            ),
            
            # Interprétation dynamique pour les assureurs
            html.Div(
                className="mt-4 p-4 bg-info/10 border border-info/30 rounded-lg",
                children=[
                    html.Div(
                        className="flex items-start gap-3",
                        children=[
                            html.I(className="fa-solid fa-chart-line text-info mt-1"),
                            html.Div([
                                html.P("Interprétation Actuarielle", className="font-semibold text-foreground mb-1"),
                                html.P(
                                    f"Nous sommes sûrs à {confidence}% que la valeur vie de ce client se situera entre ${lower:,.0f} et ${upper:,.0f}. "
                                    f"La prédiction centrale est de ${prediction:,.0f} avec une marge d'erreur de ±${range_val/2:,.0f}.",
                                    className="text-sm text-muted-foreground mb-2"
                                ),
                                html.P(
                                    f"Ratio d'incertitude: {risk_ratio:.1%} — "
                                    f"{'Cette estimation est fiable pour la prise de décision.' if risk_ratio < 0.4 else 'Une surveillance modérée est recommandée.' if risk_ratio < 0.7 else 'Prudence accrue recommandée pour ce profil.'}",
                                    className="text-xs text-info italic"
                                )
                            ])
                        ]
                    )
                ]
            ),
            
            # Risk Indicator Badge
            html.Div(
                className=f"flex items-center justify-center gap-2 py-3 rounded-lg mt-4 {colors['bg']}/10 {colors['text']}",
                children=[
                    html.Div(className=f"w-2 h-2 rounded-full {colors['bg']}"),
                    html.Span(f"Niveau d'Incertitude : {risk_text} ({risk_ratio:.0%})", className="text-sm font-medium")
                ]
            )
        ]
    )


def create_uncertainty_gauge(value, label="Incertitude"):
    """Jauge d'incertitude simple (pour compatibilité)"""
    percentage = min(100, max(0, value * 100))
    
    if percentage < 15:
        color = "bg-success"
        status = "Faible"
    elif percentage < 30:
        color = "bg-info"
        status = "Modérée"
    elif percentage < 50:
        color = "bg-warning"
        status = "Élevée"
    else:
        color = "bg-danger"
        status = "Très élevée"
    
    return html.Div(
        className="bg-card border border-border rounded-xl p-6",
        children=[
            html.Div(
                className="flex justify-between items-center mb-3",
                children=[
                    html.Span(label, className="text-sm font-medium text-muted-foreground uppercase tracking-wide"),
                    html.Span(status, className="text-sm text-foreground")
                ]
            ),
            html.Div(
                className="w-full bg-slate-700 rounded-full h-3 overflow-hidden mb-3",
                children=[
                    html.Div(
                        className=f"{color} h-full rounded-full transition-all duration-500",
                        style={"width": f"{percentage}%"}
                    )
                ]
            ),
            html.Div(
                className="text-center",
                children=[
                    html.Span(f"{percentage:.1f}%", className="text-2xl font-bold text-foreground")
                ]
            )
        ]
    )


def create_dynamic_insight(prediction, lower, upper):
    """Create dynamic insight card based on prediction - React style"""
    range_val = upper - lower
    risk_ratio = range_val / prediction if prediction > 0 else 0.5
    
    is_high_potential = prediction > 8000 and risk_ratio < 0.5
    is_risky = risk_ratio > 0.6 or prediction < 4000
    
    if is_high_potential:
        insight_type = "success"
        icon = "fa-solid fa-arrow-trend-up"
        title = "Client Haut Potentiel"
        description = "Ce profil présente une valeur vie client élevée avec une incertitude maîtrisée. Recommandation : Programme de fidélisation premium."
    elif is_risky:
        insight_type = "danger"
        icon = "fa-solid fa-triangle-exclamation"
        title = "Profil Risqué"
        description = "L'incertitude sur ce client est élevée ou sa valeur estimée est faible. Recommandation : Surveillance accrue et offres ciblées."
    else:
        insight_type = "warning"
        icon = "fa-solid fa-arrow-trend-down"
        title = "Profil Standard"
        description = "Client dans la moyenne du portefeuille. Recommandation : Maintenir la relation et identifier des opportunités de cross-sell."
    
    styles = {
        "success": {"bg": "bg-success/10", "border": "border-success/30", "text": "text-success"},
        "warning": {"bg": "bg-warning/10", "border": "border-warning/30", "text": "text-warning"},
        "danger": {"bg": "bg-danger/10", "border": "border-danger/30", "text": "text-danger"}
    }
    style = styles[insight_type]
    
    return html.Div(
        className=f"rounded-xl p-4 border flex items-start gap-4 {style['bg']} {style['border']} {style['text']}",
        children=[
            html.Div(
                className="p-2 rounded-lg bg-background/50",
                children=[html.I(className=f"{icon} text-lg")]
            ),
            html.Div([
                html.H4(title, className="font-semibold mb-1"),
                html.P(description, className="text-sm opacity-90")
            ])
        ]
    )


# ============================================================================
# PAGES
# ============================================================================

def create_portfolio_page():
    """Page Portfolio Insights"""
    return html.Div(
        className="space-y-8",
        children=[
            html.Div(
                className="mb-8 space-y-2",
                children=[
                    html.Div(
                        className="flex items-center gap-3",
                        children=[
                            html.Div(
                                className="p-2 rounded-lg bg-gradient-to-br from-primary to-emerald-400",
                                children=[
                                    html.I(className="fa-solid fa-layer-group text-white text-lg")
                                ]
                            ),
                            html.H1(
                                children=[
                                    "Portfolio ",
                                    html.Span("Insights", className="bg-gradient-to-r from-primary to-emerald-400 bg-clip-text", style={"WebkitBackgroundClip": "text", "WebkitTextFillColor": "transparent"})
                                ],
                                className="text-2xl font-black tracking-tight text-foreground",
                                style={"fontFamily": "'Montserrat', sans-serif"}
                            )
                        ]
                    ),
                    html.P("Vue d'ensemble des métriques CLV et performance du modèle Random Forest", 
                           className="text-muted-foreground text-sm")
                ]
            ),
            
            # KPIs Grid
            html.Div(
                className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4",
                children=[
                    create_kpi_card(
                        "CLV Moyen", 
                        f"${PORTFOLIO_STATS['mean_clv']:,.2f}",
                        f"Médiane: ${PORTFOLIO_STATS['median_clv']:,.0f}",
                        "fa-solid fa-coins", "primary"
                    ),
                    create_kpi_card(
                        "Clients Analysés",
                        f"{PORTFOLIO_STATS['total_customers']:,}",
                        "Base de données active",
                        "fa-solid fa-users", "info"
                    ),
                    create_kpi_card(
                        "Score R²",
                        f"{PORTFOLIO_STATS['r2_score']:.1%}",
                        "Performance du modèle",
                        "fa-solid fa-chart-line", "success"
                    ),
                    create_kpi_card(
                        "RMSE",
                        f"${PORTFOLIO_STATS['rmse']:,.0f}",
                        "Erreur quadratique moyenne",
                        "fa-solid fa-chart-bar", "warning"
                    ),
                ]
            ),
            
            # Charts Row 1
            html.Div(
                className="grid grid-cols-1 lg:grid-cols-2 gap-6",
                children=[
                    html.Div(
                        className="bg-card border border-info/30 rounded-xl p-6 glow-info",
                        children=[
                            html.Div(
                                className="flex items-center gap-2 mb-4",
                                children=[
                                    html.I(className="fa-solid fa-chart-area text-info"),
                                    html.H3("Distribution CLV", className="text-lg font-semibold text-foreground")
                                ]
                            ),
                            dcc.Graph(
                                figure=create_clv_distribution_chart(),
                                config={'displayModeBar': False, 'responsive': True},
                                style={'height': '350px'}
                            )
                        ]
                    ),
                    html.Div(
                        className="bg-card border border-primary/30 rounded-xl p-6 glow-primary",
                        children=[
                            html.Div(
                                className="flex items-center gap-2 mb-4",
                                children=[
                                    html.I(className="fa-solid fa-ranking-star text-primary"),
                                    html.H3("Importance des Variables", className="text-lg font-semibold text-foreground")
                                ]
                            ),
                            dcc.Graph(
                                figure=create_feature_importance_chart(),
                                config={'displayModeBar': False, 'responsive': True},
                                style={'height': '350px'}
                            )
                        ]
                    )
                ]
            ),
            
            # Chart Row 2
            html.Div(
                className="bg-card border border-success/30 rounded-xl p-6 glow-success",
                children=[
                    html.Div(
                        className="flex items-center gap-2 mb-4",
                        children=[
                            html.I(className="fa-solid fa-crosshairs text-success"),
                            html.H3("Prédictions vs Valeurs Réelles avec IC 95%", className="text-lg font-semibold text-foreground")
                        ]
                    ),
                    dcc.Graph(
                        figure=create_prediction_chart(),
                        config={'displayModeBar': False, 'responsive': True},
                        style={'height': '350px'}
                    )
                ]
            ),
            
            # Model Info
            html.Div(
                className="bg-card border border-border rounded-xl p-6",
                children=[
                    html.H3("Performance du Modèle", className="text-lg font-semibold text-foreground mb-6"),
                    html.Div(
                        className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4",
                        children=[
                            html.Div(
                                className="bg-background rounded-lg p-4 text-center",
                                children=[
                                    html.Span("R²", className="text-xs text-muted-foreground block mb-1"),
                                    html.Span(f"{PORTFOLIO_STATS['r2_score']:.4f}", className="font-mono font-bold text-primary")
                                ]
                            ),
                            html.Div(
                                className="bg-background rounded-lg p-4 text-center",
                                children=[
                                    html.Span("MAE", className="text-xs text-muted-foreground block mb-1"),
                                    html.Span(f"${PORTFOLIO_STATS['mae']:,.0f}", className="font-mono font-bold text-info")
                                ]
                            ),
                            html.Div(
                                className="bg-background rounded-lg p-4 text-center",
                                children=[
                                    html.Span("RMSE", className="text-xs text-muted-foreground block mb-1"),
                                    html.Span(f"${PORTFOLIO_STATS['rmse']:,.0f}", className="font-mono font-bold text-warning")
                                ]
                            ),
                            html.Div(
                                className="bg-background rounded-lg p-4 text-center",
                                children=[
                                    html.Span("Coverage", className="text-xs text-muted-foreground block mb-1"),
                                    html.Span(f"{PORTFOLIO_STATS['coverage_empirical']:.1f}%", className="font-mono font-bold text-success")
                                ]
                            ),
                            html.Div(
                                className="bg-background rounded-lg p-4 text-center",
                                children=[
                                    html.Span("Target", className="text-xs text-muted-foreground block mb-1"),
                                    html.Span(f"{PORTFOLIO_STATS['coverage_target']}%", className="font-mono font-bold text-muted-foreground")
                                ]
                            ),
                            html.Div(
                                className="bg-background rounded-lg p-4 text-center",
                                children=[
                                    html.Span("Largeur IC", className="text-xs text-muted-foreground block mb-1"),
                                    html.Span(f"${PORTFOLIO_STATS['mean_interval_width']:,}", className="font-mono font-bold text-secondary")
                                ]
                            ),
                        ]
                    )
                ]
            )
        ]
    )


def create_simulator_page():
    """Page Smart Simulator - React style"""
    return html.Div(
        className="space-y-8",
        children=[
            html.Div(
                className="mb-8 space-y-2",
                children=[
                    html.Div(
                        className="flex items-center gap-3",
                        children=[
                            html.Div(
                                className="p-2 rounded-lg bg-gradient-to-br from-warning to-amber-400",
                                children=[
                                    html.I(className="fa-solid fa-wand-magic-sparkles text-white text-lg")
                                ]
                            ),
                            html.H1(
                                children=[
                                    "Smart ",
                                    html.Span("Simulator", className="bg-gradient-to-r from-warning to-amber-400 bg-clip-text", style={"WebkitBackgroundClip": "text", "WebkitTextFillColor": "transparent"})
                                ],
                                className="text-2xl font-black tracking-tight text-foreground",
                                style={"fontFamily": "'Montserrat', sans-serif"}
                            )
                        ]
                    ),
                    html.P("Prédiction CLV personnalisée avec intervalle de confiance Bootstrap 95%", 
                           className="text-muted-foreground text-sm")
                ]
            ),
            
            # Info Banner - like React
            html.Div(
                className="bg-info/10 border border-info/30 rounded-xl p-4 flex items-start gap-3",
                children=[
                    html.I(className="fa-solid fa-lightbulb text-info mt-0.5"),
                    html.Div([
                        html.P("Comment ça marche ?", className="font-medium text-foreground mb-1"),
                        html.P(
                            "Remplissez le formulaire avec les caractéristiques du client. Notre modèle Random Forest calcule une prédiction CLV avec intervalle de confiance à 95% via Bootstrap (1500 itérations).",
                            className="text-sm text-muted-foreground"
                        )
                    ])
                ]
            ),
            
            html.Div(
                className="grid grid-cols-1 lg:grid-cols-2 gap-8",
                children=[
                    html.Div(
                        className="bg-card border border-primary/30 rounded-xl p-6 glow-primary",
                        children=[
                            html.Div(
                                className="flex items-center gap-2 mb-6",
                                children=[
                                    html.I(className="fa-solid fa-sliders text-primary"),
                                    html.H3("Paramètres Client", className="text-lg font-semibold text-foreground")
                                ]
                            ),
                            
                            html.Div(
                                className="grid grid-cols-1 md:grid-cols-2 gap-6",
                                children=[
                                    # Coverage
                                    html.Div([
                                        html.Label("Couverture", className="block text-sm font-medium text-muted-foreground mb-2"),
                                        dcc.Dropdown(
                                            id='input-coverage',
                                            options=[
                                                {'label': 'Basic', 'value': 'Basic'},
                                                {'label': 'Extended', 'value': 'Extended'},
                                                {'label': 'Premium', 'value': 'Premium'}
                                            ],
                                            value='Extended',
                                            clearable=False,
                                            className="dash-dropdown"
                                        )
                                    ]),
                                    
                                    # Vehicle Class
                                    html.Div([
                                        html.Label("Classe de Véhicule", className="block text-sm font-medium text-muted-foreground mb-2"),
                                        dcc.Dropdown(
                                            id='input-vehicle',
                                            options=[
                                                {'label': 'Two-Door Car', 'value': 'Two-Door Car'},
                                                {'label': 'Four-Door Car', 'value': 'Four-Door Car'},
                                                {'label': 'SUV', 'value': 'SUV'},
                                                {'label': 'Sports Car', 'value': 'Sports Car'},
                                                {'label': 'Luxury Car', 'value': 'Luxury Car'},
                                                {'label': 'Luxury SUV', 'value': 'Luxury SUV'}
                                            ],
                                            value='Four-Door Car',
                                            clearable=False,
                                            className="dash-dropdown"
                                        )
                                    ]),
                                    
                                    # Employment
                                    html.Div([
                                        html.Label("Statut d'Emploi", className="block text-sm font-medium text-muted-foreground mb-2"),
                                        dcc.Dropdown(
                                            id='input-employment',
                                            options=[
                                                {'label': 'Employed', 'value': 'Employed'},
                                                {'label': 'Unemployed', 'value': 'Unemployed'},
                                                {'label': 'Medical Leave', 'value': 'Medical Leave'},
                                                {'label': 'Disabled', 'value': 'Disabled'},
                                                {'label': 'Retired', 'value': 'Retired'}
                                            ],
                                            value='Employed',
                                            clearable=False,
                                            className="dash-dropdown"
                                        )
                                    ]),
                                    
                                    # Education
                                    html.Div([
                                        html.Label("Éducation", className="block text-sm font-medium text-muted-foreground mb-2"),
                                        dcc.Dropdown(
                                            id='input-education',
                                            options=[
                                                {'label': 'High School or Below', 'value': 'High School or Below'},
                                                {'label': 'College', 'value': 'College'},
                                                {'label': 'Bachelor', 'value': 'Bachelor'},
                                                {'label': 'Master', 'value': 'Master'},
                                                {'label': 'Doctor', 'value': 'Doctor'}
                                            ],
                                            value='Bachelor',
                                            clearable=False,
                                            className="dash-dropdown"
                                        )
                                    ]),
                                    
                                    # Gender
                                    html.Div([
                                        html.Label("Genre", className="block text-sm font-medium text-muted-foreground mb-2"),
                                        dcc.Dropdown(
                                            id='input-gender',
                                            options=[
                                                {'label': 'Male', 'value': 'M'},
                                                {'label': 'Female', 'value': 'F'}
                                            ],
                                            value='M',
                                            clearable=False,
                                            className="dash-dropdown"
                                        )
                                    ]),
                                    
                                    # Marital Status
                                    html.Div([
                                        html.Label("Statut Marital", className="block text-sm font-medium text-muted-foreground mb-2"),
                                        dcc.Dropdown(
                                            id='input-marital',
                                            options=[
                                                {'label': 'Single', 'value': 'Single'},
                                                {'label': 'Married', 'value': 'Married'},
                                                {'label': 'Divorced', 'value': 'Divorced'}
                                            ],
                                            value='Married',
                                            clearable=False,
                                            className="dash-dropdown"
                                        )
                                    ]),
                                    
                                    # Monthly Premium
                                    html.Div([
                                        html.Label("Prime Mensuelle ($)", className="block text-sm font-medium text-muted-foreground mb-2"),
                                        dcc.Input(
                                            id='input-premium',
                                            type='number',
                                            value=120,
                                            min=50,
                                            max=500,
                                            className="w-full px-4 py-3 bg-background border border-border rounded-lg text-foreground focus:outline-none focus:ring-2 focus:ring-primary"
                                        )
                                    ]),
                                    
                                    # Total Claims
                                    html.Div([
                                        html.Label("Total Réclamations ($)", className="block text-sm font-medium text-muted-foreground mb-2"),
                                        dcc.Input(
                                            id='input-claims',
                                            type='number',
                                            value=500,
                                            min=0,
                                            className="w-full px-4 py-3 bg-background border border-border rounded-lg text-foreground focus:outline-none focus:ring-2 focus:ring-primary"
                                        )
                                    ]),
                                    
                                    # Number of Policies
                                    html.Div([
                                        html.Label("Nombre de Polices", className="block text-sm font-medium text-muted-foreground mb-2"),
                                        dcc.Input(
                                            id='input-policies',
                                            type='number',
                                            value=3,
                                            min=1,
                                            max=10,
                                            className="w-full px-4 py-3 bg-background border border-border rounded-lg text-foreground focus:outline-none focus:ring-2 focus:ring-primary"
                                        )
                                    ]),
                                    
                                    # Income
                                    html.Div([
                                        html.Label("Revenu Annuel ($)", className="block text-sm font-medium text-muted-foreground mb-2"),
                                        dcc.Input(
                                            id='input-income',
                                            type='number',
                                            value=50000,
                                            min=0,
                                            className="w-full px-4 py-3 bg-background border border-border rounded-lg text-foreground focus:outline-none focus:ring-2 focus:ring-primary"
                                        )
                                    ]),
                                ]
                            ),
                            
                            # Submit Button
                            html.Button(
                                children=[
                                    html.I(className="fa-solid fa-bullseye mr-2"),
                                    "Prédire CLV"
                                ],
                                id='btn-predict',
                                className="w-full mt-6 px-6 py-4 bg-gradient-to-r from-primary to-info text-white font-semibold rounded-lg hover:opacity-90 transition-all duration-300 hover:shadow-lg hover:shadow-primary/30 cursor-pointer"
                            )
                        ]
                    ),
                    
                    # Results
                    html.Div(
                        id='prediction-result',
                        className="space-y-6"
                    )
                ]
            )
        ]
    )


def create_audit_page():
    """Page Scientific Audit - React style"""
    # Check validation status
    validation_passed = STATISTICAL_TESTS["z_test_coverage"]["passed"] and PORTFOLIO_STATS["coverage_empirical"] >= 93
    
    return html.Div(
        className="space-y-8",
        children=[
            # Header - React style with icon and colored text
            html.Div(
                className="mb-8 space-y-2",
                children=[
                    html.Div(
                        className="flex items-center gap-3",
                        children=[
                            html.Div(
                                className="p-2 rounded-lg bg-gradient-to-br from-info to-cyan-400",
                                children=[
                                    html.I(className="fa-solid fa-microscope text-white text-lg")
                                ]
                            ),
                            html.H1(
                                children=[
                                    "Audit ",
                                    html.Span("Scientifique", className="bg-gradient-to-r from-info to-cyan-400 bg-clip-text", style={"WebkitBackgroundClip": "text", "WebkitTextFillColor": "transparent"})
                                ],
                                className="text-2xl font-black tracking-tight text-foreground",
                                style={"fontFamily": "'Montserrat', sans-serif"}
                            )
                        ]
                    ),
                    html.P("Validation statistique du modèle et certification de fiabilité", 
                           className="text-muted-foreground text-sm")
                ]
            ),
            
            # Overall Validation Certificate - React style
            html.Div(
                className=f"bg-card border-2 {'border-success/50 glow-success' if validation_passed else 'border-warning/50 glow-warning'} rounded-xl p-8",
                children=[
                    html.Div(
                        className="flex flex-col lg:flex-row items-center gap-6",
                        children=[
                            # Status icon
                            html.Div(
                                className=f"p-6 rounded-full {'bg-success/20' if validation_passed else 'bg-warning/20'}",
                                children=[
                                    html.I(
                                        className=f"fa-solid {'fa-circle-check text-success' if validation_passed else 'fa-circle-xmark text-warning'} text-5xl"
                                    )
                                ]
                            ),
                            
                            # Info
                            html.Div(
                                className="flex-1 text-center lg:text-left",
                                children=[
                                    html.H2(
                                        "Modèle Certifié Valide" if validation_passed else "Validation Partielle",
                                        className="text-2xl font-bold text-foreground mb-2"
                                    ),
                                    html.P(
                                        "Les intervalles de prédiction à 95% sont statistiquement calibrés. Le Z-test confirme que la couverture observée ne diffère pas significativement de la cible." if validation_passed else "Certains tests indiquent des écarts. Une analyse approfondie est recommandée.",
                                        className="text-muted-foreground mb-4 max-w-xl"
                                    ),
                                    html.Div(
                                        className="grid grid-cols-3 gap-4 max-w-md mx-auto lg:mx-0",
                                        children=[
                                            html.Div(
                                                className="text-center",
                                                children=[
                                                    html.P(f"{PORTFOLIO_STATS['coverage_empirical']:.1f}%", className="text-2xl font-bold font-mono text-success"),
                                                    html.P("Couverture", className="text-xs text-muted-foreground")
                                                ]
                                            ),
                                            html.Div(
                                                className="text-center",
                                                children=[
                                                    html.P("95%", className="text-2xl font-bold font-mono text-info"),
                                                    html.P("Cible", className="text-xs text-muted-foreground")
                                                ]
                                            ),
                                            html.Div(
                                                className="text-center",
                                                children=[
                                                    html.P("1692/1795", className="text-2xl font-bold font-mono text-warning"),
                                                    html.P("Points couverts", className="text-xs text-muted-foreground")
                                                ]
                                            )
                                        ]
                                    )
                                ]
                            ),
                            
                            # Certificate badge
                            html.Div(
                                className="hidden lg:block",
                                children=[
                                    html.Div(
                                        className=f"px-6 py-4 rounded-lg border-2 transform rotate-3 {'bg-success/10 border-success/40 text-success' if validation_passed else 'bg-warning/10 border-warning/40 text-warning'}",
                                        children=[
                                            html.P("CERTIFIED" if validation_passed else "REVIEW", className="text-xl font-bold font-mono"),
                                            html.P("Bootstrap 95% IC", className="text-xs opacity-70")
                                        ]
                                    )
                                ]
                            )
                        ]
                    )
                ]
            ),
            
            # Normality Tests Section
            html.Div(
                className="space-y-4",
                children=[
                    html.Div(
                        className="flex items-center gap-3",
                        children=[
                            html.I(className="fa-solid fa-flask text-info"),
                            html.H3("Tests de Normalité des Résidus", className="text-lg font-semibold text-foreground")
                        ]
                    ),
                    # Info box
                    html.Div(
                        className="bg-info/5 border border-info/20 rounded-lg p-4 flex items-start gap-3",
                        children=[
                            html.I(className="fa-solid fa-info-circle text-info flex-shrink-0 mt-0.5"),
                            html.Div([
                                html.P("Pourquoi ces tests sont \"rejetés\" mais le modèle reste valide ?", className="font-medium text-foreground mb-1"),
                                html.P([
                                    "Les tests de Shapiro-Wilk et KS vérifient la normalité des résidus. Le rejet de H0 signifie simplement que les erreurs ne suivent pas une loi normale parfaite (queues lourdes observées). Cependant, la méthode Bootstrap est ",
                                    html.Strong("non-paramétrique"),
                                    " et ne nécessite pas cette hypothèse. C'est précisément pourquoi nous l'avons choisie."
                                ], className="text-sm text-muted-foreground")
                            ])
                        ]
                    ),
                    
                    # Statistical Tests
                    html.Div(
                        className="grid grid-cols-1 lg:grid-cols-3 gap-6",
                        children=[
                            create_stat_badge(
                                "Test de Shapiro-Wilk",
                                STATISTICAL_TESTS["shapiro_wilk"]["passed"],
                                STATISTICAL_TESTS["shapiro_wilk"]["statistic"],
                                STATISTICAL_TESTS["shapiro_wilk"]["p_value"],
                                "Résidus non-normaux (attendu pour Random Forest)"
                            ),
                            create_stat_badge(
                                "Test de Kolmogorov-Smirnov",
                                STATISTICAL_TESTS["kolmogorov_smirnov"]["passed"],
                                STATISTICAL_TESTS["kolmogorov_smirnov"]["statistic"],
                                STATISTICAL_TESTS["kolmogorov_smirnov"]["p_value"],
                                "Distribution non-normale confirmée"
                            ),
                            create_stat_badge(
                                "Z-Test Couverture",
                                STATISTICAL_TESTS["z_test_coverage"]["passed"],
                                STATISTICAL_TESTS["z_test_coverage"]["statistic"],
                                STATISTICAL_TESTS["z_test_coverage"]["p_value"],
                                "Couverture IC conforme à 95%"
                            ),
                        ]
                    ),

                    html.Div(
                        className=(
                            "mt-6 border rounded-lg p-4 flex items-start gap-3 "
                            + ("bg-success/5 border-success/20" if STATISTICAL_TESTS["z_test_coverage"]["passed"] else "bg-warning/5 border-warning/20")
                        ),
                        children=[
                            html.I(
                                className=(
                                    "fa-solid "
                                    + ("fa-circle-check text-success" if STATISTICAL_TESTS["z_test_coverage"]["passed"] else "fa-triangle-exclamation text-warning")
                                    + " mt-0.5"
                                )
                            ),
                            html.Div([
                                html.P(
                                    "Statistiquement Validé" if STATISTICAL_TESTS["z_test_coverage"]["passed"] else "Validation à surveiller",
                                    className="font-semibold text-foreground"
                                ),
                                html.P(
                                    f"Z-test (p-value = {STATISTICAL_TESTS['z_test_coverage']['p_value']:.3f}) : la couverture observée n'est pas significativement différente de 95%.",
                                    className="text-sm text-muted-foreground"
                                )
                            ])
                        ]
                    ),
                ]
            ),
            
            # Model Info
            html.Div(
                className="bg-card border border-border rounded-xl p-6",
                children=[
                    html.H3("Informations du Modèle", className="text-lg font-semibold text-foreground mb-6"),
                    html.Div(
                        className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4",
                        children=[
                            html.Div(
                                className="bg-background rounded-lg p-4 text-center",
                                children=[
                                    html.Span("Type", className="text-xs text-muted-foreground block mb-1"),
                                    html.Span("Random Forest", className="font-mono font-bold text-primary")
                                ]
                            ),
                            html.Div(
                                className="bg-background rounded-lg p-4 text-center",
                                children=[
                                    html.Span("Estimateurs", className="text-xs text-muted-foreground block mb-1"),
                                    html.Span("100", className="font-mono font-bold text-info")
                                ]
                            ),
                            html.Div(
                                className="bg-background rounded-lg p-4 text-center",
                                children=[
                                    html.Span("Profondeur Max", className="text-xs text-muted-foreground block mb-1"),
                                    html.Span("15", className="font-mono font-bold text-secondary")
                                ]
                            ),
                            html.Div(
                                className="bg-background rounded-lg p-4 text-center",
                                children=[
                                    html.Span("Bootstrap", className="text-xs text-muted-foreground block mb-1"),
                                    html.Span("1500 iter", className="font-mono font-bold text-warning")
                                ]
                            ),
                            html.Div(
                                className="bg-background rounded-lg p-4 text-center",
                                children=[
                                    html.Span("IC Level", className="text-xs text-muted-foreground block mb-1"),
                                    html.Span("95%", className="font-mono font-bold text-success")
                                ]
                            ),
                            html.Div(
                                className="bg-background rounded-lg p-4 text-center",
                                children=[
                                    html.Span("Mode", className="text-xs text-muted-foreground block mb-1"),
                                    html.Span(
                                        "ML RÉEL" if MODEL_LOADED else "SIMULATION",
                                        className=f"font-mono font-bold {'text-success' if MODEL_LOADED else 'text-warning'}"
                                    )
                                ]
                            ),
                        ]
                    )
                ]
            ),
            
            # Diagnostic Charts
            html.Div(
                className="grid grid-cols-1 lg:grid-cols-2 gap-6",
                children=[
                    html.Div(
                        className="bg-card border border-info/30 rounded-xl p-6 glow-info",
                        children=[
                            html.Div(
                                className="flex items-center gap-2 mb-4",
                                children=[
                                    html.I(className="fa-solid fa-chart-simple text-info"),
                                    html.H3("QQ-Plot des Résidus", className="text-lg font-semibold text-foreground")
                                ]
                            ),
                            dcc.Graph(
                                figure=create_qq_plot(),
                                config={'displayModeBar': False, 'responsive': True},
                                style={'height': '350px'}
                            )
                        ]
                    ),
                    html.Div(
                        className="bg-card border border-primary/30 rounded-xl p-6 glow-primary",
                        children=[
                            html.Div(
                                className="flex items-center gap-2 mb-4",
                                children=[
                                    html.I(className="fa-solid fa-wave-square text-primary"),
                                    html.H3("Distribution des Résidus", className="text-lg font-semibold text-foreground")
                                ]
                            ),
                            dcc.Graph(
                                figure=create_residuals_chart(),
                                config={'displayModeBar': False, 'responsive': True},
                                style={'height': '350px'}
                            )
                        ]
                    )
                ]
            ),

            # Graphique de performance (Réel vs Prédit) + enveloppe d'incertitude
            html.Div(
                className="bg-card border border-border rounded-xl p-6",
                children=[
                    html.Div(
                        className="flex items-center gap-2 mb-4",
                        children=[
                            html.I(className="fa-solid fa-bullseye text-info"),
                            html.H3("Valeurs Réelles vs Valeurs Prédites", className="text-lg font-semibold text-foreground"),
                        ],
                    ),
                    dcc.Graph(
                        figure=create_real_vs_predicted_scatter(),
                        config={'displayModeBar': False, 'responsive': True},
                        style={'height': '380px'}
                    )
                ]
            ),
            
            # Methodology
            html.Div(
                className="bg-card border border-success/30 rounded-xl p-6 glow-success",
                children=[
                    html.Div(
                        className="flex items-center gap-2 mb-4",
                        children=[
                            html.I(className="fa-solid fa-book text-success"),
                            html.H3("Méthodologie Bootstrap", className="text-lg font-semibold text-foreground")
                        ]
                    ),
                    html.Div(
                        className="prose prose-invert max-w-none",
                        children=[
                            html.P([
                                "Les intervalles de confiance sont construits via ",
                                html.Span("Bootstrap résiduel", className="text-primary font-semibold"),
                                " avec 1500 itérations. Cette méthode est robuste aux violations de normalité des résidus,",
                                " ce qui est particulièrement adapté aux modèles non-paramétriques comme Random Forest."
                            ], className="text-muted-foreground mb-4"),
                            html.Div(
                                className="bg-background rounded-lg p-4 font-mono text-sm text-muted-foreground",
                                children=[
                                    html.Code("IC_95% = [percentile(ŷ + ε*, 2.5%), percentile(ŷ + ε*, 97.5%)]"),
                                    html.Br(),
                                    html.Code("où ε* = échantillon bootstrap des résidus log-transformés")
                                ]
                            )
                        ]
                    )
                ]
            )
        ]
    )


# ============================================================================
# LAYOUT PRINCIPAL
# ============================================================================

app.layout = html.Div(
    className="flex min-h-screen",
    children=[
        dcc.Location(id='url', refresh=False),
        create_sidebar(),
        html.Main(
            className="flex-1 ml-64 p-8",
            children=[
                html.Div(id='page-content')
            ]
        )
    ]
)


# ============================================================================
# CALLBACKS
# ============================================================================

@callback(
    Output('page-content', 'children'),
    Input('url', 'pathname')
)
def display_page(pathname):
    if pathname == '/simulator':
        return create_simulator_page()
    elif pathname == '/audit':
        return create_audit_page()
    else:
        return create_portfolio_page()


@callback(
    Output('prediction-result', 'children'),
    Input('btn-predict', 'n_clicks'),
    State('input-coverage', 'value'),
    State('input-vehicle', 'value'),
    State('input-employment', 'value'),
    State('input-education', 'value'),
    State('input-gender', 'value'),
    State('input-marital', 'value'),
    State('input-premium', 'value'),
    State('input-claims', 'value'),
    State('input-policies', 'value'),
    State('input-income', 'value'),
    prevent_initial_call=True
)
def predict_clv(n_clicks, coverage, vehicle, employment, education, gender,
                marital, premium, claims, policies, income):
    
    prediction, lower, upper = predict_with_model(
        coverage, vehicle, employment, education, gender,
        marital, premium, claims, policies, income
    )
    
    # Prediction visualization
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=['CLV Prédit'],
        y=[upper - lower],
        base=[lower],
        marker=dict(color='rgba(59, 130, 246, 0.3)'),
        name='Intervalle 95%',
        hovertemplate=f'IC 95%: [${lower:,.0f} - ${upper:,.0f}]<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=['CLV Prédit'],
        y=[prediction],
        mode='markers',
        marker=dict(color='#3b82f6', size=20, symbol='diamond'),
        name='Prédiction',
        hovertemplate=f'CLV: ${prediction:,.0f}<extra></extra>'
    ))
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter', color='#9ca3af'),
        yaxis=dict(title='Valeur ($)', gridcolor='rgba(30, 58, 95, 0.3)', tickformat='$,.0f'),
        xaxis=dict(showticklabels=False),
        margin=dict(l=80, r=40, t=20, b=40),
        height=250,
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.02)
    )
    
    return [
        # Jauge de Gestion du Risque Professionnelle avec zone de confiance
        create_risk_management_gauge(prediction, lower, upper),
        
        # Dynamic Insight Card
        create_dynamic_insight(prediction, lower, upper),
        
        # Chart
        html.Div(
            className="bg-card border border-success/30 rounded-xl p-6 glow-success",
            children=[
                html.Div(
                    className="flex items-center gap-2 mb-4",
                    children=[
                        html.I(className="fa-solid fa-chart-column text-success"),
                        html.H3("Intervalle de Confiance Bootstrap 95%", className="text-lg font-semibold text-foreground")
                    ]
                ),
                dcc.Graph(
                    figure=fig,
                    config={'displayModeBar': False, 'responsive': True},
                    style={'height': '250px'}
                )
            ]
        ),
        
        # Input Summary
        html.Div(
            className="bg-card border border-info/30 rounded-xl p-6 glow-info",
            children=[
                html.Div(
                    className="flex items-center gap-2 mb-4",
                    children=[
                        html.I(className="fa-solid fa-list-check text-info"),
                        html.H3("Résumé des Paramètres", className="text-lg font-semibold text-foreground")
                    ]
                ),
                html.Div(
                    className="grid grid-cols-2 md:grid-cols-5 gap-3",
                    children=[
                        html.Div(
                            className="bg-background rounded-lg p-3 text-center",
                            children=[
                                html.Span("Couverture", className="text-xs text-muted-foreground block"),
                                html.Span(coverage, className="font-medium text-foreground text-sm")
                            ]
                        ),
                        html.Div(
                            className="bg-background rounded-lg p-3 text-center",
                            children=[
                                html.Span("Véhicule", className="text-xs text-muted-foreground block"),
                                html.Span(vehicle, className="font-medium text-foreground text-sm")
                            ]
                        ),
                        html.Div(
                            className="bg-background rounded-lg p-3 text-center",
                            children=[
                                html.Span("Prime", className="text-xs text-muted-foreground block"),
                                html.Span(f"${premium}/mois", className="font-medium text-foreground text-sm")
                            ]
                        ),
                        html.Div(
                            className="bg-background rounded-lg p-3 text-center",
                            children=[
                                html.Span("Polices", className="text-xs text-muted-foreground block"),
                                html.Span(str(policies), className="font-medium text-foreground text-sm")
                            ]
                        ),
                        html.Div(
                            className="bg-background rounded-lg p-3 text-center",
                            children=[
                                html.Span("Revenu", className="text-xs text-muted-foreground block"),
                                html.Span(f"${income:,}", className="font-medium text-foreground text-sm")
                            ]
                        ),
                    ]
                )
            ]
        )
    ]


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print(" CLV Prediction Dashboard avec Tailwind CSS")
    print(f"{'='*60}")
    print(f" Mode: {'ML RÉEL' if MODEL_LOADED else 'SIMULATION'}")
    print(f" URL: http://127.0.0.1:8057")
    
    app.run(debug=True, port=8057)