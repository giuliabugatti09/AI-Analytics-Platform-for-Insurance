import streamlit as st
import pandas as pd
import pydeck as pdk
import re
from utils.geo_utils import has_geo, risk_layer
from scipy.spatial import cKDTree
import numpy as np
import requests
from urllib.parse import urlparse

from components.Agente_Acidentes import renderizar_pagina_agente_acidentes # Importe a função que você criou

# ==============================================================================
# 1. CONFIGURAÇÃO DA PÁGINA E CSS (TEMA LARANJA & CLARO)
# ==============================================================================

st.set_page_config(
    page_title="Análise Exploratória - EDA",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS Unificado para um visual coeso
st.markdown("""
<style>
    /* --- CABEÇALHO PRINCIPAL --- */
    .main-header {
        font-size: 3.5rem;
        text-align: center;
        padding: 1rem 0;
        background: #ff5100;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
    }

    /* --- CONTAINERS E CARDS (TEMA CLARO) --- */
    .feature-card {
        background-color: #FFFFFF;
        border: 1px solid #FFDDC9;
        border-left: 6px solid #ff5100;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        color: #333;
    }
    .feature-card h3 {
        color: #D9480F;
        margin-top: 0;
    }

    /* --- BOTÕES --- */
    .stButton>button, .stDownloadButton>button {
        background-color: #ff5100;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        border: none;
        padding: 0.5rem 1rem;
        transition: all 0.3s;
    }
    .stButton>button:hover, .stDownloadButton>button:hover {
        background-color: #D9480F;
        box-shadow: 0 5px 15px rgba(255, 81, 0, 0.3);
    }
    div[data-testid="stAlertContainer"] {
    background-color: rgba(255, 81, 0, 0.15); /* Fundo laranja bem suave */
    }

    /* 2. O parágrafo (texto) dentro do st.info */
    div[data-testid="stAlertContentInfo"] p {
        color: #D9480F; /* Texto em um tom de laranja mais escuro para legibilidade */
    }
    div[data-testid="stAlertContentSuccess"] p {
        color: #D9480F; /* Texto em um tom de laranja mais escuro para legibilidade */
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header"> Mapa de Risco de Acidentes</div>', unsafe_allow_html=True)
st.markdown("<div style='text-align: center; font-size: 1.2em; color: #666;'>Visualize geograficamente as áreas de maior concentração de acidentes e identifique pontos críticos em um mapa de calor interativo.</div>", unsafe_allow_html=True)
st.markdown("---")


# ===============================
# Leitura do arquivo CSV fixo
# ===============================
CSV_PATH = "data/acidentes2025_todas_causas_tipos.csv"  # ajuste o caminho conforme necessário

try:
    file = open(CSV_PATH, "rb")
except FileNotFoundError:
    st.error(f"Arquivo não encontrado: {CSV_PATH}")
    st.stop()


# ===============================
# Processamento do CSV
# ===============================
df = pd.read_csv(file)

ok, lat_col, lon_col = has_geo(df)
if not ok:
    st.error("Não encontrei colunas de latitude e longitude.")
    st.stop()


def normalize_coord(val, coord_type):
    if pd.isnull(val):
        return None
    val_str = str(val).replace(" ", "")
    if val_str.count(",") > 1:
        parts = val_str.split(",")
        val_str = "".join(parts[:-1]) + "." + parts[-1]
    else:
        val_str = val_str.replace(",", ".")
    try:
        num = float(val_str)
        if coord_type == "lat" and -90 <= num <= 90:
            return num
        if coord_type == "lon" and -180 <= num <= 180:
            return num
    except Exception:
        return None
    return None


def is_in_brazil(lat, lon):
    return (
        -33.75 <= lat <= 5.27 and
        -73.99 <= lon <= -34.79
    )


# Conversão e normalização para float
df[lat_col] = df[lat_col].apply(lambda x: normalize_coord(x, "lat"))
df[lon_col] = df[lon_col].apply(lambda x: normalize_coord(x, "lon"))
df = df.dropna(subset=[lat_col, lon_col])

# Filtra apenas coordenadas dentro do Brasil
df = df[df.apply(lambda row: is_in_brazil(row[lat_col], row[lon_col]), axis=1)]

# Adiciona coluna de acidentes próximos (raio ~0.001 grau, ~100m)
RAIO = 0.001
coords = df[[lat_col, lon_col]].to_numpy()
if len(coords) > 0:
    tree = cKDTree(coords)
    counts = tree.query_ball_point(coords, r=RAIO)
    df["acidentes_proximos"] = [len(c)-1 for c in counts]  # -1 para não contar ele mesmo
else:
    df["acidentes_proximos"] = 0

st.success(f"Detectei colunas geográficas: {lat_col}, {lon_col}. Total de coordenadas utilizadas no mapa: {len(df)}")


# ===============================
# Visualização com PyDeck
# ===============================
st.subheader("Mapa de Risco (Heatmap)")

heatmap_layer = pdk.Layer(
    "HeatmapLayer",
    data=df,
    get_position=[lon_col, lat_col],
    radiusPixels=60,
)

scatter_layer = pdk.Layer(
    "ScatterplotLayer",
    data=df,
    get_position=[lon_col, lat_col],
    get_radius=80,
    get_color=[255, 0, 0, 120],
    pickable=True,
)

view_state = pdk.ViewState(
    latitude=df[lat_col].mean() if len(df) > 0 else 0,
    longitude=df[lon_col].mean() if len(df) > 0 else 0,
    zoom=10,
    pitch=0,
)

deck = pdk.Deck(
    layers=[heatmap_layer, scatter_layer],
    initial_view_state=view_state,
    tooltip={
        "text": f"Lat: {{{lat_col}}}\nLon: {{{lon_col}}}\nAcidentes próximos: {{acidentes_proximos}}"
    }
)

st.pydeck_chart(deck)

st.info("Passe o mouse sobre os pontos vermelhos para ver as coordenadas e o número de acidentes próximos.")

# --- Botão para acionar o Agente IA ---
if st.sidebar.button("Conversar com Agente IA"):
    
    # A mágica acontece aqui!
    @st.dialog("🤖", width="large")
    def agent_dialog():
        """Define o conteúdo do pop-up."""
        # Chamamos a função que renderiza a interface completa do chat
        renderizar_pagina_agente_acidentes()
    agent_dialog()  # Abre o pop-up quando o botão é clicado
st.sidebar.divider()
