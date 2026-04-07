import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tempfile
import requests
from urllib.parse import urlparse

from utils.data_utils import load_data, basic_profile, filter_dataframe, to_excel_download

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
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">Análise Exploratória de Dados (EDA)</div>', unsafe_allow_html=True)
st.markdown("<div style='text-align: center; font-size: 1.2em; color: #666;'>Visualizando e resumindo as principais características e padrões dos seus dados.</div>", unsafe_allow_html=True)
st.markdown("---")

# ===============================
# Upload de dados
# ===============================
with st.expander("Upload de dados", expanded=True):
    file = st.file_uploader("Envie seu CSV", type=["csv"])
    use_example = st.checkbox("Usar dataset de exemplo", value=not bool(file))

if use_example:
    df = pd.DataFrame({
        "hora": np.random.randint(0,24,200),
        "chuva": np.random.choice([0,1],200,p=[0.7,0.3]),
        "velocidade": np.random.normal(60,10,200).round(1),
        "feridos": np.random.poisson(0.6,200),
        "fatal": np.random.choice([0,1],200,p=[0.9,0.1]),
        "lat": -23.5 + np.random.rand(200)*0.2,
        "lon": -46.6 + np.random.rand(200)*0.2,
    })
else:
    df = load_data(file) if file else None

if df is None:
    st.info("Envie um CSV ou marque 'Usar dataset de exemplo'.")
    st.stop()

st.write("**Dimensão:**", df.shape)
st.dataframe(df.head(50), use_container_width=True)

# ===============================
# Download amostra
# ===============================
st.download_button(
    "⬇Baixar amostra (Excel)",
    data=to_excel_download(df.head(500)),
    file_name="amostra_eda.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

# ===============================
# Visão geral dos dados
# ===============================
st.subheader("Visão geral dos dados")
st.write(df.describe(include='all'))
st.text(str(df.info()))
st.write("Valores ausentes:", df.isna().sum().sum())

# ===============================
# Gráficos exploratórios
# ===============================
def plot_hbar(df, column, top_n=None, title=None):
    st.subheader(title or column)
    counts = df[column].value_counts()
    if top_n:
        counts = counts.head(top_n)
    fig, ax = plt.subplots()
    counts.sort_values().plot(kind='barh', ax=ax)
    fig.tight_layout()
    st.pyplot(fig)

def plot_hist(df, column, bins=10, color='steelblue', title=None):
    st.subheader(title or column)
    fig, ax = plt.subplots()
    df[column].plot.hist(bins=bins, color=color, ax=ax)
    fig.tight_layout()
    st.pyplot(fig)

def plot_box(df, x, y, title=None):
    st.subheader(title or f"{y} por {x}")
    fig, ax = plt.subplots()
    sns.boxplot(data=df, x=x, y=y, ax=ax)
    plt.xticks(rotation=45)
    fig.tight_layout()
    st.pyplot(fig)

def plot_countplot(df, x, hue=None, title=None):
    st.subheader(title or x)
    fig, ax = plt.subplots()
    sns.countplot(data=df, x=x, hue=hue, ax=ax)
    plt.xticks(rotation=45)
    fig.tight_layout()
    st.pyplot(fig)

# Colunas opcionais
if 'uf' in df.columns:
    plot_hbar(df,'uf', title="Quantidade de registros por UF")

if 'tipo_acidente' in df.columns:
    plot_hbar(df,'tipo_acidente', top_n=10, title="Top 10 tipos de acidente")

if 'estado_fisico' in df.columns:
    plot_hbar(df,'estado_fisico', top_n=4, title="Gravidade dos acidentes")

if 'idade' in df.columns:
    plot_hist(df[df['idade'].between(0,120)], 'idade', bins=24, title="Distribuição da idade das vítimas")
    plot_box(df[df['idade'].between(0,120)], x='estado_fisico', y='idade', title="Boxplot: Idade por estado físico da vítima")

if 'sexo' in df.columns and 'estado_fisico' in df.columns:
    plot_countplot(df, x='sexo', hue='estado_fisico', title="Estado físico das vítimas por sexo")

if 'dia_semana' in df.columns:
    plot_hbar(df,'dia_semana', title="Acidentes por dia da semana")

if 'fase_dia' in df.columns:
    plot_hbar(df,'fase_dia', title="Acidentes por fase do dia")

if 'condicao_metereologica' in df.columns:
    plot_hbar(df,'condicao_metereologica', title="Acidentes por condição meteorológica")

if 'tipo_pista' in df.columns:
    plot_hbar(df,'tipo_pista', title="Acidentes por tipo de pista")

if 'tipo_veiculo' in df.columns:
    plot_hbar(df,'tipo_veiculo', top_n=10, title="Tipos de veículos envolvidos (Top 10)")

if 'ano_fabricacao_veiculo' in df.columns:
    df_year = df[df['ano_fabricacao_veiculo'].notna() & (df['ano_fabricacao_veiculo']>=1970)]
    plot_hist(df_year, 'ano_fabricacao_veiculo', bins=range(1970, int(df_year['ano_fabricacao_veiculo'].max())+2), color='darkgreen', title="Ano de fabricação dos veículos envolvidos")

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

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

