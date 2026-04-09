import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils.data_utils import load_data, to_excel_download
from components.Agente_Acidentes import renderizar_pagina_agente_acidentes

# ==============================================================================
# 1. PAGE CONFIGURATION & UI STYLING
# ==============================================================================
st.set_page_config(page_title="EDA - Smart Insurance Analytics", layout="wide")

# Unified CSS for branding (Sompo Orange Theme)
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        text-align: center;
        color: #ff5100;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .stMetric {
        background-color: rgba(255, 81, 0, 0.05);
        padding: 15px;
        border-radius: 10px;
        border: 1px solid rgba(255, 81, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. PLOTTING UTILITIES
# ==============================================================================
def set_plot_style():
    """Sets global aesthetic parameters for Matplotlib/Seaborn."""
    sns.set_theme(style="whitegrid")
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.titleweight'] = 'bold'
    plt.rcParams['figure.facecolor'] = 'white'

def render_chart(fig, title):
    """Utility to display a plot in Streamlit and clear memory."""
    st.markdown(f"#### {title}")
    st.pyplot(fig)
    plt.close(fig)

# ==============================================================================
# 3. SIDEBAR CONTROLS & DATA INGESTION
# ==============================================================================
st.sidebar.title("Data Management")

with st.sidebar.expander("📂 Data Source", expanded=True):
    file = st.file_uploader("Upload Incident CSV", type=["csv"])
    use_example = st.checkbox("Use Demo Dataset", value=not bool(file))

# AI Agent Integration
if st.sidebar.button("🤖 Launch AI Assistant", use_container_width=True):
    @st.dialog("Smart Insurance Agent", width="large")
    def agent_dialog():
        """Renders the AI chat interface in a modal dialog."""
        renderizar_pagina_agente_acidentes()
    agent_dialog()

st.sidebar.divider()

# Load Data Logic
if use_example:
    # Synthetic data generated for demonstration purposes
    df = pd.DataFrame({
        "uf": np.random.choice(['SP', 'RJ', 'MG', 'PR', 'SC'], 500),
        "estado_fisico": np.random.choice(['Sem Vítimas', 'Feridos', 'Fatais'], 500, p=[0.6, 0.35, 0.05]),
        "idade": np.random.randint(18, 85, 500),
        "fase_dia": np.random.choice(['Full Day', 'Full Night', 'Dawn'], 500),
        "tipo_acidente": np.random.choice(['Frontal Collision', 'Rollover', 'Pedestrian'], 500)
    })
else:
    df = load_data(file) if file else None

if df is None:
    st.warning("⚠️ Waiting for data input. Please upload a CSV or use the Demo mode.")
    st.stop()

# ==============================================================================
# 4. DASHBOARD HEADER & EXECUTIVE KPIs
# ==============================================================================
st.markdown('<div class="main-header">Exploratory Data Analysis</div>', unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #666;'>Risk profiling and statistical intelligence for road safety management.</p>", unsafe_allow_html=True)

# Key Performance Indicators (KPIs)
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Records", f"{df.shape[0]:,}")
with col2:
    if 'estado_fisico' in df.columns:
        fatal_pct = (df['estado_fisico'] == 'Fatais').mean() * 100
        st.metric("Lethality Rate", f"{fatal_pct:.2f}%")
with col3:
    st.metric("Total Features", df.shape[1])
with col4:
    missing = df.isna().sum().sum()
    st.metric("Missing Values", missing)

st.divider()

# ==============================================================================
# 5. ANALYTICAL WORKSPACE (TABBED INTERFACE)
# ==============================================================================
set_plot_style()
tabs = st.tabs(["🌍 Geospatial & Severity", "👤 Victim Profile", "📉 Incident Context"])

# --- TAB 1: Geography and Severity ---
with tabs[0]:
    c1, c2 = st.columns(2)
    
    if 'uf' in df.columns:
        with c1:
            fig, ax = plt.subplots()
            df['uf'].value_counts().head(10).plot(kind='bar', ax=ax, color='#ff5100')
            ax.set_title("Top 10 Regions by Incident Volume")
            render_chart(fig, "Regional Distribution")

    if 'estado_fisico' in df.columns:
        with c2:
            fig, ax = plt.subplots()
            df['estado_fisico'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax, colors=['#ffbd9b', '#ff8142', '#ff5100'])
            ax.set_ylabel("")
            render_chart(fig, "Severity Breakdown")

# --- TAB 2: Victim Demographics ---
with tabs[1]:
    c1, c2 = st.columns(2)
    
    if 'idade' in df.columns:
        with c1:
            fig, ax = plt.subplots()
            sns.histplot(df['idade'], bins=20, kde=True, color='#ff5100', ax=ax)
            render_chart(fig, "Age Distribution of Victims")
        
        if 'estado_fisico' in df.columns:
            with c2:
                fig, ax = plt.subplots()
                sns.boxplot(data=df, x='estado_fisico', y='idade', palette='Oranges', ax=ax)
                render_chart(fig, "Age Correlation with Severity")

# --- TAB 3: Incident Circumstances ---
with tabs[2]:
    if 'tipo_acidente' in df.columns:
        fig, ax = plt.subplots(figsize=(10, 4))
        df['tipo_acidente'].value_counts().head(8).sort_values().plot(kind='barh', color='#ff5100', ax=ax)
        render_chart(fig, "Primary Incident Typology")

# ==============================================================================
# 6. RAW DATA INSPECTION
# ==============================================================================
with st.expander("🔍 Deep Data Inspection"):
    st.write("Displaying top 100 rows for verification:")
    st.dataframe(df.head(100), use_container_width=True)
    
    # Export functionality for offline auditing
    st.download_button(
        label="📥 Export Selected Data to Excel",
        data=to_excel_download(df.head(1000)),
        file_name="sompo_eda_report.xlsx",
        use_container_width=True
    )