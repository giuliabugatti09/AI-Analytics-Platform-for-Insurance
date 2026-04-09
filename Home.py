import streamlit as st
from pathlib import Path
import sys
import os
from components.Agente_Acidentes import renderizar_pagina_agente_acidentes

# Adding root directory to sys.path for utility imports
sys.path.append(str(Path(__file__).parent.parent))
from utils.security_utils import obter_hosts_locais_permitidos, fazer_requisicao_segura, obter_paginas_streamlit_validas, normalizar_url

# ==============================================================================
# 1. PAGE CONFIGURATION & UI STYLING
# ==============================================================================
st.set_page_config(
    page_title="Road Safety AI Hub - Home",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Professional branding (Sompo Orange Theme)
st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem;
        text-align: center;
        padding: 1rem 0;
        background: #ff5100;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
    }
    .feature-card {
        background-color: #FFFFFF;
        border: 1px solid #FFDDC9;
        border-left: 6px solid #ff5100;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        min-height: 250px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    .feature-card h3 { color: #D9480F; margin-top: 0; }
    .stButton>button {
        background-color: #ff5100;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. HEADER & INTRODUCTION
# ==============================================================================
st.markdown('<div class="main-header">Smart Road Safety Intelligence Platform</div>', unsafe_allow_html=True)

col_left, col_mid, col_right = st.columns([1, 2, 1])
with col_mid:
    st.markdown("""
    <div style='text-align: center; font-size: 1.2em; color: #666;'>
        An end-to-end interactive platform for <b>traffic accident analysis, visualization, and predictive modeling</b>.<br>
        Leveraging Data Science and AI to drive safety insights.
    </div>
    """, unsafe_allow_html=True)
st.divider()

# ==============================================================================
# 3. SECURITY & SSRF VALIDATION (ADMIN UTILS)
# ==============================================================================
with st.expander("🛡️ System Security & URL Validation (SSRF Lab)", expanded=False):
    st.caption("This section validates safe URL requests within the platform architecture.")
    
    hosts_locais = obter_hosts_locais_permitidos()
    outros_hosts = {"www.google.com", "sompo.com.br"}
    all_allowed_hosts = hosts_locais.union(outros_hosts)
    valid_pages = obter_paginas_streamlit_validas()

    c1, c2 = st.columns(2)
    with c1:
        st.info("**Allowed Hosts**")
        st.json(list(all_allowed_hosts))
    with c2:
        st.info("**Validated Application Pages**")
        st.json(list(valid_pages))

    user_url = st.text_input("Enter URL for security check:", placeholder="https://example.com")
    if st.button("Validate & Request URL"):
        if user_url:
            processed_url = normalizar_url(user_url)
            content, msg = fazer_requisicao_segura(
                processed_url,
                hosts_permitidos=all_allowed_hosts,
                protocolos_permitidos={"http", "https"},
                paginas_validas=valid_pages
            )
            if "Erro" in msg: st.error(msg)
            else: st.success(msg)
            if content: st.code(content[:500], language="html")

# ==============================================================================
# 4. CORE FUNCTIONALITIES
# ==============================================================================
st.markdown("## Platform Modules")

# Dictionary containing all English metadata for the dashboard cards
page_metadata = {
    "Análise_Exploratória.py": {
        "title": "Exploratory Data Analysis",
        "desc": "Statistical patterns, dynamic filters, and comprehensive correlation charts.",
        "points": ["Interactive Visuals", "Dynamic Filtering", "Descriptive Statistics"]
    },
    "Modelo_de_previsão.py": {
        "title": "Predictive Intelligence",
        "desc": "Machine Learning models to forecast incident risk and severity outcomes.",
        "points": ["XGBoost Integration", "Real-time Inference", "Risk Scoring"]
    },
    "Heatmap.py": {
        "title": "Geospatial Risk Mapping",
        "desc": "Interactive Heatmaps identifying high-density accident hotspots across Brazil.",
        "points": ["PyDeck Integration", "Density Analysis", "Hotspot ID"]
    },
    "Classificador_imagem.py": {
        "title": "Computer Vision Diagnostic",
        "desc": "Deep Learning engine for automatic accident detection from visual evidence.",
        "points": ["CNN Inference", "CCTV/Upload Support", "Visual Verification"]
    }
}

# Scan pages directory and render cards
pages_path = Path(__file__).parent / "pages"
available_files = [f.name for f in pages_path.glob("*.py")] if pages_path.exists() else []

cols = st.columns(2)
idx = 0
for filename, info in page_metadata.items():
    if filename in available_files:
        with cols[idx % 2]:
            st.markdown(f"""
            <div class="feature-card">
                <h3>{info['title']}</h3>
                <p>{info['desc']}</p>
                <ul>
                    {''.join([f'<li>✓ {p}</li>' for p in info['points']])}
                </ul>
            </div>
            """, unsafe_allow_html=True)
        idx += 1

# ==============================================================================
# 5. SIDEBAR & FOOTER
# ==============================================================================
with st.sidebar:
    st.title("Navigation")
    
    if st.button("🤖 Talk to AI Specialist", use_container_width=True):
        @st.dialog("Incident Analyst Agent", width="large")
        def agent_dialog():
            renderizar_pagina_agente_acidentes()
        agent_dialog()
    st.divider()

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888; padding-bottom: 2rem;'>
    <h4>Educational Project - Road Safety Data Science</h4>
    <p>Developed for academic research and forensic analysis simulation • FIAP • 2026</p>
    <p><i>"Data saves lives when transformed into actionable knowledge."</i></p>
</div>
""", unsafe_allow_html=True)