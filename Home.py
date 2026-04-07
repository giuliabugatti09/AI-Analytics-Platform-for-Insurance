import streamlit as st
from pathlib import Path
import sys
import os

from components.Agente_Acidentes import renderizar_pagina_agente_acidentes # Importe a função que você criou

# Adiciona o diretório raiz ao path para importar utils
sys.path.append(str(Path(__file__).parent.parent))

from utils.security_utils import obter_hosts_locais_permitidos, fazer_requisicao_segura, obter_paginas_streamlit_validas, normalizar_url

st.set_page_config(
    page_title="Acidentes de Trânsito — Analytics & Predição",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ===============================
# 💅 CSS Personalizado
# ===============================
st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem;
        text-align: center;
        padding: 1rem 0;
        background: #ff5100; /* Cor laranja principal */
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
    }
    /* --- CONTAINERS E CARDS (TEMA CLARO) --- */
    .feature-card, .stats-container, .navigation-info {
        background-color: #FFFFFF; /* Fundo branco para os cards */
        border: 1px solid #FFDDC9; /* Borda laranja bem clara */
        border-left: 6px solid #ff5100; /* Destaque laranja na lateral esquerda */
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        color: #333; /* Cor do texto escura para legibilidade */
    }
    /* Faz o texto dentro do card usar a cor laranja escura para títulos */
    .feature-card h3, .stats-container h3, .navigation-info h4 {
        color: #D9480F;
    }

    /* Ajustes específicos para manter a identidade de cada card */
    .stats-container {
        padding: 2rem;
        text-align: center;
    }
    .navigation-info {
        padding: 1.2rem;
        background-color: #FFF8F5; /* Fundo um pouco diferente para o card de dica */
    }
    /* Estilo das Abas (Tabs) */
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0px 0px;
        padding: 10px;
    }
    .stTabs [aria-selected="true"] {}
            
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
</style>
""", unsafe_allow_html=True)

# ===============================
# 🎯 Título principal
# ===============================
st.markdown('<div class="main-header">Acidentes de Trânsito — Analytics & Predição</div>', unsafe_allow_html=True)

# ===============================
# 📘 Introdução
# ===============================
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("""
    <div style='text-align: center; font-size: 1.2em; color: #666;'>
        Plataforma interativa para análise, visualização e predição de acidentes de trânsito.<br>
        <strong>Transformando dados em insights para salvar vidas.</strong>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

# ===============================
# 🔒 Validação de URL Segura (SSRF)
# ===============================

# Obter hosts locais permitidos dinamicamente
hosts_locais = obter_hosts_locais_permitidos()

# Adicionar outros domínios externos que sua aplicação precisa acessar
outros_hosts_confiaveis = {"www.google.com", "172.16.0.2", "sompo.com.br"}

# A lista final de hosts permitidos
hosts_permitidos_final = hosts_locais.union(outros_hosts_confiaveis)

paginas_validas_app = obter_paginas_streamlit_validas()

col1, col2 = st.columns(2)

# Expander para os Hosts Permitidos
with st.expander("🌐 Ver Hosts Permitidos (Dinâmicos + Confiáveis)"):
    # O st.json agora fica dentro do expander
    st.json(list(hosts_permitidos_final))

# Expander para as Páginas Válidas
with st.expander("📄 Ver Páginas Válidas Encontradas na Aplicação"):
    # O st.json agora fica dentro do expander
    st.json(list(paginas_validas_app))

url_usuario = st.text_input(
    "Insira uma URL para validar e fazer a requisição:", 
    placeholder="digite uma URL"
)

if st.button("Validar e Requisitar URL"):
    if not url_usuario.strip():
        st.warning("⚠️ Por favor, insira uma URL antes de validar.")
    else:
        url_processada = normalizar_url(url_usuario)
        # Passar a lista de páginas válidas para a função
        conteudo, msg = fazer_requisicao_segura(
            url_processada,
            hosts_permitidos=hosts_permitidos_final,
            protocolos_permitidos={"http", "https"},
            paginas_validas=paginas_validas_app
        )
        
        if "Erro" in msg:
            st.error(msg)
        else:
            st.success(msg)

        if conteudo:
            st.info("Conteúdo recebido (primeiros 500 caracteres):")
            st.code(conteudo[:500], language="html", line_numbers=True)

# ===============================
# 🧭 Funcionalidades
# ===============================
st.markdown("## Funcionalidades Disponíveis")

pages_dir = Path(__file__).parent / "pages"
page_files = sorted([f for f in pages_dir.glob("*.py") if not f.name.startswith('_.')]) if pages_dir.exists() else []

page_descriptions = {
    "Análise_Exploratória.py": {
        "icon": "",
        "title": "Análise Exploratória",
        "description": "Explore padrões, distribuições e correlações nos dados de acidentes com filtros dinâmicos, gráficos interativos e estatísticas descritivas completas.",
        "features": ["Gráficos interativos", "Filtros dinâmicos", "Estatísticas descritivas", "Análise de correlações"]
    },
    "Modelo_de_previsão.py": {
        "icon": "",
        "title": "Modelos de Predição",
        "description": "Utilize algoritmos de machine learning avançados para prever riscos e desfechos de acidentes com base em variáveis históricas.",
        "features": ["Modelos Python (.pkl)", "Integração com R", "Predições em tempo real", "Avaliação de performance"]
    },
    "Heatmap.py": {
        "icon": "",
        "title": "Mapa de Risco (Heatmap)",
        "description": "Visualize geograficamente as áreas de maior concentração de acidentes, identifique pontos críticos e obtenha insights espaciais.",
        "features": ["Mapas interativos", "Heatmaps de densidade", "Análise geográfica", "Identificação de hotspots"]
    },
    "Classificador_imagem.py": {
        "icon": "",
        "title": "Classificação de acidentes por imagens",
        "description": "Utilize modelos de deep learning para análise inteligente de imagens, detectando automaticamente acidentes de trânsito com alta precisão.",
        "features": ["Detecção por IA", "Upload múltiplo de imagens", "Análise por câmera", "Histórico de análises"]
    },
    "Agente_Acidentes.py": {
        "icon": "",
        "title": "Agente Inteligente de Acidentes",
        "description": "Converse com um especialista virtual em acidentes de trânsito powered by Google Gemini.",
        "features": ["IA Google Gemini", "Chat em tempo real", "Análise técnica especializada", "Histórico de conversas"]
    }
}

cols = st.columns(2)
for i, page_file in enumerate(page_files):
    name = page_file.name
    col = cols[i % 2]
    with col:
        if name in page_descriptions:
            page_info = page_descriptions[name]
            st.markdown(f"""
            <div class="feature-card">
                <h3>{page_info['icon']} {page_info['title']}</h3>
                <p>{page_info['description']}</p>
                <ul style="margin: 1rem 0;">
                    {''.join([f'<li>✓ {feature}</li>' for feature in page_info['features']])}
                </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="feature-card">
                <h3>📄 {name.replace('.py','')}</h3>
                <p>Página personalizada ou módulo auxiliar do sistema.</p>
            </div>
            """, unsafe_allow_html=True)

# ===============================
# 📘 Como usar
# ===============================
st.markdown("---")
col1, col2 = st.columns([2, 1])
with col1:
    st.markdown("""
    ## Como Usar a Plataforma
    1. **Navegue pelo menu lateral** para acessar as diferentes funcionalidades
    2. **Carregue seus dados** ou utilize datasets de exemplo
    3. **Explore os dados** na seção de Análise Exploratória
    4. **Visualize mapas** e identifique padrões geográficos
    5. **Realize predições** com machine learning
    6. **Converse com o Agente IA** para dúvidas específicas
    """)
with col2:
    st.markdown("""
    <div class="navigation-info">
        <h4>💡 Dica de Navegação</h4>
        <p>Use o <strong>menu lateral</strong> (→) para navegar entre as páginas.</p>
    </div>
    """, unsafe_allow_html=True)

# ===============================
# 📚 Rodapé
# ===============================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem 0;'>
    <h4>Projeto Educacional - Análise de Acidentes de Trânsito</h4>
    <p>Desenvolvido para fins acadêmicos e de pesquisa • FIAP • 2025</p>
    <p><em>💡  "Dados salvam vidas quando transformados em conhecimento"</em></p>
</div>
""", unsafe_allow_html=True)

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
