# Importação das bibliotecas necessárias
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import time
from datetime import datetime
import io
import base64

from components.Agente_Acidentes import renderizar_pagina_agente_acidentes # Importe a função que você criou

# --- Configurações da Página ---
st.set_page_config(
    page_title="Analisador de Imagens de Acidentes",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

    /* --- BOTÕES PRINCIPAIS --- */
    .stButton>button {
        background-color: #ff5100; /* Cor laranja principal */
        color: white;
        font-weight: bold;
        border-radius: 10px;
        border: none;
        padding: 0.5rem 1rem;
        font-size: 1.1rem;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #D9480F; /* Laranja mais escuro no hover */
        box-shadow: 0 5px 15px rgba(255, 81, 0, 0.3); /* Sombra laranja */
    }

    /* --- CAIXAS DE RESULTADO DA PREVISÃO --- */
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        color: white; /* Texto branco para contraste com gradientes */
        font-weight: bold;
    }
    .accident-box {
        /* Gradiente laranja para indicar risco/acidente */
        background: linear-gradient(135deg, #ff5100 0%, #ff8c00 100%);
    }
    .safe-box {
        /* Verde mantido para indicar segurança (contraste funcional) */
        background: linear-gradient(135deg, #00C851 0%, #00FF7F 100%);
    }

    /* --- CONTAINER DE MÉTRICAS --- */
    .metric-container {
        background: #FFF8F5; /* Fundo pêssego bem claro, da nossa paleta */
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border: 1px solid #FFDDC9; /* Borda laranja clara */
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

# --- Constantes e Configurações ---
MODEL_PATH = 'modelo_acidentes.keras'
IMG_SIZE = (224, 224)
CLASS_NAMES = ['Não Acidente', 'Acidente']
CONFIDENCE_THRESHOLD = 0.5  # Valor fixo, não será mais ajustável pelo usuário

# --- Inicialização do Estado da Sessão ---
if 'history' not in st.session_state:
    st.session_state.history = []
if 'total_analyzed' not in st.session_state:
    st.session_state.total_analyzed = 0
if 'accidents_detected' not in st.session_state:
    st.session_state.accidents_detected = 0

# --- Funções Auxiliares ---
@st.cache_resource
def load_keras_model():
    """
    Carrega o modelo Keras com tratamento de erro aprimorado.
    """
    try:
        with st.spinner('Carregando modelo de IA...'):
            model = tf.keras.models.load_model(MODEL_PATH)
            # Compilar modelo se necessário
            if not model.optimizer:
                model.compile(
                    optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy']
                )
        return model
    except FileNotFoundError:
        st.error("⚠️ **Arquivo do modelo não encontrado!**")
        st.info(f"Certifique-se de que o arquivo '{MODEL_PATH}' está no diretório correto.")
        return None
    except Exception as e:
        st.error(f"❌ **Erro ao carregar o modelo:** {e}")
        st.info("Verifique se o modelo está no formato correto (.keras)")
        return None

def preprocess_image(image, img_size):
    """
    Pré-processa a imagem para o formato esperado pelo modelo.
    """
    # Converter para RGB se necessário
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Redimensionar
    image = image.resize(img_size, Image.Resampling.LANCZOS)
    
    # Converter para array numpy
    img_array = np.array(image)
    
    # Normalizar pixels para [0, 1]
    img_array = img_array.astype('float32') / 255.0
    
    # Adicionar dimensão do batch
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def predict_image(image, model, img_size, class_names_list):
    try:
        img = image.convert('RGB')
        img = img.resize(img_size)
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        predictions = model.predict(img_array)
        probability = float(predictions[0][0])
        predicted_class_index = int(probability >= 0.5)
        predicted_class_name = class_names_list[predicted_class_index]
        confidence = probability if predicted_class_index == 1 else 1 - probability
        return {
            'class': predicted_class_name,
            'probability': probability,
            'confidence': confidence,
            'inference_time': 0  # Adapte se quiser medir o tempo
        }
    except Exception as e:
        st.error(f"❌ Erro durante a predição: {e}")
        return None

def get_image_download_link(img, filename):
    """
    Gera link para download da imagem analisada.
    """
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/png;base64,{img_str}" download="{filename}">📥 Baixar Imagem Analisada</a>'
    return href

# --- Sidebar ---
with st.sidebar:

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

    
    st.header("📊 Estatísticas da Sessão")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Analisado", st.session_state.total_analyzed)
    with col2:
        st.metric("Acidentes Detectados", st.session_state.accidents_detected)
    
    if st.session_state.total_analyzed > 0:
        accident_rate = (st.session_state.accidents_detected / st.session_state.total_analyzed) * 100
        st.progress(accident_rate / 100)
        st.caption(f"Taxa de Detecção: {accident_rate:.1f}%")
    
    st.divider()
    
    st.header("⚙️ Configurações")
    
    # Removido o slider de threshold
    # threshold = st.slider(
    #     "Limiar de Confiança",
    #     min_value=0.0,
    #     max_value=1.0,
    #     value=CONFIDENCE_THRESHOLD,
    #     step=0.05,
    #     help="Ajuste o limiar para classificação como acidente"
    # )
    # CONFIDENCE_THRESHOLD = threshold
    
    # Opção para mostrar detalhes técnicos
    show_technical = st.checkbox("Mostrar Detalhes Técnicos", value=False)
    
    # Botão para limpar histórico
    if st.button("🗑️ Limpar Histórico"):
        st.session_state.history = []
        st.session_state.total_analyzed = 0
        st.session_state.accidents_detected = 0
        st.rerun()
    
    st.divider()
    
    # Informações sobre o modelo
    st.header("Sobre o Modelo")
    st.info(
        "Este modelo utiliza Deep Learning para detectar acidentes em imagens. "
        "Foi treinado com milhares de exemplos para garantir alta precisão."
    )

# --- Interface Principal ---
st.markdown('<div class="main-header">Central de Análise de Acidentes de Trânsito</div>', unsafe_allow_html=True)
st.markdown("<div style='text-align: center; font-size: 1.2em; color: #666;'>Detecção Inteligente de Acidentes usando Inteligência Artificial.</div>", unsafe_allow_html=True)

# Tabs para diferentes funcionalidades
tab1, tab2, tab3 = st.tabs(["📸 **Análise de Imagem**", "📜 **Histórico**", "📚 **Como Usar**"])

with tab1:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### 📤 Upload de Imagem")
        
        # Upload de arquivo com drag and drop
        uploaded_file = st.file_uploader(
            "Arraste uma imagem ou clique para selecionar",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            help="Formatos suportados: JPG, JPEG, PNG, BMP, WEBP"
        )
        
        # Opção de usar câmera (se disponível)
        camera_input = st.camera_input("Ou tire uma foto agora")
        
        # Determinar qual imagem usar
        image_to_process = camera_input if camera_input else uploaded_file
        
        if image_to_process:
            image = Image.open(image_to_process)
            st.image(image, caption='Imagem Carregada', use_column_width=True)
            
            # Mostrar informações da imagem
            st.caption(f"Dimensões: {image.size[0]}x{image.size[1]} pixels")
            st.caption(f"Formato: {image.format if hasattr(image, 'format') else 'N/A'}")
    
    with col2:
        st.markdown("#### 🔍 Resultado da Análise")
        
        if image_to_process:
            # Carregar modelo
            model = load_keras_model()
            
            if model is None:
                st.stop()
            
            # Botão de análise
            if st.button('🚀 Iniciar Análise', use_container_width=True):
                with st.spinner('Processando imagem...'):
                    # Fazer predição
                    result = predict_image(image, model, IMG_SIZE, CLASS_NAMES)
                    
                    if result:
                        # Atualizar estatísticas
                        st.session_state.total_analyzed += 1
                        if result['class'] == 'Acidente':
                            st.session_state.accidents_detected += 1
                        
                        # Adicionar ao histórico
                        st.session_state.history.append({
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'result': result['class'],
                            'confidence': result['confidence']
                        })
                        
                        # Exibir resultado com estilo
                        if result['class'] == 'Acidente':
                            st.markdown(
                                f"""
                                <div class="prediction-box accident-box">
                                    <h2>⚠️ ACIDENTE DETECTADO</h2>
                                    <h3>Confiança: {result['confidence']:.1%}</h3>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                            st.error("Um acidente foi identificado nesta imagem.")
                        else:
                            st.markdown(
                                f"""
                                <div class="prediction-box safe-box">
                                    <h2>✅ NENHUM ACIDENTE</h2>
                                    <h3>Confiança: {result['confidence']:.1%}</h3>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                            st.success("Nenhum acidente foi detectado nesta imagem.")
                        
                        # Barra de progresso visual
                        st.markdown("##### Nível de Confiança")
                        st.progress(result['confidence'])
                        
                        # Detalhes técnicos (se habilitado)
                        if show_technical:
                            with st.expander("🔧 Detalhes Técnicos"):
                                st.write(f"**Probabilidade bruta:** {result['probability']:.4f}")
                                st.write(f"**Tempo de inferência:** {result['inference_time']*1000:.2f} ms")
                                st.write(f"**Limiar usado:** {CONFIDENCE_THRESHOLD}")  # Pode manter para referência técnica
                                st.write(f"**Dimensão de entrada:** {IMG_SIZE}")
                        
                        # Link para download
                        st.markdown(
                            get_image_download_link(
                                image,
                                f"analise_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                            ),
                            unsafe_allow_html=True
                        )
        else:
            st.info("👆 Faça o upload de uma imagem para começar a análise")

with tab2:
    st.markdown("#### 📊 Histórico de Análises")
    
    if st.session_state.history:
        # Converter histórico para DataFrame para melhor visualização
        import pandas as pd
        df = pd.DataFrame(st.session_state.history)
        
        # Filtros
        col1, col2 = st.columns(2)
        with col1:
            filter_result = st.selectbox(
                "Filtrar por resultado:",
                ["Todos", "Acidente", "Não Acidente"]
            )
        
        # Aplicar filtro
        if filter_result != "Todos":
            df_filtered = df[df['result'] == filter_result]
        else:
            df_filtered = df
        
        # Exibir tabela
        st.dataframe(
            df_filtered,
            use_container_width=True,
            hide_index=True,
            column_config={
                "timestamp": "Data/Hora",
                "result": "Resultado",
                "confidence": st.column_config.ProgressColumn(
                    "Confiança",
                    help="Nível de confiança da predição",
                    format="%.1%%",
                    min_value=0,
                    max_value=1,
                ),
            }
        )
        
        # Estatísticas do histórico
        st.markdown("##### 📈 Estatísticas")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total de Análises", len(df))
        with col2:
            accidents = len(df[df['result'] == 'Acidente'])
            st.metric("Acidentes Detectados", accidents)
        with col3:
            avg_confidence = df['confidence'].mean()
            st.metric("Confiança Média", f"{avg_confidence:.1%}")
    else:
        st.info("Nenhuma análise foi realizada ainda.")

with tab3:
    st.markdown("#### 📖 Guia de Uso")
    
    st.markdown("""
    ### Como usar o analisador:
    
    1. **Upload de Imagem** 📤
       - Clique no botão de upload ou arraste uma imagem
       - Formatos aceitos: JPG, PNG, BMP, WEBP
       - Você também pode tirar uma foto usando a câmera
    
    2. **Análise** 🔍
       - Clique em "Iniciar Análise"
       - Aguarde o processamento (geralmente < 1 segundo)
       - Veja o resultado com o nível de confiança
    
    3. **Interpretação dos Resultados** 📊
       - **Acidente Detectado**: O modelo identificou características de acidente
       - **Nenhum Acidente**: A imagem parece não conter acidentes
       - **Nível de Confiança**: Indica o quão certo o modelo está (0-100%)
    
    ### Dicas para melhores resultados:
    
    - Use imagens claras e bem iluminadas
    - Centralize o acidente na imagem (se houver)
    - Evite imagens muito distorcidas ou com baixa resolução
    - Imagens de câmeras de segurança funcionam bem
    
    ### Configurações avançadas:
    
    - **Detalhes Técnicos**: Ative para ver informações técnicas da análise
    
    ### Limitações:
    
    ⚠️ Este modelo é uma ferramenta de auxílio e não substitui a análise humana
    ⚠️ Pode haver falsos positivos ou negativos
    ⚠️ Funciona melhor com imagens de veículos rodoviários
    """)

# Footer
st.divider()
st.markdown(
    """
    <div style='text-align: center; color: #888;'>
        <p>Sistema de Detecção de Acidentes v2.0 | Powered by TensorFlow & Streamlit</p>
    </div>
    """,
    unsafe_allow_html=True
)