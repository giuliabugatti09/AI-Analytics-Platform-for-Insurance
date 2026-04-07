import streamlit as st
import google.generativeai as genai


st.set_page_config(
    page_title="Agente Especializado - ChatBot",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ============================================================================
# CSS CUSTOMIZADO (TEMA LARANJA & CLARO)
# ============================================================================
st.markdown("""
<style>
    /* --- CABEÇALHO --- */
    .main-header {
        font-size: 3.5rem;
        text-align: center;
        padding: 1rem 0;
        background: #ff5100;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
    }
    
    /* --- CONTAINER DE INTRODUÇÃO --- */
    .intro-card {
        background-color: #FFF8F5; /* Fundo pêssego bem claro */
        border: 1px solid #FFDDC9; /* Borda laranja clara */
        border-left: 6px solid #ff5100; /* Destaque laranja na lateral */
        padding: 1rem 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
    }

    /* --- ESTILOS DOS COMPONENTES DE CHAT --- */
    /* Mensagem do Assistente (IA) */
    div[data-testid="stChatMessage"]:has(span[data-testid="stMarkdownContainer-assistant"]) {
        background-color: #F0F2F6; /* Fundo cinza claro para o assistente */
        border-radius: 10px;
        padding: 1rem;
    }
    
    /* Mensagem do Usuário */
    div[data-testid="stChatMessage"]:has(span[data-testid="stMarkdownContainer-user"]) {
        background-color: #ff5100; /* Fundo laranja para o usuário */
        border-radius: 10px;
        padding: 1rem;
    }
    
    /* Cor do texto dentro da mensagem do usuário */
    div[data-testid="stChatMessage"]:has(span[data-testid="stMarkdownContainer-user"]) p {
        color: white;
    }
    
    /* Campo de input do chat */
    div[data-testid="stChatInput"] {
        border-top: 2px solid #FFDDC9;
    }

    /* --- BOTÕES --- */
    .stButton>button {
        background-color: #ff5100;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        border: none;
        padding: 0.5rem 1rem;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #D9480F;
        box-shadow: 0 5px 15px rgba(255, 81, 0, 0.3);
    }
""", unsafe_allow_html=True)

def renderizar_pagina_agente_acidentes():
    # ============================================================================
    # CONFIGURAÇÃO DA API (MANTENHA O SEU CÓDIGO)
    # ============================================================================
    # Insira sua chave da API do Gemini aqui
    # Obtenha sua chave em: https://aistudio.google.com/app/apikey
    API_KEY = "AIzaSyA5sfGXnhjm8sM7GhYNo4UXRLg2pVIaC3c"

    # ============================================================================
    # CABEÇALHO E INTRODUÇÃO DA APLICAÇÃO
    # ============================================================================
    st.markdown('<div class="main-header">Agente Especialista em Acidentes</div>', unsafe_allow_html=True)
    st.markdown("<div style='text-align: center; font-size: 1.2em; color: #666;'>Esse é o seu assistência virtual para análise e esclarecimento de sinistros de trânsito.</div>", unsafe_allow_html=True)
    st.markdown("---")

    # A introdução agora está dentro de um card estilizado
    st.markdown("""
    <div class="intro-card">
    Bem-vindo ao assistente virtual especializado em acidentes de trânsito.
    Aqui você pode esclarecer dúvidas sobre:
    <br>
    <br>
    <ul>
        <li><strong>Avaliação técnica de acidentes</strong></li>
        <li><strong>Dinâmica do acidente e possíveis causas</strong></li>
        <li><strong>Responsabilidades e cobertura do sinistro</strong></li>
        <li><strong>Legislação e normas de trânsito aplicáveis</strong></li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    # ============================================================================
    # INICIALIZAÇÃO DO CLIENTE DA API
    # ============================================================================
    @st.cache_resource
    def init_model():
        """Configura a API do Gemini e inicializa o modelo generativo"""
        try:
            genai.configure(api_key=API_KEY)
            
            # ⬇️ USE ESTE NOME DE MODELO OBTIDO DA SUA LISTA ⬇️
            model = genai.GenerativeModel(
                'models/gemini-flash-latest', 
                system_instruction = """
                    Você é um especialista em acidentes de trânsito, com experiência em perícia técnica, legislação de trânsito e regulação de sinistros em seguradoras.

                    Seu papel é:
                    - Apoiar o analista da seguradora na avaliação de acidentes de trânsito, fornecendo respostas claras, técnicas e fundamentadas.
                    - Explicar a dinâmica do acidente, apontando possíveis causas e responsabilidades de acordo com normas de trânsito.
                    - Indicar inconsistências em relatos ou versões dos envolvidos, sempre com base em critérios técnicos.
                    - Relacionar o caso às normas legais aplicáveis e práticas de perícia reconhecidas no setor.
                    - Evitar linguagem vaga ou opinativa, priorizando análises objetivas que auxiliem na tomada de decisão sobre o sinistro.

                    Formato da resposta:
                    1. Síntese objetiva da análise (2-3 frases).
                    2. Detalhamento técnico (dinâmica do acidente, evidências, pontos de atenção).
                    3. Possíveis implicações para a seguradora (responsabilidade, cobertura, inconsistências).
                    4. Referência a norma/lei aplicável, quando pertinente.
                    """
            )
            return model
        except Exception as e:
            st.error(f"Erro ao configurar a API do Gemini. Verifique sua chave. Detalhes: {e}")
            return None

    model = init_model()

    # ============================================================================
    # GERENCIAMENTO DO HISTÓRICO DE CONVERSAS
    # ============================================================================
    # Inicializa o histórico de mensagens na sessão do Streamlit
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # ============================================================================
    # EXIBIÇÃO DO HISTÓRICO DE CHAT
    # ============================================================================
    # Exibe todas as mensagens anteriores
    for message in st.session_state.messages:
        # O Gemini usa 'model' para o assistente, mas o Streamlit usa 'assistant'
        role = "assistant" if message["role"] == "model" else message["role"]
        with st.chat_message(role):
            # O conteúdo no Gemini fica dentro de 'parts', que é uma lista
            st.markdown(message["parts"][0])

    # ============================================================================
    # INTERFACE DE ENTRADA DO USUÁRIO
    # ============================================================================
    # Campo de entrada para nova mensagem. Só habilita se o modelo foi carregado.
    if prompt := st.chat_input("Digite sua pergunta...", disabled=not model):
        
        # Adiciona a mensagem do usuário ao histórico (formato do Gemini)
        st.session_state.messages.append({"role": "user", "parts": [prompt]})
        
        # Exibe a mensagem do usuário
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # ========================================================================
        # GERAÇÃO DA RESPOSTA DO ASSISTENTE
        # ========================================================================
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            try:
                # Inicia uma sessão de chat com o histórico existente
                chat = model.start_chat(history=st.session_state.messages)
                
                # Envia a nova mensagem para a API do Gemini
                # Usar stream=True para uma resposta mais interativa (efeito de digitação)
                full_response = ""
                stream = chat.send_message(prompt, stream=True)
                for chunk in stream:
                    full_response += chunk.text
                    message_placeholder.markdown(full_response + "▌") # Adiciona um cursor para feedback visual
                
                # Exibe a resposta final completa
                message_placeholder.markdown(full_response)
                
                # Adiciona a resposta do assistente ao histórico (formato do Gemini)
                # NOTA: O histórico já é atualizado internamente pelo `chat.send_message`,
                # mas para exibição na UI, precisamos atualizar o session_state.
                st.session_state.messages.append({
                    "role": "model", 
                    "parts": [full_response]
                })
                
            except Exception as e:
                # Tratamento de erros
                error_message = f"⚠️ Erro ao processar sua solicitação: {str(e)}"
                message_placeholder.error(error_message)

    # ============================================================================
    # BOTÃO PARA LIMPAR HISTÓRICO
    # ============================================================================
    st.divider()
    if st.button("🗑️ Limpar Conversa"):
        st.session_state.messages = []
        st.rerun()

    # ============================================================================
    # RODAPÉ COM INFORMAÇÕES
    # ============================================================================
    st.markdown("---")
    st.caption("💡 Dica: Este assistente fornece informações gerais. Para casos específicos, consulte sempre um profissional especializado.")
