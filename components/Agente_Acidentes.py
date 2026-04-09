import streamlit as st
import google.generativeai as genai

# NOTE: Do NOT use st.set_page_config() here. 
# It is already defined in the main page (Análise_Exploratória.py).

def renderizar_pagina_agente_acidentes():
    """
    Renders the Specialized AI Agent interface for accident analysis.
    This function is designed to be called inside a dialog or a main page.
    """
    
    # --- API CONFIGURATION ---
    # Using your provided key (Consider moving this to st.secrets for production)
    API_KEY = "AIzaSyAvAjYC8WY6OvFQU2WBji2Kf_anK-YEGRI"

    # --- STYLING (Scoped to the Component) ---
    st.markdown("""
    <style>
        .agent-header {
            font-size: 2.5rem;
            text-align: center;
            background: #ff5100;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: bold;
        }
        .intro-card {
            background-color: #FFF8F5;
            border: 1px solid #FFDDC9;
            border-left: 6px solid #ff5100;
            padding: 1rem 1.5rem;
            border-radius: 8px;
            margin-bottom: 1rem;
            color: #333;
        }
        /* Chat bubble customization */
        div[data-testid="stChatMessage"]:has(span[data-testid="stMarkdownContainer-user"]) {
            background-color: #ff5100;
            color: white;
            border-radius: 15px;
        }
    </style>
    """, unsafe_allow_html=True)

    # --- UI HEADER ---
    st.markdown('<div class="agent-header">Accident Specialist AI</div>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("""
    <div class="intro-card">
    <b>Welcome to the Technical Incident Assistant.</b><br>
    Ask about:
    <ul>
        <li>Accident dynamics and technical causes.</li>
        <li>Liability and insurance coverage assessment.</li>
        <li>Traffic legislation and forensic standards.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    # --- INITIALIZE GEMINI MODEL ---
    @st.cache_resource
    def init_model():
        try:
            genai.configure(api_key=API_KEY)
            model = genai.GenerativeModel(
                'models/gemini-1.5-flash-latest', 
                system_instruction = """
                    You are an expert in traffic accidents, forensic analysis, and insurance claims.
                    Your goal is to support insurance analysts with clear, technical, and objective answers.
                    
                    Response Format:
                    1. Objective Summary (2-3 sentences).
                    2. Technical Details (Dynamics, evidence, red flags).
                    3. Insurance Implications (Liability, coverage, inconsistencies).
                    4. Legal/Regulatory references.
                    """
            )
            return model
        except Exception as e:
            st.error(f"API Configuration Error: {e}")
            return None

    model = init_model()

    # --- CHAT HISTORY MANAGEMENT ---
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display History
    for message in st.session_state.messages:
        role = "assistant" if message["role"] == "model" else message["role"]
        with st.chat_message(role):
            st.markdown(message["parts"][0])

    # --- USER INPUT & AI RESPONSE ---
    if prompt := st.chat_input("Describe the accident or ask a technical question...", disabled=not model):
        
        # Add user message to history
        st.session_state.messages.append({"role": "user", "parts": [prompt]})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate Assistant Response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            try:
                chat = model.start_chat(history=st.session_state.messages)
                full_response = ""
                
                # Stream the response for better UX
                stream = chat.send_message(prompt, stream=True)
                for chunk in stream:
                    full_response += chunk.text
                    message_placeholder.markdown(full_response + "▌")
                
                message_placeholder.markdown(full_response)
                
                # Update History
                st.session_state.messages.append({
                    "role": "model", 
                    "parts": [full_response]
                })
                
            except Exception as e:
                st.error(f"⚠️ Error processing request: {str(e)}")

    # --- UTILITIES ---
    st.divider()
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()
    with col2:
        st.caption("Note: This AI assists analysis; final decisions must follow Sompo's official internal guidelines.")