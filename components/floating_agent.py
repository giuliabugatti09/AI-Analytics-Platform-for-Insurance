import streamlit as st

def add_floating_agent_icon():
    """
    Adiciona um ícone flutuante de agente IA a uma página do Streamlit.
    
    Este componente usa apenas HTML e CSS para criar um link estilizado que
    navega para a página 'Agente_Acidentes', sem a necessidade de JavaScript
    para a funcionalidade de clique.
    """
    st.markdown("""
    <style>
        /* Define a classe para o nosso link flutuante */
        .floating-agent-icon {
            position: fixed;
            bottom: 325px;
            right: 325px;
            width: 50px;
            height: 50px;
            background-color: #ff5100; /* Laranja principal */
            color: white !important; /* Cor do ícone (emoji) */
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 26px; /* Tamanho do emoji */
            text-decoration: none !important; /* Remove o sublinhado do link */
            cursor: pointer;
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
            transition: all 0.3s ease;
            z-index: 1000;
            border: 2px solid #FFDDC9; /* Borda laranja clara */
        }
        
        .floating-agent-icon:hover {
            transform: scale(1.2);
            box-shadow: 0 6px 20px rgba(255, 81, 0, 0.4); /* Sombra laranja */
            background-color: #D9480F; /* Laranja mais escuro */
            color: white !important;
            text-decoration: none !important;
        }
        
        /* Animação de pulso no emoji */
        .floating-agent-icon span {
            animation: pulse 2s infinite;
            display: inline-block;
        }
        
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.2); }
            100% { transform: scale(1); }
        }
        /* Tooltip */
        .floating-agent-icon::after {
            content: 'Agente IA - Clique para conversar';
            position: absolute;
            right: 55px;
            top: 50%;
            transform: translateY(-50%);
            background: #ff5100; /* Fundo laranja claro */
            color: white;
            padding: 8px 12px;
            border-radius: 4px;
            font-size: 12px;
            white-space: nowrap;
            opacity: 0;
            transition: opacity 0.3s ease;
            pointer-events: none;
        }
        
        .floating-agent-icon:hover::after {
            opacity: 1;
        }
    </style>

    <a href="Agente_Acidentes" class="floating-agent-icon">
        <span>🤖</span>
    </a>
    """, unsafe_allow_html=True)