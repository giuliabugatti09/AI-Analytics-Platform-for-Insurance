import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import io
import base64
from datetime import datetime
from components.Agente_Acidentes import renderizar_pagina_agente_acidentes

# ==============================================================================
# 1. PAGE CONFIGURATION & GLOBAL STYLES
# ==============================================================================
st.set_page_config(
    page_title="AI Incident Analyzer - Road Safety",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional branding with Sompo Orange theme
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        text-align: center;
        color: #ff5100;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .stButton>button {
        background-color: #ff5100;
        color: white;
        border-radius: 8px;
        width: 100%;
        font-weight: bold;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #D9480F;
        border-color: #ff5100;
    }
    .prediction-card {
        padding: 2rem;
        border-radius: 12px;
        text-align: center;
        color: white;
        font-weight: bold;
        margin-top: 1rem;
    }
    .status-alert {
        background-color: rgba(255, 81, 0, 0.1);
        border-left: 5px solid #ff5100;
        padding: 1rem;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

# ==============================================================================
# 2. CONSTANTS & MODEL CACHING
# ==============================================================================
# Ensure these match your notebook exports
MODEL_PATH = 'models/accident_detection_final_optimized.keras' 
IMG_SIZE = (224, 224)
CLASS_NAMES = ['Normal Flow', '🚨 INCIDENT DETECTED']

@st.cache_resource
def load_incident_model():
    """Loads the pre-trained EfficientNetV2 model with caching."""
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading AI model: {e}")
        return None

# ==============================================================================
# 3. ANALYTICS ENGINE
# ==============================================================================
def process_and_predict(image, model):
    """Handles image preprocessing and model inference."""
    try:
        # Preprocessing aligned with our Transfer Learning notebook
        img = image.convert('RGB').resize(IMG_SIZE)
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0) # Create batch axis
        
        # Inference
        predictions = model.predict(img_array, verbose=0)
        probability = float(predictions[0][0])
        
        # Binary Classification Logic (Sigmoid)
        is_accident = probability >= 0.5
        confidence = probability if is_accident else 1 - probability
        
        return {
            'class': CLASS_NAMES[1] if is_accident else CLASS_NAMES[0],
            'confidence': confidence,
            'is_accident': is_accident
        }
    except Exception as e:
        st.error(f"Inference error: {e}")
        return None

# ==============================================================================
# 4. SIDEBAR & SESSION STATE
# ==============================================================================
if 'history' not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.title("Control Center")
    
    if st.button("🤖 Launch AI Specialist", use_container_width=True):
        @st.dialog("Accident Specialist Agent", width="large")
        def agent_dialog():
            renderizar_pagina_agente_acidentes()
        agent_dialog()

    st.divider()
    st.subheader("📊 Session Metrics")
    total = len(st.session_state.history)
    accidents = sum(1 for x in st.session_state.history if x['is_accident'])
    
    col_a, col_b = st.columns(2)
    col_a.metric("Analyzed", total)
    col_b.metric("Incidents", accidents)
    
    if st.button("🗑️ Reset Session History"):
        st.session_state.history = []
        st.rerun()

# ==============================================================================
# 5. MAIN INTERFACE
# ==============================================================================
st.markdown('<div class="main-header">Computer Vision Incident Analyzer</div>', unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #666;'>Real-time Deep Learning diagnostic for road safety and claim verification.</p>", unsafe_allow_html=True)

tab_analyze, tab_history, tab_guide = st.tabs(["📸 Image Analysis", "📜 History Logs", "📖 User Guide"])

with tab_analyze:
    col_input, col_result = st.columns([1, 1])
    
    with col_input:
        st.markdown("#### 📤 Input Source")
        uploaded_file = st.file_uploader("Upload incident photo", type=["jpg", "jpeg", "png"])
        camera_file = st.camera_input("Take a real-time photo")
        
        target_file = camera_file if camera_file else uploaded_file
        
        if target_file:
            image = Image.open(target_file)
            st.image(image, caption="Current Input", use_container_width=True)

    with col_result:
        st.markdown("#### 🔍 AI Diagnostic")
        if target_file:
            model = load_incident_model()
            if model and st.button("🚀 Run AI Diagnosis"):
                with st.spinner('Analyzing visual patterns...'):
                    results = process_and_predict(image, model)
                    
                    if results:
                        # Styling based on result
                        bg_color = "linear-gradient(135deg, #ff5100 0%, #ff8c00 100%)" if results['is_accident'] else "linear-gradient(135deg, #00C851 0%, #00FF7F 100%)"
                        
                        st.markdown(f"""
                            <div class="prediction-card" style="background: {bg_color};">
                                <h2 style='color: white;'>{results['class']}</h2>
                                <h3 style='color: white;'>Confidence: {results['confidence']:.2%}</h3>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Save to history
                        st.session_state.history.append({
                            'time': datetime.now().strftime("%H:%M:%S"),
                            'result': results['class'],
                            'confidence': results['confidence'],
                            'is_accident': results['is_accident']
                        })
        else:
            st.info("Waiting for image input to begin diagnosis.")

with tab_history:
    if st.session_state.history:
        history_df = pd.DataFrame(st.session_state.history)
        st.dataframe(history_df[['time', 'result', 'confidence']], use_container_width=True)
    else:
        st.write("No incidents analyzed in this session.")

with tab_guide:
    st.markdown("""
    ### 📖 Strategic Usage Guide
    1. **Visual Clarity:** Ensure the incident is centered and well-lit.
    2. **Insurance Triage:** Use this tool to cross-verify driver reports with visual evidence.
    3. **Legal Value:** Download the analyzed report for internal documentation.
    """)

# Footer
st.divider()
st.markdown("<div style='text-align: center; color: #888;'>AI Analytics Platform v2.0 | Integrated Computer Vision Engine</div>", unsafe_allow_html=True)