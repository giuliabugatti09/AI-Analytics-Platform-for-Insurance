import streamlit as st
import pandas as pd
import numpy as np
import joblib  # Used to load/save the model and other objects
from sklearn.preprocessing import LabelEncoder
import datetime
import io
import warnings
import traceback
import plotly.graph_objects as go

from components.Agente_Acidentes import renderizar_pagina_agente_acidentes
# Import the function you created


# Custom CSS for page appearance
st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem;
        text-align: center;
        padding: 1rem 0;
        background: #ff5100ff;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
    }
    
    /* --- MAIN BUTTONS --- */
    .stButton>button {
        background-color: #ff5100;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        border: none;
        padding: 0.5rem 1rem;
        font-size: 1.1rem;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #D9480F;
        box-shadow: 0 5px 15px rgba(255, 81, 0, 0.3);
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }

    div[data-testid="stAlertContainer"] {
        background-color: rgba(255, 81, 0, 0.15);
    }

    div[data-testid="stAlertContentInfo"] p,
    div[data-testid="stAlertContentSuccess"] p {
        color: #D9480F;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# PREPROCESSING FUNCTION FOR FILE UPLOAD
# ==============================================================================
def pre_processamento_df_completo(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering function for PRF dataset.
    Receives the raw DataFrame and returns it with new features.
    Optimized for imbalanced classes with focus on fatal accidents.
    """

    df = df.copy()

    # --------------------------
    # 1. Date Processing (SIMPLIFIED)
    # --------------------------
    df['data_inversa'] = pd.to_datetime(
        df['data_inversa'], format="%Y-%m-%d", errors='coerce'
    )
    df['ano'] = df['data_inversa'].dt.year

    df['mes'] = df['data_inversa'].dt.month
    df['dia_semana_num'] = df['data_inversa'].dt.weekday

    df['fim_semana'] = df['dia_semana_num'].isin([5, 6]).astype(int)

    df['mes_sin'] = np.sin(2 * np.pi * df['mes'] / 12)
    df['mes_cos'] = np.cos(2 * np.pi * df['mes'] / 12) 
# --------------------------
    # 2. Time Processing (SIMPLIFIED)
    # --------------------------
    df['horario'] = pd.to_datetime(
        df['horario'], format="%H:%M:%S", errors='coerce'
    )
    df['hora'] = df['horario'].dt.hour
    
    def time_period(hour):
        if pd.isna(hour):
            return np.nan
        if 0 <= hour < 6:
            return "dawn"
        elif 6 <= hour < 12:
            return "morning"
        elif 12 <= hour < 18:
            return "afternoon"
        else:
            return "night"

    df['periodo_dia'] = df['hora'].apply(time_period)

    df['is_dawn'] = (df['periodo_dia'] == 'dawn').astype(int)

    # --------------------------
    # 3. Vehicle Categorization (MORE DETAILED)
    # --------------------------
    light_vehicles = [
        'Automóvel', 'Utilitário', 'Caminhonete'
    ]
    
    vulnerable_vehicles = [
        'Motocicleta', 'Ciclomotor',
        'Motoneta', 'Bicicleta', 'Triciclo'
    ]

    heavy_vehicles = [
        'Caminhão', 'Caminhão-trator', 'Ônibus',
        'Micro-ônibus', 'Trator de rodas', 'Trator misto'
    ]

    df['categoria_veiculo'] = np.where(
        df['tipo_veiculo'].isin(light_vehicles), 'light',
        np.where(
            df['tipo_veiculo'].isin(vulnerable_vehicles), 'vulnerable',
            np.where(df['tipo_veiculo'].isin(heavy_vehicles), 'heavy', 'other')
        )
    )

    df['is_vulnerable_vehicle'] = (
        df['categoria_veiculo'] == 'vulnerable'
    ).astype(int)

    # --------------------------
    # 4. Accident Conditions
    # --------------------------
    poor_visibility = [
        'Chuva', 'Garoa/Chuvisco',
        'Nevoeiro/Neblina', 'Vento', 'Nublado'
    ]

    df['poor_visibility'] = (
        df['condicao_metereologica']
        .isin(poor_visibility)
        .astype(int)
    )

    # --------------------------
    # 5. CRITICAL FEATURES FOR FATAL ACCIDENTS
    # --------------------------
    severe_accidents = [
        "Atropelamento de Pedestre",
        "Atropelamento de Animal",
        "Capotamento",
        "Colisão frontal",
        "Colisão transversal",
        "Colisão traseira",
        "Colisão lateral sentido oposto",
        "Saída de leito carroçável",
        "Queda de ocupante de veículo",
        "Tombamento",
        "Incêndio"
    ]

    df['severe_accident_type'] = (
        df['tipo_acidente']
        .isin(severe_accidents)
        .astype(int)
    )

    critical_causes = [
        "Ausência de reação do condutor",
        "Reação tardia ou ineficiente do condutor",
        "Velocidade Incompatível",
        "Condutor Dormindo",
        "Ingestão de álcool pelo condutor",
        "Ingestão de substâncias psicoativas pelo condutor",
        "Transitar na contramão",
        "Ultrapassagem Indevida",
        "Condutor usando celular",
        "Manobra de mudança de faixa",
        "Condutor deixou de manter distância do veículo da frente",
        "Condutor desrespeitou a iluminação vermelha do semáforo",
        "Participar de racha",
        "Mal súbito do condutor",
        "Atropelamento por entrada inopinada do pedestre",
        "Pedestre cruzava a pista fora da faixa",
        "Pedestre andava na pista"
    ]

    df['critical_cause'] = (
        df['causa_principal']
        .isin(critical_causes)
        .astype(int)
    )

    # Dangerous road layouts
    dangerous_layouts = [
        "Curva",
        "Curva;Aclive",
        "Curva;Declive",
        "Curva;Interseção de Vias",
        "Curva;Rotatória",
        "Declive",
        "Declive;Curva",
        "Declive;Interseção de Vias",
        "Declive;Rotatória",
        "Interseção de Vias",
        "Rotatória",
        "Viaduto;Curva",
        "Viaduto;Declive",
        "Ponte;Curva",
        "Ponte;Declive",
        "Túnel;Curva",
        "Túnel;Declive"
    ]

    df['dangerous_layout'] = (
        df['tracado_via']
        .str.contains('|'.join(dangerous_layouts), case=False, na=False)
        .astype(int)
    )

    df['single_lane'] = (
        df['tipo_pista']
        .str.contains('Simples', case=False, na=False)
        .astype(int)
    ) 
# --------------------------
    # 6. HIGH-RISK INTERACTIONS (FOCUS ON FATALITY)
    # --------------------------

    # Interaction 1: Severe accident type + vulnerable vehicle
    df['severe_accident_x_vulnerable'] = (
        (df['severe_accident_type'] == 1) &
        (df['is_vulnerable_vehicle'] == 1)
    ).astype(int)

    # Interaction 2: Head-on collision (extremely lethal)
    df['is_head_on_collision'] = (
        df['tipo_acidente']
        .str.contains('frontal', case=False, na=False)
    ).astype(int)

    # Interaction 3: Speed + poor visibility
    df['speed_x_visibility'] = (
        df['causa_principal']
        .str.contains('velocidade', case=False, na=False) &
        (df['poor_visibility'] == 1)
    ).astype(int)

    # Interaction 4: Alcohol / Drugs
    df['is_alcohol_drugs'] = (
        df['causa_principal']
        .str.contains('álcool|drogas', case=False, na=False)
    ).astype(int)

    # Interaction 5: Dawn + weekend
    df['dawn_weekend'] = (
        (df['is_dawn'] == 1) &
        (df['fim_semana'] == 1)
    ).astype(int)

    # Interaction 6: Poor visibility + night or dawn
    df['poor_visibility_x_period'] = (
        (df['poor_visibility'] == 1) &
        (df['periodo_dia'].isin(['night', 'dawn']))
    ).astype(int)

    # Interaction 7: Heavy vehicle + single lane road
    df['heavy_vehicle_single_lane'] = (
        (df['categoria_veiculo'] == 'heavy') &
        (df['single_lane'] == 1)
    ).astype(int)

    # Interaction 8: Dangerous layout + poor visibility
    df['dangerous_layout_x_visibility'] = (
        (df['dangerous_layout'] == 1) &
        (df['poor_visibility'] == 1)
    ).astype(int)

    # Interaction 9: Accident type + cause
    df['accident_type_x_cause'] = (
        df['tipo_acidente'].astype(str) + '_' +
        df['causa_principal'].astype(str)
    )

    # Interaction 10: Vehicle category + accident type
    df['vehicle_category_x_accident_type'] = (
        df['categoria_veiculo'].astype(str) + '_' +
        df['tipo_acidente'].astype(str)
    )

    # --------------------------
    # 7. FATALITY RISK SCORE (AGGREGATED)
    # --------------------------
    df['fatality_risk_score'] = (
        df['severe_accident_type'] * 3 +
        df['critical_cause'] * 2 +
        df['is_vulnerable_vehicle'] * 2 +
        df['is_head_on_collision'] * 3 +
        df['is_alcohol_drugs'] * 2 +
        df['poor_visibility'] +
        df['dawn_weekend'] +
        df['dangerous_layout'] +
        df['single_lane']
    )

    # --------------------------
    # 8. DROP ORIGINAL COLUMNS USED
    # --------------------------
    drop_cols = [
        'data_inversa', 'horario',
        'ano_fabricacao_veiculo', 'tipo_veiculo',
        'idade', 'condicao_metereologica',
        'id', 'pesid', 'id_veiculo',
        'marca', 'km', 'latitude', 'longitude',
        'municipio', 'delegacia', 'uop',
        'regional', 'estado_fisico',
        'ilesos', 'feridos_leves',
        'hora', 'feridos_graves', 'mortos',
        'sexo', 'mes', 'dia',
        'dia_semana_num', 'dia_semana',
        'causa_acidente', 'ordem_tipo_acidente',
        'tipo_envolvido', 'ano'
    ]

    df = df.drop(
        columns=[c for c in drop_cols if c in df.columns]
    )

    return df 
# ==============================================================================
# 1. AUXILIARY DATABASE (MAPPING)
# Maps (State, BR) to road characteristics.
# ==============================================================================
@st.cache_data
def carregar_dados_brs():
    """
    Reads the CSV file with BR characteristics and caches it.
    """
    df = pd.read_csv('data/caracteristicas_brs.csv')
    return df


# ==============================================================================
# 2. PREPROCESSING FUNCTION FOR ROUTE PREDICTION
# ==============================================================================
def pre_processamento_df_manual(
    data_hora_viagem: datetime,
    tipo_veiculo_usuario: str,
    condicao_visibilidade_usuario: str,
    rota: list,
    db_brs: pd.DataFrame
) -> pd.DataFrame:
    """
    Prepares Streamlit input data for the prediction model.

    Args:
        data_hora_viagem (datetime): Travel start date and time.
        tipo_veiculo_usuario (str): Vehicle type.
        condicao_visibilidade_usuario (str): Visibility condition.
        rota (list): List of route segments.
        db_brs (pd.DataFrame): BR characteristics database.

    Returns:
        pd.DataFrame: Processed DataFrame ready for prediction.
    """

    # --- STEP A: DEFINE RISK SCENARIOS ---
    cenarios_de_risco = [
        {'tipo_acidente': 'Colisão frontal', 'causa_principal': 'Velocidade Incompatível'},
        {'tipo_acidente': 'Saída de leito carroçável', 'causa_principal': 'Condutor Dormindo'},
        {'tipo_acidente': 'Atropelamento de Pedestre', 'causa_principal': 'Falta de Atenção do Pedestre'},
        {'tipo_acidente': 'Colisão traseira', 'causa_principal': 'Falta de Atenção à Condução'}
    ]

    # --- STEP B: BUILD DATAFRAME FROM USER INPUTS ---
    dados_para_prever = []

    for trecho in rota:
        info_br = db_brs[
            (db_brs['uf'] == trecho['uf']) &
            (db_brs['br'] == trecho['br'])
        ]

        if not info_br.empty:
            tipo_pista = info_br.iloc[0]['tipo_pista']
            tracado_via = info_br.iloc[0]['tracado_via']
        else:
            tipo_pista = 'Simples'
            tracado_via = 'Reta'
            
        for cenario in cenarios_de_risco:
            linha = {
                'data_inversa': data_hora_viagem.strftime('%Y-%m-%d'),
                'horario': data_hora_viagem.strftime('%H:%M:%S'),
                'tipo_veiculo': tipo_veiculo_usuario,
                'condicao_metereologica': condicao_visibilidade_usuario,
                'uf': trecho['uf'],
                'br': trecho['br'],
                'tipo_pista': tipo_pista,
                'tracado_via': tracado_via,
                'tipo_acidente': cenario['tipo_acidente'],
                'causa_principal': cenario['causa_principal']
            }
            dados_para_prever.append(linha)

    if not dados_para_prever:
        return pd.DataFrame()

    df = pd.DataFrame(dados_para_prever)

    # -------------------------------------------------------------------------
    # STEP C: APPLY ORIGINAL FEATURE ENGINEERING LOGIC
    # -------------------------------------------------------------------------
    df['data_inversa'] = pd.to_datetime(df['data_inversa'], format="%Y-%m-%d")
    df['mes'] = df['data_inversa'].dt.month
    df['dia_semana_num'] = df['data_inversa'].dt.weekday
    df['fim_semana'] = df['dia_semana_num'].isin([5, 6]).astype(int)
    df['mes_sin'] = np.sin(2 * np.pi * df['mes'] / 12)
    df['mes_cos'] = np.cos(2 * np.pi * df['mes'] / 12)

    df['horario'] = pd.to_datetime(df['horario'], format="%H:%M:%S")
    df['hora'] = df['horario'].dt.hour

    def time_period(hour):
        if 0 <= hour < 6: return "dawn"
        elif 6 <= hour < 12: return "morning"
        elif 12 <= hour < 18: return "afternoon"
        else: return "night"

    df['periodo_dia'] = df['hora'].apply(time_period)
    df['is_dawn'] = (df['periodo_dia'] == 'dawn').astype(int)

    vulnerable_vehicles = ['Motocicleta', 'Ciclomotor', 'Motoneta', 'Bicicleta']
    heavy_vehicles = ['Caminhão', 'Caminhão-trator', 'Ônibus', 'Micro-ônibus']

    df['categoria_veiculo'] = np.where(
        df['tipo_veiculo'].isin(vulnerable_vehicles), 'vulnerable',
        np.where(df['tipo_veiculo'].isin(heavy_vehicles), 'heavy', 'light')
    )

    df['is_vulnerable_vehicle'] = (df['categoria_veiculo'] == 'vulnerable').astype(int)

    poor_visibility_map = ['Chuva ou Neblina', 'Poeira ou Fumaça']
    df['poor_visibility'] = df['condicao_metereologica'].isin(poor_visibility_map).astype(int) 
# 5. Critical features (based on simulated scenarios)
    severe_accidents = [
        "Atropelamento de Pedestre",
        "Capotamento",
        "Colisão frontal",
        "Saída de leito carroçável"
    ]

    df['severe_accident_type'] = (
        df['tipo_acidente']
        .isin(severe_accidents)
        .astype(int)
    )

    critical_causes = [
        "Velocidade Incompatível",
        "Condutor Dormindo",
        "Ingestão de álcool pelo condutor",
        "Transitar na contramão"
    ]

    df['critical_cause'] = (
        df['causa_principal']
        .isin(critical_causes)
        .astype(int)
    )

    dangerous_layouts = [
        "Curva",
        "Curva;Aclive",
        "Curva;Declive",
        "Declive"
    ]

    df['dangerous_layout'] = (
        df['tracado_via']
        .isin(dangerous_layouts)
        .astype(int)
    )

    df['single_lane'] = (
        df['tipo_pista']
        .str.contains('Simples', case=False, na=False)
        .astype(int)
    )

    # 6. High-risk interactions
    df['severe_accident_x_vulnerable'] = (
        (df['severe_accident_type'] == 1) &
        (df['is_vulnerable_vehicle'] == 1)
    ).astype(int)

    df['is_head_on_collision'] = (
        df['tipo_acidente']
        .str.contains('frontal', case=False, na=False)
    ).astype(int)

    df['speed_x_visibility'] = (
        df['causa_principal']
        .str.contains('velocidade', case=False, na=False) &
        (df['poor_visibility'] == 1)
    ).astype(int)

    df['is_alcohol_drugs'] = (
        df['causa_principal']
        .str.contains('álcool|drogas', case=False, na=False)
    ).astype(int)

    df['dawn_weekend'] = (
        (df['is_dawn'] == 1) &
        (df['fim_semana'] == 1)
    ).astype(int)

    df['poor_visibility_x_period'] = (
        (df['poor_visibility'] == 1) &
        (df['periodo_dia'].isin(['night', 'dawn']))
    ).astype(int)

    df['heavy_vehicle_single_lane'] = (
        (df['categoria_veiculo'] == 'heavy') &
        (df['single_lane'] == 1)
    ).astype(int)

    df['dangerous_layout_x_visibility'] = (
        (df['dangerous_layout'] == 1) &
        (df['poor_visibility'] == 1)
    ).astype(int)

    df['accident_type_x_cause'] = (
        df['tipo_acidente'].astype(str) + '_' +
        df['causa_principal'].astype(str)
    )

    df['vehicle_category_x_accident_type'] = (
        df['categoria_veiculo'].astype(str) + '_' +
        df['tipo_acidente'].astype(str)
    )

    # 7. Risk score
    df['fatality_risk_score'] = (
        df['severe_accident_type'] * 3 +
        df['critical_cause'] * 2 +
        df['is_vulnerable_vehicle'] * 2 +
        df['is_head_on_collision'] * 3 +
        df['is_alcohol_drugs'] * 2 +
        df['poor_visibility'] +
        df['dawn_weekend'] +
        df['dangerous_layout'] +
        df['single_lane']
    )

    # 8. Drop original columns
    drop_cols = [
        'data_inversa', 'horario', 'tipo_veiculo',
        'condicao_metereologica', 'hora',
        'mes', 'dia_semana_num'
    ]

    df = df.drop(
        columns=[c for c in drop_cols if c in df.columns],
        errors='ignore'
    )

    return df 
# --- Load Models and Objects ---
@st.cache_resource
def load_objects():
    """Loads the model and encoder dictionaries from disk."""
    try:
        model = joblib.load('models/modelo_gravidade_xgb.pkl')
        feature_encoders = joblib.load('models/label_encoders_xgb.pkl')
        target_encoder = joblib.load('models/target_encoder_xgb.pkl')
        selected_features = joblib.load('models/colunas_modelo_xgb.pkl')

        return model, feature_encoders, target_encoder, selected_features

    except FileNotFoundError as e:
        st.error(f"Model file not found: {e.filename}")
        return None, None, None, None
    except Exception as e:
        st.error(f"Error loading objects: {e}")
        return None, None, None, None


# --- Unified Prediction Logic (Corrected) ---
def fazer_previsao_df_completo(df_input, model, feature_encs, selected_features):
    """
    Unified function to preprocess, encode, select features, and predict.

    Args:
        df_input (pd.DataFrame): Raw input data.
        model: Trained model.
        feature_encs (dict): Dictionary of saved LabelEncoders.
        selected_features (list): List of features expected by the model.

    Returns:
        np.array: Class probabilities.
    """
    print("Starting preprocessing for inference...")

    df_processed = pre_processamento_df_completo(df_input.copy())
    df_encoded = df_processed.copy()

    for col, encoder in feature_encs.items():
        if col in df_encoded.columns:
            known_classes = set(encoder.classes_)

            def safe_transform(value):
                if value in known_classes:
                    return encoder.transform([value])[0]
                else:
                    return -1

            df_encoded[col] = df_encoded[col].astype(str).apply(safe_transform)

    X_to_predict = df_encoded.drop(
        'classificacao_acidente', axis=1, errors='ignore'
    )

    try:
        X_final = X_to_predict[selected_features]
    except KeyError as e:
        print(
            f"Error: Feature '{e}' not found after preprocessing."
        )
        return None

    probabilities = model.predict_proba(X_final)
    print("Predictions completed.")

    return probabilities


# --- Prediction Function for Manual Input ---
def fazer_previsao_df_manual(
    df_processado: pd.DataFrame,
    model,
    feature_encs: dict,
    selected_features: list
) -> np.array:
    """
    Performs prediction from a preprocessed DataFrame.

    Steps:
    1. Encode categorical features.
    2. Align columns with model expectations.
    3. Return class probabilities.
    """
    warnings.filterwarnings('ignore')

    try:
        df_encoded = df_processado.copy()

        for col in df_encoded.columns:
            if col in feature_encs:
                encoder = feature_encs[col]
                known_classes = set(encoder.classes_)

                df_encoded[col] = df_encoded[col].apply(
                    lambda x: encoder.transform([str(x)])[0]
                    if str(x) in known_classes else -1
                )

        for feat in selected_features:
            if feat not in df_encoded.columns:
                df_encoded[feat] = 0

        X_final = df_encoded[selected_features]
        probabilities = model.predict_proba(X_final)

        return probabilities

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        traceback.print_exc()
        return None 
def rederizar_pagina_upload_arquivo():
    st.sidebar.header("Input Parameters")
    uploaded_file = st.sidebar.file_uploader(
        "Select CSV file",
        type="csv",
        help="Upload a CSV file with accident data"
    )
    
    if uploaded_file:
        try:
            file_content = uploaded_file.getvalue()
            
            try:
                data_string = file_content.decode('utf-8')
                st.success("File read as UTF-8")
                data_io = io.StringIO(data_string)
                df_input = pd.read_csv(data_io, sep=',')
            except UnicodeDecodeError:
                data_string = file_content.decode('latin-1')
                st.warning("File read using LATIN-1 encoding")
                data_io = io.StringIO(data_string)
                df_input = pd.read_csv(data_io, sep=';')

            # ==========================================
            # SECTION 1: DATASET INFORMATION
            # ==========================================
            st.header("Loaded Dataset Analysis")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Records", f"{len(df_input):,}")
            with col2:
                st.metric("Total Columns", len(df_input.columns))
            with col3:
                memory_mb = df_input.memory_usage(deep=True).sum() / 1024**2
                st.metric("Memory Size", f"{memory_mb:.1f} MB")
            with col4:
                missing_pct = (
                    df_input.isnull().sum().sum() /
                    (len(df_input) * len(df_input.columns))
                ) * 100
                st.metric("Missing Data %", f"{missing_pct:.1f}%")
            
            with st.expander("View Data Sample (First 10 rows)", expanded=False):
                st.dataframe(df_input.head(10), use_container_width=True)
            
            if len(df_input) > 100000:
                st.warning(
                    f"The file contains {len(df_input):,} rows. "
                    "Processing may take a few minutes."
                )
                
                use_sample = st.checkbox(
                    "Process only a random sample of 10,000 records",
                    value=False
                )
                if use_sample:
                    df_input = df_input.sample(n=10000, random_state=42)
                    st.info(f"Processing sample of {len(df_input):,} records")

            # ==========================================
            # SECTION 2: PROCESSING AND PREDICTIONS
            # ==========================================
            st.header("Severity Predictions")
            
            with st.spinner('Applying preprocessing and generating predictions...'):
                import time
                start_time = time.time()
                
                probabilities = fazer_previsao_df_completo(
                    df_input,
                    modelo,
                    encoder_features,
                    colunas_modelo
                )
                
                processing_time = time.time() - start_time
                
                df_results = pd.DataFrame(
                    probabilities,
                    columns=nomes_das_classes
                )

                for col in nomes_das_classes:
                    df_results[col] = pd.to_numeric(
                        df_results[col], errors='coerce'
                    )

                df_results['Predicted_Class'] = (
                    df_results[nomes_das_classes].idxmax(axis=1)
                )
                df_results['Confidence'] = (
                    df_results[nomes_das_classes].max(axis=1)
                )

                df_input_com_pred = df_input.copy()
                df_input_com_pred['Predicted_Severity'] = (
                    df_results['Predicted_Class'].values
                )
                df_input_com_pred['Prediction_Confidence'] = (
                    df_results['Confidence'].values
                )
            
            st.success(
                f"Prediction completed in {processing_time:.2f} seconds!"
            ) 
# ==========================================
            # SECTION 3: PREDICTION DISTRIBUTION
            # ==========================================
            st.subheader("Prediction Distribution")
            
            col1, col2 = st.columns(2)
            
            with col1:
                class_counts = df_results['Predicted_Class'].value_counts()
                
                fig_pie = go.Figure(data=[go.Pie(
                    labels=class_counts.index,
                    values=class_counts.values,
                    hole=0.4,
                    marker=dict(colors=['#2ca02c', "#ffdb0e", '#d62728']),
                    textinfo='label+percent',
                    textfont_size=12
                )])
                
                fig_pie.update_layout(
                    title='Accident Severity Distribution',
                    height=400,
                    showlegend=True
                )
                
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                fig_bar = go.Figure(data=[go.Bar(
                    x=class_counts.index,
                    y=class_counts.values,
                    text=class_counts.values,
                    textposition='auto',
                    marker=dict(
                        color=['#2ca02c', "#ffdb0e", '#d62728'],
                        line=dict(color='black', width=1.5)
                    )
                )])
                
                fig_bar.update_layout(
                    title='Accident Count by Severity',
                    xaxis_title='Severity',
                    yaxis_title='Number of Accidents',
                    height=400
                )
                
                st.plotly_chart(fig_bar, use_container_width=True)
            
            # Distribution metrics
            st.markdown("### Distribution Statistics")
            col1, col2, col3 = st.columns(3)
            
            total = len(df_results)
            for col, severity in zip([col1, col2, col3], nomes_das_classes):
                count = class_counts.get(severity, 0)
                pct = (count / total) * 100 if total > 0 else 0
                with col:
                    st.metric(
                        severity,
                        f"{count:,} ({pct:.1f}%)"
                    )
            
            # ==========================================
            # SECTION 4: CONFIDENCE ANALYSIS
            # ==========================================
            st.subheader("Prediction Confidence Analysis")
            
            fig_hist = go.Figure()
            
            for severity in nomes_das_classes:
                mask = df_results['Predicted_Class'] == severity
                confidences = df_results.loc[mask, 'Confidence']
                
                fig_hist.add_trace(go.Histogram(
                    x=confidences * 100,
                    name=severity,
                    opacity=0.7,
                    nbinsx=30
                ))
            
            fig_hist.update_layout(
                title='Confidence Distribution by Class',
                xaxis_title='Confidence (%)',
                yaxis_title='Frequency',
                barmode='overlay',
                height=400
            )
            
            st.plotly_chart(fig_hist, use_container_width=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "Average Confidence",
                    f"{df_results['Confidence'].mean():.1%}"
                )
            with col2:
                st.metric(
                    "Median Confidence",
                    f"{df_results['Confidence'].median():.1%}"
                )
            with col3:
                low_conf = (df_results['Confidence'] < 0.5).sum()
                st.metric(
                    "Low Confidence Predictions (<50%)",
                    f"{low_conf:,}"
                ) 
# ==========================================
            # SECTION 5: CRITICAL CASES (FATAL)
            # ==========================================
            fatal_cases = df_results[
                df_results['Predicted_Class'] == 'Com Vítimas Fatais'
            ]
            
            if len(fatal_cases) > 0:
                st.subheader(
                    f"Fatal Accident Analysis ({len(fatal_cases):,} cases)"
                )
                
                col1, col2 = st.columns(2)
                
                with col1:
                    avg_conf_fatal = fatal_cases['Confidence'].mean()
                    high_conf = (fatal_cases['Confidence'] >= 0.7).sum()
                    
                    st.metric(
                        "Average Confidence (Fatal)",
                        f"{avg_conf_fatal:.1%}"
                    )
                    st.metric(
                        "High Confidence Cases (≥70%)",
                        f"{high_conf:,}"
                    )
                
                with col2:
                    fig_box = go.Figure()
                    
                    for severity in nomes_das_classes:
                        fig_box.add_trace(go.Box(
                            y=fatal_cases[severity] * 100,
                            name=severity,
                            boxmean='sd'
                        ))
                    
                    fig_box.update_layout(
                        title='Probability Distribution (Fatal Cases)',
                        yaxis_title='Probability (%)',
                        height=300
                    )
                    
                    st.plotly_chart(
                        fig_box,
                        use_container_width=True
                    )
            
            # ==========================================
            # SECTION 6: DETAILED PROBABILITY TABLE
            # ==========================================
            st.subheader("Detailed Probability Table")
            
            col1, col2 = st.columns(2)
            with col1:
                class_filter = st.multiselect(
                    "Filter by Predicted Class",
                    options=nomes_das_classes,
                    default=nomes_das_classes
                )
            with col2:
                min_conf = st.slider(
                    "Minimum Confidence (%)",
                    min_value=0,
                    max_value=100,
                    value=0,
                    step=5
                ) / 100
            
            mask = (
                df_results['Predicted_Class'].isin(class_filter) &
                (df_results['Confidence'] >= min_conf)
            )
            df_filtered = df_results[mask]
            
            st.info(
                f"Showing {len(df_filtered):,} of "
                f"{len(df_results):,} records"
            )
            
            st.dataframe(
                df_filtered.style
                .format({
                    col: "{:.2%}"
                    for col in nomes_das_classes + ['Confidence']
                })
                .background_gradient(
                    cmap='RdYlGn_r',
                    subset=nomes_das_classes
                )
                .highlight_max(
                    axis=1,
                    subset=nomes_das_classes,
                    props='font-weight: bold; border: 2px solid green;'
                ),
                use_container_width=True,
                height=400
            )
            
            # ==========================================
            # SECTION 7: DOWNLOAD RESULTS
            # ==========================================
            st.subheader("Export Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                csv_probs = (
                    df_results
                    .to_csv(index=False)
                    .encode('utf-8')
                )
                st.download_button(
                    label="Download Probabilities (CSV)",
                    data=csv_probs,
                    file_name="accident_probabilities.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col2:
                csv_full = (
                    df_input_com_pred
                    .to_csv(index=False)
                    .encode('utf-8')
                )
                st.download_button(
                    label="Download Full Dataset + Predictions (CSV)",
                    data=csv_full,
                    file_name="dataset_with_predictions.csv",
                    mime="text/csv",
                    use_container_width=True
                ) 
# ==========================================
            # SECTION 8: ADDITIONAL ANALYSIS (GEOGRAPHIC)
            # ==========================================
            if 'uf' in df_input.columns:
                st.subheader("Geographic Analysis")
                
                df_geo = (
                    df_input_com_pred
                    .groupby('uf')
                    .agg({
                        'Predicted_Severity': lambda x: x.value_counts().to_dict(),
                        'Prediction_Confidence': 'mean'
                    })
                    .reset_index()
                )
                
                severity_by_state = []
                for _, row in df_geo.iterrows():
                    state = row['uf']
                    counts = row['Predicted_Severity']
                    for severity, count in counts.items():
                        severity_by_state.append({
                            'State': state,
                            'Severity': severity,
                            'Count': count
                        })
                
                df_geo_expanded = pd.DataFrame(severity_by_state)
                
                fig_geo = go.Figure()
                
                for severity in nomes_das_classes:
                    df_temp = df_geo_expanded[
                        df_geo_expanded['Severity'] == severity
                    ]
                    fig_geo.add_trace(go.Bar(
                        name=severity,
                        x=df_temp['State'],
                        y=df_temp['Count'],
                        text=df_temp['Count'],
                        textposition='auto'
                    ))
                
                fig_geo.update_layout(
                    title='Severity Distribution by State',
                    xaxis_title='State (UF)',
                    yaxis_title='Number of Accidents',
                    barmode='stack',
                    height=500
                )
                
                st.plotly_chart(fig_geo, use_container_width=True)
            
            st.balloons()

        except Exception as e:
            st.error(f"An error occurred during processing: {e}")
            st.error(
                "Please verify that the CSV file has the expected format "
                "and all required columns."
            )
            
            with st.expander("View Error Details"):
                st.code(traceback.format_exc())
    
    else:
        st.markdown("---")
        st.info("Waiting for CSV file upload...")
        
        st.markdown("""
        ### How to use:
        
        1. Click **\"Browse files\"** in the sidebar
        2. Select a CSV file with accident data
        3. The file will be automatically processed
        4. View predictions, charts, and statistics
        5. Download results if needed
        
        ### Expected file format:
        
        The CSV file must contain the following main columns:
        - `data_inversa`: Accident date (YYYY-MM-DD)
        - `horario`: Accident time (HH:MM:SS)
        - `uf`: State
        - `tipo_veiculo`: Vehicle type
        - `tipo_acidente`: Accident type
        - `causa_principal`: Main cause
        
        ### Tips:
        - For large files (>100k rows), consider sampling
        - Results can be downloaded as CSV
        - Use filters to explore specific cases
        """) 
def renderizar_pagina_entrada_manual():
    st.set_page_config(
        page_title="Prediction Models - Accident Analysis",
        layout="wide",
    )

    tab1, tab2 = st.tabs(
        ["**Run Prediction**", "**Route Prediction History**"]
    )

    with tab1:
        st.header("Mode: Route Risk Prediction")
        st.info(
            "Plan your trip! Add the BR segments you will travel through "
            "and get a risk analysis based on your trip conditions."
        )

        if 'rota' not in st.session_state:
            st.session_state.rota = []
        if 'historico_rotas' not in st.session_state:
            st.session_state.historico_rotas = []

        def carregar_rota_selecionada():
            selected_route = st.session_state.rotas_selectbox
            if selected_route in ROTAS_PREDEFINIDAS:
                st.session_state.rota = ROTAS_PREDEFINIDAS[selected_route]

        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("🗓️ Trip Parameters")

            data_hora_viagem = st.date_input(
                "Trip Start Date and Time",
                datetime.datetime.now()
            )

            tipo_veiculo_usuario = st.selectbox(
                "Your Vehicle Type",
                ['Automóvel', 'Motocicleta', 'Caminhão', 'Ônibus', 'Utilitário']
            )

            condicao_visibilidade_usuario = st.selectbox(
                "Visibility Conditions",
                [
                    'Tempo Bom (Céu Claro/Nublado)',
                    'Chuva ou Neblina',
                    'Poeira ou Fumaça'
                ]
            )

        with col2:
            st.subheader("Build Your Route")

            st.selectbox(
                label="Or choose a suggested route (this will replace the current route):",
                options=list(ROTAS_PREDEFINIDAS.keys()),
                key="rotas_selectbox",
                on_change=carregar_rota_selecionada
            )

            st.markdown("---")

            with st.form("form_add_trecho"):
                ufs = (
                    sorted(db_brs['uf'].unique())
                    if db_brs is not None else ['SP']
                )

                uf_selecionada = st.selectbox("Segment State (UF)", ufs)
                br_selecionada = st.number_input(
                    "BR Number",
                    min_value=1,
                    max_value=499,
                    step=1,
                    value=116
                )

                submitted = st.form_submit_button("Add Segment")

                if submitted:
                    st.session_state.rota.append(
                        {'uf': uf_selecionada, 'br': int(br_selecionada)}
                    )
                    st.success(
                        f"Segment BR-{br_selecionada}/{uf_selecionada} added!"
                    )

        if st.session_state.rota:
            st.write("**Current Route:**")
            df_rota = pd.DataFrame(st.session_state.rota)
            st.dataframe(df_rota, use_container_width=True)

            if st.button("Clear Route"):
                st.session_state.rota = []
                st.rerun()

        if st.button(
            "**Analyze Full Route**",
            type="primary",
            use_container_width=True
        ):
            if not st.session_state.rota:
                st.warning(
                    "Please add at least one route segment."
                )

            elif modelo is None or db_brs is None:
                st.error(
                    "Application not properly initialized. "
                    "Check model and data files."
                )

            else:
                with st.spinner(
                    '🔄 Analyzing route and calculating risks...'
                ):
                    df_processado = pre_processamento_df_manual(
                        data_hora_viagem=data_hora_viagem,
                        tipo_veiculo_usuario=tipo_veiculo_usuario,
                        condicao_visibilidade_usuario=condicao_visibilidade_usuario,
                        rota=st.session_state.rota,
                        db_brs=db_brs
                    ) 
# STEP 2: Prediction
                    probabilities = fazer_previsao_df_manual(
                        df_processado=df_processado,
                        model=modelo,
                        feature_encs=encoder_features,
                        selected_features=colunas_modelo
                    )

                    if probabilities is not None:
                        st.success("✅ Risk analysis completed!")

                        # Aggregate probabilities across all segments and scenarios
                        aggregated_risk = np.mean(probabilities, axis=0)

                        predicted_class = nomes_das_classes[
                            aggregated_risk.argmax()
                        ]
                        predicted_prob = aggregated_risk.max()

                        history_record = {
                            "timestamp": datetime.datetime.now(),
                            "route": pd.DataFrame(st.session_state.rota),
                            "parameters": {
                                "Date": data_hora_viagem.strftime('%d/%m/%Y %H:%M'),
                                "Vehicle": tipo_veiculo_usuario,
                                "Visibility": condicao_visibilidade_usuario
                            },
                            "aggregated_risk": aggregated_risk,
                            "predicted_class": predicted_class,
                            "predicted_prob": predicted_prob
                        }

                        st.session_state.historico_rotas.insert(
                            0, history_record
                        )

                        # STEP 3: Display aggregated results
                        st.subheader("📊 Aggregated Route Risk")
                        st.write(
                            "Overall route risk is the average probability "
                            "across all segments and simulated risk scenarios."
                        )

                        df_results = pd.DataFrame(
                            [aggregated_risk],
                            columns=nomes_das_classes
                        )

                        st.dataframe(
                            df_results.style
                            .format("{:.2%}")
                            .background_gradient(
                                cmap='YlOrRd',
                                axis=1
                            ),
                            use_container_width=True
                        )

                        fig = go.Figure(data=[go.Bar(
                            x=nomes_das_classes,
                            y=aggregated_risk * 100,
                            text=[
                                f'{p:.1f}%' for p in aggregated_risk * 100
                            ],
                            textposition='auto',
                            marker_color=[
                                '#d62728', "#ffdb0e", '#2ca02c'
                            ]
                        )])

                        fig.update_layout(
                            title='Route Risk Probability Distribution',
                            yaxis_title='Probability (%)',
                            yaxis_range=[0, 100]
                        )

                        st.plotly_chart(
                            fig,
                            use_container_width=True
                        )

                        # Risk alerts
                        if predicted_class == 'Com Vítimas Fatais':
                            st.error(
                                f"🚨 **HIGH RISK**: Highest probability is "
                                f"**{predicted_class}** ({predicted_prob:.1%}). "
                                "Increase attention!"
                            )

                        elif predicted_class == 'Com Vítimas Feridas':
                            st.warning(
                                f"⚠️ **MODERATE RISK**: Highest probability is "
                                f"**{predicted_class}** ({predicted_prob:.1%}). "
                                "Drive carefully."
                            )

                        else:
                            st.success(
                                f"✅ **LOW RISK**: Highest probability is "
                                f"**{predicted_class}** ({predicted_prob:.1%}). "
                                "Have a safe trip!"
                            )

                        with st.expander(
                            "View scenario-level analysis details"
                        ):
                            st.write(
                                "The table below shows probabilities for each "
                                "route segment and simulated risk scenario."
                            )

                            df_details = pd.DataFrame(
                                probabilities,
                                columns=nomes_das_classes
                            )

                            st.dataframe(
                                df_details.style
                                .format("{:.2%}")
                                .background_gradient(
                                    cmap='YlOrRd',
                                    axis=1
                                ),
                                use_container_width=True
                            ) 
# ==========================================================================
    # TAB 2: HISTORY
    # ==========================================================================
    with tab2:
        st.header("📜 Session Route Prediction History")

        if not st.session_state.historico_rotas:
            st.info(
                "No route predictions have been made in this session yet. "
                "Run an analysis in the tab on the left."
            )
        else:
            if st.button("🗑️ Clear History"):
                st.session_state.historico_rotas = []
                st.rerun()

            for i, record in enumerate(st.session_state.historico_rotas):
                timestamp_str = record['timestamp'].strftime(
                    '%d/%m/%Y at %H:%M:%S'
                )

                risk_color = (
                    "red"
                    if record['predicted_class'] == 'Com Vítimas Fatais'
                    else "orange"
                    if record['predicted_class'] == 'Com Vítimas Feridas'
                    else "green"
                )

                expander_title = (
                    f"**Prediction #{len(st.session_state.historico_rotas) - i}** "
                    f"({timestamp_str}) - Dominant Risk: "
                    f"**:{risk_color}[{record['predicted_class']}]**"
                )

                with st.expander(expander_title):
                    st.write("**Trip Parameters:**")
                    st.json(record['parameters'])

                    st.write("**Analyzed Route:**")
                    st.dataframe(
                        record['route'],
                        use_container_width=True
                    )

                    st.write("**Risk Analysis Result:**")
                    df_hist = pd.DataFrame(
                        [record['aggregated_risk']],
                        columns=nomes_das_classes
                    )

                    st.dataframe(
                        df_hist.style
                        .format("{:.2%}")
                        .background_gradient(
                            cmap='YlOrRd',
                            axis=1
                        ),
                        use_container_width=True
                    ) 
# --- Predefined Routes ---
ROTAS_PREDEFINIDAS = {
    "Trip SP ➔ RJ (Via Dutra)": [
        {'uf': 'SP', 'br': 116},
        {'uf': 'RJ', 'br': 116}
    ],
    "South Coast of SP (Régis Bittencourt + Rio-Santos)": [
        {'uf': 'SP', 'br': 116},
        {'uf': 'SP', 'br': 101}
    ],
    "Trip PR ➔ SC (Coastal Route)": [
        {'uf': 'PR', 'br': 116},
        {'uf': 'PR', 'br': 101},
        {'uf': 'SC', 'br': 101}
    ],
    "Main Route of Minas Gerais (Fernão Dias)": [
        {'uf': 'MG', 'br': 381}
    ]
}

nomes_das_classes = [
    'Com Vítimas Fatais',
    'Com Vítimas Feridas',
    'Ilesos'
]

modelo, encoder_features, encoder_target, colunas_modelo = load_objects()
db_brs = carregar_dados_brs()

# ==============================================================================
# STREAMLIT APPLICATION
# ==============================================================================
pd.set_option("styler.render.max_elements", 2000000)

st.set_page_config(
    page_title="Prediction Models - Accident Analysis",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Main Header ---
st.markdown(
    '<div class="main-header">🤖 Accident Severity Prediction Models</div>',
    unsafe_allow_html=True
)
st.markdown(
    "<div style='text-align: center; font-size: 1.2em; color: #666;'>"
    "Interact with Machine Learning models to predict accident severity "
    "under different scenarios."
    "</div>",
    unsafe_allow_html=True
)

# Check if models were loaded correctly
if modelo is None:
    st.error(
        "Model files not found! Make sure the .pkl files are in the correct directory."
    )
else:
    # --- Button to trigger AI Agent ---
    if st.sidebar.button("Talk to AI Agent"):
        
        @st.dialog("🤖", width="large")
        def agent_dialog():
            renderizar_pagina_agente_acidentes()

        agent_dialog()

    st.sidebar.divider()

    # --- Sidebar Mode Selection ---
    st.sidebar.header("Operation Mode")
    input_mode = st.sidebar.radio(
        "Choose prediction type:",
        ("Batch Prediction (Upload)", "Route Prediction")
    )
    
    # --- Render Selected Page ---
    if input_mode == "Batch Prediction (Upload)":
        rederizar_pagina_upload_arquivo()

    elif input_mode == "Route Prediction":
        renderizar_pagina_entrada_manual() 