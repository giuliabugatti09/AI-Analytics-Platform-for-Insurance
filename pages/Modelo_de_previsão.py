import streamlit as st
import pandas as pd
import numpy as np
import joblib  # Usado para carregar/salvar o modelo e outros objetos
from sklearn.preprocessing import LabelEncoder
import datetime
import io
import warnings
import traceback
import plotly.graph_objects as go

from components.Agente_Acidentes import renderizar_pagina_agente_acidentes # Importe a função que você criou


# CSS personalizado para a aparência da página
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
    .stTabs [aria-selected="true"] {
    }
    div[data-testid="stAlertContainer"] {
    background-color: rgba(255, 81, 0, 0.15); /* Fundo laranja bem suave */
    }

    /* 2. O parágrafo (texto) dentro do st.info */
    div[data-testid="stAlertContentInfo"] p {
        color: #D9480F; /* Texto em um tom de laranja mais escuro para legibilidade */
    div[data-testid="stAlertContentSuccess"] p {
        color: #D9480F; /* Texto em um tom de laranja mais escuro para legibilidade */
    }

</style>
""", unsafe_allow_html=True)


# ==============================================================================
# FUNÇÃO DE PRÉ-PROCESSAMENTO PARA O UPLOAD
# ==============================================================================
def pre_processamento_df_completo(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Função de engenharia de atributos para dataset da PRF.
    Recebe o DataFrame cru e retorna com novas features.
    Otimizado para classes desbalanceadas com foco em acidentes fatais.
    '''

    df = df.copy()

    # --------------------------
    # 1. Processamento de datas (SIMPLIFICADO)
    # --------------------------
    df['data_inversa'] = pd.to_datetime(df['data_inversa'], format="%Y-%m-%d", errors='coerce')
    df['ano'] = df['data_inversa'].dt.year

    # Extração de componentes
    df['mes'] = df['data_inversa'].dt.month
    df['dia_semana_num'] = df['data_inversa'].dt.weekday

    # Feature Binária
    df['fim_semana'] = df['dia_semana_num'].isin([5,6]).astype(int)

    # Transformações Cíclicas APENAS para mês
    df['mes_sin'] = np.sin(2 * np.pi * df['mes'] / 12)
    df['mes_cos'] = np.cos(2 * np.pi * df['mes'] / 12)

    # --------------------------
    # 2. Processamento de horário (SIMPLIFICADO)
    # --------------------------
    df['horario'] = pd.to_datetime(df['horario'], format="%H:%M:%S", errors='coerce')
    df['hora'] = df['horario'].dt.hour
    
    # Criar períodos do dia
    def periodo_do_dia(hora):
        if pd.isna(hora): return np.nan
        if 0 <= hora < 6: return "madrugada"
        elif 6 <= hora < 12: return "manha"
        elif 12 <= hora < 18: return "tarde"
        else: return "noite"
    df['periodo_dia'] = df['hora'].apply(periodo_do_dia)
    
    # Feature específica: Madrugada (período com mais acidentes fatais)
    df['eh_madrugada'] = (df['periodo_dia'] == 'madrugada').astype(int)

    # --------------------------
    # 3. Categorização de veículos (COM MAIS DETALHES)
    # --------------------------
    veiculos_leves = [
        'Automóvel', 'Utilitário', 'Caminhonete'
    ]
    
    veiculos_vulneraveis = [
        'Motocicleta', 'Ciclomotor', 'Motoneta', 'Bicicleta', 'Triciclo'
    ]

    veiculos_pesados = [
        'Caminhão', 'Caminhão-trator', 'Ônibus', 'Micro-ônibus', 
        'Trator de rodas', 'Trator misto'
    ]

    df['categoria_veiculo'] = np.where(
        df['tipo_veiculo'].isin(veiculos_leves), 'leve',
        np.where(df['tipo_veiculo'].isin(veiculos_vulneraveis), 'vulneravel',
        np.where(df['tipo_veiculo'].isin(veiculos_pesados), 'pesado', 'outros'))
    )
    
    # Feature crítica: veículos vulneráveis (motos têm alta letalidade)
    df['eh_veiculo_vulneravel'] = (df['categoria_veiculo'] == 'vulneravel').astype(int)

    # --------------------------
    # 4. Condições do acidente
    # --------------------------
    visibilidade_ruim = ['Chuva','Garoa/Chuvisco','Nevoeiro/Neblina','Vento','Nublado']
    df['visibilidade_ruim'] = df['condicao_metereologica'].isin(visibilidade_ruim).astype(int)

    # --------------------------
    # 5. FEATURES CRÍTICAS PARA ACIDENTES FATAIS
    # --------------------------
    
    # Tipos de acidente com alta letalidade
    acidentes_graves = [
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
    
    df['tipo_acidente_grave'] = df['tipo_acidente'].isin(acidentes_graves).astype(int)
    
    # Causas com alta letalidade
    causas_criticas = [
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
    
    df['causa_critica'] = df['causa_principal'].isin(causas_criticas).astype(int)
    
    # Traçados perigosos
    tracados_perigosos = [
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
    
    df['tracado_perigoso'] = df['tracado_via'].str.contains('|'.join(tracados_perigosos), case=False, na=False).astype(int)
    
    # Pista simples (mais acidentes fatais)
    df['pista_simples'] = df['tipo_pista'].str.contains('Simples', case=False, na=False).astype(int)

    # --------------------------
    # 6. INTERAÇÕES DE ALTO RISCO (FOCO EM FATALIDADE)
    # --------------------------
    
    # Interação 1: Tipo de acidente grave + veículo vulnerável (CRÍTICO para motos)
    df['acidente_grave_x_vulneravel'] = (
        (df['tipo_acidente_grave'] == 1) & 
        (df['eh_veiculo_vulneravel'] == 1)
    ).astype(int)
    
    # Interação 2: Colisão frontal (extremamente letal)
    df['eh_colisao_frontal'] = (
        df['tipo_acidente'].str.contains('frontal', case=False, na=False)
    ).astype(int)
    
    # Interação 3: Velocidade + visibilidade ruim
    df['velocidade_x_visibilidade'] = (
        df['causa_principal'].str.contains('velocidade', case=False, na=False) & 
        (df['visibilidade_ruim'] == 1)
    ).astype(int)
    
    # Interação 4: Álcool/Drogas (fator crítico)
    df['eh_alcool_drogas'] = (
        df['causa_principal'].str.contains('álcool|drogas', case=False, na=False)
    ).astype(int)
    
    # Interação 5: Madrugada + fim de semana (padrão de acidentes fatais)
    df['madrugada_fds'] = (
        (df['eh_madrugada'] == 1) & (df['fim_semana'] == 1)
    ).astype(int)
    
    # Interação 6: Visibilidade ruim + período noturno
    df['visibilidade_ruim_periodo_dia'] = (
        (df['visibilidade_ruim'] == 1) & 
        (df['periodo_dia'].isin(['noite', 'madrugada']))
    ).astype(int)
    
    # Interação 7: Veículo pesado + pista simples
    df['veiculo_pesado_pista_simples'] = (
        (df['categoria_veiculo'] == 'pesado') & 
        (df['pista_simples'] == 1)
    ).astype(int)
    
    # Interação 8: Traçado perigoso + visibilidade ruim
    df['tracado_perigoso_x_visibilidade'] = (
        (df['tracado_perigoso'] == 1) & (df['visibilidade_ruim'] == 1)
    ).astype(int)
    
    # Interação 9: Tipo acidente + causa (específica)
    df['tipo_acidente_x_causa'] = (
        df['tipo_acidente'].astype(str) + '_' + df['causa_principal'].astype(str)
    )
    
    # Interação 10: Categoria veículo + tipo acidente
    df['categoria_veiculo_x_tipo_acidente'] = (
        df['categoria_veiculo'].astype(str) + '_' + df['tipo_acidente'].astype(str)
    )

    # --------------------------
    # 7. SCORE DE RISCO DE FATALIDADE (agregado)
    # --------------------------
    df['score_risco_fatal'] = (
        df['tipo_acidente_grave'] * 3 +           # Peso 3
        df['causa_critica'] * 2 +                 # Peso 2
        df['eh_veiculo_vulneravel'] * 2 +         # Peso 2
        df['eh_colisao_frontal'] * 3 +            # Peso 3
        df['eh_alcool_drogas'] * 2 +              # Peso 2
        df['visibilidade_ruim'] +                 # Peso 1
        df['madrugada_fds'] +                     # Peso 1
        df['tracado_perigoso'] +                  # Peso 1
        df['pista_simples']                       # Peso 1
    )

    # --------------------------
    # 8. Drop de colunas originais usadas
    # --------------------------
    drop_cols = [
        'data_inversa', 'horario',
        'ano_fabricacao_veiculo', 'tipo_veiculo',
        'idade', 'condicao_metereologica', 'id', 'pesid',
        'id_veiculo', 'marca', 'km', 'latitude', 'longitude',
        'municipio','delegacia', 'uop', 'regional',
        'estado_fisico', 'ilesos', 'feridos_leves',
        'hora', 'feridos_graves', 'mortos', 'sexo', 'mes',
        'dia', 'dia_semana_num', 'dia_semana', 'causa_acidente',
        'ordem_tipo_acidente', 'tipo_envolvido', 'ano'
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    return df

# ==============================================================================
# 1. BASE DE DADOS AUXILIAR (DE-PARA)
# Esta é a base que mapeia (UF, BR) para suas características.
# Em um projeto real, isso viria de uma fonte de dados mais robusta.
# Para o nosso app, podemos começar com este exemplo.
# ==============================================================================
@st.cache_data # Decorator mágico do Streamlit!
def carregar_dados_brs():
    """
    Lê o arquivo CSV com as características das BRs e o mantém em cache.
    """
    df = pd.read_csv('caracteristicas_brs.csv')
    return df

# ==============================================================================
# 2. FUNÇÃO DE PRÉ-PROCESSAMENTO PARA A PREVISAO DA ROTA
# ==============================================================================
def pre_processamento_df_manual(
    data_hora_viagem: datetime, 
    tipo_veiculo_usuario: str, 
    condicao_visibilidade_usuario: str, 
    rota: list, 
    db_brs: pd.DataFrame
) -> pd.DataFrame:
    '''
    Prepara os dados de input do Streamlit para o modelo de previsão.

    Args:
        data_hora_viagem (datetime): Data e hora da viagem informada pelo usuário.
        tipo_veiculo_usuario (str): Categoria do veículo (ex: 'Motocicleta').
        condicao_visibilidade_usuario (str): Condição de visibilidade (ex: 'Chuva ou Neblina').
        rota (list): Lista de dicionários, onde cada um representa um trecho. Ex: [{'uf': 'SP', 'br': 116}].
        db_brs (pd.DataFrame): DataFrame com as características das BRs.

    Returns:
        pd.DataFrame: DataFrame processado e pronto para a previsão.
    '''
    
    # --- ETAPA A: DEFINIR CENÁRIOS DE RISCO ---
    # Como não temos o tipo/causa do acidente, vamos simular cenários críticos
    # para avaliar a robustez da rota a diferentes tipos de perigo.
    cenarios_de_risco = [
        {'tipo_acidente': 'Colisão frontal', 'causa_principal': 'Velocidade Incompatível'},
        {'tipo_acidente': 'Saída de leito carroçável', 'causa_principal': 'Condutor Dormindo'},
        {'tipo_acidente': 'Atropelamento de Pedestre', 'causa_principal': 'Falta de Atenção do Pedestre'},
        {'tipo_acidente': 'Colisão traseira', 'causa_principal': 'Falta de Atenção à Condução'}
    ]

    # --- ETAPA B: CONSTRUIR O DATAFRAME A PARTIR DOS INPUTS DO USUÁRIO ---
    dados_para_prever = []

    for trecho in rota:
        # Busca as características da BR no nosso "banco de dados"
        info_br = db_brs[(db_brs['uf'] == trecho['uf']) & (db_brs['br'] == trecho['br'])]

        if not info_br.empty:
            tipo_pista = info_br.iloc[0]['tipo_pista']
            tracado_via = info_br.iloc[0]['tracado_via']
        else:
            # Valores padrão caso a BR não seja encontrada
            tipo_pista = 'Simples'
            tracado_via = 'Reta'
            
        # Para cada trecho da rota, criamos uma linha para cada cenário de risco
        for cenario in cenarios_de_risco:
            linha = {
                # Inputs diretos da viagem
                'data_inversa': data_hora_viagem.strftime('%Y-%m-%d'),
                'horario': data_hora_viagem.strftime('%H:%M:%S'),
                'tipo_veiculo': tipo_veiculo_usuario,
                'condicao_metereologica': condicao_visibilidade_usuario,
                # Inputs baseados na rota
                'uf': trecho['uf'],
                'br': trecho['br'],
                'tipo_pista': tipo_pista,
                'tracado_via': tracado_via,
                # Inputs do cenário de risco simulado
                'tipo_acidente': cenario['tipo_acidente'],
                'causa_principal': cenario['causa_principal']
            }
            dados_para_prever.append(linha)

    if not dados_para_prever:
        return pd.DataFrame() # Retorna DF vazio se a rota estiver vazia

    df = pd.DataFrame(dados_para_prever)
    
    # =================================================================================
    # --- ETAPA C: APLICAR A SUA LÓGICA DE ENGENHARIA DE FEATURES ORIGINAL ---
    # O código abaixo é 99% idêntico ao que você enviou. Apenas adaptamos
    # os nomes das variáveis para o contexto do Streamlit.
    # =================================================================================

    # 1. Processamento de datas
    df['data_inversa'] = pd.to_datetime(df['data_inversa'], format="%Y-%m-%d")
    df['mes'] = df['data_inversa'].dt.month
    df['dia_semana_num'] = df['data_inversa'].dt.weekday
    df['fim_semana'] = df['dia_semana_num'].isin([5, 6]).astype(int)
    df['mes_sin'] = np.sin(2 * np.pi * df['mes'] / 12)
    df['mes_cos'] = np.cos(2 * np.pi * df['mes'] / 12)

    # 2. Processamento de horário
    df['horario'] = pd.to_datetime(df['horario'], format="%H:%M:%S")
    df['hora'] = df['horario'].dt.hour
    
    def periodo_do_dia(hora):
        if 0 <= hora < 6: return "madrugada"
        elif 6 <= hora < 12: return "manha"
        elif 12 <= hora < 18: return "tarde"
        else: return "noite"
    df['periodo_dia'] = df['hora'].apply(periodo_do_dia)
    df['eh_madrugada'] = (df['periodo_dia'] == 'madrugada').astype(int)

    # 3. Categorização de veículos
    veiculos_vulneraveis = ['Motocicleta', 'Ciclomotor', 'Motoneta', 'Bicicleta']
    veiculos_pesados = ['Caminhão', 'Caminhão-trator', 'Ônibus', 'Micro-ônibus']
    
    df['categoria_veiculo'] = np.where(
        df['tipo_veiculo'].isin(veiculos_vulneraveis), 'vulneravel',
        np.where(df['tipo_veiculo'].isin(veiculos_pesados), 'pesado', 'leve') # 'leve' é o default
    )
    df['eh_veiculo_vulneravel'] = (df['categoria_veiculo'] == 'vulneravel').astype(int)

    # 4. Condições do acidente
    visibilidade_ruim_map = ['Chuva ou Neblina', 'Poeira ou Fumaça']
    df['visibilidade_ruim'] = df['condicao_metereologica'].isin(visibilidade_ruim_map).astype(int)

    # 5. Features críticas (baseadas nos cenários)
    acidentes_graves = ["Atropelamento de Pedestre", "Capotamento", "Colisão frontal", "Saída de leito carroçável"]
    df['tipo_acidente_grave'] = df['tipo_acidente'].isin(acidentes_graves).astype(int)
    
    causas_criticas = ["Velocidade Incompatível", "Condutor Dormindo", "Ingestão de álcool pelo condutor", "Transitar na contramão"]
    df['causa_critica'] = df['causa_principal'].isin(causas_criticas).astype(int)
    
    tracados_perigosos = ["Curva", "Curva;Aclive", "Curva;Declive", "Declive"]
    df['tracado_perigoso'] = df['tracado_via'].isin(tracados_perigosos).astype(int)
    
    df['pista_simples'] = df['tipo_pista'].str.contains('Simples', case=False, na=False).astype(int)

    # 6. Interações de alto risco
    df['acidente_grave_x_vulneravel'] = ((df['tipo_acidente_grave'] == 1) & (df['eh_veiculo_vulneravel'] == 1)).astype(int)
    df['eh_colisao_frontal'] = (df['tipo_acidente'].str.contains('frontal', case=False, na=False)).astype(int)
    df['velocidade_x_visibilidade'] = (df['causa_principal'].str.contains('velocidade', case=False, na=False) & (df['visibilidade_ruim'] == 1)).astype(int)
    df['eh_alcool_drogas'] = (df['causa_principal'].str.contains('álcool|drogas', case=False, na=False)).astype(int)
    df['madrugada_fds'] = ((df['eh_madrugada'] == 1) & (df['fim_semana'] == 1)).astype(int)
    df['visibilidade_ruim_periodo_dia'] = ((df['visibilidade_ruim'] == 1) & (df['periodo_dia'].isin(['noite', 'madrugada']))).astype(int)
    df['veiculo_pesado_pista_simples'] = ((df['categoria_veiculo'] == 'pesado') & (df['pista_simples'] == 1)).astype(int)
    df['tracado_perigoso_x_visibilidade'] = ((df['tracado_perigoso'] == 1) & (df['visibilidade_ruim'] == 1)).astype(int)
    df['tipo_acidente_x_causa'] = df['tipo_acidente'].astype(str) + '_' + df['causa_principal'].astype(str)
    df['categoria_veiculo_x_tipo_acidente'] = df['categoria_veiculo'].astype(str) + '_' + df['tipo_acidente'].astype(str)

    # 7. Score de Risco
    df['score_risco_fatal'] = (
        df['tipo_acidente_grave'] * 3 +
        df['causa_critica'] * 2 +
        df['eh_veiculo_vulneravel'] * 2 +
        df['eh_colisao_frontal'] * 3 +
        df['eh_alcool_drogas'] * 2 +
        df['visibilidade_ruim'] +
        df['madrugada_fds'] +
        df['tracado_perigoso'] +
        df['pista_simples']
    )

    # 8. Drop de colunas originais
    drop_cols = [
        'data_inversa', 'horario', 'tipo_veiculo', 'condicao_metereologica',
        'hora', 'mes', 'dia_semana_num'
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')

    return df

# --- Carregamento dos Modelos e Objetos ---
@st.cache_resource
def load_objects():
    """Carrega o modelo e os dicionários de encoders do disco."""
    try:
        model = joblib.load('modelo_gravidade_xgb.pkl')
        feature_encoders = joblib.load('label_encoders_xgb.pkl')
        target_encoder = joblib.load('target_encoder_xgb.pkl')
        selected_features = joblib.load('colunas_modelo_xgb.pkl')
        return model, feature_encoders, target_encoder, selected_features
    except FileNotFoundError:
        return None, None, None

# --- Lógica de Previsão Unificada (Corrigida) ---
def fazer_previsao_df_completo(df_input, model, feature_encs, selected_features):
    """
    Função unificada para pré-processar, codificar, selecionar features e prever.
    
    Args:
        df_input (pd.DataFrame): Dados brutos de entrada.
        model: O modelo treinado (LGBMClassifier).
        feature_encs (dict): Dicionário de LabelEncoders salvos.
        selected_features (list): Lista de features que o modelo final espera.
        
    Returns:
        np.array: Probabilidades de cada classe.
    """
    print("Iniciando pré-processamento para inferência...")
    
    # 1. Pré-processamento e Engenharia de Features
    # (A função 'pre_processamento_df_completo' deve incluir todas as transformações cíclicas)
    df_processed = pre_processamento_df_completo(df_input.copy())
    
    # 2. Codificação (Label Encoding com tratamento de desconhecidos)
    df_encoded = df_processed.copy()
    
    for col, encoder in feature_encs.items():
        if col in df_encoded.columns:
            # Substituir valores desconhecidos por um valor de placeholder (-1) 
            # antes de usar o encoder, se a coluna for categórica (object)
            
            # Criar um set de classes conhecidas para acesso rápido
            known_classes = set(encoder.classes_)
            
            # Função para mapear: se for conhecido, usa transform, senão, usa um placeholder
            def safe_transform(value):
                # O encoder só pode transformar valores que ele viu no treino.
                if value in known_classes:
                    return encoder.transform([value])[0]
                else:
                    # Usar um valor que o modelo já viu (ex: o valor mais frequente)
                    # ou uma flag. Aqui, usamos 99999 como um placeholder grande e único.
                    # É crucial que este valor seja tratado corretamente no treino.
                    return -1 
            
            # Aplica a transformação segura
            df_encoded[col] = df_encoded[col].astype(str).apply(safe_transform)


    # 3. Preparação Final do DataFrame para o Modelo
    # Remove a coluna alvo, se ela existir nos dados de entrada
    X_to_predict = df_encoded.drop('classificacao_acidente', axis=1, errors='ignore')
    
    # Selecionar apenas as features que o modelo espera, na ordem correta
    try:
        X_final = X_to_predict[selected_features]
    except KeyError as e:
        print(f"Erro: Feature '{e}' não encontrada após o pré-processamento. Verifique se o seu 'preprocessamento_df_completo' está correto.")
        return None

    # 4. Fazer a Previsão
    probabilities = model.predict_proba(X_final)
    print("Previsões concluídas.")
    
    return probabilities

# --- Função de Previsão para Entrada Manual (Robusta) ---
def fazer_previsao_df_manual(df_processado: pd.DataFrame, model, feature_encs: dict, selected_features: list) -> np.array:
    """
    Realiza a previsão a partir de um DataFrame já pré-processado.
    Responsável por:
    1. Codificar features categóricas usando encoders pré-treinados.
    2. Alinhar as colunas do DataFrame com as esperadas pelo modelo.
    3. Retornar as probabilidades da previsão.

    Args:
        df_processado (pd.DataFrame): DataFrame após a etapa de engenharia de features.
        model: Modelo treinado (XGBoost/LightGBM).
        feature_encs (dict): Dicionário de LabelEncoders salvos do treinamento.
        selected_features (list): Lista exata de features que o modelo espera, na ordem correta.

    Returns:
        np.array: Array com as probabilidades de cada classe, ou None em caso de erro.
    """
    warnings.filterwarnings('ignore')
    
    try:
        # Etapa 1: Codificação com tratamento de valores desconhecidos
        df_encoded = df_processado.copy()
        
        for col in df_encoded.columns:
            if col in feature_encs:  # Apenas codifica colunas que têm um encoder
                encoder = feature_encs[col]
                known_classes = set(encoder.classes_)
                
                # Aplica a transformação, tratando valores que não estavam no treino
                df_encoded[col] = df_encoded[col].apply(
                    lambda x: encoder.transform([str(x)])[0] if str(x) in known_classes else -1
                )
        
        # Etapa 2: Garantir que todas as features do modelo existam no DataFrame
        # Adiciona colunas faltantes com valor 0 (ou outro valor padrão)
        for feat in selected_features:
            if feat not in df_encoded.columns:
                df_encoded[feat] = 0 # Valor padrão para features ausentes
        
        # Etapa 3: Selecionar e ordenar as features exatamente como o modelo espera
        X_final = df_encoded[selected_features]
        
        # Etapa 4: Fazer a previsão
        probabilities = model.predict_proba(X_final)
        
        return probabilities
        
    except Exception as e:
        st.error(f"Ocorreu um erro durante a previsão: {e}")
        traceback.print_exc()
        return None

def rederizar_pagina_upload_arquivo():
    st.sidebar.header("Parâmetros de Entrada")
    uploaded_file = st.sidebar.file_uploader(
        "Selecione o arquivo CSV", 
        type="csv",
        help="Faça upload de um arquivo CSV com dados de acidentes"
    )
    
    if uploaded_file:
        try:
            # Tenta ler como UTF-8
            file_content = uploaded_file.getvalue()
            
            try:
                # Tenta decodificar com UTF-8 (o padrão esperado)
                data_string = file_content.decode('utf-8')
                st.success("Arquivo lido como UTF-8")
                data_io = io.StringIO(data_string)
                df_input = pd.read_csv(data_io, sep=',') 
            except UnicodeDecodeError:
                # Se falhar, tenta LATIN-1 (comum em arquivos brasileiros)
                data_string = file_content.decode('latin-1')
                st.warning("Arquivo lido usando codificação LATIN-1")
                data_io = io.StringIO(data_string)
                df_input = pd.read_csv(data_io, sep=';')

            # ==========================================
            # SEÇÃO 1: INFORMAÇÕES DO DATASET
            # ==========================================
            st.header("Análise do Dataset Carregado")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total de Registros", f"{len(df_input):,}")
            with col2:
                st.metric("Total de Colunas", len(df_input.columns))
            with col3:
                memoria_mb = df_input.memory_usage(deep=True).sum() / 1024**2
                st.metric("Tamanho em Memória", f"{memoria_mb:.1f} MB")
            with col4:
                missing_pct = (df_input.isnull().sum().sum() / (len(df_input) * len(df_input.columns))) * 100
                st.metric("% Dados Faltantes", f"{missing_pct:.1f}%")
            
            # Amostra dos dados
            with st.expander("Visualizar Amostra dos Dados (Primeiras 10 linhas)", expanded=False):
                st.dataframe(df_input.head(10), use_container_width=True)
            
            # Aviso de processamento pesado
            if len(df_input) > 100000:
                st.warning(
                    f"O arquivo contém {len(df_input):,} linhas. "
                    "O processamento pode demorar alguns minutos. "
                    "Considere processar uma amostra menor para testes rápidos."
                )
                
                # Opção para processar amostra
                usar_amostra = st.checkbox(
                    "Processar apenas uma amostra aleatória de 10.000 registros",
                    value=False
                )
                if usar_amostra:
                    df_input = df_input.sample(n=10000, random_state=42)
                    st.info(f"Processando amostra de {len(df_input):,} registros")

            # ==========================================
            # SEÇÃO 2: PROCESSAMENTO E PREVISÕES
            # ==========================================
            st.header("Previsões de Gravidade")
            
            with st.spinner('Aplicando pré-processamento e fazendo previsões...'):
                import time
                start_time = time.time()
                
                probabilities = fazer_previsao_df_completo(
                    df_input,
                    modelo,
                    encoder_features, 
                    colunas_modelo
                )
                
                tempo_processamento = time.time() - start_time
                
                # Criar DataFrame com resultados
                df_results = pd.DataFrame(probabilities, columns=nomes_das_classes)

                # Garantir que todas as colunas são numéricas
                for col in nomes_das_classes:
                    df_results[col] = pd.to_numeric(df_results[col], errors='coerce')

                # Adicionar classe predita
                df_results['Classe_Predita'] = df_results[nomes_das_classes].idxmax(axis=1)
                df_results['Confianca'] = df_results[nomes_das_classes].max(axis=1)

                # Adicionar ao dataframe original
                df_input_com_pred = df_input.copy()
                df_input_com_pred['Gravidade_Predita'] = df_results['Classe_Predita'].values
                df_input_com_pred['Confianca_Predicao'] = df_results['Confianca'].values
            
            st.success(f"Previsão concluída em {tempo_processamento:.2f} segundos!")
            
            # ==========================================
            # SEÇÃO 3: DISTRIBUIÇÃO DAS PREVISÕES
            # ==========================================
            st.subheader("Distribuição das Previsões")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Contagem de classes preditas
                contagem_classes = df_results['Classe_Predita'].value_counts()
                
                fig_pie = go.Figure(data=[go.Pie(
                    labels=contagem_classes.index,
                    values=contagem_classes.values,
                    hole=0.4,
                    marker=dict(colors=['#2ca02c', "#ffdb0e", '#d62728']),
                    textinfo='label+percent',
                    textfont_size=12
                )])
                
                fig_pie.update_layout(
                    title='Distribuição de Gravidade dos Acidentes',
                    height=400,
                    showlegend=True
                )
                
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                # Gráfico de barras
                fig_bar = go.Figure(data=[go.Bar(
                    x=contagem_classes.index,
                    y=contagem_classes.values,
                    text=contagem_classes.values,
                    textposition='auto',
                    marker=dict(
                        color=['#2ca02c', "#ffdb0e", '#d62728'],
                        line=dict(color='black', width=1.5)
                    )
                )])
                
                fig_bar.update_layout(
                    title='Quantidade por Gravidade',
                    xaxis_title='Gravidade',
                    yaxis_title='Quantidade de Acidentes',
                    height=400
                )
                
                st.plotly_chart(fig_bar, use_container_width=True)
            
            # Métricas de distribuição
            st.markdown("### Estatísticas de Distribuição")
            col1, col2, col3 = st.columns(3)
            
            total = len(df_results)
            for i, (col, classe) in enumerate(zip([col1, col2, col3], nomes_das_classes)):
                count = contagem_classes.get(classe, 0)
                pct = (count / total) * 100
                with col:
                    st.metric(
                        classe,
                        f"{count:,} ({pct:.1f}%)",
                        delta=None
                    )
            
            # ==========================================
            # SEÇÃO 4: ANÁLISE DE CONFIANÇA
            # ==========================================
            st.subheader("Análise de Confiança das Previsões")
            
            # Histograma de confiança
            fig_hist = go.Figure()
            
            for classe in nomes_das_classes:
                mask = df_results['Classe_Predita'] == classe
                confidences = df_results[mask]['Confianca']
                
                fig_hist.add_trace(go.Histogram(
                    x=confidences * 100,
                    name=classe,
                    opacity=0.7,
                    nbinsx=30
                ))
            
            fig_hist.update_layout(
                title='Distribuição de Confiança por Classe',
                xaxis_title='Confiança (%)',
                yaxis_title='Frequência',
                barmode='overlay',
                height=400
            )
            
            st.plotly_chart(fig_hist, use_container_width=True)
            
            # Estatísticas de confiança
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Confiança Média", f"{df_results['Confianca'].mean():.1%}")
            with col2:
                st.metric("Confiança Mediana", f"{df_results['Confianca'].median():.1%}")
            with col3:
                baixa_confianca = (df_results['Confianca'] < 0.5).sum()
                st.metric("Previsões Baixa Confiança (<50%)", f"{baixa_confianca:,}")
            
            # ==========================================
            # SEÇÃO 5: CASOS CRÍTICOS (FATAIS)
            # ==========================================
            casos_fatais = df_results[df_results['Classe_Predita'] == 'Com Vítimas Fatais']
            
            if len(casos_fatais) > 0:
                st.subheader(f"Análise de Casos Fatais ({len(casos_fatais):,} casos)")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Confiança dos casos fatais
                    confianca_media_fatal = casos_fatais['Confianca'].mean()
                    confianca_alta = (casos_fatais['Confianca'] >= 0.7).sum()
                    
                    st.metric(
                        "Confiança Média (Fatais)",
                        f"{confianca_media_fatal:.1%}"
                    )
                    st.metric(
                        "Casos com Alta Confiança (≥70%)",
                        f"{confianca_alta:,}"
                    )
                
                with col2:
                    # Distribuição de probabilidades para casos fatais
                    fig_box = go.Figure()
                    
                    for classe in nomes_das_classes:
                        fig_box.add_trace(go.Box(
                            y=casos_fatais[classe] * 100,
                            name=classe,
                            boxmean='sd'
                        ))
                    
                    fig_box.update_layout(
                        title='Distribuição de Probabilidades (Casos Fatais)',
                        yaxis_title='Probabilidade (%)',
                        height=300
                    )
                    
                    st.plotly_chart(fig_box, use_container_width=True)
            
            # ==========================================
            # SEÇÃO 6: TABELA DE PROBABILIDADES
            # ==========================================
            st.subheader("Tabela de Probabilidades Detalhada")
            
            # Opções de filtro
            col1, col2 = st.columns(2)
            with col1:
                filtro_classe = st.multiselect(
                    "Filtrar por Classe Predita",
                    options=nomes_das_classes,
                    default=nomes_das_classes
                )
            with col2:
                min_confianca = st.slider(
                    "Confiança Mínima (%)",
                    min_value=0,
                    max_value=100,
                    value=0,
                    step=5
                ) / 100
            
            # Aplicar filtros
            mask = (
                df_results['Classe_Predita'].isin(filtro_classe) &
                (df_results['Confianca'] >= min_confianca)
            )
            df_filtrado = df_results[mask]
            
            st.info(f"Mostrando {len(df_filtrado):,} de {len(df_results):,} registros")
            
            # Mostrar tabela estilizada
            st.dataframe(
                df_filtrado.style.format({
                    col: "{:.2%}" for col in nomes_das_classes + ['Confianca']
                }).background_gradient(
                    cmap='RdYlGn_r',
                    subset=nomes_das_classes
                ).highlight_max(
                    axis=1,
                    subset=nomes_das_classes,
                    props='font-weight: bold; border: 2px solid green;'
                ),
                use_container_width=True,
                height=400
            )
            
            # ==========================================
            # SEÇÃO 7: DOWNLOAD DOS RESULTADOS
            # ==========================================
            st.subheader("Exportar Resultados")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Download apenas probabilidades
                csv_probs = df_results.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Probabilidades (CSV)",
                    data=csv_probs,
                    file_name="probabilidades_acidentes.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col2:
                # Download dataset completo com previsões
                csv_completo = df_input_com_pred.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Dataset Completo + Previsões (CSV)",
                    data=csv_completo,
                    file_name="dataset_com_previsoes.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            # ==========================================
            # SEÇÃO 8: ANÁLISE ADICIONAL (SE HOUVER COLUNAS RELEVANTES)
            # ==========================================
            if 'uf' in df_input.columns:
                st.subheader("Análise Geográfica")
                
                # Criar tabela de análise por UF
                df_geo = df_input_com_pred.groupby('uf').agg({
                    'Gravidade_Predita': lambda x: x.value_counts().to_dict(),
                    'Confianca_Predicao': 'mean'
                }).reset_index()
                
                # Expandir dicionário de contagem
                gravidade_por_uf = []
                for _, row in df_geo.iterrows():
                    uf = row['uf']
                    counts = row['Gravidade_Predita']
                    for gravidade, count in counts.items():
                        gravidade_por_uf.append({
                            'UF': uf,
                            'Gravidade': gravidade,
                            'Quantidade': count
                        })
                
                df_geo_expanded = pd.DataFrame(gravidade_por_uf)
                
                # Gráfico de barras empilhadas por UF
                fig_geo = go.Figure()
                
                for gravidade in nomes_das_classes:
                    df_temp = df_geo_expanded[df_geo_expanded['Gravidade'] == gravidade]
                    fig_geo.add_trace(go.Bar(
                        name=gravidade,
                        x=df_temp['UF'],
                        y=df_temp['Quantidade'],
                        text=df_temp['Quantidade'],
                        textposition='auto'
                    ))
                
                fig_geo.update_layout(
                    title='Distribuição de Gravidade por Estado',
                    xaxis_title='Estado (UF)',
                    yaxis_title='Quantidade de Acidentes',
                    barmode='stack',
                    height=500
                )
                
                st.plotly_chart(fig_geo, use_container_width=True)
            
            st.balloons()

        except Exception as e:
            st.error(f"Ocorreu um erro durante o processamento: {e}")
            st.error(
                "Verifique se o arquivo CSV tem o formato esperado e "
                "todas as colunas necessárias."
            )
            
            with st.expander("Ver Detalhes do Erro"):
                import traceback
                st.code(traceback.format_exc())
    
    else:
        st.markdown("---")
        st.info("Aguardando o upload de um arquivo CSV...")
        
        # Instruções quando não há arquivo
        st.markdown("""
        ### Como usar:
        
        1. Clique no botão **"Browse files"** na barra lateral
        2. Selecione um arquivo CSV com dados de acidentes
        3. O arquivo será automaticamente processado e analisado
        4. Visualize as previsões, gráficos e estatísticas gerados
        5. Faça download dos resultados quando necessário
        
        ### Formato esperado do arquivo:
        
        O arquivo CSV deve conter as seguintes colunas principais:
        - `data_inversa`: Data do acidente (formato YYYY-MM-DD)
        - `horario`: Horário do acidente (formato HH:MM:SS)
        - `uf`: Estado onde ocorreu o acidente
        - `tipo_veiculo`: Tipo de veículo envolvido
        - `tipo_acidente`: Tipo de acidente
        - `causa_principal`: Causa principal do acidente
        - E outras colunas relevantes...
        
        ### Dicas:
        - Para arquivos grandes (>100k linhas), considere usar a opção de amostragem
        - Os resultados podem ser baixados em formato CSV
        - Use os filtros para explorar casos específicos
        """)

def renderizar_pagina_entrada_manual():
    st.set_page_config(
    page_title="Modelos de Predição - Análise de Acidentes",
    layout="wide",
)   
    tab1, tab2 = st.tabs(["**Fazer Previsão**", "**Histórico de Previsão de Rotas**"])
    with tab1:
        st.header("Modo: Previsão de Risco por Rota")
        st.info("Planeje sua viagem! Adicione os trechos de BRs que você irá percorrer e veja uma análise de risco baseada nas condições da sua viagem.")

        # Inicializa a rota na sessão se não existir
        if 'rota' not in st.session_state:
            st.session_state.rota = []
        if 'historico_rotas' not in st.session_state: # <--- NOVO: Inicializa o histórico
            st.session_state.historico_rotas = []

        # --- NOVA FUNÇÃO DE CALLBACK ---
        def carregar_rota_selecionada():
            """Callback para carregar a rota pré-definida no session_state."""
            nome_da_rota = st.session_state.rotas_selectbox # Acessa o valor do selectbox pela sua chave
            if nome_da_rota in ROTAS_PREDEFINIDAS:
                st.session_state.rota = ROTAS_PREDEFINIDAS[nome_da_rota]

        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("🗓️ Parâmetros da Viagem")
            data_hora_viagem = st.date_input(
                "Data e Hora de Início da Viagem",
                datetime.datetime.now()
            )
            tipo_veiculo_usuario = st.selectbox(
                "Seu Tipo de Veículo",
                ['Automóvel', 'Motocicleta', 'Caminhão', 'Ônibus', 'Utilitário']
            )
            condicao_visibilidade_usuario = st.selectbox(
                "Condições de Visibilidade",
                ['Tempo Bom (Céu Claro/Nublado)', 'Chuva ou Neblina', 'Poeira ou Fumaça']
            )

        with col2:
            st.subheader("Monte sua Rota")

            # --- NOVO SELECTBOX PARA ROTAS PRÉ-DEFINIDAS ---
            st.selectbox(
                label="Ou escolha uma rota sugerida (isso substituirá a rota atual):",
                options=list(ROTAS_PREDEFINIDAS.keys()),
                key="rotas_selectbox",  # Chave para acessar o valor no session_state
                on_change=carregar_rota_selecionada # A mágica acontece aqui!
            )

            st.markdown("---")

            with st.form("form_add_trecho"):
                ufs = sorted(db_brs['uf'].unique()) if db_brs is not None else ['SP']
                uf_selecionada = st.selectbox("UF do Trecho", ufs)
                br_selecionada = st.number_input("Número da BR", min_value=1, max_value=499, step=1, value=116)
                submitted = st.form_submit_button("Adicionar Trecho")

                if submitted:
                    st.session_state.rota.append({'uf': uf_selecionada, 'br': int(br_selecionada)})
                    st.success(f"Trecho BR-{br_selecionada}/{uf_selecionada} adicionado!")

        if st.session_state.rota:
            st.write("**Rota Atual:**")
            df_rota = pd.DataFrame(st.session_state.rota)
            st.dataframe(df_rota, use_container_width=True)

            if st.button("Limpar Rota"):
                st.session_state.rota = []
                st.rerun()

        if st.button("**Analisar Rota Completa**", type="primary", use_container_width=True):
            if not st.session_state.rota:
                st.warning("Por favor, adicione pelo menos um trecho à rota.")

            elif modelo is None or db_brs is None:
                st.error("Aplicação não inicializada corretamente. Verifique os arquivos de modelo e dados.")

            else:
                with st.spinner('🔄 Analisando rota e calculando riscos...'):
                    # ETAPA 1: Pré-processamento
                    df_processado = pre_processamento_df_manual(
                        data_hora_viagem=data_hora_viagem,
                        tipo_veiculo_usuario=tipo_veiculo_usuario,
                        condicao_visibilidade_usuario=condicao_visibilidade_usuario,
                        rota=st.session_state.rota,
                        db_brs=db_brs
                    )

                    # ETAPA 2: Previsão
                    probabilidades = fazer_previsao_df_manual(
                        df_processado=df_processado,
                        model=modelo,
                        feature_encs=encoder_features,
                        selected_features=colunas_modelo
                    )

                    if probabilidades is not None:
                        st.success("✅ Análise de risco concluída!")
                        # Calcula a média das probabilidades em todos os trechos e cenários
                        risco_agregado = np.mean(probabilidades, axis=0)   

                        # --- NOVO: SALVAR RESULTADO NO HISTÓRICO ---
                        classe_predita = nomes_das_classes[risco_agregado.argmax()]
                        prob_predita = risco_agregado.max()
                        
                        registro_historico = {
                            "timestamp": datetime.datetime.now(),
                            "rota": pd.DataFrame(st.session_state.rota),
                            "parametros": {
                                "Data": data_hora_viagem.strftime('%d/%m/%Y %H:%M'),
                                "Veículo": tipo_veiculo_usuario,
                                "Visibilidade": condicao_visibilidade_usuario
                            },
                            "risco_agregado": risco_agregado,
                            "classe_predita": classe_predita,
                            "prob_predita": prob_predita
                        }
                        st.session_state.historico_rotas.insert(0, registro_historico) # Insere no início da lista

                        # ETAPA 3: Agregar e Exibir Resultados
                        st.subheader("📊 Risco Agregado para a Rota")
                        st.write("O risco geral da rota é a média das probabilidades em todos os trechos e cenários de risco simulados.")

                        df_results = pd.DataFrame([risco_agregado], columns=nomes_das_classes)

                        st.dataframe(
                            df_results.style.format("{:.2%}").background_gradient(cmap='YlOrRd', axis=1),
                            use_container_width=True
                        )

                        # Gráfico de barras (adaptado da sua visualização)
                        fig = go.Figure(data=[go.Bar(
                            x=nomes_das_classes, y=risco_agregado * 100,
                            text=[f'{p:.1f}%' for p in risco_agregado * 100], textposition='auto',
                            marker_color=['#d62728', "#ffdb0e", '#2ca02c'] # Verde, Laranja, Vermelho
                        )])

                        fig.update_layout(title='Distribuição de Probabilidade de Risco na Rota', yaxis_title='Probabilidade (%)', yaxis_range=[0,100])
                        st.plotly_chart(fig, use_container_width=True)

                        # Alerta de Risco
                        classe_predita = nomes_das_classes[risco_agregado.argmax()]
                        prob_predita = risco_agregado.max()

                        if classe_predita == 'Com Vítimas Fatais':
                            st.error(f"🚨 **RISCO ALTO**: A maior probabilidade na rota é de acidentes **{classe_predita}** ({prob_predita:.1%}). Redobre a atenção!")
                        
                        elif classe_predita == 'Com Vítimas Feridas':
                            st.warning(f"⚠️ **RISCO MODERADO**: A maior probabilidade na rota é de acidentes **{classe_predita}** ({prob_predita:.1%}). Dirija com cuidado.")
                        
                        else:
                            st.success(f"✅ **RISCO BAIXO**: A maior probabilidade na rota é de acidentes **{classe_predita}** ({prob_predita:.1%}). Boa viagem!")

                        with st.expander("Ver detalhes da análise por cenário"):
                            st.write("A tabela abaixo mostra as probabilidades para cada trecho da rota e cada cenário de risco simulado.")
                            df_detalhes = pd.DataFrame(probabilidades, columns=nomes_das_classes)
                            st.dataframe(df_detalhes.style.format("{:.2%}").background_gradient(cmap='YlOrRd', axis=1), use_container_width=True)

    # ==========================================================================
    # ABA 2: HISTÓRICO
    # ==========================================================================
    with tab2:
        st.header("📜 Histórico de Previsões da Sessão")

        if not st.session_state.historico_rotas:
            st.info("Nenhuma previsão de rota foi realizada nesta sessão ainda. Faça uma análise na aba ao lado.")
        else:
            if st.button("🗑️ Limpar Histórico"):
                st.session_state.historico_rotas = []
                st.rerun()

            for i, registro in enumerate(st.session_state.historico_rotas):
                timestamp_str = registro['timestamp'].strftime('%d/%m/%Y às %H:%M:%S')
                cor_risco = "red" if registro['classe_predita'] == 'Com Vítimas Fatais' else "orange" if registro['classe_predita'] == 'Com Vítimas Feridas' else "green"

                expander_title = f"**Previsão #{len(st.session_state.historico_rotas) - i}** ({timestamp_str}) - Risco Dominante: **:{cor_risco}[{registro['classe_predita']}]**"
                
                with st.expander(expander_title):
                    st.write("**Parâmetros da Viagem:**")
                    st.json(registro['parametros'])

                    st.write("**Rota Analisada:**")
                    st.dataframe(registro['rota'], use_container_width=True)

                    st.write("**Resultado da Análise de Risco:**")
                    df_res_hist = pd.DataFrame([registro['risco_agregado']], columns=nomes_das_classes)
                    st.dataframe(
                        df_res_hist.style.format("{:.2%}").background_gradient(cmap='YlOrRd', axis=1),
                        use_container_width=True
                    )

# --- Carregando Artefatos e Classes ---
ROTAS_PREDEFINIDAS = {
        "Viagem SP ➔ RJ (Via Dutra)": [
            {'uf': 'SP', 'br': 116},
            {'uf': 'RJ', 'br': 116}
        ],
        "Litoral Sul de SP (Régis Bittencourt + Rio-Santos)": [
            {'uf': 'SP', 'br': 116}, # Trecho da Régis Bittencourt
            {'uf': 'SP', 'br': 101}  # Trecho da Rio-Santos
        ],
        "Viagem PR ➔ SC (Via Litoral)": [
            {'uf': 'PR', 'br': 116},
            {'uf': 'PR', 'br': 101},
            {'uf': 'SC', 'br': 101}
        ],
        "Principal Rota de Minas Gerais (Fernão Dias)": [
            {'uf': 'MG', 'br': 381}
        ]
}
nomes_das_classes = ['Com Vítimas Fatais','Com Vítimas Feridas','Ilesos']
modelo, encoder_features, encoder_target, colunas_modelo = load_objects()
db_brs = carregar_dados_brs()

# ==============================================================================
# APLICAÇÃO STREAMLIT
# ==============================================================================
pd.set_option("styler.render.max_elements", 2000000)
st.set_page_config(
    page_title="Modelos de Predição - Análise de Acidentes",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Header Principal ---
st.markdown('<div class="main-header">🤖 Modelos de Predição de Gravidade</div>', unsafe_allow_html=True)
st.markdown("<div style='text-align: center; font-size: 1.2em; color: #666;'>Interaja com os modelos de Machine Learning para prever a gravidade de acidentes em diferentes cenários.</div>", unsafe_allow_html=True)

# Verifica se os modelos foram carregados corretamente
if modelo is None:
    st.error(" Arquivos de modelo não encontrados! Garanta que os arquivos .pkl estão no diretório correto.")
else:
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

    # --- Sidebar para seleção de modo ---
    st.sidebar.header("Modo de Operação")
    input_mode = st.sidebar.radio(
        "Escolha o tipo de previsão:",
        ("Previsão em Lote (Upload)", "Previsão de Rota")
    )
    
    # --- Renderização da página selecionada ---
    if input_mode == "Previsão em Lote (Upload)":
        rederizar_pagina_upload_arquivo() 

    elif input_mode == "Previsão de Rota":
        renderizar_pagina_entrada_manual()