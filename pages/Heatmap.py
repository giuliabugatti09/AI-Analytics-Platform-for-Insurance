import streamlit as st
import pandas as pd
import pydeck as pdk
import numpy as np
from scipy.spatial import cKDTree
from utils.geo_utils import has_geo
from components.Agente_Acidentes import renderizar_pagina_agente_acidentes

# ==============================================================================
# 1. PAGE CONFIGURATION & UI STYLING
# ==============================================================================
st.set_page_config(
    page_title="Risk Heatmap - Accident Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        text-align: center;
        color: #ff5100;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .stAlert {
        background-color: rgba(255, 81, 0, 0.1);
        border-left: 5px solid #ff5100;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">Accident Risk Heatmap</div>', unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #666;'>Interactive geospatial analysis identifying high-density accident hotspots.</p>", unsafe_allow_html=True)
st.divider()

# ==============================================================================
# 2. DATA PROCESSING (OPTIMIZED WITH CACHING)
# ==============================================================================
CSV_PATH = "data/acidentes2025_todas_causas_tipos.csv"

@st.cache_data(show_spinner="Processing geographic data...")
def get_processed_geo_data(path):
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        return None, None, None

    ok, lat_col, lon_col = has_geo(df)
    if not ok:
        return None, None, None

    # Helper for coordinate cleaning
    def normalize_coord(val, coord_type):
        if pd.isnull(val): return None
        val_str = str(val).replace(" ", "").replace(",", ".")
        try:
            num = float(val_str)
            if coord_type == "lat" and -90 <= num <= 90: return num
            if coord_type == "lon" and -180 <= num <= 180: return num
        except: return None
        return None

    df[lat_col] = df[lat_col].apply(lambda x: normalize_coord(x, "lat"))
    df[lon_col] = df[lon_col].apply(lambda x: normalize_coord(x, "lon"))
    df = df.dropna(subset=[lat_col, lon_col])

    # Filter strictly for Brazilian territory boundaries
    df = df[
        (df[lat_col] >= -33.75) & (df[lat_col] <= 5.27) &
        (df[lon_col] >= -73.99) & (df[lon_col] <= -34.79)
    ]

    # Calculate density using cKDTree (Accidents within ~100m)
    coords = df[[lat_col, lon_col]].to_numpy()
    if len(coords) > 0:
        tree = cKDTree(coords)
        counts = tree.query_ball_point(coords, r=0.001)
        df["nearby_accidents"] = [len(c)-1 for c in counts]
    
    return df, lat_col, lon_col

df, lat_col, lon_col = get_processed_geo_data(CSV_PATH)

if df is None:
    st.error("🚨 Geographic data source not found or missing Lat/Lon columns.")
    st.stop()

# ==============================================================================
# 3. SIDEBAR & AI AGENT
# ==============================================================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/854/854878.png", width=80)
    st.title("Map Controls")
    
    st.metric("Plotted Points", f"{len(df):,}")
    
    if st.button("🤖 Launch AI Assistant", use_container_width=True):
        @st.dialog("Geospatial Risk Analyst", width="large")
        def agent_dialog():
            renderizar_pagina_agente_acidentes()
        agent_dialog()

# ==============================================================================
# 4. PYDECK GEOSPATIAL VISUALIZATION
# ==============================================================================
st.subheader("Interactive Risk Hotspots")

# Layer 1: Heatmap for density visualization
heatmap_layer = pdk.Layer(
    "HeatmapLayer",
    data=df,
    get_position=[lon_col, lat_col],
    radiusPixels=50,
    intensity=1,
    threshold=0.3,
)

# Layer 2: Scatterplot for precise location auditing
scatter_layer = pdk.Layer(
    "ScatterplotLayer",
    data=df,
    get_position=[lon_col, lat_col],
    get_radius=100,
    get_color=[255, 81, 0, 140], # Sompo Orange with alpha
    pickable=True,
)

view_state = pdk.ViewState(
    latitude=df[lat_col].mean(),
    longitude=df[lon_col].mean(),
    zoom=4,
    pitch=45,
)

r = pdk.Deck(
    layers=[heatmap_layer, scatter_layer],
    initial_view_state=view_state,
    map_style="mapbox://styles/mapbox/dark-v10", # Dark theme makes hotspots pop
    tooltip={
        "html": "<b>Incidents nearby:</b> {nearby_accidents}<br/><b>Lat:</b> {lat}<br/><b>Lon:</b> {lon}",
        "style": {"color": "white", "backgroundColor": "#ff5100"}
    }
)

st.pydeck_chart(r)

st.success(f"Successfully rendered {len(df):,} accident locations across Brazil.")
st.info("💡 **Pro Tip:** Use 'Ctrl + Click' to rotate the map in 3D and visualize terrain risk.")