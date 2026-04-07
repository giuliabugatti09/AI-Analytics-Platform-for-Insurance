import pandas as pd

def has_geo(df: pd.DataFrame):
    for la, lo in [("lat", "lon"), ("latitude", "longitude")]:
        if la in df.columns and lo in df.columns:
            return True, la, lo
    return False, None, None

def to_geodf(df: pd.DataFrame, lat_col: str, lon_col: str):
    return df

def risk_layer(df: pd.DataFrame, lat_col: str, lon_col: str, grid_size=0.01):
    g = df.copy()
    g["_grid_lat"] = (g[lat_col] // grid_size) * grid_size
    g["_grid_lon"] = (g[lon_col] // grid_size) * grid_size
    return g.groupby(["_grid_lat", "_grid_lon"]).size().reset_index(name="count")
