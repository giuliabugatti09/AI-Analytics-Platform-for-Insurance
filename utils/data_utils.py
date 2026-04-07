import pandas as pd
import io
import streamlit as st

def load_data(file):
    return pd.read_csv(file)

def basic_profile(df: pd.DataFrame):
    n_rows, n_cols = df.shape
    null_pct = (df.isna().sum().sum() / (n_rows * n_cols) * 100) if n_rows*n_cols else 0.0
    return {"n_rows": n_rows, "n_cols": n_cols, "null_pct": null_pct}

def filter_dataframe(df: pd.DataFrame):
    return df  # simplificado aqui

def to_excel_download(df: pd.DataFrame):
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="amostra")
    return out.getvalue()
