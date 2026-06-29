import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
import plotly.graph_objects as go
from engine import FinancialEngine

# Configurazione Pagina
st.set_page_config(page_title="Piattaforma Analisi AI", layout="wide")

# Cache per evitare chiamate ripetute API inutili
@st.cache_data(ttl=3600)
def get_stock_data(ticker):
    stock = yf.Ticker(ticker)
    df = stock.history(period="6mo", interval="1d", auto_adjust=True)
    if not df.empty and df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df

# Stile CSS (mantenuto invariato)
st.markdown("""
    <style>
    .hot-card { background-color: #1e293b; border-left: 4px solid #3b82f6; padding: 15px; border-radius: 6px; }
    .fusion-box { background-color: #0f172a; border: 2px solid #a855f7; padding: 20px; border-radius: 10px; }
    </style>
""", unsafe_allow_html=True)

engine = FinancialEngine()

# Sidebar
st.sidebar.title("📊 Navigazione")
tipo_analisi = st.sidebar.radio("Moduli:", ["📈 Dashboard Predittiva", "🔄 Modello Opzioni & Rischio"])
ticker_global = st.sidebar.text_input("Ticker:", "MU").upper().strip()

# --- LOGICA MODULO 2 (Esempio di ottimizzazione) ---
if tipo_analisi == "🔄 Modello Opzioni, Stop-Loss & Rischio" and ticker_global:
    df = get_stock_data(ticker_global)
    
    if len(df) > 50:
        # Calcoli vettoriali (più veloci dei cicli)
        df['RSI'] = 100 - (100 / (1 + (df['Close'].diff(1).clip(lower=0).rolling(14).mean() / 
                                     df['Close'].diff(1).clip(upper=0).abs().rolling(14).mean())))
        
        # ... (restante logica indicatori)
        
        # Validazione e ML
        # NOTA: Assicurati di gestire i NaN generati dai rolling prima del training
        df_clean = df.dropna()
        
        # ... logica del modello
    else:
        st.error("Dati insufficienti.")
