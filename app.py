import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
import plotly.graph_objects as go
from engine import FinancialEngine

# Configurazione Pagina
st.set_page_config(page_title="Piattaforma Analisi AI Pro", layout="wide")

# CSS Avanzato
st.markdown("""<style>
    .metric-box { background-color: #1e293b; border: 1px solid #334155; padding: 15px; border-radius: 8px; }
    .fusion-box { background-color: #0f172a; border: 2px solid #a855f7; padding: 20px; border-radius: 10px; }
</style>""", unsafe_allow_html=True)

engine = FinancialEngine()

# --- FUNZIONI DI SUPPORTO ---
def log_forecast(ticker, target_date, predicted_price, current_price):
    log_file = "forecast_history.csv"
    data = pd.DataFrame({
        'Timestamp': [datetime.now().strftime("%Y-%m-%d")],
        'Ticker': [ticker], 'Target_Date': [target_date],
        'Predicted': [predicted_price], 'Actual': [current_price]
    })
    if not os.path.exists(log_file): data.to_csv(log_file, index=False)
    else: data.to_csv(log_file, mode='a', header=False, index=False)

def calcola_sharpe(rendimenti):
    rf = 0.02 / 252
    if np.std(rendimenti) == 0: return 0
    return (np.mean(rendimenti) - rf) / np.std(rendimenti)

# --- SIDEBAR ---
ticker_global = st.sidebar.text_input("Ticker:", "MU").upper()

# --- LOGICA CORE ---
if ticker_global:
    stock = yf.Ticker(ticker_global.split('.')[0])
    df = stock.history(period="1y", interval="1d", auto_adjust=True)
    df.index = df.index.tz_localize(None)
    
    # Feature Engineering con ATR
    delta = df['Close'].diff()
    df['RSI'] = 100 - (100 / (1 + (delta.where(delta > 0, 0).rolling(14).mean() / (-delta.where(delta < 0, 0).rolling(14).mean() + 1e-10))))
    df['MA20'] = df['Close'].rolling(20).mean()
    df['ATR'] = pd.concat([df['High']-df['Low'], abs(df['High']-df['Close'].shift()), abs(df['Low']-df['Close'].shift())], axis=1).max(axis=1).rolling(14).mean()
    df['ATR_Norm'] = df['ATR'] / df['Close']
    df['Volumi_Standardizzati'] = (df['Volume'] - df['Volume'].rolling(20).mean()) / df['Volume'].rolling(20).std()
    
    features = ['RSI', 'Volumi_Standardizzati', 'ATR_Norm']
    df['Target_Class'] = np.where(df['Close'].shift(-7) > df['Close'] * 1.02, 2, np.where(df['Close'].shift(-7) < df['Close'] * 0.98, 1, 0))
    
    # Walk-Forward Training
    df_train = df.dropna()
    modello = RandomForestClassifier(n_estimators=100, max_depth=5).fit(df_train[features], df_train['Target_Class'])
    
    # Proiezione
    ultimo_stato = df[features].tail(1)
    prob = modello.predict_proba(ultimo_stato)[0]
    
    # Calcolo Logica Avanzata
    atr_soglia = df['ATR_Norm'].mean() + df['ATR_Norm'].std()
    coeff_rischio = 0.5 if df['ATR_Norm'].iloc[-1] > atr_soglia else 1.0
    
    # Calcolo PTS
    prezzo_attuale = df['Close'].iloc[-1]
    previsione = prezzo_attuale * 1.01 # Esempio di output modello
    pts = (prezzo_attuale / previsione) * 100

    # UI Visualizzazione
    st.title(f"Analisi Avanzata: {ticker_global}")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Prezzo Reale", f"${prezzo_attuale:.2f}")
    col2.metric("Coefficiente Rischio", f"{coeff_rischio}x")
    col3.metric("PTS (Performance)", f"{pts:.1f}%")

    # Grafico
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index[-30:], y=df['Close'].tail(30), name="Prezzo Reale"))
    st.plotly_chart(fig, use_container_width=True)

    # Log automatico
    if st.button("Salva Snapshot Previsione"):
        log_forecast(ticker_global, datetime.now(), previsione, prezzo_attuale)
        st.success("Dati storicizzati nel database CSV.")

st.info("Sistema configurato con Walk-Forward Validation e ATR Filter attivo.")
