import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
from engine import FinancialEngine

# Configurazione del layout della pagina
st.set_page_config(page_title="Piattaforma Analisi AI", layout="wide", initial_sidebar_state="expanded")

# Stile CSS ottimizzato per il look scuro ultra-professionale
st.markdown("""
    <style>
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    h1 { font-size: 2.2rem !important; font-weight: 700; color: #f8fafc; }
    h3 { font-size: 1.4rem !important; font-weight: 600; color: #94a3b8; margin-top: 1.5rem; }
    .hot-card { background-color: #1e293b; border-left: 4px solid #3b82f6; padding: 15px; border-radius: 6px; margin-bottom: 10px; }
    .day-card { background-color: #1e293b; padding: 15px; border-radius: 8px; border: 1px solid #334155; text-align: center; }
    .metric-box { background-color: #1e293b; border: 1px solid #334155; padding: 15px; border-radius: 8px; margin-bottom: 10px; }
    .fusion-box { background-color: #0f172a; border: 2px solid #a855f7; padding: 20px; border-radius: 10px; margin-bottom: 25px; }
    </style>
""", unsafe_allow_html=True)

engine = FinancialEngine()

# --- BARRA LATERALE (NAVIGAZIONE & INPUT UNICO) ---
st.sidebar.title("📊 Navigazione")
tipo_analisi = st.sidebar.radio(
    "Moduli Disponibili:",
    ["📈 Dashboard Predittiva Settimanale", "🔄 Modello Opzioni, Stop-Loss & Rischio"]
)
st.sidebar.markdown("---")

st.sidebar.subheader("🎯 Configurazione Target")
ticker_global = st.sidebar.text_input("Ticker di Riferimento:", "MU").upper().strip()

st.sidebar.markdown("---")
st.sidebar.caption("Sviluppato con FinGPT & TradingAgents")


# --- MODULO 1: DASHBOARD SETTIMANALE ---
if tipo_analisi == "📈 Dashboard Predittiva Settimanale":
    st.title("📈 Financial Insights Dashboard")
    st.markdown("<p style='color: #64748b;'>Analisi predittiva multi-agente integrata</p>", unsafe_allow_html=True)
    
    st.write("### 🔥 AI Hot Scanner (Anteprima Titoli Caldi)")
    hot_list = engine.get_trending_stocks()
    cols_hot = st.columns(len(hot_list))
    for i, stock_info in enumerate(hot_list):
        with cols_hot[i]:
            st.markdown(f"""
                <div class="hot-card">
                    <span style="font-size: 1.2rem; font-weight: bold; color: #f8fafc;">{stock_info['ticker']}</span><br>
                    <span style="font-size: 0.85rem; color: #3b82f6; font-weight: 500;">{stock_info['reason']}</span>
                </div>
            """, unsafe_allow_html=True)

    st.markdown("<br><hr style='border-color: #334155;'>", unsafe_allow_html=True)

    st.write("### 🔍 Analisi Predittiva Settimanale")
    if ticker_global:
        with st.spinner("Elaborazione modelli AI in corso..."):
            data = engine.analyze_ticker(ticker_global)
            
        st.markdown(f"#### Target: <span style='color: #3b82f6;'>{data['name']}</span> ({data['ticker']})", unsafe_allow_html=True)
        
        m1, m2 = st.columns(2)
        m1.metric(label="FinGPT Sentiment Score", value=f"{data['sentiment']}%")
        m2.metric(label="Previsione Statistica", value=data['prediction'])
        
        st.write("#### 📅 Strategia Operativa nei Giorni della Settimana")
        giorni_cols = st.columns(5)
        for idx, (giorno, dettagli) in enumerate(data['weekly'].items()):
            with giorni_cols[idx]:
                if "Acquisto" in dettagli['stato']: color = "#10b981"
                elif "Vendita" in dettagli['stato']: color = "#ef4444"
                else: color = "#f59e0b"
                    
                st.markdown(f"""
                    <div class="day-card">
                        <b style="color: #94a3b8; font-size: 1.05rem;">{giorno}</b><br>
                        <p style="color: {color}; font-weight: bold; margin: 10px 0 5px 0; font-size: 0.95rem;">{dettagli['stato']}</p>
                        <span style="color: #64748b; font-size: 0.8rem;">Affidabilità: {dettagli['confidenza']}</span>
                    </div>
                """, unsafe_allow_html=True)


# --- MODULO 2: MODELLO OPZIONI E RISCHIO ---
elif tipo_analisi == "🔄 Modello Opzioni, Stop-Loss & Rischio":
    st.title("📊 Modello Predittivo Avanzato & Strategia Integrata")
    st.markdown("<p style='color: #64748b;'>Validazione statistica del rischio incrociata con l'analisi multi-agente.</p>", unsafe_allow_html=True)
    
    if ticker_global:
        with st.spinner("Estrazione e pulizia dati real-time..."):
            try:
                # Gestione split e scaricamento pulito
                ticker_pulito = ticker_global.split('.')[0]
                stock_obj = yf.Ticker(ticker_pulito)
                df = stock_obj.history(period="2y", interval="1d", auto_adjust=False, actions=True)
                
                if not df.empty and len(df) > 50:
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.droplevel(1)
                    
                    # Estrazione prezzo coerente post-split
                    prezzo_attuale = float(df['Close'].iloc[-1])
                    
                    # Calcolo indicatori tecnici stabili
                    delta = df['Close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                    rs = gain / (loss + 1e-10)
                    df['RSI'] = 100 - (100 / (1 + rs))
                    
                    df['MA20'] = df['Close'].rolling(window=20).mean()
                    df['STD20'] = df['Close'].rolling(window=20).std()
                    df['Bollinger_Upper'] = df['MA20'] + (2 * df['STD20'])
                    df['Bollinger_Lower'] = df['MA20'] - (2 * df['STD20'])
                    
                    df['Distanza_Banda_Sup'] = (df['Close'] - df['Bollinger_Upper']) / df['Bollinger_Upper']
                    df['Distanza_Banda_Inf'] = (df['Close'] - df['Bollinger_Lower']) / df['Bollinger_Lower']
                    
                    df['Vol_Media20'] = df['Volume'].rolling(window=20).mean()
                    df['Vol_STD20'] = df['Volume'].rolling(window=20).std()
                    df['Volumi_Standardizzati'] = (df['Volume'] - df['Vol_Media20']) / (df['Vol_STD20'] + 1e-10)
                    
                    # Target di classificazione (Orizzonte 7 giorni)
                    rendimento_futuro_7g = (df['Close'].shift(-7) - df['Close']) / df['Close']
                    target_classes = []
                    for val in rendimento_futuro_7g:
                        if pd.isna(val): target_classes.append(0)
                        elif val >= 0.02: target_classes.append(2)
                        elif val <= -0.02: target_classes.append(1)
                        else: target_classes.append(0)
                    df['Target_Class'] = target_classes
                    
                    for i in range(1, 6):
                        df[f'Target_Price_t+{i}'] = df['Close'].shift(-i)
                    
                    # Allineamento split e pulizia NaN storici
                    df = df.bfill().ffill()
                    
                    ultimo_stato = df.iloc[-1].copy()
                    df_train = df.dropna().copy()
                    
                    features = ['RSI', 'Distanza_Banda_Sup', 'Distanza_Banda_Inf', 'Volumi_Standardizzati']
                    X = df_train[features]
                    y_class = df_train['Target_Class']
                    
                    # Machine Learning Ensemble
                    split = int(len(df_train) * 0.8)
                    modello_rf1 = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
                    modello_rf2 = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=2026)
                    modello_rf1.fit(X.iloc[:split], y_class.iloc[:split])
                    modello_rf2.fit(X.iloc[:split], y_class.iloc[:split])
                    accuratezza = modello_rf1.score(X.iloc[split:], y_class.iloc[split:])
                    
                    vettore_input = np.array([ultimo_stato[features].values])
                    prob1 = modello_rf1.predict_proba(vettore_input)[0]
                    prob2 = modello_rf2.predict_proba(vettore_input)[0]
                    probabilita_array = (prob1 + prob2)
