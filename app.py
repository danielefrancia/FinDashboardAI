import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
import plotly.graph_objects as go
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

# --- BARRA LATERALE ---
st.sidebar.title("📊 Navigazione")
tipo_analisi = st.sidebar.radio(
    "Moduli Disponibili:",
    ["📈 Dashboard Predittiva Settimanale", "🔄 Modello Opzioni, Stop-Loss & Rischio"]
)

st.sidebar.markdown("---")

# --- CONFIGURAZIONE TARGET NELLA BARRA LATERALE ---
st.sidebar.subheader("🎯 Configurazione Target")
ticker_global = st.sidebar.text_input("Ticker di Riferimento:", "MU").upper().strip()
st.sidebar.markdown("---")

# --- GENERAZIONE GUIDA DINAMICA E SPECIFICA IN ITALIANO ---
st.sidebar.subheader(f"📖 Guida Operativa: {ticker_global}")

if ticker_global:
    try:
        ticker_pulito = ticker_global.split('.')[0]
        stock_guida = yf.Ticker(ticker_pulito)
        df_guida = stock_guida.history(period="30d", interval="1d", auto_adjust=True)
        
        if not df_guida.empty and len(df_guida) > 20:
            chiusura_attuale = float(df_guida['Close'].iloc[-1])
            
            # Calcolo RSI per Guida
            delta = df_guida['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / (loss + 1e-10)
            rsi_attuale = float(100 - (100 / (1 + rs)).iloc[-1])
            
            # Calcolo Bollinger per Guida
            ma20 = df_guida['Close'].rolling(window=20).mean()
            std20 = df_guida['Close'].rolling(window=20).std()
            b_upper = float((ma20 + (2 * std20)).iloc[-1])
            b_lower = float((ma20 - (2 * std20)).iloc[-1])
            
            if rsi_attuale > 70:
                condizione_rsi = "IPERCOMPRATO 🔥"
                consiglio_do = "Valuta prese di profitto parziali o l'acquisto di opzioni Put protettive. Il titolo è surriscaldato a breve termine."
                consiglio_dont = "Evita assolutamente ingressi 'FOMO' (inseguendo il rialzo) sui massimi della sessione attuale."
            elif rsi_attuale < 30:
                condizione_rsi = "IPERVENDUTO 📉"
                consiglio_do = "Inizia ad accumulare posizioni long in scala (Dollar Cost Averaging). Il prezzo è statisticamente a sconto."
                consiglio_dont = "Non vendere in preda al panico se il prezzo rompe i supporti di breve termine; potresti liquidare sui minimi."
            else:
                condizione_rsi = "NEUTRO / TREND ATTIVO ↕️"
                if chiusura_attuale > ma20.iloc[-1]:
                    consiglio_do = f"Mantieni la posizione long assecondando l'inerzia rialzista. Usa la media a 20 giorni (${ma20.iloc[-1]:.2f}) come Stop di protezione."
                    consiglio_dont = "Evita di metterti contro-trend (short selvaggi) senza inversioni strutturali macro confermate."
                else:
                    consiglio_do = "Attendi il breakout della resistenza volumetrica o il test del supporto inferiore prima di esporti."
                    consiglio_dont = "Non anticipare il mercato immettendo size importanti senza una chiara conferma di inversioni."

            with st.sidebar.expander("🔍 Stato Tecnico Attuale"):
                st.markdown(f"""
                * **Prezzo Ultimo:** ${chiusura_attuale:.2f}
                * **RSI (14gg):** {rsi_attuale:.1f} ({condizione_rsi})
                * **Bande Bollinger:** Min ${b_lower:.2f} | Max ${b_upper:.2f}
                """)
                
            with st.sidebar.expander("🎯 COSA FARE (Do's)"):
                st.markdown(f"✅ **{consiglio_do}**")
                
            with st.sidebar.expander("❌ COSA EVITARE (Don'ts)"):
                st.markdown(f"⚠️ **{consiglio_dont}**")
                st.markdown("⚠️ *Evita di operare se l'Affidabilità Algoritmica nel Modulo 2 scende sotto il 45%.*")
        else:
            st.sidebar.warning("Storico insufficiente per generare consigli personalizzati.")
    except Exception as e:
        st.sidebar.error("Guida non disponibile: inserisci un ticker valido.")

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


# --- MODULO 2: MODELLO OPZIONI, STOP-LOSS & VALIDAZIONE FORECAST ---
elif tipo_analisi == "🔄 Modello Opzioni, Stop-Loss & Rischio":
    st.title("📊 Modello Predittivo Avanzato & Validazione Forecast")
    st.markdown("<p style='color: #64748b;'>Validazione statistica e confronto tra i dati reali del mese e il Forecast simulato a inizio mese.</p>", unsafe_allow_html=True)
    
    if ticker_global:
        with st.spinner("Estrazione e pulizia dati real-time..."):
            try:
                ticker_pulito = ticker_global.split('.')[0]
                stock_obj = yf.Ticker(ticker_pulito)
                df = stock_obj.history(period="6mo", interval="1d", auto_adjust=True)
                
                if not df.empty and len(df) > 50:
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = [col[0] for col in df.columns]
                    
                    # --- RIMOZIONE TIMESTAMPS / FUSO ORARIO PER EVITARE L'ERRORE ---
                    if df.index.tz is not None:
                        df.index = df.index.tz_localize(None)

                    df = pd.DataFrame({
                        'Open': df['Open'].squeeze(),
                        'High': df['High'].squeeze(),
                        'Low': df['Low'].squeeze(),
                        'Close': df['Close'].squeeze(),
                        'Volume': df['Volume'].squeeze()
                    }, index=df.index)
                    
                    prezzo_attuale = float(df['Close'].iloc[-1])
                    
                    # --- CALCOLO INDICATORI SULLO STORICO ---
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
                    df
