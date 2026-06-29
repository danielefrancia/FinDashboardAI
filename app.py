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

# --- BARRA LATERALE ---
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
                ticker_pulito = ticker_global.split('.')[0]
                stock_obj = yf.Ticker(ticker_pulito)
                df = stock_obj.history(period="2y", interval="1d", auto_adjust=True, actions=True)
                
                if not df.empty and len(df) > 50:
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.droplevel(1)
                    
                    prezzo_attuale = float(df['Close'].iloc[-1])
                    
                    # Indicatori Tecnici
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
                    
                    # Target Classificazione Rischio
                    rendimento_futuro_7g = (df['Close'].shift(-7) - df['Close']) / df['Close']
                    target_classes = []
                    for val in rendimento_futuro_7g:
                        if pd.isna(val): target_classes.append(0)
                        elif val >= 0.02: target_classes.append(2)
                        elif val <= -0.02: target_classes.append(1)
                        else: target_classes.append(0)
                    df['Target_Class'] = target_classes
                    
                    # CORREZIONE: Addestriamo la regressione sul RENDIMENTO futuro (variazione %), non sul prezzo assoluto
                    for i in range(1, 6):
                        df[f'Target_Return_t+{i}'] = (df['Close'].shift(-i) - df['Close']) / df['Close']
                    
                    df = df.bfill().ffill()
                    ultimo_stato = df.iloc[-1].copy()
                    df_train = df.dropna().copy()
                    
                    features = ['RSI', 'Distanza_Banda_Sup', 'Distanza_Banda_Inf', 'Volumi_Standardizzati']
                    X = df_train[features]
                    y_class = df_train['Target_Class']
                    
                    # Machine Learning Ensemble (Classificazione)
                    split = int(len(df_train) * 0.8)
                    modello_rf1 = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
                    modello_rf2 = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=2026)
                    modello_rf1.fit(X.iloc[:split], y_class.iloc[:split])
                    modello_rf2.fit(X.iloc[:split], y_class.iloc[:split])
                    accuratezza = modello_rf1.score(X.iloc[split:], y_class.iloc[split:])
                    
                    vettore_input = np.array([ultimo_stato[features].values])
                    prob1 = modello_rf1.predict_proba(vettore_input)[0]
                    prob2 = modello_rf2.predict_proba(vettore_input)[0]
                    probabilita_array = (prob1 + prob2) / 2
                    
                    classi_modello = list(modello_rf1.classes_)
                    prob_laterale = probabilita_array[classi_modello.index(0)] if 0 in classi_modello else 0.0
                    prob_stop_loss = probabilita_array[classi_modello.index(1)] if 1 in classi_modello else 0.0
                    prob_take_profit = probabilita_array[classi_modello.index(2)] if 2 in classi_modello else 0.0
                    
                    # CORREZIONE REGRESSIONE LINEARE: Calcolo variazioni e ricostruzione dinamica del prezzo basata su prezzo_attuale
                    previsioni_prezzo = []
                    variaz_reg_salvate = []
                    for i in range(1, 6):
                        y_reg = df_train[f'Target_Return_t+{i}']
                        modello_reg = LinearRegression()
                        modello_reg.fit(X, y_reg)
                        variazione_stimata = modello_reg.predict(vettore_input)[0]
                        
                        # Impediamo oscillazioni assurde dovute a code storiche anomale
                        variazione_stimata = np.clip(variazione_stimata, -0.15, 0.15)
                        variaz_reg_salvate.append(variazione_stimata)
                        previsioni_prezzo.append(prezzo_attuale * (1 + variazione_stimata))
                    
                    df['Daily_Range'] = (df['High'] - df['Low']) / df['Close']
                    volatilia_media = float(df['Daily_Range'].tail(10).mean())
                    
                    macro_data = engine.analyze_ticker(ticker_global)
                    sentiment_macro = macro_data['sentiment']
                    
                    st.write(f"### 🎯 Matrice Operativa Integrata (Sentiment + Rischio)")
                    
                    if sentiment_macro > 65 and prob_take_profit > 0.40:
                        conclusione_hub = "🟢 CONDIZIONE DI ACQUISTO (BULLISH CONVERGENCE)"
                        colore_hub = "#10b981"
                        desc_hub = f"Sia l'analisi multi-agente ({sentiment_macro}% sentiment) sia il modello probabilistico statistico confermano un'elevata probabilità di rialzo."
                    elif prob_stop_loss > 0.45:
                        conclusione_hub = "🔴 EVITARE INGRESSI / ATTESA (BEARISH DOMINANCE)"
                        colore_hub = "#ef4444"
                        desc_hub = f"Attenzione: L'Ensemble di Machine Learning rileva un rischio di Stop Loss del {prob_stop_loss:.1%}. Struttura tecnica debole."
                    else:
                        conclusione_hub = "🟡 STRATEGIA NEUTRA DI COMPRESSIONE (CONSOLIDAMENTO)"
                        colore_hub = "#f59e0b"
                        desc_hub = "I modelli divergono o indicano lateralità. Ottima stabilità per strategie basate sul trading di range."

                    target_prossima_sessione = previsioni_prezzo[0]
                    var_prossima_sessione = variaz_reg_salvate[0]
                    estensione_attesa = prezzo_attuale * volatilia_media

                    st.markdown(f"""
                        <div class="fusion-box">
                            <span style="color: #94a3b8; font-size: 0.85rem; font-weight: bold;">FUSIONE INTELLIGENTE MULTI-MODELLO</span>
                            <h3 style="margin: 5px 0 12px 0; color: {colore_hub}; font-size: 1.35rem !important;">{conclusione_hub}</h3>
                            <p style="color: #cbd5e1; font-size: 0.95rem; margin-bottom: 12px;">{desc_hub}</p>
                            <hr style="border-color: #1e293b; margin: 10px 0;">
                            <p style="font-size: 1rem; margin-bottom: 4px; color: #f8fafc;">
                                📅 <b>Target Prossima Sessione (t+1):</b> ${target_prossima_sessione:.2f} <span style="color:{colore_hub}; font-weight:bold;">({var_prossima_sessione:+.2%})</span>
                            </p>
                            <p style="font-size: 0.95rem; margin-bottom: 0; color: #94a3b8;">
                                🎯 <b>Range Operativo Stimato:</b> Min: <span style="color:#ef4444;">${(target_prossima_sessione - estensione_attesa/2):.2f}</span> | Max: <span style="color:#10b981;">${(target_prossima_sessione + estensione_attesa/2):.2f}</span>
                            </p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("#### 📈 Indicatori Tecnici Certificati")
                        st.markdown(f"""
                            <div class="metric-box">
                                <b>Ultimo Prezzo di Chiusura:</b> ${prezzo_attuale:.2f}<br>
                                <b>RSI (14 giorni):</b> <span style="color:#3b82f6; font-weight:bold;">{ultimo_stato['RSI']:.2f}</span><br>
                                <b>Resistenza (Bollinger Sup.):</b> ${ultimo_stato['Bollinger_Upper']:.2f}<br>
                                <b>Supporto (Bollinger Inf.):</b> ${ultimo_stato['Bollinger_Lower']:.2f}<br>
                                <b>Z-Score Volumi:</b> {ultimo_stato['Volumi_Standardizzati']:.2f}
                            </div>
                        """, unsafe_allow_html=True)
                        
                    with col2:
                        st.markdown("#### 🎯 Probabilità dell'Ensemble Model")
                        st.markdown(f"""
                            <div class="metric-box">
                                <b>Affidabilità Algoritmo:</b> {accuratezza:.2%}<br>
                                <span style="color:#10b981; font-weight:bold;">🚀 Probabilità Take Profit (≥ +2%):</span> {prob_take_profit:.1%}<br>
                                <span style="color:#3b82f6; font-weight:bold;">↔️ Probabilità Lateralità:</span> {prob_laterale:.1%}<br>
                                <span style="color:#ef4444; font-weight:bold;">⚠️ Probabilità Stop Loss (≤ -2%):</span> {prob_stop_loss:.1%}
                            </div>
                        """, unsafe_allow_html=True)
                    
                    st.write("---")
                    st.subheader("📅 Proiezione Lineare Adattiva (t+1 a t+5)")
                    st.write("Prezzi puntuali attesi calcolati tramite regressione logica adattata alle variazioni percentuali.")
                    
                    giorni_settimana = []
                    data_corrente = datetime.now()
                    passo = 1
                    while len(giorni_settimana) < 5:
                        giorno_futuro = data_corrente + timedelta(days=passo)
                        if giorno_futuro.weekday() < 5:
                            giorni_settimana.append(giorno_futuro.strftime('%A (%d/%m)'))
                        passo += 1
                    
                    df_previsioni = pd.DataFrame({
                        'Giorno Previsto': giorni_settimana,
                        'Prezzo Target': [f"${p:.2f}" for p in previsioni_prezzo],
                        'Variazione Attesa': [f"{v:+.2%}" for v in variaz_reg_salvate]
                    })
                    
                    c_tab, c_graf = st.columns([1, 1])
                    with c_tab:
                        st.dataframe(df_previsioni, use_container_width=True, hide_index=True)
                    with c_graf:
                        df_chart = pd.DataFrame({'Prezzo Target': previsioni_prezzo}, index=giorni_settimana)
                        st.line_chart(df_chart)
                        
                    st.success("Sincronizzazione dati e report eseguiti correttamente.")
                else:
                    st.error("Dati storici non disponibili o non allineati per il ticker inserito.")
            except Exception as e:
                st.error(f"Errore di calcolo nell'Hub di integrazione: {str(e)}")
