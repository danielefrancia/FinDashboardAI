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
                    df['Distanza_Banda_Inf'] = (df['Close'] - df['Bollinger_Lower']) / df['Bollinger_Lower']
                    df['Vol_Media20'] = df['Volume'].rolling(window=20).mean()
                    df['Vol_STD20'] = df['Volume'].rolling(window=20).std()
                    df['Volumi_Standardizzati'] = (df['Volume'] - df['Vol_Media20']) / (df['Vol_STD20'] + 1e-10)
                    
                    features = ['RSI', 'Distanza_Banda_Sup', 'Distanza_Banda_Inf', 'Volumi_Standardizzati']
                    
                    # Target Shiftati per l'addestramento (senza data leakage)
                    for i in range(1, 6):
                        df[f'Target_Return_t+{i}'] = (df['Close'].shift(-i) - df['Close']) / df['Close']
                        
                    rendimento_futuro_7g = (df['Close'].shift(-7) - df['Close']) / df['Close']
                    target_classes = []
                    for val in rendimento_futuro_7g:
                        if pd.isna(val): target_classes.append(np.nan)
                        elif val >= 0.02: target_classes.append(2)
                        elif val <= -0.02: target_classes.append(1)
                        else: target_classes.append(0)
                    df['Target_Class'] = target_classes

                    # --- ADDESTRAMENTO MODELLI SUI DATI STORICI ---
                    df_pulito = df.dropna(subset=['Target_Class'] + [f'Target_Return_t+{i}' for i in range(1, 6)] + features).copy()
                    X_all = df_pulito[features]
                    y_class_all = df_pulito['Target_Class'].astype(int)
                    
                    modello_rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
                    modello_rf.fit(X_all, y_class_all)
                    
                    # Regressori lineari per i 5 giorni di proiezione
                    regressori = {}
                    for i in range(1, 6):
                        reg_m = LinearRegression()
                        reg_m.fit(X_all, df_pulito[f'Target_Return_t+{i}'])
                        regressori[i] = reg_m

                   # --- SIMULAZIONE FORECAST (CORRETTO PER VEDERE DATI AD OGGI) ---
oggi = datetime.now()
# Cambiamo la logica: invece di inizio_mese fisso, usiamo una finestra mobile di 30 giorni
data_di_riferimento = oggi - timedelta(days=30) 
dati_mese_corrente = df[df.index >= pd.Timestamp(data_di_riferimento)]

if len(dati_mese_corrente) >= 2:
    # Qui manteniamo la tua logica di indicizzazione
    idx_inizio = dati_mese_corrente.index[0]
    prezzo_inizio = float(df.loc[idx_inizio, 'Close'])
    features_inizio = df.loc[[idx_inizio], features]
    
    date_forecast_storico = [idx_inizio]
    prezzi_forecast_storico = [prezzo_inizio]
    
    pos_iniziale = df.index.get_loc(idx_inizio)
    # ... (il resto del tuo ciclo for rimane identico)
    for i in range(1, 6):
        if pos_iniziale + i < len(df):
            data_f = df.index[pos_iniziale + i]
            var_stimata = regressori[i].predict(features_inizio)[0]
            var_stimata = np.clip(var_stimata, -0.15, 0.15)
            date_forecast_storico.append(data_f)
            prezzi_forecast_storico.append(prezzo_inizio * (1 + var_stimata))
    
    show_backtest = True
else:
    # Fallback se non ci sono dati: forziamo show_backtest a False per evitare errori grafici
    show_backtest = False
    st.warning("Dati insufficienti per il backtest (finestra 30 giorni troppo corta o dati mancanti).")

                    # --- PROIEZIONE FUTURA IN AVANTI AD OGGI (t+1 a t+5) ---
                    ultimo_stato_df = df[features].tail(1).ffill()
                    previsioni_prezzo_futuro = []
                    variaz_reg_salvate = []
                    
                    mappa_giorni = {
                        'Monday': 'Lunedì', 'Tuesday': 'Martedì', 'Wednesday': 'Mercoledì',
                        'Thursday': 'Giovedì', 'Friday': 'Venerdì', 'Saturday': 'Sabato', 'Sunday': 'Domenica'
                    }
                    giorni_futuri = []
                    data_c = datetime.now()
                    p = 1
                    while len(giorni_futuri) < 5:
                        g_futuro = data_c + timedelta(days=p)
                        if g_futuro.weekday() < 5:
                            giorni_futuri.append(f"{mappa_giorni.get(g_futuro.strftime('%A'))} ({g_futuro.strftime('%d/%m')})")
                        p += 1

                    for i in range(1, 6):
                        var_stimata = regressori[i].predict(ultimo_stato_df)[0]
                        var_stimata = np.clip(var_stimata, -0.15, 0.15)
                        variaz_reg_salvate.append(var_stimata)
                        previsioni_prezzo_futuro.append(prezzo_attuale * (1 + var_stimata))

                    # --- INTERFACCIA GRAFICA MATRICE DI SENTIMENT ---
                    macro_data = engine.analyze_ticker(ticker_global)
                    sentiment_macro = macro_data['sentiment']
                    prob_array = modello_rf.predict_proba(ultimo_stato_df)[0]
                    classi = list(modello_rf.classes_)
                    prob_take_profit = prob_array[classi.index(2)] if 2 in classi else 0.0
                    prob_stop_loss = prob_array[classi.index(1)] if 1 in classi else 0.0
                    
                    st.write("### 🎯 Matrice Operativa Integrata (Sentiment + Rischio)")
                    if sentiment_macro > 65 and prob_take_profit > 0.40:
                        conclusione_hub, colore_hub = "🟢 CONDIZIONE DI ACQUISTO (BULLISH CONVERGENCE)", "#10b981"
                    elif prob_stop_loss > 0.45:
                        conclusione_hub, colore_hub = "🔴 EVITARE INGRESSI / ATTESA (BEARISH DOMINANCE)", "#ef4444"
                    else:
                        conclusione_hub, colore_hub = "🟡 STRATEGIA NEUTRA DI COMPRESSIONE (CONSOLIDAMENTO)", "#f59e0b"

                    st.markdown(f"""
                        <div class="fusion-box">
                            <h3 style="margin: 0 0 10px 0; color: {colore_hub}; font-size: 1.35rem !important;">{conclusione_hub}</h3>
                            <p style="color: #cbd5e1; margin-bottom: 0;">Analisi probabilistica calcolata su {len(df_pulito)} sessioni storiche.</p>
                        </div>
                    """, unsafe_allow_html=True)

                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("#### 📈 Indicatori Tecnici Certificati")
                        ultimo_valore_tecnico = df.iloc[-1]
                        st.markdown(f"""
                            <div class="metric-box">
                                <b>Ultimo Prezzo di Chiusura:</b> ${prezzo_attuale:.2f}<br>
                                <b>RSI (14 giorni):</b> <span style="color:#3b82f6; font-weight:bold;">{ultimo_valore_tecnico['RSI']:.2f}</span><br>
                                <b>Resistenza (Bollinger Sup.):</b> ${ultimo_valore_tecnico['Bollinger_Upper']:.2f}<br>
                                <b>Supporto (Bollinger Inf.):</b> ${ultimo_valore_tecnico['Bollinger_Lower']:.2f}<br>
                                <b>Z-Score Volumi:</b> {ultimo_valore_tecnico['Volumi_Standardizzati']:.2f}
                            </div>
                        """, unsafe_allow_html=True)
                        
                    with col2:
                        st.markdown("#### 🎯 Probabilità dell'Ensemble Model")
                        st.markdown(f"""
                            <div class="metric-box">
                                <b>Affidabilità Algoritmo:</b> {modello_rf.score(X_all, y_class_all):.2%}<br>
                                <span style="color:#10b981; font-weight:bold;">🚀 Probabilità Take Profit (≥ +2%):</span> {prob_take_profit:.1%}<br>
                                <span style="color:#3b82f6; font-weight:bold;">↔️ Probabilità Lateralità:</span> {1-(prob_take_profit+prob_stop_loss):.1%}<br>
                                <span style="color:#ef4444; font-weight:bold;">⚠️ Probabilità Stop Loss (≤ -2%):</span> {prob_stop_loss:.1%}
                            </div>
                        """, unsafe_allow_html=True)

                    # --- GRAFICO DI VALIDAZIONE (REALE vs FORECAST) ---
                    st.write("---")
                    st.subheader("📉 Analisi di Validazione: Reale vs Modello Predictor")
                    st.write("Questo grafico mette a confronto l'andamento reale del prezzo nel mese corrente con il forecast generato artificialmente dal modello a inizio mese.")
                    
                    fig_val = go.Figure()
                    
                    # Linea 1: Prezzo Reale
                    fig_val.add_trace(go.Scatter(
                        x=dati_mese_corrente.index, y=dati_mese_corrente['Close'],
                        mode='lines+markers', name='Prezzo Reale (Mercato)',
                        line=dict(color='#3b82f6', width=3)
                    ))
                    
                    # Linea 2: Forecast
                    if show_backtest:
                        fig_val.add_trace(go.Scatter(
                            x=date_forecast_storico, y=prezzi_forecast_storico,
                            mode='lines+markers', name='Forecast Simulato Inizio Mese',
                            line=dict(color='#ef4444', width=2, dash='dash'),
                            marker=dict(symbol='x', size=7)
                        ))
                    
                    fig_val.update_layout(
                        height=350, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                        xaxis=dict(showgrid=True, gridcolor='#1e293b', tickfont=dict(color='#94a3b8')),
                        yaxis=dict(showgrid=True, gridcolor='#1e293b', tickfont=dict(color='#94a3b8'), autorange=True),
                        legend=dict(font=dict(color='#f8fafc'), bgcolor='rgba(15,23,42,0.8)', orientation='h', y=1.1)
                    )
                    st.plotly_chart(fig_val, use_container_width=True, config={'displayModeBar': False})

                    # --- PROIEZIONE FUTURA IN AVANTI ---
                    st.write("---")
                    st.subheader("📅 Proiezione Lineare Adattiva (Prossimi 5 Giorni)")
                    
                    df_previsioni = pd.DataFrame({
                        'Giorno Previsto': giorni_futuri,
                        'Prezzo Target': [f"${p:.2f}" for p in previsioni_prezzo_futuro],
                        'Variazione Attesa': [f"{v:+.2%}" for v in variaz_reg_salvate]
                    })
                    
                    c_tab, c_graf = st.columns([1, 1])
                    with c_tab:
                        st.dataframe(df_previsioni, use_container_width=True, hide_index=True)
                    with c_graf:
                        fig_fut = go.Figure()
                        fig_fut.add_trace(go.Scatter(
                            x=giorni_futuri, y=previsioni_prezzo_futuro,
                            mode='lines+markers', line=dict(color='#a855f7', width=3),
                            marker=dict(size=6, color='#3b82f6'), name='Prezzo Posteriore'
                        ))
                        fig_fut.update_layout(
                            margin=dict(l=20, r=20, t=10, b=10), height=220,
                            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                            xaxis=dict(showgrid=True, gridcolor='#1e293b', tickfont=dict(color='#94a3b8')),
                            yaxis=dict(showgrid=True, gridcolor='#1e293b', tickfont=dict(color='#94a3b8'), autorange=True)
                        )
                        st.plotly_chart(fig_fut, use_container_width=True, config={'displayModeBar': False})

                    st.success("Analisi di distacco modello-reale completata correttamente.")
                else:
                    st.error("Dati storici non disponibili o non allineati per il ticker inserito.")
            except Exception as e:
                st.error(f"Errore di calcolo nell'Hub di integrazione: {str(e)}")
