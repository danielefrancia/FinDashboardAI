import streamlit as st
from engine import FinancialEngine

# Configurazione del layout della pagina (ottimizzata per PC e Smartphone)
st.set_page_config(page_title="AI Financial Dashboard", layout="wide", initial_sidebar_state="collapsed")

# Stile CSS aggiuntivo per migliorare la visualizzazione da mobile
st.markdown("""
    <style>
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    h1 { font-size: 2.2rem !important; }
    h3 { font-size: 1.3rem !important; }
    </style>
""", unsafe_allow_html=True)

st.title("📊 Financial Insights Dashboard")
st.caption("Analisi integrata basata su TradingAgents & FinGPT Sentiment")

# Inizializziamo il motore di calcolo
engine = FinancialEngine()

# --- SEZIONE 1: ANTEPRIMA TITOLI CALDI ---
st.write("### 🔥 AI Hot Scanner (Anteprima Titoli Caldi)")
hot_list = engine.get_trending_stocks()

# Crea colonne dinamiche: su PC sono affiancate, su mobile si impilano da sole
cols_hot = st.columns(len(hot_list))
for i, stock_info in enumerate(hot_list):
    with cols_hot[i]:
        st.info(f"**{stock_info['ticker']}**\n\n{stock_info['reason']}")

st.markdown("---")

# --- SEZIONE 2: ANALISI DETTAGLIATA ---
st.write("### 🔍 Analisi Predittiva Settimanale")
ticker_input = st.text_input("Inserisci un Ticker manualmente (es. AAPL, NVDA, TSLA):", "NVDA").upper()

if ticker_input:
    with st.spinner(f"Integrazione report e calcolo metriche per {ticker_input}..."):
        data = engine.analyze_ticker(ticker_input)
        
    st.write(f"#### Analisi per: **{data['name']}** ({data['ticker']})")
    
    # KPI Principali in evidenza
    m1, m2 = st.columns(2)
    m1.metric(label="FinGPT Sentiment Score", value=f"{data['sentiment']}%")
    m2.metric(label="Previsione Statistica", value=data['prediction'])
    
    st.write("#### 📆 Strategia Operativa per i Giorni della Settimana")
    
    # Generiamo 5 colonne flessibili per i giorni lavorativi
    giorni_cols = st.columns(5)
    for idx, (giorno, dettagli) in enumerate(data['weekly'].items()):
        with giorni_cols[idx]:
            st.markdown(f"**{giorno}**")
            
            # Colorazione condizionale in base all'operazione consigliata
            if "Acquisto" in dettagli['stato']:
                st.success(dettagli['stato'])
            elif "Vendita" in dettagli['stato']:
                st.error(dettagli['stato'])
            else:
                st.warning(dettagli['stato'])
                
            st.caption(f"Affidabilità: {dettagli['confidenza']}")