import streamlit as st
from engine import FinancialEngine

# Configurazione del layout della pagina
st.set_page_config(page_title="AI Financial Dashboard", layout="wide", initial_sidebar_state="collapsed")

# Stile CSS per un look scuro ultra-professionale e bordi curvi
st.markdown("""
    <style>
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    h1 { font-size: 2.2rem !important; font-weight: 700; color: #f8fafc; }
    h3 { font-size: 1.4rem !important; font-weight: 600; color: #94a3b8; margin-top: 1.5rem; }
    
    /* Box personalizzati per i titoli caldi */
    .hot-card {
        background-color: #1e293b;
        border-left: 4px solid #3b82f6;
        padding: 15px;
        border-radius: 6px;
        margin-bottom: 10px;
    }
    /* Box per i giorni della settimana */
    .day-card {
        background-color: #1e293b;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #334155;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

st.title("📈 Financial Insights Dashboard")
st.markdown("<p style='color: #64748b;'>Analisi predittiva multi-agente integrata con modelli FinGPT & TradingAgents</p>", unsafe_allow_html=True)

engine = FinancialEngine()

# --- SEZIONE 1: ANTEPRIMA TITOLI CALDI ---
st.write("### 🔥 AI Hot Scanner (Anteprima Titoli Caldi)")
hot_list = engine.get_trending_stocks()

cols_hot = st.columns(len(hot_list))
for i, stock_info in enumerate(hot_list):
    with cols_hot[i]:
        # Usiamo il container HTML personalizzato per un look premium
        st.markdown(f"""
            <div class="hot-card">
                <span style="font-size: 1.2rem; font-weight: bold; color: #f8fafc;">{stock_info['ticker']}</span><br>
                <span style="font-size: 0.85rem; color: #3b82f6; font-weight: 500;">{stock_info['reason']}</span>
            </div>
        """, unsafe_allow_html=True)

st.markdown("<br><hr style='border-color: #334155;'>", unsafe_allow_html=True)

# --- SEZIONE 2: ANALISI DETTAGLIATA ---
st.write("### 🔍 Analisi Predittiva Settimanale")
ticker_input = st.text_input("Inserisci un Ticker manualmente:", "NVDA").upper()

if ticker_input:
    with st.spinner("Elaborazione modelli AI in corso..."):
        data = engine.analyze_ticker(ticker_input)
        
    st.markdown(f"#### Target: <span style='color: #3b82f6;'>{data['name']}</span> ({data['ticker']})", unsafe_allow_html=True)
    
    # KPI Principali bicolore su sfondo scuro
    m1, m2 = st.columns(2)
    m1.metric(label="FinGPT Sentiment Score", value=f"{data['sentiment']}%")
    m2.metric(label="Previsione Statistica", value=data['prediction'])
    
    st.write("#### 📅 Strategia Operativa nei Giorni della Settimana")
    
    giorni_cols = st.columns(5)
    for idx, (giorno, dettagli) in enumerate(data['weekly'].items()):
        with giorni_cols[idx]:
            # Colore del testo in base all'azione consigliata
            if "Acquisto" in dettagli['stato']:
                color = "#10b981" # Verde smeraldo
            elif "Vendita" in dettagli['stato']:
                color = "#ef4444" # Rosso flat
            else:
                color = "#f59e0b" # Ambra
                
            st.markdown(f"""
                <div class="day-card">
                    <b style="color: #94a3b8; font-size: 1.05rem;">{giorno}</b><br>
                    <p style="color: {color}; font-weight: bold; margin: 10px 0 5px 0; font-size: 0.95rem;">{dettagli['stato']}</p>
                    <span style="color: #64748b; font-size: 0.8rem;">Affidabilità: {dettagli['confidenza']}</span>
                </div>
            """, unsafe_allow_html=True)