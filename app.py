import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas_ta as ta
import pandas as pd
from newsapi import NewsApiClient
from textblob import TextBlob
from deep_translator import GoogleTranslator

# --- 1. CONFIGURACIÓN ---
NEWS_API_KEY = 'c25a21c32be844a882214b74f02a1d47' 

st.set_page_config(page_title="Terminal Pro AI - Algorithmic Engine", layout="wide")

# --- 2. LISTA MAESTRA (S&P 100) ---
@st.cache_data
def obtener_lista_completa():
    data = {
        'Symbol': [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B', 'V', 'UNH', 
            'JNJ', 'WMT', 'JPM', 'MA', 'PG', 'AVGO', 'ORCL', 'HD', 'CVX', 'LLY', 
            'ABBV', 'KO', 'PEP', 'MRK', 'BAC', 'COST', 'ADBE', 'CSCO', 'TMO', 'ACN',
            'CRM', 'NFLX', 'ABT', 'LIN', 'DIS', 'AMD', 'WFC', 'TXN', 'PM', 'INTC',
            'CAT', 'VZ', 'INTU', 'AMGN', 'IBM', 'QCOM', 'LOW', 'PFE', 'UNP', 'SPGI',
            'GS', 'HON', 'RTX', 'T', 'GE', 'BLK', 'ISRG', 'BA', 'AXP', 'ELV',
            'AMAT', 'SYK', 'MDLZ', 'BKNG', 'GILD', 'ADI', 'TJX', 'ADP', 'C', 'LRCX',
            'VRTX', 'MMC', 'REGN', 'ETN', 'PANW', 'BSX', 'BMY', 'PLD', 'CVS', 'LMT',
            'CB', 'ZTS', 'SBUX', 'DE', 'MU', 'CI', 'MDT', 'MO', 'FI', 'TMUS'
        ],
        'Name': [
            'Apple', 'Microsoft', 'Google', 'Amazon', 'Nvidia', 'Meta', 'Tesla', 'Berkshire', 'Visa', 'UnitedHealth',
            'J&J', 'Walmart', 'JPMorgan', 'Mastercard', 'Procter & Gamble', 'Broadcom', 'Oracle', 'Home Depot', 'Chevron', 'Eli Lilly',
            'AbbVie', 'Coca-Cola', 'PepsiCo', 'Merck', 'Bank of America', 'Costco', 'Adobe', 'Cisco', 'Thermo Fisher', 'Accenture',
            'Salesforce', 'Netflix', 'Abbott', 'Linde', 'Disney', 'AMD', 'Wells Fargo', 'Texas Inst.', 'Philip Morris', 'Intel',
            'Caterpillar', 'Verizon', 'Intuit', 'Amgen', 'IBM', 'Qualcomm', 'Lowe\'s', 'Pfizer', 'Union Pacific', 'S&P Global',
            'Goldman Sachs', 'Honeywell', 'Raytheon', 'AT&T', 'GE', 'BlackRock', 'Intuitive Surg.', 'Boeing', 'American Express', 'Elevance',
            'Applied Mat.', 'Stryker', 'Mondelez', 'Booking', 'Gilead', 'Analog Devices', 'TJX', 'ADP', 'Citigroup', 'Lam Research',
            'Vertex', 'Marsh McLennan', 'Regeneron', 'Eaton', 'Palo Alto', 'Boston Scient.', 'Bristol-Myers', 'Prologis', 'CVS Health', 'Lockheed Martin',
            'Chubb', 'Zoetis', 'Starbucks', 'Deere', 'Micron', 'Cigna', 'Medtronic', 'Altria', 'Fiserv', 'T-Mobile'
        ]
    }
    df = pd.DataFrame(data)
    df['Display'] = df['Name'] + " (" + df['Symbol'] + ")"
    return df.sort_values('Name').reset_index(drop=True)

# --- 3. LÓGICA DEL ALGORITMO ---
def calcular_score_maestro(df, noticias):
    score = 50 
    rsi = df['RSI'].iloc[-1]
    precio = df['Close'].iloc[-1]
    ema = df['EMA_20'].iloc[-1]
    
    # Análisis Técnico (RSI y EMA)
    if rsi < 35: score += 15 
    if rsi > 65: score -= 15
    if precio > ema: score += 10
    
    # Análisis de Volumen (Smart Money)
    vol_reciente = df['Volume'].tail(5).mean()
    vol_previo = df['Volume'].tail(20).head(15).mean()
    if vol_reciente > vol_previo * 1.3: score += 15

    # Análisis de Sentimiento
    sent_val = 0
    if noticias:
        for n in noticias:
            sent_val += TextBlob(n).sentiment.polarity
        score += (sent_val / len(noticias)) * 25
        
    return min(max(score, 0), 100)

# --- 4. INTERFAZ Y TEMPORALIDADES ---
st.sidebar.title("🤖 Algorithmic Engine")
df_empresas = obtener_lista_completa()

try:
    idx_def = int(df_empresas[df_empresas['Symbol'] == 'NVDA'].index[0])
except:
    idx_def = 0

seleccion = st.sidebar.selectbox("Empresa:", df_empresas['Display'].tolist(), index=idx_def)
ticker = seleccion.split('(')[-1].strip(')')

# Temporalidades solicitadas
opciones_tiempo = {
    "4 Horas": {"p": "60d", "i": "90m"},
    "1 Día": {"p": "1y", "i": "1d"},
    "1 Semana": {"p": "2y", "i": "1wk"},
    "1 Mes": {"p": "5y", "i": "1mo"},
    "1 Año": {"p": "max", "i": "1mo"},
    "5 Años": {"p": "max", "i": "1mo"}
}
temp_sel = st.sidebar.selectbox("Temporalidad:", list(opciones_tiempo.keys()), index=1)

# --- 5. MOTOR DE DATOS ---
def cargar_todo(ticker, conf):
    df = yf.download(ticker, period=conf["p"], interval=conf["i"], auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    
    # Indicadores
    df['RSI'] = ta.rsi(df['Close'], length=14)
    df['EMA_20'] = ta.ema(df['Close'], length=20)
    
    # Cálculo de Soportes y Resistencias (Puntos máximos/mínimos de 20 periodos)
    df['Resistencia'] = df['High'].rolling(window=20).max()
    df['Soporte'] = df['Low'].rolling(window=20).min()
    
    noticias_txt, noticias_display = [], []
    try:
        api = NewsApiClient(api_key=NEWS_API_KEY)
        raw = api.get_everything(q=ticker, language='en', page_size=5)
        trans = GoogleTranslator(source='en', target='es')
        for art in raw['articles']:
            noticias_txt.append(art['title'])
            noticias_display.append(f"🔹 **{trans.translate(art['title'])}**\n*({art['source']['name']})*")
    except:
        noticias_display = ["ℹ️ Feed de noticias no disponible."]
    
    return df, noticias_txt, noticias_display

df, n_raw, n_show = cargar_todo(ticker, opciones_tiempo[temp_sel])

if not df.empty:
    score = calcular_score_maestro(df, n_raw)
    
    st.title(f"🚀 Análisis Algorítmico: {seleccion}")
    
    # --- VEREDICTO ---
    col_s, col_v = st.columns([1, 3])
    col_s.metric("SCORE ALGORÍTMICO", f"{int(score)}/100")
    
    with col_v:
        if score > 70:
            st.success("🔥 VEREDICTO: COMPRA FUERTE. Alta confluencia técnica, de volumen y noticias.")
        elif score < 40:
            st.error("⚠️ VEREDICTO: RIESGO DE VENTA. Debilidad en indicadores y sentimiento negativo.")
        else:
            st.warning("⚖️ VEREDICTO: NEUTRAL. Mercado en equilibrio o consolidación lateral.")

    c1, c2 = st.columns([2, 1])
    with c1:
        # Gráfico con Subplots para Velas y Volumen
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
        
        # Velas
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Precio"), row=1, col=1)
        
        # EMA 20 (Tendencia)
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA_20'], line=dict(color='orange', width=1.5), name="EMA 20"), row=1, col=1)
        
        # Soporte y Resistencia actuales (Líneas horizontales)
        ult_soporte = df['Soporte'].iloc[-1]
        ult_resistencia = df['Resistencia'].iloc[-1]
        fig.add_hline(y=ult_soporte, line_dash="dot", line_color="green", annotation_text="Soporte", row=1, col=1)
        fig.add_hline(y=ult_resistencia, line_dash="dot", line_color="red", annotation_text="Resistencia", row=1, col=1)
        
        # Volumen
        colores_vol = ['green' if df['Close'].iloc[i] >= df['Open'].iloc[i] else 'red' for i in range(len(df))]
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name="Volumen", marker_color=colores_vol), row=2, col=1)
        
        fig.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False, height=750, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with c2:
        st.subheader("📡 Prensa Traducida")
        for n in n_show:
            st.markdown(n)
            st.divider()