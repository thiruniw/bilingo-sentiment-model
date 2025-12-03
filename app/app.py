# app.py
import streamlit as st
import sys, os
import warnings
from datetime import datetime
import pandas as pd

# Suppress all warnings before imports
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_PATH = os.path.join(CURRENT_DIR, "..", "src")
sys.path.append(SRC_PATH)

from predict import load_artifacts, predict_sentiment
# Streamlit Page Config
st.set_page_config(
    page_title="Bilingo Sentiment Analysis",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Enhanced CSS for professional styling with TOP NAV BAR
st.markdown("""
<style>
/* Force white background everywhere */
.stApp {
    background-color: #ffffff !important;
}

body {
    background-color: #ffffff !important;
}

.main {
    background-color: #ffffff !important;
    padding: 0 !important;
    margin: 0 !important;
}

[data-testid="stAppViewContainer"] {
    background-color: #ffffff !important;
    padding-top: 0 !important;
}

[data-testid="stHeader"] {
    background-color: #ffffff !important;
}

/* TOP NAVIGATION BAR */
.top-nav {
    background: linear-gradient(135deg, #1a237e 0%, #3949ab 100%);
    padding: 20px 50px;
    box-shadow: 0 4px 12px rgba(26, 35, 126, 0.3);
    position: sticky;
    top: 0;
    z-index: 999;
    margin: 0;
    width: 100%;
}

.nav-brand {
    color: white !important;
    font-size: 32px;
    font-weight: 800;
    letter-spacing: 2px;
    text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
    margin: 0;
    padding: 0;
}

/* Hide default streamlit header */
header[data-testid="stHeader"] {
    display: none;
}

/* Main content area */
.content-area {
    padding: 40px 50px;
    max-width: 1400px;
    margin: 0 auto;
}

/* Title styling */
h1 {
    color: #000000 !important;
    font-size: 42px !important;
    font-weight: 700 !important;
    text-align: center !important;
    margin-bottom: 10px !important;
}

h2 {
    color: #000000 !important;
    font-size: 32px !important;
    font-weight: 600 !important;
    margin-top: 30px !important;
}

h3 {
    color: #000000 !important;
    font-size: 24px !important;
    font-weight: 600 !important;
}

p, div, span, label {
    color: #000000 !important;
}

/* Subtitle */
.subtitle {
    color: #000000 !important;
    font-size: 18px;
    text-align: center;
    margin-bottom: 40px;
    font-weight: 400;
}

/* Labels */
label {
    color: #000000 !important;
    font-size: 18px !important;
    font-weight: 600 !important;
}

/* Text area styling */
.stTextArea>div>div>textarea {
    font-size: 18px !important;
    color: #000000 !important;
    background-color: #ffffff !important;
    border: 3px solid #1a237e !important;
    border-radius: 12px !important;
    padding: 15px !important;
    min-height: 150px !important;
}

.stTextArea>div>div>textarea::placeholder {
    color: #666666 !important;
}

.stTextArea>div>div>textarea:focus {
    border-color: #3949ab !important;
    box-shadow: 0 0 0 2px rgba(26, 35, 126, 0.2) !important;
}

/* Navigation buttons styling */
.nav-buttons {
    margin-top: -70px;
    margin-bottom: 20px;
    display: flex;
    justify-content: flex-end;
    padding: 0 50px;
}

/* Button styling */
.stButton>button {
    background: rgba(255, 255, 255, 0.9) !important;
    color: #1a237e !important;
    font-size: 15px !important;
    font-weight: 600 !important;
    padding: 10px 24px !important;
    border-radius: 8px !important;
    border: 2px solid white !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2) !important;
}

.stButton>button:hover {
    background: white !important;
    color: #0d47a1 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3) !important;
}

/* Analyze button - different style */
.analyze-btn button {
    background: linear-gradient(135deg, #1a237e 0%, #3949ab 100%) !important;
    color: white !important;
    font-size: 18px !important;
    padding: 14px 40px !important;
}

.analyze-btn button:hover {
    background: linear-gradient(135deg, #0d47a1 0%, #1976d2 100%) !important;
}

/* Result container */
.result-container {
    background-color: #ffffff;
    border: 3px solid #1a237e;
    border-radius: 16px;
    padding: 30px;
    margin-top: 30px;
    box-shadow: 0 8px 24px rgba(26, 35, 126, 0.15);
    text-align: center;
}

.result-label {
    color: #000000 !important;
    font-size: 24px;
    font-weight: 600;
    margin-bottom: 15px;
}

.result-sentiment {
    font-size: 42px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 2px;
    padding: 20px;
    border-radius: 12px;
    margin-top: 10px;
}

.positive-sentiment {
    color: #1b5e20 !important;
    background-color: #e8f5e9;
    border: 2px solid #4caf50;
}

.negative-sentiment {
    color: #b71c1c !important;
    background-color: #ffebee;
    border: 2px solid #f44336;
}

.neutral-sentiment {
    color: #e65100 !important;
    background-color: #fff3e0;
    border: 2px solid #ff9800;
}

/* History item */
.history-item {
    background-color: #f8f9fa;
    padding: 20px;
    border-radius: 12px;
    margin-bottom: 15px;
    border-left: 5px solid #1a237e;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.history-text {
    color: #000000 !important;
    font-size: 16px;
    margin-bottom: 10px;
}

.history-meta {
    color: #000000 !important;
    font-size: 14px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

/* Stats */
.stat-box {
    background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
    padding: 25px;
    border-radius: 12px;
    text-align: center;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}

.stat-number {
    font-size: 42px;
    font-weight: 700;
    color: #1a237e !important;
}

.stat-label {
    font-size: 16px;
    color: #000000 !important;
    margin-top: 8px;
}

/* Info box */
.info-box {
    background-color: #e3f2fd;
    padding: 20px;
    border-radius: 12px;
    border-left: 5px solid #1a237e;
    margin: 20px 0;
    
}

.info-box p {
    color: #000000 !important;
    margin: 5px 0;
    text-align: center;
}

/* Metric override */
[data-testid="stMetricValue"] {
    color: #000000 !important;
}

[data-testid="stMetricLabel"] {
    color: #000000 !important;
}
            
.nav-btn-group {
    display: flex;
    align-items: right;
    gap: 15px;            /* spacing between buttons */
    margin-left:1420px;
}

.nav-btn-group button {
    background: white !important;
    color: #1a237e !important;
    border: none !important;
    padding: 10px 20px !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 15px !important;
    cursor: pointer;
    box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    transition: 0.2s ease;
}

.nav-btn-group button:hover {
    background: #e3e3e3 !important;
}


</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Home"


# ======================================================
# TOP NAV BAR WITH BUTTONS MOVED *INSIDE* THE BLUE BAR
# ======================================================
st.markdown("""
<div class="top-nav">
    <div class="nav-brand">📝 BILINGO SENTIMENT ANALYZER</div>
</div>
</div>
""", unsafe_allow_html=True)

# Sync URL buttons
page = st.query_params.get("page")
if page:
    st.session_state.current_page = page

# =======================
# LOAD MODEL
# =======================
@st.cache_resource
def load_model():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return load_artifacts()

st.markdown('<div class="content-area">', unsafe_allow_html=True)

try:
    tokenizer, model, label_encoder = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# ============================================================================
# HOME PAGE
# ============================================================================
if st.session_state.current_page == "Home":
    st.title("🏠 Sentiment Analysis")
    st.markdown("<p class='subtitle'>Enter text in Sinhala or English to detect sentiment</p>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        user_input = st.text_area(
            "Enter your text:",
            placeholder="Type your text here...",
            height=200
        )
        
        st.markdown('<div class="analyze-btn">', unsafe_allow_html=True)
        analyze_button = st.button("🔍 Analyze Sentiment", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    
    with col2:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("### 💡 Quick Guide")
        st.markdown("""
        **Supported:**
        - Sinhala text
        - English text
        
        **Tips:**
        - Use complete sentences
        - Keep under 128 words
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Prediction
    if analyze_button:
        if user_input.strip() != "":
            with st.spinner("Analyzing..."):
                sentiment = predict_sentiment(user_input, tokenizer, model, label_encoder, show_details=False)
            
            # Add to history
            st.session_state.history.insert(0, {
                'text': user_input,
                'sentiment': sentiment,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            
            # Display result
            sentiment_lower = sentiment.lower()
            
            if sentiment_lower == "positive":
                sentiment_class = "positive-sentiment"
                emoji = "😊"
            elif sentiment_lower == "negative":
                sentiment_class = "negative-sentiment"
                emoji = "😞"
            else:
                sentiment_class = "neutral-sentiment"
                emoji = "😐"
            
            st.markdown(f"""
            <div class='result-container'>
                <div class='result-label'>Predicted Sentiment:</div>
                <div class='result-sentiment {sentiment_class}'>
                    {emoji} {sentiment.upper()}
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("⚠️ Please enter some text to analyze!")
    
   