# app.py
import streamlit as st
import sys, os
sys.path.append(os.path.abspath("../src"))
from predict import predict_sentiment

# Streamlit Page Config
st.set_page_config(
    page_title="Bilingo Sentiment Analysis",
    page_icon="📝",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Enhanced CSS for professional styling
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
    padding: 2rem;
}

[data-testid="stAppViewContainer"] {
    background-color: #ffffff !important;
}

[data-testid="stHeader"] {
    background-color: #ffffff !important;
}

/* Title styling */
h1 {
    color: #0066cc !important;
    font-size: 48px !important;
    font-weight: 700 !important;
    text-align: center !important;
    margin-bottom: 10px !important;
    letter-spacing: 1px !important;
}

.main h1 {
    color: #0066cc !important;
}

[data-testid="stMarkdownContainer"] h1 {
    color: #0066cc !important;
}

/* Subtitle/description text */
.subtitle {
    color: #1a237e;
    font-size: 20px;
    text-align: center;
    margin-bottom: 40px;
    font-weight: 500;
}

/* Labels */
label {
    color: #1a237e !important;
    font-size: 22px !important;
    font-weight: 600 !important;
    margin-bottom: 10px !important;
}

/* Text area styling */
.stTextArea>div>div>textarea {
    font-size: 20px !important;
    color: #000000 !important;
    background-color: #ffffff !important;
    border: 3px solid #1a237e !important;
    border-radius: 12px !important;
    padding: 15px !important;
    min-height: 150px !important;
}

.stTextArea>div>div>textarea::placeholder {
    color: #757575 !important;
    opacity: 1 !important;
}

.stTextArea>div>div>textarea:focus {
    border-color: #3949ab !important;
    box-shadow: 0 0 0 2px rgba(26, 35, 126, 0.2) !important;
    background-color: #ffffff !important;
    color: #000000 !important;
}

/* Button styling */
.stButton>button {
    background: linear-gradient(135deg, #1a237e 0%, #3949ab 100%);
    color: white;
    font-size: 22px;
    font-weight: 600;
    padding: 16px 48px;
    border-radius: 12px;
    border: none;
    width: 100%;
    margin-top: 20px;
    transition: all 0.3s ease;
    box-shadow: 0 4px 12px rgba(26, 35, 126, 0.3);
    text-transform: uppercase;
    letter-spacing: 1.5px;
}

.stButton>button:hover {
    background: linear-gradient(135deg, #0d47a1 0%, #1976d2 100%);
    box-shadow: 0 6px 20px rgba(26, 35, 126, 0.4);
    transform: translateY(-2px);
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
    color: #1a237e;
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
    color: #1b5e20;
    background-color: #e8f5e9;
    border: 2px solid #4caf50;
}

.negative-sentiment {
    color: #b71c1c;
    background-color: #ffebee;
    border: 2px solid #f44336;
}

.neutral-sentiment {
    color: #e65100;
    background-color: #fff3e0;
    border: 2px solid #ff9800;
}

/* Warning styling */
.stWarning {
    background-color: #fff3cd;
    color: #856404;
    font-size: 18px;
    padding: 15px;
    border-radius: 8px;
    border-left: 4px solid #ffc107;
}

/* Input instruction box */
.instruction-box {
    background-color: #e3f2fd;
    padding: 20px;
    border-radius: 12px;
    margin-bottom: 25px;
    border-left: 5px solid #1a237e;
}

.instruction-text {
    color: #1a237e;
    font-size: 18px;
    font-weight: 600;
    margin: 0;
}
</style>
""", unsafe_allow_html=True)

# Title Section
st.markdown("<h1>📝 Bilingo Sentiment Analysis</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Advanced AI-powered sentiment detection for Sinhala & English text</p>", unsafe_allow_html=True)

# Instruction Box
st.markdown("""
<div class='instruction-box'>
    <p class='instruction-text'>
        💡 Enter your text below in Sinhala or English for instant sentiment analysis
    </p>
</div>
""", unsafe_allow_html=True)

# User Input
user_input = st.text_area("Enter your text here:", placeholder="Type or paste your text here...")

# Prediction Button
if st.button("🔍 Analyze Sentiment"):
    if user_input.strip() != "":
        with st.spinner("Analyzing sentiment..."):
            sentiment = predict_sentiment(user_input)
        
        # Display result with professional styling
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

# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #757575; font-size: 16px;'> Machine Learning | Bilingo AI</p>", unsafe_allow_html=True)