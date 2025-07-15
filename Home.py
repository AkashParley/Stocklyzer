import streamlit as st

st.set_page_config(
    page_title="Stock Analysis Dashboard",
    page_icon="📈",
    layout="wide"
)
import streamlit as st

# Text logo using Ma
st.markdown(
    """
    <div style='background: linear-gradient(90deg, #23272f 0%, #2c313a 100%); padding: 1.5rem 1rem; border-radius: 12px; margin-bottom: 2rem; position: relative; overflow: hidden;'>
         <h1 style='text-align: center; color: #F5F5F5;'>📈 Stocklyzer</h1>
         <h4 style='text-align: center; color: #00B894;'>Analyze | Predict | Grow</h4>
        <div style="width:100%;overflow:hidden;position:relative;height:2.2em;">
            <div style="
                color: #b0b8c1;
                font-size: 1.2em;
                margin-top: 0.5em;
                white-space: nowrap;
                position: absolute;
                will-change: transform;
                animation: scroll-left 12s linear infinite;
            ">
                Forecast the future. Analyze the past. Make smarter decisions.
            </div>
        </div>
    </div>
    <style>
    @keyframes scroll-left {
        0% { left: 100%; }
        100% { left: -100%; }
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("""
### Available Tools:
1. **CAPM Calculator** 📊
   - Calculate Beta and Expected Returns using CAPM
   - Analyze stock performance against market
   - View interactive price charts

2. **Stock Analysis** 📉
   - Technical indicators
   - Price trends
   - Volume analysis

3. **Stock Prediction** 📈

   - Predict the next 30 days of stock prices using ARIMA models and visualize the results with interactive charts. Enter a stock ticker or search by company name in the Stock Prediction page to get started!

### How to Use:
Select a tool from the sidebar to get started with your analysis.
""")


# Add some styling
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Add a footer

st.markdown(
    """
    <hr style='margin-top:2em; margin-bottom:1em; border-color:#444;'>
    <div style='text-align:center; color:#888; font-size:0.95em;'>
        Made with ❤️  by PARLE | <a href='https://github.com/AkashParley/Stocklyzer' style='color:#4F8BF9;'>GitHub</a>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
    [data-testid="stSidebar"]::after {
        content: "Made by Parle";
        display: block;
        position: fixed;
        left: 0;
        bottom: 0;
        width: 15rem;
        background: #23272f;
        color: #e0e0e0;
        text-align: center;
        padding: 0.8em 0;
        font-size: 1em;
        border-top: 1px solid #444;
        z-index: 100;
        letter-spacing: 0.5px;
    }
    </style>
    """,
    unsafe_allow_html=True
)