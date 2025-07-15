import streamlit as st
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.model_train import get_data, get_rolling_mean, get_differencing_order, evaluate_model, scaling, get_forecast, inverse_scaling
from utils.ploty_figure import plotly_table, Moving_average_forecast
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime, timedelta
import numpy as np
import requests

@st.cache_data
def search_stocks(query):
    try:
        url = f"https://query2.finance.yahoo.com/v1/finance/search?q={query}"
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=5)
        if response.status_code != 200:
            st.warning("Yahoo Finance search is temporarily unavailable. Please try again later.")
            return []
        data = response.json()
        results = data.get("quotes", [])
        stocks = [r for r in results if r.get("quoteType") == "EQUITY"]
        return [
            f"{stock['symbol']} - {stock.get('shortname', stock.get('longname', ''))}"
            for stock in stocks
        ]
    except Exception:
        st.warning("Could not fetch suggestions. Please check your internet connection or try again later.")
        return []

st.set_page_config(page_title="Stock Prediction", page_icon=":chart_with_upwards_trend:", layout="wide")



st.title("Stock Prediction")


col1, col2, col3 = st.columns(3)
with col1:
    search_query = st.text_input("Search for a stock (name or symbol)", value="TSLA")

matches = search_stocks(search_query) if search_query else []

if matches:
    selected = st.selectbox("Suggestions", matches, key="stock_suggestion")
    ticker = selected.split(" - ")[0]
    st.success(f"Selected symbol: {ticker}")
elif search_query and len(search_query) <= 6 and search_query.isalnum():
    # Accept direct symbol input if it looks like a ticker (e.g., "AAPL")
    ticker = search_query.upper()
    st.info(f"Using direct symbol: {ticker}")
else:
    st.info("Type a company name or symbol to see suggestions.")
    st.stop()

st.subheader('Predicting Next 30 days Close Price for: ' + ticker)

close_price = get_data(ticker)
rolling_price = get_rolling_mean(close_price)

# If 'Close' is not a column, rename the only column to 'Close'
if 'Close' not in rolling_price.columns:
    if len(rolling_price.columns) == 1:
        rolling_price.columns = ['Close']
    else:
        st.error(f"rolling_price columns are: {list(rolling_price.columns)}. No 'Close' column found.")
        st.stop()

differencing_order = get_differencing_order(rolling_price)
scaled_data, scaler = scaling(rolling_price)
rmse = evaluate_model(scaled_data, differencing_order)

forecast = get_forecast(scaled_data, rolling_price, differencing_order)
forecast['Close'] = inverse_scaling(scaler, forecast['Close'])

# Show forecast table
st.write('##### Forecast Data (Next 30 days)')
fig_tail = plotly_table(forecast.sort_index(ascending=True).round(3))
fig_tail.update_layout(
    height=220,
    margin=dict(l=0, r=0, t=0, b=0),
    paper_bgcolor="#f7fafd"
)
st.plotly_chart(fig_tail, use_container_width=True)

# After rolling_price and forecast are created
last_hist_date = rolling_price.index[-1]
first_forecast_date = forecast.index[0]

if (first_forecast_date - last_hist_date).days > 1:
    fill_dates = pd.date_range(start=last_hist_date + pd.Timedelta(days=1), end=first_forecast_date - pd.Timedelta(days=1), freq='D')
    fill_values = close_price.loc[fill_dates.intersection(close_price.index), 'Close']
    fill_df = pd.DataFrame({'Close': fill_values}, index=fill_values.index)
    rolling_price = pd.concat([rolling_price, fill_df])
    rolling_price = rolling_price.sort_index()

# Now combine and plot as before
combined_data = pd.concat([rolling_price, forecast])
combined_data = combined_data[~combined_data.index.duplicated(keep='first')]

# Flatten MultiIndex columns if present
if isinstance(combined_data.columns, pd.MultiIndex):
    combined_data.columns = ['_'.join([str(i) for i in col if i]) for col in combined_data.columns.values]

st.text(f"Combined data columns: {list(combined_data.columns)}")

# Use the correct column name for plotting
# Use the correct column name for plotting
for col in combined_data.columns:
    if 'Close' in str(col):
        close_col = col
        break

combined_data = combined_data.dropna(subset=[close_col])

# Debug after combining
st.write("Combined data shape:", combined_data.shape)
st.write("Combined data head:", combined_data.head())
st.write("Combined data tail:", combined_data.tail())
st.write("Any NaNs in combined:", combined_data[close_col].isna().any())

st.write("Rolling price index head:", rolling_price.index[:5])
st.write("Rolling price index tail:", rolling_price.index[-5:])
st.write("Forecast index head:", forecast.index[:5])
st.write("Forecast index tail:", forecast.index[-5:])
st.write("Rolling price columns:", list(rolling_price.columns))

# Pass the correct column to the plotting function


# After combining and deduplicating
combined_data['Close'] = pd.to_numeric(combined_data['Close'], errors='coerce')
combined_data = combined_data.dropna(subset=['Close'])

st.write("Combined data head (for plotting):", combined_data.head(10))
st.write("Combined data tail (for plotting):", combined_data.tail(10))
st.write("Combined data dtypes:", combined_data.dtypes)

# Clean up both DataFrames
rolling_price = rolling_price.copy()
forecast = forecast.copy()
rolling_price['Close'] = pd.to_numeric(rolling_price['Close'], errors='coerce')
forecast['Close'] = pd.to_numeric(forecast['Close'], errors='coerce')
rolling_price = rolling_price.dropna(subset=['Close'])
forecast = forecast.dropna(subset=['Close'])

# Plot
st.plotly_chart(Moving_average_forecast(rolling_price, forecast), use_container_width=True)

colA, colB = st.columns(2)
with colA:
    st.metric("Model RMSE", f"{rmse:.2f}")
with colB:
    last_close = float(close_price['Close'].values[-1])
    st.metric("Last Close Price", f"${last_close:.2f}")

st.markdown(
    """
    <hr style='margin-top:2em; margin-bottom:1em;'>
    <div style='text-align:center; color:#888; font-size:0.95em;'>
        Made with ❤️ using Streamlit & Plotly | <a href='https://github.com/your-repo' style='color:#0078ff;'>GitHub</a>
    </div>
    """,
    unsafe_allow_html=True
)