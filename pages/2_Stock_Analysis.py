import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta, date # Import date
import requests

st.set_page_config(page_title="Stock Analysis", page_icon="📉", layout="wide")

st.title("Stock Technical Analysis 📉")

# Sidebar for stock selection (remains for general page context)
st.sidebar.header("Stock Selection")

st.sidebar.markdown(
    "Type a company name or symbol, then select a suggestion from the dropdown below."
)

# Default to "AAPL" in the search bar
search_query = st.sidebar.text_input("Search for a stock (name or symbol)", value="AAPL")

@st.cache_data
def search_stocks(query):
    try:
        url = f"https://query2.finance.yahoo.com/v1/finance/search?q={query}"
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=5)
        if response.status_code != 200:
            st.sidebar.warning("Yahoo Finance search is temporarily unavailable. Please try again later.")
            return []
        try:
            data = response.json()
        except Exception:
            st.sidebar.warning("No results found or service temporarily unavailable.")
            return []
        results = data.get("quotes", [])
        stocks = [r for r in results if r.get("quoteType") == "EQUITY"]
        return [
            f"{stock['symbol']} - {stock.get('shortname', stock.get('longname', ''))}"
            for stock in stocks
        ]
    except Exception:
        st.sidebar.warning("Could not fetch suggestions. Please check your internet connection or try again later.")
        return []

matches = search_stocks(search_query) if search_query else []

stocks_list = []
if matches:
    selected = st.sidebar.selectbox("Suggestions", matches, key="stock_suggestion")
    selected_symbol = selected.split(" - ")[0]
    stocks_list = [selected_symbol]
    st.sidebar.success(f"Selected symbol: {selected_symbol}")
elif search_query and len(search_query) <= 6 and search_query.isalnum():
    # Accept direct symbol input if it looks like a ticker (e.g., "AAPL")
    stocks_list = [search_query.upper()]
    st.sidebar.info(f"Using direct symbol: {search_query.upper()}")
else:
    st.sidebar.info("Type a company name or symbol to see suggestions.")
    st.sidebar.info("No stock selected.")

# Main content area
if not stocks_list:
    st.warning("Please select at least one stock from the sidebar.")
else:
    # Focus on the first selected stock for detailed view (used by Chart View)
    selected_symbol_for_chart = stocks_list[0]

    # Create tabs for different analysis types
    tab1, tab2 = st.tabs(["Chart View", "About & Details"])

    with tab1:
        # --- Existing Chart View Logic ---
        # Horizontal time period selector
        period = st.radio(
            "Select Time Period",
            ["1D", "1W", "1M", "3M", "6M", "YTD", "1Y", "2Y", "5Y", "max"],
            index=2,  # Default to 1M
            horizontal=True
        )

        # Map period string to yfinance format
        yf_period_map = {
            "1D": "1d", "1W": "7d", "1M": "1mo", "3M": "3mo", "6M": "6mo",
            "YTD": "ytd", "1Y": "1y", "2Y": "2y", "5Y": "5y", "max": "max"
        }
        yf_period = yf_period_map.get(period, "1mo") # Default to 1mo if not found

        @st.cache_data
        def load_chart_data(symbol, period):
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period)
            info = ticker.info
            return hist, info

        try:
            hist_data, stock_info = load_chart_data(selected_symbol_for_chart, yf_period)
            
            if hist_data.empty:
                st.error(f"No data found for {selected_symbol_for_chart} for the selected period.")
            else:
                # Header: Stock Symbol and Company Name
                st.markdown(f"## {selected_symbol_for_chart} {stock_info.get('longName', '')}")

                # Current Price and Change
                current_price = hist_data['Close'][-1]
                previous_close = hist_data['Close'][-2] if len(hist_data) > 1 else current_price
                price_change = current_price - previous_close
                percent_change = (price_change / previous_close) * 100 if previous_close != 0 else 0
                
                # Determine color and arrow
                if percent_change < 0:
                    pill_color = "#faeaea"      # Light red background
                    pill_text_color = "#b71c1c" # Red text inside pill
                    arrow = "&#8595;"           # Down arrow
                else:
                    pill_color = "#eafaea"      # Light green background
                    pill_text_color = "#1b5e20" # Green text inside pill
                    arrow = "&#8593;"   

                st.markdown(f"""
                <div style="display: flex; align-items: center; gap: 1rem;">
                    <span style="font-size: 2.5em; font-weight: bold;">${current_price:.2f}</span>
                    <span style="
                        background: {pill_color};
                        color: {pill_text_color};
                        border-radius: 1em;
                        padding: 0.2em 0.8em;
                        font-weight: 600;
                        font-size: 1.2em;
                        display: flex;
                        align-items: center;
                        gap: 0.3em;
                    ">
                        {arrow} {abs(percent_change):.2f}%
                    </span>
                    <span style="color: {pill_text_color}; font-size: 1.2em; font-weight: 500;">
                        {price_change:.2f} {period}
                    </span>
                </div>
                """, unsafe_allow_html=True)
                st.markdown(f"Past {period}")
                
                exchange_name = stock_info.get('exchange', '')
                display_exchange = 'NMS' if 'Nasdaq' in exchange_name else exchange_name
                st.markdown(f"{display_exchange} \u00B7 {stock_info.get('currency', 'USD')}")

                # Chart Type Selector
                chart_type = st.radio(
                    "Select Chart Type",
                    ["Line Chart", "Candlestick Chart"],
                    index=0, # Default to Line Chart
                    horizontal=True
                )

                # Main Price Chart
                fig = go.Figure()
                if chart_type == "Line Chart":
                    fig.add_trace(go.Scatter(
                        x=hist_data.index,
                        y=hist_data['Close'],
                        mode='lines',
                        name=selected_symbol_for_chart,
                        line=dict(color='red' if price_change < 0 else 'green'),
                        hovertemplate=
                            "<b>Date:</b> %{x|%m/%d %I:%M %p}<br>" +
                            "<b>Close:</b> %{y:.2f}<br>" +
                            "<b>Open:</b> %{customdata[0]:.2f}<br>" +
                            "<b>High:</b> %{customdata[1]:.2f}<br>" +
                            "<b>Low:</b> %{customdata[2]:.2f}<br>" +
                            "<b>Volume:</b> %{customdata[3]:,}<br>" +
                            "<extra></extra>",
                        customdata=np.stack([
                            hist_data['Open'],
                            hist_data['High'],
                            hist_data['Low'],
                            hist_data['Volume']
                        ], axis=-1)
                    ))
                elif chart_type == "Candlestick Chart":
                    fig.add_trace(go.Candlestick(
                        x=hist_data.index,
                        open=hist_data['Open'],
                        high=hist_data['High'],
                        low=hist_data['Low'],
                        close=hist_data['Close'],
                        name="candle",
                        increasing_line_color='#4CAF50',      # Light green
                        increasing_fillcolor='#4CAF50',      # Very light green
                        decreasing_line_color='#E53935',     # Bright red
                        decreasing_fillcolor='#E53935',      # Light red
                        showlegend=True
                    ))
                    # Overlay a line (e.g., 10-period moving average)
                    fig.add_trace(go.Scatter(
                        x=hist_data.index,
                        y=hist_data['Close'].rolling(window=10).mean(),
                        mode='lines',
                        name="line",
                        line=dict(color='#4FC3F7', width=0.5),  # Light blue and thin
                        showlegend=True
                    ))
                
                fig.update_layout(
                    title=f"{selected_symbol_for_chart} Stock Price ({chart_type})",
                    yaxis_title="Stock Price (USD)",
                    xaxis_title="Date",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)

                # Generate a professional chart summary
                close_prices = hist_data['Close']
                high = hist_data['High'].max()
                low = hist_data['Low'].min()
                avg = close_prices.mean()
                std = close_prices.std()
                trend = "an uptrend" if close_prices[-1] > close_prices[0] else "a downtrend"
                volatility = "high" if std / avg > 0.03 else "moderate" if std / avg > 0.015 else "low"

                summary_md = (
                    "### Chart Summary\n\n"
                    f"- **Trend:** {selected_symbol_for_chart} has experienced {trend} over the selected period.\n"
                    f"- **Opening price:** ${close_prices[0]:,.2f}\n"
                    f"- **Closing price:** ${close_prices[-1]:,.2f}\n"
                    f"- **Overall change:** {'Gain' if close_prices[-1] > close_prices[0] else 'Loss'} of {abs((close_prices[-1] - close_prices[0]) / close_prices[0] * 100):.2f}%\n"
                    f"- **Highest price:** ${high:,.2f}\n"
                    f"- **Lowest price:** ${low:,.2f}\n"
                    f"- **Average closing price:** ${avg:,.2f}\n"
                    f"- **Price volatility:** {volatility} (std dev: {std:.2f})"
                )
                st.markdown(summary_md)

                st.markdown("---")
                st.subheader(f"Research Analysis: {selected_symbol_for_chart}")

                col1, col2, col3, col4 = st.columns(4, gap="medium")

                # 1. Earnings Per Share (EPS) - Visual Card
                with col1:
                    with st.container():
                        st.markdown("**Earnings Per Share**")
                        # Try to get the latest EPS estimate from yfinance
                        eps_estimate = stock_info.get('earningsEstimate', None)
                        if eps_estimate is None:
                            # Fallback: try trailing EPS
                            eps_estimate = stock_info.get('trailingEps', None)

                        if eps_estimate is not None and isinstance(eps_estimate, (int, float)):
                            eps_estimate_str = f"Estimate: {eps_estimate:+.2f}"
                        else:
                            eps_estimate_str = "Estimate: N/A"

                        # You can still use mock data for the chart, or use real if available
                        eps_data = pd.DataFrame({
                            "Quarter": ["Q3'24", "Q4'24", "Q1'25", "Q2'25"],
                            "EPS": [1.42, 1.38, 1.45, 1.50]
                        })
                        eps_fig = go.Figure(go.Bar(
                            x=eps_data["Quarter"],
                            y=eps_data["EPS"],
                            marker_color="#90caf9"
                        ))
                        eps_fig.update_layout(
                            height=250, width=250, margin=dict(l=0, r=0, t=30, b=0),
                            yaxis=dict(range=[0, 1.7]), showlegend=False,
                            plot_bgcolor="#23272f", paper_bgcolor="#23272f"
                        )
                        st.plotly_chart(eps_fig, use_container_width=True)
                        st.markdown(
                            f"<div style='text-align:center; color:#4caf50; font-size:1.2em; font-weight:bold;'>{eps_estimate_str}</div>",
                            unsafe_allow_html=True
                        )
                        st.markdown("&nbsp;", unsafe_allow_html=True)

                # 2. Revenue vs. Earnings (Bar Chart)
                with col2:
                    with st.container():
                        st.markdown("**Revenue vs. Earnings**")
                        df = pd.DataFrame({
                            "Revenue": [95.36, 98.0, 120.0, 100.0],
                            "Earnings": [24.78, 25.0, 30.0, 22.0]
                        }, index=["Q2'24", "Q3'24", "Q4'24", "Q1'25"])
                        rev_fig = go.Figure()
                        rev_fig.add_bar(name="Revenue", x=df.index, y=df["Revenue"], marker_color="#90caf9")
                        rev_fig.add_bar(name="Earnings", x=df.index, y=df["Earnings"], marker_color="#1976d2")
                        rev_fig.update_layout(
                            barmode='group', height=250, width=250, margin=dict(l=0, r=0, t=30, b=0),
                            plot_bgcolor="#23272f", paper_bgcolor="#23272f"
                        )
                        st.plotly_chart(rev_fig, use_container_width=True)
                        st.markdown("&nbsp;", unsafe_allow_html=True)

                # 3. Analyst Recommendations (Stacked Bar)
                with col3:
                    with st.container():
                        st.markdown("**Analyst Recommendations**")
                        rec_data = {
                            "Month": ["Mar", "Apr", "May", "Jun"],
                            "Strong Buy": [7, 7, 7, 7],
                            "Buy": [21, 23, 21, 20],
                            "Hold": [14, 16, 16, 17],
                            "Underperform": [2, 2, 2, 2],
                            "Sell": [1, 0, 1, 0]
                        }
                        rec_fig = go.Figure()
                        rec_fig.add_bar(name="Strong Buy", x=rec_data["Month"], y=rec_data["Strong Buy"], marker_color="green")
                        rec_fig.add_bar(name="Buy", x=rec_data["Month"], y=rec_data["Buy"], marker_color="lightgreen")
                        rec_fig.add_bar(name="Hold", x=rec_data["Month"], y=rec_data["Hold"], marker_color="gold")
                        rec_fig.add_bar(name="Underperform", x=rec_data["Month"], y=rec_data["Underperform"], marker_color="orange")
                        rec_fig.add_bar(name="Sell", x=rec_data["Month"], y=rec_data["Sell"], marker_color="red")
                        rec_fig.update_layout(
                            barmode='stack', height=250, width=250, margin=dict(l=0, r=0, t=30, b=0),
                            plot_bgcolor="#23272f", paper_bgcolor="#23272f", showlegend=False
                        )
                        st.plotly_chart(rec_fig, use_container_width=True)
                        st.markdown("&nbsp;", unsafe_allow_html=True)

                # 4. Analyst Price Targets (Gauge)
                with col4:
                    with st.container():
                        st.markdown("**Analyst Price Targets**")
                        price_target_fig = go.Figure(go.Indicator(
                            mode = "gauge+number+delta",
                            value = current_price,
                            delta = {'reference': 228.85, 'increasing': {'color': "green"}, 'decreasing': {'color': "red"}},
                            gauge = {
                                'axis': {'range': [170, 300]},
                                'steps' : [
                                    {'range': [170, 201], 'color': "#e0e0e0"},
                                    {'range': [201, 228.85], 'color': "#b0e0e6"},
                                    {'range': [228.85, 300], 'color': "#c8e6c9"}
                                ],
                                'threshold' : {'line': {'color': "blue", 'width': 4}, 'thickness': 0.75, 'value': 228.85}
                            },
                            title = {'text': "Current vs. Target"}
                        ))
                        price_target_fig.update_layout(height=250, width=250, margin=dict(l=0, r=0, t=40, b=0),
                                                      paper_bgcolor="#23272f", plot_bgcolor="#23272f")
                        st.plotly_chart(price_target_fig, use_container_width=True)
                        st.markdown("&nbsp;", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error loading chart data for {selected_symbol_for_chart}. Please check the stock symbol and try again.")
            st.error(str(e))

    with tab2:
        # --- New Summary & Details View Logic ---
        st.subheader("Detailed Stock Summary")

        today = date.today()
        col1, col2, col3 = st.columns(3)
        
        with col1:
            ticker = st.text_input("Stock Ticker", selected_symbol_for_chart).upper() # Default to Chart View selection

        with col2:
            start_date = st.date_input("Choose Start Date", date(today.year - 1, today.month, today.day))
        with col3:
            end_date = st.date_input("Choose End Date", today)
        
        if ticker:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info

                st.subheader(ticker)

                st.write(info.get('longBusinessSummary', 'N/A'))

                st.write("**Sector:**", info.get('sector', 'N/A'))
                st.write("**Full Time Employees:**", info.get('fullTimeEmployees', 'N/A'))
                st.write("**Website:**", info.get('website', 'N/A'))

                st.markdown("### Key Financial Ratios")
                col1, col2 = st.columns(2)
                
                # Handle potential missing keys gracefully
                market_cap = info.get('marketCap', 'N/A')
                beta_val = info.get('beta', 'N/A')
                eps = info.get('trailingEps', 'N/A')
                pe_ratio = info.get('trailingPE', 'N/A')

                with col1:
                    st.metric("Market Cap", f"${market_cap:,}" if isinstance(market_cap, (int, float)) else market_cap)
                    st.metric("EPS", f"{eps:.2f}" if isinstance(eps, (int, float)) else eps)

                with col2:
                    st.metric("Beta", f"{beta_val:.2f}" if isinstance(beta_val, (int, float)) else beta_val)
                    st.metric("P/E Ratio", f"{pe_ratio:.2f}" if isinstance(pe_ratio, (int, float)) else pe_ratio)

                # Example: Replace these with your real values from stock_info
                ratios = {
                    "Market Cap": market_cap,
                    "Beta": beta_val,
                    "EPS": eps,
                    "PE Ratio": pe_ratio,
                    "Quick Ratio": info.get('quickRatio', 'N/A'),
                    "Revenue per share": info.get('revenuePerShare', 'N/A'),
                    "Profit Margins": info.get('profitMargins', 'N/A'),
                    "Debt to Equity": info.get('debtToEquity', 'N/A'),
                    "Return on Equity": info.get('returnOnEquity', 'N/A'),
                }

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("""
                    <table style="width:100%; border-radius:8px; overflow:hidden;">
                        <tr style="background:#1976d2; color:white;">
                            <th colspan="2" style="padding:8px;"></th>
                        </tr>
                        <tr style="background:#111; color:white;"><td>Market Cap</td><td>{}</td></tr>
                        <tr style="background:#fff; color:black;"><td>Beta</td><td>{}</td></tr>
                        <tr style="background:#f5faff; color:black;"><td>EPS</td><td>{}</td></tr>
                        <tr style="background:#fff; color:black;"><td>PE Ratio</td><td>{}</td></tr>
                    </table>
                    """.format(
                        ratios["Market Cap"], ratios["Beta"], ratios["EPS"], ratios["PE Ratio"]
                    ), unsafe_allow_html=True)

                with col2:
                    st.markdown("""
                    <table style="width:100%; border-radius:8px; overflow:hidden;">
                        <tr style="background:#1976d2; color:white;">
                            <th colspan="2" style="padding:8px;">&nbsp;</th>
                        </tr>
                        <tr style="background:#111; color:white;"><td>Quick Ratio</td><td>{}</td></tr>
                        <tr style="background:#fff; color:black;"><td>Revenue per share</td><td>{}</td></tr>
                        <tr style="background:#f5faff; color:black;"><td>Profit Margins</td><td>{}</td></tr>
                        <tr style="background:#fff; color:black;"><td>Debt to Equity</td><td>{}</td></tr>
                        <tr style="background:#f5faff; color:black;"><td>Return on Equity</td><td>{}</td></tr>
                    </table>
                    """.format(
                        ratios["Quick Ratio"], ratios["Revenue per share"], ratios["Profit Margins"], ratios["Debt to Equity"], ratios["Return on Equity"]
                    ), unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error loading data for {ticker}. Please check the ticker symbol or selected dates.")
                st.error(str(e)) 


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