import streamlit as st
import pandas as pd
import yfinance as yf
import datetime
import capm_functions

st.set_page_config(page_title="CAPM", page_icon=":chart_with_upwards_trend:",layout="wide")

st.markdown("""
<style>
.tremble {
  display: inline-block;
  animation: tremble 0.7s infinite;
}
@keyframes tremble {
  0% { transform: translateX(0); }
  20% { transform: translateX(-2px); }
  40% { transform: translateX(2px); }
  60% { transform: translateX(-2px); }
  80% { transform: translateX(2px); }
  100% { transform: translateX(0); }
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
# CAPM Calculator: Smarter Investing Starts Here 🚀

Welcome to your professional-grade **CAPM (Capital Asset Pricing Model) Calculator**.

---

### Why Use CAPM?

- **Objective Analysis:** Quantify the expected return for any stock based on its risk.
- **Risk-Reward Clarity:** Instantly see if a stock is offering enough return for its volatility.
- **Smart Comparisons:** Evaluate all your investment options on a level playing field.

---

### How Does It Work?

We use the CAPM formula to estimate a stock's fair return:
- **Risk-Free Rate:** The baseline return from ultra-safe investments (like government bonds).
- **Beta:** How much the stock moves compared to the overall market.
- **Market Return:** The average return of the market as a whole.

**CAPM Expected Return = Risk-Free Rate + Beta × (Market Return − Risk-Free Rate)**

---

### What Can You Do Here?

- **Select any stock** and see its risk (Beta) and expected return.
- **Compare** the CAPM return to the stock's actual performance.
- **Spot opportunities:** Find undervalued stocks or avoid overhyped risks.

---

### Example

> **Stock:** AlphaTech  
> **Beta:** 1.4  
> **CAPM Expected Return:** 12.2%  
> **Current Avg Return:** 9.0%  
>  
> _AlphaTech is underperforming for its risk. Consider analyzing further before investing._

---

<span class="tremble" style="font-weight:bold; font-size:1.1em;">
Ready to make smarter, data-driven investment decisions? Scroll down and start analyzing!
</span>
""", unsafe_allow_html=True)

st.title("Capital Asset Pricing Model")

#getting input from user
col1, col2 = st.columns(2)

with col1:
    stocks_list = st.multiselect("Select the stocks",("TSLA","AAPL","NFLX","MSFT","WIPRO","INFY","RELIANCE","AMZN","NVDA"),["TSLA","AAPL","NFLX","MSFT"])
with col2:
    years = st.number_input("Number of years",1,10)

#downloading data for SP500
try: 
    end = datetime.date.today()
    start = datetime.date(datetime.date.today().year-years, datetime.date.today().month, datetime.date.today().day)
    SP500 = yf.download("^GSPC", start=start, end=end)

    stocks_df = pd.DataFrame()

    for stock in stocks_list:
        data = yf.download(stock, start=start, end=end)
        stocks_df[f'{stock}'] = data['Close']

    stocks_df.reset_index(inplace=True)
    SP500.reset_index(inplace=True)

    # Get only the Close price column from SP500
    SP500 = SP500[['Date', 'Close']].copy()
    SP500.columns = ['Date', 'GSPC']
    stocks_df['Date'] = pd.to_datetime(stocks_df['Date'])
    SP500['Date'] = pd.to_datetime(SP500['Date'])

    # Merge the dataframes
    stocks_df = pd.merge(stocks_df, SP500, on='Date', how='inner')

    #for displaying 
    col1, col2 = st.columns([1,1])
    with col1:
        st.markdown("### Opening Price")
        st.dataframe(stocks_df.head(),use_container_width=True)
    with col2:
        st.markdown("### Closing Price")
        st.dataframe(stocks_df.tail(),use_container_width=True)

    col1, col2 = st.columns([1,1])
    with col1:
        st.markdown("### Price of all the Stocks")
        st.plotly_chart(capm_functions.interactive_plot(stocks_df))

    #plotting after normlization
    with col2:
        st.markdown("### Price of all the Stocks(After Normalization)")
        st.plotly_chart(capm_functions.interactive_plot(capm_functions.normalize(stocks_df)))

    stocks_daily_return = capm_functions.daily_return(stocks_df)

    beta = {}
    alpha = {}

    for i in stocks_daily_return.columns:
        if i !='Date' and i !='GSPC':
            b, a = capm_functions.calculate_beta(stocks_daily_return,i)
            beta[i] = b
            alpha[i] = a

    beta_df = pd.DataFrame(columns=['Stocks', 'Beta Value'])
    beta_df['Stocks'] = beta.keys()
    beta_df['Beta Value'] = [str(round(i,2)) for i in beta.values()]

    rf = 0
    rm = stocks_daily_return['GSPC'].mean() * 252

    return_df = pd.DataFrame()
    return_value = []
    for i in stocks_daily_return.columns:
        if i != 'Date' and i != 'GSPC':
            return_value.append(str(round(rf + beta[i] * (rm - rf), 2)))
    return_df['Stock'] = stocks_list
    return_df['Return Value'] = return_value

    # Create two columns for side-by-side display of Beta and Returns
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('### Calculated Beta Value')
        st.dataframe(beta_df, use_container_width=True)
    with col2:
        st.markdown("### Calculated Return using CAPM")
        st.dataframe(return_df, use_container_width=True)

except:
    st.write("Please check the stock symbols and try again.") 

