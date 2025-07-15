import plotly.graph_objects as go
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
from datetime import timedelta
import streamlit as st



import plotly.graph_objects as go

def plotly_table(dataframe):
    headerColor = '#0078ff'
    rowEvenColor = '#f8fafd'
    rowOddColor = '#e1efff'
    # Prepare header and cell values
    header_values = ["<b>Index</b>"] + ["<b>" + str(col)[:10] + "</b>" for col in dataframe.columns]
    cell_values = [[str(i) for i in dataframe.index]] + [dataframe[col].tolist() for col in dataframe.columns]
    # Alternate row colors
    n_rows = len(dataframe)
    fill_colors = []
    for i in range(n_rows):
        fill_colors.append(rowEvenColor if i % 2 == 0 else rowOddColor)
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=header_values,
            line_color=headerColor,
            fill_color=headerColor,
            align='center',
            font=dict(color='black', size=15),
            height=35
        ),
        cells=dict(
            values=cell_values,
            fill_color=[fill_colors] * (len(dataframe.columns) + 1),
            align='left',
            line_color='white',
            font=dict(color="black", size=15)
        )
    )])
    fig.update_layout(height=400, margin=dict(l=0, r=0, t=0, b=0))
    return fig

def Moving_average_forecast(historical, forecast):
    fig = go.Figure()
    # Plot historical line (black)
    if len(historical) > 0:
        fig.add_trace(go.Scatter(
            x=historical.index,
            y=historical['Close'],
            mode='lines',
            name='Close Price',
            line=dict(width=3, color='black')
        ))
    # Plot forecast line (red)
    if len(forecast) > 0:
        fig.add_trace(go.Scatter(
            x=forecast.index,
            y=forecast['Close'],
            mode='lines',
            name='Future Close Price',
            line=dict(width=3, color='red')
        ))
    fig.update_xaxes(rangeslider_visible=True, color='black', tickfont=dict(color='black'), title_font=dict(color='black'))
    fig.update_yaxes(color='black', tickfont=dict(color='black'), title_font=dict(color='black'))
    fig.update_layout(
        title=dict(text="Stock Price Forecast", font=dict(color='black')),
        xaxis_title="Date",
        yaxis_title="Price",
        height=500,
        margin=dict(l=0, r=20, t=40, b=0),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='black'),
        legend=dict(
            yanchor="top",
            xanchor="right",
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='black',
            borderwidth=1,
            font=dict(color='black')
        )
    )
    return fig

def get_rolling_mean(close_price):
    """Calculate 7-day rolling mean of closing prices"""
    # Ensure we are working with a Series
    if isinstance(close_price, pd.DataFrame):
        close_price = close_price['Close']
    rolling_price = close_price.rolling(window=7).mean().dropna()
    # Only convert to DataFrame if it's a Series
    if isinstance(rolling_price, pd.Series):
        rolling_price = rolling_price.to_frame(name='Close')
    return rolling_price