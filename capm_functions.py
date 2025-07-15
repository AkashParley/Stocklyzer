import plotly.express as px
import pandas as pd
import numpy as np


# functions to plort interactive ploty charts
def interactive_plot(df):
    fig = px.line()
    for i in df.columns[1:]:
        fig.add_scatter(x = df['Date'],y = df[i], name = i)
    fig.update_layout(width = 450, margin = dict(l=20, r =20, t=50 , b=20), legend =dict(orientation = 'h', yanchor ='bottom' , y = 1.02 , xanchor = 'right' , x = 1 ,))
    return fig


#fuction to normalize the prices based on the initial price
def normalize(df_2):
    df = df_2.copy()
    for i in df.columns[1:]:
        df[i] = df[i]/df[i][0]
    return df


#fuction to calulate daily return
def daily_return(df):
    df_daily_return = df.copy()
    # Calculate daily returns for all columns except 'Date'
    for column in df.columns[1:]:
        df_daily_return[column] = df[column].pct_change() * 100
    # Set first row to 0 for all columns except 'Date'
    df_daily_return.iloc[0, 1:] = 0
    return df_daily_return


#fuction to calculate beta
def calculate_beta(stocks_daily_return , stocks):
    rm = stocks_daily_return['GSPC'].mean()*252

    b,a = np.polyfit(stocks_daily_return['GSPC'], stocks_daily_return[stocks], 1)
    return b,a


