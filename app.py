import os
import streamlit as st
from dotenv import load_dotenv, dotenv_values
from datetime import datetime, timedelta
from predict import get_data, runner
import plotly.express as px
import plotly.graph_objs as go

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

ss = st.session_state




def info_graph(df):
    fig = px.line(df, x='Date', y='Close', title='Stock Close Prices Over Time')
    fig.update_traces(fill='tozeroy', line_color='blue')  
    fig.update_layout(
        title='Stock Close Prices Over Time',
        xaxis=dict(title='Date'),
        yaxis=dict(title='Close'),
        xaxis_rangeslider_visible=True
    )
    return fig
    
    
def prediction_graph(df, df_new, forecast_price, close_price, scaler):
    
    split_date = start=df['Date'].max() + pd.Timedelta(days=1)

    trace_before = df_new[df_new['Date'] <= split_date]
    trace_after = df_new[df_new['Date'] > split_date]
    
    fig = px.line(trace_before, x='Date', y='Close', title='Stock Close Prices Over Time')
    fig.update_traces(fill='tozeroy', line_color='blue')
    
    fig.add_trace(px.line(trace_after, x='Date', y='Close').data[0])
    fig.data[1].line.color = 'orange'

    fig.update_traces(selector=dict(type='scatter', mode='lines'), fill='tozeroy', line_color='orange')
    
    fig.update_layout(
        title='Stock Close Prices Over Time',
        xaxis=dict(title='Date'),
        yaxis=dict(title='Close'),
        xaxis_rangeslider_visible=True
    )
    return fig
    
def merge_data(df, df_pred):
    df_new = pd.concat([df, df_pred], ignore_index=True)
    df_new.sort_values(by='Date', inplace=True)
    return df_new

st.set_page_config(page_title='InvestWise', page_icon ='📈')
st.header('Your Investment partner  🤑')

st.sidebar.header('Settings')
end_date = st.sidebar.date_input('End Date', datetime.today())
start_date = st.sidebar.date_input('Start Date', datetime.today() - timedelta(5* 365))
ss.stock_name = st.sidebar.text_input('Stock Name', 'NVDA')
ss.num_days_pred = st.sidebar.number_input('Number of Days to Predict', 1)




if  st.sidebar.button('Get Data'):
    ss.df_raw, ss.df = get_data(ss.stock_name, end_date, start_date)
    # st.dataframe(ss.df_raw)
    ss.info_graph = info_graph(ss.df)
    ss.df_pred, close_price ,forecast_price, scaler  = runner(ss.df, ss.num_days_pred)
    ss.df_new = merge_data(ss.df, ss.df_pred)
    # st.dataframe(ss.df_new)
    ss.prediction_graph = prediction_graph(ss.df, ss.df_new, forecast_price, close_price, scaler)


if 'df_raw' in ss:
    st.header('Current Stock Price History 🧾')
    st.dataframe(ss.df_raw)

if 'info_graph' in ss:
    st.plotly_chart(ss.info_graph)

if 'df_new' in ss:
    st.header(' My prediction on next '+ str(ss.num_days_pred) +' days', "🪙")
    st.dataframe(ss.df_new)

if 'prediction_graph' in ss:
    st.plotly_chart(ss.prediction_graph)


    
    

