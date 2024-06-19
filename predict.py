import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from util import to_sequences, create_model, forecast

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import yfinance as yf
from datetime import datetime, timedelta

seq_size = 75
# num_days_pred = 30
 
def get_data(name, end_date, start_date):
    
    stock_data = yf.download(name, start=start_date, end=end_date)
    df = pd.DataFrame(stock_data)
    df_raw = df.copy()
    df_raw.reset_index(inplace=True)
    df.reset_index(inplace=True)
    

    df = df[['Date', 'Close']]
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.dropna(subset=['Close'])
    return df_raw, df

def scale_data(df):
    scaler = MinMaxScaler()
    close_price = scaler.fit_transform(np.array(df['Close']).reshape(-1,1))
    return close_price, scaler

def train_test_split(close_price):
    train_sz = int(len(close_price) * 0.80)
    test_sz = len(close_price) - train_sz
    train, test = close_price[:train_sz], close_price[train_sz:]
    return train, test



def get_predictions(model, trainX, testX, scaler):
    train_pred = model.predict(trainX)
    test_pred = model.predict(testX)
    train_pred = scaler.inverse_transform(train_pred)
    test_pred = scaler.inverse_transform(test_pred)
    return train_pred, test_pred

def get_rmse(trainY, testY, train_pred, test_pred):
    train_rmse = np.sqrt(mean_squared_error(trainY, train_pred))
    test_rmse = np.sqrt(mean_squared_error(testY, test_pred))
    return train_rmse, test_rmse

def forecast_stocks(test, model, num_days_pred):
    x_input = test[len(test) - seq_size :].reshape(1,-1)
    temp_input = list(x_input)
    temp_input = temp_input[0].tolist()
    forecast_price = forecast(seq_size, num_days_pred, temp_input, x_input, model)
    return forecast_price

def forecasted_data(df, forecast_price, scaler, close_price, num_days_pred):
    
    dates = pd.date_range(start=df['Date'].max() + pd.Timedelta(days=1), periods= num_days_pred)  
    pred_close_values = scaler.inverse_transform(forecast_price).ravel()
    df_pred = pd.DataFrame({'Date': dates, 'Close': pred_close_values})
    return df_pred



def runner(df, num_days_pred=30):
    # path = 'stocks/BEL.BEL.csv'
    # name = 'NVDA'
    # df = get_data(name)
    close_price, scaler = scale_data(df)
    train, test = train_test_split(close_price)
    
    trainX, trainY = to_sequences(train, seq_size)
    testX, testY = to_sequences(test, seq_size)
    trainX =trainX.reshape(trainX.shape[0],trainX.shape[1] , 1)
    testX = testX.reshape(testX.shape[0],testX.shape[1] , 1)

    model = create_model(seq_size)
    model.fit(trainX,trainY, validation_split=0.1, epochs=2, batch_size=64, verbose=1)
    
    # get_predictions(model, trainX, testX, scaler) # not needed as such
    
    
    forecast_price = forecast_stocks(test, model, num_days_pred)
    df_pred = forecasted_data(df, forecast_price, scaler, close_price, num_days_pred)
    return (df_pred, close_price ,forecast_price, scaler)
    
# if __name__ == "__main__":
#     main()
    
    
    
    
    










