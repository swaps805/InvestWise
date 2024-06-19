import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from keras.models import Sequential
from keras.layers import LSTM, Input, Dropout
from keras.layers import Dense
from keras.models import Model


def to_sequences(data, seq_size=1):
    dataX = []
    dataY = []
    for i in range(len(data)-seq_size-1):
        # print(i)
        dataX.append(data[i: (i + seq_size), 0])
        dataY.append(data[i + seq_size, 0])

    return np.array(dataX), np.array(dataY)

def create_model(seq_size):
    model = Sequential([
        Input(shape=(seq_size, 1)),  # batch_sz = defined later,seq-size, features (1)
        LSTM(128, return_sequences=True),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(64),
        Dense(1),
    ])
    model.compile(loss="mean_squared_error", optimizer= "adam")
    return model

def forecast(seq_size, num_days_pred, temp_input, x_input, model):
    forecast_price=[]
    i = 0
    while(i < num_days_pred):
        
        if(len(temp_input) > seq_size):
            #print(temp_input)
            x_input = np.array(temp_input[1:]) # selected the next 30 days exclude the first day (1 | 30)
            #print("{} day input {}".format(i,x_input))
            x_input = x_input.reshape(1,-1)
            x_input = x_input.reshape((1, seq_size, 1))
            #print(x_input)
            yhat = model.predict(x_input, verbose=0)
            #print("{} day output {}".format(i,yhat))
            temp_input.extend(yhat[0].tolist())
            temp_input = temp_input[1:]
            #print(temp_input)
            forecast_price.extend(yhat.tolist())
            i += 1
        else:
            x_input = x_input.reshape((1, seq_size,1)) # give to LSTM a single sample 
            yhat = model.predict(x_input, verbose=0) # returns 
            #print(yhat[0])
            temp_input.extend(yhat[0].tolist()) # add that predicted value to temp_input
            #print(temp_input)
            #print(len(temp_input))
            forecast_price.extend(yhat.tolist())
            i += 1
    return forecast_price
