

import datetime as dt
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
import streamlit as st

default_company = 'AAPL'

#start and end dates of stock data, free to change
start = dt.datetime(2010,1,1)
end =  dt.datetime(2019,12,31)

st.title('Stock Trend Prediction')
user_input = st.text_input('Enter Stock Ticker', default_company)
# Read Stock Price Data from yahoo finance's website
df = yf.download(user_input, start , end)

#Describing Data
st.subheader('Data from 2010-2019')
st.write(df.describe())

#Visualizations
st.subheader('Closing Price vs Time chart')
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close, 'b')
st.pyplot(fig)

#Plotting 100 day Moving Average and Closing Price
st.subheader('Closing Price vs Time chart with 100 day MA')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close, 'b')
plt.plot(ma100, 'r')
st.pyplot(fig)

#Plotting 200 day MA, 100 day MA and Closing Price
st.subheader('Closing Price vs Time chart with 100MA and 200 MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close, 'b')
plt.plot(ma100, 'r')
plt.plot(ma200, 'g')
st.pyplot(fig)

#Splittig data into a Training Set(70%) and a Testing Set(30%)
data_length = len(df)
data_training = pd.DataFrame(df['Close'][0:int(data_length*0.70)])
data_testing = pd.DataFrame(df['Close'][int(data_length*0.70):int(len(df))])
print(data_training.shape)
print(data_testing.shape)

#closing price values are scaled from 0-1 per LTSM model
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
data_training_array = scaler.fit_transform(data_training) #scaling training data

#Load my model
model = load_model('my_model.keras')

#Testing Part
past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True) #data testing has prepended 100 days data
input_data = scaler.fit_transform(final_df)

x_test=[]
y_test=[]

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)#converting into numpy arrays
# Making Predictions
y_predicted = model.predict(x_test)
scaler = scaler.scale_

scale_factor = 1/scaler[0] #scaling up data
y_predicted = y_predicted*scale_factor
y_test = y_test*scale_factor

#Final Graph
st.subheader('Predictions vs Original')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test, 'b', label='Original Price')
plt.plot(y_predicted, 'r', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
st.pyplot(fig2)
