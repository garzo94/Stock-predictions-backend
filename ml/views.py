from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from datetime import datetime
import yfinance as yf

# data libraries
import math
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM


class Data(APIView):
    def get(self, request):
        #Get the stock quote
        stock = request.GET.get('stock')
        start = request.GET.get('start')
        end = request.GET.get('end')

        # print(stock, start, end,'stockstartend')
        print(start, type(start), 'startttt')
        yf.pdr_override()
        df = yf.download(stock,  start=start, end=end)

        df.reset_index(inplace=True)

        # Create a new dataframe wit Close column
        data = df.filter(['Close'])
        time = df.filter(['Date'])
        #Convert the dataframe to numpy array
        dataset = data.values
        dataset
        # #Get the number of rows to train the model on
        training_data_len = math.ceil(len(dataset) * .8)

        #Scale the data
        scaler = MinMaxScaler(feature_range=(0,1))
        scaled_data = scaler.fit_transform(dataset)

        # Training the dataset
        # Create the sacaled training data set
        train_data = scaled_data[0:training_data_len, :]

        #split the data into x_train and y_train data sets
        x_train = []
        y_train = []

        for i in range(60,len(train_data)):
            x_train.append(train_data[i-60:i,0])
            y_train.append(train_data[i,0])

        #Convert the x_train and y_train to numpy arraYS
        x_train, y_train = np.array(x_train), np.array(y_train)

        #Reshape the data
        x_train = np.reshape(x_train,(x_train.shape[0], x_train.shape[1],1))

        #Build LSTM model
        model = Sequential()
        model.add(LSTM(50,return_sequences=True, input_shape=(x_train.shape[1],1)))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dense(25))
        model.add(Dense(1))

        # Compilb the model
        model.compile(optimizer="adam",loss="mean_squared_error")

        #Train the model
        model.fit(x_train, y_train, batch_size=1, epochs=1)

        #create the test data set
        #create a new array containing scaled valued from index 1543 to 2003
        test_data = scaled_data[training_data_len-60:,:]
        #create the dat sets x_test and train
        x_test = []
        y_test = dataset[training_data_len:,:]
        for i in range(60, len(test_data)):
            x_test.append(test_data[i-60:i,0])

        #convert the data to anumpy array
        x_test = np.array(x_test)

        #reshape the data
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

        #get the models predicted price values
        predictions = model.predict(x_test)

        predictions = scaler.inverse_transform(predictions)

        #get the root mean squared error (rmse)
        rmse = np.sqrt(np.mean(predictions - y_test)**2)

        train = data[:training_data_len]
        train['timeTrain'] = time[:training_data_len]
        valid = data[training_data_len:]
        valid['Predictions'] = predictions
        valid['timeValid'] = time[training_data_len:]


        # predicting
      #Get the last 60days closing price values and conver the dataframe to an array
        last_60_days = data[-60:].values
        #scale the dta to be values betwen 0 and 1
        last_60_days_scaled = scaler.transform(last_60_days)
        #create an empty list
        X_test = []
        #Append the past 60 days
        X_test.append(last_60_days_scaled)
        #Convert the X_TEST DATA SET TO A numpy array
        X_test = np.array(X_test)
        #reshape the data
        X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
        #predicted scale price
        pred_price = model.predict(X_test)
        #undo the scaling
        pred_price = scaler.inverse_transform(pred_price)


        return Response(
            {
                'data':{
                    'prices':df['Close'],
                    "time":df['Date'],
                    "train":train,
                    "valid":valid,
                    "price":pred_price,
                    "rmse":rmse
                    }
            }

                    )
