import numpy as np
import pandas as pd
# import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
pd.options.mode.chained_assignment = None
import matplotlib.pyplot as plt

from darts import TimeSeries
from darts.models import FFT, AutoARIMA, ExponentialSmoothing, Theta
from darts.metrics import mae
from darts.utils.missing_values import fill_missing_values

from datetime import datetime

from statsmodels.tsa.arima.model import ARIMA
from pandas.plotting import autocorrelation_plot
from matplotlib import pyplot
from math import sqrt
from sklearn.metrics import mean_squared_error

from skopt import Optimizer
from skopt.space.space import Real, Integer
from skopt.plots import plot_gaussian_process, plot_objective
from sklearn.gaussian_process.kernels import Matern
from skopt.learning import GaussianProcessRegressor as GPR
import random
from datetime import datetime
import os

def non_lin_reg(prices, time, lookahead, num_days, order):
    cols2 = time.shape[0]
    # time_add = np.linspace(num_days, num_days+lookahead, lookahead)
    # time2 = np.concatenate((time, time_add), axis=0)
    time = np.linspace(1, cols2, cols2)
    time2 = np.linspace(1, cols2+lookahead, cols2+lookahead)
    mymodel = np.poly1d(np.polyfit(time[:cols2], prices, order))
    y_prediction = mymodel(time2)
    # print(y_prediction.shape)
    y_pred = y_prediction[y_prediction.shape[0]-lookahead:]
    # print(y_pred.shape)
    
    fig0 = plt.figure(figsize=(10,5), facecolor = 'white')
    ax = fig0.add_subplot(111)
    plt.plot(mymodel(time2))
    plt.plot(prices)
    # plt.ylim([10,30])
    plt.title('Price Prediction using Non-Linear Regression')
    plt.xlabel('Time [days]')
    plt.ylabel('Price [$]')
    plt.grid('on')
    return y_pred

def lstm_alg(prices, time, lookahead, lookback, epoch, batchsize, plot):
    y = prices[:(prices.shape[0]-lookahead)].reshape(-1,1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(y)
    y = scaler.transform(y)
    X = []
    Y = []

    for i in range(lookback, len(y) - lookahead + 1):
        X.append(y[i - lookback: i])
        Y.append(y[i: i + lookahead])

    X = np.array(X)
    Y = np.array(Y)
    # print(X.shape)
    # print(Y.shape)
    # fit the model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(lookback, 1)))
    model.add(LSTM(units=50))
    model.add(Dense(lookahead))

    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X, Y, epochs=epoch, batch_size=batchsize, verbose=2)
    
    # generate the forecasts
    X_ = y[- lookback:]  # last available input sequence
    X_ = X_.reshape(1, lookback, 1)

    Y_ = model.predict(X_).reshape(-1, 1)
    Y_ = scaler.inverse_transform(Y_)
    
    results = np.append(prices,Y_.flatten())
    
    prediction_y = Y_.flatten()
    
    # time2 = np.arange(time[0], time_diff*len_original + len_prediction+time[0], time_diff)
    if(plot == 0):
        return prediction_y
    res = np.concatenate((prices, prediction_y), axis=0)
    fig1 = plt.figure(figsize=(10,5), facecolor = 'white')
    plt.plot(res)
    plt.title('Price Prediction with LSTM')
    plt.xlabel('Time [days]')
    plt.ylabel('Price [$]')
    # ax.set_aspect(30)
    return prediction_y

def fft_alg(prices, time, lookahead, freq):
    data = np.reshape(prices, (prices.shape[0],1))
    cols_2 = data.shape[0]

    start_date = datetime.strptime("2020-05-29-12", "%Y-%m-%d-%H")
    time = pd.date_range(start_date, periods=cols_2, freq='6H')

    data_df = pd.DataFrame(data, columns = ['data'])
    time_df = pd.DataFrame(time, columns = ['time'])
    df = pd.concat([time_df, data_df], axis=1)
    series = TimeSeries.from_dataframe(df,
                                   time_col = 'time',  
                                   value_cols = 'data',
                                   fill_missing_dates=True, freq='6H')
    model = FFT(trend="exp", nr_freqs_to_keep=freq)
    model.fit(series)
    pred_val = model.predict(lookahead) 
    fig2 = plt.figure(figsize=(10,5), facecolor = 'white')
    series.plot(label="train")
    pred_val.plot(label="forecast")
    # plot.ylabel('Price [$]')
    
    pred_arr = pred_val.pd_dataframe()
    pred_arr = pred_arr.to_numpy()

    prices = data
    prediction_y = pred_arr

    res = np.concatenate((prices, prediction_y), axis=0)
    # print(res)

    fig3 = plt.figure(figsize=(10,5), facecolor = 'white')
    # plt.clf()
    plt.plot(res)
    # print(res.shape)
    plt.title('Price Prediction with FFT')
    plt.xlabel('Time [days]')
    plt.ylabel('Price [$]')
    return prediction_y[:,0]


def ARMA_alg(prices, order_val):

    ## Autocorrelation plot to determine AR weighting of model
    autocorrelation_plot(prices)
    pyplot.show()

    model = ARIMA(prices, order=(order_val,1,0))
    model.initialize_approximate_diffuse()
    model_fit = model.fit()
    res = model_fit.forecast()
    
    # plotting
    size = int(len(prices) * 0.5)
    training, test = prices[0:size], prices[size:2*size]
    history = [x for x in training]
    predictions = list()
    for t in range(len(test)):
        # creates model and predicts for each point in test
        model = ARIMA(history, order=(order_val,1,0))
        model.initialize_approximate_diffuse()
        model_fit = model.fit()
        predictions.append(model_fit.forecast()[0])
        history.append(test[t])
    ## rmse = sqrt(mean_squared_error(test, predictions))
    pyplot.plot(test)
    pyplot.plot(predictions, color='red')
    pyplot.show()

    #returns one predicted point
    return res

def rmse(a, b):
  return np.sqrt(np.mean((a-b)**2))
