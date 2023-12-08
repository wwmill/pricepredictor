# Price Prediction

Our project focuses on creating a price prediction algorithm. We used 4 different algorithms, specifically the Long Short Term Memory, Auto-Regressive Moving Average, Fourier Transform Forcasting, and Non-Linear Regression. 

**LSTM**

Long Short-Term Memory is a type of recurrent neural network. It is similar to a neural network where data is input and moves down the chain until it reaches a final state where the output is calculated. An LSTM allows data to go from the beginning/lookback time to the end so that it will use data over a longer period of time. 

**Fourier Transform**

Fourier transform forecasting takes advantage of the FFT, the algorithmic equivalent to the DFT, to determine the individual sinusoidal components of a signal -- in this case, the price of a product. This frequency domain representation can then be used to predict the future behavior of the signal once the inverse DFT is taken.

**Polynomial Regression**

The nonlinear regression model expands on the idea of linear regression. Given an order, n, the objective is to find the coefficients of each component of the polynomial in x that best fit existing data.

**ARMA**

ARMA uses a combination of regression and of the moving average to predict one future value at a time. Specifically, it uses a window of previous values to estimate noise and predict the next value.

The algorithms are implemented in the algs.py file, and the runfile.ipynb uses it to generate predictions. 

## Runfile.ipynb

You can upload the processed file into first line inthe second cell as data. Look at the data-preprocessing file to see how to convert an image to a npy file that can be imported to this dataset. 

### Price Predition for Future Prices

num_chop - Number of points to remove from the beginning and end of the dataset if it was filtered.

non_lin_reg(Prices, time of collection, numbr of days to predict, Time period data was collected over, order of fit)

lstm_alg(prices, time, days to predict, days in the past you look at for prediction, number of iterations, batch size, 0 (no plots) 1 (include plots))

fft_alg(prices, time, number of days to predit for, number of frequencies with maximum amplitude to use)

arma(prices[::(numer to condense data by)], number of values in the past to look at to see how correlated it is)


### Plotting the Full Dataset
    
prediction_y = algorithm result you want to plot with the full dataset

ax.set_aspect(set number to increace/decreace width of the graph)
    

### Price Prediction with Test vs Train Dataset

num_chop = points to remove from beginning and end for filtered data

num_test_pts = number of points to test on

you can comment out whichever algorithms you want to find the results for and the corresponding rmse calculation for that algorithm. 


### Plotting the Training and Predicted Data

prediction_y = result from algorithm you want to plot, eg pred_fft for FFT, pred_arma for ARMA, pred_lstm for LSTM, pred_lin_reg for non-linear regression


### Optional-Baysian Optimization of Parameters for LSTM

if you want to optimize the lstm parameters you can run these two cells

nBOiter = number of iterations of optimizations you can run

nu = smoothness of the kernel

Set the upper and lower bounds for testing for lookback, lookahead, epoch and batch size and run


## Data-Preprocess File

If you have images of the dataset, you can upload it to data-preprocess to process it into a npy array to put into the runfile. 

To process the data you input the image, maximum price, minimum prices and the days over which it was collected. 

    image_original = image location
    max_price = maximum price of the product 
    min_price = minimum price of the product
    ndays = number of days the product price was recorded for
