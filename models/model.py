# Importing the libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
import requests
import json
import os
import glob
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import MinMaxScaler

from statsmodels.tsa.arima_model import ARIMA
from math import sqrt

import math
from sklearn.metrics import mean_squared_error, mean_absolute_error

def arima_model(file_name,choice,dir1):
    os.chdir(dir1)
    data=pd.read_csv(file_name+'.csv')
    if choice==0:
        train=data["Open Price"]
    elif choice==1:
        train=data["High Price"]
    elif choice==2:
        train=data["Low Price"]
    elif choice==3:
        train=data["No of Trades"]
    train=train[::-1]
    model = ARIMA(train, order=(1,1,0))
    model_fit = model.fit(disp=0)
    #print(model_fit)
    next_day_pred = model_fit.forecast()[0]
    #print(yhat)
    return next_day_pred
