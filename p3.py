import os, io
import numpy as np
from pylab import *
import pandas as pd
from sklearn import preprocessing, cross_validation, svm, model_selection
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.ensemble import RandomForestRegressor

data = pd.read_csv("data.csv")[['ITIN_ID', 'ORIGIN', 'ORIGIN_AIRPORT_ID', 'ORIGIN_STATE_ABR', 'ROUNDTRIP', 'ITIN_YIELD', 'ITIN_FARE', 'DISTANCE']]
model = LinearRegression()

model.fit(data.as_matrix(['ORIGIN_AIRPORT_ID', 'ROUNDTRIP', 'DISTANCE']), data.as_matrix(['ITIN_FARE']))

forecast = model.predict(10135, 1, 850)

print(forecast)