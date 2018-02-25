# http://www.learndatasci.com/predicting-housing-prices-linear-regression-using-python-pandas-statsmodels/

import numpy as np
import pandas as pd

from sklearn import preprocessing, svm, model_selection
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.ensemble import RandomForestRegressor

import statsmodels.api as sm
from statsmodels.formula.api import ols

data = pd.read_csv("data.csv")[['ITIN_ID', 'ORIGIN', 'ORIGIN_AIRPORT_ID', 'ORIGIN_STATE_ABR', 'ROUNDTRIP', 'ITIN_YIELD', 'ITIN_FARE', 'DISTANCE']]
data_model = ols("ITIN_FARE ~ DISTANCE", data=data).fit()

summary = data_model.summary()
