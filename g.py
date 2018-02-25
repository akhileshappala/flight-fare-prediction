# https://datascienceplus.com/linear-regression-in-python-predict-the-bay-areas-home-prices/

import os, io
import numpy as np
from pylab import *
import pandas as pd
from sklearn import preprocessing, cross_validation, svm, model_selection
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.ensemble import RandomForestRegressor

data = pd.read_csv("data.csv")[['ITIN_ID', 'ORIGIN', 'ORIGIN_AIRPORT_ID', 'ORIGIN_STATE_ABR', 'ROUNDTRIP', 'ITIN_YIELD', 'ITIN_FARE', 'DISTANCE']]
data['ORIGIN'] = data['ORIGIN'].str.replace(',', '')
data['ORIGIN'] = data['ORIGIN'].convert_objects(convert_numeric=True)

data['ORIGIN_STATE_ABR'] = data['ORIGIN_STATE_ABR'].str.replace(',', '')
data['ORIGIN_STATE_ABR'] = data['ORIGIN_STATE_ABR'].convert_objects(convert_numeric=True)

# data.info()
print(data.describe())


# %matplotlib inline
import matplotlib.pyplot as plt
# data.hist(bins=50, figsize=(20,15))
# plt.savefig("attribute_histogram_plots")
# plt.show()

data.plot(kind="scatter", x="DISTANCE", y="ITIN_FARE", alpha=0.2)
plt.savefig('map1.png')

data.plot(kind="scatter", x="DISTANCE", y="ITIN_FARE", alpha=0.4, figsize=(10,7), c="ROUNDTRIP", cmap=plt.get_cmap("jet"), colorbar=True, sharex=False)
plt.savefig('map2.png')


# data.lastsolddate.min(), data.lastsolddate.max()