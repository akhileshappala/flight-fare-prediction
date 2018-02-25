#https://blog.trifork.com/2017/02/16/machine-learning-predicting-house-prices/
# There is different models (algorithms) that can be used to predict prices. The sklearn library offers many. Each algorithm can have a different performance on a given problem. To choose between different models we can compare their cross validation scores and pick the best performing one.
# Lowest score has the best performance

import numpy as np
import pandas as pd
from sklearn import preprocessing, svm, model_selection
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.ensemble import RandomForestRegressor

data = pd.read_csv("data.csv")[['ITIN_ID', 'ORIGIN', 'ORIGIN_AIRPORT_ID', 'ORIGIN_STATE_ABR', 'ROUNDTRIP', 'ITIN_YIELD', 'ITIN_FARE', 'DISTANCE']]
X = data[['ITIN_ID', 'ORIGIN_AIRPORT_ID', 'ROUNDTRIP', 'ITIN_YIELD', 'DISTANCE']]
Y = data['ITIN_FARE']


# model = sklearn.linear_model.LinearRegression()
# validation_scores = cross_val_score(sklearn.linear_model.LinearRegression(),
models = {
    'linear_regression': LinearRegression(),
    # 'elastic_net': ElasticNet(),
    'svr': svm.SVR(kernel='sigmoid'),
    'random_forest': RandomForestRegressor()
}

for name, model in models.items():
    validation_scores = model_selection.cross_val_score(model,
        data.as_matrix(['ITIN_ID', 'ORIGIN_AIRPORT_ID', 'ROUNDTRIP', 'ITIN_YIELD', 'DISTANCE']),
        data.as_matrix(['ITIN_FARE']),
        cv=10,
        scoring='neg_mean_squared_error')
    # print("Model %s had average error: %f" % (name, math.sqrt(-np.mean(validation_scores))))
    print(name)
    print(math.sqrt(-np.mean(validation_scores)))