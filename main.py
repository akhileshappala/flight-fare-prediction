import os
import io
import numpy as np
from pylab import *
import pandas as pd
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.naive_bayes import MultinomialNB

import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()

# data = pd.read_csv("data.csv")[['ITIN_ID', 'ORIGIN', 'ORIGIN_AIRPORT_ID', 'ORIGIN_STATE_ABR', 'ROUNDTRIP', 'ITIN_YIELD', 'ITIN_FARE', 'DISTANCE']]

data = pd.read_csv("data.csv")[['ITIN_ID', 'ORIGIN', 'ORIGIN_AIRPORT_ID', 'ORIGIN_STATE_ABR', 'ROUNDTRIP', 'ITIN_YIELD', 'ITIN_FARE', 'DISTANCE']]
# data = data.pivot_table(index=['ITIN_ID'],columns=['title'],values='rating')


# clf = svm.SVC(gamma=0.001, C=100.)
# clf.fit(digits.data[:-1], digits.target[:-1])
X = data[['ITIN_ID', 'ORIGIN_AIRPORT_ID', 'ROUNDTRIP', 'ITIN_YIELD', 'DISTANCE']]
Y = data['ITIN_FARE']

X[['ITIN_ID', 'ORIGIN_AIRPORT_ID', 'ROUNDTRIP', 'ITIN_YIELD', 'DISTANCE']] = scale.fit_transform(X[['ITIN_ID', 'ORIGIN_AIRPORT_ID', 'ROUNDTRIP', 'ITIN_YIELD', 'DISTANCE']].as_matrix())

# print(X)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        # est = sm.OLS(y, X).fit()
est = sm.OLS(y, X).fit()

est.summary()                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        # est.summary()
y.groupby(df.ITIN_FARE).mean()

# vectorizer = CountVectorizer()
# counts = vectorizer.fit_transform(data['ORIGIN', 'ORIGIN_AIRPORT_ID', 'ORIGIN_STATE_ABR', 'ROUNDTRIP', 'ITIN_YIELD', 'DISTANCE'])

# print(counts)
# classifier = MultinomialNB()
# targets = data['ITIN_FARE'].values
# classifier.fit(counts, targets)


# examples = ['ABE', ]
# examples_counts = vectorizer.transform(examples)
# predictions = classifier.predict(examples_counts)

# print(predictions)
