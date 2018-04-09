# -*- coding: utf-8 -*
import numpy as np 
from sklearn import model_selection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import metrics
import matplotlib.pyplot as plt

dataset = np.loadtxt('watermelon_3.3.csv', delimiter=",")

# separate the data from the target attributes
X = dataset[:,1:3]
y = dataset[:,3]

# generalization of train and test set
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.5, random_state=0)

# model fitting
lda_model = LinearDiscriminantAnalysis(solver='lsqr', shrinkage=None).fit(X_train, y_train)

# model validation
y_pred = lda_model.predict(X_test)

# summarize the fit of the model
print(metrics.confusion_matrix(y_test, y_pred))
print(metrics.classification_report(y_test, y_pred))

