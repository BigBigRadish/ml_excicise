# -*- coding: utf-8 -*-
'''
Created on 2019年3月1日

@author: Zhukun Luo
Jiangxi university of finance and economics
'''
#svm imports
from sklearn.svm import SVC

import pickle

class predictModel(object):

    def __init__(self):
        """Simple NLP
        Attributes:
            clf: sklearn classifier model
            vectorizor: TFIDF vectorizer or similar
        """
        self.svc = SVC(kernel="rbf")


    def predict(self, X):
        """Returns the predicted class in an array
        """
        y_pred = self.svc.predict(X)
        return y_pred


