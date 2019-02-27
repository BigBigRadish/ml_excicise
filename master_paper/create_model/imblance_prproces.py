# -*- coding: utf-8 -*-
'''
Created on 2019年2月27日

@author: Zhukun Luo
Jiangxi university of finance and economics
'''
#非平衡数据预处理
'''
Over-sampling  上采样
针对不平衡数据, 最简单的一种方法就是生成少数类的样本, 这其中最基本的一种方法就是: 从少数类的样本中进行随机采样来增加新的样本,
RandomOverSampler 函数就能实现上述的功能.
'''
import sklearn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from sklearn import manifold
from test.support import gzip
dataset=pd.read_csv('../data/dataset_1.csv')
print(dataset.count())
'''
GRADE         240897
day          1044628
deepnight    1044628
hour         1044628
workday      1044628
消费时间         1044628
消费类型         1044628
账号           1044628
金额           1044628
dtype: int64
'''
predict_set=dataset[dataset['GRADE'].isnull()]#预测集
dataset_1=dataset[dataset['GRADE'].notnull()]#训练集
print(dataset_1.count())
train_set=dataset_1.drop(columns=['消费时间','账号'])
print(train_set.count())
train_set_1=pd.get_dummies(train_set,columns=['deepnight','workday','消费类型']).astype("float")#onehot
print(train_set_1.columns.values.tolist())
'''
Index(['GRADE', 'day', 'hour', '金额', 'deepnight_afternoon',
       'deepnight_deepnight', 'deepnight_evening', 'deepnight_morning',
       'workday_星期一', 'workday_星期三', 'workday_星期二', 'workday_星期五',
       'workday_星期六', 'workday_星期四', 'workday_星期日', '消费类型_上机费', '消费类型_其他消费',
       '消费类型_图书借阅', '消费类型_打印复印', '消费类型_水电费', '消费类型_网络费', '消费类型_餐厅消费'],
      dtype='object')
'''
label=train_set_1['GRADE']
train_set_1_1=train_set_1.drop(columns=['GRADE'])
from collections import Counter
from imblearn.over_sampling import RandomOverSampler#简单随机过采样
from sklearn.model_selection import train_test_split
ros = RandomOverSampler(random_state=0)
X_resampled, y_resampled = ros.fit_sample(train_set_1_1,label)
print(sorted(Counter(y_resampled).items()))#[(1.0, 175026), (2.0, 175026), (3.0, 175026)]
print(X_resampled)
x_train,x_test,y_train,y_test=train_test_split(X_resampled,y_resampled ,random_state=1)
'''
Synthetic Minority Oversampling Technique (SMOTE)
SMOTE: 对于少数类样本a, 随机选择一个最近邻的样本b, 然后从a与b的连线上随机选取一个点c作为新的少数类样本
'''
from imblearn.over_sampling import SMOTE,ADASYN
X_resampled_smote, y_resampled_smote = SMOTE().fit_sample(train_set_1_1, label)
print('smote',sorted(Counter(y_resampled_smote).items()))#
x_train_smote,x_test_smote,y_train_smote,y_test_smote=train_test_split(X_resampled_smote, y_resampled_smote ,random_state=1)

'''
Adaptive Synthetic (ADASYN) 
ADASYN: 关注的是在那些基于K最近邻分类器被错误分类的原始样本附近生成新的少数类样本
'''
X_resampled_adasyn, y_resampled_adasyn = ADASYN().fit_sample(train_set_1_1, label)
print('ADASYN',sorted(Counter(y_resampled_adasyn).items()))
x_train_adasyn,x_test_adasyn,y_train_adasyn,y_test_adasyn=train_test_split(X_resampled_adasyn, y_resampled_adasyn ,random_state=1)
'''
导入svm模块
'''

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.externals import joblib

'''
一般现实问题中，线性模型难以解决问题，可以采用非线性SVM
对于非线性回归问题，可以先添加一个多项式的特征，即将一维数据转化为多项式数据，然后采用线性SVM；或者也可以直接采用多项式的kernel，直接进行非线性SVM的分类
SVC中的参数C越大，对于训练集来说，其误差越小，但是很容易发生过拟合；C越小，则允许有更多的训练集误分类，相当于soft margin
SVC中的参数coef0反映了高阶多项式相对于低阶多项式对模型的影响，如果发生了过拟合的现象，则可以减小coef0；如果发生了欠拟合的现象，可以试着增大coef0
'''

svm_clf = Pipeline([( "scaler", StandardScaler()),
                    ("svm_clf", SVC(kernel="rbf"))
                        ])
svm_clf.fit(x_train, y_train)
joblib.dump(svm_clf,'../model/simple_sample_model.pkl')

#评估
from sklearn.model_selection import cross_val_score
scores=cross_val_score(svm_clf,x_test,y_test,cv=5)#[1. 1. 1. 1. 1.]
print(scores)
pred = svm_clf.predict(x_test)
print('simple_accuracy_score:',metrics.accuracy_score(y_test, pred))
print('simple_f1_score:',metrics.f1_score(y_test, pred,average="micro"))
from sklearn.metrics import cohen_kappa_score#Kappa系数是基于混淆矩阵的计算得到的模型评价参数
kappa = cohen_kappa_score(y_test,pred)
print('simple_cohen_kappa_score:',kappa)
from sklearn.metrics import hamming_loss#铰链损失
hamloss=hamming_loss(y_test,pred)
print('simple_hamming_loss',hamloss)
'''
poly
[0.34648638 0.34873357 0.34698153 0.34365596 0.34454518]
accuracy_score: 0.35074274396282473
f1_score: 0.35074274396282473
cohen_kappa_score: 0.026231322019858894
hamming_loss 0.6492572560371753
'''
svm_clf.fit(x_train_smote, y_train_smote)
joblib.dump(svm_clf,'../model/smote_sample_model.pkl')
#smote评估
from sklearn.model_selection import cross_val_score
scores=cross_val_score(svm_clf,x_test_smote,y_test_smote,cv=5)#[1. 1. 1. 1. 1.]
print('smotes_score:',scores)
pred1 = svm_clf.predict(x_test_smote)
print('smote_accuracy_score:',metrics.accuracy_score(y_test_smote, pred))
print('smote_f1_score:',metrics.f1_score(y_test_smote, pred,average="micro"))
from sklearn.metrics import cohen_kappa_score#Kappa系数是基于混淆矩阵的计算得到的模型评价参数
kappa = cohen_kappa_score(y_test_smote,pred)
print('smote_cohen_kappa_score:',kappa)
from sklearn.metrics import hamming_loss#铰链损失
hamloss=hamming_loss(y_test_smote,pred)
print('smote_hamming_loss',hamloss)

#
svm_clf.fit(x_train_adasyn, y_train_adasyn)
joblib.dump(svm_clf,'../model/adasyn_sample_model.pkl')

#smote评估
from sklearn.model_selection import cross_val_score
scores=cross_val_score(svm_clf,x_test_adasyn,y_test_adasyn,cv=5)#[1. 1. 1. 1. 1.]
print('adasyn_score:',scores)
pred1 = svm_clf.predict(x_test_adasyn)
print('adasyn_smote_accuracy_score:',metrics.accuracy_score(y_test_adasyn, pred))
print('adasyn_f1_score:',metrics.f1_score(y_test_adasyn, pred,average="micro"))
from sklearn.metrics import cohen_kappa_score#Kappa系数是基于混淆矩阵的计算得到的模型评价参数
kappa = cohen_kappa_score(y_test_adasyn,pred)
print('adasyn_cohen_kappa_score:',kappa)
from sklearn.metrics import hamming_loss#铰链损失
hamloss=hamming_loss(y_test_adasyn,pred)
print('adasyn_hamming_loss',hamloss)
