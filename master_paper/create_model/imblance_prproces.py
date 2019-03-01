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

# # ros = RandomOverSampler(random_state=0)
# # X_resampled, y_resampled = ros.fit_sample(train_set_1_1,label)
# # print(sorted(Counter(y_resampled).items()))#[(1.0, 175026), (2.0, 175026), (3.0, 175026)]
# # print(X_resampled)
# # x_train,x_test,y_train,y_test=train_test_split(X_resampled,y_resampled ,random_state=1)
# '''
# Synthetic Minority Oversampling Technique (SMOTE)
# SMOTE: 对于少数类样本a, 随机选择一个最近邻的样本b, 然后从a与b的连线上随机选取一个点c作为新的少数类样本
# '''
# from imblearn.over_sampling import SMOTE,ADASYN
# X_resampled_smote, y_resampled_smote = SMOTE().fit_sample(train_set_1_1, label)
# print('smote',sorted(Counter(y_resampled_smote).items()))#
# x_train_smote,x_test_smote,y_train_smote,y_test_smote=train_test_split(X_resampled_smote, y_resampled_smote ,random_state=1)
# 
# '''
# Adaptive Synthetic (ADASYN) 
# ADASYN: 关注的是在那些基于K最近邻分类器被错误分类的原始样本附近生成新的少数类样本
# '''
# X_resampled_adasyn, y_resampled_adasyn = ADASYN().fit_sample(train_set_1_1, label)
# print('ADASYN',sorted(Counter(y_resampled_adasyn).items()))
# x_train_adasyn,x_test_adasyn,y_train_adasyn,y_test_adasyn=train_test_split(X_resampled_adasyn, y_resampled_adasyn ,random_state=1)
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
# # svm_clf.fit(x_train, y_train)
# # joblib.dump(svm_clf,'../model/simple_sample_model.pkl')
# # 
# # #评估
# # from sklearn.model_selection import cross_val_score
# # scores=cross_val_score(svm_clf,x_test,y_test,cv=5)#[1. 1. 1. 1. 1.]
# # print(scores)
# # pred = svm_clf.predict(x_test)
# # print('simple_accuracy_score:',metrics.accuracy_score(y_test, pred))
# # print('simple_f1_score:',metrics.f1_score(y_test, pred,average="micro"))
# # from sklearn.metrics import cohen_kappa_score#Kappa系数是基于混淆矩阵的计算得到的模型评价参数
# # kappa = cohen_kappa_score(y_test,pred)
# # print('simple_cohen_kappa_score:',kappa)
# # from sklearn.metrics import hamming_loss#铰链损失
# # hamloss=hamming_loss(y_test,pred)
# # print('simple_hamming_loss',hamloss)
# '''
# poly
# [0.34648638 0.34873357 0.34698153 0.34365596 0.34454518]
# accuracy_score: 0.35074274396282473
# f1_score: 0.35074274396282473
# cohen_kappa_score: 0.026231322019858894
# hamming_loss 0.6492572560371753
# '''
# svm_clf.fit(x_train_smote, y_train_smote)
# joblib.dump(svm_clf,'../model/smote_sample_model.pkl')
# #smote评估
# from sklearn.model_selection import cross_val_score
# scores=cross_val_score(svm_clf,x_test_smote,y_test_smote,cv=5)#[1. 1. 1. 1. 1.]
# print('smotes_score:',scores)
# pred1 = svm_clf.predict(x_test_smote)
# print('smote_accuracy_score:',metrics.accuracy_score(y_test_smote, pred1))
# print('smote_f1_score:',metrics.f1_score(y_test_smote, pred1,average="micro"))
# from sklearn.metrics import cohen_kappa_score#Kappa系数是基于混淆矩阵的计算得到的模型评价参数
# kappa = cohen_kappa_score(y_test_smote,pred1)
# print('smote_cohen_kappa_score:',kappa)
# from sklearn.metrics import hamming_loss#铰链损失
# hamloss=hamming_loss(y_test_smote,pred1)
# print('smote_hamming_loss',hamloss)
# 
# #
# svm_clf.fit(x_train_adasyn, y_train_adasyn)
# joblib.dump(svm_clf,'../model/adasyn_sample_model.pkl')
# 
# #smote评估
# from sklearn.model_selection import cross_val_score
# scores=cross_val_score(svm_clf,x_test_adasyn,y_test_adasyn,cv=5)#[1. 1. 1. 1. 1.]
# print('adasyn_score:',scores)
# pred2 = svm_clf.predict(x_test_adasyn)
# print('adasyn_smote_accuracy_score:',metrics.accuracy_score(y_test_adasyn, pred2))
# print('adasyn_f1_score:',metrics.f1_score(y_test_adasyn, pred2,average="micro"))
# from sklearn.metrics import cohen_kappa_score#Kappa系数是基于混淆矩阵的计算得到的模型评价参数
# kappa = cohen_kappa_score(y_test_adasyn,pred2)
# print('adasyn_cohen_kappa_score:',kappa)
# from sklearn.metrics import hamming_loss#铰链损失
# hamloss=hamming_loss(y_test_adasyn,pred2)
# print('adasyn_hamming_loss',hamloss)

'''
score: [0.36168349 0.35958865 0.36012188 0.35584505 0.35284931]
simple_accuracy_score: 0.35359945151215055
simple_f1_score: 0.35359945151215055
simple_cohen_kappa_score: 0.030477865657530856
simple_hamming_loss 0.6464005484878494

smote [(1.0, 175026), (2.0, 175026), (3.0, 175026)]
ADASYN [(1.0, 172397), (2.0, 175026), (3.0, 181828)]
smotes_score: [0.36023615 0.36278804 0.35604647 0.35794004 0.35547768]
smote_accuracy_score: 0.36559762321931893
smote_f1_score: 0.36559762321931893
smote_cohen_kappa_score: 0.048463089465324405
smote_hamming_loss 0.634402376780681
adasyn_score: [0.36834946 0.36481125 0.3630489  0.35949664 0.36565641]
adasyn_smote_accuracy_score: 0.3661544972905157
adasyn_f1_score: 0.3661544972905157
adasyn_cohen_kappa_score: 0.04467966548542168
adasyn_hamming_loss 0.6338455027094844
'''





'''
下采样(Under-sampling)
原型生成(prototype generation)
给定数据集S,原型生成算法将生成一个子集S’,其中|S’|<|S|,但是子集并非来自于原始数据集. 
意思就是说:原型生成方法将减少数据集的样本数量,剩下的样本是由原始数据集生成的,而不是直接来源于原始数据集.
ClusterCentroids函数实现了上述功能: 每一个类别的样本都会用K-Means算法的中心点来进行合成, 而不是随机从原始样本进行抽取.
'''



from imblearn.under_sampling import ClusterCentroids
cc = ClusterCentroids(random_state=0)
X_resampled_cc, y_resampled_cc = cc.fit_sample(train_set_1_1, label)
print('ClusterCentroids:',sorted(Counter(y_resampled_cc).items()))
x_train_cc,x_test_cc,y_train_cc,y_test_cc=train_test_split(X_resampled_cc, y_resampled_cc ,random_state=1)
# ClusterCentroids函数提供了一种很高效的方法来减少样本的数量, 但需要注意的是, 该方法要求原始数据集最好能聚类成簇. 
# 此外, 中心点的数量应该设置好, 这样下采样的簇能很好地代表原始数据.
svm_clf.fit(x_train_cc, y_train_cc)
joblib.dump(svm_clf,'../model/cc_sample_model.pkl')

#smote评估
from sklearn.model_selection import cross_val_score
scores=cross_val_score(svm_clf,x_test_cc,y_test_cc,cv=5)
print('cc_score:',scores)
pred3 = svm_clf.predict(x_test_cc)
print('cc_accuracy_score:',metrics.accuracy_score(y_test_cc, pred3))
print('cc_f1_score:',metrics.f1_score(y_test_cc, pred3,average="micro"))
from sklearn.metrics import cohen_kappa_score#Kappa系数是基于混淆矩阵的计算得到的模型评价参数
kappa = cohen_kappa_score(y_test_cc,pred3)
print('cc_cohen_kappa_score:',kappa)
from sklearn.metrics import hamming_loss#铰链损失
hamloss=hamming_loss(y_test_cc,pred3)
print('cc_hamming_loss',hamloss)







'''
原型选择(prototype selection)
与原型生成不同的是, 原型选择算法是直接从原始数据集中进行抽取. 
抽取的方法大概可以分为两类:(i)可控的下采样技术(the controlled under-sampling techniques)
(ii)the cleaning under-sampling techniques
第一类的方法可以由用户指定下采样抽取的子集中样本的数量;第二类方法则不接受这种用户的干预.
'''
#RandomUnderSampler函数是一种快速并十分简单的方式来平衡各个类别的数据: 随机选取数据的子集.
from imblearn.under_sampling import RandomUnderSampler#下采样函数
rus = RandomUnderSampler(random_state=0, replacement=True)
X_resampled_rs, y_resampled_rs = rus.fit_sample(train_set_1_1, label)
print('RandomUnderSampler:',sorted(Counter(y_resampled_rs).items()))
x_train_rs,x_test_rs,y_train_rs,y_test_rs=train_test_split(X_resampled_rs, y_resampled_rs ,random_state=1)
svm_clf.fit(x_train_rs, y_train_rs)
joblib.dump(svm_clf,'../model/RS_sample_model.pkl')

#rs评估
from sklearn.model_selection import cross_val_score
scores=cross_val_score(svm_clf,x_test_rs,y_test_rs,cv=5)
print('rs_score:',scores)
pred4 = svm_clf.predict(x_test_rs)
print('rs_accuracy_score:',metrics.accuracy_score(y_test_rs, pred4))
print('rs_f1_score:',metrics.f1_score(y_test_rs, pred4,average="micro"))
from sklearn.metrics import cohen_kappa_score#Kappa系数是基于混淆矩阵的计算得到的模型评价参数
kappa = cohen_kappa_score(y_test_rs,pred4)
print('rs_cohen_kappa_score:',kappa)
from sklearn.metrics import hamming_loss#铰链损失
hamloss=hamming_loss(y_test_rs,pred4)
print('rs_hamming_loss',hamloss)


# #通过设置RandomUnderSampler中的replacement=True参数, 可以实现自助法(boostrap)抽样
# import numpy as np
# print(np.vstack({tuple(row) for row in X_resampled}).shape)
# #(192, 2) 不重复抽样
# 
# rus = RandomUnderSampler(random_state=0, replacement=True)
# X_resampled, y_resampled = rus.fit_sample(X, y)
# print(sorted(Counter(y_resampled).items()))
# print(np.vstack({tuple(row) for row in X_resampled}).shape)
# #(181, 2) 重复抽样

'''
NearMiss函数则添加了一些启发式(heuristic)的规则来选择样本,通过设定version参数来实现三种启发式的规则.
假设正样本是需要下采样的(多数类样本),负样本是少数类的样本.
NearMiss-1:选择离N个近邻的负样本的平均距离最小的正样本;
NearMiss-2:选择离N个负样本最远的平均距离最小的正样本;
NearMiss-3:是一个两段式的算法.首先,对于每一个负样本,保留它们的M个近邻样本;接着,那些到N个近邻样本平均距离最大的正样本将被选择.
'''
# from imblearn.under_sampling import NearMiss
# nm1 = NearMiss(random_state=0, version=1)
# X_resampled_nm1, y_resampled = nm1.fit_sample(X, y)
# print(sorted(Counter(y_resampled).items()))

'''
Cleaning under-sampling techniques
omek’s links
TomekLinks:样本x与样本y来自于不同的类别,满足以下条件,它们之间被称之为TomekLinks;
不存在另外一个样本z,使得d(x,z)<d(x,y)或者 d(y,z)<d(x,y)成立.其中d(.)表示两个样本之间的距离,也就是说两个样本之间互为近邻关系.
这个时候,样本x或样本y很有可能是噪声数据,或者两个样本在边界的位置附近.
TomekLinks函数中的auto参数控制Tomek’s links中的哪些样本被剔除.
默认的ratio='auto'移除多数类的样本,当ratio='all'时,两个样本均被移除.
'''
from imblearn.under_sampling import TomekLinks
tl =TomekLinks(random_state=0,ratio='all')
X_resampled_tl, y_resampled_tl = tl.fit_sample(train_set_1_1, label)
print('TomekLinks ；',sorted(Counter(y_resampled_cc).items()))
x_train_tl,x_test_tl,y_train_tl,y_test_tl=train_test_split(X_resampled_tl, y_resampled_tl ,random_state=1)
svm_clf.fit(x_train_tl, y_train_tl)
joblib.dump(svm_clf,'../model/tl_sample_model.pkl')

#tl评估
from sklearn.model_selection import cross_val_score
scores=cross_val_score(svm_clf,x_test_tl,y_test_tl,cv=5)
print('tl_score:',scores)
pred6 = svm_clf.predict(x_test_cc)
print('tl_accuracy_score:',metrics.accuracy_score(y_test_cc, pred3))
print('tl_f1_score:',metrics.f1_score(y_test_tl, pred6,average="micro"))
from sklearn.metrics import cohen_kappa_score#Kappa系数是基于混淆矩阵的计算得到的模型评价参数
kappa = cohen_kappa_score(y_test_tl,pred6)
print('tl_cohen_kappa_score:',kappa)
from sklearn.metrics import hamming_loss#铰链损失
hamloss=hamming_loss(y_test_tl,pred6)
print('tl_hamming_loss',hamloss)
'''
[(0, 55), (1, 249), (2, 4654)]
'''
'''
Edited data set using nearest neighbours
EditedNearestNeighbours这种方法应用最近邻算法来编辑(edit)数据集,
找出那些与邻居不太友好的样本然后移除.对于每一个要进行下采样的样本,那些不满足一些准则的样本将会被移除;
他们的绝大多数(kind_sel='mode')或者全部(kind_sel='all')的近邻样本都属于同一个类,这些样本会被保留在数据集中.
'''
print(sorted(Counter(y).items()))
from imblearn.under_sampling import EditedNearestNeighbours
enn = EditedNearestNeighbours(random_state=0)
X_resampled, y_resampled = enn.fit_sample(X, y)
print(sorted(Counter(y_resampled).items()))
'''
[(0, 64), (1, 262), (2, 4674)]
[(0, 64), (1, 213), (2, 4568)]
'''
'''
在此基础上, 延伸出了RepeatedEditedNearestNeighbours算法, 重复基础的EditedNearestNeighbours算法多次
'''
from imblearn.under_sampling import RepeatedEditedNearestNeighbours
renn = RepeatedEditedNearestNeighbours(random_state=0)
X_resampled, y_resampled = renn.fit_sample(X, y)
print(sorted(Counter(y_resampled).items()))
#[(0, 64), (1, 208), (2, 4551)]
#与RepeatedEditedNearestNeighbours算法不同的是, ALLKNN算法在进行每次迭代的时候, 最近邻的数量都在增加.
from imblearn.under_sampling import AllKNN
allknn = AllKNN(random_state=0)
X_resampled, y_resampled = allknn.fit_sample(X, y)
print(sorted(Counter(y_resampled).items()))
#[(0, 64), (1, 220), (2, 4601)]
#Condensed nearest neighbors and derived algorithms
'''
CondensedNearestNeighbour使用1近邻的方法来进行迭代,来判断一个样本是应该保留还是剔除,具体的实现步骤如下:
集合C:所有的少数类样本;
选择一个多数类样本(需要下采样)加入集合C,其他的这类样本放入集合S;
使用集合S训练一个1-NN的分类器,对集合S中的样本进行分类;
将集合S中错分的样本加入集合C;
重复上述过程, 直到没有样本再加入到集合C.
'''
from imblearn.under_sampling import CondensedNearestNeighbour
cnn = CondensedNearestNeighbour(random_state=0)
X_resampled, y_resampled = cnn.fit_sample(X, y)
print(sorted(Counter(y_resampled).items()))
#显然,CondensedNearestNeighbour方法对噪音数据是很敏感的,也容易加入噪音数据到集合C中.
#因此,OneSidedSelection函数使用 TomekLinks方法来剔除噪声数据(多数类样本).
from imblearn.under_sampling import OneSidedSelection
oss = OneSidedSelection(random_state=0)
X_resampled, y_resampled = oss.fit_sample(X, y)
print(sorted(Counter(y_resampled).items()))

'''
NeighbourhoodCleaningRule 算法主要关注如何清洗数据而不是筛选(considering)他们. 因此,该算法将使用
EditedNearestNeighbours和 3-NN分类器结果拒绝的样本之间的并集.
'''
from imblearn.under_sampling import NeighbourhoodCleaningRule
ncr = NeighbourhoodCleaningRule(random_state=0)
X_resampled, y_resampled = ncr.fit_sample(X, y)
print(sorted(Counter(y_resampled).items()))

#InstanceHardnessThreshold是一种很特殊的方法,是在数据上运用一种分类器,然后将概率低于阈值的样本剔除掉.
from sklearn.linear_model import LogisticRegression
from imblearn.under_sampling import InstanceHardnessThreshold
iht = InstanceHardnessThreshold(random_state=0,
                              estimator=LogisticRegression())
X_resampled, y_resampled = iht.fit_sample(X, y)
print(sorted(Counter(y_resampled).items()))

