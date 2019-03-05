# -*- coding: utf-8 -*-
'''
Created on 2019年3月5日

@author: Zhukun Luo
Jiangxi university of finance and economics
'''
#获取小批量数据集
from sklearn import linear_model,tree
import sklearn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from sklearn import manifold
from test.support import gzip
dataset=pd.read_csv('./data/dataset_1.csv')
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
train_set=dataset_1.drop(columns=['consumption_time','账号'])
print(train_set.count())
train_set_1=pd.get_dummies(train_set,columns=['deepnight','workday','consumption_kind']).astype("float")#onehot
print(train_set_1.count())
mini_train_data=train_set_1.sample(frac=0.1)
print(mini_train_data.count())
# import seaborn as sns
# import warnings; warnings.filterwarnings(action='once')
# # 绘制热点图
# plt.figure(figsize=(12,10), dpi= 80)
# sns.heatmap(train_set_1.corr(), xticklabels=train_set_1.corr().columns, yticklabels=train_set_1.corr().columns, cmap='RdYlGn', center=0, annot=True)
# #组装
# plt.title('Correlogram of mtcars', fontsize=22)
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
# plt.show()
'''
Index(['GRADE', 'day', 'hour', '金额', 'deepnight_afternoon',
       'deepnight_deepnight', 'deepnight_evening', 'deepnight_morning',
       'workday_星期一', 'workday_星期三', 'workday_星期二', 'workday_星期五',
       'workday_星期六', 'workday_星期四', 'workday_星期日', '消费类型_上机费', '消费类型_其他消费',
       '消费类型_图书借阅', '消费类型_打印复印', '消费类型_水电费', '消费类型_网络费', '消费类型_餐厅消费'],
      dtype='object')
'''
from sklearn.model_selection import train_test_split
label=mini_train_data['GRADE']
train_set_1_1=mini_train_data.drop(columns=['GRADE'])
feature_name=train_set_1_1.columns.values.tolist()
x_train,x_test,y_train,y_test=train_test_split(train_set_1_1,label,random_state=1)
print(y_train.count())#180672
'''t-SNE'''
# tsne = manifold.TSNE(n_components=3, random_state=501)
# X_tsne = tsne.fit_transform(x_train)
#  
# print("Org data dimension is {}.Embedded data dimension is {}".format(x_train.shape[-1], X_tsne.shape[-1]))
#  
# '''嵌入空间可视化'''
# x_min, x_max = X_tsne.min(0), X_tsne.max(0)
# X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
# plt.figure(figsize=(8, 8))
# for i in range(X_norm.shape[0]):
#     plt.text(X_norm[i, 0], X_norm[i, 1], str(y_train[i]), color=plt.cm.Set1(y_train[i]), 
#              fontdict={'weight': 'bold', 'size': 9})
# plt.xticks([])
# plt.yticks([])
# plt.show()
dt = tree. DecisionTreeClassifier(random_state=0)
dt.fit(x_train, y_train)
from sklearn.model_selection import cross_val_score
scores=cross_val_score(dt,x_test,y_test,cv=5)#[1. 1. 1. 1. 1.]
y_importances=dt.feature_importances_
print(y_importances)
x_importances=feature_name
y_pos=np.arange(len(x_importances))
#横向柱状图
plt.barh(y_pos,y_importances,align='center')
plt.yticks(y_pos,x_importances)
plt.xlabel('importances')
plt.xlim(0,1)
plt.title('feature importance')
plt.show()
print(scores)
dtpred=dt.predict(x_test)
print(metrics.accuracy_score(y_test, dtpred))
print(metrics.f1_score(y_test, dtpred,average="micro"))
from sklearn.metrics import cohen_kappa_score#Kappa系数是基于混淆矩阵的计算得到的模型评价参数
kappa = cohen_kappa_score(y_test,dtpred)
print(kappa)
print(y_test)
from sklearn.metrics import hamming_loss#铰链损失
hamloss=hamming_loss(y_test,dtpred)
print(hamloss)
'''
1.0
1.0
1.0
1.0
0.0
'''
def plot_decision_boundary(pred_func, data, labels):
    '''绘制分类边界图'''
    # 设置最大值和最小值并增加0.5的边界（0.5 padding）
    x_min, x_max = data[:, 0].min() - 0.5, data[:, 0].max() + 0.5
    y_min, y_max = data[:, 1].min() - 0.5, data[:, 1].max() + 0.5
    h = 0.01
    # 生成一个点阵网格，点阵间距离为h
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    # 预测整个网格当中的函数值
    z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    z = z.reshape(xx.shape)
    # 绘制轮廓和训练样本
    plt.contourf(xx, yy, z, cmap=plt.cm.Spectral)
    plt.scatter(data[:, 0], data[:, 1], s=40, c=labels, cmap=plt.cm.Spectral)
def lin_regplot(X, y):
    plt.plot(X, y, color='blue', linewidth=2)
#     plt.xticks(fontsize=12,label='solver'); plt.yticks(fontsize=12,label='f1-scores')
#     plt.title("Bubble Plot with Encircling", fontsize=22)  
    plt.show()
#构建人工神经网路o
from sklearn.neural_network import MLPClassifier
# import gzip
# import pickle
# with gzip.open('./poverty.pkl.gz') as f_gz:
#     train_data,valid_data,test_data = pickle.load(f_gz)
solve=['adam','lbfgs','sgd']
activiton=['identity','logistic','tanh','relu']
max_it=[20,30,40,50,100]
hidden_layer=[50,100,150]
scores=[]
for i in hidden_layer:
    clf = MLPClassifier(solver='sgd',activation = 'identity',max_iter = 30,alpha = 1e-5,hidden_layer_sizes = (i,i),random_state = 1,verbose = True)
    clf.fit(x_train,y_train)
    mlppred=clf.predict(x_test)
    # print(clf.predict(x_test))
    print(clf.score(x_test,y_test))
    print(metrics.accuracy_score(y_test, mlppred))
    print(metrics.f1_score(y_test, mlppred,average="micro"))
    scores.append(metrics.f1_score(y_test, mlppred,average="micro"))
    kappa = cohen_kappa_score(y_test,mlppred)
    print(kappa)
    hamloss=hamming_loss(y_test,mlppred)
    print(hamloss)
lin_regplot(hidden_layer, scores)
'''
1.0
1.0
1.0
1.0
0.0
'''
# print(clf.predict_proba(test_data[0][:10]))
# from sklearn.model_selection import cross_val_predict
# from sklearn import metrics
# 
# predicted = cross_val_predict(clf, x_test, y_test, cv=5)
# 
# print(predicted)
# print(metrics.accuracy_score(predicted, y_test))