# -*- coding: utf-8 -*-
'''
Created on 2019年2月18日

@author: Zhukun Luo
Jiangxi university of finance and economics
'''
#不平衡数据处理的常用方法
#当我们的样本数据中, 正负样本的数据占比极其不均衡的时候, 模型的效果就会偏向于多数类的结果。
#https://imbalanced-learn.org/en/stable/introduction.html
'''
Over-sampling
针对不平衡数据, 最简单的一种方法就是生成少数类的样本, 这其中最基本的一种方法就是: 从少数类的样本中进行随机采样来增加新的样本,
RandomOverSampler 函数就能实现上述的功能.
'''
from sklearn.datasets import make_classification
'''
n_features :特征个数= n_informative（） + n_redundant + n_repeated
n_informative：多信息特征的个数
n_redundant：冗余信息，informative特征的随机线性组合
n_repeated ：重复信息，随机提取n_informative和n_redundant 特征
n_classes：分类类别
n_clusters_per_class ：某一个类别是由几个cluster构成的

weights:列表类型，权重比

class_sep:乘以超立方体大小的因子。 较大的值分散了簇/类，并使分类任务更容易。默认为1

random_state: 如果是int，random_state是随机数发生器使用的种子; 如果RandomState实例，random_state是随机数生成器; 如果没有，则随机数生成器是np.random使用的RandomState实例。

返回值：

X：形状数组[n_samples，n_features]
生成的样本。

y：形状数组[n_samples]
每个样本的类成员的整数标签。
'''
from collections import Counter
X, y = make_classification(n_samples=5000, n_features=2, n_informative=2,
                           n_redundant=0, n_repeated=0, n_classes=3,
                           n_clusters_per_class=1,
                           weights=[0.01, 0.05, 0.94],
                           class_sep=0.8, random_state=0)
y_1=Counter(y)
print(Counter(y))
from imblearn.over_sampling import RandomOverSampler#简单随机过采样
ros = RandomOverSampler(random_state=0)
X_resampled, y_resampled = ros.fit_sample(X, y)
print(sorted(Counter(y_resampled).items()))
print(X_resampled)


#Synthetic Minority Oversampling Technique (SMOTE)
#SMOTE: 对于少数类样本a, 随机选择一个最近邻的样本b, 然后从a与b的连线上随机选取一个点c作为新的少数类样本
from imblearn.over_sampling import SMOTE,ADASYN
X_resampled_smote, y_resampled_smote = SMOTE().fit_sample(X, y)
print(sorted(Counter(y_resampled_smote).items()))


#Adaptive Synthetic (ADASYN) 
#ADASYN: 关注的是在那些基于K最近邻分类器被错误分类的原始样本附近生成新的少数类样本
X_resampled_adasyn, y_resampled_adasyn = ADASYN().fit_sample(X, y)
sorted(Counter(y_resampled_adasyn).items())

'''
SMOTE的变体
对于基本的SMOTE算法,关注的是所有的少数类样本,这些情况可能会导致产生次优的决策函数, 
因此SMOTE就产生了一些变体:这些方法关注在最优化决策函数边界的一些少数类样本,然后在最近邻类的相反方向生成样本.
SMOTE函数中的kind参数控制了选择哪种变体,(i)borderline1,(ii)borderline2,(iii) svm:
'''
from imblearn.over_sampling import SMOTE, ADASYN
X_resampled, y_resampled = SMOTE(kind='borderline1').fit_sample(X, y)
print(sorted(Counter(y_resampled).items()))

'''
数学意义
SMOTE算法与ADASYN都是基于同样的算法来合成新的少数类样本:对于少数类样本a,从它的最近邻中选择一个样本b,
然后在两点的连线上随机生成一个新的少数类样本, 不同的是对于少数类样本的选择.
原始的SMOTE:kind='regular',随机选取少数类的样本.
The borderline SMOTE: kind='borderline1' or kind='borderline2'
此时, 少数类的样本分为三类:(i)噪音样本(noise),该少数类的所有最近邻样本都来自于不同于样本a的其他类别;(ii) 危险样本(in danger),
至少一半的最近邻样本来自于同一类(不同于a的类别);(iii)安全样本(safe),所有的最近邻样本都来自于同一个类.
这两种类型的SMOTE使用的是危险样本来生成新的样本数据,对于 Borderline-1 SMOTE,最近邻中的随机样本b与该少数类样本a来自于不同的类;
不同的是,对于 Borderline-2 SMOTE,随机样本b可以是属于任何一个类的样本;
SVM SMOTE:kind='svm',使用支持向量机分类器产生支持向量然后再生成新的少数类样本.
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
X_resampled, y_resampled = cc.fit_sample(X, y)
print(sorted(Counter(y_resampled).items()))





