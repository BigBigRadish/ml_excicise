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
Over-sampling  上采样
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
# ClusterCentroids函数提供了一种很高效的方法来减少样本的数量, 但需要注意的是, 该方法要求原始数据集最好能聚类成簇. 
# 此外, 中心点的数量应该设置好, 这样下采样的簇能很好地代表原始数据.
'''
原型选择(prototype selection)
与原型生成不同的是, 原型选择算法是直接从原始数据集中进行抽取. 
抽取的方法大概可以分为两类:(i)可控的下采样技术(the controlled under-sampling techniques)
(ii)the cleaning under-sampling techniques
第一类的方法可以由用户指定下采样抽取的子集中样本的数量;第二类方法则不接受这种用户的干预.
'''
#RandomUnderSampler函数是一种快速并十分简单的方式来平衡各个类别的数据: 随机选取数据的子集.
from imblearn.under_sampling import RandomUnderSampler#下采样函数
rus = RandomUnderSampler(random_state=0)
X_resampled, y_resampled = rus.fit_sample(X, y)
print(sorted(Counter(y_resampled).items()))

#通过设置RandomUnderSampler中的replacement=True参数, 可以实现自助法(boostrap)抽样
import numpy as np
print(np.vstack({tuple(row) for row in X_resampled}).shape)
#(192, 2) 不重复抽样

rus = RandomUnderSampler(random_state=0, replacement=True)
X_resampled, y_resampled = rus.fit_sample(X, y)
print(sorted(Counter(y_resampled).items()))
print(np.vstack({tuple(row) for row in X_resampled}).shape)
#(181, 2) 重复抽样

'''
NearMiss函数则添加了一些启发式(heuristic)的规则来选择样本,通过设定version参数来实现三种启发式的规则.
假设正样本是需要下采样的(多数类样本),负样本是少数类的样本.
NearMiss-1:选择离N个近邻的负样本的平均距离最小的正样本;
NearMiss-2:选择离N个负样本最远的平均距离最小的正样本;
NearMiss-3:是一个两段式的算法.首先,对于每一个负样本,保留它们的M个近邻样本;接着,那些到N个近邻样本平均距离最大的正样本将被选择.
'''
from imblearn.under_sampling import NearMiss
nm1 = NearMiss(random_state=0, version=1)
X_resampled_nm1, y_resampled = nm1.fit_sample(X, y)
print(sorted(Counter(y_resampled).items()))

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
X_resampled, y_resampled = tl.fit_sample(X, y)
print(sorted(Counter(y_resampled).items()))
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
#[(0, 64), (1, 64), (2, 64)]


'''
过采样与下采样的结合
在之前的SMOTE方法中,当由边界的样本与其他样本进行过采样差值时,很容易生成一些噪音数据. 
因此,在过采样之后需要对样本进行清洗.这样,第三节中涉及到的TomekLink与 EditedNearestNeighbours方法就能实现上述的要求.
所以就有了两种结合过采样与下采样的方法:(i) SMOTETomek and (ii) SMOTEENN.
'''
from imblearn.combine import SMOTEENN
smote_enn = SMOTEENN(random_state=0)
X_resampled, y_resampled = smote_enn.fit_sample(X, y)
print(sorted(Counter(y_resampled).items()))

from imblearn.combine import SMOTETomek
smote_tomek = SMOTETomek(random_state=0)
X_resampled, y_resampled = smote_tomek.fit_sample(X, y)
print(sorted(Counter(y_resampled).items()))

'''
Ensemble的例子
一个不均衡的数据集能够通过多个均衡的子集来实现均衡, imblearn.ensemble模块能实现上述功能.
EasyEnsemble通过对原始的数据集进行随机下采样实现对数据集进行集成.
'''
from imblearn.ensemble import EasyEnsemble
ee = EasyEnsemble(random_state=0, n_subsets=10)
X_resampled, y_resampled = ee.fit_sample(X, y)

print(X_resampled.shape)
print(sorted(Counter(y_resampled[0]).items()))
#EasyEnsemble有两个很重要的参数: (i) n_subsets控制的是子集的个数  (ii)replacement决定是有放回还是无放回的随机采样.

'''
BalanceCascade(级联平衡)的方法通过使用分类器(estimator参数)来确保那些被错分类的样本在下一次进行子集选取的时候也能被采样到. 
同样,n_max_subset参数控制子集的个数,以及可以通过设置bootstrap=True来使用bootstraping(自助法).
'''
from imblearn.ensemble import BalanceCascade
from sklearn.linear_model import LogisticRegression
bc = BalanceCascade(random_state=0,
                    estimator=LogisticRegression(random_state=0),
                    n_max_subset=4)
X_resampled, y_resampled = bc.fit_sample(X, y)
print(X_resampled.shape)
print(sorted(Counter(y_resampled[0]).items()))


'''
Chaining ensemble of samplers and estimators
在集成分类器中,装袋方法(Bagging)在不同的随机选取的数据集上建立了多个估计量.
在scikit-learn中这个分类器叫做BaggingClassifier.然而,该分类器并不允许对每个数据集进行均衡.
因此,在对不均衡样本进行训练的时候,分类器其实是有偏的,偏向于多数类.
'''
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
bc = BaggingClassifier(base_estimator=DecisionTreeClassifier(),
                       random_state=0)
bc.fit(X_train, y_train) 
y_pred = bc.predict(X_test)
print(confusion_matrix(y_test, y_pred))

'''
BalancedBaggingClassifier 允许在训练每个基学习器之前对每个子集进行重抽样. 
简而言之,该方法结合了EasyEnsemble采样器与分类器(如BaggingClassifier)的结果.
'''
from imblearn.ensemble import BalancedBaggingClassifier
bbc = BalancedBaggingClassifier(base_estimator=DecisionTreeClassifier(),
                                ratio='auto',
                                replacement=False,
                                random_state=0)
bbc.fit(X, y) 

y_pred = bbc.predict(X_test)
print(confusion_matrix(y_test, y_pred))

'''
imblearn.datasets包与sklearn.datasets包形成了很好的互补.
该包主要有以下两个功能:(i)提供一系列的不平衡数据集来实现测试;(ii) 提供一种工具将原始的平衡数据转换为不平衡数据.
'''











