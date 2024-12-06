#使用支持向量机（SVM）来构建垃圾邮件分类器
#线形决策边界，训练线性支持向量机来学习分类边界
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import warnings

warnings.simplefilter("ignore")
data1 = pd.read_csv('svmdata1.csv')
data1.head()
positive = data1[data1['y'].isin([1])]
negative = data1[data1['y'].isin([0])]
fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(positive['X1'], positive['X2'], s=50, marker='x', label='Positive')
ax.scatter(negative['X1'], negative['X2'], s=50, marker='o', label='Negative')
ax.legend()
from sklearn import svm
#在正则化参数为1下的准确率（模型对错误分类的惩罚较小，会倾向于得到一个更宽松的决策边界）
svc = svm.LinearSVC(C=1, loss='hinge', max_iter=1000)
svc.fit(data1[['X1', 'X2']], data1['y'])
print(svc.score(data1[['X1', 'X2']], data1['y']))
#在正则化参数为100下的准确率（模型对错误分类的惩罚较大，会尽量使所有训练数据点都被正确分类）
svc2 = svm.LinearSVC(C=100, loss='hinge', max_iter=1000)
svc2.fit(data1[['X1', 'X2']], data1['y'])
print(svc2.score(data1[['X1', 'X2']], data1['y']))

data1['SVM 1 Confidence'] = svc.decision_function(data1[['X1', 'X2']])
fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(data1['X1'],
           data1['X2'],
           s=50,
           c=data1['SVM 1 Confidence'],
           cmap='seismic')
ax.set_title('SVM (C=1) Decision Confidence')
data1['SVM 2 Confidence'] = svc2.decision_function(data1[['X1', 'X2']])

fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(data1['X1'], data1['X2'], s=50, c=data1['SVM 2 Confidence'], cmap='seismic')
ax.set_title('SVM (C=100) Decision Confidence')
plt.show()