#非线性决策边界，训练非线性支持向量机来学习分类边界
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import warnings
from sklearn import svm
data2 = pd.read_csv('svmdata2.csv')
data2.head()
positive = data2[data2['y'].isin([1])]
negative = data2[data2['y'].isin([0])]

fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(positive['X1'], positive['X2'], s=30, marker='x', label='Positive')
ax.scatter(negative['X1'], negative['X2'], s=30, marker='o', label='Negative')
ax.legend()
svc = svm.SVC(C=100, gamma=10, probability=True)
svc
svc.fit(data2[['X1', 'X2']], data2['y'])
print(svc.score(data2[['X1', 'X2']], data2['y']))
#为了可视化决策边界，这一次我们将根据实例具有负类标签的预测概率来对点做阴影。
data2['Probability'] = svc.predict_proba(data2[['X1', 'X2']])[:, 0]
fig, ax = plt.subplots(figsize=(12, 8))
#用点的颜色来表示每个样本属于负类的预测概率，颜色映射采用Reds，颜色越深表示属于负类的概率越高，
ax.scatter(data2['X1'], data2['X2'], s=30, c=data2['Probability'], cmap='Reds')
plt.show()
