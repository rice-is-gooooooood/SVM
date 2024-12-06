#给出了训练和验证集，并且基于验证集性能为SVM模型找到最优超参数。
#一个是训练数据集（data3），另一个是验证数据集（data3val），用于评估模型在未见过的数据上的性能。
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn import svm
import warnings

warnings.simplefilter("ignore")
data3=pd.read_csv('svmdata3.csv')
data3val=pd.read_csv('svmdata3val .csv')
X = data3[['X1','X2']]
Xval = data3val[['X1','X2']]
y = data3['y'].ravel()
yval = data3val['yval'].ravel()
C_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
gamma_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
#gamma:非线性参数，用于调整图像的亮度和对比度，为1.0表示线性响应，而Gamma值大于1会使图像显示得更暗。
best_score = 0
best_params = {'C': None, 'gamma': None}

for C in C_values:
    for gamma in gamma_values:
        svc = svm.SVC(C=C, gamma=gamma)
        svc.fit(X, y)
        score = svc.score(Xval, yval)

        if score > best_score:
            best_score = score
            best_params['C'] = C
            best_params['gamma'] = gamma
#在验证集上得到的最佳准确率（best_score）以及对应的最佳超参数组合（best_params，包含C和gamma的值）。
print(best_score, best_params)