import itertools
from copy import deepcopy
import numpy as np
from tqdm import tqdm
import pandas as pd
from getData import *
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

if __name__ == '__main__':
    classes = ["Iris-setosa", "Iris-virginica", "Iris-versicolor"]
    array = [0, 1, 2]
    perm = list(itertools.permutations(array))  # 要list一下，不然它只是一个对象

    # path = "iris.data"
    path = "bezdekIris.data"
    df = getData(path)
    X = df.values[:, :-1]  # 转为ndarray并且去掉类别
    y = df.iloc[:, -1].values  # list型
    n = len(y)

    # X = StandardScaler().fit_transform(X)
    max_accu = -1  # 最大准确率
    res = {}

    kmeans = KMeans(n_clusters=3)
    kmeans.fit(X)
    n_clusters = len([i for i in set(kmeans.labels_)])  # 统计聚类个数（-1为异常点）
    if n_clusters == 3:
        for tup in perm:
            x = 0
            for i in range(n):
                if classes[tup[kmeans.labels_[i]]] == y[i]:
                    x += 1
            if x > max_accu:
                max_accu = x
                res = {'accuracy': float(x / n),
                       'labels': deepcopy(kmeans.labels_)}
    print(path.split('.')[0], "  ", "k-means")
    print(res)
