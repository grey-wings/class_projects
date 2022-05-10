import itertools
from copy import deepcopy
import numpy as np
from getData import *
from sklearn.cluster import Birch
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


def get_test_data(ans, one, two, three):
    k = 1
    while k <= 3:
        node_data = ans[k]
        a = 0
        b = 0
        c = 0
        for line in node_data:
            if line[4] == 'Iris-setosa':
                a = a + 1
            if line[4] == 'Iris-versicolor':
                b = b + 1
            if line[4] == 'Iris-virginica':
                c = c + 1
            test_list = [a, b, c]
        jiaozhun = max(test_list)
        if jiaozhun == a:
            print("分类为‘Iris-setosa’的准确率为：")
            print(a / one)
        if jiaozhun == b:
            print("分类为‘Iris-versicolor’的准确率为：")
            print(b / two)
        if jiaozhun == c:
            print("分类为‘Iris-virginica’的准确率为：")
            print(c / three)
        k = k + 1
    pass


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

    X = StandardScaler().fit_transform(X)
    max_accu = -1  # 最大准确率
    res = {}
    for thre in tqdm(np.arange(0.01, 0.6, 0.01)):
        for bf in range(10, 30):
            birch = Birch(threshold=thre, branching_factor=bf, n_clusters=3)
            birch.fit(X)
            n_clusters = len([i for i in set(birch.labels_) if i != -1])
            if n_clusters == 3:
                for tup in perm:
                    x = 0
                    for i in range(n):
                        if classes[tup[birch.labels_[i]]] == y[i]:
                            x += 1
                    if x > max_accu:
                        max_accu = x
                        res = {'threshold': thre, 'branching_factor': bf, 'accuracy': float(x / n),
                               'labels': deepcopy(birch.labels_)}
    print(path.split('.')[0], "  ", "Birch")
    print(res)
