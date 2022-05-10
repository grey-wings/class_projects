import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def load_data(dataset_path):
    df = pd.read_csv(dataset_path, index_col='编号', encoding='gbk')
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    return X, y


def getPCA(X):
    pca = PCA(n_components=2)
    reduced_x = pca.fit_transform(X)
    return reduced_x


if __name__ == '__main__':
    x, y = load_data('scores.csv')
    pca = PCA(n_components=2)
    reduced_x = pca.fit_transform(x)

    red_1 = reduced_x[np.where(y == 1)][:, 0]
    red_2 = reduced_x[np.where(y == 1)][:, 1]
    blue_1 = reduced_x[np.where(y == 0)][:, 0]
    blue_2 = reduced_x[np.where(y == 0)][:, 1]

    plt.scatter(red_1, red_2, c='r')
    plt.scatter(blue_1, blue_2, c='b')
    plt.savefig('PCA_result.png')
    plt.show()
