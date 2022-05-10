import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def genConCircle(filePath, r1, r2, eps=0.1):
    """

    :param filePath:
    :param r1:
    :param r2:
    :param eps:
    :return:
    """

    def getRandom(r1, eps):
        return r1 + eps * r1 * random.random() - 0.5 * eps * r1

    with open(filePath, 'w+') as f:
        for i in np.arange(0, 2 * np.pi, 0.01 * np.pi):
            f.write('{} {}\n'.format(getRandom(r1, eps) * np.cos(i), getRandom(r1, eps) * np.sin(i)))
            f.write('{} {}\n'.format(getRandom(r2, eps) * np.cos(i), getRandom(r2, eps) * np.sin(i)))


def draw2DTxt(filePath):
    data = np.loadtxt(filePath)
    x = data[:, 0]
    y = data[:, 1]
    plt.scatter(x, y)
    plt.show()


if __name__ == '__main__':
    # genConCircle('a.txt', 2.5, 3.5, 0.2)
    # draw2DTxt('a.txt')
    path = r'C:\Users\15594\Desktop\data.txt'
    df = pd.read_csv(path, sep=' ')
    df = df.iloc[:, :-1]
    data = df.values
    x = data[:, 0]
    y = data[:, 1]
    plt.scatter(x, y)
    plt.show()
