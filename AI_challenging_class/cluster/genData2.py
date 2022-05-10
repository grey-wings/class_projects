import random
import numpy as np
import matplotlib.pyplot as plt


def genConCircle(filePath, r1, r2, eps):
    """
    :param filePath:
    :param r1:
    :param r2:
    :param eps:
    :return:
    """
    x1 = np.linspace(-5, 5, num=200)
    y1 = 0.5 * x1 + [np.random.random() for _ in range(200)]
    x2 = np.linspace(-5, 5, num=200)
    y2 = -0.5 * x2 + [np.random.random() for _ in range(200)]

    def getRandom(r1, eps):
        return r1 + eps * r1 * random.random() - 0.5 * eps * r1

    def gett(x, y):
        if x < 0:
            if y < 0:
                xx = 1
            else:
                xx = 2
        else:
            if y < 0:
                xx = 3
            else:
                xx = 4
        return '{} {} {}\n'.format(x, y, xx)
    with open(filePath, 'w+') as f:
        for i in np.arange(0, 2 * np.pi, 0.01 * np.pi):
            f.write(gett(getRandom(r1, eps) * np.cos(i), getRandom(r1, eps) * np.sin(i)))
        for i in np.arange(0, 2 * np.pi, 0.01 * np.pi):
            f.write(gett(getRandom(r2, eps) * np.cos(i), getRandom(r2, eps) * np.sin(i)))
        for i in range(200):
            f.write(gett(x1[i], y1[i]))
        for i in range(200):
            f.write(gett(x2[i], y2[i]))


def draw2DTxt(filePath):
    data = np.loadtxt(filePath)
    n = data.size
    x = data[:, 0]
    y = data[:, 1]
    label = data[:, 2]
    idx1 = np.where(label == 1)
    idx2 = np.where(label == 2)
    idx3 = np.where(label == 3)
    idx4 = np.where(label == 4)
    plt.scatter(x[idx1], y[idx1], c='b')
    plt.scatter(x[idx2], y[idx2], c='r')
    plt.scatter(x[idx3], y[idx3], c='y')
    plt.scatter(x[idx4], y[idx4], c='g')
    plt.show()


if __name__ == '__main__':
    genConCircle('a.txt', 2.5, 3.5, 0.2)
    draw2DTxt('a.txt')
