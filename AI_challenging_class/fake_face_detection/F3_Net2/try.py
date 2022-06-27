import os
import numpy as np


if __name__ == '__main__':
    l = np.loadtxt('./datasets/YuYong/labels.txt')
    l[np.argwhere(l == -1)] = 0
    print(l)