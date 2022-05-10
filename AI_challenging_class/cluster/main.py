import numpy as np
from matplotlib.colors import ListedColormap
import itertools
from copy import deepcopy
from getData import *
from sklearn.cluster import Birch, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2

X = np.loadtxt('a.txt')
print(X[:, 0].shape)
X[:, 0], X[:, 1] = cv2.cartToPolar(np.squeeze(X[:, 0]), np.squeeze(X[:, 1]))
X = StandardScaler().fit_transform(X)
eps = 0.4
min_samples = 2
# dbscan = DBSCAN(eps=eps, min_samples=min_samples)
# dbscan.fit(X)
thre = 1.5
bf = 4
result = Birch(threshold=thre, branching_factor=bf, n_clusters=2)
result.fit(X)
# result = KMeans(n_clusters=4)
# result.fit(X)
print(X)
y = result.labels_
y_set = set(result.labels_)

ax1 = plt.subplot(projection='polar')
for i, j in enumerate(y_set):
    ax1.scatter(X[y == j, 0], X[y == j, 1],
                c=ListedColormap(('red', 'green', 'b'))(i), label=j)
ax1.legend()
ax1.show()

# for i, j in enumerate(y_set):
#     plt.scatter(X[y == j, 0], X[y == j, 1],
#                 c=ListedColormap(('red', 'green', 'b'))(i), label=j)
# plt.legend()
# plt.show()
