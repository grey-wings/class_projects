import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn.svm import SVC
from sklearn.metrics import *
from task2_PCA import load_data, getPCA


if __name__ == '__main__':
    x, y = load_data('scores.csv')
    p = np.array([[88, 74, 89, 92],
                  [80, 75, 74, 69]]).astype(float)
    x = getPCA(x)
    p = getPCA(p)

    # xx = np.vstack((x, p))
    # xx = getPCA(xx)
    # x = xx[:-2, :]
    # p = xx[-2:, :]

    c = 1.0
    svm = SVC(C=c, kernel='rbf')
    svm.fit(x, y)

    print('SVM PCA rbf kernel, C =', c)
    accu = svm.score(x, y)
    print('accurancy: {}'.format(accu))
    cf_matrix = confusion_matrix(y, svm.predict(x))
    print('confusion matrix: ')
    print(cf_matrix)
    f1 = f1_score(y, svm.predict(x))
    print('f1_score: {}'.format(f1))

    fpr, tpr, _ = roc_curve(y, svm.predict(x))
    plt.plot(fpr, tpr)
    plt.xlabel('False positive rates')
    plt.ylabel('Talse positive rates')
    plt.savefig('roc_curve_svm_pca_linear.png')
    plt.show()

    auc = roc_auc_score(y, svm.predict(x))
    print('roc_auc_score:', auc)

    print('prediction:')
    a, b = svm.predict(p)
    print('A: {}   B: {}'.format(a, b))
    y_p = [a, b]

    X_set = x
    y_set = y
    X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                         np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))
    plt.contourf(X1, X2, svm.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                 alpha=0.75, cmap=ListedColormap(('red', 'green')))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                    c=ListedColormap(('yellow', 'green'))(i), label='train={}'.format(j))
        plt.scatter(p[y_p == j, 0], p[y_p == j, 1],
                    c=ListedColormap(('Pink', 'Aqua'))(i), label='test={}'.format(j))
    plt.title('SVM C={}'.format(c))
    plt.xlabel('PCA_feature1')
    plt.ylabel('PCA_feature2')
    plt.legend()
    plt.show()
