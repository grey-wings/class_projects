import os
import struct
import time
import random
import cv2
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from tqdm import tqdm
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn import naive_bayes

SZ = 28
affine_flags = cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR
train_numbers = 60000
test_numbers = 10000

bNB = naive_bayes.BernoulliNB()
coNB = naive_bayes.ComplementNB()
gNB = naive_bayes.GaussianNB()
mNB = naive_bayes.MultinomialNB()
svm = SVC(kernel='rbf')
model_list = [bNB, coNB, gNB, mNB, svm]
model_name_list = ['BernoulliNB', 'ComplementNB', 'GaussianNB', 'MultinomialNB', 'SVM-rbf']


def deskew(img):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11'] / m['mu02']
    M = np.float32([[1, skew, -0.5 * SZ * skew], [0, 1, 0]])
    img = cv2.warpAffine(img, M, (SZ, SZ), flags=affine_flags)
    return img


def dataLoader(nsamples, type='train'):
    """

    :param nsamples:
    :param type: 'train'或'test'，表示测试集或训练集
    :return:
    """
    if type == 'train':
        root = './datasets/train/'
    elif type == 'test':
        root = './datasets/test/'
    else:
        return
    data = np.empty([nsamples, 28, 28])
    label = np.empty([nsamples])
    cnt = 0
    for i in range(10):
        path = root + str(i) + '/'
        filelist = os.listdir(path)
        for j in tqdm(range(len(filelist))):
            img = cv2.imread(path + filelist[j], 0)
            # img = cv2.imread(path + filelist[random.randrange(len(filelist))], 0)
            img = deskew(img)
            # img = my_tilt_correction(img)
            data[cnt + j] = img
            label[cnt + j] = i
        cnt += len(filelist)

    # 数据和标签以相同形式打乱：
    state = np.random.get_state()
    np.random.shuffle(data)
    np.random.set_state(state)
    np.random.shuffle(label)
    return data, label


def get_model(model_idx, x_train, y_train, x_test, y_test):
    """
    针对不同的模型进行训练
    :param model_idx: 要训练的模型的下标（见model_list）
    :param x_train:
    :param y_train:
    :param x_test:
    :param y_test:
    :return:
    """
    # print(model_name_list[model_idx], '倾斜校正')
    print(model_name_list[model_idx])
    time1 = time.time()
    classifier = model_list[model_idx]
    classifier.fit(x_train, y_train)
    print('训练时间：%.2fs' % (time.time() - time1))

    time1 = time.time()
    y_pred = classifier.predict(x_test)
    print('测试时间：%.2fs' % (time.time() - time1))

    cm = confusion_matrix(y_test, y_pred)
    print('混淆矩阵：', cm, sep='\n')

    print('正确率：', np.diag(cm).sum() / (test_numbers))
    print()


if __name__ == '__main__':

    time1 = time.time()
    x_train, y_train = dataLoader(train_numbers, type='train')
    x_test, y_test = dataLoader(test_numbers, type='test')
    print('数据处理时间：%.2fs' % (time.time() - time1))
    x_train = x_train.reshape(-1, 784)
    x_test = x_test.reshape(-1, 784)

    for i in range(len(model_list) - 1):
        get_model(i, x_train, y_train, x_test, y_test)



