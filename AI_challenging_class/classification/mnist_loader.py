import os
import cv2
import numpy as np
import sklearn


import os
import struct
import cv2
from tqdm import tqdm
import pickle
import numpy as np
import matplotlib.pyplot as plt


# 读MNIST数据集的图片数据
def mnist_load_img(img_path):
    with open(img_path, "rb") as fp:
        # >是以大端模式读取，i是整型模式，读取前四位的标志位，
        # unpack()函数：是将4个字节联合后再解析成一个数，(读取后指针自动后移)
        msb = struct.unpack('>i', fp.read(4))[0]
        # 标志位为2051，后存图像数据；标志位为2049，后存图像标签
        # fp.read后面是字节数
        if msb == 2051:
            # 读取样本个数60000，存入cnt
            cnt = struct.unpack('>i', fp.read(4))[0]
            # rows：行数28；cols：列数28
            rows = struct.unpack('>i', fp.read(4))[0]
            cols = struct.unpack('>i', fp.read(4))[0]
            imgs = np.empty((cnt, rows, cols), dtype="int")
            for i in tqdm(range(cnt)):
                for j in range(0, rows):
                    for k in range(0, cols):
                        # 16进制转10进制
                        pxl = int(hex(fp.read(1)[0]), 16)
                        imgs[i][j][k] = pxl
            return imgs
        else:
            return np.empty(1)


# 读MNIST数据集的图片标签
def mnist_load_label(label_path):
    with open(label_path, "rb") as fp:
        msb = struct.unpack('>i', fp.read(4))[0]
        if msb == 2049:
            cnt = struct.unpack('>i', fp.read(4))[0]
            labels = np.empty(cnt, dtype="int")
            for i in range(0, cnt):
                label = int(hex(fp.read(1)[0]), 16)
                labels[i] = label
            return labels
        else:
            return np.empty(1)


# 分割训练、测试集的图片数据与图片标签
def mnist_load_data(train_img_path, train_label_path, test_img_path, test_label_path):
    x_train = mnist_load_img(train_img_path)
    y_train = mnist_load_label(train_label_path)
    x_test = mnist_load_img(test_img_path)
    y_test = mnist_load_label(test_label_path)
    return x_train, y_train, x_test, y_test


# 输出打印图片
def mnist_plot_img(img):
    (rows, cols) = img.shape
    plt.figure()
    plt.gray()
    plt.imshow(img)
    plt.show()


# 按指定位置保存图片
def mnist_save_img(img, path, name):
    if not os.path.exists(path):
        os.mkdir(path)
    cv2.imwrite(path + name, img)
    # (rows, cols) = img.shape
    # plt.tight_layout(pad=0)
    # fig = plt.figure()
    # plt.gray()
    # current_axes = plt.axes()
    # current_axes.xaxis.set_visible(False)
    # current_axes.yaxis.set_visible(False)
    # plt.imshow(img)
    # # 在既定路径里保存图片
    # fig.savefig(path + name)
    # plt.close('all')


def generateTrainImages(beginNum, endNum):
    """
    取训练集中前num个图片，以png形式保存。
    :param num:
    :return:
    """
    with open(os.path.join('train-images.dat'), 'rb') as f:
        x_train = pickle.load(f)
    with open(os.path.join('train-labels.dat'), 'rb') as f:
        y_train = pickle.load(f)
    # 按图片标签的不同，打印MNIST数据集的图片存入不同文件夹下
    for i in tqdm(range(beginNum, endNum)):
        path = "./datasets/train/" + str(y_train[i]) + "/"
        name = str(i) + ".png"
        mnist_save_img(x_train[i], path, name)


def generateTestImages(beginNum, endNum):
    """
    取训练集中前num个图片，以png形式保存。
    :param num:
    :return:
    """
    with open(os.path.join('test-images.dat'), 'rb') as f:
        x_test = pickle.load(f)
    with open(os.path.join('test-labels.dat'), 'rb') as f:
        y_test = pickle.load(f)

    for i in tqdm(range(beginNum, endNum)):
        path = "./datasets/test/" + str(y_test[i]) + "/"
        name = str(i) + ".png"
        mnist_save_img(x_test[i], path, name)


def dataLoader(nsamples=10000, type='train', shuffle=True):
    """

    :param nsamples: train: 60000个，test: 10000个
    :param type: 'train'或'test'，表示测试集或训练集
    :return:
    """
    if type == 'train':
        root = './datasets/train/'
    elif type == 'test':
        root = './datasets/test/'
    else:
        return
    data = np.empty([nsamples * 10, 28, 28])
    label = np.empty([nsamples * 10])
    for i in range(10):
        path = root + str(i) + '/'
        filelist = os.listdir(path)
        for j in range(nsamples):
            img = cv2.imread(path + filelist[j], 0)
            # img = cv2.imread(path + filelist[random.randrange(len(filelist))], 0)
            # img = deskew(img)
            # img = my_tilt_correction(img)
            data[i * nsamples + j] = img
            label[i * nsamples + j] = i

    if shuffle:
        # 数据和标签以相同形式打乱：
        state = np.random.get_state()
        np.random.shuffle(data)
        np.random.set_state(state)
        np.random.shuffle(label)
    return data, label


if __name__ == '__main__':
    # x_train, y_train, x_test, y_test = mnist_load_data('train-images.idx3-ubyte',
    #                                                    'train-labels.idx1-ubyte',
    #                                                    't10k-images.idx3-ubyte',
    #                                                    't10k-labels.idx1-ubyte')
    # with open(os.path.join('train-images.dat'), 'wb') as f:
    #     pickle.dump(x_train, f)
    # with open(os.path.join('train-labels.dat'), 'wb') as f:
    #     pickle.dump(y_train, f)
    # with open(os.path.join('test-images.dat'), 'wb') as f:
    #     pickle.dump(x_test, f)
    # with open(os.path.join('test-labels.dat'), 'wb') as f:
    #     pickle.dump(y_test, f)

    with open(os.path.join('train-images.dat'), 'rb') as f:
        x_train = pickle.load(f)
    with open(os.path.join('train-labels.dat'), 'rb') as f:
        y_train = pickle.load(f)
    with open(os.path.join('test-images.dat'), 'rb') as f:
        x_test = pickle.load(f)
    with open(os.path.join('test-labels.dat'), 'rb') as f:
        y_test = pickle.load(f)
    # 按图片标签的不同，打印MNIST数据集的图片存入不同文件夹下
    for i in tqdm(range(60000)):
        path = "./datasets/train/" + str(y_train[i]) + "/"
        # path = "./datasets/train/"
        if not os.path.exists(path):
            os.makedirs(path)
        name = str(i) + ".png"
        mnist_save_img(x_train[i], path, name)

    # for i in tqdm(range(10000)):
    #     path = "./datasets/test/" + str(y_test[i]) + "/"
    #     if not os.path.exists(path):
    #         os.makedirs(path)
    #     name = str(i) + ".png"
    #     mnist_save_img(x_test[i], path, name)


