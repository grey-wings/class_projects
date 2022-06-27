import os

import sys
import time
import torch
import torch.nn
import dlib
from torch.utils import data
from torchvision import transforms as trans
from PIL import Image, ImageFile
from sklearn.metrics import roc_curve, accuracy_score
from sklearn.metrics import auc as cal_auc
from trainer import Trainer
import numpy as np
import random
from utils import FFDataset


def evaluate(model, data_path, photo_num=5000, mode='valid'):
    root = data_path
    origin_root = root
    root = os.path.join(data_path, mode)
    real_root = os.path.join(root, 'real')
    fake_root = os.path.join(root, 'fake')
    dataset_real = FFDataset(dataset_root=real_root, size=299, photo_num=photo_num, augment=False)
    # dataset_fake, _ = get_dataset(name=mode, root=origin_root, size=299, augment=False)
    dataset_fake = FFDataset(dataset_root=fake_root, size=299, photo_num=photo_num, augment=False)
    dataset_img = torch.utils.data.ConcatDataset([dataset_real, dataset_fake])

    bz = 63
    # torch.cache.empty_cache()
    with torch.no_grad():
        y_true, y_pred = [], []

        for i, d in enumerate(dataset_img.datasets):
            dataloader = torch.utils.data.DataLoader(
                dataset=d,
                batch_size=bz,
                shuffle=True,
                num_workers=8
            )
            for img in dataloader:
                if i == 0:
                    label = torch.zeros(img.size(0))
                else:
                    label = torch.ones(img.size(0))
                img = img.detach().cuda()
                output = model.forward(img)
                y_pred.extend(output.sigmoid().flatten().tolist())
                y_true.extend(label.flatten().tolist())

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=1)
    AUC = cal_auc(fpr, tpr)

    idx_real = np.where(y_true == 0)[0]
    idx_fake = np.where(y_true == 1)[0]

    r_acc = accuracy_score(y_true[idx_real], y_pred[idx_real] > 0.5)
    f_acc = accuracy_score(y_true[idx_fake], y_pred[idx_fake] > 0.5)
    t_acc = (r_acc * dataset_real.__len__() + f_acc * dataset_fake.__len__()) / \
            (dataset_fake.__len__() + dataset_real.__len__())

    return AUC, r_acc, f_acc, t_acc


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
osenvs = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
# config
dataset_path = '/home/PointCloud/F3_Net2/tests/group3/'
pretrained_path = './pretrained/xception-b5690688.pth'
gpu_ids = [*range(osenvs)]
max_epoch = 1
loss_freq = 40
mode = ['Original', 'FAD', 'Both', 'Original', 'Both']
modeinfo = ['Original', 'FAD', 'Both', '摆烂', '究极摆烂']
# ['Original', 'FAD', 'LFS', 'Both', 'Mix']
ckpt_dir = './checkpoints/220427'
load_paths = ['/home/PointCloud/F3_Net2/checkpoints/220427/Base4_bz128/epoch_1_1.pth',
              '/home/PointCloud/F3_Net2/checkpoints/220427/FAD4_bz128/epoch_1_2.pth',
              '/home/PointCloud/F3_Net2/checkpoints/220427/Both4_bz128/epoch_2_4.pth',
              '/home/PointCloud/F3_Net2/checkpoints/220427/Base4_bz128/epoch_0_3.pth',
              '/home/PointCloud/F3_Net2/checkpoints/220427/Both4_bz128/epoch_3_3.pth']

if __name__ == '__main__':
    print("group 3")
    for i in range(len(load_paths)):
        model = Trainer(gpu_ids, mode[i], pretrained_path)
        load_path = load_paths[i]
        model.load(load_path)
        model.total_steps = 0
        epoch = 0

        model.model.eval()
        auc, r_acc, f_acc, t_acc = evaluate(model, dataset_path, mode='test')
        print(f'mode:{modeinfo[i]}, auc: {auc}, r_acc: {r_acc}, f_acc:{f_acc}, t_acc:{t_acc}')
        # auc, acc = evaluate_labels(model, dataset_path, label_path)
