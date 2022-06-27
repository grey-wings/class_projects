import os

import sys
import time
import torch
import torch.nn

from utils import evaluate, get_dataset, FFDataset, setup_logger
from trainer import Trainer
import numpy as np
import random

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
osenvs = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
# config
dataset_path = '/home/PointCloud/F3_Net2/dataset_ffpp/ffpp_face_only'
pretrained_path = './pretrained/xception-b5690688.pth'
batch_size = 32
gpu_ids = [*range(osenvs)]
max_epoch = 4
loss_freq = 40
mode = 'FAD'  # ['Original', 'FAD', 'LFS', 'Both', 'Mix']
ckpt_dir = './checkpoints/220428'
ckpt_name = 'FAD4_bz128'

if __name__ == '__main__':
    dataset = FFDataset(dataset_root=os.path.join(dataset_path, 'train', 'real'), size=299, photo_num=11200,
                        augment=True)
    dataloader_real = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size // 2,
        shuffle=True,
        num_workers=8)

    len_dataloader = dataloader_real.__len__()

    # dataset_img, total_len = get_dataset(name='train', size=299, root=dataset_path, photo_num=8400, augment=True)
    dataset_img = FFDataset(dataset_root=os.path.join(dataset_path, 'train', 'fake'), size=299, photo_num=11200,
                            augment=True)
    # dataset_img = get_dataset(name='train', size=299, root=dataset_path, photo_num=4800, augment=True)
    dataloader_fake = torch.utils.data.DataLoader(
        dataset=dataset_img,
        batch_size=batch_size // 2,
        shuffle=True,
        num_workers=8
    )

    # init checkpoint and logger
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    logger = setup_logger(ckpt_path, 'result.log', 'logger')
    logger.info('batch_size:{}'.format(batch_size))
    logger.info('max_epoch:{}'.format(max_epoch))
    logger.info('loss_freq:{}'.format(loss_freq))
    logger.info('pretrained_path:{}'.format(pretrained_path))
    best_val = 0.
    ckpt_model_name = 'best.pkl'

    # train
    model = Trainer(gpu_ids, mode, pretrained_path)
    model.total_steps = 0
    epoch = 0

    while epoch < max_epoch:

        fake_iter = iter(dataloader_fake)
        real_iter = iter(dataloader_real)

        logger.debug(f'No {epoch}')
        i = 0
        cnt = 0

        while i < len_dataloader:

            i += 1
            model.total_steps += 1

            try:
                data_real = real_iter.next()
                data_fake = fake_iter.next()
            except StopIteration:
                break
            # -------------------------------------------------

            if data_real.shape[0] != data_fake.shape[0]:
                continue

            bz = data_real.shape[0]

            data = torch.cat([data_real, data_fake], dim=0)
            label = torch.cat([torch.zeros(bz).unsqueeze(dim=0), torch.ones(bz).unsqueeze(dim=0)], dim=1).squeeze(dim=0)

            # manually shuffle
            idx = list(range(data.shape[0]))
            random.shuffle(idx)
            data = data[idx]
            label = label[idx]

            data = data.detach()
            label = label.detach()

            model.set_input(data, label)
            loss = model.optimize_weight()

            if model.total_steps % loss_freq == 0:
                logger.debug(f'loss: {loss} at step: {model.total_steps}')
            if i % int(len_dataloader / 4) == 0:
                cnt += 1
                model.model.eval()
                auc, r_acc, f_acc = evaluate(model, dataset_path, photo_num=208, mode='valid')
                logger.debug(f'(Val @ epoch {epoch}, {cnt}th save) auc: {auc}, r_acc: {r_acc}, f_acc:{f_acc}')
                # auc, r_acc, f_acc = evaluate(model, dataset_path, mode='test')
                # logger.debug(f'(Test @ epoch {epoch}, {cnt}th save) auc: {auc}, r_acc: {r_acc}, f_acc:{f_acc}')
                model.save(os.path.join(ckpt_path, 'epoch_{}_{}.pth'.format(epoch, cnt)))
                model.model.train()
        epoch = epoch + 1

    model.model.eval()
    auc, r_acc, f_acc = evaluate(model, dataset_path, mode='test')
    logger.debug(f'(Test @ epoch {epoch}) auc: {auc}, r_acc: {r_acc}, f_acc:{f_acc}')
