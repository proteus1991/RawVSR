"""
paper: Exploit Camera Raw Data for Video Super-Resolution via Hidden Markov Model Inference
file: utils.py
author: Xiaohong Liu
date: 09/09/19
"""

import time
import torch
import os
import torch.nn.functional as F
import torchvision.utils as utils
import numpy as np
from math import log10


def linear2raw(linear):
    h, w, _ = linear.shape
    raw = np.zeros((h, w))

    # red
    raw[::2, ::2] = linear[::2, ::2, 0]
    # green 1
    raw[::2, 1::2] = linear[::2, 1::2, 1]
    # green 2
    raw[1::2, ::2] = linear[1::2, ::2, 1]
    # blue
    raw[1::2, 1::2] = linear[1::2, 1::2, 2]

    return raw


def to_psnr(derain, gt):
    mse = F.mse_loss(derain, gt, reduction='none')
    mse_split = torch.split(mse, 1, dim=0)
    mse_list = [torch.mean(torch.squeeze(mse_split[ind])).item() for ind in range(len(mse_split))]

    # ToTensor scales input images to [0.0, 1.0]
    intensity_max = 1.0
    psnr_list = [10.0 * log10(intensity_max / mse) for mse in mse_list]
    return psnr_list


def test(net, val_data_loader, device, dataset_name, save_tag=False):
    psnr_list = []
    for batch_id, val_data in enumerate(val_data_loader):

        with torch.no_grad():

            lrs = val_data['LQs'].to(device)
            gt = val_data['GT'].to(device)

            name_gt = [i.split('/')[-1] for i in val_data['gt_name']]

            ref = val_data['ref'].to(device)
            sr, _ = net(lrs, ref)
            # To calculate average PSNR
            psnr_list.extend(to_psnr(sr, gt))

            # Save image
            if save_tag:
                save_image(sr.clone(), name_gt, dataset_name)

    avr_psnr = sum(psnr_list) / len(psnr_list)

    print('--- testing results ---')
    print('store: {0:.2f}dB'.format(sum(psnr_list[0:31]) / len(psnr_list[0:31])))
    print('painting: {0:.2f}dB'.format(sum(psnr_list[31:62]) / len(psnr_list[31:62])))
    print('train: {0:.2f}dB'.format(sum(psnr_list[62:93]) / len(psnr_list[62:93])))
    print('city: {0:.2f}dB'.format(sum(psnr_list[93:124]) / len(psnr_list[93:124])))
    print('tree: {0:.2f}dB'.format(sum(psnr_list[124:155]) / len(psnr_list[124:155])))
    print('avg_psnr: {0:.2f}dB'.format(avr_psnr))
    print('--- end ---')
    return avr_psnr


def save_image(image, image_name, category):
    images = torch.split(image, 1, dim=0)
    batch_num = len(images)

    path = './{}_results/'.format(category)
    if not os.path.exists(path):
        os.mkdir(path)

    for ind in range(batch_num):
        # this function changes the image to range(0, 255)
        utils.save_image(images[ind], path + '{}'.format(image_name[ind].split('.')[0] + '.png'))


def print_log(epoch, num_epochs, one_epoch_time, train_psnr, val_psnr, category):
    print('({0:.0f}s) Epoch [{1}/{2}], Train_PSNR:{3:.2f}, Val_PSNR:{4:.2f}'
          .format(one_epoch_time, epoch, num_epochs, train_psnr, val_psnr))
    # write training log
    path = './training_log/'
    if not os.path.exists(path):
        os.mkdir(path)

    with open(path + '{}_log.txt'.format(category), 'a+') as f:
        print(
            'Date: {0}s, Time_Cost: {1:.0f}s, Epoch: [{2}/{3}], Train_PSNR: {4:.2f}, Val_PSNR: {5:.2f}'
                .format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                        one_epoch_time, epoch, num_epochs, train_psnr, val_psnr), file=f)


def adjust_learning_rate(optimizer, epoch, num_epochs, lr_decay=0.5):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    step = num_epochs // 1

    if not epoch % step and epoch > 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_decay
            print('Learning rate sets to {}.'.format(param_group['lr']))
    else:
        for param_group in optimizer.param_groups:
            print('Learning rate sets to {}.'.format(param_group['lr']))


if __name__ == '__main__':
    import torchvision

    print(torchvision.__version__)
