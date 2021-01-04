"""
paper: Exploit Camera Raw Data for Video Super-Resolution via Hidden Markov Model Inference
file: train.py
author: Xiaohong Liu
date: 17/09/19
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
from data.data_util import create_dataloader
from config import get_config
from utils.utils import to_psnr, print_log, test, adjust_learning_rate
from data.train_data_raw import TrainData
from data.test_data_raw import TestData
from models.model import RawVSR
from pytorch_msssim import SSIM

# --- Parse hyper-parameters  --- #
parser = argparse.ArgumentParser(description='RawVSR')
parser.add_argument('--scale_ratio', default=4, type=int, help='Set the SR scaling rate.')
parser.add_argument('--save_image', action='store_true', help='Save image')
args = parser.parse_args()

opt = get_config(args)

# Gpu device
device_ids = [Id for Id in range(torch.cuda.device_count())]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net = RawVSR(nf=opt['network']['nf'], nframes=opt['network']['nframes'], scale=opt['train']['scale'],
             groups=opt['network']['groups'], back_RBs=opt['network']['back_RBs'])

# build optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=opt['train']['lr'])


class SSIM_Loss(SSIM):
    def forward(self, img1, img2):
        return 1 - super(SSIM_Loss, self).forward(img1, img2)


ssim_loss = SSIM_Loss(data_range=1.0, size_average=True, channel=3)

# multi-GPU
net = net.to(device)
net = nn.DataParallel(net, device_ids=device_ids)

# Load Network weight
try:
    weight_name = './weight_checkpoint/rawvsr_{}_{}_{}_{}_raw_best.pkl'.format(opt['train']['scale'],
                                                                              opt['network']['nframes'],
                                                                              opt['network']['nf'],
                                                                              opt['network']['back_RBs'])
    net.load_state_dict(torch.load(weight_name))
    print('Weight loading succeeds: {}'.format(weight_name))
except:
    print('Weight loading fails.')

# calculate all trainable parameters in network
pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print("Total_params: {}".format(pytorch_total_params))


# print configs
print('----- Hyper-parameter default settings -----')
for k, v in opt.items():
    print('--- {} settings ---'.format(k))
    for kk, vv in v.items():
        print('{}: {}'.format(kk, vv))

# Load training data and test data.
# Additional dimension about batch size is added to all the data in Dataset.
train_data = TrainData(opt['train'])
test_data = TestData(opt['test'])

train_data_loader = create_dataloader(train_data, opt['train'])
test_data_loader = create_dataloader(test_data, opt['test'])

# previous test PSNR
dataset_name = opt['dataset']['dataset_name']

previous_test_psnr = test(net, test_data_loader, device, dataset_name)
print('previous_test_psnr:{0:.2f}'.format(previous_test_psnr))

num_epochs = opt['train']['num_epochs']
for epoch in range(num_epochs):
    psnr_list = []
    loss_list = []
    running_loss = []
    loss_mse_list = []
    loss_ssim_list = []
    loss_ssim_lin_list = []

    start_time = time.time()
    adjust_learning_rate(optimizer, epoch, num_epochs)

    for batch_id, train_data in enumerate(train_data_loader):

        lrs = train_data['LQs'].to(device)
        lin_gt = train_data['LIN'].to(device)
        gt = train_data['GT'].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        net.train()

        ref = train_data['ref'].to(device)
        sr, lin_sr = net(lrs, ref)
        loss_mse = F.mse_loss(sr, gt)
        loss_ssim = ssim_loss(sr, gt)
        loss_ssim_lin = ssim_loss(lin_sr, gt)
        loss = loss_mse + 0.005 * loss_ssim + 0.005 * loss_ssim_lin

        loss_mse_list.append(loss_mse.item())
        loss_ssim_list.append(0.005 * loss_ssim.item())
        loss_ssim_lin_list.append(0.005 * loss_ssim_lin.item())

        loss.backward()
        optimizer.step()

        # To save PSNR and running loss
        psnr_list.extend(to_psnr(sr, gt))
        running_loss.append(loss.item())

        # print out
        if not (batch_id % 100):
            print('Epoch:{0}, Iteration:{1}'.format(epoch, batch_id))

    # Average PSNR on one epoch train_data
    train_psnr = sum(psnr_list) / len(psnr_list)
    train_loss = sum(running_loss) / len(running_loss)

    print('MSE loss: {0:.4f}, SSIM loss: {1:.4f}, SSIM linear loss: {2:.4f}'.format(np.mean(loss_mse_list),
                                                                                    np.mean(loss_ssim_list),
                                                                                    np.mean(loss_ssim_lin_list)))

    # save the network parameters
    torch.save(net.state_dict(),
               './weight_checkpoint/rawvsr_{}_{}_{}_{}_{}_raw.pkl'.format(opt['train']['scale'],
                                                                         opt['network']['nframes'],
                                                                         opt['network']['nf'],
                                                                         opt['network']['back_RBs'], epoch))

    # use evaluation models during the net evaluating
    net.eval()

    test_psnr = test(net, test_data_loader, device, dataset_name)
    one_epoch_time = time.time() - start_time
    print_log(epoch + 1, num_epochs, one_epoch_time, train_psnr, test_psnr, dataset_name)

    if test_psnr >= previous_test_psnr:
        torch.save(net.state_dict(),
                   './weight_checkpoint/rawvsr_{}_{}_{}_{}_raw_best.pkl'.format(opt['train']['scale'],
                                                                               opt['network']['nframes'],
                                                                               opt['network']['nf'],
                                                                               opt['network']['back_RBs']))
        previous_test_psnr = test_psnr
