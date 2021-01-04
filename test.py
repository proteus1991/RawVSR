"""
paper: Exploit Camera Raw Data for Video Super-Resolution via Hidden Markov Model Inference
file: test.py
author: Xiaohong Liu
date: 01/08/19
"""

import torch
import argparse
import torch.nn as nn
from data.data_util import create_dataloader
from config import get_config
from utils.utils import test
from data.test_data_raw import TestData
from models.model import RawVSR

# --- Parse hyper-parameters  --- #
parser = argparse.ArgumentParser(description='RawVSR')
parser.add_argument('--scale_ratio', default=4, type=int, help='Set the SR scaling rate.')
parser.add_argument('--save_image', action='store_true', help='Save image')
args = parser.parse_args()


# Gpu device
device_ids = [Id for Id in range(torch.cuda.device_count())]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# test data loader
opt = get_config(args)
test_data = TestData(opt['test'])
test_data_loader = create_dataloader(test_data, opt['test'])

# establish network
net = RawVSR(nf=opt['network']['nf'], nframes=opt['network']['nframes'], groups=opt['network']['groups'],
             scale=opt['train']['scale'], back_RBs=opt['network']['back_RBs'])

# multi-GPU
net = net.to(device)
net = nn.DataParallel(net, device_ids=device_ids)

# load data
net.load_state_dict(torch.load(
    './weight_checkpoint/rawvsr_{}_{}_{}_{}_raw_best.pkl'.format(opt['train']['scale'], opt['network']['nframes'],
                                                                opt['network']['nf'], opt['network']['back_RBs'])))

# print configs
print('----- Hyper-parameter default settings -----')
for k, v in opt.items():
    print('--- {} settings ---'.format(k))
    for kk, vv in v.items():
        print('{}: {}'.format(kk, vv))


# network evaluation
net.eval()
dataset_name = opt['dataset']['dataset_name']
test_psnr = test(net, test_data_loader, device, dataset_name, opt['test']['save_image'])
