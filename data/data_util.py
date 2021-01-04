"""
paper: Exploit Camera Raw Data for Video Super-Resolution via Hidden Markov Model Inference
file: data_util.py
author: Xiaohong Liu
date: 27/09/19
"""

import os
import numpy as np
import glob
import torch
from PIL import Image
from skimage import io


def glob_file_list(root):
    return sorted(glob.glob(os.path.join(root, '*')))


def read_img(path, israw=False, islin=False):
    user_black = 2047
    user_sat = 16200

    if israw:
        img = Image.open(path)
        img = np.array(img)
        img = (img - user_black) / (user_sat - user_black)
        img = np.clip(img, 0, 1).astype(np.float32)
        img = img[:, :, None]

    elif islin:
        img = io.imread(path)
        img = img / (2 ** 16 - 1)
        img = np.clip(img, 0, 1).astype(np.float32)

    else:
        img = Image.open(path)
        w, h = img.size
        img = np.array(img)
        img = img / 255.
        img = np.clip(img, 0, 1).astype(np.float32)

    return img


def create_dataloader(dataset, dataset_opt):
    phase = dataset_opt['phase']
    if phase == 'train':
        num_workers = dataset_opt['n_workers']
        batch_size = dataset_opt['batch_size']
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                           num_workers=num_workers, drop_last=True,
                                           pin_memory=False)
    else:
        num_workers = dataset_opt['n_workers']
        batch_size = dataset_opt['batch_size']
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                           pin_memory=False)