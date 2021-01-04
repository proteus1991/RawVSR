"""
paper: Exploit Camera Raw Data for Video Super-Resolution via Hidden Markov Model Inference
file: train_data_raw.py
author: Xiaohong Liu
date: 17/09/19
"""

import os.path as osp
import random
import numpy as np
import torch
import torch.utils.data as data
import data.data_util as util


class TrainData(data.Dataset):

    def __init__(self, opt):
        super(TrainData, self).__init__()
        self.opt = opt
        self.half_N_frames = opt['N_frames'] // 2
        self.LQ_size = opt['LQ_size']
        self.scale = opt['scale']
        self.GT_root, self.LQ_root = opt['dataroot_GT'], opt['dataroot_LQ']
        self.data_info = {'path_LQ': [], 'path_GT': [], 'folder': [], 'idx': [], 'border': []}

        # Generate data info and cache data
        self.imgs_LQ, self.imgs_GT = {}, {}
        subfolders_LQ = util.glob_file_list(self.LQ_root)
        subfolders_GT = util.glob_file_list(self.GT_root)
        for subfolder_LQ, subfolder_GT in zip(subfolders_LQ, subfolders_GT):
            subfolder_name = osp.basename(subfolder_GT)
            img_paths_LQ = util.glob_file_list(subfolder_LQ)
            img_paths_GT = util.glob_file_list(subfolder_GT)
            max_idx = len(img_paths_LQ)
            assert max_idx == len(
                img_paths_GT), 'Different number of images in LQ and GT folders'
            self.data_info['path_LQ'].extend(img_paths_LQ)
            self.data_info['path_GT'].extend(img_paths_GT)
            self.data_info['folder'].extend([subfolder_name] * max_idx)
            for i in range(max_idx):
                self.data_info['idx'].append('{}/{}'.format(i, max_idx))
            border_l = [0] * max_idx
            for i in range(self.half_N_frames):
                border_l[i] = 1
                border_l[max_idx - i - 1] = 1
            self.data_info['border'].extend(border_l)

    def __getitem__(self, index):
        border = self.data_info['border'][index]
        img_paths_LQ = []
        img_paths_GT = self.data_info['path_GT'][index]
        if border == 1:
            img_paths_LQ = [self.data_info['path_LQ'][index] for _ in range(self.half_N_frames * 2 + 1)]
        else:
            for i in range(self.half_N_frames, -1, -1):
                img_paths_LQ.append(self.data_info['path_LQ'][index - i])
            for i in range(1, self.half_N_frames + 1):
                img_paths_LQ.append(self.data_info['path_LQ'][index + i])

        img_LQ_l = []
        img_GT = util.read_img(img_paths_GT)

        img_paths_lin = img_paths_GT.replace('rgb', 'lin')
        img_paths_lin = img_paths_lin.replace('png', 'tiff')
        img_lin = util.read_img(img_paths_lin, islin=True)

        img_paths_ref = img_paths_LQ[self.half_N_frames]
        img_paths_ref = img_paths_ref.replace('raw', 'rgb')
        img_paths_ref = img_paths_ref.replace('tiff', 'jpg')
        img_paths_ref = img_paths_ref.replace('rgb', 'raw', 1)

        img_ref_l = util.read_img(img_paths_ref)

        for LQ_path in img_paths_LQ:
            img_LQ = util.read_img(LQ_path, israw=True)
            img_LQ_l.append(img_LQ)

        LQ_size = self.LQ_size
        scale = self.scale
        H, W, C = img_LQ.shape
        rnd_h = random.randint(0, max(0, H - LQ_size))
        rnd_w = random.randint(0, max(0, W - LQ_size))
        img_LQ_l = [v[rnd_h:rnd_h + LQ_size, rnd_w:rnd_w + LQ_size, :] for v in img_LQ_l]
        img_ref_l = img_ref_l[rnd_h:rnd_h + LQ_size, rnd_w:rnd_w + LQ_size, :]
        rnd_h_HR, rnd_w_HR = int(rnd_h * scale), int(rnd_w * scale)
        img_GT = img_GT[rnd_h_HR:rnd_h_HR + LQ_size * scale, rnd_w_HR:rnd_w_HR + LQ_size * scale, :]
        img_lin = img_lin[rnd_h_HR:rnd_h_HR + LQ_size * scale, rnd_w_HR:rnd_w_HR + LQ_size * scale, :]

        img_LQs = np.stack(img_LQ_l, axis=0)
        img_GT = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1)))).float()
        img_lin = torch.from_numpy(np.ascontiguousarray(np.transpose(img_lin, (2, 0, 1)))).float()
        img_ref_l = torch.from_numpy(np.ascontiguousarray(np.transpose(img_ref_l, (2, 0, 1)))).float()
        img_LQs = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQs,
                                                                     (0, 3, 1, 2)))).float()
        return {'LQs': img_LQs, 'GT': img_GT, 'LIN': img_lin, 'ref': img_ref_l, 'index': index}

    def __len__(self):
        return len(self.data_info['path_GT'])
