"""
paper: Exploit Camera Raw Data for Video Super-Resolution via Hidden Markov Model Inference
file: test_data_raw.py
author: Xiaohong Liu
date: 17/09/19
"""
import os.path as osp
import numpy as np
import torch
import torch.utils.data as data
import data.data_util as util


class TestData(data.Dataset):

    def __init__(self, opt):
        super(TestData, self).__init__()
        self.opt = opt
        self.half_N_frames = opt['N_frames'] // 2
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
        folder = self.data_info['folder'][index]
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

        img_paths_ref = img_paths_LQ[self.half_N_frames]
        img_paths_ref = img_paths_ref.replace('raw', 'rgb')
        img_paths_ref = img_paths_ref.replace('tiff', 'jpg')
        img_paths_ref = img_paths_ref.replace('rgb', 'raw', 1)

        img_ref_l = util.read_img(img_paths_ref)

        for LQ_path in img_paths_LQ:
            img_LQ = util.read_img(LQ_path, israw=True)
            img_LQ_l.append(img_LQ)

        img_LQs = np.stack(img_LQ_l, axis=0)
        img_GT = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1)))).float()
        img_ref_l = torch.from_numpy(np.ascontiguousarray(np.transpose(img_ref_l, (2, 0, 1)))).float()
        img_LQs = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQs,
                                                                     (0, 3, 1, 2)))).float()
        return {'LQs': img_LQs, 'GT': img_GT, 'ref': img_ref_l, 'idx': index, 'folder': folder, 'gt_name': img_paths_GT}

    def __len__(self):
        return len(self.data_info['path_GT'])
