"""
paper: Exploit Camera Raw Data for Video Super-Resolution via Hidden Markov Model Inference
file: models.py
author: Xiaohong Liu
date: 17/09/19
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.align_net import AlignNet
from models.backbone import BackBone
from models.feature_extraction import FeatureExtraction


class RawVSR(nn.Module):
    def __init__(self, nf=64, nframes=7, scale=4, groups=8, back_RBs=3, kernel_size=3):
        super(RawVSR, self).__init__()

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # demosaicing
        self.demosaic = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=kernel_size, padding=(kernel_size - 1) // 2),
            self.lrelu,
            nn.Conv2d(64, 64, kernel_size=kernel_size, padding=(kernel_size - 1) // 2),
            self.lrelu,
            nn.PixelShuffle(2),
            self.lrelu,
            nn.Conv2d(16, 64, kernel_size=kernel_size, padding=(kernel_size - 1) // 2),
            self.lrelu,
            nn.Conv2d(64, 64, kernel_size=kernel_size, padding=(kernel_size - 1) // 2),
            self.lrelu,
        )

        # linear rgb to feature
        self.conv_first = nn.Conv2d(3, 64, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.feature_extraction = FeatureExtraction()

        # alignment
        self.align_net = AlignNet(nf=nf, groups=groups, num_frames=nframes)

        # build backbone
        self.recon_trunk = BackBone(nf, back_RBs)

        # upsampling
        if scale == 4:
            self.sr_conv1 = nn.Conv2d(nf, nf * 4, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
            self.sr_conv2 = nn.Conv2d(nf, nf * 4, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
            self.pixel_shuffle = nn.PixelShuffle(2)
            self.sr_conv3 = nn.Conv2d(nf, nf, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
            self.sr_conv4 = nn.Conv2d(nf, 3, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)

            # color reference
            self.c_conv1 = nn.Conv2d(nf, nf * 4, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
            self.c_conv2 = nn.Conv2d(nf, nf * 4, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
            self.c_conv3 = nn.Conv2d(nf, nf, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
            self.c_conv4 = nn.Conv2d(nf, 9, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)

        elif scale == 2:
            self.sr_conv1 = nn.Conv2d(nf, nf * 4, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
            self.pixel_shuffle = nn.PixelShuffle(2)
            self.sr_conv3 = nn.Conv2d(nf, nf, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
            self.sr_conv4 = nn.Conv2d(nf, 3, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)

            # color reference
            self.c_conv1 = nn.Conv2d(nf, nf * 4, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
            self.c_conv3 = nn.Conv2d(nf, nf, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
            self.c_conv4 = nn.Conv2d(nf, 9, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        else:
            raise Exception('scale {} is not supported!'.format(scale))

        # output
        self.conv_out = nn.Conv2d(3, 3, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)

        # parameters
        self.scale = scale

    def forward(self, x, ref):
        # N video frames
        B, N, C, H, W = x.size()

        lins = []
        for i in range(N):
            raw = x[:, i, :, :, :].clone()

            r = raw[:, :, ::2, ::2]
            g1 = raw[:, :, ::2, 1::2]
            g2 = raw[:, :, 1::2, ::2]
            b = raw[:, :, 1::2, 1::2]

            lin = torch.cat([r, g1, g2, b], dim=1)
            lins.append(lin)

        # 4 -> 64, 64 -> 64, pixel_shuffle
        feas = [self.demosaic(lin) for lin in lins]

        # linear rgb to feature
        feas = [self.feature_extraction(fea) for fea in feas]

        # alignment
        align_fea = self.align_net(feas)

        # build backbone
        fea = self.recon_trunk(align_fea)

        # color reference
        ref_fea = self.lrelu(self.conv_first(ref))
        ref_fea = self.feature_extraction(ref_fea)

        ref_fea = self.recon_trunk(ref_fea)

        # upscale
        if self.scale == 4:
            sr = self.lrelu(self.pixel_shuffle(self.sr_conv1(fea)))
            sr = self.lrelu(self.pixel_shuffle(self.sr_conv2(sr)))
            sr = self.lrelu(self.sr_conv3(sr))
            sr = self.lrelu(self.sr_conv4(sr))

            color_matrix = self.lrelu(self.pixel_shuffle(self.c_conv1(ref_fea)))
            color_matrix = self.lrelu(self.pixel_shuffle(self.c_conv2(color_matrix)))
            color_matrix = self.lrelu(self.c_conv3(color_matrix))
            color_matrix = self.lrelu(self.c_conv4(color_matrix))

        elif self.scale == 2:
            sr = self.lrelu(self.pixel_shuffle(self.sr_conv1(fea)))
            sr = self.lrelu(self.sr_conv3(sr))
            sr = self.lrelu(self.sr_conv4(sr))

            color_matrix = self.lrelu(self.pixel_shuffle(self.c_conv1(ref_fea)))
            color_matrix = self.lrelu(self.c_conv3(color_matrix))
            color_matrix = self.lrelu(self.c_conv4(color_matrix))
        else:
            raise Exception('scale {} is not supported!'.format(self.scale))

        lin_out = sr.clone()

        # color correction
        sr_cat = torch.cat([sr, sr, sr], dim=1)
        sr_m = sr_cat * color_matrix
        n, c, h, w = sr_m.size()
        out = sr_m.reshape(n, c//3, 3, h, w).sum(2)
        out = F.relu(out)

        return out, lin_out
