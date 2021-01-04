"""
paper: Exploit Camera Raw Data for Video Super-Resolution via Hidden Markov Model Inference
file: color_correction.py
author: Xiaohong Liu
date: 17/09/19
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from blocks import ChannelAttention


class RawVSR(nn.Module):
    def __init__(self):
        super(RawVSR, self).__init__()

        self.fuseGreen = nn.Conv2d(2, 1, kernel_size=1, stride=1)

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(64, 16, kernel_size=3, stride=2, padding=1)
        self.conv_out = nn.Conv2d(32, 9, kernel_size=3, stride=1, padding=1)
        self.white_balance = ChannelAttention(3)

    def forward(self, x, ref_rgb_lr):
        r = x[:, :, ::2, ::2]
        g1 = x[:, :, ::2, 1::2]
        g2 = x[:, :, 1::2, ::2]
        b = x[:, :, 1::2, 1::2]

        r_bilinear = F.interpolate(r, scale_factor=2, mode='bilinear', align_corners=True)
        b_bilinear = F.interpolate(b, scale_factor=2, mode='bilinear', align_corners=True)

        g = torch.cat([g1, g2], dim=1)
        g = self.fuseGreen(g)
        g_bilinear = F.interpolate(g, scale_factor=2, mode='bilinear', align_corners=True)

        rgb = torch.cat([r_bilinear, g_bilinear, b_bilinear], dim=1)
        wb = self.white_balance(rgb)
        rgb = rgb * wb

        conv1 = F.relu(self.conv1(ref_rgb_lr))
        conv2 = F.relu(self.conv2(conv1))
        conv3 = F.relu(self.conv3(conv2))
        conv4 = F.relu(self.conv4(conv3))
        deconv3 = F.relu(self.deconv3(conv4, conv2.size()))
        deconv2 = F.relu(self.deconv2(torch.cat([deconv3, conv2], dim=1), conv1.size()))
        c3 = F.relu(self.conv_out(torch.cat([deconv2, conv1], dim=1)))

        # color correction
        sr_cat = torch.cat([rgb, rgb, rgb], dim=1)
        sr_m = sr_cat * c3
        n, c, h, w = sr_m.size()
        rgb_corrected = sr_m.reshape(n, c // 3, 3, h, w).sum(2)

        return rgb_corrected
