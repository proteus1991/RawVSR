"""
paper: Exploit Camera Raw Data for Video Super-Resolution via Hidden Markov Model Inference
file: config.py
author: Xiaohong Liu
date: 17/09/19
"""


def get_config(args):
    scale = args.scale_ratio
    save_tag = args.save_image

    if scale not in [2, 4]:
        raise Exception('scale {} is not supported!'.format(scale))

    opt = {'train': {'dataroot_GT': './dataset/train/1080p_gt_rgb',
                     'dataroot_LQ': './dataset/train/1080p_lr_d_raw_{}'.format(scale),
                     'lr': 2e-4,
                     'num_epochs': 100,
                     'N_frames': 7,
                     'n_workers': 12,
                     'batch_size': 24 if scale == 4 else 8,
                     'GT_size': 256,
                     'LQ_size': 256 // scale,
                     'scale': scale,
                     'phase': 'train',
                     },

           'test': {'dataroot_GT': './dataset/test/1080p_gt_rgb',
                    'dataroot_LQ': './dataset/test/1080p_lr_d_raw_{}'.format(scale),
                    'N_frames': 7,
                    'n_workers': 12,
                    'batch_size': 2,
                    'phase': 'test',
                    'save_image': save_tag,
                    },

           'network': {'nf': 64,
                       'nframes': 7,
                       'groups': 8,
                       'back_RBs': 4},

           'dataset': {'dataset_name': 'RawVD'
                       }
           }

    return opt
