import sys
from os.path import dirname, abspath

import torch

labels = [
    'ETT - Abnormal',
    'ETT - Borderline',
    'ETT - Normal',
    'NGT - Abnormal',
    'NGT - Borderline',
    'NGT - Incompletely Imaged',
    'NGT - Normal',
    'CVC - Abnormal',
    'CVC - Borderline',
    'CVC - Normal',
    'Swan Ganz Catheter Present'
]

prefix = dirname(dirname(dirname(dirname(abspath(__file__)))))
model_dir = f'{prefix}/resources/models'

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

num_workers = 2
image_size = 512
batch_size = 8

enet_type_seg = 'timm-efficientnet-b1'
kernel_type_seg = 'unet++_6epo'

enet_type_cls = 'tf_efficientnet_b1_ns'
kernel_type_cls = 'Cls_6epo'
