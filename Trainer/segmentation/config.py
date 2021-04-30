import os
import warnings
from os.path import dirname, abspath, join

import albumentations
import pandas as pd

import torch
import torch.cuda.amp as amp

is_resume_train = False
resume_fold = 0  # zero based
resume_epoch = 1  # one based

assert albumentations.__version__ >= '0.5.2'

warnings.simplefilter('ignore')
scaler = amp.GradScaler()
device = torch.device('cuda')

kernel_type = 'unet++_6epo'
enet_type = 'timm-efficientnet-b1'
num_workers = 4
image_size = 1024
batch_size = 4
init_lr = 1e-4
warmup_epo = 1
cosine_epo = 5
n_epochs = warmup_epo + cosine_epo
use_amp = True

base_dir = join(dirname(dirname(dirname(abspath(__file__)))), 'resources')
data_dir = join(base_dir, 'input/ranzcr-clip-catheter-line-classification')
image_folder = 'train'

train_annot_csv = join(data_dir, 'train_annotations.csv')
train_csv = join(base_dir, 'input/train_v2.csv')

log_dir = join(base_dir, 'logs')
model_dir = join(base_dir, 'models')

os.makedirs(log_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)
log_file = join(log_dir, f'log_{kernel_type}.txt')

output_dir = join(base_dir, f'mask_{kernel_type}')
os.makedirs(output_dir, exist_ok=True)

df_train = pd.read_csv(train_csv)
df_train_anno = pd.read_csv(train_annot_csv)

# random seed
random_seed = 1234 % 2 ** 32 - 1
import random
import torch
import os
import numpy as np
import tensorflow as tf

random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
np.random.seed(random_seed)
tf.random.set_seed(random_seed)
os.environ['PYTHONHASHSEED'] = str(random_seed)
