import os
import warnings
from os.path import dirname, abspath, join

import pandas as pd
import albumentations
import torch
import torch.cuda.amp as amp

is_resume_train = False
resume_fold = 0  # zero based
resume_epoch = 1  # one based

assert albumentations.__version__ >= '0.5.2'

warnings.simplefilter('ignore')
scaler = amp.GradScaler()
device = torch.device('cuda')

kernel_type = 'Cls_6epo'
enet_type = 'tf_efficientnet_b1_ns'
num_workers = 4
num_classes = 12
n_ch = 5
image_size = 512
batch_size = 32
init_lr = 3e-4
warmup_epo = 1
cosine_epo = 5
n_epochs = warmup_epo + cosine_epo
print(f'running {n_epochs} epochs: ')

loss_weights = [1., 9.]

base_dir = join(dirname(dirname(dirname(abspath(__file__)))), 'resources')
mask_folder = join(base_dir, 'input/pred_masks/mask_unet++_6epo')
os.makedirs(mask_folder, exist_ok=True)

data_dir = join(base_dir, 'input')
image_folder = 'train_images'

train_csv = join(base_dir, 'input/train_v2.csv')

log_dir = join(base_dir, 'logs')
model_dir = join(base_dir, 'models')

os.makedirs(log_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)
log_file = join(log_dir, f'log_{kernel_type}.txt')

df_train = pd.read_csv(train_csv)

no_ETT = (df_train[['ETT - Abnormal', 'ETT - Borderline', 'ETT - Normal']].values.max(1) == 0).astype(int)
df_train.insert(4, column='no_ETT', value=no_ETT)

# random seed
random_seed = 1234 % 2 ** 32 - 1
import random
import torch
import os
import tensorflow as tf
import numpy as np

random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
np.random.seed(random_seed)
tf.random.set_seed(random_seed)
os.environ['PYTHONHASHSEED'] = str(random_seed)
