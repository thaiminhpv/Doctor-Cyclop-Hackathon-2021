import ast
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from config import *


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class DC_Dataset(Dataset):

    def __init__(self, df, mode, transform=None):

        self.df = df.reset_index(drop=True)
        self.mode = mode
        self.transform = transform

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        row = self.df.iloc[index]
        image = cv2.imread(os.path.join(data_dir, image_folder, row.StudyInstanceUID + '.jpg'))[:, :, ::-1]

        if self.mode == 'test':
            mask = None
            res = self.transform(image=image)
        else:
            df_this = df_train_anno.query(f'StudyInstanceUID == "{row.StudyInstanceUID}"')
            mask = np.zeros((image.shape[0], image.shape[1], 2)).astype(np.uint8)
            for _, anno in df_this.iterrows():
                anno_this = np.array(ast.literal_eval(anno["data"]))
                mask1 = mask[:, :, 0].copy()
                mask1 = cv2.polylines(mask1, np.int32([anno_this]), isClosed=False, color=1, thickness=15, lineType=16)
                mask[:, :, 0] = mask1
                mask2 = mask[:, :, 1].copy()
                mask2 = cv2.circle(mask2, (anno_this[0][0], anno_this[0][1]), radius=15, color=1, thickness=25)
                mask2 = cv2.circle(mask2, (anno_this[-1][0], anno_this[-1][1]), radius=15, color=1, thickness=25)
                mask[:, :, 1] = mask2

            mask = cv2.resize(mask, (image_size, image_size))
            mask = (mask > 0.5).astype(np.uint8)
            res = self.transform(image=image, mask=mask)

        image = res['image'].astype(np.float32).transpose(2, 0, 1) / 255.

        if self.mode == 'test':
            return torch.tensor(image)
        else:
            mask = res['mask'].astype(np.float32)
            mask = mask.transpose(2, 0, 1).clip(0, 1)
            return torch.tensor(image), torch.tensor(mask)

