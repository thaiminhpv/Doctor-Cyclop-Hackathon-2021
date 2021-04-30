import cv2
import numpy as np
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
        mask = cv2.imread(os.path.join(mask_folder, row.StudyInstanceUID + '.png')).astype(np.float32)[:, :, :2]

        res = self.transform(image=image, mask=mask)
        image = res['image'].astype(np.float32).transpose(2, 0, 1) / 255.
        mask = res['mask'].astype(np.float32).transpose(2, 0, 1) / 255.

        image = np.concatenate([image, mask], 0)

        if self.mode == 'test':
            return torch.tensor(image)
        else:
            label = row[[
                'ETT - Abnormal',
                'ETT - Borderline',
                'ETT - Normal',
                'no_ETT',
                'NGT - Abnormal',
                'NGT - Borderline',
                'NGT - Incompletely Imaged',
                'NGT - Normal',
                'CVC - Abnormal',
                'CVC - Borderline',
                'CVC - Normal',
                'Swan Ganz Catheter Present'
            ]].values.astype(float)
            return torch.tensor(image).float(), torch.tensor(label).float()
