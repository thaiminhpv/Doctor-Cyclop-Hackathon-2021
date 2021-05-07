import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class UserUploadDataset(Dataset):
    def __init__(self, df, images):
        self.df = df.reset_index(drop=True)
        self.images = images

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        row = self.df.iloc[index]

        image = np.repeat(self.images[row.StudyInstanceUID][:, :, np.newaxis]
                          , 3, axis=2)  # expand grayscale to 3 channels

        image1024 = cv2.resize(image, (1024, 1024)).astype(np.float32).transpose(2, 0, 1) / 255.
        image512 = cv2.resize(image, (512, 512)).astype(np.float32).transpose(2, 0, 1) / 255.

        return {
            '1024': torch.tensor(image1024),
            '512': torch.tensor(image512),
            'caseId': row.StudyInstanceUID
        }
