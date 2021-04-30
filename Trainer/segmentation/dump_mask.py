import cv2
import numpy as np
from torch.utils.data import TensorDataset
from tqdm import tqdm

from augmentations import transforms_val
from config import *
from dataset import DC_Dataset
from model import SegModel
from seg_train import df_train
from train_valid import valid_epoch


def dump_mask():
    for fold in range(5):
        valid_ = df_train.query(f'w_anno==True and fold=={fold}').copy()
        dataset_valid = DC_Dataset(valid_, 'valid', transform=transforms_val)
        valid_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=batch_size, shuffle=False,
                                                   num_workers=num_workers)

        model = SegModel(enet_type)
        model = model.to(device)
        model_file = os.path.join(model_dir, f'{kernel_type}_best_fold{fold}.pth')
        model.load_state_dict(torch.load(model_file), strict=False)
        model.eval()

        outputs = valid_epoch(model, valid_loader, get_output=True).numpy()

        for i, (_, row) in tqdm(enumerate(valid_.iterrows())):
            png = (outputs[i] * 255).astype(np.uint8).transpose(1, 2, 0)
            png = np.concatenate([png, np.zeros((png.shape[0], png.shape[1], 1))], -1)
            cv2.imwrite(os.path.join(output_dir, f'{row.StudyInstanceUID}.png'), png)

    df_train_wo_anno = df_train.query(f'w_anno==False').copy().reset_index(drop=True)
    dataset_test = DC_Dataset(df_train_wo_anno, 'test', transform=transforms_val)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False,
                                              num_workers=num_workers)
    models = []
    for fold in range(5):
        model = SegModel(enet_type)
        model = model.to(device)
        model_file = os.path.join(model_dir, f'{kernel_type}_best_fold{fold}.pth')
        model.load_state_dict(torch.load(model_file), strict=False)
        model.eval()
        models.append(model)

    with torch.no_grad():
        for id, data in tqdm(enumerate(test_loader), total=len(test_loader)):
            data = data.to(device)
            outputs = torch.stack([model(data).sigmoid() for model in models], 0).mean(0).cpu().numpy()
            for i in range(outputs.shape[0]):
                row = df_train_wo_anno.loc[id * batch_size + i]
                png = (outputs[i] * 255).astype(np.uint8).transpose(1, 2, 0)
                png = np.concatenate([png, np.zeros((png.shape[0], png.shape[1], 1))], -1)
                cv2.imwrite(os.path.join(output_dir, f'{row.StudyInstanceUID}.png'), png)

