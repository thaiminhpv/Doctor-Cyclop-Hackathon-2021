import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from threading import Thread

from server.model_inference.model import SegModel, enetv2
from server.model_inference.config import labels, model_dir
from server.model_inference.dataset import RANZCRDatasetTest
from server.util.push_mask import push_mask_list

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

num_workers = 2
image_size = 512
batch_size = 8

enet_type_seg = 'timm-efficientnet-b1'
kernel_type_seg = 'unet++_6epo'

enet_type_cls = 'tf_efficientnet_b1_ns'
kernel_type_cls = 'Cls_6epo'

models_seg = []
for fold in range(5):
    model = SegModel(enet_type_seg)
    model = model.to(device)
    model_file = os.path.join(model_dir, f'{kernel_type_seg}_best_fold{fold}.pth')
    model.load_state_dict(torch.load(model_file), strict=True)
    model.eval()
    models_seg.append(model)

models_cls = []
for fold in range(5):
    model = enetv2(enet_type_cls)
    model = model.to(device)
    model_file = os.path.join(model_dir, f'{kernel_type_cls}_best_fold{fold}.pth')
    model.load_state_dict(torch.load(model_file), strict=True)
    model.eval()
    models_cls.append(model)


def get_model_prediction(df_sub, images):
    # Load Data
    dataset_test = RANZCRDatasetTest(df_sub, images)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False,
                                              num_workers=num_workers)

    # Making prediction
    PROBS = []
    with torch.no_grad():
        for batch_id, data in tqdm(enumerate(test_loader), total=len(test_loader)):
            # for k in data.keys():
            #     data[k] = data[k].cuda
            data['1024'], data['512'] = data['1024'].cuda( ) , data['512'].cuda( )
            mask = torch.stack([model(data['1024']).sigmoid() for model in models_seg], 0).mean(0)

            # Submit mask image
            Thread(target=push_mask_list, args=(data['caseId'], data['1024'], mask)).  start()

            logits = torch.stack([model(data['512'], mask) for model in models_cls], 0)
            logits[:, :, :4] = logits[:, :, :4].softmax(2)
            logits[:, :, 4:] = logits[:, :, 4:].sigmoid()

            PROBS.append(logits.cpu())
    PROBS = torch.cat(PROBS, 1)
    PROBS = PROBS[:, :, [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11]]
    PROBS = PROBS.numpy()

    # Submit
    df_sub[labels] = PROBS.mean(0)
    return df_sub
