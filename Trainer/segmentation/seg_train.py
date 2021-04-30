import time

import numpy as np
import pandas as pd
import torch.optim as optim
from torch.utils.data import TensorDataset

from augmentations import transforms_train, transforms_val
from config import *
from dataset import DC_Dataset, seed_worker
from model import SegModel
from strategy import GradualWarmupSchedulerV2
from train_valid import train_epoch, valid_epoch


def run(fold):
    content = 'Fold: ' + str(fold)
    print(content)
    with open(log_file, 'a') as appender:
        appender.write(content + '\n')
    train_ = df_train.query(f'w_anno==True and fold!={fold}').copy()
    valid_ = df_train.query(f'w_anno==True and fold=={fold}').copy()

    dataset_train = DC_Dataset(train_, 'train', transform=transforms_train)
    dataset_valid = DC_Dataset(valid_, 'valid', transform=transforms_val)
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True,
                                               num_workers=num_workers, worker_init_fn=seed_worker)
    valid_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=batch_size, shuffle=False,
                                               num_workers=num_workers, worker_init_fn=seed_worker)

    model = SegModel(enet_type)

    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.seg.segmentation_head.parameters():
        parameter.requires_grad = True
    for parameter in model.seg.decoder.parameters():
        parameter.requires_grad = True

    if is_resume_train:
        model_file = os.path.join(model_dir, f'{kernel_type}_best_fold{fold}_epo{resume_epoch - 1}.pth')
        model.load_state_dict(torch.load(model_file))

    model = model.to(device)
    val_loss_min = np.Inf

    optimizer = optim.Adam(model.parameters(), lr=init_lr)
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cosine_epo)
    scheduler_warmup = GradualWarmupSchedulerV2(optimizer, multiplier=10, total_epoch=warmup_epo,
                                                after_scheduler=scheduler_cosine)
    for epoch in range(1 if not is_resume_train else resume_epoch, n_epochs + 1):
        print('Epoch:', epoch)
        scheduler_warmup.step(epoch - 1)

        train_loss = train_epoch(model, train_loader, optimizer)
        val_loss = valid_epoch(model, valid_loader)

        content = f'Fold {fold} Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, train loss: {train_loss:.5f}, valid loss: {(val_loss):.5f}.'
        print(content)
        with open(log_file, 'a') as appender:
            appender.write(content + '\n')

        if val_loss_min > val_loss:
            model_file = os.path.join(model_dir, f'{kernel_type}_best_fold{fold}_epo{epoch}.pth')
            torch.save(model.state_dict(), model_file)
            val_loss_min = val_loss

    torch.save(model.state_dict(), os.path.join(model_dir, f'{kernel_type}_best_fold{fold}.pth'))
