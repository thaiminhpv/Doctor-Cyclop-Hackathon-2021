import time

import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader

from augmentations import transforms_train, transforms_val
from config import *
from dataset import DC_Dataset, seed_worker
from model import B1_ns
from strategy import GradualWarmupSchedulerV2
from train_valid import train_epoch, valid_epoch


def run(fold):
    content = 'Fold: ' + str(fold)
    print(content)
    with open(log_file, 'a') as appender:appender.write(content + '\n')
    train_ = df_train.query(f'fold!={fold}').copy()
    valid_ = df_train.query(f'fold=={fold}').copy()

    dataset_train = DC_Dataset(train_, 'train', transform=transforms_train)
    dataset_valid = DC_Dataset(valid_, 'valid', transform=transforms_val)
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, worker_init_fn=seed_worker)
    valid_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=batch_size, shuffle=False, num_workers=num_workers, worker_init_fn=seed_worker)

    model = B1_ns(enet_type, num_classes)

    if is_resume_train:
        model_file = os.path.join(model_dir, f'{kernel_type}_best_fold{fold}_epo{resume_epoch - 1}.pth')
        model.load_state_dict(torch.load(model_file))

    model = model.to(device)
    aucs_max = 0

    optimizer = optim.Adam(model.parameters(), lr=init_lr)
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cosine_epo)
    scheduler_warmup = GradualWarmupSchedulerV2(optimizer, multiplier=10, total_epoch=warmup_epo,
                                                after_scheduler=scheduler_cosine)
    for epoch in range(1 if not is_resume_train else resume_epoch, n_epochs + 1):
        print('Epoch:', epoch)
        scheduler_warmup.step(epoch - 1)

        train_loss = train_epoch(model, train_loader, optimizer)
        val_loss, aucs = valid_epoch(model, valid_loader)

        content = f'Fold {fold} Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, train loss: {train_loss:.4f}, valid loss: {(val_loss):.4f}, aucs: {np.mean(aucs):.4f}.'
        content += '\n' + ' '.join([f'{x:.4f}' for x in aucs])
        print(content)
        with open(log_file, 'a') as appender:
            appender.write(content + '\n')

        if aucs_max < np.mean(aucs):
            model_file = os.path.join(model_dir, f'{kernel_type}_best_fold{fold}_epo{epoch}.pth')
            torch.save(model.state_dict(), model_file)
            aucs_max = np.mean(aucs)

    torch.save(model.state_dict(), os.path.join(model_dir, f'{kernel_type}_best_fold{fold}.pth'))
