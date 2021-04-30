from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from config import *
from strategy import criterion



def train_epoch(model, loader, optimizer):
    model.train()
    train_loss = []
    bar = tqdm(loader)
    for (data, targets) in bar:
        optimizer.zero_grad()
        data, targets = data.to(device), targets.to(device)

        with amp.autocast():
            logits = model(data)
            loss = criterion(logits, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loss_np = loss.item()
        train_loss.append(loss_np)
        smooth_loss = sum(train_loss[-50:]) / min(len(train_loss), 50)
        bar.set_description('loss: %.4f, smth: %.4f' % (loss_np, smooth_loss))

    return np.mean(train_loss)


def valid_epoch(model, loader, get_output=False):
    model.eval()
    val_loss = []
    LOGITS = []
    TARGETS = []
    with torch.no_grad():
        for (data, targets) in tqdm(loader):
            data, targets = data.to(device), targets.to(device)
            logits = model(data)
            loss = criterion(logits, targets)
            val_loss.append(loss.item())
            LOGITS.append(logits.cpu())
            TARGETS.append(targets.cpu())

    val_loss = np.mean(val_loss)
    LOGITS = torch.cat(LOGITS)
    LOGITS[:, :4] = LOGITS[:, :4].softmax(1)
    LOGITS[:, 4:] = LOGITS[:, 4:].sigmoid()
    TARGETS = torch.cat(TARGETS).numpy()

    if get_output:
        return LOGITS
    else:
        aucs = []
        for cid in range(num_classes):
            if cid == 3: continue
            try:
                aucs.append(roc_auc_score(TARGETS[:, cid], LOGITS[:, cid]))
            except:
                aucs.append(0.5)
        return val_loss, aucs
