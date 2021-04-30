import torch.nn as nn
from warmup_scheduler import GradualWarmupScheduler

from config import loss_weights

bce = nn.BCEWithLogitsLoss()
ce = nn.CrossEntropyLoss()


def criterion(logits, targets, lw=loss_weights):
    loss1 = ce(logits[:, :4], targets[:, :4].argmax(1)) * lw[0]
    loss2 = bce(logits[:, 4:], targets[:, 4:]) * lw[1]
    return (loss1 + loss2) / sum(lw)


class GradualWarmupSchedulerV2(GradualWarmupScheduler):
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        super(GradualWarmupSchedulerV2, self).__init__(optimizer, multiplier, total_epoch, after_scheduler)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]
        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in
                    self.base_lrs]
