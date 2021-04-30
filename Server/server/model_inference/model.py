import segmentation_models_pytorch as smp
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class SegModel(nn.Module):
    def __init__(self, backbone):
        super(SegModel, self).__init__()
        self.seg = smp.UnetPlusPlus(encoder_name=backbone, encoder_weights=None, classes=2, activation=None)

    def forward(self, x):
        global_features = self.seg.encoder(x)
        seg_features = self.seg.decoder(*global_features)
        seg_features = self.seg.segmentation_head(seg_features)
        return seg_features


class enetv2(nn.Module):
    def __init__(self, backbone):
        super(enetv2, self).__init__()
        self.enet = timm.create_model(backbone, False)
        self.enet.conv_stem.weight = nn.Parameter(self.enet.conv_stem.weight.repeat(1, 5 // 3 + 1, 1, 1)[:, :5])
        self.myfc = nn.Linear(self.enet.classifier.in_features, 12)
        self.enet.classifier = nn.Identity()

    def extract(self, x):
        return self.enet(x)

    def forward(self, x, mask):
        mask = F.interpolate(mask, x.shape[2])
        x = torch.cat([x, mask], 1)
        x = self.extract(x)
        x = self.myfc(x)
        return x


