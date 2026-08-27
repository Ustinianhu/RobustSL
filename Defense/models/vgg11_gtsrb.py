# models/vgg11_gtsrb.py
import torch
import torch.nn as nn


class VGGConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Head(nn.Module):
    """First VGG11 convolution block for split learning."""

    def __init__(self):
        super().__init__()
        self.block = VGGConv(3, 64)

    def forward(self, x):
        return self.block(x)


class Backbone(nn.Module):
    """VGG11 feature extractor adapted to 32x32 GTSRB images."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.MaxPool2d(2, 2),
            VGGConv(64, 128),
            nn.MaxPool2d(2, 2),
            VGGConv(128, 256),
            VGGConv(256, 256),
            nn.MaxPool2d(2, 2),
            VGGConv(256, 512),
            VGGConv(512, 512),
            nn.MaxPool2d(2, 2),
            VGGConv(512, 512),
            VGGConv(512, 512),
            nn.MaxPool2d(2, 2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 512)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


class Tail(nn.Module):
    """43-class GTSRB classifier."""

    def __init__(self):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Dropout(0.3),
            nn.Linear(512, 43),
        )

    def forward(self, x):
        return self.classifier(x)
