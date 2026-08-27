# models/simple_cnn_gtsrb.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, pool=False, dropout=0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class Head(nn.Module):
    """
    GTSRB Simple CNN head.
    Input:  (B, 3, 32, 32)
    Output: (B, 32, 32, 32)
    """
    def __init__(self):
        super().__init__()
        self.conv = ConvBlock(3, 32, stride=1, pool=False, dropout=0.0)

    def forward(self, x):
        return self.conv(x)


class Backbone(nn.Module):
    """
    Split backbone for Simple CNN.
    Input:  (B, 32, 32, 32)
    Output: (B, 256)
    """
    def __init__(self):
        super().__init__()
        self.stage1 = nn.Sequential(
            ConvBlock(32, 64, stride=1, pool=True, dropout=0.05),
            ConvBlock(64, 64, stride=1, pool=False, dropout=0.05),
        )
        self.stage2 = nn.Sequential(
            ConvBlock(64, 128, stride=1, pool=True, dropout=0.10),
            ConvBlock(128, 128, stride=1, pool=False, dropout=0.10),
        )
        self.stage3 = nn.Sequential(
            ConvBlock(128, 256, stride=1, pool=True, dropout=0.15),
            ConvBlock(256, 256, stride=1, pool=False, dropout=0.15),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, 256)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return F.relu(self.fc(x))


class Tail(nn.Module):
    """Classifier head for GTSRB."""
    def __init__(self):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(256, 43),
        )

    def forward(self, x):
        return self.classifier(x)
