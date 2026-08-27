# models/micronnet_gtsrb.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class Head(nn.Module):
    """MicronNet input projection, adapted for RGB GTSRB images."""

    def __init__(self):
        super().__init__()
        self.projection = nn.Conv2d(3, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        return F.relu(self.projection(x), inplace=True)


class Backbone(nn.Module):
    """Compact MicronNet feature extractor for 32x32 traffic-sign images."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 29, kernel_size=5, stride=2, padding=0)
        self.conv2 = nn.Conv2d(29, 59, kernel_size=3, stride=2, padding=0)
        self.conv3 = nn.Conv2d(59, 74, kernel_size=3, stride=1, padding=0)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Linear(74 * 4 * 4, 300)

    def forward(self, x):
        x = F.relu(self.conv1(x), inplace=True)
        x = F.relu(self.conv2(x), inplace=True)
        x = F.relu(self.conv3(x), inplace=True)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return F.relu(self.fc(x), inplace=True)


class Tail(nn.Module):
    """43-class GTSRB classifier."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(300, 43)

    def forward(self, x):
        return self.fc(x)
