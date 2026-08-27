# models/googlenet_gtsrb.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Inception(nn.Module):
    def __init__(self, in_ch, ch1x1, ch3x3_reduce, ch3x3,
                 ch5x5_reduce, ch5x5, pool_proj):
        super().__init__()
        self.branch1 = ConvBNReLU(in_ch, ch1x1, 1)
        self.branch2 = nn.Sequential(
            ConvBNReLU(in_ch, ch3x3_reduce, 1),
            ConvBNReLU(ch3x3_reduce, ch3x3, 3, padding=1),
        )
        self.branch3 = nn.Sequential(
            ConvBNReLU(in_ch, ch5x5_reduce, 1),
            ConvBNReLU(ch5x5_reduce, ch5x5, 5, padding=2),
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            ConvBNReLU(in_ch, pool_proj, 1),
        )

    def forward(self, x):
        return torch.cat(
            [self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)],
            dim=1,
        )


class Head(nn.Module):
    """GTSRB-adapted GoogLeNet stem."""

    def __init__(self):
        super().__init__()
        self.stem = ConvBNReLU(3, 64, 3, padding=1)

    def forward(self, x):
        return self.stem(x)


class Backbone(nn.Module):
    """GoogLeNet body adapted for 32x32 GTSRB images."""

    def __init__(self):
        super().__init__()
        self.conv2 = nn.Sequential(
            ConvBNReLU(64, 64, 3, padding=1),
            ConvBNReLU(64, 192, 3, padding=1),
        )
        self.pool1 = nn.MaxPool2d(2, stride=2)

        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.pool2 = nn.MaxPool2d(2, stride=2)

        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.pool3 = nn.MaxPool2d(2, stride=2)

        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, 512)

    def forward(self, x):
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.pool2(x)
        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.pool3(x)
        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return F.relu(self.fc(x), inplace=True)


class Tail(nn.Module):
    """43-class GTSRB classifier."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(512, 43)

    def forward(self, x):
        return self.fc(x)
