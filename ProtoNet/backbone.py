import torch.nn as nn
from torchvision import models

"""
默认的简单backbone，一个卷积块，用于测试
"""


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class ConvEncoder(nn.Module):
    def __init__(self, input_img_channels, hidden_channels):
        super(ConvEncoder, self).__init__()
        self.hidden_channels = hidden_channels
        self.input_img_channels = input_img_channels

        self.encoder = nn.Sequential(
            self.conv_block(self.input_img_channels, self.hidden_channels),
            self.conv_block(self.hidden_channels, self.hidden_channels),
            self.conv_block(self.hidden_channels, self.hidden_channels),
            self.conv_block(self.hidden_channels, self.hidden_channels),
            Flatten()
        )

    def forward(self, x):
        x = self.encoder(x)
        return x

    @staticmethod
    def conv_block(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=7, padding=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )


"""
Teacher-Student模型的backbone，ResNet50
用于无微调场景下测试
"""


class TS_ResNet50(nn.Module):

    def __init__(self, num_classes=31):
        super(TS_ResNet50, self).__init__()
        self.backbone = ResNet50()
        self.head = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        inner = self.head(x)
        y = self.classifier(inner)
        return inner, y


"""
ResNet50，默认模型
"""


class ResNet50(nn.Module):

    def __init__(self):
        super(ResNet50, self).__init__()
        self.encoder = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.encoder.fc = nn.Sequential()

    def forward(self, x):
        x = self.encoder(x)
        return x
