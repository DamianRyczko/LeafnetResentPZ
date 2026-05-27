import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from torchvision.transforms import v2

class residual_block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, channel_expansion = 1):
        super().__init__()
        self.expanded_channels = out_channels * channel_expansion
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.conv4 = nn.Conv2d(out_channels, self.expanded_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn4 = nn.BatchNorm2d(self.expanded_channels)
        self.leaky_relu = nn.LeakyReLU(0.1, inplace=True)


        self.downsample = None
        if in_channels != self.expanded_channels or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, self.expanded_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(self.expanded_channels)
            )

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.leaky_relu(x)
        x = self.conv4(x)
        x = self.bn4(x)

        if self.downsample is not None:
            residual = self.downsample(residual)

        x = x + residual
        x = self.leaky_relu(x)
        return x


class residual_cnn(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.leaky_relu = nn.LeakyReLU(0.1, inplace=True)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.residual_block1 = residual_block(64, 64,channel_expansion=4)
        self.residual_block2 = residual_block(256, 64,channel_expansion=4)
        self.residual_block3 = residual_block(256, 64, channel_expansion=4)
        self.residual_block4 = residual_block(256, 128, channel_expansion=4,stride=2)
        self.residual_block5 = residual_block(512, 128,channel_expansion=4)
        self.residual_block6 = residual_block(512, 128,channel_expansion = 4)
        self.residual_block7 = residual_block(512, 128,channel_expansion=4)
        self.residual_block8 = residual_block(512, 256, channel_expansion=2)
        self.residual_block9 = residual_block(512, 256,channel_expansion=2,stride=2)
        self.residual_block10 = residual_block(512, 256, channel_expansion=2)
        self.residual_block11 = residual_block(512, 256,channel_expansion=2)
        self.residual_block12 = residual_block(512, 256,channel_expansion=2)
        self.residual_block13 = residual_block(512, 256, channel_expansion=2)
        self.residual_block14 = residual_block(512, 256, channel_expansion=2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.3)


        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.leaky_relu(x)
        x = self.max_pool(x)

        x = self.residual_block1(x)
        x = self.residual_block2(x)
        x = self.residual_block3(x)
        x = self.residual_block4(x)
        x = self.residual_block5(x)
        x = self.residual_block6(x)
        x = self.residual_block7(x)
        x = self.residual_block8(x)
        x = self.residual_block9(x)
        x = self.residual_block10(x)
        x = self.residual_block11(x)
        x = self.residual_block12(x)
        x = self.residual_block13(x)
        x = self.residual_block14(x)
        x = self.avg_pool(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.leaky_relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.leaky_relu(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

transforms_val = v2.Compose([
    v2.Resize((336, 336)),
    v2.RGB(), # convert timage to rgb values
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True), # convert to tensor
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # default valeus from pytorch dosumentation
])