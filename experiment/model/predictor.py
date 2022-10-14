import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(2,2), stride=2)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class StatePredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.eb1 = EncoderBlock(16, 32)
        self.eb2 = EncoderBlock(32, 64)
        self.eb3 = EncoderBlock(64, 128)
        self.fc1 = nn.Linear(384, 256)
        self.fc2 = nn.Linear(256, 64)

    def forward(self, x, z):
        x = self.eb1(x)
        x = self.eb2(x)
        x = self.eb3(x)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = torch.cat([x, z], 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
