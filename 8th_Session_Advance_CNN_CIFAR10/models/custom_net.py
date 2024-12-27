import torch
import torch.nn as nn

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, groups=in_channels, dilation=dilation
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class CustomNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        
        # C1 Block - First Convolution Block (RF: 1->3->5)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=0),  # 32x32 -> 30x30
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 24, kernel_size=3, padding=0),  # 30x30 -> 28x28
            nn.BatchNorm2d(24),
            nn.ReLU(),
        )
        
        # C2 Block - Second Convolution Block with Downsampling (RF: 5->9->13)
        self.conv2 = nn.Sequential(
            DepthwiseSeparableConv(24, 32, kernel_size=3, padding=2, dilation=2),  # 28x28 -> 28x28
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 40, kernel_size=3, padding=1, stride=2),  # 28x28 -> 14x14
            nn.BatchNorm2d(40),
            nn.ReLU(),
        )
        
        # C3 Block - Third Convolution Block with Dilation (RF: 13->29->33)
        self.conv3 = nn.Sequential(
            nn.Conv2d(40, 64, kernel_size=3, padding=4, dilation=4),  # 14x14 -> 14x14
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 80, kernel_size=3, padding=2, dilation=2),  # 14x14 -> 14x14
            nn.BatchNorm2d(80),
            nn.ReLU(),
        )
        
        # C40 Block - Final Block with Output (RF: 33->41->41)
        self.conv4 = nn.Sequential(
            nn.Conv2d(80, 96, kernel_size=3, padding=3, dilation=3),  # 14x14 -> 14x14
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.Conv2d(96, 64, kernel_size=1)  # 14x14 -> 14x14 (1x1 conv)
        )
        
        self.gap = nn.AdaptiveAvgPool2d(1)  # 14x14 -> 1x1
        
        # Larger FC layers with dropout
        self.fc = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)  # Flatten: 1x1x64 -> 64
        x = self.fc(x)  # 64 -> 128 -> 10
        return x 