import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.downsample = nn. Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        x = self.downsample(x)
        out += x
        out = F.relu(out)

        return out


class ResNet18(nn.Module):
    def __init__(self, N):
        super(ResNet18, self).__init__()

        self.in_planes = 64
        self.N = N

        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.conv2 = nn.Conv2d(self.in_planes, self.in_planes, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.in_planes)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)

        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 + N, 3)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x, parameters):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x))) # (N, 64, 112, 112)
        x = self.maxpool(x) # (N, 64, 56, 56)

        x = self.layer1(x)  # (N, 64, 56, 56)
        x = self.layer2(x)  # (N, 128, 28, 28)
        x = self.layer3(x)  # (N, 256, 14, 14)
        x = self.layer4(x)  # (N, 512, 7, 7)

        x = self.avgpool(x) # (N, 512, 1, 1)
        x = x.view(x.size(0), -1)   # (N, 512)
        x = torch.cat((x, parameters), 1)
        x = self.fc(x)

        return x


class ResNet18_2D(nn.Module):
    def __init__(self, N):
        super(ResNet18, self).__init__()

        self.in_planes = 64
        self.N = N

        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.conv2 = nn.Conv2d(self.in_planes, self.in_planes, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.in_planes)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)

        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 + N, 2)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x, parameters):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x))) # (N, 64, 112, 112)
        x = self.maxpool(x) # (N, 64, 56, 56)

        x = self.layer1(x)  # (N, 64, 56, 56)
        x = self.layer2(x)  # (N, 128, 28, 28)
        x = self.layer3(x)  # (N, 256, 14, 14)
        x = self.layer4(x)  # (N, 512, 7, 7)

        x = self.avgpool(x) # (N, 512, 1, 1)
        x = x.view(x.size(0), -1)   # (N, 512)
        x = torch.cat((x, parameters), 1)
        x = self.fc(x)

        return x
