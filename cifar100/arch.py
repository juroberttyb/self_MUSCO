import torch
import torch.nn as nn
import torch.nn.functional as F

def conv3x3(_in, _out):
    return nn.Conv2d(in_channels=_in, out_channels=_out, kernel_size=3, padding='same')

class Net(nn.Module):
    def __init__(self):
        super().__init__()

        n1, n2, n3 = 12, 12, 12

        self.conv1_1 = conv3x3(3,n1)
        self.conv1_2 = conv3x3(n1,n2)
        self.bn1_1 = nn.BatchNorm2d(n1)
        self.bn1_2 = nn.BatchNorm2d(n2)
        self.conv2_1 = conv3x3(n2,n2)
        self.conv2_2 = conv3x3(n2,n3)
        self.bn2_1 = nn.BatchNorm2d(n2)
        self.bn2_2 = nn.BatchNorm2d(n3)
        self.conv3_1 = conv3x3(n3,n3)
        self.conv3_2 = conv3x3(n3,n3)
        self.bn3_1 = nn.BatchNorm2d(n3)
        self.bn3_2 = nn.BatchNorm2d(n3)

        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(16 * n3, 10)

    def forward(self, x):
        x = F.relu(self.conv1_1(x))
        x = self.bn1_1(x)
        x = F.relu(self.conv1_2(x))
        x = self.bn1_2(x)
        x = self.pool(x)

        x = F.relu(self.conv2_1(x))
        x = self.bn2_1(x)
        x = F.relu(self.conv2_2(x))
        x = self.bn2_2(x)
        x = self.pool(x)

        x = F.relu(self.conv3_1(x))
        x = self.bn3_1(x)
        x = F.relu(self.conv3_2(x))
        x = self.bn3_2(x)
        x = self.pool(x)

        x = torch.flatten(x, 1) # flatten all dimensions except batch

        x = self.fc(x)
        return x