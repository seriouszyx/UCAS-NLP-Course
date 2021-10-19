import torch
from torch import nn
import torch.nn.functional as F


class MyCNNNet(nn.Module):
    def __init__(self):
        super(MyCNNNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, (5, 5))
        self.maxpool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, (5, 5))
        self.fc1 = nn.Linear(16 * 53 * 53, 1024)
        nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, 512)
        nn.Dropout(0.5)
        self.fc3 = nn.Linear(512, 2)

    def forward(self, x):
        x = self.maxpool(F.relu(self.conv1(x)))
        x = self.maxpool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 53 * 53)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x
