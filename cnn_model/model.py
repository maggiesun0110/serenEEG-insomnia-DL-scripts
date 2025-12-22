import torch
import torch.nn as nn

class BaseEEGCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        #block1
        self.conv1 = nn.Conv1d(3, 32, kernel_size=64, stride=16, padding = 0)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(4)

        self.conv2 = nn.Conv1d(32, 64, kernel_size = 8, stride = 2, padding =3)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2)

        self.conv3 = nn.Conv1d(64, 128, kernel_size = 8, stride = 2, padding =3)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(2)

        self.global_avg = nn.AdaptiveAvgPool1d(1)

        self.dropout = nn.Dropout(0.5)
        
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 1)

        self.relu = nn.ReLU()


    def forward(self, x):
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu(self.bn3(self.conv3(x))))

        x = self.global_avg(x)
        x = x.squeeze(-1)

        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x