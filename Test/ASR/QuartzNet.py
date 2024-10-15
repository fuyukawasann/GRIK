import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio

class QuartzNet(nn.Module):
    def __init__(self):
        super(QuartzNet, slef).__init_()
        self.conv1 = nn.Conv1d(1, 256, kernel_size=3)
        self.conv2 = nn.Conv1d(256, 256, kernel_size=3)
        self.conv3 = nn.Conv1d(256, 256, kernel_size=3)
        self.conv4 = nn.Conv1d(256, 256, kernel_size=3)
        self.conv5 = nn.Conv1d(256, 256, kernel_size=3)
        self.fc = nn.Linear(256, 29)
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = torch.relu(self.conv5(x))
        x = x.view(-1, 256)
        x = self.fc(x)
        return x