import torch.nn as nn

class MyCnn(nn.Module):
    def __init__(self, out_sz, ch):
        super().__init__()
        self.channels = ch
        self.out_sz = out_sz
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.conv1_1 = nn.Conv2d(1, ch, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(ch, ch*2, kernel_size=3, stride=1, padding=1)
        self.lin1 = nn.Linear(ch*2 * 14 * 14, 32)
        self.lin2 = nn.Linear(32, out_sz)
    def forward(self, x):
        x = self.conv1_1(x)
        x = self.relu(x)
        x = self.conv1_2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = x.view(-1, self.channels*2 * 14 * 14)
        x = self.lin1(x)
        x = self.relu(x)
        x = self.lin2(x)
        return x
