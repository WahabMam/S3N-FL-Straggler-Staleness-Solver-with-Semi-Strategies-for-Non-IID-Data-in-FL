import torch.nn as nn
import torch.nn.functional as F

class FC2Layer(nn.Module):
    def __init__(self, input_size, output_size):
        super(FC2Layer, self).__init__()
        self.input_size = input_size
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, output_size)

    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x


class CNN2Layer(nn.Module):
    def __init__(self, in_channels, output_size, data_type, n_feature=6):
        super(CNN2Layer, self).__init__()
        self.n_feature = n_feature
        self.intemidiate_size = 4 if data_type == 'mnist' else 5
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=n_feature, kernel_size=5)
        self.conv2 = nn.Conv2d(n_feature, n_feature, kernel_size=5)
        self.fc1 = nn.Linear(n_feature * self.intemidiate_size * self.intemidiate_size, 50)  # 4*4 for MNIST 5*5 for CIFAR10
        self.fc2 = nn.Linear(50, output_size)

    def forward(self, x, verbose=False):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)
        x = x.view(-1, self.n_feature * self.intemidiate_size * self.intemidiate_size)  # 4*4 for MNIST 5*5 for CIFAR10
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

