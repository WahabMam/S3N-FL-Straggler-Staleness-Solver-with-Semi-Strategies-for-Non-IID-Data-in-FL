import torch.nn as nn
import torch.nn.functional as F

class FC2Layer(nn.Module):
    def __init__(self, input_size, output_size, intemidiate_size_1=32, intemidiate_size_2=16):
        super(FC2Layer, self).__init__()
        self.input_size = input_size
        self.fc1 = nn.Linear(input_size, intemidiate_size_1)
        self.fc2 = nn.Linear(intemidiate_size_1, intemidiate_size_2)
        self.fc3 = nn.Linear(intemidiate_size_2, output_size)

    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x


class CNN2Layer(nn.Module):
    def __init__(self, in_channels, output_size, data, kernel_size=5, intemidiate_size_1=6, intemidiate_size_2=50):
        super(CNN2Layer, self).__init__()
        self.intemidiate_size_1 = intemidiate_size_1
        self.data = data
        self.data_size = 28 if data == 'mnist' else 32

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=intemidiate_size_1, kernel_size=kernel_size)
        self.data_size = (self.data_size - kernel_size + 1) / 2

        self.conv2 = nn.Conv2d(in_channels=intemidiate_size_1, out_channels=intemidiate_size_1, kernel_size=kernel_size)
        self.data_size = int((self.data_size - kernel_size + 1) / 2)

        self.fc1 = nn.Linear(intemidiate_size_1 * self.data_size * self.data_size, intemidiate_size_2)
        self.fc2 = nn.Linear(intemidiate_size_2, output_size)


    def forward(self, x, verbose=False):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)
        x = x.view(-1, self.intemidiate_size_1 * self.data_size * self.data_size)  # 4*4 for MNIST 5*5 for CIFAR10
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

'''VGG11/13/16/19 in Pytorch from github.com/kuangliu/pytorch-cifar'''

cfg_vgg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg_vgg[vgg_name])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)