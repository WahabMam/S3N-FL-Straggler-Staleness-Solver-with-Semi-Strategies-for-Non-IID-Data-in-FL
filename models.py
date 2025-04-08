import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import torch

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
    
# for Mnist
# class LeNet5(nn.Module):
#     def __init__(self, input_channels=1, num_classes=10):
#         super(LeNet5, self).__init__()
        
#         # Convolutional layers
#         self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2)
#         self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        
#         # Fully connected layers
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, num_classes)
        
#     def forward(self, x):
#         x = F.tanh(self.conv1(x))
#         x = F.avg_pool2d(x, kernel_size=2, stride=2)
#         x = F.tanh(self.conv2(x))
#         x = F.avg_pool2d(x, kernel_size=2, stride=2)
        
#         x = x.view(x.size(0), -1)  # Flatten
#         x = F.tanh(self.fc1(x))
#         x = F.tanh(self.fc2(x))
#         x = self.fc3(x)  # Output layer (no activation, as softmax is applied in loss)
        
#         return x
    

# for Cifar_10

class LeNet5(nn.Module):
    def __init__(self, input_channels=3, num_classes=10):
        super(LeNet5, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=6, kernel_size=5, stride=1, padding=2)  # Output: (6, 32, 32)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)  # Output: (16, 14, 14)

        # Compute correct FC input size dynamically
        self._to_linear = None
        self._compute_fc_input()

        # Fully connected layers
        self.fc1 = nn.Linear(self._to_linear, 120)  
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def _compute_fc_input(self):
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 32, 32)  # CIFAR-10 size
            x = self.conv1(dummy_input)
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
            x = self.conv2(x)
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
            self._to_linear = x.numel()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.avg_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv2(x))
        x = F.avg_pool2d(x, kernel_size=2, stride=2)
        
        x = x.view(x.size(0), -1)  # Flatten properly
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class SimpleAlexNet(nn.Module):
    def __init__(self, input_channels=3, num_classes=10):
        super(SimpleAlexNet, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1),  # 32x32 → 32x32
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32x32 → 16x16

            nn.Conv2d(64, 192, kernel_size=3, padding=1),  # 16x16
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16x16 → 8x8

            nn.Conv2d(192, 384, kernel_size=3, padding=1),  # 8x8
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1),  # 8x8
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # 8x8
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 8x8 → 4x4
        )

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x
    

class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.skip = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.skip(x)
        return F.relu(out)

class ResNet18(nn.Module):
    def __init__(self, input_channels=3, num_classes=10):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, in_ch, out_ch, blocks, stride):
        layers = [BasicBlock(in_ch, out_ch, stride)]
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_ch, out_ch))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)