import torch
import torch.nn as nn

class block(nn.Module):
    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):
        super(block, self).__init__()
        self.expansion = 4   
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.conv3 = nn.Conv1d(out_channels, self.expansion*out_channels, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm1d(out_channels*self.expansion)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample

    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x = x + identity       # skip connection
        x = self.relu(x)
        return x
    

class Resnet(nn.Module):

    def __init__(self, block, layers, image_channels, num_classes):
        super(Resnet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv1d(image_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.maxpool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, layers[0], out_channels=64, stride=1)
        self.layer2 = self._make_layer(block, layers[1], out_channels=128, stride=2)
        self.layer3 = self._make_layer(block, layers[2], out_channels=256, stride=2)
        self.layer4 = self._make_layer(block, layers[3], out_channels=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool1d((1))
        # self.sigmoid = nn.LogSoftmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        # self.dropout = nn.Dropout1d(p=0.2)

        self.fc1 = nn.Linear(512*4, num_classes)



    def forward(self, x, return_features=False):
        x = torch.permute(x, (0, 2, 1))
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.dropout(x)
    
        x = self.maxpool1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = x.reshape(x.shape[0], -1)
        if return_features:
            return x
        x = self.fc1(x)
        x = self.sigmoid(x)
        return x
        
    def _make_layer(self, block, num_residual_blocks, out_channels, stride):
        identity_downsample = None
        layers = []
        if stride != 1 or self.in_channels != out_channels*4:
            identity_downsample = nn.Sequential(nn.Conv1d(self.in_channels, out_channels*4, kernel_size=1, stride=stride),
                                               nn.BatchNorm1d(out_channels*4))
        layers.append(block(self.in_channels, out_channels, identity_downsample, stride))
        self.in_channels = out_channels*4
        for i in range(num_residual_blocks - 1):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)
    

def resnet50(channels, num_classes, n_layers=50):
    assert n_layers in [50, 101, 152], "n_layers has to be 50 or 101 or 152"
    n_blocks = [3, 4, 6, 3]
    if n_layers == 101:
        n_blocks = [3, 4, 23, 3]
    elif n_layers == 152:
        n_blocks = [3, 8, 36, 3]
    
    return Resnet(block, n_blocks, channels, num_classes)
