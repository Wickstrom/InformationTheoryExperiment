import torch.nn as nn
from network import Network


class CNN_Hero(nn.Module, Network):
    def __init__(self, activation, a_type):
        super(CNN_Hero, self).__init__()

        self.layer1 = nn.Sequential(
                *([nn.Conv2d(1, 4, 3, padding=1),
                   activation]))

        self.layer2 = nn.Sequential(
                *([nn.Conv2d(4, 8, 3, padding=1),
                   activation]))

        self.layer3 = nn.Sequential(
                *([nn.Conv2d(8, 16, 3, padding=1),
                   activation]))

        self.layer4 = nn.Sequential(
                *([nn.Linear(7*7*16, 256),
                   activation]))

        self.layer5 = nn.Sequential(
                *([nn.Linear(256, 10)]))

        self.a_type = a_type
        for m in self.modules():
            self.weight_init(m)

        self.pool_layer = nn.MaxPool2d(2, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):

        layer1 = self.layer1(x)
        layer2 = self.layer2(layer1)
        layer2_p = self.pool_layer(layer2)
        layer3 = self.layer3(layer2_p)
        layer3_p = self.pool_layer(layer3)

        N, C, H, W = layer3_p.size()

        layer4 = self.layer4(layer3_p.view(N, -1))
        layer5 = self.layer5(layer4)

        return [layer5, layer4, layer3, layer2, layer1]
