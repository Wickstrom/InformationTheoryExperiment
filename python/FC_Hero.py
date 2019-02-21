import torch.nn as nn
from network import Network


class FC_Hero(nn.Module, Network):
    def __init__(self, activation, a_type):
        super(FC_Hero, self).__init__()

        self.layer1 = nn.Sequential(
                *([nn.Linear(784, 1024),
                   activation]))

        self.layer2 = nn.Sequential(
                *([nn.Linear(1024, 20),
                   activation]))

        self.layer3 = nn.Sequential(
                *([nn.Linear(20, 20),
                   activation]))

        self.layer4 = nn.Sequential(
                *([nn.Linear(20, 20),
                   activation]))

        self.layer5 = nn.Sequential(
                *([nn.Linear(20, 10)]))

        self.a_type = a_type

        for m in self.modules():
            self.weight_init(m)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):

        N, C, H, W = x.size()
        x = x.view(N, -1)

        layer1 = self.layer1(x)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        layer5 = self.layer5(layer4)

        return [layer5, layer4, layer3, layer2, layer1]
