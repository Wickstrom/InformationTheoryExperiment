import torch.nn as nn
from network import Network


class CNN4L(nn.Module, Network):
    def __init__(self, n_f, n_n, activation):
        super(CNN4L, self).__init__()

        self.layer1 = nn.Sequential(
                *([nn.Conv2d(1, n_f, 3, padding=1),
                   activation]))

        self.layer2 = nn.Sequential(
                *([nn.Conv2d(n_f, n_f, 3, padding=1),
                   activation]))

        self.layer3 = nn.Sequential(
                *([nn.Linear(n_f*7*7, n_n),
                   activation]))

        self.layer4 = nn.Sequential(
                *([nn.Linear(n_n, 10)]))

        self.pool_layer = nn.MaxPool2d(2, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):

        layer1 = self.layer1(x)
        layer1_p = self.pool_layer(layer1)
        layer2 = self.layer2(layer1_p)
        layer2_p = self.pool_layer(layer2)
        N, C, H, W = layer2_p.size()

        layer3 = self.layer3(layer2_p.view(N, -1))
        layer4 = self.layer4(layer3)

        return [layer4, layer3, layer2, layer1]
