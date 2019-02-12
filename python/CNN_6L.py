import torch.nn as nn
from network import Network


class CNN6L(nn.Module, Network):
    def __init__(self, n_f, n_n, activation):
        super(CNN6L, self).__init__()

        self.layer1 = nn.Sequential(
                *([nn.Conv2d(1, n_f, 3, padding=1),
                   activation]))

        self.layer2 = nn.Sequential(
                *([nn.Conv2d(n_f, n_f, 3, padding=1),
                   activation]))

        self.layer3 = nn.Sequential(
                *([nn.Conv2d(n_f, n_f, 3, padding=1),
                   activation]))

        self.layer4 = nn.Sequential(
                *([nn.Conv2d(n_f, n_f, 3, padding=1),
                   activation]))

        self.layer5 = nn.Sequential(
                *([nn.Linear(n_f*7*7, n_n),
                   activation]))

        self.layer6 = nn.Sequential(
                *([nn.Linear(n_n, 10)]))

        self.pool_layer = nn.MaxPool2d(2, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):

        layer1 = self.layer1(x)
        layer1_p = self.pool_layer(layer1)
        layer2 = self.layer2(layer1_p)
        layer2_p = self.pool_layer(layer2)
        layer3 = self.layer3(layer2_p)
        layer4 = self.layer4(layer3)
        N, C, H, W = layer4.size()

        layer5 = self.layer5(layer4.view(N, -1))
        layer6 = self.layer6(layer5)

        return [layer6, layer5, layer4, layer3, layer2, layer1]
