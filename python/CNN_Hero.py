import torch as th
import torch.nn as nn
from network import Network


class CNN_Hero(nn.Module, Network):
    def __init__(self, a_type, n_iterations):
        super(CNN_Hero, self).__init__()

        self.a_type = a_type

        if a_type == 'relu':
            self.activation = nn.ReLU()
        elif a_type == 'tanh':
            self.activation = nn.Tanh()
        elif a_type == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif a_type == 'leaky_relu':
            self.activation = nn.LeakyReLU()
        else:
            print('Not implemented')
            raise

        self.layer1 = nn.Sequential(
                *([nn.Conv2d(1, 4, 3, padding=1),
                   self.activation]))

        self.layer2 = nn.Sequential(
                *([nn.Conv2d(4, 8, 3, padding=1),
                   self.activation]))

        self.layer3 = nn.Sequential(
                *([nn.Conv2d(8, 16, 3, padding=1),
                   self.activation]))

        self.layer4 = nn.Sequential(
                *([nn.Linear(7*7*16, 256),
                   self.activation]))

        self.layer5 = nn.Sequential(
                *([nn.Linear(256, 10)]))

        for m in self.modules():
            self.weight_init(m)

        self.pool_layer = nn.MaxPool2d(2, 2)
        self.softmax = nn.Softmax(dim=1)

        self.sigmas = th.zeros((7, n_iterations)).cuda()
        self.cost = []
        self.score = []
        self.MI = th.zeros((n_iterations, 5, 2)).cuda()

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
