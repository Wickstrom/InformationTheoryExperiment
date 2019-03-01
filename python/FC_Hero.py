import torch.nn as nn
from network import Network


class FC_HERO(nn.Module, Network):
    def __init__(self, a_type):
        super(FC_HERO, self).__init__()
        Network.__init__(self)

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
                *([nn.Linear(784, 1024),
                   self.activation,
                   nn.BatchNorm1d(1024)]))

        self.layer2 = nn.Sequential(
                *([nn.Linear(1024, 20),
                   self.activation,
                   nn.BatchNorm1d(20)]))

        self.layer3 = nn.Sequential(
                *([nn.Linear(20, 20),
                   self.activation,
                   nn.BatchNorm1d(20)]))

        self.layer4 = nn.Sequential(
                *([nn.Linear(20, 20),
                   self.activation,
                   nn.BatchNorm1d(20)]))

        self.layer5 = nn.Sequential(
                *([nn.Linear(20, 10)]))

        for m in self.modules():
            self.weight_init(m)

    def forward(self, x):

        N, C, H, W = x.size()
        x = x.view(N, -1)

        layer1 = self.layer1(x)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        layer5 = self.layer5(layer4)

        return [layer5, layer4, layer3, layer2, layer1]
