import torch.nn as nn
from network import Network


class FC_Hero(nn.Module, Network):
    def __init__(self, activation, a_type, mode, dirichlet):
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
        self.dirichlet = dirichlet
        self.mode = mode

        for m in self.modules():
            self.weight_init(m)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):

        N, C, H, W = x.size()
        x = x.view(N, -1)

        if self.mode == 'after':

            layer1 = self.layer1(x)
            layer2 = self.layer2(layer1)
            layer3 = self.layer3(layer2)
            layer4 = self.layer4(layer3)
            layer5 = self.layer5(layer4)

            return [layer5, layer4, layer3, layer2, layer1]

        elif self.mode == 'before':
            layer1_pre = self.layer1[0](x)
            layer1 = self.layer1[1](layer1_pre)
            layer2_pre = self.layer2[0](layer1)
            layer2 = self.layer2[1](layer2_pre)
            layer3_pre = self.layer3[0](layer2)
            layer3 = self.layer3[1](layer3_pre)
            layer4_pre = self.layer4[0](layer3)
            layer4 = self.layer4[1](layer4_pre)
            layer5 = self.layer5(layer4)

            return [layer5, layer4_pre, layer3_pre, layer2_pre, layer1_pre]

        else:
            print('Not implemented')
            raise
