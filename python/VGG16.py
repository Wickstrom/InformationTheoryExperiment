import torch.nn as nn
from network import Network


class VGG16(nn.Module, Network):
    def __init__(self, activation):
        super(VGG16, self).__init__()

        # First encoder
        self.layer1 = nn.Sequential(
                *([nn.Conv2d(3, 64, kernel_size=3, padding=1),
                   nn.BatchNorm2d(64),
                   activation]))
        self.layer2 = nn.Sequential(
                *([nn.Conv2d(64, 64, kernel_size=3, padding=1),
                   nn.BatchNorm2d(64),
                   activation]))
        # Second encoder
        self.layer3 = nn.Sequential(
                *([nn.Conv2d(64, 128, kernel_size=3, padding=1),
                   nn.BatchNorm2d(128),
                   activation]))
        self.layer4 = nn.Sequential(
                *([nn.Conv2d(128, 128, kernel_size=3, padding=1),
                   nn.BatchNorm2d(128),
                   activation]))

        # Third encoder
        self.layer5 = nn.Sequential(
                *([nn.Conv2d(128, 256, kernel_size=3, padding=1),
                   nn.BatchNorm2d(256),
                   activation]))
        self.layer6 = nn.Sequential(
                *([nn.Conv2d(256, 256, kernel_size=3, padding=1),
                   nn.BatchNorm2d(256),
                   activation]))
        self.layer7 = nn.Sequential(
                *([nn.Conv2d(256, 256, kernel_size=3, padding=1),
                   nn.BatchNorm2d(256),
                   activation]))

        # Fourth encoder
        self.layer8 = nn.Sequential(
                *([nn.Conv2d(256, 512, kernel_size=3, padding=1),
                   nn.BatchNorm2d(512),
                   activation]))
        self.layer9 = nn.Sequential(
                *([nn.Conv2d(512, 512, kernel_size=3, padding=1),
                   nn.BatchNorm2d(512),
                   activation]))
        self.layer10 = nn.Sequential(
                *([nn.Conv2d(512, 512, kernel_size=3, padding=1),
                   nn.BatchNorm2d(512),
                   activation]))

        # Fifth encoder
        self.layer11 = nn.Sequential(
                *([nn.Conv2d(512, 512, kernel_size=3, padding=1),
                   nn.BatchNorm2d(512),
                   activation]))
        self.layer12 = nn.Sequential(
                *([nn.Conv2d(512, 512, kernel_size=3, padding=1),
                   nn.BatchNorm2d(512),
                   activation]))
        self.layer13 = nn.Sequential(
                *([nn.Conv2d(512, 512, kernel_size=3, padding=1),
                   nn.BatchNorm2d(512),
                   activation]))
    
        # Classifier
        self.fc1 = nn.Sequential(*([
                nn.Dropout(),
                nn.Linear(512, 4096),
                activation]))
        self.fc2 = nn.Sequential(*([
                nn.Dropout(),
                nn.Linear(4096, 4096),
                activation]))
        self.classifier = nn.Sequential(*([
                nn.Linear(4096, 10)]))

        self.pool = nn.MaxPool2d(2, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):

        # Encoder 2
        layer1 = self.layer1(x)
        layer2 = self.layer2(layer1)
        pool1 = self.pool(layer2)

        # Encoder 2
        layer3 = self.layer3(pool1)
        layer4 = self.layer4(layer3)
        pool2 = self.pool(layer4)

        # Encoder 3
        layer5 = self.layer5(pool2)
        layer6 = self.layer6(layer5)
        layer7 = self.layer7(layer6)
        pool3 = self.pool(layer7)

        # Encoder 4
        layer8 = self.layer8(pool3)
        layer9 = self.layer9(layer8)
        layer10 = self.layer10(layer9)
        pool4 = self.pool(layer10)

        # Encoder 5
        layer11 = self.layer11(pool4)
        layer12 = self.layer12(layer11)
        layer13 = self.layer13(layer12)
        pool5 = self.pool(layer13)

        # Classifier

        fc1 = self.fc1(pool5.view(pool5.size(0), -1))
        fc2 = self.fc2(fc1)
        classifier = self.classifier(fc2)

        return [classifier, fc1, fc2, layer13, layer12, layer11,
                layer10, layer9, layer8, layer7, layer6, layer5,
                layer4, layer3, layer2, layer1]
