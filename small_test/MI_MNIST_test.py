# %%
import sys
import scipy
import torch as th
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
sys.path.insert(0, '/home/kristoffer/scripts/random/')
from numpy import linalg as LA
from load_mnist import load_mnist
from torch.autograd import Variable
from sklearn.metrics import confusion_matrix
from sklearn.metrics import  classification_report
from scipy.spatial.distance import pdist, squareform
from torch.distributions.dirichlet import Dirichlet

x_tr, y_tr, x_te, y_te = load_mnist('/home/kristoffer/data/mnist/')

x_tr = x_tr[:500]
y_tr = y_tr[:500]

x_te = x_te[:200]
y_te = y_te[:200]

# %%


class CNN(nn.Module):
    def __init__(self, n_f, activation):
        super(CNN, self).__init__()

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
                *([nn.Linear(n_f*7*7, 50),
                   activation]))

        self.layer6 = nn.Sequential(
                *([nn.Linear(50, 10)]))

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

    def dist_mat(self, x, y=None):

        try:
            x = th.from_numpy(x)
        except TypeError:
            x = x

        dist = th.norm(x[:, None] - x, dim=2, p=2)
        return dist / dist.max()

    def calc_kernel(self, x, n_n, sigma=None):

        k = self.dist_mat(x)

        if sigma is None:
            sigma = th.sort(k)[0][:, :n_n].mean()
        else:
            sigma = sigma
        k = th.exp(-k ** 2 / sigma ** 2)
        return k / th.trace(k)

    def kernel_mat(self, x, n_n, sigma=None):

        try:
            x = th.from_numpy(x)
        except TypeError:
            x = x
        if len(x.shape) == 2:
            return self.calc_kernel(x, n_n, sigma=sigma)
        else:
            x = x.view(x.size(0), -1)
            return self.calc_kernel(x, n_n, sigma=sigma)

    def entropy(self, x, n_n, alpha, sigma=None):

        eigv = th.abs(th.symeig(x)[0])
        eig_pow = eigv**alpha
        return (1/(1-alpha))*th.log2(eig_pow.sum())

    def j_entropy(self, x, y, n_n, alpha, sigma=None):

        k = x*y / (th.trace(x*y))
        eigv = th.abs(th.symeig(k)[0])
        eig_pow = eigv**alpha
        return (1/(1-alpha))*th.log2(eig_pow.sum())

    def compute_mi(self, x, y, n_n, alpha, sigma=None):

        data = self.forward(x)
        data.reverse()
        data[-1] = self.softmax(data[-1])
        data.insert(0, x)
        data.append(self.one_hot(y))

        k_list = [self.kernel_mat(i, n_n) for i in data]
        e_list = [self.entropy(i, n_n, alpha, sigma) for i in k_list]
        j_XT =  [self.j_entropy(k_list[0], k_i, n_n, alpha) for k_i in k_list[1:-1]]
        j_TY =  [self.j_entropy(k_i, k_list[-1], n_n, alpha) for k_i in k_list[1:-1]]

        MI = []

        for idx, val in enumerate(e_list[1:-1]):
            MI.append([e_list[0].data.numpy()+val.data.numpy()-j_XT[idx].data.numpy(),
                       e_list[-1].data.numpy()+val.data.numpy()-j_TY[idx].data.numpy()])

        return MI

    def one_hot(self, y):

        try:
            y = th.from_numpy(y)
        except TypeError:
            None

        y_1d = y
        y_hot = th.zeros((y.size(0), th.max(y).int()+1))

        for i in range(y.size(0)):
            y_hot[i, y_1d[i].int()] = 1

        return y_hot

    def one_hot_dirichlet(self, y):

        try:
            y = th.from_numpy(y)
        except TypeError:
            None

        y_1d = y
        y_hot = th.zeros((y.size(0), th.max(y).int()+1))
        y_dir = th.zeros((y.size(0), th.max(y).int()+1))

        for i in range(y.size(0)):
            y_hot[i, y_1d[i].int()] = 10000

        for i in range(y.size(0)):
            m = Dirichlet(y_hot[i] + 100)
            y_dir[i] = m.sample()

        return y_dir

# %%


N, mi_sample = 1, []

for i in range(N):
    model = CNN(5, nn.Tanh())
    criterion = nn.CrossEntropyLoss()
    optimizer = th.optim.Adam(model.parameters())
    cost = []
    mi_mat = []
    epochs = 350

    for epoch in range(epochs):

        optimizer.zero_grad()
        output = model(Variable(th.from_numpy(x_tr).float()))
        loss = criterion(output[0], Variable(th.from_numpy(y_tr).long()))

        loss.backward()
        optimizer.step()
        cost.append(np.float32(loss.data))
        print('Run number {}'.format(i), 'Epoch number {}'.format(epoch), cost[-1])
        with th.no_grad():
            mi_mat.append(model.compute_mi(th.from_numpy(x_te).float(),
                                           th.from_numpy(y_te).long(), 50, 1.01))
    mi_sample.append(mi_mat)


#output = model(Variable(th.from_numpy(x_te[:100].reshape(100, 1, 28, 28)).float()))
#cf_mat = confusion_matrix(y_te[:100], np.argmax(output[0].data.numpy(), 1))
#rapport = classification_report(y_te[:100], np.argmax(output[0].data.numpy(), 1))


# %%

xy1 = np.zeros((len(mi_sample), len(mi_sample[0]), len(mi_sample[0][0]), 2))

plt.figure(1, figsize=(10, 6))
plt.style.use('ggplot')
c_lab = ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds']
layer = [1, 2, 3, 4, 5, 6]

for i1, s1 in enumerate(mi_sample):
    for i2, s2 in enumerate(s1):
        for i3, s3 in enumerate(s2):
            xy1[i1, i2, i3, 0] = s3[0]
            xy1[i1, i2, i3, 1] = s3[1]     

for m in range(N):
    for j in range(len(mi_sample[0][0])):
                plt.scatter(xy1[m, :, j, 0], xy1[m, :, j, 1], cmap=c_lab[j], c=np.arange(0, xy1.shape[1], 1), edgecolor=c_lab[j][:-1], s=30)


for j in range(len(mi_sample[0][0])):
            plt.scatter(xy1[0, -1, j, 0], xy1[0, -1, j, 1], c=c_lab[j][:-1], label='Layer {}'.format(j+1), s=30)

plt.legend(facecolor='white')
plt.xlabel('MI(X,T)')
plt.ylabel('MI(T,Y)')
plt.tight_layout()
plt.show()
# %%

plt.style.use('ggplot')
for samples in mi_sample:
    xy1 = np.zeros((6, epochs//5-1, 2))
    c_lab = ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds']
    layer = [1, 2, 3, 4, 5, 6]

    for j, idx in enumerate(layer):
        for i, value in enumerate(mi_mat):
            xy1[j, i, 0] = samples[i][0]
            xy1[j, i, 1] = samples[i][1]
    plt.figure(1, figsize=(10, 6))
    for m, hei in enumerate(layer):
        plt.scatter(xy1[m, :, 0], xy1[m, :, 1], cmap=c_lab[m], c=np.arange(0,xy1.shape[1], 1), edgecolor=c_lab[m][:-1], s=30)
#        if hei == layer[0]:
#            plt.colorbar(pad=-0.12)
#        elif hei == layer[-1]:
#            plt.colorbar(ticks=[], pad=0.001)
#        else:
#        250    plt.colorbar(ticks=[], pad=-0.12)
for m, hei in enumerate(layer):
    plt.scatter(xy1[m, -1, 0], xy1[m, -1, 1], c=c_lab[m][:-1], label='Layer {}'.format(hei), s=30)
plt.legend(facecolor='white')
plt.xlabel('MI(X,T)')
plt.ylabel('MI(T,Y)')
plt.tight_layout()
plt.show()

# %%

mi = np.zeros((N, len(mi_mat), 8, 8))

for i, sample in enumerate(mi_sample):
    for j, mat in enumerate(sample):
        mi[i, j] = mat

mi = np.mean(mi, 0)

xy1 = np.zeros((6, epochs//5-1, 2))
c_lab = ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds']
layer = [1, 2, 3, 4, 5, 6]

for j, idx in enumerate(layer):
    for i, value in enumerate(mi_mat):
        xy1[j, i, 0] = mi[i, idx, 0]
        xy1[j, i, 1] = mi[i, 7, idx]
plt.figure(1, figsize=(10, 6))
for m, hei in enumerate(layer):
    plt.scatter(xy1[m, :, 0], xy1[m, :, 1], cmap=c_lab[m], c=np.arange(0,xy1.shape[1], 1), edgecolor=c_lab[m][:-1], s=30)
    if hei == layer[0]:
        plt.colorbar(pad=-0.12)
    elif hei == layer[-1]:
        plt.colorbar(ticks=[], pad=0.001)
    else:
        plt.colorbar(ticks=[], pad=-0.12)
for m, hei in enumerate(layer):
    plt.scatter(xy1[m, -1, 0], xy1[m, -1, 1], c=c_lab[m][:-1], label='Layer {}'.format(hei), s=30)
plt.legend(facecolor='white')
plt.xlabel('MI(X,T)')
plt.ylabel('MI(T,Y)')
plt.tight_layout()
plt.show()
