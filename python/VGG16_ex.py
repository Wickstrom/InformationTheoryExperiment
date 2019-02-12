import torch as th
import numpy as np
import torch.nn as nn
from VGG16 import VGG16
from load_cifar10 import load_cifar10

gpu = th.cuda.is_available()
if gpu:
    th.cuda.set_device(1)
    path = '/root/data/'
else:
    path = '/home/kristoffer/data/cifar10/'

N = 10
batch_size_tr = 2000
batch_size_te = 200
epochs = 500
n_n = 50
alpha = 1.01

all_costs, all_scores, mi_list = [], [], []
all_scores = []

x_tr, y_tr, x_te, y_te = load_cifar10(path)

for n in range(N):

    cost, score, mi_sample = [], [], []

    if gpu:
        model = VGG16(nn.ReLU()).cuda()
    else:
        model = VGG16(nn.ReLU())

    for epoch in range(epochs):
        cost.append(model.train_model(x_tr, y_tr, model,
                                      batch_size_tr, gpu))
        print('Run Number: {}'.format(n), '\n',
              'Epoch number: {}'.format(epoch), '\n',
              'Cost: {}'.format(cost[-1]))
        with th.no_grad():
            mi_sample.append(model.compute_mi(x_te, y_te, n_n,
                                              alpha, batch_size_te,
                                              model, gpu))

    all_costs.append(cost)
    mi_list.append(mi_sample)
    all_scores.append(model.predict(x_te, y_te, model, batch_size_te, gpu))
    np.savez_compressed('/root/output/VGG16_ex_results_50.npz',
                        a=mi_list, b=all_costs, c=all_scores)