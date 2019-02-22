import torch as th
import numpy as np
import torch.nn as nn
from VGG16 import VGG16
from load_cifar10 import load_cifar10

gpu = th.cuda.is_available()
if gpu:
    th.cuda.set_device(0)
    path = '/root/data/cifar10/'
else:
    path = '/home/kristoffer/data/cifar10/'

N = 10
batch_size_tr = 1000
batch_size_te = 200
epochs = 100
n_n = 20

all_costs, all_scores, mi_list = [], [], []
all_scores = []

x_tr, y_tr, x_te, y_te = load_cifar10(path, gpu)

for n in range(N):

    cost, score, mi_sample = [], [], []

    if gpu:
        model = VGG16(10, nn.ReLU()).cuda()
    else:
        model = VGG16(10, nn.ReLU())

    for epoch in range(epochs):

        batches_tr = list(model.make_batches(x_tr.shape[0], batch_size_tr))
        batches_te = list(model.make_batches(x_te.shape[0], batch_size_te))

        for idx_tr, idx_te in zip(batches_tr, batches_te):

            x_tr_b = x_tr[idx_tr]
            y_tr_b = y_tr[idx_tr]

            x_te_b = x_te[idx_te]
            y_te_b = y_te[idx_te]

            cost.append(model.train_model(x_tr, y_tr, model,
                                          batch_size_tr // 2, gpu))

            with th.no_grad():
                mi_sample.append(model.compute_mi(x_te, y_te, n_n,
                                                  batch_size_te // 2,
                                                  model, gpu))
                score.append(model.predict(x_te_b, y_te_b, model,
                                           batch_size_te // 2, gpu))
            print('Run Number: {}'.format(n), '\n',
                  'Epoch number: {}'.format(epoch), '\n',
                  'Cost: {}'.format(cost[-1]), '\n',
                  'Score: {}'.format(score[-1]))

    all_costs.append(cost)
    all_scores.append(score)
    mi_list.append(mi_sample)

    np.savez_compressed('/root/output/iteration_cifar10_results.npz',
                        a=mi_list, b=all_costs, c=all_scores)
