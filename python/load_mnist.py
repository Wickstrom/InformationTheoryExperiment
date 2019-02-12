import numpy as np
import torch as th


def load_mnist(path, gpu):

    data = np.load(path + 'MNIST.npz')

    if gpu:

        x_tr = th.from_numpy(data['a'].reshape(60000, 1, 28, 28)).float().cuda()
        y_tr = th.from_numpy(data['b']).long().cuda()

        x_te = th.from_numpy(data['c'].reshape(10000, 1, 28, 28)).float().cuda()
        y_te = th.from_numpy(data['d']).long().cuda()

    else:

        x_tr = th.from_numpy(data['a'].reshape(60000, 1, 28, 28)).float()
        y_tr = th.from_numpy(data['b']).long()
        x_te = th.from_numpy(data['c'].reshape(10000, 1, 28, 28)).float()
        y_te = th.from_numpy(data['d']).long()

    return x_tr, y_tr, x_te, y_te
