import numpy as np
import torch as th


def load_mnist(path, gpu):

    data = np.load(path + 'MNIST.npz')

    if gpu:

        x_tr = th.from_numpy(data['a'].reshape(60000, 1, 28, 28)).cuda()
        y_tr = th.from_numpy(data['b']).cuda()

        x_te = th.from_numpy(data['c'].reshape(10000, 1, 28, 28)).cuda()
        y_te = th.from_numpy(data['d']).cuda()

    else:

        x_tr = th.from_numpy(data['a'].reshape(60000, 1, 28, 28))
        y_tr = th.from_numpy(data['b'])
        x_te = th.from_numpy(data['c'].reshape(10000, 1, 28, 28))
        y_te = th.from_numpy(data['d'])

    return x_tr, y_tr, x_te, y_te
