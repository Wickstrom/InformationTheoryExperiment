import numpy as np


def load_mnist(path, version='HE'):

    data = np.load(path + 'MNIST.npz')

    if version == 'HE':

        x_tr = data['a'][:6000].reshape(6000, 1, 28, 28)
        y_tr = data['b'][:6000]

        x_te = data['c'][:1000].reshape(1000, 1, 28, 28)
        y_te = data['d'][:1000]

    if version == 'full':

        x_tr = data['a'].reshape(60000, 1, 28, 28)
        y_tr = data['b']

        x_te = data['c'].reshape(10000, 1, 28, 28)
        y_te = data['d']

    return x_tr, y_tr, x_te, y_te
