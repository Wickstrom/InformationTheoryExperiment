import pickle
import numpy as np


def unpickle(file):

    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def load_cifar10(path):

    data_x_tr = []
    data_y_tr = []

    for i in range(1, 6):
        data_tr = unpickle(path + 'data_batch_{}'.format(i))

        data_x_tr.append(data_tr[b'data'].reshape(10000, 3, 32, 32))
        data_y_tr.append(data_tr[b'labels'])

    x_tr = np.concatenate(data_x_tr)
    y_tr = np.concatenate(data_y_tr)

    data_te = unpickle(path + 'test_batch')

    x_te = data_te[b'data']
    x_te = x_te.reshape(10000, 3, 32, 32)

    y_te = np.array(data_te[b'labels'])

    return x_tr, y_tr, x_te, y_te
