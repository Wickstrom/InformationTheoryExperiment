import pickle
import numpy as np
import torch as th


def unpickle(file):

    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def load_cifar10(path, gpu):

    if gpu:
        data_x_tr = []
        data_y_tr = []

        for i in range(1, 6):
            data_tr = unpickle(path + 'data_batch_{}'.format(i))

            data_x_tr.append(data_tr[b'data'].reshape(10000, 3, 32, 32))
            data_y_tr.append(data_tr[b'labels'])

        x_tr = th.from_numpy(np.concatenate(data_x_tr)).float().cuda()
        y_tr = th.from_numpy(np.concatenate(data_y_tr)).long().cuda()

        data_te = unpickle(path + 'test_batch')

        x_te = data_te[b'data']
        x_te = th.from_numpy(x_te.reshape(10000, 3, 32, 32)).float().cuda()

        y_te = th.from_numpy(np.array(data_te[b'labels'])).long().cuda()
    else:
        data_x_tr = []
        data_y_tr = []

        for i in range(1, 6):
            data_tr = unpickle(path + 'data_batch_{}'.format(i))

            data_x_tr.append(data_tr[b'data'].reshape(10000, 3, 32, 32))
            data_y_tr.append(data_tr[b'labels'])

        x_tr = th.from_numpy(np.concatenate(data_x_tr)).float()
        y_tr = th.from_numpy(np.concatenate(data_y_tr)).long()

        data_te = unpickle(path + 'test_batch')

        x_te = data_te[b'data']
        x_te = th.from_numpy(x_te.reshape(10000, 3, 32, 32)).float()

        y_te = th.from_numpy(np.array(data_te[b'labels'])).long()       

    return x_tr, y_tr, x_te, y_te
