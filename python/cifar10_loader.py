import pickle
import numpy as np


def unpickle(file):

    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def load_cifar10(version, path):

    if version == 'batch_1':

        data_tr = unpickle(path + 'data_batch_1')

        x_tr = data_tr[b'data']
        x_tr = x_tr.reshape(x_tr.shape[0], 3, 32, 32)

        y_tr = data_tr[b'labels']

        data_te = unpickle(path + 'test_batch')

        x_te = data_te[b'data']
        x_te = x_te.reshape(x_te.shape[0], 3, 32, 32)

        y_te = data_te[b'labels']

    if version == 'batch_2':

        data_tr = unpickle(path + 'data_batch_2')

        x_tr = data_tr[b'data']
        x_tr = x_tr.reshape(x_tr.shape[0], 3, 32, 32)

        y_tr = data_tr[b'labels']

        data_te = unpickle(path + 'test_batch')

        x_te = data_te[b'data']
        x_te = x_te.reshape(x_te.shape[0], 3, 32, 32)

        y_te = data_te[b'labels']

    if version == 'batch_3':

        data_tr = unpickle(path + 'data_batch_3')

        x_tr = data_tr[b'data']
        x_tr = x_tr.reshape(x_tr.shape[0], 3, 32, 32)

        y_tr = data_tr[b'labels']

        data_te = unpickle(path + 'test_batch')

        x_te = data_te[b'data']
        x_te = x_te.reshape(x_te.shape[0], 3, 32, 32)

        y_te = data_te[b'labels']

    if version == 'batch_4':

        data_tr = unpickle(path + 'data_batch_4')

        x_tr = data_tr[b'data']
        x_tr = x_tr.reshape(x_tr.shape[0], 3, 32, 32)

        y_tr = data_tr[b'labels']

        data_te = unpickle(path + 'test_batch')

        x_te = data_te[b'data']
        x_te = x_te.reshape(x_te.shape[0], 3, 32, 32)

        y_te = data_te[b'labels']

    if version == 'batch_5':

        data_tr = unpickle(path + 'data_batch_1')

        x_tr = data_tr[b'data']
        x_tr = x_tr.reshape(x_tr.shape[0], 3, 32, 32)

        y_tr = data_tr[b'labels']

        data_te = unpickle(path + 'test_batch')

        x_te = data_te[b'data']
        x_te = x_te.reshape(x_te.shape[0], 3, 32, 32)

        y_te = data_te[b'labels']

    if version == 'full':

        data_tr = unpickle(path + 'data_batch_1')

        x_tr = data_tr[b'data']
        x_tr = x_tr.reshape(x_tr.shape[0], 3, 32, 32)

        y_tr = data_tr[b'labels']

        for i in [2, 3, 4, 5]:

            data_tr = unpickle(path + 'data_batch_{}'.format(i))

            x_temp = data_tr[b'data']
            x_temp = x_temp.reshape(x_temp.shape[0], 3, 32, 32)

            y_temp = data_tr[b'labels']

            x_tr = np.concatenate((x_tr, x_temp))
            y_tr = np.concatenate((y_tr, y_temp))

        data_te = unpickle(path + 'test_batch')

        x_te = data_te[b'data']
        x_te = x_te.reshape(x_te.shape[0], 3, 32, 32)

        y_te = data_te[b'labels']

    return x_tr, y_tr, x_te, y_te
