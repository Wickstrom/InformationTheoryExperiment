import pickle
import numpy as np
import torch as th


def unpickle(file):

    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def load_cifar100(path, gpu):

    if gpu:
        data_tr = unpickle(path + 'train')

        x_tr = data_tr[b'data']
        x_tr = x_tr.reshape(50000, 3, 32, 32)
        y_tr = np.array(data_tr[b'fine_labels'])

        x_tr = th.from_numpy(x_tr).float().cuda()
        y_tr = th.from_numpy(y_tr).long().cuda()        

        data_te = unpickle(path + 'test')

        x_te = data_te[b'data']
        x_te = x_te.reshape(10000, 3, 32, 32)
        y_te = np.array(data_te[b'fine_labels'])

        x_te = th.from_numpy(x_te).float().cuda()
        y_te = th.from_numpy(y_te).long().cuda()     
    else:
        data_tr = unpickle(path + 'train')

        x_tr = data_tr[b'data']
        x_tr = x_tr.reshape(50000, 3, 32, 32)
        y_tr = np.array(data_tr[b'fine_labels'])

        x_tr = th.from_numpy(x_tr).float()
        y_tr = th.from_numpy(y_tr).long()       

        data_te = unpickle(path + 'test')

        x_te = data_te[b'data']
        x_te = x_te.reshape(10000, 3, 32, 32)
        y_te = np.array(data_te[b'fine_labels'])
 
        x_te = th.from_numpy(x_te).float()
        y_te = th.from_numpy(y_te).long()

    return x_tr, y_tr, x_te, y_te
