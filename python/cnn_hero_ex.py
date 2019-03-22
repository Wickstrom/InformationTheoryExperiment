import random
import torch as th
import numpy as np
from CNN_Hero import CNN_Hero
from load_mnist import load_mnist

gpu = th.cuda.is_available()
if gpu:
    th.cuda.set_device(0)
    path = '/root/data/'
else:
    path = '/home/kristoffer/data/mnist/'

N = 10
batch_size_tr = 100
batch_size_te = 200
epochs = 10
tr_size = 60000
n_iterations = (tr_size // batch_size_tr)*epochs

activation = 'relu'

x_tr, y_tr, x_te, y_te = load_mnist(path, gpu)

for n in range(N):
    current_iteration = 0
    if gpu:
        model = CNN_Hero(activation, n_iterations).cuda()
    else:
        model = CNN_Hero(activation, n_iterations)

    for epoch in range(epochs):

        batches_tr = list(model.make_batches(tr_size, batch_size_tr))

        for idx_tr in batches_tr:

            x_tr_b = x_tr[idx_tr]
            y_tr_b = y_tr[idx_tr]

            idx_te = random.sample(range(0, 10000), batch_size_te)

            x_te_b = x_tr_b #x_te[idx_te]
            y_te_b = y_tr_b #y_te[idx_te]

            model.train_model(x_tr_b, y_tr_b, model, gpu)
            with th.no_grad():
                model.predict(x_te_b, y_te_b, model, gpu)
                model.compute_mi(x_te_b, y_te_b, model, gpu, current_iteration)
                current_iteration += 1

        print('Run Number: {}'.format(n), '\n',
              'Epoch number: {}'.format(epoch), '\n',
              'Cost: {}'.format(model.cost[-1]), '\n',
              'Acc: {}'.format(model.score[-1]))

    if n == 0:
        mi = model.MI.cpu().detach().numpy().reshape(1, n_iterations, 5, 2)
        c_out = np.array(model.cost).reshape(1, -1)
        s_out = np.array(model.score).reshape(1, -1)
    else:
        mi = np.concatenate((mi, model.MI.cpu().detach().numpy().reshape(1, n_iterations, 5, 2)))
        c_out = np.concatenate((c_out, np.array(model.cost).reshape(1, -1)))
        s_out = np.concatenate((s_out, np.array(model.score).reshape(1, -1)))
    np.savez_compressed('/root/output/cnn_train_relu.npz',
                        a=mi, b=c_out, c=s_out)
