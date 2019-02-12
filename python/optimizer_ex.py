import torch as th
import numpy as np
import torch.nn as nn
from CNN_5L import CNN5L
from load_mnist import load_mnist


gpu = th.cuda.is_available()
if gpu:
    th.cuda.set_device(0)
    path = '/root/data/'
else:
    path = '/home/kristoffer/data/mnist/'

N = 10
batch_size_tr = 100
batch_size_te = 50
epochs = 250
n_n = 25
number_filters = 3
number_neurons = 50
alpha = 1.01
number_neurons = 50

all_costs, all_scores, mi_list = [], [], []
all_scores = []
optimizers = [th.optim.SGD, th.optim.Adagrad, th.optim.Adam]

x_tr, y_tr, x_te, y_te = load_mnist(path, 'full')

for n in range(N):

    temp_cost, temp_score, temp_mi = [], [], []
    for optimizer in optimizers:

        cost, score, mi_sample = [], [], []

        if gpu:
            model = CNN5L(number_filters, number_neurons, nn.ReLU()).cuda()
        else:
            model = CNN5L(number_filters, number_neurons, nn.ReLU())

        for epoch in range(epochs):
            cost.append(model.train_model(x_tr, y_tr, model,
                                          batch_size_tr, gpu, optimizer))
            print('Run Number: {}'.format(n), '\n',
                  'Optimizer is: {}'.format(optimizer), '\n',
                  'Epoch number: {}'.format(epoch), '\n',
                  'Cost: {}'.format(cost[-1]))
            with th.no_grad():
                mi_sample.append(model.compute_mi(x_tr, y_tr, n_n, alpha,
                                                  batch_size_te, model, gpu))

        temp_cost.append(cost)
        temp_mi.append(mi_sample)

        temp_score.append(model.predict(x_te, y_te, model, batch_size_te, gpu))

    all_costs.append(temp_cost)
    mi_list.append(temp_mi)
    all_scores.append(temp_score)
    np.savez_compressed('/root/output/optimization_ex_results_25.npz',
                        a=mi_list, b=all_costs, c=all_scores)
