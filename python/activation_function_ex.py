import torch as th
import numpy as np
import torch.nn as nn
from load_mnist import load_mnist
from FC_Hero import FC_Hero

gpu = th.cuda.is_available()
if gpu:
    th.cuda.set_device(0)
    path = '/root/data/'
else:
    path = '/home/kristoffer/data/mnist/'

N = 10
batch_size_tr = 250
batch_size_te = 100
epochs = 150
n_n = 15

all_costs, all_scores, mi_list = [], [], []
all_scores = []
activation_func = [nn.Sigmoid(), nn.Tanh(), nn.ReLU(),
                   nn.LeakyReLU()]
a_type = ['sigmoid', 'tanh', 'relu', 'leaky_relu']

x_tr, y_tr, x_te, y_te = load_mnist(path, gpu)

for n in range(N):

    temp_cost, temp_score, temp_mi = [], [], []
    for a_idx, a_func in enumerate(activation_func):

        cost, score, mi_sample = [], [], []

        if gpu:
            model = FC_Hero(a_func, a_type[a_idx]).cuda()
        else:
            model = FC_Hero(a_func, a_type[a_idx])

        for epoch in range(epochs):
            cost.append(model.train_model(x_tr, y_tr, model,
                                          batch_size_tr, gpu))
            with th.no_grad():
                mi_sample.append(model.compute_mi(x_te, y_te, n_n,
                                                  batch_size_te,
                                                  model, gpu))
                score.append(model.predict(x_te, y_te, model,
                                           batch_size_te, gpu))
            print('Run Number: {}'.format(n), '\n',
                  'Activation function is: {}'.format(a_type[a_idx]), '\n',
                  'Epoch number: {}'.format(epoch), '\n',
                  'Cost: {}'.format(cost[-1]), '\n',
                  'Acc: {}'.format(score[-1]))

        temp_cost.append(cost)
        temp_score.append(score)
        temp_mi.append(mi_sample)

    all_costs.append(temp_cost)
    mi_list.append(temp_mi)
    all_scores.append(temp_score)
    np.savez_compressed('/root/output/activation_results_mean.npz',
                        a=mi_list, b=all_costs, c=all_scores)
