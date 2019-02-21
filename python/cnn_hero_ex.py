import torch as th
import numpy as np
import torch.nn as nn
from load_mnist import load_mnist
from CNN_Hero import CNN_Hero

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
n_n = 40

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
            model = CNN_Hero(a_func, a_type[a_idx]).cuda()
        else:
            model = CNN_Hero(a_func, a_type[a_idx])

        for epoch in range(epochs):
            cost.append(model.train_model(x_tr, y_tr, model,
                                          batch_size_tr, gpu))
            print('Run Number: {}'.format(n), '\n',
                  'Activation function is: {}'.format(a_type[a_idx]), '\n',
                  'Epoch number: {}'.format(epoch), '\n',
                  'Cost: {}'.format(cost[-1]))
            with th.no_grad():
                mi_sample.append(model.compute_mi(x_te, y_te, n_n,
                                                  batch_size_te,
                                                  model, gpu))

        temp_cost.append(cost)
        temp_mi.append(mi_sample)

        temp_score.append(model.predict(x_te, y_te, model, batch_size_te, gpu))

    all_costs.append(temp_cost)
    mi_list.append(temp_mi)
    all_scores.append(temp_score)
    np.savez_compressed('/root/output/cnn_hero_results_40.npz',
                        a=mi_list, b=all_costs, c=all_scores)
