import torch as th
import numpy as np
import torch.nn as nn
from CNN_4L import CNN4L
from CNN_5L import CNN5L
from CNN_6L import CNN6L
from load_mnist import load_mnist

gpu = th.cuda.is_available()
if gpu:
    th.cuda.set_device(0)
    path = '/root/data/'
else:
    path = '/home/kristoffer/data/mnist/'

N = 10
batch_size_tr = 2000
batch_size_te = 200
epochs = 200
n_n = 25
alpha = 1.01
number_filters = 10
number_neurons = 50

all_costs, all_scores, mi_list = [], [], []
all_scores = []
a_func = nn.ReLU()
models = [CNN4L, CNN5L, CNN6L]

x_tr, y_tr, x_te, y_te = load_mnist(path, 'full')


for n in range(N):
    temp_cost, temp_score, temp_mi = [], [], []
    for model in models:

        cost, score, mi_sample = [], [], []

        if gpu:
            model = model(number_filters, number_neurons, a_func).cuda()
        else:
            model = model(number_filters, number_neurons, a_func)

        for epoch in range(epochs):
            cost.append(model.train_model(x_tr, y_tr,
                                          model, batch_size_tr, gpu))
            print('Run Number: {}'.format(n), '\n',
                  'Model is: {}'.format(str(model)[:5]), '\n',
                  'Epoch number {}'.format(epoch), '\n',
                  'Cost: {}'.format(cost[-1]))
            with th.no_grad():
                mi_sample.append(model.compute_mi(x_te, y_te, n_n,
                                                  alpha, batch_size_te,
                                                  model, gpu))

        temp_cost.append(cost)
        temp_mi.append(mi_sample)

        temp_score.append(model.predict(x_te, y_te, model, batch_size_te, gpu))

    all_costs.append(temp_cost)
    mi_list.append(temp_mi)
    all_scores.append(temp_score)
    np.savez_compressed('/root/output/multi_layer_IP_ex_results_50j.npz',
                        a=mi_list, b=all_costs, c=all_scores)
