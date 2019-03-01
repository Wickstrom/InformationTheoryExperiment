import torch as th
import numpy as np
import torch.nn as nn
from load_mnist import load_mnist
from FC_Hero import FC_HERO

gpu = th.cuda.is_available()
if gpu:
    th.cuda.set_device(0)
    path = '/root/data/'
else:
    path = '/home/kristoffer/data/mnist/'

N = 10
batch_size_tr = 1200
batch_size_te = 200
epochs = 50
n_n = 20
all_costs, all_scores, mi_list = [], [], []
all_scores = []
activation_func = ['sigmoid', 'tanh', 'relu', 'leaky_relu']

x_tr, y_tr, x_te, y_te = load_mnist(path, gpu)

for n in range(N):

    temp_cost, temp_score, temp_mi = [], [], []
    for a_func in activation_func:

        cost, score, mi_sample = [], [], []

        if gpu:
            model = FC_HERO(a_func).cuda()
        else:
            model = FC_HERO(a_func)

        for epoch in range(epochs):

            batches_tr = list(model.make_batches(x_tr.shape[0], batch_size_tr))
            batches_te = list(model.make_batches(x_te.shape[0], batch_size_te))

            for idx_tr, idx_te in zip(batches_tr, batches_te):

                x_tr_b = x_tr[idx_tr]
                y_tr_b = y_tr[idx_tr]

                x_te_b = x_te[idx_te]
                y_te_b = y_te[idx_te]

                cost.append(model.train_model(x_tr_b, y_tr_b, model,
                                              batch_size_tr // 2, gpu))
                with th.no_grad():
                    mi_sample.append(model.compute_mi(x_te_b, y_te_b, n_n,
                                                      batch_size_te // 2,
                                                      model, gpu))
                    score.append(model.predict(x_te_b, y_te_b, model,
                                               batch_size_te // 2, gpu))
            print('Run Number: {}'.format(n), '\n',
                  'Activation function is: {}'.format(a_func), '\n',
                  'Epoch number: {}'.format(epoch), '\n',
                  'Cost: {}'.format(cost[-1]), '\n',
                  'Acc: {}'.format(score[-1]))

        temp_cost.append(cost)
        temp_score.append(score)
        temp_mi.append(mi_sample)

    all_costs.append(temp_cost)
    mi_list.append(temp_mi)
    all_scores.append(temp_score)
    np.savez_compressed('/root/output/iteration_results_20.npz',
                        a=mi_list, b=all_costs, c=all_scores)
