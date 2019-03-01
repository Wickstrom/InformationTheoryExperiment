import random
import torch as th
import numpy as np
import torch.nn as nn
import torch.nn.init as init
from scipy.stats import chi2
from torch.distributions.dirichlet import Dirichlet


class Network():
    def train_model(self, x, y, model, batch_size, gpu,
                    optimizer=th.optim.SGD):

        optimizer = optimizer(model.parameters(), lr=0.09)
        criterion = nn.CrossEntropyLoss()

        model.train()
        cost = []
        batches = self.make_batches(x.shape[0], batch_size)

        for batch in batches:
            optimizer.zero_grad()
            batch_x = x[batch]
            batch_y = y[batch]
            output = model(batch_x)
            loss = criterion(output[0], batch_y)
            loss.backward()
            optimizer.step()
            cost.append(np.float32(loss.cpu().data))

        return np.mean(cost)

    def predict(self, x, y, model, batch_size, gpu):

        model.eval()
        scores = []
        batches = self.make_batches(x.shape[0], batch_size)

        for batch in batches:

            batch_x = x[batch]
            batch_y = y[batch]

            output = model(batch_x)
            y_hat = th.argmax(self.softmax(output[0]), 1)
            scores.append(th.eq(batch_y, y_hat).sum().cpu().data.numpy() /
                          batch_x.shape[0])

        return np.mean(scores)

    def dist_mat(self, x, y=None):

        try:
            x = th.from_numpy(x)
        except TypeError:
            x = x

        if len(x.size()) == 4:
            x = x.view(x.size()[0], -1)

        dist = th.norm(x[:, None] - x, dim=2, p=2)
        return dist

    def kernel_mat(self, x, n_n, sigma=None):

        d = self.dist_mat(x)
        if sigma is None:
            if len(x.size()) == 4:
                dim = x.size(2)*x.size(3)
            else:
                dim = x.size(1)
            sigma = th.sort(d)[0][:, n_n].mean() / np.log(chi2.ppf(q=0.05, df=dim-1))
        else:
            sigma = sigma
        k = th.exp(-d ** 2 / (2*sigma ** 2))
        return k

    def entropy(self, *args):

        for idx, val in enumerate(args):
            if idx == 0:
                k = val.clone()
            else:
                k *= val

        k = k / k.trace()
        eigv = th.symeig(k)[0].abs()

        return -(eigv*(eigv.log2())).sum()

    def compute_mi(self, x, y, n_n, batch_size, model, gpu):

        model.eval()
        batches = list(self.make_batches(x.shape[0], batch_size))
        MI = []

        for idx, batch in enumerate(batches):

            MI_temp = []

            batch_x = x[batch]
            batch_y = y[batch]

            data = self.forward(batch_x)
            data.reverse()
            data[-1] = self.softmax(data[-1])
            data.insert(0, batch_x)
            data.append(self.one_hot(batch_y, gpu))

            k_list = [self.kernel_mat(data[0], n_n, sigma=10.0)]

            for idx, val in enumerate(data[1:-2]):
                k_list.append(self.kernel_mat(val, n_n))

            k_list.append(self.kernel_mat(data[5], n_n, sigma=0.3))
            k_list.append(self.kernel_mat(data[6], n_n, sigma=0.3))

            e_list = [self.entropy(i) for i in k_list]
            j_XT = [self.entropy(k_list[0], k_i) for k_i in k_list[1:-1]]
            j_TY = [self.entropy(k_i, k_list[-1]) for k_i in k_list[1:-1]]

            for idx, val in enumerate(e_list[1:-1]):
                MI_temp.append([e_list[0].cpu().data.numpy() +
                                val.cpu().data.numpy() -
                                j_XT[idx].cpu().data.numpy(),
                                e_list[-1].cpu().data.numpy() +
                                val.cpu().data.numpy() -
                                j_TY[idx].cpu().data.numpy()])
            MI.append(MI_temp)
        return np.array(MI).mean(0)

    def one_hot(self, y, gpu):

        try:
            y = th.from_numpy(y)
        except TypeError:
            None

        y_1d = y
        if gpu:
            y_hot = th.zeros((y.size(0), th.max(y).int()+1)).cuda()
        else:
            y_hot = th.zeros((y.size(0), th.max(y).int()+1))

        for i in range(y.size(0)):
            y_hot[i, y_1d[i].int()] = 1

        return y_hot

    def make_batches(self, N, batch_size):

        idx = random.sample(range(0, N), N)

        for i in range(0, N, batch_size):
            yield idx[i:i+batch_size]

    def weight_init(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            if self.a_type == 'relu':
                init.kaiming_normal_(m.weight.data, nonlinearity=self.a_type)
                init.constant_(m.bias.data, 0)
            elif self.a_type == 'leaky_relu':
                init.kaiming_normal_(m.weight.data, nonlinearity=self.a_type)
                init.constant_(m.bias.data, 0)
            elif self.a_type == 'tanh':
                g = init.calculate_gain(self.a_type)
                init.xavier_normal_(m.weight.data, gain=g)
                init.constant_(m.bias.data, 0)
            elif self.a_type == 'sigmoid':
                g = init.calculate_gain(self.a_type)
                init.xavier_normal_(m.weight.data, gain=g)
                init.constant_(m.bias.data, 0)
            else:
                raise
                return NotImplemented

    def one_hot_dirichlet(self, y, gpu):

        try:
            y = th.from_numpy(y)
        except TypeError:
            None

        y_1d = y
        if gpu:
            y_hot = th.zeros((y.size(0), th.max(y).int()+1)).cuda()
            y_dir = th.zeros((y.size(0), th.max(y).int()+1)).cuda()
        else:
            y_hot = th.zeros((y.size(0), th.max(y).int()+1))
            y_dir = th.zeros((y.size(0), th.max(y).int()+1))

        for i in range(y.size(0)):
            y_hot[i, y_1d[i].int()] = 1000000

        for i in range(y.size(0)):
            m = Dirichlet(y_hot[i] + 1000)
            y_dir[i] = m.sample()

        return y_dir
