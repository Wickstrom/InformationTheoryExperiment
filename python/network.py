import random
import torch as th
import torch.nn as nn
import torch.nn.init as init


class Network():
    def train_model(self, x, y, model, gpu, optimizer=th.optim.SGD):

        optimizer = optimizer(model.parameters(), lr=0.09)
        criterion = nn.CrossEntropyLoss()

        model.train()

        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output[0], y)
        loss.backward()
        optimizer.step()

        self.cost.append(loss.item())

        return

    def predict(self, x, y, model, gpu):

        model.eval()
        output = model(x)
        y_hat = th.argmax(self.softmax(output[0]), 1)
        score = th.eq(y, y_hat).sum().float() / x.size(0)
        self.score.append(score.item())

        return

    def dist_mat(self, x):

        try:
            x = th.from_numpy(x)
        except TypeError:
            x = x

        if len(x.size()) == 4:
            x = x.view(x.size()[0], -1)

        dist = th.norm(x[:, None] - x, dim=2)
        return dist

    def entropy(self, *args):

        for idx, val in enumerate(args):
            if idx == 0:
                k = val.clone()
            else:
                k *= val

        k /= k.trace()
        eigv = th.symeig(k)[0].abs()

        return -(eigv*(eigv.log2())).sum()

    def kernel_mat(self, x, k_x, k_y, sigma=None, epoch=None, idx=None):

        d = self.dist_mat(x)
        if sigma is None:
            if epoch > 20:
                sigma_vals = th.linspace(0.3, 10*d.mean(), 100)
            else:
                sigma_vals = th.linspace(0.3, 10*d.mean(), 300)
            L = []
            for sig in sigma_vals:
                k_l = th.exp(-d ** 2 / (sig ** 2)) / d.size(0)
                L.append(self.kernel_loss(k_x, k_y, k_l))

            if epoch == 0:
                self.sigmas[idx+1, epoch] = sigma_vals[L.index(max(L))]
            else:
                self.sigmas[idx+1, epoch] = 0.9*self.sigmas[idx+1, epoch-1] + 0.1*sigma_vals[L.index(max(L))]
        
            sigma = self.sigmas[idx+1, epoch]
        return th.exp(-d ** 2 / (sigma ** 2))

    def kernel_loss(self, k_x, k_y, k_l):

        beta = 1.0

        L = th.norm(k_l)
        Y = th.norm(k_y) ** beta
        X = th.norm(k_x) ** (1-beta)

        LY = th.trace(th.matmul(k_l, k_y))**beta
        LX = th.trace(th.matmul(k_l, k_x))**(1-beta)

        return 2*th.log2((LY*LX)/(L*Y*X))

    def compute_mi(self, x, y, model, gpu, current_iteration):

        model.eval()

        data = self.forward(x)
        data.reverse()
        data[-1] = self.softmax(data[-1])
        data.insert(0, x)
        data.append(self.one_hot(y, gpu))

        k_x = self.kernel_mat(data[0], [], [], sigma=th.tensor(8.0))
        k_y = self.kernel_mat(data[-1], [], [], sigma=th.tensor(0.1))

        k_list = [k_x]
        for idx_l, val in enumerate(data[1:-1]):
            k_list.append(self.kernel_mat(val.reshape(data[0].size(0), -1),
                                          k_x, k_y, epoch=current_iteration,
                                          idx=idx_l))
        k_list.append(k_y)

        e_list = [self.entropy(i) for i in k_list]
        j_XT = [self.entropy(k_list[0], k_i) for k_i in k_list[1:-1]]
        j_TY = [self.entropy(k_i, k_list[-1]) for k_i in k_list[1:-1]]

        for idx_mi, val_mi in enumerate(e_list[1:-1]):
            self.MI[current_iteration, idx_mi, 0] = e_list[0]+val_mi-j_XT[idx_mi]
            self.MI[current_iteration, idx_mi, 1] = e_list[-1]+val_mi-j_TY[idx_mi]

        return

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
                init.xavier_uniform_(m.weight.data, gain=g)
                init.constant_(m.bias.data, 0)
            elif self.a_type == 'sigmoid':
                g = init.calculate_gain(self.a_type)
                init.xavier_uniform_(m.weight.data, gain=g)
                init.constant_(m.bias.data, 0)
            else:
                raise
                return NotImplemented

    def make_batches(self, N, batch_size):

        idx = random.sample(range(0, N), N)

        for i in range(0, N, batch_size):
            yield idx[i:i+batch_size]

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
