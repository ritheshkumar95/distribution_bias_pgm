import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn as sns


def wrap_arr(arr, vol=False, req_grad=True):
    return Variable(torch.from_numpy(arr), req_grad, vol).cuda()


def get_data(batch_size=128, seq_len=6):
    batch_data = np.zeros((batch_size, seq_len + 1, 1)).astype('float32')
    data = 1
    N = batch_size
    for i in xrange(seq_len):
        # data1 = np.random.normal(loc=1.5 * data, scale=2., size=N)
        # data2 = np.random.uniform(-np.abs(data), np.abs(data), size=N)
        # mask = np.random.binomial(1, 0.5, N)

        data = np.random.uniform(-np.abs(data), np.abs(data), size=N)
        batch_data[:, i + 1, 0] = data
    return batch_data[:, :-1], batch_data[:, 1:]


def process_batch(network, criterion, opt=None, train=True):
    source, target = get_data()
    source = wrap_arr(source)
    target = wrap_arr(target, req_grad=False)

    pred = network(source)
    args = [target] + list(pred)[:-1]
    loss = criterion(*args).mean()

    if train:
        opt.zero_grad()
        loss.backward()
        opt.step()

    return loss.cpu().data.tolist()[0]


# def plot_distributions(model, N=4096):
#     fig, axs = plt.subplots(1, 6, sharex=False, sharey=False)
#     fig.subplots_adjust(wspace=0)
#     fig.set_size_inches(20, 3)
#     for i in xrange(6):
#         gen_data = model.inference(N).cpu().data.numpy()
#         true_data = get_data(N)[1]

#         sns.distplot(gen_data[:, i].flatten(), color="red", hist=True, ax=axs[i])
#         sns.distplot(true_data[:, i].flatten(), color="green", hist=True, ax=axs[i])
#     fig.savefig('all_timesteps.png')
#     plt.close('all')


def plot_distributions(model, N=4096):
    fig = plt.figure()
    gen_data = model.inference(N).cpu().data.numpy()
    true_data = get_data(N)[1]
    sns.distplot(gen_data[:, 0].flatten(), color="red", hist=True)
    sns.distplot(true_data[:, 0].flatten(), color="green", hist=True)
    fig.savefig('zeroth_timestep.png')
    plt.close('all')


class StandardGaussianRNN(nn.Module):
    def __init__(self):
        super(StandardGaussianRNN, self).__init__()
        self.rnn = nn.LSTM(1, 16, batch_first=True)
        self.output = nn.Linear(16, 1)

    def forward(self, x, hidden=None):
        out, hidden = self.rnn(x, hidden)
        mu = self.output(out)
        return mu, hidden

    def inference(self, batch_size=64):
        x = Variable(torch.zeros(batch_size, 1, 1)).float().cuda()
        hidden = None
        outputs = []
        for i in xrange(10):
            mu, hidden = self.forward(x, hidden)
            x = torch.normal(mu, torch.ones_like(mu)*.5)
            outputs.append(x)
        return torch.cat(outputs, 1)


class GaussianRNN(nn.Module):
    def __init__(self):
        super(GaussianRNN, self).__init__()
        self.rnn = nn.LSTM(1, 16, batch_first=True)
        self.output = nn.Linear(16, 2)

    def forward(self, x, hidden=None):
        out, hidden = self.rnn(x, hidden)
        mu, logvar = self.output(out).split(1, dim=-1)
        return mu, logvar, hidden

    def inference(self, batch_size=64):
        x = Variable(torch.zeros(batch_size, 1, 1)).float().cuda()
        hidden = None
        outputs = []
        for i in xrange(10):
            mu, logvar, hidden = self.forward(x, hidden)
            x = torch.normal(mu, logvar.mul(.5).exp())
            outputs.append(x)
        return torch.cat(outputs, 1)


class SkewedGaussianRNN(nn.Module):
    def __init__(self):
        super(SkewedGaussianRNN, self).__init__()
        self.rnn = nn.LSTM(1, 16, batch_first=True)
        self.output = nn.Linear(16, 3)

    def forward(self, x, hidden=None):
        out, hidden = self.rnn(x, hidden)
        mu, logvar1, logvar2 = self.output(out).split(1, dim=-1)
        return mu, logvar1, logvar2, hidden

    def inference(self, batch_size=64):
        x = Variable(torch.zeros(batch_size, 1, 1)).float().cuda()
        hidden = None
        outputs = []
        for i in xrange(10):
            mu, logvar1, logvar2, hidden = self.forward(x, hidden)
            sigma1 = logvar1.mul(.5).exp()
            sigma2 = logvar2.mul(.5).exp()
            x1 = torch.abs(torch.normal(mu, sigma1) - mu) + mu
            x2 = -torch.abs(torch.normal(mu, sigma2) - mu) + mu
            probs = torch.ones_like(x1)*0.5
            # probs = 0.5*(1+sigma2/sigma1)
            mask = torch.bernoulli(probs).float()
            x = mask*x1 + (1-mask)*x2
            outputs.append(x)
        return torch.cat(outputs, 1)


class GaussianMixtureRNN(nn.Module):
    def __init__(self, K=20):
        super(GaussianMixtureRNN, self).__init__()
        self.K = K
        self.rnn = nn.LSTM(1, 16, batch_first=True)
        self.output = nn.Linear(16, 3 * K)

    def forward(self, x, hidden=None):
        out, hidden = self.rnn(x, hidden)
        return list(self.output(out).split(self.K, dim=-1)) + [hidden]

    def inference(self, batch_size=64):
        x = Variable(torch.zeros(batch_size, 1, 1)).float().cuda()
        hidden = None
        outputs = []
        for i in xrange(10):
            mu, logvar, weights, hidden = self.forward(x, hidden)
            idxs = F.softmax(weights[:, 0], -1).multinomial()
            mu_select = mu[:, 0].gather(-1, idxs)
            std_select = logvar[:, 0].mul(0.5).exp().gather(-1, idxs)
            x = torch.normal(mu_select, std_select).unsqueeze(1)
            outputs.append(x)
        return torch.cat(outputs, 1)
