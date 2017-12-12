import torch
import torch.nn.functional as F
import numpy as np


def StandardGaussianNLL(x, means, sigma=.5):
    exponent = torch.pow(x - means, 2) / sigma**2
    constant = np.log(2 * np.pi * sigma**2)
    loglikelihood = -0.5 * (exponent + constant)
    return -loglikelihood


def GaussianNLL(x, means, logvars):
    exponent = torch.pow(x - means, 2)
    exponent = exponent / logvars.exp()
    constant = np.log(2 * np.pi) + logvars

    loglikelihood = -0.5 * (exponent + constant)
    return -loglikelihood


def SkewedGaussianNLL(x, means, logvars1, logvars2):
    exponent = torch.pow(x - means, 2)

    sum_stds = logvars1.mul(0.5).exp() + logvars2.mul(0.5).exp()
    A = 0.5 * np.log(2/np.pi) - torch.log(sum_stds)

    exponent1 = -0.5 * exponent / logvars1.exp()
    loglikelihood1 = A + exponent1

    exponent2 = -0.5 * exponent / logvars2.exp()
    loglikelihood2 = A + exponent2

    mask = x.lt(means).float()
    loglikelihood = mask*loglikelihood1 + (1-mask)*loglikelihood2
    return -loglikelihood


def GaussianMixtureNLL(x, means, logvars, weights):
    exponent = torch.pow(x - means, 2)
    exponent = exponent / logvars.exp()
    constant = np.log(2 * np.pi) + logvars

    likelihood = -0.5 * (exponent + constant)
    likelihood += F.log_softmax(weights, -1)

    def logsumexp(x, dim=-1):
        a = x.max(dim, keepdim=True)[0]
        return a + torch.log((x - a).exp().sum(dim, keepdim=True))

    return -logsumexp(likelihood)
