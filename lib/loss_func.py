import math
import torch
import torch.nn as nn


class MDN_loss(nn.Module):
    def __init__(self):
        super(MDN_loss, self).__init__()
        self.ONEOVERSQRT2PI = 1.0 / math.sqrt(2 * math.pi)

    def gaussian_probability(self, sigma, mu, target):
        target = target.unsqueeze(1).expand_as(sigma)
        ret = self.ONEOVERSQRT2PI * torch.exp(-0.5*((target-mu)/sigma)**2) / sigma
        return torch.prod(ret, 2)

    def forward(self, pi, sigma, mu, target):
        # Calculates the error, given the MoG parameters and the target
        # The loss is the negative log likelihood of the data given the MoG parameters.
        prob = pi * self.gaussian_probability(sigma, mu, target)
        nll  = -torch.log(torch.sum(prob, dim=1))
        loss = torch.mean(nll)
        #print("pi",pi)
        #print("sigma",sigma)
        #print("mu",mu)
        #print("target",target)
        #print(stop)
        return loss