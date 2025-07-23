import torch
import torch.nn as nn
import math

class MDN_loss(nn.Module):
    def __init__(self):
        super(MDN_loss, self).__init__()

    def forward(self, pi, sigma, mu, target, reduce=True):
        target = target.unsqueeze(1).expand_as(mu)
        gauss = 1.0 / torch.sqrt(2.0 * torch.pi * sigma**2) * torch.exp(-0.5 * ((target - mu) / sigma)**2)
        weighted_gauss = pi * gauss
        weighted_gauss = weighted_gauss.sum(dim=1)
        mdn_loss = -torch.log(weighted_gauss)
        
        if reduce:
            return mdn_loss.mean()
        else:
            return mdn_loss
