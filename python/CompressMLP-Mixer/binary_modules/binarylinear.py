
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

from binary_modules.STEs import *

__all__ = ["BinaryLinearReCU", "BinaryLinear", "BinaryLinearMean", "BinaryLinearClamp"]






class BinaryLinearReCU(nn.Linear):

    def __init__(self, *kargs, **kwargs):
        super(BinaryLinearReCU, self).__init__(*kargs, **kwargs)
        self.alpha = nn.Parameter(torch.rand([1]), requires_grad=True)
        self.register_buffer('tau', torch.tensor(1.))

    def forward(self, input):
        w = self.weight
        a = input
        w0 = w - w.mean()

        w1 = w0 / (torch.sqrt(w0.var(()) + 1e-5) / 2 / np.sqrt(2))
        EW = torch.mean(torch.abs(w1))
        Q_tau = (- EW * torch.log(2-2 * self.tau)).detach().cpu().item()
        w2 = torch.clamp(w1, -Q_tau, Q_tau)
        if self.training:
            a0 = a / torch.sqrt(a.var() + 1e-5)
        else:
            a0 = a
        bw = BinaryQuantize().apply(w2)
        ba = BinaryQuantize().apply(a0)
        output = F.linear(input=ba, weight=bw, bias=self.bias)
        output = output * self.alpha
        #assert torch.sum(torch.isnan(output)) == 0
        return output


class BinaryLinear(nn.Linear):
    def __init__(self, *kargs, **kwargs):
        super(BinaryLinear, self).__init__(*kargs, **kwargs)

    def forward(self, input):
        bw = BinaryQuantize().apply(self.weight) * torch.mean(torch.abs(self.weight))
        ba = BinaryQuantize().apply(input) * torch.mean(torch.abs(input))
        output = F.linear(input=ba, weight=bw, bias=self.bias)
        return output


class BinaryLinearMean(nn.Linear):
    def __init__(self, *kargs, **kwargs):
        super(BinaryLinearMean, self).__init__(*kargs, **kwargs)

    def forward(self, input):
        w = self.weight
        w0 = w - w.mean()
        #w1 = w0 / (torch.sqrt(w0.var(()) + 1e-5) / 2 / np.sqrt(2))
        bw = BinaryQuantize().apply(w0) * torch.mean(torch.abs(w0))
        ba = BinaryQuantize().apply(input) * torch.mean(torch.abs(input))
        output = F.linear(input=ba, weight=bw, bias=self.bias)
        return output



class BinaryLinearClamp(nn.Linear):
    def __init__(self, *kargs, **kwargs):
        super(BinaryLinearClamp, self).__init__(*kargs, **kwargs)
        self.alpha = nn.Parameter(torch.rand([1]), requires_grad=True)


    def forward(self, input):
        a = input.clamp(min=-1, max=0)
        bw = BinaryQuantize().apply(self.weight)

        ba = BinaryQuantize().apply(a)
        ba = self.alpha * ba
        output = F.linear(input=ba, weight=bw, bias=self.bias)
        return output



class BinaryLinearNaive(nn.Linear):
    def __init__(self, *kargs, **kwargs):
        super(BinaryLinearNaive, self).__init__(*kargs, **kwargs)

    def forward(self, input):
        bw = BinaryQuantize().apply(self.weight)
        ba = BinaryQuantize().apply(input)
        output = F.linear(input=ba, weight=bw, bias=self.bias)
        return output
