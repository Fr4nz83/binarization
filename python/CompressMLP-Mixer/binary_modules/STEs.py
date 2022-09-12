from torch.autograd import Function, Variable
from torch.cuda.amp import custom_fwd, custom_bwd, autocast
import torch


class BinaryQuantize(Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input):
        out = torch.sign(input)
        return out

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input


class BinaryQuantize_a(Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.sign(input)
        return out

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        input = ctx.saved_tensors[0]
        grad_input = (2 - torch.abs(2*input))
        grad_input = grad_input.clamp(min=0) * grad_output.clone()
        return grad_input