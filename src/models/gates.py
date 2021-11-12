import torch

class ReTanh(torch.autograd.Function):
    """
    ReTanh gate
    """

    @staticmethod
    def forward(ctx, input):
        y = input.tanh()
        ctx.save_for_backward(y)
        return y.clamp(min=0, max=1)

    @staticmethod
    def backward(ctx, grad_output):
        y, = ctx.saved_tensors
        grad_new = (1 - y**2)
        grad_new[y <= 0] = 0.0

        return grad_output* grad_new


class HeavisideReLU(torch.autograd.Function):
    """
    Heaviside activation function. ReLU substitute used to approximate gradients.
    """

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return torch.ceil(input).clamp(min=0, max=1)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input <= 0] = 0.0
        return grad_input


class HeavisideST(torch.autograd.Function):
    """
    Heaviside activation function with straight through estimator
    """

    @staticmethod
    def forward(ctx, input):
        return torch.ceil(input).clamp(min=0, max=1)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input


class Sigmoid(torch.autograd.Function):
    """
    Sigmoid gate
    """

    @staticmethod
    def forward(ctx, input):
        y = input.sigmoid()
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        y, = ctx.saved_tensors
        grad_new = y * (1 - y)
        return grad_output* grad_new