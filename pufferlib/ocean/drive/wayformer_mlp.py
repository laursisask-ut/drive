"""BF16 Wayformer MLP with a fused cuBLAS linear/ReLU forward operation.

The caller selects this path only for the specialized B200 execution. The
autograd formula retains the existing two linear layers and their parameters.
"""

import torch
from torch.nn import functional as F


class _FusedMLP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w1, b1, w2, b2, residual):
        # Inductor otherwise emits the ReLU and its saved mask as a separate
        # pass over the large hidden activation. This ATen operation retains
        # cuBLAS's fused epilogue, but requires an explicit autograd formula.
        activation = torch._addmm_activation(b1, x, w1.T, use_gelu=False)
        result = F.linear(activation, w2, b2) + residual
        ctx.save_for_backward(x, w1, w2, activation)
        return result

    @staticmethod
    def backward(ctx, grad):
        x, w1, w2, activation = ctx.saved_tensors
        grad = grad.contiguous()
        hidden_grad = torch.where(activation > 0, grad @ w2, 0.0)
        return (
            hidden_grad @ w1,
            hidden_grad.T @ x,
            hidden_grad.sum(0),
            grad.T @ activation,
            grad.sum(0),
            grad,
        )


def fused_mlp(x, layers, residual):
    """Apply the existing Linear/ReLU/Linear MLP and residual in BF16."""
    shape = x.shape
    return _FusedMLP.apply(
        x.to(torch.bfloat16).reshape(-1, shape[-1]),
        layers[0].weight.to(torch.bfloat16),
        layers[0].bias.to(torch.bfloat16),
        layers[2].weight.to(torch.bfloat16),
        layers[2].bias.to(torch.bfloat16),
        residual.reshape(-1, shape[-1]),
    ).reshape(shape)
