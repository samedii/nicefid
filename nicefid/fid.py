from multiprocessing.sharedctypes import Value
import torch


class _MatrixSquareRootEig(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a):
        vals, vecs = torch.linalg.eigh(a)
        ctx.save_for_backward(vals, vecs)
        return vecs @ vals.abs().sqrt().diag_embed() @ vecs.transpose(-2, -1)

    @staticmethod
    def backward(ctx, grad_output):
        vals, vecs = ctx.saved_tensors
        d = vals.abs().sqrt().unsqueeze(-1).repeat_interleave(vals.shape[-1], -1)
        vecs_t = vecs.transpose(-2, -1)
        return vecs @ (vecs_t @ grad_output @ vecs / (d + d.transpose(-2, -1))) @ vecs_t


def sqrtm_eig(a):
    if a.ndim < 2:
        raise ValueError("Tensor of matrices must have at least 2 dimensions")
    if a.shape[-2] != a.shape[-1]:
        raise ValueError("Tensor must be batches of square matrices")
    return _MatrixSquareRootEig.apply(a)


def fid(x, y, eps=1e-8):
    x_mean = x.mean(dim=0)
    y_mean = y.mean(dim=0)
    mean_term = (x_mean - y_mean).pow(2).sum()
    x_cov = torch.cov(x.T)
    y_cov = torch.cov(y.T)
    eps_eye = torch.eye(x_cov.shape[0], device=x_cov.device, dtype=x_cov.dtype) * eps
    x_cov_sqrt = sqrtm_eig(x_cov + eps_eye)
    cov_term = torch.trace(
        x_cov + y_cov - 2 * sqrtm_eig(x_cov_sqrt @ y_cov @ x_cov_sqrt + eps_eye)
    )
    return mean_term + cov_term
