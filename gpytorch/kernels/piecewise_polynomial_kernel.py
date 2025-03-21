import math
from typing import Optional

import torch
from torch import Tensor

from .kernel import Kernel


def _fmax(r: Tensor, j: int, q: int) -> Tensor:
    return torch.max(torch.tensor(0.0, dtype=r.dtype, device=r.device), 1 - r).pow(j + q)


def _get_cov(r: Tensor, j: int, q: int) -> Tensor:
    if q == 0:
        return 1
    if q == 1:
        return (j + 1) * r + 1
    if q == 2:
        return 1 + (j + 2) * r + ((j + 4 * j + 3) / 3.0) * (r**2)
    if q == 3:
        return (
            1
            + (j + 3) * r
            + ((6 * j**2 + 36 * j + 45) / 15.0) * r.square()
            + ((j**3 + 9 * j**2 + 23 * j + 15) / 15.0) * (r**3)
        )


class PiecewisePolynomialKernel(Kernel):
    r"""
    Computes a covariance matrix based on the Piecewise Polynomial kernel
    between inputs :math:`\mathbf{x_1}` and :math:`\mathbf{x_2}`:

    .. math::

        \begin{align}
            r &= \left\Vert x1 - x2 \right\Vert \\
            j &= \lfloor \frac{D}{2} \rfloor + q +1 \\
            K_{\text{ppD, 0}}(\mathbf{x_1}, \mathbf{x_2}) &= (1-r)^j_+ , \\
            K_{\text{ppD, 1}}(\mathbf{x_1}, \mathbf{x_2}) &= (1-r)^{j+1}_+ ((j + 1)r + 1), \\
            K_{\text{ppD, 2}}(\mathbf{x_1}, \mathbf{x_2}) &= (1-r)^{j+2}_+ ((1 + (j+2)r +
                \frac{j^2 + 4j + 3}{3}r^2), \\
            K_{\text{ppD, 3}}(\mathbf{x_1}, \mathbf{x_2}) &= (1-r)^{j+3}_+
                (1 + (j+3)r + \frac{6j^2 + 36j + 45}{15}r^2 +
                \frac{j^3 + 9j^2 + 23j +15}{15}r^3) \\
        \end{align}

    where :math:`K_{\text{ppD, q}}` is positive semidefinite in :math:`\mathbb{R}^{D}` and
    :math:`q` is the smoothness coefficient. See `Rasmussen and Williams (2006)`_ Equation 4.21.

    .. note:: This kernel does not have an `outputscale` parameter. To add a scaling parameter,
        decorate this kernel with a :class:`gpytorch.kernels.ScaleKernel`.

    :param int q: (default= 2) The smoothness parameter.
    :type q: int (0, 1, 2 or 3)
    :param ard_num_dims: (Default: `None`) Set this if you want a separate lengthscale for each
        input dimension. It should be `d` if x1 is a `... x n x d` matrix.
    :type ard_num_dims: int, optional
    :param batch_shape: (Default: `None`) Set this if you want a separate lengthscale for each
         batch of input data. It should be `torch.Size([b1, b2])` for a `b1 x b2 x n x m` kernel output.
    :type batch_shape: torch.Size, optional
    :param active_dims: (Default: `None`) Set this if you want to
        compute the covariance of only a few input dimensions. The ints
        corresponds to the indices of the dimensions.
    :type active_dims: Tuple(int)
    :param lengthscale_prior: (Default: `None`)
        Set this if you want to apply a prior to the lengthscale parameter.
    :type lengthscale_prior: ~gpytorch.priors.Prior, optional
    :param lengthscale_constraint: (Default: `Positive`) Set this if you want
        to apply a constraint to the lengthscale parameter.
    :type lengthscale_constraint: ~gpytorch.constraints.Positive, optional
    :param eps: (Default: 1e-6) The minimum value that the lengthscale can take (prevents divide by zero errors).
    :type eps: float, optional

    .. _Rasmussen and Williams (2006):
        http://www.gaussianprocess.org/gpml/

    Example:
        >>> x = torch.randn(10, 5)
        >>> # Non-batch option
        >>> covar_module = gpytorch.kernels.ScaleKernel(
                                gpytorch.kernels.PiecewisePolynomialKernel(q = 2))
        >>> # Non-batch: ARD (different lengthscale for each input dimension)
        >>> covar_module = gpytorch.kernels.ScaleKernel(
                            gpytorch.kernels.PiecewisePolynomialKernel(q = 2, ard_num_dims=5)
                            )
        >>> covar = covar_module(x)  # Output: LinearOperator of size (10 x 10)
        >>> batch_x = torch.randn(2, 10, 5)
        >>> # Batch: different lengthscale for each batch
        >>> covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.PiecewisePolynomialKernel(q = 2, batch_shape=torch.Size([2]))
            )
        >>> covar = covar_module(batch_x)  # Output: LinearOperator of size (2 x 10 x 10)
    """
    has_lengthscale = True

    def __init__(self, q: Optional[int] = 2, **kwargs):
        super(PiecewisePolynomialKernel, self).__init__(**kwargs)
        if q not in {0, 1, 2, 3}:
            raise ValueError("q expected to be 0, 1, 2 or 3")
        self.q = q

    def forward(self, x1: Tensor, x2: Tensor, diag: bool = False, **params) -> Tensor:
        x1_ = x1.div(self.lengthscale)
        x2_ = x2.div(self.lengthscale)
        D = x1.shape[-1]
        j = math.floor(D / 2.0) + self.q + 1
        if diag:
            r = self.covar_dist(x1_, x2_, diag=True)
        else:
            r = self.covar_dist(x1_, x2_)
        cov_matrix = _fmax(r, j, self.q) * _get_cov(r, j, self.q)
        return cov_matrix
