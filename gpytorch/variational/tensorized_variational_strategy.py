#!/usr/bin/env python3

from typing import Optional

import torch

from linear_operator import to_dense

from linear_operator.operators import CholLinearOperator, LinearOperator, MatmulLinearOperator, SumLinearOperator
from linear_operator.utils.cholesky import psd_safe_cholesky
from torch import Tensor

from gpytorch.settings import trace_mode

from ..distributions import MultivariateNormal
from .variational_strategy import VariationalStrategy


class VariationalStrategyAlgebra(torch.autograd.Function):
    def forward(
        ctx,
        chol: Tensor,
        covar_data_induc: Tensor,
        middle: Tensor,
        induc_mean: Tensor,
    ) -> tuple[Tensor, Tensor]:
        r"""
        Returns:
            A tuple fo tensors. The first tensor is the predictive mean, and the second tensor is the diagonal of
            predictive covariance.
        """
        interp_term = torch.linalg.solve_triangular(chol, covar_data_induc.mT, upper=False)
        predictive_mean = (interp_term.transpose(-1, -2) @ induc_mean.unsqueeze(-1)).squeeze(-1)
        predictive_covar_diag = torch.sum(
            (interp_term.transpose(-1, -2) @ middle) * interp_term.transpose(-1, -2),
            dim=-1,
        )

        # NOTE: No need to save `covar_data_induc`. Access to it is always through `interp_term`.
        ctx.save_for_backward(chol, middle, induc_mean, interp_term)
        return predictive_mean, predictive_covar_diag

    def backward(
        ctx,
        d_predictive_mean: Tensor,
        d_predictive_variance: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        r"""
        We re-arrange the matmuls in the derivatives to make the computation more efficient.
        """
        chol, middle, induc_mean, interp_term = ctx.saved_tensors

        # The derivative of `K_XZ K_ZZ^{-1/2} m` with respect to `m`
        d_induc_mean = interp_term @ d_predictive_mean.unsqueeze(-1)
        d_induc_mean = d_induc_mean.squeeze(-1)

        # This term `(S - I) @ K_ZZ^{-1/2}` will be used multiple times
        middle_times_inv_chol = torch.linalg.solve_triangular(chol, middle, upper=True, left=False)

        d_middle = interp_term @ (d_predictive_variance.unsqueeze(-1) * interp_term.transpose(-1, -2))

        # The derivative of `K_XZ` received from the predictive covariance
        d_covar_data_induc = 2.0 * d_predictive_variance.unsqueeze(-1) * interp_term.mT @ middle_times_inv_chol.mT

        # And then add the derivative of `K_XZ` received from the predictive mean
        inv_chol_times_induc_mean = torch.linalg.solve_triangular(chol, induc_mean.unsqueeze(-1), upper=False)
        d_chol = d_covar_data_induc + d_predictive_mean.unsqueeze(-1) @ inv_chol_times_induc_mean.mT
        # This is the hardest one.
        d_chol = -2.0 * middle_times_inv_chol.mT @ interp_term @ (d_predictive_variance.unsqueeze(-1) * interp_term.mT)
        d_chol = d_chol.tril()

        return d_chol, d_covar_data_induc, d_middle, d_induc_mean


class TensorizedVariationalStrategy(VariationalStrategy):
    r"""
    An ultra-light-weight variational strategy that perform linear algebra operations on torch tensors.
    """

    def forward(
        self,
        x: Tensor,
        inducing_points: Tensor,
        inducing_values: Tensor,
        variational_inducing_covar: Optional[LinearOperator] = None,
        **kwargs,
    ) -> MultivariateNormal:
        # Compute full prior distribution
        full_inputs = torch.cat([inducing_points, x], dim=-2)
        full_output = self.model.forward(full_inputs, **kwargs)
        full_covar = full_output.lazy_covariance_matrix

        # Covariance terms
        num_induc = inducing_points.size(-2)
        test_mean = full_output.mean[..., num_induc:]
        induc_induc_covar = full_covar[..., :num_induc, :num_induc].add_jitter(self.jitter_val)
        induc_data_covar = full_covar[..., :num_induc, num_induc:].to_dense()
        data_data_covar = full_covar[..., num_induc:, num_induc:]

        # Compute interpolation terms
        # K_ZZ^{-1/2} K_ZX
        # K_ZZ^{-1/2} \mu_Z
        L = psd_safe_cholesky(to_dense(induc_induc_covar))
        if L.shape != induc_induc_covar.shape:
            raise NotImplementedError()

        interp_term = torch.linalg.solve_triangular(
            L.to(full_inputs.dtype),
            # induc_data_covar.type(_linalg_dtype_cholesky.value()),
            induc_data_covar,
            upper=False,
        ).to(full_inputs.dtype)

        # Compute the mean of q(f)
        # k_XZ K_ZZ^{-1/2} (m - K_ZZ^{-1/2} \mu_Z) + \mu_X
        predictive_mean = (interp_term.transpose(-1, -2) @ inducing_values.unsqueeze(-1)).squeeze(-1) + test_mean

        # Compute the covariance of q(f)
        # K_XX + k_XZ K_ZZ^{-1/2} (S - I) K_ZZ^{-1/2} k_ZX
        assert isinstance(variational_inducing_covar, CholLinearOperator)
        root_times_interp = variational_inducing_covar.root.transpose(-1, -2) @ interp_term

        if trace_mode.on():
            raise NotImplementedError
        else:
            predictive_covar = SumLinearOperator(
                data_data_covar.add_jitter(self.jitter_val),
                MatmulLinearOperator(interp_term.transpose(-1, -2), interp_term).mul(-1),
                MatmulLinearOperator(root_times_interp.transpose(-1, -2), root_times_interp),
            )

        # Return the distribution
        return MultivariateNormal(predictive_mean, predictive_covar)
