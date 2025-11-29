#!/usr/bin/env python3

from typing import Optional

import torch

from linear_operator import to_dense

from linear_operator.operators import DiagLinearOperator, LinearOperator, SumLinearOperator
from linear_operator.utils.cholesky import psd_safe_cholesky
from torch import Tensor

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

        L = psd_safe_cholesky(to_dense(induc_induc_covar))
        if L.shape != induc_induc_covar.shape:
            raise NotImplementedError()

        predictive_mean, predictive_covar_diag = VariationalStrategyAlgebra.apply(
            L,
            induc_data_covar,
            to_dense(variational_inducing_covar) - to_dense(self.variational_distribution.covariance_matrix),
            inducing_values,
        )

        predictive_mean = predictive_mean + test_mean

        return MultivariateNormal(
            predictive_mean,
            SumLinearOperator(
                data_data_covar.add_jitter(self.jitter_val),
                DiagLinearOperator(predictive_covar_diag),
            ),
        )
