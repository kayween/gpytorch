#!/usr/bin/env python3

from typing import Optional

import torch

from linear_operator import to_dense

from linear_operator.operators import LinearOperator, MatmulLinearOperator, SumLinearOperator
from linear_operator.utils.cholesky import psd_safe_cholesky
from torch import Tensor

from gpytorch.settings import _linalg_dtype_cholesky, trace_mode

from ..distributions import MultivariateNormal
from .variational_strategy import VariationalStrategy


class VariationalStrategyLDLDecomposition(VariationalStrategy):
    r"""
    An ultra-light-weight variational strategy that perform linear algebra operations on torch tensors.

    This class does LDL decomposition on `S - I`.
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
        L = psd_safe_cholesky(to_dense(induc_induc_covar).type(_linalg_dtype_cholesky.value()))
        if L.shape != induc_induc_covar.shape:
            raise NotImplementedError()

        middle_term = self.prior_distribution.lazy_covariance_matrix.mul(-1)
        if variational_inducing_covar is not None:
            middle_term = to_dense(variational_inducing_covar) + to_dense(middle_term)

        # Compute the mean of q(f)
        # k_XZ K_ZZ^{-1/2} (m - K_ZZ^{-1/2} \mu_Z) + \mu_X

        # Compute the covariance of q(f)
        # K_XX + k_XZ K_ZZ^{-1/2} (S - I) K_ZZ^{-1/2} k_ZX
        try:
            middle_term = middle_term.type(torch.float64)
            lu, _ = torch.linalg.lu_factor(middle_term, pivot=False)
            D = lu.diagonal(dim1=-2, dim2=-1)

            # TODO: Make it work for batched case
            L_lu = lu.tril(-1) + torch.eye(lu.size(-1), device=lu.device, dtype=lu.dtype)

            left = induc_data_covar.mT.to(torch.float64) @ torch.linalg.solve_triangular(
                L.mT, L_lu.to(_linalg_dtype_cholesky.value()), upper=True
            )
            right = D.unsqueeze(-1) * left.mT

            left = left.to(full_inputs.dtype)
            right = right.to(full_inputs.dtype)

            vec = torch.linalg.solve_triangular(L.mT.type(full_inputs.dtype), inducing_values.unsqueeze(-1), upper=True)
            predictive_mean = (induc_data_covar.mT @ vec).squeeze(-1) + test_mean

        except:
            print("LU decomposition failed; falling back to the original GPyTorch implementation.")

            # `middle_term` has already been converted to FP64 if the try fails
            middle_term = middle_term.type(full_inputs.dtype)

            interp_term = torch.linalg.solve_triangular(
                L, induc_data_covar.to(_linalg_dtype_cholesky.value()), upper=False
            )
            interp_term = interp_term.type(full_inputs.dtype)

            predictive_mean = (interp_term.transpose(-1, -2) @ inducing_values.unsqueeze(-1)).squeeze(-1) + test_mean
            left = interp_term.mT
            right = middle_term @ interp_term

        if trace_mode.on():
            raise NotImplementedError
        else:
            predictive_covar = SumLinearOperator(
                data_data_covar.add_jitter(self.jitter_val),
                MatmulLinearOperator(left, right),
            )

        # Return the distribution
        return MultivariateNormal(predictive_mean, predictive_covar)
