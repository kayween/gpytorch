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
