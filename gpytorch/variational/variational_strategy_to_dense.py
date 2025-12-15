#!/usr/bin/env python3

from typing import Optional

import torch
from linear_operator.operators import LinearOperator, MatmulLinearOperator, SumLinearOperator
from torch import Tensor

from ..distributions import MultivariateNormal
from ..settings import _linalg_dtype_cholesky, trace_mode
from ..utils.errors import CachingError
from ..utils.memoize import pop_from_cache_ignore_args

from .variational_strategy import VariationalStrategy


class VariationalStrategyToDense(VariationalStrategy):
    r"""
    An ultra-light-weight variational strategy that perform linear algebra operations on torch tensors.

    This class convert all linear operators to tensors.
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
        L = self._cholesky_factor(induc_induc_covar)
        if L.shape != induc_induc_covar.shape:
            # Aggressive caching can cause nasty shape incompatibilies when evaluating with different batch shapes
            # TODO: Use a hook fo this
            try:
                pop_from_cache_ignore_args(self, "cholesky_factor")
            except CachingError:
                pass
            L = self._cholesky_factor(induc_induc_covar)
        interp_term = L.solve(induc_data_covar.type(_linalg_dtype_cholesky.value())).to(full_inputs.dtype)

        # Compute the mean of q(f)
        # k_XZ K_ZZ^{-1/2} (m - K_ZZ^{-1/2} \mu_Z) + \mu_X
        predictive_mean = (interp_term.transpose(-1, -2) @ inducing_values.unsqueeze(-1)).squeeze(-1) + test_mean

        # Compute the covariance of q(f)
        # K_XX + k_XZ K_ZZ^{-1/2} (S - I) K_ZZ^{-1/2} k_ZX
        middle_term = self.prior_distribution.lazy_covariance_matrix.mul(-1)
        if variational_inducing_covar is not None:
            middle_term = SumLinearOperator(variational_inducing_covar, middle_term)

        middle_term = middle_term.to_dense()

        if trace_mode.on():
            predictive_covar = (
                data_data_covar.add_jitter(self.jitter_val).to_dense()
                + interp_term.transpose(-1, -2) @ middle_term.to_dense() @ interp_term
            )
        else:
            predictive_covar = SumLinearOperator(
                data_data_covar.add_jitter(self.jitter_val),
                MatmulLinearOperator(interp_term.transpose(-1, -2), middle_term @ interp_term),
            )

        # Return the distribution
        return MultivariateNormal(predictive_mean, predictive_covar)
