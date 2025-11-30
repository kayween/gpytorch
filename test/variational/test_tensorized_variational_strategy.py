#!/usr/bin/env python3

import unittest

import torch

import gpytorch
from gpytorch.models.approximate_gp import ApproximateGP
from gpytorch.test.base_test_case import BaseTestCase
from gpytorch.variational.cholesky_variational_distribution import CholeskyVariationalDistribution
from gpytorch.variational.tensorized_variational_strategy import (
    TensorizedVariationalStrategy,
    VariationalStrategyAlgebra,
)
from gpytorch.variational.variational_strategy import VariationalStrategy


class _GPModel(ApproximateGP):
    def __init__(self, inducing_points, variational_strategy_class: type[VariationalStrategy] = VariationalStrategy):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = variational_strategy_class(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.RBFKernel()

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class TestVariationalStrategyAlgebra(unittest.TestCase, BaseTestCase):
    def test_forward_backward(self):
        torch.set_default_dtype(torch.float32)
        n = 3
        m = 2

        chol = torch.rand(m, m).tril_()
        covar_data_induc = torch.rand(n, m)

        middle = torch.rand(m, m)
        middle = middle + middle.mT

        induc_mean = torch.randn(m)

        chol.requires_grad_(True)
        covar_data_induc.requires_grad_(True)
        middle.requires_grad_(True)
        induc_mean.requires_grad_(True)

        # Custom autograd function
        predictive_mean, predictive_covar_diag = VariationalStrategyAlgebra.apply(
            chol,
            covar_data_induc,
            middle,
            induc_mean,
        )

        loss = predictive_mean.sum() + predictive_covar_diag.sum() * 0.0
        loss.backward()

        # Ground truth derivatives
        chol_ref = chol.detach().clone().requires_grad_(True)
        covar_data_induc_ref = covar_data_induc.detach().clone().requires_grad_(True)
        middle_ref = middle.detach().clone().requires_grad_(True)
        induc_mean_ref = induc_mean.detach().clone().requires_grad_(True)

        interp_term_ref = torch.linalg.solve_triangular(chol_ref, covar_data_induc_ref.mT, upper=False)
        predictive_mean_ref = (interp_term_ref.transpose(-1, -2) @ induc_mean_ref.unsqueeze(-1)).squeeze(-1)
        predictive_covar_diag_ref = torch.sum(
            (interp_term_ref.transpose(-1, -2) @ middle_ref) * interp_term_ref.transpose(-1, -2),
            dim=-1,
        )

        loss_ref = predictive_mean_ref.sum() + predictive_covar_diag_ref.sum() * 0.0
        loss_ref.backward()

        # Assert that the forward outputs are the same
        self.assertAllClose(predictive_mean, predictive_mean_ref)
        self.assertAllClose(predictive_covar_diag, predictive_covar_diag_ref)

        # Now assert that the derivatives are the same
        self.assertAllClose(chol.grad, chol_ref.grad)
        self.assertAllClose(covar_data_induc.grad, covar_data_induc_ref.grad)
        self.assertAllClose(middle.grad, middle_ref.grad)
        self.assertAllClose(induc_mean.grad, induc_mean_ref.grad)


class TestTensorizedVariationalStrategy(unittest.TestCase, BaseTestCase):
    def test_train_mode(self):
        torch.set_default_dtype(torch.float32)

        inducing_points = torch.rand(2, 2)
        train_x = torch.rand(5, 2)

        torch.manual_seed(42)
        model1 = _GPModel(
            inducing_points=inducing_points.clone(),
            variational_strategy_class=TensorizedVariationalStrategy,
        )
        model1.train()
        output1 = model1(train_x)

        loss1 = output1.mean.mean() + output1.covariance_matrix.diag().mean()
        loss1.backward()

        torch.manual_seed(42)
        model2 = _GPModel(
            inducing_points=inducing_points.clone(),
            variational_strategy_class=VariationalStrategy,
        )
        model2.train()
        output2 = model2(train_x)

        loss2 = output2.mean.mean() + output2.covariance_matrix.diag().mean()
        loss2.backward()

        self.assertAllClose(output1.mean, output2.mean)
        self.assertAllClose(output1.covariance_matrix.diag(), output2.covariance_matrix.diag())

        self.assertAllClose(
            model1.mean_module.constant.grad,
            model2.mean_module.constant.grad,
        )
        self.assertAllClose(
            model1.covar_module.raw_lengthscale.grad,
            model2.covar_module.raw_lengthscale.grad,
        )
        self.assertAllClose(
            model1.variational_strategy.inducing_points.grad,
            model2.variational_strategy.inducing_points.grad,
            atol=1e-5,
        )
        self.assertAllClose(
            model1.variational_strategy._variational_distribution.variational_mean.grad,
            model2.variational_strategy._variational_distribution.variational_mean.grad,
            atol=1e-4,
            rtol=1e-4,
        )
        self.assertAllClose(
            model1.variational_strategy._variational_distribution.chol_variational_covar.grad,
            model2.variational_strategy._variational_distribution.chol_variational_covar.grad,
            atol=1e-4,
            rtol=1e-4,
        )


if __name__ == "__main__":
    unittest.main()
