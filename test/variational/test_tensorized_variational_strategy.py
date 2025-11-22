#!/usr/bin/env python3

import unittest

import torch

import gpytorch
from gpytorch.models.approximate_gp import ApproximateGP
from gpytorch.variational.cholesky_variational_distribution import CholeskyVariationalDistribution
from gpytorch.variational.tensorized_variational_strategy import TensorizedVariationalStrategy
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


class TestTensorizedVariationalStrategy(unittest.TestCase):
    def test_train_mode(self):
        torch.set_default_dtype(torch.float32)

        inducing_points = torch.randn(2, 2)
        train_x = torch.randn(5, 2)

        torch.manual_seed(42)
        model1 = _GPModel(
            inducing_points=inducing_points.clone(),
            variational_strategy_class=TensorizedVariationalStrategy,
        )
        model1.train()
        output1 = model1(train_x)

        torch.manual_seed(42)
        model2 = _GPModel(
            inducing_points=inducing_points.clone(),
            variational_strategy_class=VariationalStrategy,
        )
        model2.train()
        output2 = model2(train_x)

        self.assertTrue(torch.allclose(output1.mean, output2.mean))
        self.assertTrue(torch.allclose(output1.covariance_matrix, output2.covariance_matrix))


if __name__ == "__main__":
    unittest.main()
