"""SPD manifold batch normalization from Kobler et al. (2022)."""

from enum import Enum

import torch
from torch import nn
from torch.nn.utils.parametrize import register_parametrization

from spd.functional import (
    airm_geodesic,
    matrix_log,
    matrix_power,
    matrix_inv_sqrt,
    matrix_sqrt,
    numerical_config,
)
from spd.functional.batchnorm import frechet_mean, spd_centering, spd_rebiasing
from spd.modules.manifold import PositiveDefiniteScalar, SymmetricPositiveDefinite


class BatchNormTestStatsMode(str, Enum):
    BUFFER = "buffer"
    REFIT = "refit"
    ADAPT = "adapt"


class BatchNormDispersion(str, Enum):
    NONE = "mean"
    SCALAR = "scalar"
    VECTOR = "vector"


def _identity(shape, *, device=None, dtype=None):
    return torch.diag_embed(torch.ones(tuple(shape)[:-1], device=device, dtype=dtype))


class SPDBatchNormImpl(nn.Module):
    """General SPD batch-normalization implementation.

    ``running_*`` tracks the statistics used during optimization, whereas
    ``running_*_test`` tracks the statistics used by BUFFER evaluation.
    """

    def __init__(
            self,
            shape,
            batchdim=0,
            affine=True,
            learn_mean=True,
            learn_std=True,
            dispersion=BatchNormDispersion.SCALAR,
            karcher_steps=1,
            eta=1.0,
            eta_test=0.1,
            eps=None,
            test_stats_mode=BatchNormTestStatsMode.BUFFER,
            device=None,
            dtype=None,
    ):
        super().__init__()
        shape = torch.Size(shape)
        if len(shape) < 2 or shape[-1] != shape[-2]:
            raise ValueError("shape must end in two equal SPD matrix dimensions")
        dispersion = BatchNormDispersion(dispersion)
        if dispersion is BatchNormDispersion.VECTOR:
            raise NotImplementedError("vector dispersion is not implemented")

        self.shape = shape
        self.batchdim = int(batchdim)
        self.affine = bool(affine)
        self.learn_mean = bool(learn_mean)
        self.learn_std = bool(learn_std)
        self.dispersion = dispersion
        self.karcher_steps = int(karcher_steps)
        self.eta = float(eta)
        self.eta_test = float(eta_test)
        self.eps = eps if eps is not None else numerical_config.batchnorm_var_eps
        self.test_stats_mode = BatchNormTestStatsMode(test_stats_mode)

        identity = _identity(shape, device=device, dtype=dtype)
        variance = torch.ones((*shape[:-2], 1), device=device, dtype=dtype)
        self.register_buffer("running_mean", identity.clone())
        self.register_buffer("running_mean_test", identity.clone())
        self.register_buffer("running_var", variance.clone())
        self.register_buffer("running_var_test", variance.clone())

        if self.affine:
            self.mean = nn.Parameter(identity.clone(), requires_grad=learn_mean)
            register_parametrization(self, "mean", SymmetricPositiveDefinite())
            if dispersion is BatchNormDispersion.SCALAR:
                self.std = nn.Parameter(variance.clone(), requires_grad=learn_std)
                register_parametrization(self, "std", PositiveDefiniteScalar())
            else:
                self.std = None
        else:
            self.mean = None
            self.std = None

    def set_test_stats_mode(self, mode):
        self.test_stats_mode = BatchNormTestStatsMode(mode)

    def set_eta(self, eta=None, eta_test=None):
        if eta is not None:
            self.eta = float(eta)
        if eta_test is not None:
            self.eta_test = float(eta_test)

    def reset_running_stats(self):
        with torch.no_grad():
            identity = _identity(self.shape, device=self.running_mean.device, dtype=self.running_mean.dtype)
            self.running_mean.copy_(identity)
            self.running_mean_test.copy_(identity)
            self.running_var.fill_(1.0)
            self.running_var_test.fill_(1.0)

    def _move_batch_first(self, x):
        batchdim = self.batchdim if self.batchdim >= 0 else x.ndim + self.batchdim
        if batchdim >= x.ndim - 2:
            raise ValueError("batchdim cannot be one of the SPD matrix dimensions")
        return x.movedim(batchdim, 0), batchdim

    def _batch_mean(self, x):
        return frechet_mean(x, max_iter=max(1, self.karcher_steps)).squeeze(0)

    @staticmethod
    def _variance_at(x, reference_mean):
        centered = spd_centering(x, matrix_inv_sqrt.apply(reference_mean))
        tangent = matrix_log.apply(centered)
        return tangent.square().sum(dim=(-2, -1)).mean(dim=0).unsqueeze(-1)

    @torch.no_grad()
    def initrunningstats(self, x):
        """Fit both train and test buffers from a complete data domain."""
        xb, _ = self._move_batch_first(x)
        mean = self._batch_mean(xb)
        variance = self._variance_at(xb, mean).clamp_min(self.eps)
        self.running_mean.copy_(mean)
        self.running_mean_test.copy_(mean)
        if self.dispersion is BatchNormDispersion.SCALAR:
            self.running_var.copy_(variance)
            self.running_var_test.copy_(variance)
        return mean, variance

    def _normalize(self, x, mean, variance, target_mean, target_std):
        centered = spd_centering(x, matrix_inv_sqrt.apply(mean))
        if self.dispersion is BatchNormDispersion.SCALAR:
            if target_std is None:
                target_std = 1.0
            # Scalar dispersion is stored as ``(..., 1)``.  Add a leading
            # singleton batch axis; the trailing singleton already broadcasts
            # over the eigenvalue dimension used by ``matrix_power``.
            if torch.is_tensor(target_std):
                target_std = target_std.unsqueeze(0)
            if torch.is_tensor(variance):
                variance = variance.unsqueeze(0)
            exponent = target_std / (variance + self.eps).sqrt()
            # centered = matrix_exp.apply(exponent * matrix_log.apply(centered))
            centered = matrix_power.apply(centered, exponent)
        if target_mean is not None:
            centered = spd_rebiasing(centered, matrix_sqrt.apply(target_mean))
        return centered

    def forward(self, x, *, target_mean=None, target_std=None):
        xb, batchdim = self._move_batch_first(x)
        target_mean = self.mean if target_mean is None else target_mean
        target_std = self.std if target_std is None else target_std

        if self.training:
            batch_mean = self._batch_mean(xb)
            train_mean = airm_geodesic(self.running_mean, batch_mean, self.eta)
            test_mean = airm_geodesic(self.running_mean_test, batch_mean, self.eta_test)
            if self.dispersion is BatchNormDispersion.SCALAR:
                batch_var = self._variance_at(xb, train_mean)
                batch_var_test = self._variance_at(xb, test_mean)
                train_var = (1.0 - self.eta) * self.running_var + self.eta * batch_var
                test_var = (1.0 - self.eta_test) * self.running_var_test + self.eta_test * batch_var_test
            else:
                train_var = test_var = None

            with torch.no_grad():
                self.running_mean.copy_(train_mean.detach())
                self.running_mean_test.copy_(test_mean.detach())
                if self.dispersion is BatchNormDispersion.SCALAR:
                    self.running_var.copy_(train_var.detach())
                    self.running_var_test.copy_(test_var.detach())
            norm_mean, norm_var = train_mean, train_var
        else:
            if self.test_stats_mode is BatchNormTestStatsMode.REFIT:
                self.initrunningstats(x)
            elif self.test_stats_mode is BatchNormTestStatsMode.ADAPT:
                raise NotImplementedError("ADAPT mode is not implemented")
            norm_mean = self.running_mean_test
            norm_var = self.running_var_test if self.dispersion is BatchNormDispersion.SCALAR else None

        normalized = self._normalize(xb, norm_mean, norm_var, target_mean, target_std)
        return normalized.movedim(0, batchdim)


class SPDMBN(SPDBatchNormImpl):
    pass


__all__ = [
    "BatchNormTestStatsMode",
    "BatchNormDispersion",
    "SPDBatchNormImpl",
    "SPDMBN",
]
