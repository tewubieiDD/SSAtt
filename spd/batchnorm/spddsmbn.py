"""Domain-specific SPD manifold batch normalization."""

import torch
from torch import nn
from torch.nn.utils.parametrize import register_parametrization

from spd.modules.manifold import PositiveDefiniteScalar, SymmetricPositiveDefinite

from .spdmbn import (
    BatchNormDispersion,
    BatchNormTestStatsMode,
    SPDMBN,
    _identity,
)


class DomainSPDBatchNormImpl(nn.Module):
    """One SPDMBN per domain, optionally sharing their affine parameters."""

    domain_bn_cls = SPDMBN

    def __init__(
            self,
            shape,
            batchdim=0,
            affine=True,
            learn_mean=True,
            learn_std=True,
            dispersion=BatchNormDispersion.SCALAR,
            domains=(),
            karcher_steps=1,
            eta=1.0,
            eta_test=0.1,
            eps=None,
            test_stats_mode=BatchNormTestStatsMode.BUFFER,
            device=None,
            dtype=None,
    ):
        super().__init__()
        self.shape = torch.Size(shape)
        if self.shape[-1] != self.shape[-2]:
            raise ValueError("shape must end in two equal SPD matrix dimensions")
        self.batchdim = int(batchdim)
        self.affine = bool(affine)
        self.learn_mean = bool(learn_mean)
        self.learn_std = bool(learn_std)
        self.dispersion = BatchNormDispersion(dispersion)
        self._domain_kwargs = dict(
            shape=self.shape,
            batchdim=self.batchdim,
            affine=not self.affine,
            learn_mean=self.learn_mean,
            learn_std=self.learn_std,
            dispersion=self.dispersion,
            karcher_steps=karcher_steps,
            eta=eta,
            eta_test=eta_test,
            eps=eps,
            device=device,
            dtype=dtype,
        )
        self.test_stats_mode = BatchNormTestStatsMode(test_stats_mode)

        if self.affine:
            identity = _identity(self.shape, device=device, dtype=dtype)
            self.mean = nn.Parameter(identity, requires_grad=learn_mean)
            register_parametrization(self, "mean", SymmetricPositiveDefinite())
            if self.dispersion is BatchNormDispersion.SCALAR:
                variance = torch.ones((*self.shape[:-2], 1), device=device, dtype=dtype)
                self.std = nn.Parameter(variance, requires_grad=learn_std)
                register_parametrization(self, "std", PositiveDefiniteScalar())
            else:
                self.std = None
        else:
            self.mean = None
            self.std = None

        self.batchnorm = nn.ModuleDict()
        for domain in torch.as_tensor(domains).reshape(-1).tolist():
            self.add_domain_(domain)

    @staticmethod
    def _domain_key(domain):
        value = domain.item() if torch.is_tensor(domain) else domain
        return f"dom {int(value)}"

    def add_domain_(self, domain, *, device=None, dtype=None):
        key = self._domain_key(domain)
        if key not in self.batchnorm:
            kwargs = dict(self._domain_kwargs)
            if self.affine:
                kwargs["device"] = self.mean.device
                kwargs["dtype"] = self.mean.dtype
            else:
                kwargs["device"] = device if device is not None else kwargs["device"]
                kwargs["dtype"] = dtype if dtype is not None else kwargs["dtype"]
            self.batchnorm[key] = self.domain_bn_cls(**kwargs)
            self.batchnorm[key].train(self.training)
            self.batchnorm[key].set_test_stats_mode(self.test_stats_mode)
        return self.batchnorm[key]

    def get_domain_obj(self, domain):
        return self.batchnorm[self._domain_key(domain)]

    def set_test_stats_mode(self, mode):
        self.test_stats_mode = BatchNormTestStatsMode(mode)
        for bn in self.batchnorm.values():
            bn.set_test_stats_mode(self.test_stats_mode)

    def set_eta(self, eta=None, eta_test=None):
        for bn in self.batchnorm.values():
            bn.set_eta(eta=eta, eta_test=eta_test)
        if eta is not None:
            self._domain_kwargs["eta"] = float(eta)
        if eta_test is not None:
            self._domain_kwargs["eta_test"] = float(eta_test)

    @torch.no_grad()
    def initrunningstats(self, x, domain):
        return self.add_domain_(domain, device=x.device, dtype=x.dtype).initrunningstats(x)

    def forward_domain_(self, x, domain):
        """Normalize a single-domain batch whose batch dimension is axis 0."""
        if x.ndim < 3:
            raise ValueError("x must include a batch dimension and SPD matrix dimensions")
        child = self.add_domain_(domain, device=x.device, dtype=x.dtype)
        original_batchdim = child.batchdim
        child.batchdim = 0
        try:
            output = child(
                x,
                target_mean=self.mean if self.affine else None,
                target_std=self.std if self.affine else None,
            )
        finally:
            child.batchdim = original_batchdim
        return output

    def forward(self, x, d):
        if d is None:
            raise ValueError("domain ids d are required")
        d = torch.as_tensor(d, device=x.device).reshape(-1)
        batchdim = self.batchdim if self.batchdim >= 0 else x.ndim + self.batchdim
        if d.numel() != x.shape[batchdim]:
            raise ValueError("d must contain one domain id per batch sample")

        xb = x.movedim(batchdim, 0)
        output = torch.empty_like(xb)
        for domain in torch.unique(d):
            mask = d == domain
            output[mask] = self.forward_domain_(xb[mask], domain)
        return output.movedim(0, batchdim)


class DomainSPDBatchNorm(DomainSPDBatchNormImpl):
    pass


class SPDDSMBN(DomainSPDBatchNorm):
    """Kobler et al. SPD domain-specific manifold batch normalization."""

    @torch.no_grad()
    def domainadapt_finetune(self, x, y=None, d=None, target_domains=None):
        if d is None:
            raise ValueError("domain ids d are required")
        d = torch.as_tensor(d, device=x.device).reshape(-1)
        batchdim = self.batchdim if self.batchdim >= 0 else x.ndim + self.batchdim
        xb = x.movedim(batchdim, 0)
        self.set_test_stats_mode(BatchNormTestStatsMode.REFIT)
        for domain in torch.unique(d):
            child = self.add_domain_(domain, device=xb.device, dtype=xb.dtype)
            original_batchdim = child.batchdim
            child.batchdim = 0
            try:
                child.eval()
                child(
                    xb[d == domain],
                    target_mean=self.mean if self.affine else None,
                    target_std=self.std if self.affine else None,
                )
            finally:
                child.batchdim = original_batchdim
        self.set_test_stats_mode(BatchNormTestStatsMode.BUFFER)


__all__ = ["DomainSPDBatchNormImpl", "DomainSPDBatchNorm", "SPDDSMBN"]
