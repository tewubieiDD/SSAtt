import math
from typing import Optional, Literal

import torch
from functorch.einops import rearrange
from torch import nn
from torch.nn.utils import parametrizations

from spd.functional import init_param as spd_init

from spd.functional import (
    get_epsilon,
    bimap_transform,
    bimap_increase_dim,
    clamp_eigvals_func,
    clamp_eigvals,
    matrix_log_func,
    sym_to_upper,
    matrix_log,
    matrix_exp_func,
    vec_to_sym,
    matrix_exp,
)


class BiMap(nn.Module):
    r"""Bilinear Mapping Layer for SPD Matrices."""

    weight: nn.Parameter  # Type annotation for registered parameter

    def __init__(
            self,
            in_features: int,
            out_features: int,
            depthwise: int = 1,
            parametrized: bool = True,
            orthogonal_map: Optional[str] = None,
            init_method: Literal["kaiming_uniform", "orthogonal", "stiefel"] = "orthogonal",
            seed: Optional[int] = None,
            device=None,
            dtype=None,
    ):
        super().__init__()

        if init_method not in ["kaiming_uniform", "orthogonal", "stiefel"]:
            raise ValueError(
                f"Unknown init_method: '{init_method}'. Choose from "
                "'kaiming_uniform', 'orthogonal', 'stiefel'."
            )

        if not parametrized and orthogonal_map is not None:
            raise ValueError("orthogonal_map is only used when parametrized is True")

        self._in_features = in_features
        self._out_features = out_features
        self._depthwise = depthwise
        self.parametrized = parametrized
        self.increase_dim = None
        self.orthogonal_map = orthogonal_map
        self.init_method = init_method
        self.seed = seed

        if out_features > in_features:
            self.increase_dim = BiMapIncreaseDim(in_features, out_features, device=device, dtype=dtype)
            self._in_features = out_features

        self.register_parameter(
            "weight",
            nn.Parameter(
                torch.empty([self._depthwise, self._in_features, self._out_features], device=device, dtype=dtype, ),
                requires_grad=True,
            ),
        )

        self.reset_parameters()

        if self.parametrized:
            parametrizations.orthogonal(module=self, name="weight", orthogonal_map=self.orthogonal_map)

    @torch.no_grad()
    def reset_parameters(self) -> None:
        """Initialize weight matrix according to the specified method."""
        if self.init_method == "kaiming_uniform":
            nn.init.kaiming_uniform_(self.weight, a=0.01)
        elif self.init_method == "orthogonal":
            nn.init.orthogonal_(self.weight)
        elif self.init_method == "stiefel":
            spd_init.stiefel_(self.weight, seed=self.seed)
        else:
            raise ValueError(f"Internal error: Invalid init_method '{self.init_method}'")

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.increase_dim:
            X = self.increase_dim(X)
        return bimap_transform(X, self.weight)


class BiMapIncreaseDim(nn.Module):
    r"""Bilinear Mapping Layer for SPD Matrix Dimensionality Expansion."""

    projection_matrix: torch.Tensor  # Type annotation for registered buffer
    add: torch.Tensor  # Type annotation for registered buffer

    def __init__(
            self,
            in_features: int,
            out_features: int,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None,
    ) -> None:
        super(BiMapIncreaseDim, self).__init__()

        if out_features < in_features:
            raise ValueError("Output features must be >= input features")

        self.register_buffer("projection_matrix", torch.eye(out_features, in_features, device=device, dtype=dtype), )
        self.register_buffer("add",
                             torch.diag((torch.arange(out_features, device=device) >= in_features)).to(dtype=dtype), )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # orig_ndim = input.ndim

        # if orig_ndim == 3:
        #     input = input.unsqueeze(1)

        # projection_matrix = self.projection_matrix.view(1, 1, *self.projection_matrix.shape).to(input.dtype)
        # padding_matrix = self.add.view(1, 1, *self.add.shape).to(input.dtype)
        projection_matrix = self.projection_matrix.to(input.dtype)
        padding_matrix = self.add.to(input.dtype)

        output = bimap_increase_dim(input, projection_matrix, padding_matrix)

        # if orig_ndim == 3:
        #     output = output.squeeze(1)

        return output


class ReEig(nn.Module):
    threshold_: torch.Tensor  # Type annotation for registered buffer

    def __init__(self, threshold=None, autograd=False, device=None, dtype=None):
        super().__init__()
        self._use_dynamic_threshold = threshold is None
        if threshold is None:
            # Will be computed dynamically based on input dtype
            self.register_buffer("threshold_", torch.tensor(0.0, device=device, dtype=dtype))
        else:
            self.register_buffer("threshold_", torch.tensor(threshold, device=device, dtype=dtype))
        self.autograd_ = autograd

    def _get_threshold(self, X: torch.Tensor) -> torch.Tensor:
        if self._use_dynamic_threshold:
            threshold_value = get_epsilon(X.dtype, "eigval_clamp")
            return torch.tensor(threshold_value, device=X.device, dtype=X.dtype)
        return self.threshold_.to(device=X.device, dtype=X.dtype)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        threshold = self._get_threshold(X)
        if self.autograd_:
            return clamp_eigvals_func(X, threshold)
        return clamp_eigvals.apply(X, threshold)


class LogEig(nn.Module):
    r"""Logarithmic Eigenvalue Layer (LogEig)."""

    autograd_: torch.Tensor  # Type annotation for registered buffer

    def __init__(self, upper=True, flatten=True, autograd=False, device=None, dtype=None):
        super().__init__()
        self.upper = upper
        self.flatten = flatten
        self.dtype = dtype
        self.device = device

        self.register_buffer("autograd_", torch.tensor(autograd, device=device, dtype=dtype))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.autograd_:
            X_log = matrix_log_func(X)
        else:
            X_log = matrix_log.apply(X)

        if self.upper:
            X_log = sym_to_upper(X_log)
        elif self.flatten:
            X_log = X_log.flatten(start_dim=-2)

        return X_log.to(device=self.device)


class ExpEig(nn.Module):
    r"""Exponential Eigenvalue Layer (ExpEig)."""

    autograd_: torch.Tensor  # Type annotation for registered buffer

    def __init__(self, upper=False, flatten=False, autograd=False, device=None, dtype=None):
        super().__init__()
        self.upper = upper
        self.flatten = flatten
        self.dtype = dtype
        self.device = device

        self.register_buffer("autograd_", torch.tensor(autograd, device=device, dtype=dtype))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.upper:
            X = vec_to_sym(X)
        elif self.flatten:
            n = math.isqrt(X.shape[-1])
            if n * n != X.shape[-1]:
                raise ValueError(f"ExpEig(flatten=True) expects last dimension n*n, got {X.shape[-1]}.")
            X = rearrange(X, "... (n1 n2) -> ... n1 n2", n1=n, n2=n)
        elif X.ndim < 2 or X.shape[-1] != X.shape[-2]:
            raise ValueError("ExpEig expects matrix input with shape (..., n, n) when upper=False and flatten=False.")

        if self.autograd_:
            X_exp = matrix_exp_func(X)
        else:
            X_exp = matrix_exp.apply(X)

        return X_exp.to(device=self.device)
