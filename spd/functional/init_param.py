import warnings
from typing import Optional

import torch

from .autograd import matrix_inv_sqrt


def orthogonal_polar_factor(W: torch.Tensor) -> torch.Tensor:
    r"""Compute the orthogonal polar factor of a matrix."""
    return W @ matrix_inv_sqrt.apply(W.mT @ W)


@torch.no_grad()
def stiefel_(tensor: torch.Tensor, seed: Optional[int] = None) -> torch.Tensor:
    r"""Initialize tensor on the Stiefel manifold (in-place)."""
    if seed is None:
        warnings.warn(
            f"No seed provided for Stiefel initialization. "
            f"Using default seed 0 for reproducibility.",
            UserWarning,
        )
        seed = 0

    generator = torch.Generator(device=tensor.device).manual_seed(seed)
    _W = torch.randn(*tensor.shape, dtype=tensor.dtype, device=tensor.device, generator=generator)
    try:
        stiefel_W = orthogonal_polar_factor(_W)
    except torch.linalg.LinAlgError as e:
        warnings.warn(
            f"Stiefel initialization via eigh failed ({e}). "
            f"Falling back to QR decomposition for orthogonalization.",
            UserWarning,
        )
        Q_list = []
        if _W.ndim > 2:
            for i in range(_W.shape[0]):
                q, _ = torch.linalg.qr(_W[i])
                Q_list.append(q[..., : _W.shape[-1]])
            stiefel_W = torch.stack(Q_list, dim=0)
        else:
            q, _ = torch.linalg.qr(_W)
            stiefel_W = q[..., : _W.shape[-1]]

    tensor.copy_(stiefel_W)
    return tensor


@torch.no_grad()
def spd_identity_(tensor: torch.Tensor) -> torch.Tensor:
    """Initialize tensor as identity matrix (in-place)."""
    if tensor.shape[-1] != tensor.shape[-2]:
        raise ValueError(f"spd_identity_ requires square matrices, got shape {tensor.shape}")

    tensor.zero_()
    n = tensor.shape[-1]
    # Fill diagonal with ones
    tensor[..., range(n), range(n)] = 1.0
    return tensor
