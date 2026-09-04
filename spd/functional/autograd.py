import warnings
from math import sqrt

import torch
from torch import Tensor
from torch.autograd import Function

from .numerical import get_loewner_threshold, get_epsilon, numerical_config


def ensure_sym(matrix):
    """Ensures that a matrix is symmetric."""
    return (matrix + matrix.mT) / 2


def modeig_forward(X, applied_fct, *args):
    """Forward pass for the modified eigenvalue of a symmetric matrix."""
    s, U = torch.linalg.eigh(X)
    s_modified = applied_fct(s, *args)
    output = U @ torch.diag_embed(s_modified).to(dtype=X.dtype) @ U.transpose(-1, -2)
    return output, s, U, s_modified


def modeig_backward(grad_output, s, U, s_modified, derivative, *args):
    """Backward pass for the modified eigenvalue of a symmetric matrix."""
    # Compute Loewner matrix with adaptive threshold
    denominator = s.unsqueeze(-1) - s.unsqueeze(-1).transpose(-1, -2)

    # Use adaptive threshold that scales with eigenvalue magnitude
    threshold = get_loewner_threshold(s)
    is_eq = denominator.abs() < threshold
    denominator[is_eq] = 1.0

    # Case: sigma_i != sigma_j
    numerator = s_modified.unsqueeze(-1) - s_modified.unsqueeze(-1).transpose(-1, -2)

    # Case: sigma_i == sigma_j (use derivative instead)
    s_derivative = derivative(s, *args)
    numerator[is_eq] = (0.5 * (s_derivative.unsqueeze(-1) + s_derivative.unsqueeze(-1).transpose(-1, -2))[is_eq])
    L = numerator / denominator

    grad_input = (U @ (L * (U.transpose(-1, -2) @ ensure_sym(grad_output) @ U)) @ U.transpose(-1, -2))

    return grad_input


def bimap_transform(X: Tensor, W: Tensor) -> Tensor:
    r"""Apply bilinear transformation to SPD matrices."""
    return W.mT @ X @ W


def bimap_increase_dim(X: Tensor, projection_matrix: Tensor, padding_matrix: Tensor, ) -> Tensor:
    r"""Increase the dimension of SPD matrices via embedding."""
    return padding_matrix + projection_matrix @ X @ projection_matrix.mT


def clamp_eigvals_func(X, threshold):
    """Clamps the eigenvalues of a symmetric matrix."""
    return modeig_forward(X, lambda eigvals: eigvals.clamp(min=threshold))[0]


def clamp_complex(x: torch.Tensor, min_mag: float) -> torch.Tensor:
    """Clamp the magnitude of a complex tensor `x` so that `|x[i]| >= min_mag`."""
    mag = x.abs()
    if isinstance(min_mag, torch.Tensor):
        if min_mag.is_complex():
            min_mag = min_mag.real
        min_mag = min_mag.to(mag.dtype)
    else:
        min_mag = float(min_mag)
    mag_clamped = mag.clamp(min=min_mag)
    safe_mag = mag.clamp(min=1e-12)
    return x * (mag_clamped / safe_mag)


class clamp_eigvals(Function):
    """Rectification of the eigenvalues of a symmetric matrix."""

    @staticmethod
    def applied_fct(s, threshold):
        return s.clamp(min=threshold)

    @staticmethod
    def applied_fct_complex(s, threshold):
        return clamp_complex(s, threshold)

    @staticmethod
    def derivative(s, threshold):
        s_deriv = torch.zeros_like(s)
        s_deriv[s > threshold] = 1
        return s_deriv

    @staticmethod
    def forward(ctx, X, threshold):
        if torch.is_complex(X):
            function_clamp = clamp_eigvals.applied_fct_complex
        else:
            function_clamp = clamp_eigvals.applied_fct

        output, s, U, s_modified = modeig_forward(X, function_clamp, threshold)
        ctx.save_for_backward(s, U, s_modified)
        ctx.threshold = threshold
        return output

    @staticmethod
    def backward(ctx, grad_output):
        s, U, s_modified = ctx.saved_tensors
        threshold = ctx.threshold
        return modeig_backward(grad_output, s, U, s_modified, clamp_eigvals.derivative, threshold), None


def matrix_log_func(X):
    """Computes the matrix logarithm of a symmetric matrix."""
    threshold = get_epsilon(X.dtype, "eigval_log")
    return modeig_forward(X, lambda eigvals: eigvals.clamp(min=threshold).log())[0]


def sym_to_upper(X, preserve_norm=False, upper=True):
    r"""Vectorizes symmetric matrices by extracting triangular elements."""
    assert X.ndim >= 2
    assert X.shape[-1] == X.shape[-2]
    ndim = X.shape[-1]

    if upper:
        ixs = torch.triu_indices(ndim, ndim, offset=0)
    else:
        ixs = torch.tril_indices(ndim, ndim, offset=0)

    x_vec = X[..., ixs[0], ixs[1]]

    if preserve_norm:
        # multiply off-diagonal elements to preserve the norm
        off_diagonal_mask = ixs[0] != ixs[1]
        multipliers = torch.ones_like(x_vec)
        multipliers[..., off_diagonal_mask] = sqrt(2)
        x_vec = x_vec * multipliers

    return x_vec


class matrix_log(Function):
    r"""Matrix logarithm of a symmetric matrix."""

    @staticmethod
    def applied_fct(s):
        threshold = get_epsilon(s.dtype, "eigval_log")
        return s.clamp(min=threshold).log()

    @staticmethod
    def derivative(s):
        threshold = get_epsilon(s.dtype, "eigval_log")
        s_deriv = s.reciprocal()
        # pick subgradient 0 for clamped eigenvalues
        s_deriv[s <= threshold] = 0
        return s_deriv

    @staticmethod
    def forward(ctx, X):
        output, s, U, s_modified = modeig_forward(X, matrix_log.applied_fct)
        threshold = get_epsilon(s.dtype, "eigval_log")
        min_eigenvalue = s.min()
        if numerical_config.warn_on_clamp and threshold > min_eigenvalue:
            warnings.warn(
                f"Eigenvalue clamping occurred in matrix_log: threshold "
                f"({threshold:.2e}) > min eigenvalue ({min_eigenvalue:.2e}). "
                f"This might lead to inaccurate results.",
                UserWarning,
            )
        ctx.save_for_backward(s, U, s_modified)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        s, U, s_modified = ctx.saved_tensors
        return modeig_backward(grad_output, s, U, s_modified, matrix_log.derivative)


# class matrix_log(Function):
#     @staticmethod
#     def applied_fct(s):
#         return s.log()
#
#     @staticmethod
#     def derivative(s):
#         return s.reciprocal()
#
#     @staticmethod
#     def forward(ctx, X):
#         output, s, U, s_modified = modeig_forward(X, matrix_log.applied_fct)
#         ctx.save_for_backward(s, U, s_modified)
#         return output
#
#     @staticmethod
#     def backward(ctx, grad_output):
#         s, U, s_modified = ctx.saved_tensors
#         return modeig_backward(grad_output, s, U, s_modified, matrix_log.derivative)


def matrix_exp_func(X):
    """Computes the matrix exponential of a symmetric matrix."""
    return modeig_forward(X, lambda eigvals: eigvals.exp())[0]


def vec_to_sym(x_vec, preserve_norm=True, upper=True):
    r"""Reconstructs symmetric matrices from vectorization."""
    ndim = (sqrt(1 + 8 * x_vec.shape[-1]) - 1) / 2
    assert ndim == int(ndim)
    ndim = int(ndim)

    if upper:
        ixs = torch.triu_indices(ndim, ndim, offset=0)
    else:
        ixs = torch.tril_indices(ndim, ndim, offset=0)

    od_mask = ixs[0] != ixs[1]

    X = torch.empty((*x_vec.shape[:-1], ndim, ndim), device=x_vec.device, dtype=x_vec.dtype)
    X[..., ixs[0], ixs[1]] = x_vec

    if preserve_norm:
        # divide off-diagonal elements to undo norm-preserving scaling
        X[..., ixs[0, od_mask], ixs[1, od_mask]] /= sqrt(2)

    # Mirror to make symmetric
    X[..., ixs[1, od_mask], ixs[0, od_mask]] = X[..., ixs[0, od_mask], ixs[1, od_mask]]
    return X


class matrix_exp(Function):
    r"""Matrix exponential of a symmetric matrix."""

    @staticmethod
    def applied_fct(s):
        return s.exp()

    @staticmethod
    def derivative(s):
        return s.exp()

    @staticmethod
    def forward(ctx, X):
        output, s, U, s_modified = modeig_forward(X, matrix_exp.applied_fct)
        ctx.save_for_backward(s, U, s_modified)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        s, U, s_modified = ctx.saved_tensors
        return modeig_backward(grad_output, s, U, s_modified, matrix_exp.derivative)


def softplus(s):
    """
    Scaled SoftPlus function.
    It is scaled so that: f(0) = 1, f(x) -> 0 as x -> -inf and
    f'(x) -> 1 as x -> +inf
    """
    return torch.log2(1.0 + torch.pow(torch.tensor(2.0, dtype=s.dtype, device=s.device), s))


def inv_softplus(s):
    """Inverse of the scaled SoftPlus function"""
    return torch.log2(torch.pow(torch.tensor(2.0, device=s.device, dtype=s.dtype), s) - 1.0)


class matrix_softplus(Function):
    r"""
    Matrix (scaled) SoftPlus of a symmetric matrix.
    It is scaled so that: f(0) = 1, f(x) -> 0 as x -> -inf and
    f'(x) -> 1 as x -> +inf.
    """

    @staticmethod
    def applied_fct(s):
        return softplus(s)

    @staticmethod
    def derivative(s):
        return 1 / (1.0 + torch.pow(torch.tensor(2.0, device=s.device, dtype=s.dtype), -s))

    @staticmethod
    def forward(ctx, X):
        output, s, U, s_modified = modeig_forward(X, matrix_softplus.applied_fct)
        ctx.save_for_backward(s, U, s_modified)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        s, U, s_modified = ctx.saved_tensors
        return modeig_backward(grad_output, s, U, s_modified, matrix_softplus.derivative)


class matrix_inv_softplus(Function):
    r"""Matrix inverse (scaled) SoftPlus of a symmetric matrix."""

    @staticmethod
    def applied_fct(s):
        return inv_softplus(s)

    @staticmethod
    def derivative(s):
        return 1 / (1.0 - torch.pow(torch.tensor(2.0, device=s.device, dtype=s.dtype), -s))

    @staticmethod
    def forward(ctx, X):
        output, s, U, s_modified = modeig_forward(X, matrix_inv_softplus.applied_fct)
        ctx.save_for_backward(s, U, s_modified)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        s, U, s_modified = ctx.saved_tensors
        return modeig_backward(grad_output, s, U, s_modified, matrix_inv_softplus.derivative)


class abs_eigvals(Function):
    """Absolute value of the eigenvalues of a symmetric matrix."""

    @staticmethod
    def applied_fct(s):
        return s.abs()

    @staticmethod
    def derivative(s):
        return s.sign()

    @staticmethod
    def forward(ctx, X):
        output, s, U, s_modified = modeig_forward(X, abs_eigvals.applied_fct)
        ctx.save_for_backward(s, U, s_modified)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        s, U, s_modified = ctx.saved_tensors
        return modeig_backward(grad_output, s, U, s_modified, abs_eigvals.derivative)


# class matrix_power(Function):
#     """Computes the matrix power."""
#
#     @staticmethod
#     def applied_fct(s, exponent):
#         threshold = get_epsilon(s.dtype, "eigval_power")
#         return s.clamp(min=threshold).pow(exponent=exponent)
#
#     @staticmethod
#     def derivative(s, exponent):
#         threshold = get_epsilon(s.dtype, "eigval_power")
#         s_clamped = s.clamp(min=threshold)
#         s_deriv = exponent * s_clamped.pow(exponent=exponent - 1.0)
#         # pick subgradient 0 for clamped eigenvalues
#         s_deriv[s <= threshold] = 0
#         return s_deriv
#
#     @staticmethod
#     def forward(ctx, X, exponent):
#         output, s, U, s_modified = modeig_forward(X, matrix_power.applied_fct, exponent)
#         ctx.save_for_backward(s, U, s_modified)
#         ctx.exponent = exponent
#         return output
#
#     @staticmethod
#     def backward(ctx, grad_output):
#         s, U, s_modified = ctx.saved_tensors
#         exponent = ctx.exponent
#         return modeig_backward(grad_output, s, U, s_modified, matrix_power.derivative, exponent), None


# adapt SPDDSMBN
class matrix_power(Function):
    @staticmethod
    def applied_fct(s, exponent):
        threshold = get_epsilon(s.dtype, "eigval_power")
        return s.clamp(min=threshold).pow(exponent=exponent)

    @staticmethod
    def derivative(s, exponent):
        threshold = get_epsilon(s.dtype, "eigval_power")
        s_clamped = s.clamp(min=threshold)
        s_deriv = exponent * s_clamped.pow(exponent=exponent - 1.0)
        s_deriv[s <= threshold] = 0
        return s_deriv

    @staticmethod
    def forward(ctx, X, exponent):
        output, s, U, s_modified = modeig_forward(X, matrix_power.applied_fct, exponent)
        ctx.save_for_backward(s, U, s_modified)
        ctx.exponent = exponent
        return output

    @staticmethod
    def backward(ctx, grad_output):
        s, U, s_modified = ctx.saved_tensors
        exponent = ctx.exponent
        grad_x = modeig_backward(grad_output, s, U, s_modified, matrix_power.derivative, exponent)

        grad_exponent = None
        if torch.is_tensor(exponent) and ctx.needs_input_grad[1]:
            threshold = get_epsilon(s.dtype, "eigval_power")
            s_log = s.clamp(min=threshold).log()
            grad_in_eigenbasis = (U.mT @ ensure_sym(grad_output) @ U).diagonal(dim1=-2, dim2=-1)
            grad_exponent = (grad_in_eigenbasis * s_modified * s_log).sum(dim=-1)
            # ``exponent`` may have singleton dimensions that broadcast over
            # the input batch/eigenvalue dimensions. Reduce those dimensions
            # explicitly before restoring its original shape.
            while grad_exponent.ndim < exponent.ndim:
                grad_exponent = grad_exponent.unsqueeze(-1)
            for dim, size in enumerate(exponent.shape):
                if size == 1 and grad_exponent.shape[dim] != 1:
                    grad_exponent = grad_exponent.sum(dim=dim, keepdim=True)
            grad_exponent = grad_exponent.reshape(exponent.shape)

        return grad_x, grad_exponent


class matrix_sqrt(Function):
    """Matrix square root."""

    @staticmethod
    def applied_fct(s):
        threshold = get_epsilon(s.dtype, "eigval_sqrt")
        return s.clamp(min=threshold).sqrt()

    @staticmethod
    def derivative(s):
        threshold = get_epsilon(s.dtype, "eigval_sqrt")
        sder = s.rsqrt() / 2
        # pick subgradient 0 for clamped eigenvalues
        sder[s <= threshold] = 0
        return sder

    @staticmethod
    def forward(ctx, X):
        output, s, U, s_modified = modeig_forward(X, matrix_sqrt.applied_fct)
        ctx.save_for_backward(s, U, s_modified)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        s, U, s_modified = ctx.saved_tensors
        return modeig_backward(grad_output, s, U, s_modified, matrix_sqrt.derivative)


class matrix_inv_sqrt(Function):
    """Inverse matrix square root."""

    @staticmethod
    def applied_fct(s):
        threshold = get_epsilon(s.dtype, "eigval_inv_sqrt")
        return s.clamp(min=threshold).rsqrt()

    @staticmethod
    def derivative(s):
        threshold = get_epsilon(s.dtype, "eigval_inv_sqrt")
        sder = -0.5 * s.pow(-1.5)
        # pick subgradient 0 for clamped eigenvalues
        sder[s <= threshold] = 0
        return sder

    @staticmethod
    def forward(ctx, X):
        output, s, U, s_modified = modeig_forward(X, matrix_inv_sqrt.applied_fct)
        ctx.save_for_backward(s, U, s_modified)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        s, U, s_modified = ctx.saved_tensors
        return modeig_backward(grad_output, s, U, s_modified, matrix_inv_sqrt.derivative)


class matrix_sqrt_inv(Function):
    """Matrix square root and inverse matrix square root."""

    @staticmethod
    def forward(ctx, X):
        output_sqrt, s, U, s_sqrt = modeig_forward(X, matrix_sqrt.applied_fct)
        s_invsqrt = matrix_inv_sqrt.applied_fct(s)
        output_invsqrt = (U @ torch.diag_embed(s_invsqrt).to(dtype=X.dtype) @ U.transpose(-1, -2))
        ctx.save_for_backward(s, U, s_sqrt, s_invsqrt)
        return output_sqrt, output_invsqrt

    @staticmethod
    def backward(ctx, grad_output_sqrt, grad_output_invsqrt):
        s, U, s_sqrt, s_invsqrt = ctx.saved_tensors
        return (modeig_backward(grad_output_sqrt, s, U, s_sqrt, matrix_sqrt.derivative) +
                modeig_backward(grad_output_invsqrt, s, U, s_invsqrt, matrix_inv_sqrt.derivative))
