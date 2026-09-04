from dataclasses import dataclass, field
from typing import Optional, Literal, Dict

import torch

# Type alias for threshold names
ThresholdName = Literal[
    "eigval_clamp",
    "eigval_log",
    "eigval_sqrt",
    "eigval_inv_sqrt",
    "eigval_power",
    "loewner_equal",
    "batchnorm_var",
    "dropout",
    "trace_norm",
    "stiefel_init",
    "division_safe",
]


@dataclass
class NumericalConfig:
    r"""Global configuration for numerical stability thresholds.

    Notes
    -----
    The default scale factors are chosen to balance numerical stability with
    accuracy :cite:p:`higham2002accuracy`. More conservative (larger) values
    provide better stability but may reduce precision. Less conservative
    (smaller) values preserve more information but risk numerical issues.

    For mixed-precision training (fp16), consider using larger scale factors
    as the machine epsilon for fp16 is much larger (~9.77e-4).
    """

    # Scale factors (multiplied by machine epsilon)
    eigval_clamp_scale: float = 1e4
    eigval_log_scale: float = 1e2
    eigval_sqrt_scale: float = 1e2
    eigval_inv_sqrt_scale: float = 1e3
    eigval_power_scale: float = 1e3
    loewner_equal_scale: float = 1e2
    stiefel_init_scale: float = 1e3
    division_safe_scale: float = 1e5

    # Absolute epsilons (not scaled by machine epsilon)
    batchnorm_var_eps: float = 1e-5
    dropout_eps: float = 1e-5
    trace_norm_eps: float = 1e-6

    # Behavior flags
    warn_on_clamp: bool = True
    strict_spd_check: bool = False

    # Cache for computed thresholds per dtype
    _threshold_cache: Dict[tuple, float] = field(default_factory=dict, repr=False)

    def clear_cache(self) -> None:
        """Clear the threshold cache after configuration changes."""
        self._threshold_cache.clear()

    def summary(self, dtype: torch.dtype = torch.float32) -> str:
        """Return formatted string showing all thresholds for a given dtype."""
        # Import get_epsilon locally to avoid circular import issues
        # (get_epsilon is defined later in this module)
        lines = [f"Numerical Configuration Summary (dtype={dtype})"]
        lines.append("=" * 50)

        # Machine epsilon info
        machine_eps = torch.finfo(dtype).eps
        lines.append(f"\nMachine epsilon: {machine_eps:.2e}")

        # Scaled thresholds
        scaled_names: list[ThresholdName] = [
            "eigval_clamp",
            "eigval_log",
            "eigval_sqrt",
            "eigval_inv_sqrt",
            "eigval_power",
            "loewner_equal",
            "stiefel_init",
            "division_safe",
        ]
        lines.append("\nScaled thresholds (scale × machine_eps):")
        for name in scaled_names:
            scale = self.get_scale(name)
            eps_value = scale * machine_eps
            lines.append(f"  {name:20s}: {eps_value:.2e} (scale={scale:.0e})")

        # Absolute thresholds
        lines.append("\nAbsolute thresholds (dtype-independent):")
        absolute_names: list[ThresholdName] = [
            "batchnorm_var",
            "dropout",
            "trace_norm",
        ]
        for name in absolute_names:
            eps_value = self.get_scale(name)
            lines.append(f"  {name:20s}: {eps_value:.2e}")

        # Behavior flags
        lines.append("\nBehavior flags:")
        lines.append(f"  warn_on_clamp:      {self.warn_on_clamp}")
        lines.append(f"  strict_spd_check:   {self.strict_spd_check}")

        return "\n".join(lines)

    def get_scale(self, name: ThresholdName) -> float:
        """Get the scale factor for a given threshold name."""
        scale_map = {
            "eigval_clamp": self.eigval_clamp_scale,
            "eigval_log": self.eigval_log_scale,
            "eigval_sqrt": self.eigval_sqrt_scale,
            "eigval_inv_sqrt": self.eigval_inv_sqrt_scale,
            "eigval_power": self.eigval_power_scale,
            "loewner_equal": self.loewner_equal_scale,
            "stiefel_init": self.stiefel_init_scale,
            "division_safe": self.division_safe_scale,
            # Absolute epsilons return themselves (will be handled specially)
            "batchnorm_var": self.batchnorm_var_eps,
            "dropout": self.dropout_eps,
            "trace_norm": self.trace_norm_eps,
        }
        if name not in scale_map:
            raise ValueError(
                f"Unknown threshold name: '{name}'. "
                f"Valid names are: {list(scale_map.keys())}"
            )
        return scale_map[name]

    def is_absolute(self, name: ThresholdName) -> bool:
        """Check if a threshold uses absolute values (not scaled by eps)."""
        return name in ("batchnorm_var", "dropout", "trace_norm")


# Global configuration instance
numerical_config = NumericalConfig()


def get_epsilon(dtype: torch.dtype, name: ThresholdName = "eigval_clamp", *,
                config: Optional[NumericalConfig] = None, ) -> float:
    if config is None:
        config = numerical_config

    # Check cache first
    cache_key = (dtype, name)
    if cache_key in config._threshold_cache:
        return config._threshold_cache[cache_key]

    # Compute threshold
    if config.is_absolute(name):
        # Absolute thresholds don't scale with dtype
        threshold = config.get_scale(name)
    else:
        # Scaled thresholds multiply machine epsilon by scale factor
        machine_eps = torch.finfo(dtype).eps
        scale = config.get_scale(name)
        threshold = scale * machine_eps

    # Cache and return
    config._threshold_cache[cache_key] = threshold
    return threshold


def get_loewner_threshold(
        eigenvalues: torch.Tensor,
        *,
        config: Optional[NumericalConfig] = None,
) -> float:
    """Get threshold for detecting equal eigenvalues in Loewner matrix.

    Notes
    -----
    The threshold is computed as::

        threshold = scale * max(1, |eigenvalues|.max()) * eps

    This adaptive threshold accounts for the magnitude of eigenvalues,
    providing better numerical stability for matrices with large eigenvalues.
    """
    if config is None:
        config = numerical_config

    base_threshold = get_epsilon(eigenvalues.dtype, "loewner_equal", config=config)

    # Adaptive scaling based on eigenvalue magnitude
    max_eigval = eigenvalues.abs().max().item()
    scale = max(1.0, max_eigval)

    return base_threshold * scale
