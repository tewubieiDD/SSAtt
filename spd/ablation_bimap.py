from torch import nn

from spd.modules import BiMap, ReEig
from spd.base_module_re import (
    E2R,
    AttentionManifold,
)


class Submanifold(nn.Module):
    def __init__(
            self,
            n,
            k_dims,
            hidden_dim=128,
            context_dim=64,
            dropout=0.0,
            num_iterations=8,
            base_init="branch",
            use_tanh=False,
            use_scale=False,
            dynamic_scale=0.1,
    ):
        super().__init__()
        self.n = n
        self.k_dims = k_dims if isinstance(k_dims, list) else [k_dims]
        self.maps = nn.ModuleList([BiMap(n, k) for k in self.k_dims])
        self.re = ReEig(threshold=1e-4)

    def forward(self, X):
        if X.dim() == 4:
            X = X.squeeze(1)
        return [self.re(bimap(X)).unsqueeze(1) for bimap in self.maps]
