import torch
from torch import nn

from spd.base_module_re import (
    NewtonSchulzOrthogonalization,
    E2R,
    AttentionManifold,
)
from spd.modules import ReEig


class BaseOnlyHyperNet(nn.Module):
    def __init__(
            self,
            n,
            k_dims,
            hidden_dim=128,
            context_dim=64,
            dropout=0.0,
            base_init="branch",
            use_tanh=False,
            use_scale=False,
            dynamic_scale=0.1,
    ):
        super().__init__()
        self.n = n
        self.k_dims = k_dims if isinstance(k_dims, list) else [k_dims]
        self.total_k = sum(self.k_dims)
        self.base_init = self._normalize_base_init(base_init)
        self.base_w = nn.ParameterList([
            nn.Parameter(torch.empty(k, n)) for k in self.k_dims
        ])
        self.reset_parameters()

    @staticmethod
    def _normalize_base_init(base_init):
        aliases = {
            "branch": "branch",
            "branch_orthogonal": "branch",
            "per_branch": "branch",
            "global": "global",
            "global_orthogonal": "global",
            "large": "global",
        }
        if base_init not in aliases:
            raise ValueError(f"Unknown base_init: {base_init}")
        return aliases[base_init]

    def reset_parameters(self):
        with torch.no_grad():
            if self.base_init == "branch":
                for base in self.base_w:
                    nn.init.orthogonal_(base)
            elif self.base_init == "global":
                global_frame = torch.empty(self.total_k, self.n)
                nn.init.orthogonal_(global_frame)
                start = 0
                for base, k in zip(self.base_w, self.k_dims):
                    end = start + k
                    base.copy_(global_frame[start:end])
                    start = end

    def forward(self, X):
        bs = X.shape[0]
        return [base.unsqueeze(0).expand(bs, -1, -1) for base in self.base_w]


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
        self.hyper_net = BaseOnlyHyperNet(
            n,
            self.k_dims,
            hidden_dim=hidden_dim,
            context_dim=context_dim,
            dropout=dropout,
            base_init=base_init,
            use_tanh=use_tanh,
            use_scale=use_scale,
            dynamic_scale=dynamic_scale,
        )
        self.orthogonalization = NewtonSchulzOrthogonalization(num_iterations=num_iterations)
        self.re = ReEig(threshold=1e-4)

    def forward(self, X):
        if X.dim() == 4:
            X = X.squeeze(1)

        W_raw_list = self.hyper_net(X)

        sub_manifolds = []
        for W_raw in W_raw_list:
            W = self.orthogonalization(W_raw)
            X_i = W @ X @ W.mT
            X_i = self.re(X_i)
            sub_manifolds.append(X_i.unsqueeze(1))

        return sub_manifolds
