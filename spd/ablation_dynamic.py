from torch import nn

from spd.base_module_re import (
    NewtonSchulzOrthogonalization,
    E2R,
    AttentionManifold,
)
from spd.modules import LogEig, ReEig


class DynamicOnlyHyperNet(nn.Module):
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
        self.use_tanh = use_tanh
        self.use_scale = use_scale
        self.dynamic_scale = dynamic_scale

        flat_dim = n * (n + 1) // 2
        self.tangent = LogEig()
        self.encoder = nn.Sequential(
            nn.LayerNorm(flat_dim),
            nn.Linear(flat_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, context_dim),
            nn.GELU(),
        )
        self.branch_heads = nn.ModuleList([
            nn.Linear(context_dim, k * n) for k in self.k_dims
        ])
        self.reset_parameters()

    def reset_parameters(self):
        for head in self.branch_heads:
            nn.init.xavier_uniform_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, X):
        bs = X.shape[0]
        context = self.encoder(self.tangent(X))

        W_raw_list = []
        for head, k in zip(self.branch_heads, self.k_dims):
            raw_delta = head(context).view(bs, k, self.n)
            delta = raw_delta.tanh() if self.use_tanh else raw_delta
            delta_for_w = self.dynamic_scale * delta if self.use_scale else delta
            W_raw_list.append(delta_for_w)

        return W_raw_list


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
        self.hyper_net = DynamicOnlyHyperNet(
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
