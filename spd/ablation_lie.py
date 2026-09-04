import torch
from torch import nn

from spd.base_module_re import (
    NewtonHyperNet,
    E2R,
    AttentionManifold,
)
from spd.modules import ReEig


class LieOrthogonalization(nn.Module):
    def forward(self, W):
        bs, k, n = W.shape
        A = W.new_zeros(bs, n, n)
        A[:, :k, :] = W
        A = A - A.mT
        Q = torch.matrix_exp(A)
        return Q[:, :k, :]


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
        self.hyper_net = NewtonHyperNet(
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
        self.orthogonalization = LieOrthogonalization()
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
