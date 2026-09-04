from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

from spd.functional import log_euclidean_distance, log_euclidean_mean
from spd.modules import BiMap, LogEig, ReEig


class signal2spd(nn.Module):
    # convert signal epoch to SPD matrix
    def __init__(self):
        super().__init__()
        self.dev = torch.device('cpu')

    def forward(self, x):
        x = x.squeeze()
        mean = x.mean(axis=-1).unsqueeze(-1).repeat(1, 1, x.shape[-1])
        x = x - mean
        cov = x @ x.permute(0, 2, 1)
        cov = cov.to(self.dev)
        cov = cov / (x.shape[-1] - 1)
        tra = cov.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)
        tra = tra.view(-1, 1, 1)
        cov /= tra
        identity = torch.eye(cov.shape[-1], cov.shape[-1], device=self.dev).to(self.dev).repeat(x.shape[0], 1, 1)
        cov = cov + (1e-5 * identity)
        return cov


class E2R(nn.Module):
    def __init__(self, epochs, dim=-1):
        super().__init__()
        self.epochs = epochs
        self.signal2spd = signal2spd()
        self.dim = dim

    def patch_len(self, n, epochs):
        list_len = []
        base = n // epochs
        for i in range(epochs):
            list_len.append(base)
        for i in range(n - base * epochs):
            list_len[i] += 1

        if sum(list_len) == n:
            return list_len
        else:
            return ValueError('check your epochs and axis should be split again')

    def forward(self, x):
        # x with shape[bs, ch, time]
        list_patch = self.patch_len(x.shape[self.dim], int(self.epochs))
        x_list = list(torch.split(x, list_patch, dim=self.dim))
        for i, item in enumerate(x_list):
            x_list[i] = self.signal2spd(item)
        x = torch.stack(x_list).permute(1, 0, 2, 3)
        return x


class AttentionManifold(nn.Module):
    def __init__(self, in_features, out_features):
        super(AttentionManifold, self).__init__()

        self._in_features = in_features
        self._out_features = out_features

        self.q_trans = BiMap(self._in_features, self._out_features)
        self.k_trans = BiMap(self._in_features, self._out_features)
        self.v_trans = BiMap(self._in_features, self._out_features)

    def forward(self, x):
        Q = self.q_trans(x)
        K = self.k_trans(x)
        V = self.v_trans(x)

        Q_expand = Q.unsqueeze(2)
        K_expand = K.unsqueeze(1)

        atten_energy = log_euclidean_distance(Q_expand, K_expand)
        atten_weights = 1 / (1 + torch.log1p(atten_energy))
        atten_prob = F.softmax(atten_weights, dim=-1)

        output = log_euclidean_mean(atten_prob, V)

        return output


class SubmanifoldAttention(nn.Module):
    def __init__(self, in_dims, qk_dim, v_dim):
        super(SubmanifoldAttention, self).__init__()
        self.in_dims = in_dims
        self.qk_dim = qk_dim
        self.v_dim = v_dim

        self.q_trans = nn.ModuleList([BiMap(d, qk_dim) for d in in_dims])
        self.k_trans = nn.ModuleList([BiMap(d, qk_dim) for d in in_dims])
        self.v_trans = nn.ModuleList([BiMap(d, v_dim) for d in in_dims])

    def forward(self, x):
        Q = torch.cat([q_fn(xi) for q_fn, xi in zip(self.q_trans, x)], dim=1)
        K = torch.cat([k_fn(xi) for k_fn, xi in zip(self.k_trans, x)], dim=1)
        V = torch.cat([v_fn(xi) for v_fn, xi in zip(self.v_trans, x)], dim=1)

        Q_expand = Q.unsqueeze(2)
        K_expand = K.unsqueeze(1)

        atten_energy = log_euclidean_distance(Q_expand, K_expand)
        atten_weights = 1 / (1 + torch.log1p(atten_energy))
        atten_prob = F.softmax(atten_weights, dim=-1)

        # self.current_weights = atten_prob

        output = log_euclidean_mean(atten_prob, V)

        return output


class NewtonSchulzOrthogonalization(nn.Module):
    def __init__(self, num_iterations=8, eps=1e-6):
        super().__init__()
        self.num_iterations = num_iterations
        self.eps = eps

    def forward(self, W):
        k = W.shape[-2]
        norm = torch.linalg.norm(W, dim=(-2, -1), keepdim=True) + self.eps
        W_k = W / norm
        I = torch.eye(k, dtype=W.dtype, device=W.device)
        I = I.view(*([1] * (W.dim() - 2)), k, k).expand(*W.shape[:-2], k, k)

        for _ in range(self.num_iterations):
            WWT = W_k @ W_k.mT
            W_k = 0.5 * (3.0 * I - WWT) @ W_k

        return W_k


class NewtonHyperNet(nn.Module):
    def __init__(
            self,
            n,
            k_dims,
            hidden_dim=128,
            context_dim=64,
            dropout=0.0,
            base_init="branch",
            use_tanh=True,
            use_scale=False,
            dynamic_scale=0.1,
    ):
        super().__init__()
        self.n = n
        self.k_dims = k_dims if isinstance(k_dims, list) else [k_dims]
        self.total_k = sum(self.k_dims)
        self.base_init = self._normalize_base_init(base_init)
        self.use_tanh = use_tanh
        self.use_scale = use_scale
        self.dynamic_scale = dynamic_scale

        flat_dim = n * (n + 1) // 2
        self.tangent = LogEig()
        self.encoder = nn.Sequential(
            # nn.LayerNorm(flat_dim),
            nn.Linear(flat_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, context_dim),
            nn.GELU(),
        )
        self.branch_heads = nn.ModuleList([
            nn.Linear(context_dim, k * n) for k in self.k_dims
        ])
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

        for head in self.branch_heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, X):
        bs = X.shape[0]
        context = self.encoder(self.tangent(X))
        # context = self.encoder(self.tangent(X).unsqueeze(1)).squeeze(-1)

        W_raw_list = []
        for base, head, k in zip(self.base_w, self.branch_heads, self.k_dims):
            raw_delta = head(context).view(bs, k, self.n)
            delta = torch.tanh(raw_delta) if self.use_tanh else raw_delta
            delta_for_w = self.dynamic_scale * delta if self.use_scale else delta
            W_raw = base.unsqueeze(0) + delta_for_w
            W_raw_list.append(W_raw)

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
            use_tanh=True,
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


def _tri_dim(n):
    return n * (n + 1) // 2


class _SubmanifoldAttentionMixin:
    def _init_submanifold_blocks(
            self,
            n,
            k_dims,
            slice,
            num_classes,
            hidden_dim=128,
            context_dim=64,
    ):
        self.n = n
        self.k_dims = list(k_dims)
        self.subcov = Submanifold(
            self.n,
            self.k_dims,
            hidden_dim=hidden_dim,
            context_dim=context_dim,
        )
        self.att_dims = self.k_dims + [self.n]
        self.attentions = nn.ModuleList([AttentionManifold(d, d) for d in self.att_dims])
        self.re = ReEig(threshold=1e-4)

        self.tangent = LogEig()
        self.flat = nn.Flatten()
        feature_dim = sum(_tri_dim(d) for d in self.att_dims)
        self.linear = nn.Linear(feature_dim * slice, num_classes, bias=True)

    def _attended_features(self, x):
        sub_x_list = [self.subcov(x[:, i, :, :]) for i in range(x.size(1))]
        grouped_sub_x = zip(*sub_x_list)

        features = []
        for idx, covs_k in enumerate(grouped_sub_x):
            d_tensor = torch.cat(covs_k, dim=1)
            d_feat = self.attentions[idx](d_tensor)
            features.append(self.tangent(self.re(d_feat)))

        main_feat = self.tangent(self.re(self.attentions[-1](x)))
        features.append(main_feat)
        return torch.cat(features, dim=-1)
