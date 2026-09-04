import torch
from torch import nn

from spd.batchnorm import SPDDSMBN, BatchNormDispersion, BatchNormTestStatsMode
from spd.modules import ReEig, LogEig


def _tri_dim(n):
    return n * (n + 1) // 2


def make_eeg_kdim_classes(E2R, Submanifold, AttentionManifold):
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

    class BNCINewtonKDimNet(nn.Module, _SubmanifoldAttentionMixin):
        def __init__(
                self,
                slice,
                k_dims,
                n,
                num_classes,
                num_channels,
                temporal_filters,
                spatial_filters,
                # temp_cnn_kernel=25,
                submanifold_hidden_dim=256,
                submanifold_context_dim=64,
        ):
            super().__init__()
            # 代码实现
            # dim1 = 22
            # in_size = n
            # self.conv_block1 = nn.Sequential(nn.Conv2d(1, dim1, (num_channels, 1)), nn.BatchNorm2d(dim1), nn.ELU())
            #
            # self.conv_block2 = nn.Sequential(nn.Conv2d(dim1, dim1, (1, 13), bias=False, groups=dim1, padding=(0, 12 // 2)),
            #                                  nn.Conv2d(dim1, dim1, (1, 1), bias=False, padding=(0, 0)),
            #                                  nn.BatchNorm2d(dim1), nn.ELU(),
            #                                  nn.Conv2d(dim1, in_size, (1, 12), padding=(0, 6)),
            #                                  nn.BatchNorm2d(in_size), nn.ELU())

            # 论文实现v4
            in_size = spatial_filters
            self.conv_block1 = nn.Sequential(nn.Conv2d(1, temporal_filters, (1, 25), padding=(0, 12)),
                                             nn.BatchNorm2d(temporal_filters))
            self.conv_block2 = nn.Sequential(nn.Conv2d(temporal_filters, spatial_filters, (num_channels, 1)),
                                             nn.BatchNorm2d(spatial_filters))

            # v2
            # self.conv_block1 = nn.Sequential(
            #     nn.Conv2d(1, temporal_filters, (1, 13), bias=False, groups=1, padding=(0, 12 // 2)),
            #     nn.Conv2d(temporal_filters, temporal_filters, (1, 1), bias=False, padding=(0, 0)),
            #     nn.BatchNorm2d(temporal_filters), nn.ELU(),
            #     nn.Conv2d(temporal_filters, temporal_filters, (1, 12), padding=(0, 6)),
            #     nn.BatchNorm2d(temporal_filters), nn.ELU())
            # self.conv_block2 = nn.Sequential(nn.Conv2d(temporal_filters, spatial_filters, (num_channels, 1)),
            #                                  nn.BatchNorm2d(spatial_filters))

            # v6/v7
            # self.conv_block1 = nn.Sequential(nn.Conv2d(1, 22, (num_channels, 1)),
            #                                  nn.BatchNorm2d(22))
            # self.conv_block2 = nn.Sequential(nn.Conv2d(22, spatial_filters, (1, 12), padding=(0, 6)),
            #                                  nn.BatchNorm2d(spatial_filters))

            self.ract1 = E2R(slice)
            self.spddsmbn = SPDDSMBN(
                shape=(slice, in_size, in_size),
                batchdim=0,
                learn_mean=False,
                learn_std=True,
                dispersion=BatchNormDispersion.SCALAR,
                eta=1.0,
                eta_test=0.1,
                eps=1e-5,
            )

            self._init_submanifold_blocks(
                n=in_size,
                k_dims=k_dims,
                slice=slice,
                num_classes=num_classes,
                hidden_dim=submanifold_hidden_dim,
                context_dim=submanifold_context_dim,
            )

        def forward(self, x, d):
            x = self.conv_block1(x)
            x = self.conv_block2(x)

            x = self.ract1(x)
            x = self.spddsmbn(x, d)
            x = self._attended_features(x)
            x = self.flat(x)
            return self.linear(x)

        @torch.no_grad()
        def domainadapt_finetune(self, x, y=None, d=None, target_domains=None):
            """Refit target-domain SPDDSMBN statistics from unlabeled data."""
            if d is None:
                raise ValueError("domain ids d are required")
            d = torch.as_tensor(d, device=x.device).reshape(-1)
            self.eval()
            self.spddsmbn.set_test_stats_mode(BatchNormTestStatsMode.REFIT)
            try:
                for domain in torch.unique(d):
                    mask = d == domain
                    self.forward(x[mask], d[mask])
            finally:
                self.spddsmbn.set_test_stats_mode(BatchNormTestStatsMode.BUFFER)

    class BNCI2014001NewtonKDimNet(BNCINewtonKDimNet):
        def __init__(self, slice, k_dims):
            super().__init__(
                slice=slice,
                k_dims=k_dims,
                n=43,
                num_classes=4,
                num_channels=22,
                temporal_filters=4,
                spatial_filters=43,
            )

    class BNCI2015001NewtonKDimNet(BNCINewtonKDimNet):
        def __init__(self, slice, k_dims):
            super().__init__(
                slice=slice,
                k_dims=k_dims,
                n=44,
                num_classes=2,
                num_channels=13,
                temporal_filters=5,
                spatial_filters=44,
            )

    return BNCI2014001NewtonKDimNet, BNCI2015001NewtonKDimNet
