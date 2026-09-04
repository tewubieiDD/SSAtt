import torch
from torch import nn

from spd.base_module_re import E2R, _SubmanifoldAttentionMixin
from spd.batchnorm import BatchNormDispersion, BatchNormTestStatsMode, SPDDSMBN


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

        # 论文实现
        in_size = spatial_filters
        self.conv_block1 = nn.Sequential(nn.Conv2d(1, temporal_filters, (1, 25), padding=(0, 12)),
                                         nn.BatchNorm2d(temporal_filters))
        self.conv_block2 = nn.Sequential(nn.Conv2d(temporal_filters, spatial_filters, (num_channels, 1)),
                                         nn.BatchNorm2d(spatial_filters))

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
