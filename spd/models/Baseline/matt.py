import torch
from torch import nn

from spd.base_module_re import E2R, AttentionManifold
from spd.batchnorm import SPDDSMBN, BatchNormDispersion, BatchNormTestStatsMode
from spd.modules import BiMap, ReEig, LogEig


class MAtt(nn.Module):
    def __init__(
            self,
            slice,
            num_channels,
            num_classes,
            temporal_filters,
            spatial_filters,
    ):
        super().__init__()
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

        self.spdnet = nn.Sequential(
            BiMap(in_size, 20),
            ReEig(threshold=1e-4),
        )
        self.matt = AttentionManifold(20, 18)
        self.logeig = nn.Sequential(
            LogEig(),
            nn.Flatten(),
        )
        self.classifier = nn.Linear(9 * 19 * slice, num_classes)

    def forward(self, x, d):
        x = self.conv_block1(x)
        x = self.conv_block2(x)

        x = self.ract1(x)
        x = self.spddsmbn(x, d)
        x = self.spdnet(x)
        x = self.matt(x)
        x = self.logeig(x)

        return self.classifier(x)

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


class BNCI2014Net(MAtt):
    def __init__(self, slice):
        super().__init__(
            slice,
            num_channels=22,
            num_classes=4,
            temporal_filters=4,
            spatial_filters=43,
        )


class BNCI2015Net(MAtt):
    def __init__(self, slice):
        super().__init__(
            slice,
            num_channels=13,
            num_classes=2,
            temporal_filters=5,
            spatial_filters=44,
        )
