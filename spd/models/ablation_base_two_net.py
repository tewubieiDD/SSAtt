from spd.ablation_base import E2R, Submanifold, AttentionManifold
from spd.models.ablation_common import make_eeg_kdim_classes


BNCI2014001NewtonKDimNet, BNCI2015001NewtonKDimNet = make_eeg_kdim_classes(
    E2R,
    Submanifold,
    AttentionManifold,
)
