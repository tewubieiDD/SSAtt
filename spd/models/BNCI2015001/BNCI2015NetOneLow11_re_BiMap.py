from spd.models.ablation_bimap_two_net import BNCI2015001NewtonKDimNet


class BNCI2015Net(BNCI2015001NewtonKDimNet):
    def __init__(self, slice):
        super().__init__(slice, k_dims=[11])
