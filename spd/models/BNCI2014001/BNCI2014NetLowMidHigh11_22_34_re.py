from spd.models.two_net_re import BNCI2014001NewtonKDimNet


class BNCI2014Net(BNCI2014001NewtonKDimNet):
    def __init__(self, slice):
        super().__init__(slice, k_dims=[11, 22, 34])
