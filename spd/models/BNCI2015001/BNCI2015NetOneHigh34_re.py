from spd.models.two_net_re import BNCI2015001NewtonKDimNet


class BNCI2015Net(BNCI2015001NewtonKDimNet):
    def __init__(self, slice):
        super().__init__(slice, k_dims=[34])
