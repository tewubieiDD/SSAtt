import math

import torch
import torch.nn as nn
from SSAtt.spd import SPDTransform, SPDTangentSpace, SPDRectified


class SubManifold(nn.Module):
    def __init__(self, ks, s=1):
        super().__init__()
        self.ks = ks if isinstance(ks, list) else [ks]
        self.s = s

    def _extract_submatrices(self, x, k, s):
        bs, n, _ = x.size()
        d = int(math.sqrt(n))

        outsize = (d - k) // s + 1
        num_windows = outsize * outsize
        device = x.device
        dtype = torch.long

        start_indices = torch.arange(0, d - k + 1, step=s, device=device, dtype=dtype)
        grid_ir, grid_ic = torch.meshgrid(start_indices, start_indices, indexing="ij")
        grid_ir, grid_ic = grid_ir.flatten(), grid_ic.flatten()

        rel_range = torch.arange(k, device=device, dtype=dtype)
        rel_a, rel_b = torch.meshgrid(rel_range, rel_range, indexing="ij")
        rel_indices = (rel_a * d + rel_b).flatten()

        idx = grid_ir.unsqueeze(1) * d + grid_ic.unsqueeze(1) + rel_indices.unsqueeze(0)
        idx_batch = idx.unsqueeze(0).expand(bs, -1, -1)
        idx_rows = idx_batch.reshape(bs, num_windows * k * k).unsqueeze(-1).expand(-1, -1, n)
        sub_rows = torch.gather(x, dim=1, index=idx_rows).view(bs, num_windows, k * k, n)
        idx_cols = idx_batch.unsqueeze(2).expand(-1, -1, k * k, -1)
        submatrices = torch.gather(sub_rows, dim=3, index=idx_cols)

        return submatrices

    def forward(self, x):
        xs = []

        for k in self.ks:
            sub_x = self._extract_submatrices(x, k, self.s)
            xs.append(sub_x)

        return xs


class CgNet(nn.Module):
    def __init__(self, epochs):
        super().__init__()
        # FE
        self.trans1 = SPDTransform(100, 80)
        self.trans2 = SPDTransform(80, 50)
        self.trans31 = SPDTransform(50, 25)
        self.trans32 = SPDTransform(50, 25)
        self.trans33 = SPDTransform(50, 25)
        self.trans34 = SPDTransform(50, 25)

        # SubManifold
        self.subcov1 = SubManifold([2])
        self.subcov2 = SubManifold([3])
        self.subcov3 = SubManifold([4])
        self.subcov4 = SubManifold([5])

        # Att
        self.ract = SPDRectified()

        # R2E
        self.tangent1 = SPDTangentSpace(4)
        self.tangent2 = SPDTangentSpace(9)
        self.tangent3 = SPDTangentSpace(16)
        self.tangent4 = SPDTangentSpace(25)
        self.flat = nn.Flatten()
        # fc
        self.linear = nn.Linear(2 * 5 * 16 + 5 * 9 * 9 + 8 * 17 * 4 + 13 * 25 * 1, 9, bias=True)

    def forward(self, x):
        x1 = self.ract(self.trans1(x.squeeze()))
        x2 = self.ract(self.trans2(x1))
        x31 = self.ract(self.trans31(x2))
        x32 = self.ract(self.trans32(x2))
        x33 = self.ract(self.trans33(x2))
        x34 = self.ract(self.trans34(x2))
        x31 = self.subcov1(x31)[0]
        x32 = self.subcov2(x32)[0]
        x33 = self.subcov3(x33)[0]
        x34 = self.subcov4(x34)[0]
        shape1 = x31.shape
        shape2 = x32.shape
        shape3 = x33.shape
        shape4 = x34.shape
        x31 = self.tangent1(x31.view(-1, shape1[-2], shape1[-1])).view(shape1[0], shape1[1], -1)
        x32 = self.tangent2(x32.view(-1, shape2[-2], shape2[-1])).view(shape2[0], shape2[1], -1)
        x33 = self.tangent3(x33.view(-1, shape3[-2], shape3[-1])).view(shape3[0], shape3[1], -1)
        x34 = self.tangent4(x34.view(-1, shape4[-2], shape4[-1])).view(shape4[0], shape4[1], -1)
        x1 = self.flat(x31)
        x2 = self.flat(x32)
        x3 = self.flat(x33)
        x4 = self.flat(x34)
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.linear(x)
        return x


class FphaNet(nn.Module):
    def __init__(self, epochs):
        super().__init__()
        # FE
        self.trans1 = SPDTransform(63, 56)
        self.trans2 = SPDTransform(56, 46)
        self.trans31 = SPDTransform(46, 36)
        self.trans32 = SPDTransform(46, 36)

        # SubManifold
        self.subcov1 = SubManifold([5])
        self.subcov2 = SubManifold([6])

        # Att
        self.ract = SPDRectified()

        # R2E
        self.tangent1 = SPDTangentSpace(25)
        self.tangent2 = SPDTangentSpace(36)
        self.flat = nn.Flatten()
        # fc
        self.linear = nn.Linear(13 * 25 * 4 + 18 * 37 * 1, 45, bias=True)

    def forward(self, x):
        x1 = self.ract(self.trans1(x.squeeze()))
        x2 = self.ract(self.trans2(x1))
        x31 = self.ract(self.trans31(x2))
        x32 = self.ract(self.trans32(x2))
        x31 = self.subcov1(x31)[0]
        x32 = self.subcov2(x32)[0]
        shape1 = x31.shape
        shape2 = x32.shape
        x31 = self.tangent1(x31.view(-1, shape1[-2], shape1[-1])).view(shape1[0], shape1[1], -1)
        x32 = self.tangent2(x32.view(-1, shape2[-2], shape2[-1])).view(shape2[0], shape2[1], -1)
        x1 = self.flat(x31)
        x2 = self.flat(x32)
        x = torch.cat((x1, x2), dim=1)
        x = self.linear(x)
        return x


class MdsdNet(nn.Module):
    def __init__(self, epochs):
        super().__init__()
        # FE
        self.trans1 = SPDTransform(400, 200)
        self.trans2 = SPDTransform(200, 100)
        self.trans31 = SPDTransform(100, 49)
        self.trans32 = SPDTransform(100, 49)
        self.trans33 = SPDTransform(100, 49)

        # SubManifold
        self.subcov1 = SubManifold([2])
        self.subcov2 = SubManifold([6])
        self.subcov3 = SubManifold([7])

        # Att
        self.ract = SPDRectified()

        # R2E
        self.tangent1 = SPDTangentSpace(4)
        self.tangent2 = SPDTangentSpace(36)
        self.tangent3 = SPDTangentSpace(49)
        self.flat = nn.Flatten()
        # fc
        self.linear = nn.Linear(2 * 5 * 36 + 18 * 37 * 4 + 25 * 49 * 1, 13, bias=True)

    def forward(self, x):
        x1 = self.ract(self.trans1(x.squeeze()))
        x2 = self.ract(self.trans2(x1))
        x31 = self.ract(self.trans31(x2))
        x32 = self.ract(self.trans32(x2))
        x33 = self.ract(self.trans33(x2))
        x31 = self.subcov1(x31)[0]
        x32 = self.subcov2(x32)[0]
        x33 = self.subcov3(x33)[0]
        shape1 = x31.shape
        shape2 = x32.shape
        shape3 = x33.shape
        x31 = self.tangent1(x31.view(-1, shape1[-2], shape1[-1])).view(shape1[0], shape1[1], -1)
        x32 = self.tangent2(x32.view(-1, shape2[-2], shape2[-1])).view(shape2[0], shape2[1], -1)
        x33 = self.tangent3(x33.view(-1, shape3[-2], shape3[-1])).view(shape3[0], shape3[1], -1)
        x1 = self.flat(x31)
        x2 = self.flat(x32)
        x3 = self.flat(x33)
        x = torch.cat((x1, x2, x3), dim=1)
        x = self.linear(x)
        return x
