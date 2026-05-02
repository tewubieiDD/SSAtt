import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from SSAtt.spd import SPDTransform, SPDTangentSpace, SPDRectified


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
    def __init__(self, epochs):
        super().__init__()
        self.epochs = epochs
        self.signal2spd = signal2spd()

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
        list_patch = self.patch_len(x.shape[-1], int(self.epochs))
        x_list = list(torch.split(x, list_patch, dim=-1))
        for i, item in enumerate(x_list):
            x_list[i] = self.signal2spd(item)
        x = torch.stack(x_list).permute(1, 0, 2, 3)
        return x


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


class SubmanifoldAttention(nn.Module):
    def __init__(self, in_dims, qk_dim, v_dim):
        super(SubmanifoldAttention, self).__init__()

        self.in_dims = in_dims
        self.qk_dim = qk_dim
        self.v_dim = v_dim
        q_trans, k_trans, v_trans = [], [], []
        for i in range(len(in_dims)):
            q_trans.append(SPDTransform(self.in_dims[i], self.qk_dim).cpu())
            k_trans.append(SPDTransform(self.in_dims[i], self.qk_dim).cpu())
            v_trans.append(SPDTransform(self.in_dims[i], self.v_dim).cpu())
        self.q_trans, self.k_trans, self.v_trans = nn.ModuleList(q_trans), nn.ModuleList(k_trans), nn.ModuleList(
            v_trans)

    # def tensor_log(self, t):  # 4dim
    #     u, s, v = torch.svd(t)
    #     return u @ torch.diag_embed(torch.log(s)) @ v.permute(0, 1, 3, 2)

    def tensor_log(self, t, epsilon=1e-6):  # 4dim
        t.diagonal(dim1=-2, dim2=-1).add_(torch.rand_like(t.diagonal(dim1=-2, dim2=-1)) * epsilon)
        u, s, v = torch.svd(t)
        return u @ torch.diag_embed(torch.log(s)) @ v.permute(0, 1, 3, 2)

    def tensor_exp(self, t):  # 4dim
        # condition: t is symmetric!
        s, u = torch.linalg.eigh(t)
        return u @ torch.diag_embed(torch.exp(s)) @ u.permute(0, 1, 3, 2)

    def log_euclidean_distance(self, A, B):
        inner_term = self.tensor_log(A) - self.tensor_log(B)
        inner_multi = inner_term @ inner_term.permute(0, 1, 3, 2)
        _, s, _ = torch.svd(inner_multi)
        final = torch.sum(s, dim=-1)
        return final

    def LogEuclideanMean(self, weight, cov):
        # cov:[bs, #p, s, s]
        # weight:[bs, #p, #p]
        bs = cov.shape[0]
        num_p = cov.shape[1]
        size = cov.shape[2]
        cov = self.tensor_log(cov).view(bs, num_p, -1)
        output = weight @ cov  # [bs, #p, -1]
        output = output.view(bs, num_p, size, size)
        return self.tensor_exp(output)

    def forward(self, x, keep_only_last=True):
        Q = [];
        K = [];
        V = []
        # calculate Q K V
        for i in range(len(x)):
            bs = x[i].shape[0]
            m = x[i].shape[1]
            tmp_x = x[i].reshape(bs * m, self.in_dims[i], self.in_dims[i])
            q = self.q_trans[i](tmp_x).view(bs, m, self.qk_dim, self.qk_dim)
            k = self.k_trans[i](tmp_x).view(bs, m, self.qk_dim, self.qk_dim)
            v = self.v_trans[i](tmp_x).view(bs, m, self.v_dim, self.v_dim)
            Q.append(q);
            K.append(k);
            V.append(v)
        Q = torch.cat(Q, dim=1)
        K = torch.cat(K, dim=1)
        V = torch.cat(V, dim=1)

        # calculate the attention score
        Q_expand = Q.repeat(1, V.shape[1], 1, 1)

        K_expand = K.unsqueeze(2).repeat(1, 1, V.shape[1], 1, 1)
        K_expand = K_expand.view(K_expand.shape[0], K_expand.shape[1] * K_expand.shape[2], K_expand.shape[3],
                                 K_expand.shape[4])

        atten_energy = self.log_euclidean_distance(Q_expand, K_expand).view(V.shape[0], V.shape[1], V.shape[1])
        atten_prob = nn.Softmax(dim=-2)(1 / (1 + torch.log(1 + atten_energy))).permute(0, 2, 1)  # now row is c.c.

        # calculate outputs(v_i') of attention module
        output = self.LogEuclideanMean(atten_prob, V)

        output = output.view(V.shape[0], V.shape[1], self.v_dim, self.v_dim)

        if keep_only_last:
            output = output[:, -1:, :, :]

        shape = list(output.shape[:2])
        shape.append(-1)

        output = output.contiguous().view(-1, self.v_dim, self.v_dim)
        return output, shape


class AttentionManifold(nn.Module):
    def __init__(self, in_embed_size, out_embed_size):
        super(AttentionManifold, self).__init__()

        self.d_in = in_embed_size
        self.d_out = out_embed_size
        self.q_trans = SPDTransform(self.d_in, self.d_out).cpu()
        self.k_trans = SPDTransform(self.d_in, self.d_out).cpu()
        self.v_trans = SPDTransform(self.d_in, self.d_out).cpu()

    def tensor_log(self, t):  # 4dim
        u, s, v = torch.svd(t)
        return u @ torch.diag_embed(torch.log(s)) @ v.permute(0, 1, 3, 2)

    def tensor_exp(self, t):  # 4dim
        # condition: t is symmetric!
        s, u = torch.linalg.eigh(t)
        return u @ torch.diag_embed(torch.exp(s)) @ u.permute(0, 1, 3, 2)

    def log_euclidean_distance(self, A, B):
        inner_term = self.tensor_log(A) - self.tensor_log(B)
        inner_multi = inner_term @ inner_term.permute(0, 1, 3, 2)
        _, s, _ = torch.svd(inner_multi)
        final = torch.sum(s, dim=-1)
        return final

    def LogEuclideanMean(self, weight, cov):
        # cov:[bs, #p, s, s]
        # weight:[bs, #p, #p]
        bs = cov.shape[0]
        num_p = cov.shape[1]
        size = cov.shape[2]
        cov = self.tensor_log(cov).view(bs, num_p, -1)
        output = weight @ cov  # [bs, #p, -1]
        output = output.view(bs, num_p, size, size)
        return self.tensor_exp(output)

    def forward(self, x, shape=None):
        if len(x.shape) == 3 and shape is not None:
            x = x.view(shape[0], shape[1], self.d_in, self.d_in)
        x = x.to(torch.double)  # patch:[b, #patch, c, c]
        q_list = [];
        k_list = [];
        v_list = []
        # calculate Q K V
        bs = x.shape[0]
        m = x.shape[1]
        x = x.reshape(bs * m, self.d_in, self.d_in)
        Q = self.q_trans(x).view(bs, m, self.d_out, self.d_out)
        K = self.k_trans(x).view(bs, m, self.d_out, self.d_out)
        V = self.v_trans(x).view(bs, m, self.d_out, self.d_out)

        # calculate the attention score
        Q_expand = Q.repeat(1, V.shape[1], 1, 1)

        K_expand = K.unsqueeze(2).repeat(1, 1, V.shape[1], 1, 1)
        K_expand = K_expand.view(K_expand.shape[0], K_expand.shape[1] * K_expand.shape[2], K_expand.shape[3],
                                 K_expand.shape[4])

        atten_energy = self.log_euclidean_distance(Q_expand, K_expand).view(V.shape[0], V.shape[1], V.shape[1])
        atten_prob = nn.Softmax(dim=-2)(1 / (1 + torch.log(1 + atten_energy))).permute(0, 2, 1)  # now row is c.c.

        # calculate outputs(v_i') of attention module
        output = self.LogEuclideanMean(atten_prob, V)

        output = output.view(V.shape[0], V.shape[1], self.d_out, self.d_out)

        shape = list(output.shape[:2])
        shape.append(-1)

        output = output.contiguous().view(-1, self.d_out, self.d_out)
        return output, shape


class SSAtt_bci(nn.Module):
    def __init__(self, epochs):
        super().__init__()
        # FE
        # bs, 1, channel, sample
        self.conv1 = nn.Conv2d(1, 22, (22, 1))
        self.Bn1 = nn.BatchNorm2d(22)
        # bs, 22, 1, sample
        self.conv2 = nn.Conv2d(22, 25, (1, 12), padding=(0, 6))
        self.Bn2 = nn.BatchNorm2d(25)

        # E2R
        self.ract1 = E2R(epochs=epochs)
        # riemannian part
        self.subcov = SubManifold([2, 3, 4, 5])
        self.satt = SubmanifoldAttention([4, 9, 16, 25], 4, 25)
        self.ract2 = SPDRectified()

        self.matt = AttentionManifold(25, 18)

        # R2E
        self.tangent = SPDTangentSpace(18)
        self.flat = nn.Flatten()
        # fc
        self.linear = nn.Linear(9 * 19 * epochs, 4, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.Bn1(x)
        x = self.conv2(x)
        x = self.Bn2(x)

        x = self.ract1(x)
        x1 = self.subcov(x[:, 0, :, :])
        x2 = self.subcov(x[:, 1, :, :])
        x3 = self.subcov(x[:, 2, :, :])
        x1, shape1 = self.satt(x1)
        x1 = self.ract2(x1)
        x2, shape2 = self.satt(x2)
        x2 = self.ract2(x2)
        x3, shape3 = self.satt(x3)
        x3 = self.ract2(x3)
        x = torch.stack([x1, x2, x3], dim=1)
        x, shape = self.matt(x)
        x = self.ract2(x)

        x = self.tangent(x)
        x = x.view(shape[0], shape[1], -1)
        x = self.flat(x)
        x = self.linear(x)
        return x


class SSAtt_mamem(nn.Module):
    def __init__(self, epochs):
        super().__init__()
        # FE
        # bs, 1, channel, sample
        self.conv1 = nn.Conv2d(1, 125, (8, 1))
        self.Bn1 = nn.BatchNorm2d(125)
        # bs, 8, 1, sample
        self.conv2 = nn.Conv2d(125, 16, (1, 36), padding=(0, 18))
        self.Bn2 = nn.BatchNorm2d(16)

        # E2R
        self.ract1 = E2R(epochs)
        # riemannian part
        self.subcov = SubManifold([2, 3, 4])
        self.satt = SubmanifoldAttention([4, 9, 16], 4, 16)
        self.ract2 = SPDRectified()

        self.matt = AttentionManifold(16, 12)

        # R2E
        self.tangent = SPDTangentSpace(12)
        self.flat = nn.Flatten()
        # fc
        self.linear = nn.Linear(6 * 13 * epochs, 5, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.Bn1(x)
        x = self.conv2(x)
        x = self.Bn2(x)

        x = self.ract1(x)
        x1 = self.subcov(x[:, 0, :, :])
        x2 = self.subcov(x[:, 1, :, :])
        x3 = self.subcov(x[:, 2, :, :])
        x4 = self.subcov(x[:, 3, :, :])
        x5 = self.subcov(x[:, 4, :, :])
        x6 = self.subcov(x[:, 5, :, :])
        x7 = self.subcov(x[:, 6, :, :])
        x1, shape1 = self.satt(x1)
        x1 = self.ract2(x1)
        x2, shape2 = self.satt(x2)
        x2 = self.ract2(x2)
        x3, shape3 = self.satt(x3)
        x3 = self.ract2(x3)
        x4, shape4 = self.satt(x4)
        x4 = self.ract2(x4)
        x5, shape5 = self.satt(x5)
        x5 = self.ract2(x5)
        x6, shape6 = self.satt(x6)
        x6 = self.ract2(x6)
        x7, shape7 = self.satt(x7)
        x7 = self.ract2(x7)
        x = torch.stack([x1, x2, x3, x4, x5, x6, x7], dim=1)
        x, shape = self.matt(x)
        x = self.ract2(x)

        x = self.tangent(x)
        x = x.view(shape[0], shape[1], -1)
        x = self.flat(x)
        x = self.linear(x)
        return x


class SSAtt_cha(nn.Module):
    def __init__(self, epochs):
        super().__init__()
        # FE
        # bs, 1, channel, sample
        self.conv1 = nn.Conv2d(1, 22, (56, 1))
        self.Bn1 = nn.BatchNorm2d(22)
        # bs, 56, 1, sample
        self.conv2 = nn.Conv2d(22, 16, (1, 64), padding=(0, 32))
        self.Bn2 = nn.BatchNorm2d(16)

        # E2R
        self.ract1 = E2R(epochs=epochs)
        # riemannian part
        self.subcov = SubManifold([2, 3, 4])
        self.satt = SubmanifoldAttention([4, 9, 16], 4, 16)
        self.ract2 = SPDRectified()

        self.matt = AttentionManifold(16, 8)

        # R2E
        self.tangent = SPDTangentSpace(8)
        self.flat = nn.Flatten()
        # fc
        self.linear = nn.Linear(4 * 9 * epochs, 2, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.Bn1(x)
        x = self.conv2(x)
        x = self.Bn2(x)

        x = self.ract1(x)
        x1 = self.subcov(x[:, 0, :, :])
        x2 = self.subcov(x[:, 1, :, :])
        x3 = self.subcov(x[:, 2, :, :])
        x1, shape1 = self.satt(x1)
        x1 = self.ract2(x1)
        x2, shape2 = self.satt(x2)
        x2 = self.ract2(x2)
        x3, shape3 = self.satt(x3)
        x3 = self.ract2(x3)
        x = torch.stack([x1, x2, x3], dim=1)
        x, shape = self.matt(x)
        x = self.ract2(x)

        x = self.tangent(x)
        x = x.view(shape[0], shape[1], -1)
        x = self.flat(x)
        x = self.linear(x)
        return x


class SSAtt_cg(nn.Module):
    def __init__(self, epochs):
        super().__init__()
        # FE
        self.trans1 = SPDTransform(100, 81)
        self.trans21 = SPDTransform(81, 49)
        self.trans22 = SPDTransform(81, 49)
        self.trans23 = SPDTransform(81, 49)
        self.trans31 = SPDTransform(49, 25)
        self.trans32 = SPDTransform(49, 25)
        self.trans33 = SPDTransform(49, 25)
        self.trans34 = SPDTransform(49, 25)

        # SubManifold
        self.subcov21 = SubManifold([2])
        self.subcov22 = SubManifold([6])
        self.subcov23 = SubManifold([7])
        self.subcov31 = SubManifold([2])
        self.subcov32 = SubManifold([3])
        self.subcov33 = SubManifold([4])
        self.subcov34 = SubManifold([5])

        # Att
        self.satt1 = SubmanifoldAttention([4, 36, 49], 4, 49)
        self.satt2 = SubmanifoldAttention([4, 9, 16, 25], 4, 25)
        self.ract = SPDRectified()

        # R2E
        self.tangent = SPDTangentSpace(25)
        self.flat = nn.Flatten()
        # fc
        self.linear = nn.Linear(25 * 13, 9, bias=True)

    def forward(self, x):
        x1 = self.ract(self.trans1(x.squeeze()))
        x21 = self.ract(self.trans21(x1))
        x22 = self.ract(self.trans22(x1))
        x23 = self.ract(self.trans23(x1))
        x21 = self.subcov21(x21)
        x22 = self.subcov22(x22)
        x23 = self.subcov23(x23)
        x21.extend(x22)
        x21.extend(x23)
        x2, shape2 = self.satt1(x21)
        x2 = self.ract(x2)
        x31 = self.ract(self.trans31(x2))
        x32 = self.ract(self.trans32(x2))
        x33 = self.ract(self.trans33(x2))
        x34 = self.ract(self.trans34(x2))
        x31 = self.subcov31(x31)
        x32 = self.subcov32(x32)
        x33 = self.subcov33(x33)
        x34 = self.subcov34(x34)
        x31.extend(x32)
        x31.extend(x33)
        x31.extend(x34)
        x, shape = self.satt2(x31)
        x = self.ract(x)

        x = self.tangent(x)
        x = self.flat(x)
        x = self.linear(x)
        return x
