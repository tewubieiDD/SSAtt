import torch
from .manifold import Manifold


def _sym(x):
    return 0.5 * (x + _transpose(x))


def _transpose(x):
    """Returns the transpose for each matrix in a batch."""
    return x.transpose(dim0=-2, dim1=-1)


def _symapply(x, f, *, wmin=None, wmax=None):
    r"""Template function acting on stacked symmetric matrices that applies a
    given analytic function on them via eigenvalue decomposition.
    """
    # try:
    w, v = torch.linalg.eigh(x)
    # except RuntimeError as e:
    #    print(e)
    #     torch.save(x, "x.pt")
    #   sys.exit(0)

    # print("W", w[0])
    # print("V", v[0])
    if wmin is not None or wmax is not None:
        # w.data.clamp_(min=wmin, max=wmax)
        w = w.clamp(min=wmin, max=wmax)
    return _mvmt(v, f(w), v)


def _lult(x, u, ret_chol=False):
    l_inv, l = _invchol(x, x.shape[-1], ret_chol=ret_chol)
    lult = _axat(l_inv, u)
    return lult, l


def _invchol(x, n, ret_chol=False):
    l = torch.linalg.cholesky(x)
    eye = torch.eye(n, out=x.new(n, n))
    l_inv = torch.linalg.solve_triangular(l, eye, upper=False)
    if not ret_chol:
        return l_inv, None
    return l_inv, l


def _mvmt(u, w, v):
    r""" "The multiplication `u @ diag(w) @ v.T`. The name stands for
    matrix/vector/matrix-transposed.
    """
    return torch.einsum("...ij,...j,...kj->...ik", u, w, v)


def _axat(a, x):
    r"""Returns the product :math:`A X A^\top` for each pair of matrices in a
    batch. This should give the same result as :py:`A @ X @ A.T`.
    """
    return torch.einsum("...ij,...jk,...lk->...il", a, x, a)


def _trace(x, keepdim=False):
    """Returns the trace for each matrix in a batch."""
    traces = x.diagonal(dim1=-2, dim2=-1).sum(-1)
    return traces.view(-1, 1, 1) if keepdim else traces


class SPD(Manifold):
    r"""
    A class for the manifold of SPD matrices.


    Manifold implementation taken from
    [1] https://github.com/dalab/matrix-manifolds/blob/master/graphembed/graphembed/manifolds/spd.py
    [2] https://github.com/dalab/matrix-manifolds/blob/master/graphembed/graphembed/linalg/torch_batch.py
    """

    def __init__(self, metric="aff_inv"):
        super().__init__()
        self.wmin = 1e-8
        self.wmax = 1e8
        self.metric = metric  # "aff_inv" or "log_euc"

    def inner(self, x, u, v, keepdim=False):
        l = torch.linalg.cholesky(x)
        x_inv_u = torch.cholesky_solve(u, l)
        x_inv_v = torch.cholesky_solve(v, l)
        return _trace(x_inv_u @ x_inv_v, keepdim=keepdim)

    def norm(self, x, u, squared=False, keepdim=False):
        lult, _ = _lult(x, u)
        norm_sq = lult.pow(2).sum((-2, -1), keepdim=keepdim)

        return norm_sq if squared else norm_sq.sqrt()

    def proju(self, x, u):
        return _sym(u)

    def projx(self, x):
        return _symapply(_sym(x), lambda w: w, wmin=self.wmin, wmax=self.wmax)

    def exp(self, x, u, t=1):
        if self.metric == "aff_inv":
            return x @ torch.matrix_exp(t * torch.linalg.solve(x, u))
        elif self.metric == "log_euc":
            return torch.matrix_exp(self.log(x) + t * u)

    def log(self, x, u):
        return NotImplementedError

    def __str__(self):
        return "Symmetric Positive Definite"

    def powm(self, x, p):
        """对称矩阵幂：x^p（通过特征分解实现，并做特征值裁剪保证稳定）"""
        return _symapply(
            _sym(x),
            lambda w: w.pow(p),
            wmin=self.wmin,
            wmax=self.wmax,
        )

    def transp(self, x, y, v):
        """
        AIRM(aff_inv) 下：沿 x->y 测地线的平行传输，把 x 处切向量 v 传到 y 处。
        Log-Euclidean 下：在 log 域是欧式空间，平行传输为恒等（这里按“原空间切向量”约定返回 v）。
        """
        v = self.proju(x, v)  # 切空间是对称矩阵

        if self.metric == "aff_inv":
            # X^{1/2}, X^{-1/2}
            sqrt_x = self.powm(x, 0.5)
            inv_sqrt_x = self.powm(x, -0.5)

            # P = (X^{-1/2} Y X^{-1/2})^{1/2}
            middle = _axat(inv_sqrt_x, y)  # inv_sqrt_x @ y @ inv_sqrt_x^T (inv_sqrt_x 本身对称，T无影响)
            pdt = self.powm(middle, 0.5)

            # E = X^{1/2} P X^{-1/2}
            e = sqrt_x @ pdt @ inv_sqrt_x

            # v_y = E v E^T
            v_y = e @ v @ e.transpose(-1, -2)
            return self.proju(y, v_y)

        elif self.metric == "log_euc":
            # 若你把切向量定义在原空间（对称矩阵），则 log-euc 的平行传输可视作恒等
            return self.proju(y, v)

        else:
            raise ValueError(f"Unknown metric: {self.metric}")
