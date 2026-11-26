"""
Lattice models in the form block DMRG needs.

A model says how to start a block at either end of the chain, how to enlarge a block by one
site, and how to join an enlarged left block to an enlarged right block (the superblock).
Each block carries its Hamiltonian plus whatever operators the coupling to the rest of the
chain needs (the edge spin operators, and for the Schwinger model the charge sums that feed
the long-range Coulomb energy).

Local basis states carry an integer U(1) charge (twice S^z), so blocks can be kept
block-diagonal in charge and the superblock restricted to one total-charge sector.
"""
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp


@dataclass
class Block:
    side: str                 # "left" or "right"
    size: int                 # number of sites
    ops: dict                 # name -> sparse matrix in the block basis
    charges: np.ndarray       # integer U(1) charge (2 S^z) of each basis state, or None
    meta: dict = field(default_factory=dict)

    @property
    def dim(self):
        return self.ops["H"].shape[0]


def spin_operators(S):
    """(S^z, S^+) for spin S as sparse matrices, basis ordered m = S, S-1, ..., -S."""
    m = np.arange(S, -S - 1, -1)
    sz = sp.diags(m)
    sp_vals = np.sqrt(S * (S + 1) - m[1:] * (m[1:] + 1))   # <m+1|S+|m>
    splus = sp.diags(sp_vals, 1)
    return sp.csr_matrix(sz), sp.csr_matrix(splus), (2 * m).astype(int)


def _kron(a, b):
    return sp.kron(a, b, format="csr")


def _eye(n):
    return sp.identity(n, format="csr")


def _combine(q_first, q_second):
    if q_first is None or q_second is None:
        return None
    return np.add.outer(q_first, q_second).ravel()


class XXZChain:
    """
    H = J sum_i [ (S+_i S-_{i+1} + S-_i S+_{i+1}) / 2 + Delta S^z_i S^z_{i+1} ] + h sum_i S^z_i

    Delta = 0 is the XY model (free fermions), Delta = 1 the Heisenberg model; S = 1 with
    Delta = 1 is the Haldane chain.
    """

    def __init__(self, J=1.0, Delta=1.0, h=0.0, S=0.5):
        self.J, self.Delta, self.h, self.S = J, Delta, h, S
        self.sz, self.splus, self.q = spin_operators(S)
        self.d = self.sz.shape[0]

    def _site(self):
        return {"H": self.h * self.sz, "Sz": self.sz, "S+": self.splus}

    def _bond(self, a, b):
        """Coupling between operators a (left) and b (right), each a dict with Sz, S+."""
        ab = _kron(a["S+"], b["S+"].T) + _kron(a["S+"].T, b["S+"])
        return self.J * (0.5 * ab + self.Delta * _kron(a["Sz"], b["Sz"]))

    def initial_block(self, side):
        return Block(side, 1, self._site(), self.q.copy())

    def enlarge(self, block, side):
        s, I_b, I_s = self._site(), _eye(block.dim), _eye(self.d)
        edge = {k: block.ops[k] for k in ("Sz", "S+")}
        if side == "left":      # new site to the right of the block
            H = _kron(block.ops["H"], I_s) + _kron(I_b, s["H"]) + self._bond(edge, s)
            new = {"Sz": _kron(I_b, s["Sz"]), "S+": _kron(I_b, s["S+"])}
            q = _combine(block.charges, self.q)
        else:                   # new site to the left of the block
            H = _kron(I_s, block.ops["H"]) + _kron(s["H"], I_b) + self._bond(s, edge)
            new = {"Sz": _kron(s["Sz"], I_b), "S+": _kron(s["S+"], I_b)}
            q = _combine(self.q, block.charges)
        return Block(side, block.size + 1, {"H": H, **new}, q)

    def superblock(self, L, R):
        H = _kron(L.ops["H"], _eye(R.dim)) + _kron(_eye(L.dim), R.ops["H"])
        return H + self._bond({k: L.ops[k] for k in ("Sz", "S+")}, {k: R.ops[k] for k in ("Sz", "S+")})


class SchwingerModel:
    """
    The massive Schwinger model on a staggered lattice with open ends, written as a spin chain
    (Kogut-Susskind, Banks-Susskind-Kogut; conventions of Arguello Cruz, Tarnopolsky & Xin,
    arXiv:2412.01902):

        2H / (e^2 a) = sum_{n=0}^{N-2} ( sum_{k<=n} Q_k + theta/2pi )^2
                       + (mu/2) sum_n (-1)^n Z_n
                       + x sum_n ( s+_n s-_{n+1} + s-_n s+_{n+1} )

    with Q_k = (Z_k + (-1)^k) / 2, x = 1/(e a)^2 and mu = 2 sqrt(x) m_lat/e. The lattice mass
    m_lat = m - e^2 a / 8 (Dempsey, Klebanov, Pufu & Zan 2022) makes the lattice approach the
    continuum faster; `from_continuum` applies it. The Gauss law has eliminated the gauge field,
    which leaves the non-local Coulomb term.

    To handle that term blockwise:
      * a left block (sites 0..l-1) owns links 0..l-1 and carries Qtot = sum_k Q_k + theta/2pi;
      * a right block (sites N-r..N-1) owns its r-1 internal links and carries
        P = sum over its links n of the partial charge sum p_n (counted from its left end),
        so that with charge Qtot to its left its Coulomb energy is
        sum_n (Qtot + p_n)^2 = (r-1) Qtot^2 + 2 Qtot P + H_C,right.
    Site parity is measured from the left for left blocks and from the right for right blocks,
    which is consistent for any even chain length.
    """

    def __init__(self, x, mu, theta=0.0):
        self.x, self.mu, self.theta = x, mu, theta
        self.Z = sp.csr_matrix(np.diag([1.0, -1.0]))
        self.sp = sp.csr_matrix(np.array([[0.0, 1.0], [0.0, 0.0]]))
        self.q = np.array([1, -1])      # U(1) charge in units of 1/2: Z
        self.d = 2

    @classmethod
    def from_continuum(cls, x, m_over_e, theta=0.0):
        """Use the continuum mass m/e, with the lattice mass shift m_lat = m - e^2 a/8."""
        return cls(x, 2 * np.sqrt(x) * m_over_e - 0.25, theta)

    def _site(self, parity):
        Q = 0.5 * (self.Z + parity * _eye(2))
        return {"mass": 0.5 * self.mu * parity * self.Z, "Q": Q, "S+": self.sp}

    def _hop(self, a_sp, b_sp):
        return self.x * (_kron(a_sp, b_sp.T) + _kron(a_sp.T, b_sp))

    def initial_block(self, side):
        if side == "left":     # site 0, parity +1; owns link 0
            s = self._site(+1)
            Qtot = s["Q"] + self.theta / (2 * np.pi) * _eye(2)
            ops = {"H": s["mass"] + Qtot @ Qtot, "Qtot": Qtot, "S+": s["S+"]}
            return Block("left", 1, ops, self.q.copy())
        s = self._site(-1)     # last site of an even chain has parity -1; owns no links
        ops = {"H": s["mass"], "P": sp.csr_matrix((2, 2)), "S+": s["S+"]}
        return Block("right", 1, ops, self.q.copy(), {"links": 0})

    def enlarge(self, block, side):
        I_b, I_s = _eye(block.dim), _eye(2)
        if side == "left":
            s = self._site((-1) ** block.size)                 # new site index = block.size
            Qtot = _kron(block.ops["Qtot"], I_s) + _kron(I_b, s["Q"])
            H = (_kron(block.ops["H"], I_s) + _kron(I_b, s["mass"])
                 + self._hop(block.ops["S+"], s["S+"]) + Qtot @ Qtot)
            ops = {"H": H, "Qtot": Qtot, "S+": _kron(I_b, s["S+"])}
            return Block("left", block.size + 1, ops, _combine(block.charges, self.q))
        s = self._site(-(-1) ** block.size)                    # distance block.size from right
        n = block.meta["links"]
        Q = s["Q"]
        H = (_kron(I_s, block.ops["H"]) + _kron(s["mass"], I_b)
             + self._hop(s["S+"], block.ops["S+"])
             + (n + 1) * _kron(Q @ Q, I_b) + 2 * _kron(Q, block.ops["P"]))
        P = (n + 1) * _kron(Q, I_b) + _kron(I_s, block.ops["P"])
        ops = {"H": H, "P": P, "S+": _kron(s["S+"], I_b)}
        return Block("right", block.size + 1, ops, _combine(self.q, block.charges), {"links": n + 1})

    def superblock(self, L, R):
        Qt = L.ops["Qtot"]
        H = _kron(L.ops["H"], _eye(R.dim)) + _kron(_eye(L.dim), R.ops["H"])
        H = H + self._hop(L.ops["S+"], R.ops["S+"])
        return H + R.meta["links"] * _kron(Qt @ Qt, _eye(R.dim)) + 2 * _kron(Qt, R.ops["P"])
