"""
Independent reference results to check DMRG against.

    xy_ground_energy        open XY chain, exactly, via free fermions (Jordan-Wigner)
    exact_ground_states     brute-force diagonalization of a full chain Hamiltonian
    xxz_hamiltonian         full 2^N (or (2S+1)^N) matrix of the XXZ chain
    schwinger_hamiltonian   full 2^N matrix of the lattice Schwinger model

The full Hamiltonians are built directly from their definitions, sharing no code with the
block construction in models.py.
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

from .models import spin_operators


def xy_ground_energy(N, J=1.0):
    """
    H = (J/2) sum (S+_i S-_{i+1} + h.c.) on an open chain is free fermions with hopping J/2.
    Single-particle energies J cos(pi k/(N+1)); the ground state fills the negative ones.
    """
    eps = J * np.cos(np.pi * np.arange(1, N + 1) / (N + 1))
    return eps[eps < 0].sum()


def _site_op(op, i, N, d):
    out = sp.identity(1, format="csr")
    for j in range(N):
        out = sp.kron(out, op if j == i else sp.identity(d, format="csr"), format="csr")
    return out


def xxz_hamiltonian(N, J=1.0, Delta=1.0, h=0.0, S=0.5):
    sz, splus, _ = spin_operators(S)
    d = sz.shape[0]
    Sz = [_site_op(sz, i, N, d) for i in range(N)]
    Sp = [_site_op(splus, i, N, d) for i in range(N)]
    H = sum(h * Sz[i] for i in range(N))
    for i in range(N - 1):
        H = H + J * (0.5 * (Sp[i] @ Sp[i + 1].T + Sp[i].T @ Sp[i + 1]) + Delta * Sz[i] @ Sz[i + 1])
    return sp.csr_matrix(H)


def schwinger_hamiltonian(N, x, mu, theta=0.0):
    """2H/(e^2 a) of the staggered lattice Schwinger model with open ends (see models.py)."""
    Zs = sp.csr_matrix(np.diag([1.0, -1.0]))
    sps = sp.csr_matrix(np.array([[0.0, 1.0], [0.0, 0.0]]))
    Z = [_site_op(Zs, i, N, 2) for i in range(N)]
    Sp = [_site_op(sps, i, N, 2) for i in range(N)]
    I = sp.identity(2**N, format="csr")
    H = sp.csr_matrix((2**N, 2**N))
    field = theta / (2 * np.pi) * I
    for n in range(N - 1):
        field = field + 0.5 * (Z[n] + (-1) ** n * I)
        H = H + field @ field
    for n in range(N):
        H = H + 0.5 * mu * (-1) ** n * Z[n]
    for n in range(N - 1):
        H = H + x * (Sp[n] @ Sp[n + 1].T + Sp[n].T @ Sp[n + 1])
    return sp.csr_matrix(H)


def total_charge(N, S=0.5):
    """Diagonal of the total U(1) charge (twice total S^z) in the product basis."""
    q = spin_operators(S)[2]
    tot = np.zeros(1, dtype=int)
    for _ in range(N):
        tot = np.add.outer(tot, q).ravel()
    return tot


def exact_ground_states(H, charges=None, target=None, k=1):
    """Lowest k eigenvalues of H, optionally within the sector charges == target."""
    if target is not None:
        idx = np.flatnonzero(charges == target)
        H = H[idx][:, idx]
    if H.shape[0] <= 600:
        return np.linalg.eigvalsh(H.toarray())[:k]
    return np.sort(eigsh(H, k=k, which="SA", return_eigenvectors=False))


def entanglement_profile(H, N, charges=None, target=None, d=2):
    """Entanglement entropy S(l) of the exact ground state for every cut l = 1..N-1."""
    if target is not None:
        idx = np.flatnonzero(charges == target)
        w, v = np.linalg.eigh(H[idx][:, idx].toarray())
        psi = np.zeros(H.shape[0])
        psi[idx] = v[:, 0]
    else:
        w, v = np.linalg.eigh(H.toarray())
        psi = v[:, 0]
    out = {}
    for l in range(1, N):
        s = np.linalg.svd(psi.reshape(d**l, -1), compute_uv=False) ** 2
        s = s[s > 1e-16]
        out[l] = float(-np.sum(s * np.log(s)))
    return out
