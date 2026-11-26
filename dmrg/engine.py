"""
White's density-matrix renormalization group in its original block form.

Each step joins  [left block] + site + site + [right block]  into a superblock, finds its
lowest eigenstate(s) with Lanczos, and keeps the m most probable states of the reduced density
matrix of one half. With a conserved U(1) charge the superblock is restricted to the target
sector and the density matrix is diagonalized sector by sector, which keeps every block basis
state labelled by its charge.

    infinite_dmrg   grows a chain two sites at a time from the middle outwards
    finite_dmrg     warms up with infinite_dmrg, then sweeps back and forth at fixed length
"""
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

from .models import Block


@dataclass
class DMRGResult:
    energies: np.ndarray                 # final energies (num_states of them)
    entropy_profile: dict = field(default_factory=dict)   # cut position l -> S(l), last sweep
    truncation_error: float = 0.0
    history: list = field(default_factory=list)          # (chain length or cut, energies)


def _lowest(H, k, v0=None):
    n = H.shape[0]
    if n <= max(64, 3 * k):
        w, v = np.linalg.eigh(H.toarray())
        return w[:k], v[:, :k]
    w, v = eigsh(H, k=k, which="SA", v0=v0, tol=1e-12)
    order = np.argsort(w)
    return w[order], v[:, order]


def _solve(model, L, R, target, num_states):
    """Lowest states of the superblock of enlarged blocks L, R as matrices psi[k][iL, iR]."""
    H = model.superblock(L, R)
    if target is None:
        idx = np.arange(H.shape[0])
    else:
        idx = np.flatnonzero(np.add.outer(L.charges, R.charges).ravel() == target)
        H = H[idx][:, idx]
    E, vecs = _lowest(sp.csr_matrix(H), num_states)
    psis = []
    for j in range(vecs.shape[1]):
        full = np.zeros(L.dim * R.dim)
        full[idx] = vecs[:, j]
        psis.append(full.reshape(L.dim, R.dim))
    return E, psis


def _truncate(block, psis, m, use_symmetry):
    """
    Keep the m largest eigenvectors of the reduced density matrix of `block` (averaged over
    the target states). Returns the rotated block, the entanglement entropy and the error.
    """
    mats = psis if block.side == "left" else [p.T for p in psis]
    rho = sum(M @ M.conj().T for M in mats) / len(mats)

    candidates = []          # (weight, vector, charge)
    if use_symmetry and block.charges is not None:
        for q in np.unique(block.charges):
            sel = np.flatnonzero(block.charges == q)
            w, v = np.linalg.eigh(rho[np.ix_(sel, sel)])
            for wi, vi in zip(w, v.T):
                vec = np.zeros(block.dim)
                vec[sel] = vi
                candidates.append((wi, vec, q))
    else:
        w, v = np.linalg.eigh(rho)
        candidates = [(wi, vi, None) for wi, vi in zip(w, v.T)]

    weights = np.clip(np.array([c[0] for c in candidates]), 0, None)
    nz = weights[weights > 1e-16]
    entropy = float(-np.sum(nz * np.log(nz)))

    candidates.sort(key=lambda c: -c[0])
    keep = candidates[:min(m, len(candidates))]
    O = sp.csr_matrix(np.column_stack([c[1] for c in keep]))
    error = float(max(0.0, 1.0 - sum(c[0] for c in keep)))
    charges = np.array([c[2] for c in keep]) if keep[0][2] is not None else None

    ops = {k: sp.csr_matrix(O.T @ op @ O) for k, op in block.ops.items()}
    return Block(block.side, block.size, ops, charges, dict(block.meta)), entropy, error


def infinite_dmrg(model, L, m, target=0, num_states=1, use_symmetry=True, store=None):
    """
    Grow an open chain to length L (even) by adding two sites per step. Returns a DMRGResult
    whose history lists (chain length, energies). If `store` is a dict, the truncated left and
    right blocks of every size are put in it (used by finite_dmrg).
    """
    target = target if use_symmetry else None
    left, right = model.initial_block("left"), model.initial_block("right")
    if store is not None:
        store[("left", 1)], store[("right", 1)] = left, right
    result = DMRGResult(energies=None)
    while 2 * left.size + 2 <= L:
        Le, Re = model.enlarge(left, "left"), model.enlarge(right, "right")
        E, psis = _solve(model, Le, Re, target, num_states)
        left, S, err = _truncate(Le, psis, m, use_symmetry)
        right, _, _ = _truncate(Re, psis, m, use_symmetry)
        result.history.append((Le.size + Re.size, E))
        result.entropy_profile = {Le.size: S}
        result.energies, result.truncation_error = E, err
        if store is not None:
            store[("left", left.size)], store[("right", right.size)] = left, right
    return result


def finite_dmrg(model, L, m, sweeps=3, target=0, num_states=1, use_symmetry=True, m_warmup=None):
    """
    Finite-system DMRG on an open chain of L (even) sites: an infinite-system warm-up to
    length L, then `sweeps` full sweeps (centre -> right end -> left end -> centre).
    The energy is taken in the symmetric configuration at the end of the last sweep, and
    the entanglement entropy S(l) of every cut l (sites 0..l-1 vs the rest) is recorded
    during the last sweep.
    """
    if L % 2 or L < 6:
        raise ValueError("L must be even and at least 6")
    target = target if use_symmetry else None
    blocks = {}
    infinite_dmrg(model, L, m_warmup or m, 0 if target is None else target, num_states,
                  use_symmetry, store=blocks)

    result = DMRGResult(energies=None)
    sys_side, env_side = "left", "right"
    sys = blocks[("left", L // 2)]
    for sweep in range(1, sweeps + 1):
        while True:
            env = blocks[(env_side, L - sys.size - 2)]
            if env.size == 1:                       # reached an end: turn around
                sys, env = env, sys
                sys_side, env_side = env_side, sys_side
            Se, Ee = model.enlarge(sys, sys_side), model.enlarge(env, env_side)
            Le, Re = (Se, Ee) if sys_side == "left" else (Ee, Se)
            E, psis = _solve(model, Le, Re, target, num_states)
            sys, S, err = _truncate(Se, psis, m, use_symmetry)
            blocks[(sys_side, sys.size)] = sys
            result.history.append((Le.size, E))
            if sweep == sweeps:
                result.entropy_profile[Le.size] = S
            if sys_side == "left" and 2 * sys.size == L:
                result.energies, result.truncation_error = E, err
                break
    result.entropy_profile = dict(sorted(result.entropy_profile.items()))
    return result
