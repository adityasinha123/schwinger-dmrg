"""
White's density-matrix renormalization group in its original block form.

Each step joins  [left block] + site + site + [right block]  into a superblock, finds its
lowest eigenstate(s) with Lanczos, and keeps the m most probable states of the reduced density
matrix of one half. With a conserved U(1) charge the superblock is restricted to the target
sector and the density matrix is diagonalized sector by sector, which keeps every block basis
state labelled by its charge.

Charges are integers in units of the model's local charge labels (2 S^z for spin chains,
Z for the Schwinger model), so `target` is 2 S^z_tot.

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
    entropy_profile: dict = field(default_factory=dict)   # cut l -> ground-state S(l)
    truncation_error: float = 0.0
    history: list = field(default_factory=list)          # (chain length or cut, energies)


def _lowest(H, k):
    n = H.shape[0]
    if n <= max(64, 3 * k):
        w, v = np.linalg.eigh(H.toarray())
        return w[:k], v[:, :k]
    w, v = eigsh(H, k=k, which="SA", tol=1e-12)
    order = np.argsort(w)
    return w[order], v[:, order]


def _entropy(M):
    """Von Neumann entropy (natural log) of the bipartition encoded by the matrix M[i, j]."""
    p = np.linalg.svd(M, compute_uv=False) ** 2
    p = p[p > 1e-16]
    return float(-np.sum(p * np.log(p)))


def _sector_target(model, M, L, target):
    """
    Charge to target on a chain of M sites while working towards `target` at length L: the
    value reachable by M sites that is closest to target * M / L. Equal to `target` at M = L.
    """
    if target is None:
        return None
    q = np.unique(model.q)
    step = int(np.gcd.reduce(np.diff(q))) if len(q) > 1 else 1
    lo, hi = M * int(q.min()), M * int(q.max())
    if M == L:
        if not (lo <= target <= hi and (target - lo) % step == 0):
            raise ValueError(f"charge {target} cannot occur on a chain of {L} sites")
        return target
    val = lo + step * round((target * M / L - lo) / step)
    return int(min(max(val, lo), hi))


def _solve(model, L, R, target, num_states, strict=True):
    """
    Lowest states of the superblock of enlarged blocks L, R as matrices psi[k][iL, iR].
    If the target sector holds fewer than num_states states this is an error when `strict`,
    otherwise (warm-up on short chains) fewer states are returned.
    """
    H = model.superblock(L, R)
    if target is None:
        idx = np.arange(H.shape[0])
    else:
        idx = np.flatnonzero(np.add.outer(L.charges, R.charges).ravel() == target)
        if idx.size == 0:
            raise ValueError(f"the superblock has no states with charge {target}")
        H = H[idx][:, idx]
    k = num_states
    if idx.size < num_states:
        if strict:
            raise ValueError(f"asked for {num_states} states but the sector has only {idx.size}")
        k = idx.size
    E, vecs = _lowest(sp.csr_matrix(H), k)
    psis = []
    for j in range(vecs.shape[1]):
        full = np.zeros(L.dim * R.dim)
        full[idx] = vecs[:, j]
        psis.append(full.reshape(L.dim, R.dim))
    return E, psis


def _truncate(block, psis, m, use_symmetry):
    """
    Keep the m largest eigenvectors of the reduced density matrix of `block`, averaged over
    the target states. Returns the rotated block, the ground-state entanglement entropy of
    the cut, and the discarded weight.
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

    candidates.sort(key=lambda c: -c[0])
    keep = candidates[:min(m, len(candidates))]
    O = sp.csr_matrix(np.column_stack([c[1] for c in keep]))
    error = float(max(0.0, 1.0 - sum(c[0] for c in keep)))
    charges = np.array([c[2] for c in keep]) if keep[0][2] is not None else None

    ops = {k: sp.csr_matrix(O.T @ op @ O) for k, op in block.ops.items()}
    new = Block(block.side, block.size, ops, charges, dict(block.meta))
    return new, _entropy(psis[0]), error


def infinite_dmrg(model, L, m, target=0, num_states=1, use_symmetry=True, store=None):
    """
    Grow an open chain to length L (even, >= 4) by adding two sites per step. Shorter chains
    target the reachable charge closest to the proportional share of `target`. Returns a
    DMRGResult whose history lists (chain length, energies). If `store` is a dict, the
    truncated left and right blocks of every size are put in it (used by finite_dmrg).
    """
    if L % 2 or L < 4:
        raise ValueError("L must be even and at least 4")
    target = target if use_symmetry else None
    left, right = model.initial_block("left"), model.initial_block("right")
    if store is not None:
        store[("left", 1)], store[("right", 1)] = left, right
    result = DMRGResult(energies=None)
    while 2 * left.size + 2 <= L:
        Le, Re = model.enlarge(left, "left"), model.enlarge(right, "right")
        M = Le.size + Re.size
        E, psis = _solve(model, Le, Re, _sector_target(model, M, L, target), num_states,
                         strict=(M == L))
        left, S, err = _truncate(Le, psis, m, use_symmetry)
        right, _, _ = _truncate(Re, psis, m, use_symmetry)
        result.history.append((M, E))
        result.entropy_profile = {Le.size: S}
        result.energies, result.truncation_error = E, err
        if store is not None:
            store[("left", left.size)], store[("right", right.size)] = left, right
    return result


def finite_dmrg(model, L, m, sweeps=3, target=0, num_states=1, use_symmetry=True, m_warmup=None):
    """
    Finite-system DMRG on an open chain of L (even, >= 6) sites: an infinite-system warm-up
    to length L, then a fixed number of full sweeps (centre -> right end -> left end ->
    centre). The truncation error is reported as a diagnostic; convergence in m and sweeps
    is up to the caller.

    Returns the energies in the symmetric configuration at the end of the last sweep, and
    the ground-state entanglement entropy S(l) of every cut l = 1..L-1 (sites 0..l-1 vs the
    rest), recorded during the last sweep.
    """
    if L % 2 or L < 6:
        raise ValueError("L must be even and at least 6")
    if sweeps < 1:
        raise ValueError("sweeps must be at least 1")
    target = target if use_symmetry else None
    blocks = {}
    infinite_dmrg(model, L, m_warmup or m, target, num_states, use_symmetry, store=blocks)

    d = model.d
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
                # the outermost cuts: a size-1 block is the bare site basis
                if Le.size == 2:
                    result.entropy_profile[1] = _entropy(psis[0].reshape(d, -1))
                if Re.size == 2:
                    result.entropy_profile[L - 1] = _entropy(psis[0].reshape(-1, d))
            if sys_side == "left" and 2 * sys.size == L:
                result.energies, result.truncation_error = E, err
                break
    result.entropy_profile = dict(sorted(result.entropy_profile.items()))
    return result
