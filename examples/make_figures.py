"""
Regenerate the figures in figures/. Expensive DMRG runs are cached in examples/data/*.json;
delete a file there to recompute it. Run from the repository root:

    python examples/make_figures.py
"""
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
from dmrg import SchwingerModel, XXZChain, exact, finite_dmrg, infinite_dmrg

DATA = os.path.join(os.path.dirname(__file__), "data")
OUT = os.path.join(ROOT, "figures")
os.makedirs(DATA, exist_ok=True)
os.makedirs(OUT, exist_ok=True)

PAPER, INK, MUTED, LINE = "#f3eee3", "#1c2130", "#6a6556", "#d8cfbd"
ORANGE, TEAL, OCHRE, BRICK = "#c4561d", "#2b7a6f", "#b5862a", "#b8452f"
plt.rcParams.update({
    "figure.facecolor": PAPER, "axes.facecolor": PAPER, "savefig.facecolor": PAPER,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
    "grid.color": LINE, "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10.5, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "legend.frameon": False, "lines.linewidth": 1.5, "mathtext.fontset": "dejavusans",
})


def cached(name, compute):
    path = os.path.join(DATA, name + ".json")
    if os.path.exists(path):
        return json.load(open(path))
    t = time.time()
    data = compute()
    json.dump(data, open(path, "w"), indent=1)
    print(f"computed {name} in {time.time() - t:.0f} s")
    return json.load(open(path))          # same (string-keyed) form as a cached read


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote figures/" + name)


# ---------------------------------------------------------------- XY benchmark ---
def xy_data():
    model = XXZChain(Delta=0.0)
    r = infinite_dmrg(model, 100, m=64)
    Ls = [h[0] for h in r.history]
    Es = [h[1][0] for h in r.history]
    density = list(np.diff(Es) / np.diff(Ls))

    L, e_exact = 100, exact.xy_ground_energy(100)
    errors, timing = {}, {}
    for m in (8, 16, 32, 48, 64):
        res = finite_dmrg(model, L, m=m, sweeps=2)
        errors[m] = [abs(res.energies[0] - e_exact) / abs(e_exact), res.truncation_error]
    for m in (16, 32, 48, 64):
        row = []
        for sym in (False, True):
            t = time.time()
            infinite_dmrg(model, L, m=m, use_symmetry=sym)
            row.append(time.time() - t)
        timing[m] = row
    return {"L": Ls[1:], "density": density, "errors": errors, "timing": timing}


def fig_xy():
    d = cached("xy_benchmark", xy_data)
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8), gridspec_kw={"wspace": 0.3})

    ax = axs[0]
    ax.plot(d["L"], d["density"], color=TEAL, label="infinite DMRG, m = 64")
    ax.axhline(-1 / np.pi, color=ORANGE, ls=(0, (4, 3)), label=r"exact $-1/\pi$")
    ax.set_ylim(-0.3195, -0.3145)
    ax.set_xlabel("chain length L")
    ax.set_ylabel("energy per site (bulk)")
    ax.set_title("XY chain: energy density")
    ax.legend(loc="upper right")

    ax = axs[1]
    ms = sorted(int(k) for k in d["errors"])
    ax.semilogy(ms, [d["errors"][str(m)][0] for m in ms], "o-", color=ORANGE, label="relative energy error")
    ax.semilogy(ms, [d["errors"][str(m)][1] for m in ms], "s--", color=TEAL, label="discarded weight")
    ax.set_xlabel("bond dimension m")
    ax.set_title("L = 100, against free fermions")
    ax.legend(loc="lower left")

    ax = axs[2]
    ms = sorted(int(k) for k in d["timing"])
    t_no = [d["timing"][str(m)][0] for m in ms]
    t_yes = [d["timing"][str(m)][1] for m in ms]
    w = 5
    ax.bar(np.array(ms) - w / 2, t_no, w, color=MUTED, label="full superblock")
    ax.bar(np.array(ms) + w / 2, t_yes, w, color=TEAL, label=r"restricted to $S^z_{tot} = 0$")
    for m, a, b in zip(ms, t_no, t_yes):
        ax.text(m + w / 2, b, f"{a / b:.1f}x", ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_xlabel("bond dimension m")
    ax.set_ylabel("time to grow to L = 100 [s]")
    ax.set_title("Speed-up from charge conservation")
    ax.legend(loc="upper left")
    save(fig, "xy_benchmark.png")


# ------------------------------------------------------- Schwinger criticality ---
X, MC = 10.0, 0.3340985          # m_c(x = 10)/e from arXiv:2412.01902, Table I (criterion C1)
MASSES = {"0.20": 0.20, "m_c": MC, "0.50": 0.50}
SIZES = (20, 40, 60, 80)


def c_estimate(S, N):
    """Arguello Cruz et al. eq. (25) with N_L = N/2, N'_L = N/2 - 4 (open chain, kappa = 1)."""
    return 6 * (S[N // 2] - S[N // 2 - 4]) / np.log(1 / np.sin(np.pi * (N / 2 - 4) / N))


def schwinger_data():
    out = {"profiles": {}, "c": {}, "ratio": {}}
    for label, me in MASSES.items():
        out["c"][label] = []
        for N in SIZES:
            r = finite_dmrg(SchwingerModel.from_continuum(X, me, np.pi), N, m=48, sweeps=3)
            S = {int(k): v for k, v in r.entropy_profile.items()}
            out["c"][label].append(c_estimate(S, N))
            if N == SIZES[-1]:
                out["profiles"][label] = S
    for N in SIZES[:3]:
        r = finite_dmrg(SchwingerModel.from_continuum(X, MC, np.pi), N, m=48, sweeps=3, num_states=3)
        E = r.energies
        out["ratio"][N] = (E[2] - E[0]) / (E[1] - E[0])
    for N in (8, 10, 12, 14):     # exact diagonalization for the smallest sizes
        mod = SchwingerModel.from_continuum(X, MC, np.pi)
        E = exact.exact_ground_states(exact.schwinger_hamiltonian(N, X, mod.mu, np.pi),
                                      exact.total_charge(N), 0, k=3)
        out["ratio"][N] = (E[2] - E[0]) / (E[1] - E[0])
    return out


def fig_schwinger():
    d = cached("schwinger_criticality", schwinger_data)
    colors = {"0.20": OCHRE, "m_c": ORANGE, "0.50": TEAL}
    names = {"0.20": "m/e = 0.20", "m_c": r"m/e = $m_c$ = 0.33410", "0.50": "m/e = 0.50"}
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.9), gridspec_kw={"wspace": 0.3})

    ax = axs[0]
    N = SIZES[-1]
    for label in MASSES:
        S = d["profiles"][label]
        l = np.array(sorted(int(k) for k in S))
        s = np.array([S[str(k)] for k in l])
        chord = np.log(2 * N / np.pi * np.sin(np.pi * l / N))
        ax.plot(chord, s, ".", color=colors[label], ms=5, label=names[label])
    ref = np.linspace(1.5, np.log(2 * N / np.pi), 10)
    s_c = np.array([d["profiles"]["m_c"][str(N // 2)]])
    ax.plot(ref, s_c - (np.log(2 * N / np.pi) - ref) / 12, color=INK, lw=1, ls=(0, (4, 3)),
            label="slope c/6, c = 1/2")
    ax.set_xlabel(r"$\log\,[\frac{2N}{\pi}\sin(\pi\ell/N)]$")
    ax.set_ylabel(r"entanglement entropy $S(\ell)$")
    ax.set_title(f"Entanglement profile (N = {N}, x = 10)")
    ax.legend(loc="upper left", fontsize=9)

    ax = axs[1]
    inv = 1 / np.array(SIZES)
    for label in MASSES:
        ax.plot(inv, d["c"][label], "o-", color=colors[label], label=names[label])
    ax.axhline(0.5, color=INK, lw=1, ls=(0, (4, 3)))
    ax.text(0.003, 0.52, "Ising: c = 1/2", fontsize=9)
    ax.set_xlim(0, None)
    ax.set_xlabel("1 / N")
    ax.set_ylabel("central-charge estimator")
    ax.set_title("Central-charge estimator")
    ax.legend(loc="upper left", fontsize=9)

    ax = axs[2]
    Ns = sorted(int(k) for k in d["ratio"] if int(k) >= 14)   # smaller lattices: lattice artefacts dominate
    ax.plot(1 / np.array(Ns), [d["ratio"][str(n)] for n in Ns], "o-", color=ORANGE)
    ax.axhline(1.5, color=INK, lw=1, ls=(0, (4, 3)))
    ax.text(0.003, 1.493, r"Ising (++) boundary: $\Delta_2/\Delta_1 = 3/2$", fontsize=9)
    ax.set_xlim(0, None)
    ax.set_xlabel("1 / N")
    ax.set_ylabel(r"$(E_2 - E_0)/(E_1 - E_0)$")
    ax.set_title(r"Gap ratio at $m_c$")
    save(fig, "schwinger_criticality.png")


if __name__ == "__main__":
    fig_xy()
    fig_schwinger()
