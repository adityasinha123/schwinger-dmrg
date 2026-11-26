"""DMRG checked against exact results."""
import numpy as np
import pytest

from dmrg import SchwingerModel, XXZChain, exact, finite_dmrg, infinite_dmrg


@pytest.mark.parametrize("N", [8, 12, 20, 40])
def test_xy_matches_free_fermions(N):
    """Exact while the bond dimension covers the Hilbert space; then limited by truncation."""
    r = finite_dmrg(XXZChain(Delta=0.0), N, m=48, sweeps=2)
    assert r.energies[0] == pytest.approx(exact.xy_ground_energy(N), rel=1e-8, abs=1e-10)


def test_xy_error_falls_with_bond_dimension():
    N, e = 40, exact.xy_ground_energy(40)
    errs = [abs(finite_dmrg(XXZChain(Delta=0.0), N, m=m, sweeps=2).energies[0] - e) for m in (8, 16, 32)]
    assert errs[0] > errs[1] > errs[2]


@pytest.mark.parametrize("Delta,S,N", [(1.0, 0.5, 12), (0.5, 0.5, 12), (1.0, 1.0, 8)])
def test_xxz_matches_exact_diagonalization(Delta, S, N):
    r = finite_dmrg(XXZChain(Delta=Delta, S=S), N, m=64, sweeps=2)
    H = exact.xxz_hamiltonian(N, Delta=Delta, S=S)
    e = exact.exact_ground_states(H, exact.total_charge(N, S), 0)[0]
    assert r.energies[0] == pytest.approx(e, abs=1e-9)


@pytest.mark.parametrize("theta", [0.0, np.pi, 1.0])
@pytest.mark.parametrize("x,mu", [(1.0, 0.0), (4.0, 1.5), (9.0, -0.3)])
def test_schwinger_matches_exact_diagonalization(theta, x, mu):
    N = 10
    r = finite_dmrg(SchwingerModel(x, mu, theta), N, m=64, sweeps=2)
    H = exact.schwinger_hamiltonian(N, x, mu, theta)
    q = exact.total_charge(N)
    assert r.energies[0] == pytest.approx(exact.exact_ground_states(H, q, 0)[0], abs=1e-9)
    ee = exact.entanglement_profile(H, N, q, 0)
    for l in range(2, N - 1):
        assert r.entropy_profile[l] == pytest.approx(ee[l], abs=1e-7)


def test_schwinger_excited_states_in_sector():
    N, x, mu, theta = 10, 4.0, 1.0, np.pi
    r = finite_dmrg(SchwingerModel(x, mu, theta), N, m=64, sweeps=3, num_states=3)
    H = exact.schwinger_hamiltonian(N, x, mu, theta)
    e = exact.exact_ground_states(H, exact.total_charge(N), 0, k=3)
    np.testing.assert_allclose(r.energies, e, atol=1e-8)


def test_symmetry_projection_gives_same_energy():
    N = 16
    a = finite_dmrg(XXZChain(Delta=0.0), N, m=32, sweeps=2, use_symmetry=True)
    b = finite_dmrg(XXZChain(Delta=0.0), N, m=32, sweeps=2, use_symmetry=False)
    assert a.energies[0] == pytest.approx(b.energies[0], abs=1e-9)


def test_xy_bulk_energy_density():
    """Energy per site in the bulk approaches -1/pi (the dissertation-era benchmark)."""
    r = infinite_dmrg(XXZChain(Delta=0.0), 200, m=32)
    (L1, E1), (L2, E2) = r.history[-2], r.history[-1]
    assert (E2[0] - E1[0]) / (L2 - L1) == pytest.approx(-1 / np.pi, rel=2e-4)


def test_heisenberg_bulk_energy_density():
    """Bethe ansatz: e0 = 1/4 - ln 2."""
    r = infinite_dmrg(XXZChain(Delta=1.0), 200, m=48)
    (L1, E1), (L2, E2) = r.history[-2], r.history[-1]
    assert (E2[0] - E1[0]) / (L2 - L1) == pytest.approx(0.25 - np.log(2), rel=1e-3)


def test_lattice_mass_shift():
    m = SchwingerModel.from_continuum(x=100.0, m_over_e=0.3)
    assert m.mu == pytest.approx(2 * 10 * 0.3 - 0.25)
