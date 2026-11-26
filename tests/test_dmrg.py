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


# --- charge sectors other than zero --------------------------------------------------------
def test_nonzero_sector_with_field():
    """A field moves the global ground state out of S^z = 0; DMRG must find both answers."""
    N, h = 10, 0.8
    model = XXZChain(Delta=1.0, h=-h)
    H = exact.xxz_hamiltonian(N, Delta=1.0, h=-h)
    q = exact.total_charge(N)
    for target in (0, 2, 4):
        r = finite_dmrg(model, N, m=64, sweeps=2, target=target)
        assert r.energies[0] == pytest.approx(exact.exact_ground_states(H, q, target)[0], abs=1e-9)
    r = finite_dmrg(model, N, m=64, sweeps=2, use_symmetry=False)
    assert r.energies[0] == pytest.approx(exact.exact_ground_states(H)[0], abs=1e-9)
    assert exact.exact_ground_states(H)[0] < exact.exact_ground_states(H, q, 0)[0] - 1e-3


def test_target_unreachable_by_first_warmup_chain():
    """2 S^z = 6 does not exist on 4 sites but does on 8: the warm-up must cope."""
    N = 8
    r = finite_dmrg(XXZChain(Delta=1.0), N, m=32, sweeps=2, target=6)
    H = exact.xxz_hamiltonian(N, Delta=1.0)
    assert r.energies[0] == pytest.approx(exact.exact_ground_states(H, exact.total_charge(N), 6)[0], abs=1e-9)


# --- truncation actually happening ---------------------------------------------------------
def test_schwinger_without_symmetry_is_variational_under_truncation():
    """
    Without charge conservation the kept basis can mix charges. The energy must still be an
    upper bound on the exact one (it is <psi|H|psi> for a real state) and close to it.
    """
    N, x, mu, theta = 10, 4.0, 0.6, np.pi
    E0 = exact.exact_ground_states(exact.schwinger_hamiltonian(N, x, mu, theta))[0]
    r = finite_dmrg(SchwingerModel(x, mu, theta), N, m=12, sweeps=3, use_symmetry=False)
    assert r.truncation_error > 0
    assert r.energies[0] >= E0 - 1e-10
    assert r.energies[0] == pytest.approx(E0, abs=5e-3)


def test_multi_state_entropy_is_the_ground_state_entropy():
    """
    With three target states, truncation uses their averaged density matrix, but the reported
    S(l) must be that of the ground state, for every cut l = 1..N-1, also when truncating.
    """
    N, x, mu, theta = 12, 4.0, 1.0, np.pi
    H = exact.schwinger_hamiltonian(N, x, mu, theta)
    q = exact.total_charge(N)
    ee = exact.entanglement_profile(H, N, q, 0)
    r = finite_dmrg(SchwingerModel(x, mu, theta), N, m=24, sweeps=3, num_states=3)
    assert r.truncation_error > 0
    assert sorted(r.entropy_profile) == list(range(1, N))
    for l in range(1, N):
        assert r.entropy_profile[l] == pytest.approx(ee[l], abs=2e-3)
    np.testing.assert_allclose(r.energies, exact.exact_ground_states(H, q, 0, k=3), atol=1e-3)


# --- input validation -----------------------------------------------------------------------
def test_invalid_inputs_raise():
    model = XXZChain(Delta=1.0)
    with pytest.raises(ValueError):
        infinite_dmrg(model, 5, m=8)
    with pytest.raises(ValueError):
        finite_dmrg(model, 7, m=8)
    with pytest.raises(ValueError):
        finite_dmrg(model, 8, m=8, sweeps=0)
    with pytest.raises(ValueError):
        finite_dmrg(model, 8, m=8, target=10)             # 2 S^z = 10 impossible on 8 sites
    with pytest.raises(ValueError):
        finite_dmrg(model, 6, m=8, target=6, num_states=2)  # that sector has one state


def test_schwinger_left_block_in_charge_mixing_basis():
    """
    Rotate a 3-site left block into a random 5-state basis that mixes charges, enlarge it by a
    site, and compare with projecting the exact 4-site left Hamiltonian (all terms on sites
    0..3, Coulomb energy on links 0..3) into the same basis. This needs Qtot^2 carried
    separately: squaring the projected Qtot would be wrong here.
    """
    import scipy.sparse as sp
    from dmrg.models import Block
    x, mu, theta = 3.0, 0.7, np.pi
    model = SchwingerModel(x, mu, theta)
    blk = model.initial_block("left")
    for _ in range(2):
        blk = model.enlarge(blk, "left")                       # 3 sites, exact basis
    rng = np.random.default_rng(0)
    O, _ = np.linalg.qr(rng.normal(size=(8, 5)))               # isometry mixing all charges
    O = sp.csr_matrix(O)
    rot = Block("left", 3, {k: sp.csr_matrix(O.T @ op @ O) for k, op in blk.ops.items()}, None)
    got = model.enlarge(rot, "left").ops["H"].toarray()

    n = 4
    Z = [exact._site_op(sp.csr_matrix(np.diag([1.0, -1.0])), i, n, 2) for i in range(n)]
    S = [exact._site_op(sp.csr_matrix(np.array([[0.0, 1.0], [0.0, 0.0]])), i, n, 2) for i in range(n)]
    I = sp.identity(2**n, format="csr")
    H = sp.csr_matrix((2**n, 2**n))
    field = theta / (2 * np.pi) * I
    for k in range(n):                                         # links 0..3 belong to the block
        field = field + 0.5 * (Z[k] + (-1) ** k * I)
        H = H + field @ field + 0.5 * mu * (-1) ** k * Z[k]
    for k in range(n - 1):
        H = H + x * (S[k] @ S[k + 1].T + S[k].T @ S[k + 1])
    P = sp.kron(O, sp.identity(2), format="csr")
    np.testing.assert_allclose(got, (P.T @ H @ P).toarray(), atol=1e-12)
