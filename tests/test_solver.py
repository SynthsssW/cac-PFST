"""Tests T1–T8 from SOLVER_SPEC.md §7. Written before floodsolver/solver.py.

Tolerances are copied verbatim from §7. Where §7 leaves a parameter or a piece of
geometry unstated, the choice made here is marked `SPEC-GAP:`.

The solver is imported inside each test so that, before solver.py exists, every
test fails individually with ModuleNotFoundError rather than one collection error.
"""
import numpy as np

from floodsolver.analytic import tilted_plane_steady_h

# §7 common parameters
G = 9.81
DX = 10.0
ALPHA = 0.7
H_MIN = 1e-3
NX = NY = 128
RAIN_50MM_HR = 50e-3 / 3600.0  # 50 mm/hr = 1.3889e-5 m/s

# SPEC-GAP: §7 gives Manning n only for T3 (0.03). The same value is used for all tests.
N_MANNING = 0.03

CLOSED = {"N": "closed", "S": "closed", "E": "closed", "W": "closed"}


def _rel_residual(budget):
    """§6: abs(residual) / max(rain_volume, 1e-12)."""
    return abs(budget["residual"]) / max(budget["rain_volume"], 1e-12)


def _coords(ny, nx):
    """Cell-centre coordinates x = i*dx, y = j*dx (m)."""
    x = np.arange(nx) * DX
    y = np.arange(ny) * DX
    return np.meshgrid(x, y)  # X[j, i], Y[j, i]


def _bowl():
    """§7 T2 paraboloid bowl on a 129x129 grid, centre exactly at cell (64, 64)."""
    nx = ny = 129
    X, Y = _coords(ny, nx)
    xc = yc = 64 * DX
    return 5e-4 * ((X - xc) ** 2 + (Y - yc) ** 2)


def _dilate(mask):
    """3x3 (8-neighbour) binary dilation, pure numpy."""
    p = np.pad(mask, 1, constant_values=False)
    ny, nx = mask.shape
    out = np.zeros_like(mask)
    for dj in range(3):
        for di in range(3):
            out |= p[dj:dj + ny, di:di + nx]
    return out


def _erode(mask):
    return ~_dilate(~mask)


# ---------------------------------------------------------------------------
# T1 mass conservation
# ---------------------------------------------------------------------------
def test_T1_mass_conservation():
    from floodsolver.solver import run

    X, Y = _coords(NY, NX)
    Lx, Ly = NX * DX, NY * DX
    # SPEC-GAP: "random smooth terrain (sum of 3 low-frequency sines, amplitude 2 m)".
    # Fixed seed; integer wavenumbers 1..3 cycles per domain; random phases;
    # the sum is scaled so its peak absolute value is exactly 2 m.
    rng = np.random.default_rng(20261023)
    s = np.zeros((NY, NX))
    for _ in range(3):
        kx, ky = rng.integers(1, 4, size=2)
        phi = rng.uniform(0.0, 2.0 * np.pi)
        s += np.sin(2.0 * np.pi * (kx * X / Lx + ky * Y / Ly) + phi)
    z = 2.0 * s / np.abs(s).max() + 0.001 * X
    n = np.full((NY, NX), N_MANNING)

    t_rain = 3600.0
    res = run(z, n, DX, lambda t: RAIN_50MM_HR if t < t_rain else 0.0, t_rain + 7200.0,
              edges=CLOSED, alpha=ALPHA, h_min=H_MIN, rain_breakpoints=[t_rain])

    h = res.h_final
    rain_volume = RAIN_50MM_HR * t_rain * NX * NY * DX ** 2
    assert _rel_residual(res.budget) < 1e-10
    assert h.min() >= 0
    assert res.budget["clipped_volume"] == 0.0
    assert abs(h.sum() * DX ** 2 - rain_volume) / rain_volume < 1e-10


# ---------------------------------------------------------------------------
# T2 symmetry
# ---------------------------------------------------------------------------
def test_T2_symmetry():
    from floodsolver.solver import run

    z = _bowl()
    n = np.full(z.shape, N_MANNING)
    t_rain = 3600.0
    res = run(z, n, DX, lambda t: RAIN_50MM_HR if t < t_rain else 0.0, t_rain + 3600.0,
              edges=CLOSED, alpha=ALPHA, h_min=H_MIN, rain_breakpoints=[t_rain])

    h = res.h_final
    assert np.allclose(h, h[::-1, :], atol=1e-9)
    assert np.allclose(h, h[:, ::-1], atol=1e-9)
    assert np.allclose(h, h.T, atol=1e-9)


# ---------------------------------------------------------------------------
# T3 tilted plane steady state
# ---------------------------------------------------------------------------
def test_T3_tilted_plane_steady_state():
    from floodsolver.solver import run

    S0 = 0.01
    n_val = 0.03
    r = RAIN_50MM_HR
    L = (NX - 1) * DX
    X, _ = _coords(NY, NX)
    z = S0 * (L - X)
    n = np.full((NY, NX), n_val)
    t_end = 4 * 3600.0
    edges = {"N": "closed", "S": "closed", "E": "open", "W": "closed"}

    res = run(z, n, DX, lambda t: r, t_end, edges=edges, alpha=ALPHA, h_min=H_MIN)

    # (a) outflow rate, time-weighted average over the last 10 min.
    # SPEC-GAP: outflow_series entries are read as (end time of step, mean rate over that step).
    ts = np.array([t for t, _ in res.outflow_series], dtype=float)
    Qs = np.array([Q for _, Q in res.outflow_series], dtype=float)
    t_start = np.concatenate(([0.0], ts[:-1]))
    w0 = t_end - 600.0
    sel = ts > w0
    weights = ts[sel] - np.maximum(t_start[sel], w0)
    Q_avg = np.sum(Qs[sel] * weights) / 600.0
    Q_expected = r * L * NY * DX
    assert abs(Q_avg - Q_expected) / Q_expected < 0.02

    # (b) depth within 5% of analytic for all i >= 8, every row; rows identical to 1e-12.
    h = res.h_final
    x = np.arange(NX) * DX
    h_an = tilted_plane_steady_h(x, r, n_val, S0)
    assert np.abs(h - h[0:1, :]).max() <= 1e-12
    rel_err = np.abs(h[:, 8:] - h_an[None, 8:]) / h_an[None, 8:]
    assert np.all(rel_err < 0.05)

    # (c)
    assert _rel_residual(res.budget) < 1e-10


# ---------------------------------------------------------------------------
# T4a bowl flat surface
# ---------------------------------------------------------------------------
def test_T4a_bowl_flat_surface():
    from floodsolver.solver import run

    z = _bowl()
    n = np.full(z.shape, N_MANNING)
    t_rain = 3600.0
    r = 100e-3 / t_rain  # 100 mm total over 1 h
    res = run(z, n, DX, lambda t: r if t < t_rain else 0.0, t_rain + 6 * 3600.0,
              edges=CLOSED, alpha=ALPHA, h_min=H_MIN, rain_breakpoints=[t_rain])

    h = res.h_final
    eta = z + h
    wet = h > 0.01
    assert np.std(eta[wet]) < 1e-3

    # Wet footprint equals {z < eta_mean} up to a one-cell boundary ring.
    # SPEC-GAP: eta_mean = mean(eta[wet]); the ring is every cell within one cell
    # (8-neighbour) of the set's edge, i.e. dilate(A) & ~erode(A).
    A = z < eta[wet].mean()
    ring = _dilate(A) & ~_erode(A)
    mismatch = wet ^ A
    assert not np.any(mismatch & ~ring)

    assert _rel_residual(res.budget) < 1e-10


# ---------------------------------------------------------------------------
# T4b bowl no overshoot
# ---------------------------------------------------------------------------
def test_T4b_bowl_no_overshoot():
    from floodsolver.solver import run

    z = _bowl()
    n = np.full(z.shape, N_MANNING)
    h0 = np.zeros(z.shape)
    # SPEC-GAP: a 10x10 block cannot be centred on a 129 grid; rows/cols 60..69 used.
    h0[60:70, 60:70] = 2.0
    res = run(z, n, DX, lambda t: 0.0, 6 * 3600.0, h0=h0,
              edges=CLOSED, alpha=ALPHA, h_min=H_MIN)

    h = res.h_final
    eta = z + h
    assert res.h_max.max() <= 2.0 + 1e-9
    assert np.std(eta[h > 0.01]) < 1e-3


# ---------------------------------------------------------------------------
# T5 dry ridge
# ---------------------------------------------------------------------------
def test_T5_dry_ridge():
    from floodsolver.solver import run

    z = np.zeros((NY, NX))
    z[:, 63:65] = 1.0
    h0 = np.zeros((NY, NX))
    h0[:, 0:63] = 0.5
    n = np.full((NY, NX), N_MANNING)
    res = run(z, n, DX, lambda t: 0.0, 2 * 3600.0, h0=h0,
              edges=CLOSED, alpha=ALPHA, h_min=H_MIN)

    h = res.h_final
    eta = z + h
    assert h[:, 65:].max() == 0.0
    assert np.abs(eta[:, :63] - 0.5).max() < 1e-9


# ---------------------------------------------------------------------------
# T6 no uphill wetting
# ---------------------------------------------------------------------------
def test_T6_no_uphill_wetting():
    from floodsolver.solver import run

    z = np.full((NY, NX), 0.3)
    z[:, 0] = 0.0
    h0 = np.zeros((NY, NX))
    h0[:, 0] = 0.2
    n = np.full((NY, NX), N_MANNING)
    res = run(z, n, DX, lambda t: 0.0, 3600.0, h0=h0,
              edges=CLOSED, alpha=ALPHA, h_min=H_MIN)

    assert res.h_final[:, 1:].max() == 0.0


# ---------------------------------------------------------------------------
# T7 Dirichlet surge
# ---------------------------------------------------------------------------
def test_T7_dirichlet_surge():
    from floodsolver.solver import run

    z = np.zeros((NY, NX))
    n = np.full((NY, NX), N_MANNING)
    mask = np.zeros((NY, NX), dtype=bool)
    mask[:, 0] = True
    # SPEC-GAP: initial west column set to the surge level so "at all times" includes t = 0.
    h0 = np.zeros((NY, NX))
    h0[:, 0] = 1.0
    # SPEC-GAP: output interval is not stated; 60 s used. snapshots read as [(t, h), ...].
    res = run(z, n, DX, lambda t: 0.0, 2 * 3600.0, h0=h0,
              surge_mask=mask, surge_fn=lambda t: 1.0,
              edges=CLOSED, alpha=ALPHA, h_min=H_MIN, record_every=60.0)

    frames = [h for _, h in res.snapshots] + [res.h_final]
    assert len(frames) > 1
    for h in frames:
        assert np.all(h[:, 0] == 1.0)
        assert np.all(np.diff(h, axis=1) <= 0.0)
    assert _rel_residual(res.budget) < 1e-10


# ---------------------------------------------------------------------------
# T8 culvert
# ---------------------------------------------------------------------------
def _two_basins():
    """SPEC-GAP: basin geometry not stated in §7. Two flat 20x20-cell basins at
    z = 0 set into ground at z = 2 m, so the 2 m ridge separates them."""
    z = np.full((NY, NX), 2.0)
    west = (slice(54, 74), slice(30, 50))
    east = (slice(54, 74), slice(78, 98))
    z[west] = 0.0
    z[east] = 0.0
    return z, west, east


def test_T8_culvert():
    from floodsolver.solver import run

    z, west, east = _two_basins()
    n = np.full((NY, NX), N_MANNING)
    h0 = np.zeros((NY, NX))
    h0[west] = 1.0
    # (ja, ia, jb, ib, area_m2, C, z_invert): west-basin cell -> east-basin cell
    link = (64, 45, 64, 82, 1.0, 0.6, 0.0)
    t_end = 6 * 3600.0

    res = run(z, n, DX, lambda t: 0.0, t_end, h0=h0, culverts=[link],
              edges=CLOSED, alpha=ALPHA, h_min=H_MIN)
    eta = z + res.h_final
    # SPEC-GAP: "eta equalises across basins" read as the basin-mean surfaces.
    assert abs(eta[west].mean() - eta[east].mean()) < 1e-3

    res_no = run(z, n, DX, lambda t: 0.0, t_end, h0=h0, culverts=(),
                 edges=CLOSED, alpha=ALPHA, h_min=H_MIN)
    assert res_no.h_final[east].max() == 0.0
