# Phase 3 report: reruns each test in tests/test_solver.py unchanged, capturing every run() Result, and prints relative residuals, margins and T3 wall-clock.
import sys
import pathlib
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import floodsolver.solver as solver  # noqa: E402
import test_solver as T  # noqa: E402

captured = []
_run = solver.run


def _capturing_run(*args, **kwargs):
    res = _run(*args, **kwargs)
    captured.append((args, kwargs, res))
    return res


solver.run = _capturing_run


def rel(res):
    b = res.budget
    return abs(b["residual"]) / b["V_ref"]


tests = [
    ("T1", T.test_T1_mass_conservation), ("T2", T.test_T2_symmetry),
    ("T3", T.test_T3_tilted_plane_steady_state), ("T4a", T.test_T4a_bowl_flat_surface),
    ("T4b", T.test_T4b_bowl_no_overshoot), ("T5", T.test_T5_dry_ridge),
    ("T6", T.test_T6_no_uphill_wetting), ("T7", T.test_T7_dirichlet_surge),
    ("T8", T.test_T8_culvert),
]
rows = []
for name, fn in tests:
    captured.clear()
    t0 = time.perf_counter()
    fn()
    wall = time.perf_counter() - t0
    for k, (args, kwargs, res) in enumerate(captured):
        label = name if len(captured) == 1 else "%s (run %d%s)" % (name, k + 1, ", no link" if k else ", with link")
        b = res.budget
        print("%-22s rel residual %.2e   clipped %.1e m3   wall %.2f s" % (label, rel(res), b["clipped_volume"], wall))
        rows.append((name, k, args, kwargs, res))

print()
get = {(n, k): (a, kw, r) for n, k, a, kw, r in rows}

# T3 outlet column and outflow margin
a, kw, r = get[("T3", 0)]
z, n, dx = a[0], a[1], a[2]
x = (np.arange(z.shape[1]) + 1) * dx
rr = 50e-3 / 3600
h_an = T.tilted_plane_steady_h(x, rr, 0.03, 0.01)
err = np.abs(r.h_final - h_an[None, :]) / h_an[None, :]
print("T3  i = nx-1 relative error: %.4f%%   max error over 8 <= i <= nx-2: %.3f%% (limit 5%%)"
      % (100 * err[:, -1].max(), 100 * err[:, 8:-1].max()))
ts = np.array([t for t, _ in r.outflow_series]); Qs = np.array([q for _, q in r.outflow_series])
t0s = np.concatenate(([0.0], ts[:-1])); w0 = 4 * 3600 - 600; s = ts > w0
Qa = np.sum(Qs[s] * (ts[s] - np.maximum(t0s[s], w0))) / 600
Qe = rr * z.size * dx ** 2
print("T3  outflow last 10 min %.4f m3/s vs %.4f -> %.3f%% (limit 2%%)" % (Qa, Qe, 100 * abs(Qa - Qe) / Qe))

# T4a lake flatness margin
a, kw, r = get[("T4a", 0)]
z = a[0]; h = r.h_final; w = h > 0.01
print("T4a final std(eta[wet]) = %.3e m (limit 1e-3), wet cells %d" % (np.std((z + h)[w]), w.sum()))

# T4b margins
a, kw, r = get[("T4b", 0)]
z = a[0]; h = r.h_final; w = h > 0.01; eta_eq = np.mean((z + h)[w]); h0 = kw["h0"]
late = max(np.max((z + hs)[hs > 0.01]) for t, hs in r.snapshots if t >= 3600)
print("T4b eta_eq %.4f; max late snapshot eta %.4f vs bound %.4f; ever-wet eta_max %.4f vs initial wet max %.4f; final std %.3e (limit 1e-3)"
      % (eta_eq, late, eta_eq + 0.10, r.eta_max[r.h_max > 0.01].max(), (z + h0)[h0 > 0].max(), np.std((z + h)[w])))

# T7 margins
a, kw, r = get[("T7", 0)]
z = a[0]; h = r.h_final; w = h > 0.01
print("T7  final std(eta[wet]) %.3e (limit 2e-3), |mean-1| %.3e (limit 2e-3)" % (np.std((z + h)[w]), abs(np.mean((z + h)[w]) - 1)))

# T8 basin difference
a, kw, r = get[("T8", 0)]
z = a[0]; e = z + r.h_final
W = (slice(54, 74), slice(20, 40)); E = (slice(54, 74), slice(88, 108))
b = r.budget
print("T8  final |mean eta west - east| = %.3e m (limit 1e-3); culvert ab %.1f m3, ba %.1f m3"
      % (abs(e[W].mean() - e[E].mean()), b["culvert_volume_ab"], b["culvert_volume_ba"]))
