# v1.3: is the T4a residual slosh and the T8 basin imbalance at the test end time physical (dt-converged) or numerical? Time series at alpha 0.7 / 0.35 / 0.175.
import sys
import pathlib

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from floodsolver.solver import run  # noqa: E402

DX = 10.0
C = {"N": "closed", "S": "closed", "E": "closed", "W": "closed"}

# T4a: same bowl, rain 100 mm over 1 h, then drain; std(eta[wet]) every hour
N = 129
X = (np.arange(N) - 64) * DX
z = 2e-5 * (X[None, :] ** 2 + X[:, None] ** 2)
n = np.full((N, N), 0.03)
for a in [0.7, 0.35, 0.175]:
    res = run(z, n, DX, lambda t: 0.1 / 3600 if t < 3600 else 0.0, 7 * 3600.0, edges=C,
              rain_breakpoints=[3600.0], alpha=a, record_every=1800.0)
    row = []
    for t, h in res.snapshots:
        w = h > 0.01
        row.append("%.1fh:%.1e" % (t / 3600, np.std((z + h)[w])))
    print("T4a alpha=%.3f std(eta[wet]):  %s" % (a, "  ".join(row)))

# T8: west-minus-east basin mean eta every hour
NX = NY = 128
z8 = np.full((NY, NX), 2.0)
W = (slice(54, 74), slice(20, 40))
E = (slice(54, 74), slice(88, 108))
z8[W] = 0.0
z8[E] = 0.0
h0 = np.zeros((NY, NX))
h0[W] = 1.0
link = (63, 29, 63, 97, 1.0, 0.6, 0.0)
for a in [0.7, 0.35, 0.175]:
    res = run(z8, np.full((NY, NX), 0.03), DX, lambda t: 0.0, 6 * 3600.0, h0=h0, culverts=[link],
              edges=C, alpha=a, record_every=3600.0)
    row = []
    for t, h in res.snapshots:
        e = z8 + h
        row.append("%.0fh:%+.2e(a-b %+.2e)" % (t / 3600, e[W].mean() - e[E].mean(), e[63, 29] - e[63, 97]))
    b = res.budget
    print("T8 alpha=%.3f west-east mean:  %s  | ab=%.0f m3 ba=%.0f m3" % (a, "  ".join(row), b["culvert_volume_ab"], b["culvert_volume_ba"]))

# Ideal two-reservoir ODE (flat basins, no surface dynamics): time to reach 1e-3 m
A_b = 20 * 20 * DX ** 2
k = 0.6 * 1.0 * np.sqrt(2 * 9.81) / A_b
print("ideal reservoir ODE: head 1 m -> 1e-3 m in %.2f h" % ((1 - np.sqrt(1e-3)) / k / 3600))
