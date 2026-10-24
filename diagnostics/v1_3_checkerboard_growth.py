# v1.3: does solver.py reproduce the von Neumann prediction? Flat closed box, 1 m lake at rest + 1e-9 m checkerboard; per-step growth at alpha 0.7 vs 0.6, theta 0.8 vs 1.0.
import sys
import pathlib

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from floodsolver.solver import run  # noqa: E402

N = 64
DX = 10.0
z = np.zeros((N, N))
n = np.full((N, N), 0.03)
j, i = np.indices((N, N))
h0 = 1.0 + 1e-9 * (-1.0) ** (i + j)
C = {"N": "closed", "S": "closed", "E": "closed", "W": "closed"}
for theta, alpha in [(0.8, 0.7), (0.8, 0.6), (1.0, 0.7)]:
    amps = []
    for steps in (10, 20):
        dt = alpha * DX / np.sqrt(9.81 * h0.max())  # u = 0 at rest, so this is the section 2.7 dt
        res = run(z, n, DX, lambda t: 0.0, steps * dt * 0.999999, h0=h0, edges=C,
                  alpha=alpha, theta=theta)
        amps.append(np.abs(res.h_final - res.h_final.mean()).max())
    g = (amps[1] / amps[0]) ** (1 / 10)
    print("theta=%.1f alpha=%.1f: checkerboard amplitude after 10/20 steps %.2e / %.2e -> growth per step %.3f"
          % (theta, alpha, amps[0], amps[1], g))
