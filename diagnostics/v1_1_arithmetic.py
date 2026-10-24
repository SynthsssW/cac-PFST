# Checks spec v1.1 by arithmetic (no solver): T3 outlet depth with the zero-depth ghost, T3 spin-up outlet overdraw with the limiter skipped, T4b block cells vs eta_eq + 0.10.
import numpy as np

g = 9.81; dx = 10.; n = 0.03; S0 = 0.01; r = 50e-3 / 3600; nx = 128

# (A) T3 outlet cell with v1.1 ghost: z_ghost = 2*z_edge - z_int = -S0*dx, h_ghost = 0
q = r * nx * dx
lo, hi = 1e-4, 1.0
for _ in range(200):
    h = (lo + hi) / 2
    Sf = (S0 * dx + h) / dx
    lo, hi = (h, hi) if h ** (5 / 3) * np.sqrt(Sf) / n < q else (lo, h)
ha = (n * q / np.sqrt(S0)) ** 0.6
print("T3 i=127: steady depth with zero-depth ghost %.5f m vs h_i %.5f m -> %.1f%% low" % (h, ha, 100 * (1 - h / ha)))

# (B) T3 spin-up, v1.1 order (boundary face computed after the limiter): first wet step at the east edge
h_min = 1e-3
dt = min(0.7 * dx / np.sqrt(g * h_min), 60.0)
he = 1.0e-3 + 1e-6
qb = -g * he * dt * ((-S0 * dx - he) / dx)  # section 2.3 from q_old = 0, ghost neighbour
print("T3 spin-up: dt=%.1f s, boundary q=%.2e m2/s, outgoing %.2f m3 vs cell volume %.3f m3 (x%.0f)"
      % (dt, qb, qb * dt * dx, he * dx * dx, qb * dt * dx / (he * dx * dx)))

# (C) T4b (v1.1 bowl 5e-4): block cells' initial eta vs eta_eq + 0.10
N = 129; c = 64
X = (np.arange(N) - c) * dx
z = 5e-4 * (X[None, :] ** 2 + X[:, None] ** 2)
h0 = np.zeros((N, N)); h0[60:70, 60:70] = 2.0
V = h0.sum() * dx * dx
lo, hi = 0, 10
for _ in range(200):
    e = (lo + hi) / 2
    lo, hi = (e, hi) if np.maximum(e - z, 0).sum() * dx * dx < V else (lo, e)
wet_end = (e - z) > 0.01
eta0 = z + h0
blk = h0 > 0
print("T4b eta_eq=%.4f; block cells wet at end: %d/100; max initial eta among them=%.3f; count > eta_eq+0.10: %d"
      % (e, (wet_end & blk).sum(), eta0[wet_end & blk].max(), (wet_end & blk & (eta0 > e + 0.10)).sum()))
