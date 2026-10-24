# v1.3 T4b by arithmetic (no solver): the exact dam-break (Ritter) solution at the block edge already exceeds the sloshing bound eta_eq + 0.10 on an initially dry, finally wet cell.
import numpy as np

g = 9.81
dx = 10.0
a = 2e-5            # v1.3 bowl coefficient
h0 = 2.0            # block depth
c0 = np.sqrt(g * h0)

# final equilibrium level of the 20,000 m3 block in the bowl (discrete, 129x129 grid)
N = 129
X = (np.arange(N) - 64) * dx
z = a * (X[None, :] ** 2 + X[:, None] ** 2)
hb = np.zeros((N, N)); hb[60:70, 60:70] = h0
V = hb.sum() * dx ** 2
lo, hi = 0.0, 5.0
for _ in range(200):
    e = (lo + hi) / 2
    lo, hi = (e, hi) if np.maximum(e - z, 0).sum() * dx ** 2 < V else (lo, e)
eta_eq = e
bound = eta_eq + 0.10

# cell just outside the block's east edge, centre row: (64, 70). dry at t=0, inside the final lake
j, i = 64, 70
print("eta_eq = %.4f m, bound eta_eq + 0.10 = %.4f m; cell (%d,%d): z = %.4f m, dry at t=0: %s, wet at end: %s"
      % (eta_eq, bound, j, i, z[j, i], hb[j, i] == 0, eta_eq - z[j, i] > 0.01))


# 1D Ritter dam-break on a dry bed: h(x, t) = (2 c0 - x/t)^2 / (9 g) for 0 <= x <= 2 c0 t (x from the dam)
def ritter_cell_avg(t, x0=0.0, x1=dx, m=2001):
    x = np.linspace(x0, x1, m)
    h = np.where(x <= 2 * c0 * t, (2 * c0 - x / t) ** 2 / (9 * g), 0.0)
    return h.mean()


# The mid-edge cell is 45-55 m from the block corners. No corner signal can arrive before
# t = 45 m / (2 c0) (2 c0 is the fastest characteristic speed in the dam-break fan),
# so the 1D solution is exact there until then.
t_valid = 45.0 / (2 * c0)
for t in [1.0, 2.0, 3.0, 4.0, t_valid]:
    hc = ritter_cell_avg(t)
    print("t = %.2f s: exact cell-average depth %.3f m -> eta ~ %.3f m  (bound %.3f) %s"
          % (t, hc, hc + z[j, i], bound, "EXCEEDS" if hc + z[j, i] > bound else ""))
print("depth at the dam site itself is 4*h0/9 = %.3f m for all t > 0 (eta %.3f m)" % (4 * h0 / 9, 4 * h0 / 9 + z[j, i]))
print("1D solution exact at the mid-edge cell until t = %.2f s" % t_valid)
