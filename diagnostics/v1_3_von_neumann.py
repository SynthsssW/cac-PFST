# v1.3: von Neumann stability of spec sections 2.3 (theta-weighted) + 2.5 linearised about a lake at rest (u = 0, so friction vanishes and dt = alpha*dx/c), 1D and 2D; no solver import.
import numpy as np


def amp_matrix(kx, ky, C, theta, two_d=True):
    """Amplification matrix for (eta, qx, qy), nondimensional (q scaled by c, eta by H).

    qx' = s_x qx - C * i*Gx * eta          (sections 2.3, theta-weighted inertia along x)
    eta' = eta - C * i*(Gx qx' + Gy qy')   (section 2.5, using the NEW fluxes)
    with s = theta + (1-theta) cos k, G = 2 sin(k/2), C = c dt / dx.
    """
    sx = theta + (1 - theta) * np.cos(kx)
    sy = theta + (1 - theta) * np.cos(ky)
    Gx = 2 * np.sin(kx / 2)
    Gy = 2 * np.sin(ky / 2) if two_d else 0.0
    # step 1: q' = S q + B eta ; step 2: eta' = eta - C i (G . q')
    M_q = np.array([[0, 0, 0],
                    [-1j * C * Gx, sx, 0],
                    [-1j * C * Gy, 0, sy if two_d else 1.0]], dtype=complex)
    M_q[0, 0] = 1.0
    row_eta = np.array([1, 0, 0], dtype=complex) - 1j * C * (Gx * M_q[1] + Gy * M_q[2])
    M = M_q.copy()
    M[0] = row_eta
    return M


def rho(C, theta, two_d=True, nk=97):
    ks = np.linspace(-np.pi, np.pi, nk)
    worst = 0.0
    for kx in ks:
        for ky in (ks if two_d else [0.0]):
            lam = np.linalg.eigvals(amp_matrix(kx, ky, C, theta, two_d))
            worst = max(worst, np.abs(lam).max())
    return worst


def limit(theta, two_d):
    lo, hi = 0.05, 1.5
    for _ in range(40):
        m = (lo + hi) / 2
        lo, hi = (m, hi) if rho(m, theta, two_d, nk=49) <= 1 + 1e-12 else (lo, m)
    return lo


for theta in [1.0, 0.9, 0.8]:
    print("theta=%.1f  stable Courant limit: 1D %.4f   2D %.4f   | spectral radius at C=0.7: 1D %.4f  2D %.4f"
          % (theta, limit(theta, False), limit(theta, True), rho(0.7, theta, False), rho(0.7, theta, True)))
# closed form for the 2D (pi, pi) checkerboard: s = 2*theta - 1, need 1 + s - 8 C^2 >= -(1 + s)
th = 0.8
print("closed form 2D (pi,pi) limit for theta=0.8: C <= sqrt((1 + (2*th-1))/4) = %.4f" % np.sqrt((1 + (2 * th - 1)) / 4))
