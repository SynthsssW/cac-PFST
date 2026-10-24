"""2D overland-flow solver: local-inertial shallow water (Bates, Horritt & Fewtrell 2010).

Implements SOLVER_SPEC.md v1.2. Pure numpy, SI units throughout.

Face-flux storage: inside step() and run() the fluxes are kept on the PADDED face
arrays of §2.5, qx (ny, nx+1) and qy (ny+1, nx), because open-edge faces carry their
own q_old between steps (§4.1). Column/row 0 and nx/ny are the boundary faces; the
interior slices qx[:, 1:-1] and qy[1:-1, :] are the §1 arrays (ny, nx-1) and (ny-1, nx).
"""
from dataclasses import dataclass, field

import numpy as np

G = 9.81

CLOSED_EDGES = {"N": "closed", "S": "closed", "E": "closed", "W": "closed"}


# ---------------------------------------------------------------------------
# §2.1–§2.3 face flux
# ---------------------------------------------------------------------------
def _momentum(q_old, eta_a, eta_b, z_a, z_b, n_a, n_b, dx, dt, g, h_min):
    """§2.1 h_flow, §2.2 n_face, §2.3 semi-implicit momentum update.

    b is the +direction neighbour of a. Faces with h_flow < h_min get exactly 0.0
    and skip the friction update.
    """
    # §2.1
    h_flow = np.maximum(eta_a, eta_b) - np.maximum(z_a, z_b)
    wet = h_flow >= h_min
    # §2.2
    n_face = 0.5 * (n_a + n_b)
    # §2.3
    S = (eta_b - eta_a) / dx
    hf = np.where(wet, h_flow, 1.0)  # dry faces: value unused, avoids dividing by 0
    q_new = (q_old - g * hf * dt * S) / (1 + g * dt * n_face**2 * np.abs(q_old) / hf**(7 / 3))
    return np.where(wet, q_new, 0.0)


def _ghost_x(a):
    """Pad a (ny, nx) cell field with one ghost column on each side (W, E)."""
    return np.concatenate((a[:, :1], a, a[:, -1:]), axis=1)


def _ghost_y(a):
    """Pad a (ny, nx) cell field with one ghost row on each side (N, S)."""
    return np.concatenate((a[:1, :], a, a[-1:, :]), axis=0)


def _ghost_terrain(z_eff):
    """§4.1 ghost terrain, z_ghost = 2*z_edge - z_interior_neighbour."""
    zx = np.concatenate((2 * z_eff[:, :1] - z_eff[:, 1:2], z_eff,
                         2 * z_eff[:, -1:] - z_eff[:, -2:-1]), axis=1)
    zy = np.concatenate((2 * z_eff[:1, :] - z_eff[1:2, :], z_eff,
                         2 * z_eff[-1:, :] - z_eff[-2:-1, :]), axis=0)
    return zx, zy


# ---------------------------------------------------------------------------
# step()
# ---------------------------------------------------------------------------
def step(h, qx, qy, z_eff, n, dx, dt, rain, params, bc, state):
    """Advance one time step in the §2.8 order. Returns (h, qx, qy, state).

    h        (ny, nx) depth at the start of the step
    qx, qy   padded face fluxes (ny, nx+1), (ny+1, nx) from the previous step
    rain     scalar or (ny, nx), m/s, held constant over the step
    params   dict: g, h_min
    bc       dict: edges, surge_mask, surge_fn, culverts, Ia, f_c
    state    dict: t, P_cum, h_max, eta_max, budget (§6 volumes); not modified in place
    """
    g = params["g"]
    h_min = params["h_min"]
    edges = bc["edges"]
    open_w = edges["W"] == "open"
    open_e = edges["E"] == "open"
    open_n = edges["N"] == "open"
    open_s = edges["S"] == "open"
    budget = dict(state["budget"])
    r = np.broadcast_to(np.asarray(rain, dtype=float), h.shape)

    # 1. eta from the start-of-step depth
    eta = z_eff + h

    # 2. h_flow, n_face, S for ALL faces, built on ghost-padded cells.
    # §4.1 open-edge ghost: z_ghost extrapolated, h_ghost = h_edge (zero depth gradient).
    zx, zy = _ghost_terrain(z_eff)
    hx, hy = _ghost_x(h), _ghost_y(h)
    etax, etay = zx + hx, zy + hy
    nfx, nfy = _ghost_x(n), _ghost_y(n)

    # 3. §2.3 momentum for every padded face, from the same eta
    qx_new = _momentum(qx, etax[:, :-1], etax[:, 1:], zx[:, :-1], zx[:, 1:],
                       nfx[:, :-1], nfx[:, 1:], dx, dt, g, h_min)
    qy_new = _momentum(qy, etay[:-1, :], etay[1:, :], zy[:-1, :], zy[1:, :],
                       nfy[:-1, :], nfy[1:, :], dx, dt, g, h_min)
    # closed edges: 0.0; open edges: inward-pointing flux set to 0.0
    qx_new[:, 0] = np.minimum(qx_new[:, 0], 0.0) if open_w else 0.0
    qx_new[:, -1] = np.maximum(qx_new[:, -1], 0.0) if open_e else 0.0
    qy_new[0, :] = np.minimum(qy_new[0, :], 0.0) if open_n else 0.0
    qy_new[-1, :] = np.maximum(qy_new[-1, :], 0.0) if open_s else 0.0

    # 4. §2.4 per-cell limiter over the padded arrays (boundary faces included)
    V_out = dt * dx * ((np.maximum(qx_new[:, 1:], 0.0) + np.maximum(-qx_new[:, :-1], 0.0))
                       + (np.maximum(qy_new[1:, :], 0.0) + np.maximum(-qy_new[:-1, :], 0.0)))
    V_cell = h * dx**2
    out = V_out > 0
    scale = np.ones_like(h)
    scale[out] = np.minimum(1.0, V_cell[out] / V_out[out])
    sx = np.pad(scale, ((0, 0), (1, 1)), constant_values=1.0)
    sy = np.pad(scale, ((1, 1), (0, 0)), constant_values=1.0)
    qx_new = qx_new * np.where(qx_new > 0, sx[:, :-1], sx[:, 1:])
    qy_new = qy_new * np.where(qy_new > 0, sy[:-1, :], sy[1:, :])

    # 5. outflow volume from the limited open-edge fluxes
    v_out = 0.0
    if open_w:
        v_out += dt * dx * np.sum(-qx_new[:, 0])
    if open_e:
        v_out += dt * dx * np.sum(qx_new[:, -1])
    if open_n:
        v_out += dt * dx * np.sum(-qy_new[0, :])
    if open_s:
        v_out += dt * dx * np.sum(qy_new[-1, :])
    budget["outflow_volume"] += v_out

    # 6. §2.5 continuity on the padded arrays (rain included; infiltration in step 7)
    div = (qx_new[:, 1:] - qx_new[:, :-1]) / dx + (qy_new[1:, :] - qy_new[:-1, :]) / dx
    h_new = h + dt * (r - div)

    # 7. §2.6 infiltration sub-step
    rain_dt = r * dt
    remaining = np.maximum(bc["Ia"] - state["P_cum"], 0.0)
    abstraction = np.minimum(rain_dt, remaining)          # rain lost while P_cum < Ia
    P_cum = state["P_cum"] + rain_dt
    # fraction of the step after Ia is satisfied (handles the crossing step exactly)
    with np.errstate(invalid="ignore", divide="ignore"):
        frac_after = np.where(rain_dt > 0, (rain_dt - abstraction) / rain_dt,
                              np.where(remaining > 0, 0.0, 1.0))
    active = (h > 0) | (r > 0)
    f_dt = abstraction + np.where(active, bc["f_c"] * dt * frac_after, 0.0)
    loss = np.minimum(f_dt, h_new)
    h_new = h_new - loss
    budget["infiltrated_volume"] += np.sum(loss) * dx**2

    # 8. §4.2 Dirichlet (surge) cells overwritten
    mask = bc["surge_mask"]
    if mask is not None:
        S_t = bc["surge_fn"](state["t"] + dt)
        h_before = h_new[mask]
        h_after = np.maximum(S_t - z_eff[mask], 0.0)
        h_new[mask] = h_after
        budget["boundary_inflow_volume"] += np.sum(h_after - h_before) * dx**2

    # 9. §5 culvert transfers, a -> b only
    for ja, ia, jb, ib, A, C, z_inv in bc["culverts"]:
        eta_a = z_eff[ja, ia] + h_new[ja, ia]
        eta_b = z_eff[jb, ib] + h_new[jb, ib]
        head = max(eta_a - eta_b, 0.0) if eta_a > z_inv else 0.0
        Q = C * A * np.sqrt(2 * g * head)
        V = min(Q * dt, h_new[ja, ia] * dx**2)
        h_new[ja, ia] -= V / dx**2
        h_new[jb, ib] += V / dx**2

    # 10. clip h >= 0, record clipped volume
    budget["clipped_volume"] += np.sum(np.maximum(-h_new, 0.0)) * dx**2
    h_new = np.maximum(h_new, 0.0)

    # 11. running maxima
    h_max = np.maximum(state["h_max"], h_new)
    eta_max = np.maximum(state["eta_max"], z_eff + h_new)

    # 12. mass budget (§6)
    budget["rain_volume"] += np.sum(rain_dt) * dx**2

    new_state = dict(state, t=state["t"] + dt, P_cum=P_cum, h_max=h_max, eta_max=eta_max,
                     budget=budget, last_outflow_volume=v_out)
    return h_new, qx_new, qy_new, new_state


# ---------------------------------------------------------------------------
# run()
# ---------------------------------------------------------------------------
@dataclass
class Result:
    h_final: np.ndarray
    h_max: np.ndarray
    eta_max: np.ndarray
    t_wet_first: np.ndarray
    qx: np.ndarray
    qy: np.ndarray
    budget: dict
    outflow_series: list
    snapshots: list = field(default_factory=list)


def run(z, n, dx, rain_fn, t_end, *, rain_breakpoints=(), dz=None,
        edges=CLOSED_EDGES,
        surge_mask=None, surge_fn=None, culverts=(), Ia=None, f_c=None, h0=None,
        alpha=0.7, h_min=1e-3, dt_min=0.01, dt_max=60.0, record_every=None):
    """Run from t = 0 to t_end. Returns Result (§8).

    Result.qx, Result.qy are the §1 interior face arrays (ny, nx-1), (ny-1, nx).
    """
    g = G
    z = np.asarray(z, dtype=float)
    ny, nx = z.shape
    # §3: z_eff = z + dz, once before the run
    z_eff = z + (0.0 if dz is None else np.asarray(dz, dtype=float))
    n = np.broadcast_to(np.asarray(n, dtype=float), (ny, nx))
    h = np.zeros((ny, nx)) if h0 is None else np.array(h0, dtype=float)
    h_init = h.copy()
    qx = np.zeros((ny, nx + 1))
    qy = np.zeros((ny + 1, nx))

    params = {"g": g, "h_min": h_min}
    bc = {
        "edges": dict(CLOSED_EDGES, **edges),
        "surge_mask": None if surge_mask is None else np.asarray(surge_mask, dtype=bool),
        "surge_fn": surge_fn,
        "culverts": list(culverts),
        "Ia": np.zeros((ny, nx)) if Ia is None else np.broadcast_to(np.asarray(Ia, float), (ny, nx)),
        "f_c": np.zeros((ny, nx)) if f_c is None else np.broadcast_to(np.asarray(f_c, float), (ny, nx)),
    }
    state = {
        "t": 0.0,
        "P_cum": np.zeros((ny, nx)),
        "h_max": h.copy(),
        "eta_max": z_eff + h,
        "budget": {"rain_volume": 0.0, "infiltrated_volume": 0.0, "outflow_volume": 0.0,
                   "boundary_inflow_volume": 0.0, "clipped_volume": 0.0},
    }
    t_wet_first = np.where(h > 0, 0.0, np.inf)

    # times dt must land on exactly: forcing breakpoints and t_end
    targets = sorted({float(b) for b in rain_breakpoints if 0.0 < b < t_end} | {float(t_end)})
    k = 0
    outflow_series = []
    snapshots = []
    next_record = record_every

    t = 0.0
    while t < t_end:
        # §2.7 time step
        h_ref = max(h.max(), h_min)
        dt = alpha * dx / np.sqrt(g * h_ref)
        dt = float(np.clip(dt, dt_min, dt_max))
        while targets[k] <= t:
            k += 1
        if t + dt >= targets[k]:
            dt = targets[k] - t
            t_next = targets[k]
        else:
            t_next = t + dt

        state["t"] = t
        h, qx, qy, state = step(h, qx, qy, z_eff, n, dx, dt, rain_fn(t), params, bc, state)
        t = t_next

        outflow_series.append((t, state["last_outflow_volume"] / dt))
        newly_wet = (h > 0) & np.isinf(t_wet_first)
        t_wet_first[newly_wet] = t
        if record_every is not None and t >= next_record:
            snapshots.append((t, h.copy()))
            while next_record <= t:
                next_record += record_every

    budget = dict(state["budget"])
    stored = np.sum(h) * dx**2 - np.sum(h_init) * dx**2
    budget["residual"] = stored - (budget["rain_volume"] - budget["infiltrated_volume"]
                                   - budget["outflow_volume"] + budget["boundary_inflow_volume"])
    budget["V_ref"] = max(budget["rain_volume"] + abs(budget["boundary_inflow_volume"])
                          + np.sum(h_init) * dx**2, 1e-12)

    return Result(h_final=h, h_max=state["h_max"], eta_max=state["eta_max"],
                  t_wet_first=t_wet_first, qx=qx[:, 1:-1], qy=qy[1:-1, :], budget=budget,
                  outflow_series=outflow_series, snapshots=snapshots)
