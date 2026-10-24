# Pasco Flood Screening — 2D overland-flow solver spec

This document is the contract. Implement exactly what is written here. Do not substitute a different scheme, reorder operations, change tolerances, or "simplify" an equation. If you believe something in this spec is wrong or ambiguous, STOP and report it — do not guess and do not work around it. If a test fails, fix the implementation, never the test.

Deliverables:
- `floodsolver/solver.py` — pure numpy, no numba/jax/torch in v1. Stateless functions: arrays in, arrays out.
- `tests/test_solver.py` — the tests in §7, written FIRST, before `solver.py`.
- `floodsolver/analytic.py` — the analytic reference solutions used by tests.

Units: SI everywhere inside the solver (m, s, m/s, m²/s). Unit conversion (mm/hr, ft) happens only in I/O helpers, never inside `step()` or `run()`.

---

## 1. Grid and state

- Square cells, spacing `dx` (m). Arrays are `(ny, nx)`, index `[row=j, col=i]`, row 0 = north.
- `z[j,i]` terrain elevation (m). `n[j,i]` Manning roughness (s·m^-1/3). Both inputs, never modified by `step()`.
- `h[j,i]` water depth (m), `h >= 0` always.
- `eta = z + h` water-surface elevation.
- Face fluxes are unit-width discharge (m²/s) on a staggered grid:
  - `qx` shape `(ny, nx-1)`: `qx[j,i]` is the flux across the face between cell `(j,i)` and `(j,i+1)`. Positive = flow in +i direction (east).
  - `qy` shape `(ny-1, nx)`: `qy[j,i]` is the flux across the face between cell `(j,i)` and `(j+1,i)`. Positive = flow in +j direction (south).
- Rainfall `r` (m/s) is a scalar or `(ny, nx)` array per step. Infiltration rate `f_c` (m/s) and initial abstraction `Ia` (m) are per-cell arrays.

## 2. Scheme: local-inertial shallow water (Bates, Horritt & Fewtrell 2010)

This is NOT the diffusive wave. It is the inertial form with semi-implicit friction. Use it exactly as below.

### 2.1 Flow depth at a face (between cells a and b)

```
h_flow = max(eta_a, eta_b) - max(z_a, z_b)
```
If `h_flow < h_min` (default `h_min = 1e-3` m) the face flux is set to exactly `0.0` and the friction update is skipped for that face. Do NOT use `(h_a + h_b)/2`.

### 2.2 Face roughness

```
n_face = 0.5 * (n_a + n_b)
```

### 2.3 Momentum update (per face, vectorised)

Gradient of water surface across the face, positive-direction convention matching §1:
```
S = (eta_b - eta_a) / dx        # b is the +direction neighbour
```
Update:
```
q_new = (q_old - g * h_flow * dt * S) / (1 + g * dt * n_face**2 * abs(q_old) / h_flow**(7/3))
```
with `g = 9.81`. Compute `qx` and `qy` from the SAME `eta` (the one at the start of the step). Do not update `eta` between the x and y flux computations.

### 2.4 Flux limiter

After §2.3, limit each face flux so that one step cannot remove more water than the upwind cell holds, sharing the cell's depth across its 4 faces:
```
h_up = h_a if q_new > 0 else h_b          # depth of the cell water leaves
q_cap = h_up * dx / (4 * dt)
q_new = sign(q_new) * min(abs(q_new), q_cap)
```
Then `h` is updated (§2.5) and clipped `h = maximum(h, 0.0)`. Accumulate the clipped volume `sum(negative part) * dx²` into the diagnostic `clipped_volume`. In all tests `clipped_volume` must be exactly `0.0` — if it is not, the limiter or indexing is wrong.

### 2.5 Continuity

```
div = (qx[:, 1:] - qx[:, :-1]) / dx  +  (qy[1:, :] - qy[:-1, :]) / dx
```
where `qx` and `qy` here are the FULL face arrays including boundary faces (§4), i.e. padded to `(ny, nx+1)` and `(ny+1, nx)` with boundary values. Then:
```
h_new = h + dt * (r - f - div)
```

### 2.6 Infiltration `f`

Per cell, per step:
1. Initial abstraction: maintain cumulative rain `P_cum[j,i]`. While `P_cum < Ia`, all rain in that step is lost (`f = r` for the portion up to `Ia`). Track exactly, including the partial step where `P_cum` crosses `Ia`.
2. After abstraction is satisfied, `f = f_c` wherever `h > 0 or r > 0`, else `0`.
3. Never let infiltration drive `h` negative: `f = min(f, (h + r*dt - dt*div) / dt)` is NOT acceptable because it couples to `div`; instead apply infiltration as a separate sub-step after continuity: `loss = min(f*dt, h_new)`, `h_new -= loss`, and accumulate `loss*dx²` into `infiltrated_volume`.

Tests in §7 use `Ia = 0`, `f_c = 0` unless stated.

### 2.7 Time step

```
h_ref = max(h.max(), h_min)
dt = alpha * dx / sqrt(g * h_ref)
dt = clip(dt, dt_min, dt_max)
```
`alpha = 0.7`, `dt_min = 0.01`, `dt_max = 60.0` seconds. Recompute every step. The step must also be shortened so the simulation hits the requested end time and any forcing breakpoint (rain on/off) exactly.

### 2.8 Order of operations inside `step()`

1. `eta = z + h`
2. compute `h_flow`, `n_face`, `S` for x faces and y faces from this `eta`
3. §2.3 momentum for x and y
4. §2.4 limiter
5. boundary face fluxes (§4)
6. §2.5 continuity (rain included, infiltration not)
7. §2.6 infiltration sub-step
8. §4.2 Dirichlet (surge) cells overwritten
9. §5 culvert transfers
10. clip `h >= 0`, record `clipped_volume`
11. `h_max = maximum(h_max, h)`
12. update mass budget (§6)

## 3. Infrastructure inputs

- `dz[j,i]` (m): added to `z` once before the run (raised road = positive, excavated pond = negative). `z_eff = z + dz`.
- Culvert links: list of `(ja, ia, jb, ib, area_m2, C, z_invert)`. See §5.
- Nothing else in v1.

## 4. Boundaries

### 4.1 Edge faces
Each of the 4 edges has a mode: `"closed"` (default) or `"open"`.
- closed: boundary face flux = `0.0`.
- open (free outfall): construct a ghost cell outside the edge with `h_ghost = 0` and `z_ghost = 2*z_edge - z_interior_neighbour` (linear extrapolation of terrain). Compute the boundary face flux with the §2 equations using the ghost cell as the outside neighbour. Only outflow is allowed: if the computed flux points into the domain, set it to `0.0`. Accumulate outflow volume into `outflow_volume`.

### 4.2 Dirichlet head (storm surge)
`surge_mask[j,i]` boolean, `surge_fn(t) -> S` (m, water-surface elevation). After continuity each step, for masked cells set:
```
h[mask] = maximum(S(t) - z_eff[mask], 0.0)
```
The volume change this imposes, `sum(h_after - h_before)*dx²`, is accumulated into `boundary_inflow_volume` (may be negative).

## 5. Culverts

For each link `(a, b, A, C, z_inv)`, flow a→b only:
```
head = max(eta_a - eta_b, 0)   if eta_a > z_inv else 0
Q = C * A * sqrt(2 * g * head)              # m³/s
V = min(Q * dt, h_a * dx**2)                 # cannot drain more than cell a holds
h_a -= V / dx**2 ;  h_b += V / dx**2
```
Culverts are off-grid transfers; they do not touch `qx`/`qy`. Default `C = 0.6`.

## 6. Mass budget (required diagnostic)

Track in m³, cumulatively:
`rain_volume`, `infiltrated_volume`, `outflow_volume`, `boundary_inflow_volume`, `clipped_volume`.
Define at any time:
```
residual = (sum(h)*dx² - sum(h0)*dx²) - (rain_volume - infiltrated_volume - outflow_volume + boundary_inflow_volume)
```
`run()` returns the final residual. Relative residual `abs(residual)/max(rain_volume, 1e-12)` must be `< 1e-10` in every test.

## 7. Tests (write these first; tolerances are fixed)

Common: `g = 9.81`, `dx = 10.0`, `alpha = 0.7`, `h_min = 1e-3`, `Ia = 0`, `f_c = 0`, grid 128×128 unless stated. Rain of 50 mm/hr = `1.3889e-5` m/s.

**T1 mass conservation.** `z` = random smooth terrain (sum of 3 low-frequency sines, amplitude 2 m) + 0.001·x tilt; all edges closed; rain 50 mm/hr for 3600 s, then 7200 s no rain. Assert: relative residual `< 1e-10`; `h.min() >= 0`; `clipped_volume == 0.0`; `sum(h)*dx²` equals rain volume to rel `1e-10`.

**T2 symmetry.** Paraboloid bowl `z = 5e-4 * ((x-xc)² + (y-yc)²)` with `xc, yc` at the exact grid centre (use `nx = ny = 129` so there is a centre cell); uniform rain 50 mm/hr for 3600 s + 3600 s drain; closed. Assert `allclose(h, h[::-1, :], atol=1e-9)` and `allclose(h, h[:, ::-1], atol=1e-9)` and `allclose(h, h.T, atol=1e-9)`.

**T3 tilted plane steady state.** `z = S0 * (L - x)` with `S0 = 0.01`, `L = (nx-1)*dx`, uniform in y, `n = 0.03`; east edge open, others closed; rain 50 mm/hr held constant for 4 h. Analytic kinematic steady state: `q(x) = r * x`, `h(x) = (n * r * x / sqrt(S0))**(3/5)`. Assert at t = 4 h: (a) outflow rate (m³/s, last 10 min average) within 2% of `r * L * ny * dx`; (b) `h[j, i]` within 5% of analytic for all `i >= 8`, every row `j` (rows identical to 1e-12); (c) relative residual `< 1e-10`.

**T4a bowl flat surface.** Same bowl as T2, closed, rain 100 mm total over 1 h, then drain 6 h. Let `wet = h > 0.01`. Assert `std(eta[wet]) < 1e-3` m; wet footprint equals `{z < eta_mean}` up to a one-cell boundary ring; relative residual `< 1e-10`.

**T4b bowl no overshoot.** Same bowl, no rain, initial `h = 2.0` m in the central 10×10 block, zero elsewhere, run 6 h. Assert `h_max.max() <= 2.0 + 1e-9` (never overshoots the initial surface), final `std(eta[h > 0.01]) < 1e-3`.

**T5 dry ridge.** `z = 0` everywhere except a ridge `z = 1.0` in columns 63–64; initial `h = 0.5` m in columns 0–62, zero east of the ridge; no rain; closed; run 2 h. Assert `h[:, 65:].max() == 0.0` exactly, and `abs(eta[:, :63] - 0.5).max() < 1e-9`.

**T6 no uphill wetting.** `z = 0` in column 0, `z = 0.3` elsewhere; `h = 0.2` in column 0 only; run 1 h. Assert `h[:, 1:].max() == 0.0` exactly. (Verifies §2.1 `h_flow` definition; `(h_a+h_b)/2` fails this.)

**T7 Dirichlet surge.** Flat `z = 0`, west column is `surge_mask`, `surge_fn(t) = 1.0` constant, all edges closed, run 2 h. Assert `h[:, 0] == 1.0` exactly at all times, `h` monotone non-increasing in `i` at every output time, relative residual `< 1e-10` (boundary inflow included).

**T8 culvert.** Two basins separated by a 2 m ridge, culvert link from a cell in the west basin to a cell in the east basin, `A = 1.0`, `C = 0.6`, `z_inv = 0`; initial 1 m of water in west basin; run 6 h. Assert `eta` equalises across basins to within 1e-3 m; with the link removed, east basin `h.max() == 0.0`.

## 8. API

```python
def step(h, qx, qy, z_eff, n, dx, dt, rain, params, bc, state) -> (h, qx, qy, state)
def run(z, n, dx, rain_fn, t_end, *, dz=None, edges={"N":"closed","S":"closed","E":"closed","W":"closed"},
        surge_mask=None, surge_fn=None, culverts=(), Ia=None, f_c=None, h0=None,
        alpha=0.7, h_min=1e-3, dt_min=0.01, dt_max=60.0, record_every=None) -> Result
```
`Result` has: `h_final, h_max, t_wet_first, qx, qy, budget (dict of the §6 volumes + residual), outflow_series [(t, m³/s)], snapshots (optional)`.
`rain_fn(t) -> float or (ny,nx) array` in m/s. `run()` must shorten `dt` to land exactly on `t_end` and on any breakpoint times passed in `rain_breakpoints`.

## 9. Rules for the implementer

- Tests first. Run them, confirm they fail for the right reason (no solver), then implement.
- Do not loosen any tolerance. Do not skip, xfail, or mark a test slow to avoid it.
- Do not add a slope floor, artificial diffusion, smoothing, or "stability tweaks" not in this spec. If the scheme is unstable in a test, the implementation deviates from §2 somewhere — find it.
- Before declaring done, print the discrete equations you implemented (copy from the code, not from this file) for §2.1, §2.3, §2.4, §2.5, §2.7 so they can be diffed against the spec by eye.
- Report wall-clock time for T3 (the longest run). Target < 3 min in numpy; if slower, report, do not optimise in v1.
