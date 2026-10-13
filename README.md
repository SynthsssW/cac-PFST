# Pasco Flood Screening Tool.
Idea from PINO paper — a Fourier Neural Operator trained on your own synthetic solver runs — but the PDE is 2D overland flow (diffusive-wave shallow water) on the county’s LiDAR terrain instead of groundwater. Inputs: elevation tile, rainfall intensity or surge height at the coast, and the state of drainage (culverts, retention ponds, raised roads). Output: inundation depth across the tile.

# Two faces of the same app:
## Residents:
type an address, see depth under a 10-year storm, a 100-year storm, and a Helene-scale surge — with bands, because terrain and rainfall are uncertain.

## County:
Draw a proposed project (retention pond here, raise this road, clear this culvert) and see how many homes drop out of the inundation footprint. That’s a per-dollar ranking for the $235M, which nobody on staff can do interactively today.

###
Data is all public: Pasco/SWFWMD LiDAR DEM, FEMA flood zones, NOAA Atlas 14 rainfall, county stormwater project list, Helene/Milton high-water marks from USGS. Solver is a few hundred lines of numpy on 128×128 DEM tiles; same FNO training pipeline as before.
Honest framing: screening-grade, not a replacement for the county’s engineering models — a way to decide which neighborhoods get the full study first.
Opening line of the video: “Pasco just committed $235 million to flood mitigation with a five-year deadline. Here’s a tool that shows where each dollar removes the most water.”

# Backend (Python)

1. Solver — this week, before anything else.
 2D diffusive-wave overland flow on a grid:
State: water depth h(x,y,t). Water surface η = z + h, where z is terrain.
Flux between cells: q = −(h^(5/3) / n) · ∇η / |∇η|^(1/2) (Manning). Explicit time stepping with a CFL-limited dt.
Inputs: DEM tile (128×128, 10 m cells from Pasco/SWFWMD LiDAR), rainfall rate r(t), boundary head at coast cells (surge), optional “infrastructure mask” (cells with added storage or raised elevation).
Output: max depth over the event, h_max(x,y).
~150 lines of numpy. Validate on a tilted plane and a bowl before trusting it.

3. Dataset — weekend.
 Sample ~3,000 scenarios: 20–30 real DEM tiles from west Pasco (Hudson, Sea Pines, Port Richey, Holiday) plus synthetic bumps; rainfall from NOAA Atlas 14 return periods (2-yr to 100-yr); surge 0–8 ft; random infrastructure edits (pond, raised road). Save (inputs, h_max) as .npz. Run overnight.

4. FNO — week 2.
 neuraloperator package. Inputs stacked as channels: DEM, rainfall-total map, surge-boundary map, infrastructure mask. Output: h_max. Add a physics term to the loss — mass-balance residual (total water in = rainfall + inflow − outflow − storage) is cheap and keeps it honest. Train on a laptop GPU or Colab in a few hours. Target: relative L2 under 10% on held-out tiles.

5. Uncertainty.
 Don’t train a Bayesian net — run the operator over an ensemble: ±20% rainfall, ±2 ft surge, DEM noise. 30 forward passes in under a second. Return mean, p10, p90 per cell.

6. API — FastAPI, one endpoint.
 POST /predict with {tile_id, rainfall_return_period, surge_ft, edits: [...]} → returns depth grid (mean/p10/p90) as base64 PNG or flat array + summary stats (cells above 1 ft, homes flagged). Host on Render/Railway/Fly free tier; GPU not needed for inference at this size. Preload the model at startup.


