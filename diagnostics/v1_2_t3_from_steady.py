# Starts v1.2 T3 at the analytic steady state and shows the error growing to O(1) in ~50 steps. Solver: commit 192f777.
import sys, numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import floodsolver.solver as S
# 1-row version of T3 (rows are identical); drive step() directly and inspect
NX=128; NY=4; DX=10.; r=50e-3/3600; S0=0.01; g=9.81
X=np.arange(NX)*DX; z=np.broadcast_to(S0*((NX-1)*DX-X),(NY,NX)).copy(); n=np.full((NY,NX),0.03)
ha=((0.03*r*(np.arange(NX)+1)*DX/np.sqrt(S0))**0.6)
# start AT the analytic steady state with analytic fluxes -> should stay put
h=np.broadcast_to(ha,(NY,NX)).copy()
qx=np.zeros((NY,NX+1)); qx[:,1:]=r*(np.arange(NX)+1)*DX; qy=np.zeros((NY+1,NX))
params={"g":g,"h_min":1e-3}
bc={"edges":{"N":"closed","S":"closed","E":"open","W":"closed"},"surge_mask":None,"surge_fn":None,"culverts":[],"Ia":np.zeros((NY,NX)),"f_c":np.zeros((NY,NX))}
st={"t":0.0,"P_cum":np.zeros((NY,NX)),"h_max":h.copy(),"eta_max":z+h,"budget":dict(rain_volume=0.,infiltrated_volume=0.,outflow_volume=0.,boundary_inflow_volume=0.,clipped_volume=0.)}
for k in range(2001):
    dt=float(np.clip(0.7*DX/np.sqrt(g*max(h.max(),1e-3)),0.01,60))
    h,qx,qy,st=S.step(h,qx,qy,z,n,DX,dt,r,params,bc,st)
    if k in (0,1,2,5,10,50,200,1000,2000):
        rel=qx[0,1:]/(r*(np.arange(NX)+1)*DX)
        print(k,"dt=%.2f"%dt,"q/q_eq faces 60..65:",np.round(rel[60:66],4)," max|h/ha-1|=%.3e"%np.abs(h[0]/ha-1).max())
# friction factor k*|q| at steady state
hf=ha[:-1]; q=r*(np.arange(NX-1)+1)*DX; dt=0.7*DX/np.sqrt(g*ha.max())
kq=g*dt*0.03**2*q/hf**(7/3); print("friction factor g dt n^2 |q|/h^(7/3): min %.2f max %.2f at i=8 %.2f i=64 %.2f"%(kq.min(),kq.max(),kq[8],kq[64]))
