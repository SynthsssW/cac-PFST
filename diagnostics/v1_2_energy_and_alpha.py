# Total energy in v1.2 T4b (must not rise) plus alpha sensitivity of T7 and T8. Solver: commit 192f777.
import sys, numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import floodsolver.solver as S
from floodsolver.solver import run
DX=10.; g=9.81; C={"N":"closed","S":"closed","E":"closed","W":"closed"}
def energy(h,qx,qy,z):
    PE=g*np.sum(h*(z+h/2))*DX**2
    hx=0.5*(h[:,:-1]+h[:,1:]); hy=0.5*(h[:-1,:]+h[1:,:])
    qi=qx[:,1:-1]; qj=qy[1:-1,:]
    KE=0.5*(np.sum(np.where(hx>1e-3,qi**2/np.maximum(hx,1e-3),0))+np.sum(np.where(hy>1e-3,qj**2/np.maximum(hy,1e-3),0)))*DX**2
    return PE,KE
N=129; X=(np.arange(N)-64)*DX; z=5e-4*(X[None,:]**2+X[:,None]**2); n=np.full((N,N),0.03)
h0=np.zeros((N,N)); h0[60:70,60:70]=2.0
for a in [0.7,0.35]:
    h=h0.copy(); qx=np.zeros((N,N+1)); qy=np.zeros((N+1,N)); t=0.; nxt=0.; rows=[]
    bc={"edges":C,"surge_mask":None,"surge_fn":None,"culverts":[],"Ia":np.zeros((N,N)),"f_c":np.zeros((N,N))}
    st={"t":0.0,"P_cum":np.zeros((N,N)),"h_max":h.copy(),"eta_max":z+h,"budget":dict(rain_volume=0.,infiltrated_volume=0.,outflow_volume=0.,boundary_inflow_volume=0.,clipped_volume=0.)}
    E0=sum(energy(h,qx,qy,z)); Emin=E0; maxrise=0
    while t<6*3600:
        dt=float(np.clip(a*DX/np.sqrt(g*max(h.max(),1e-3)),0.01,60)); st["t"]=t
        h,qx,qy,st=S.step(h,qx,qy,z,n,DX,dt,0.0,{"g":g,"h_min":1e-3},bc,st); t+=dt
        E=sum(energy(h,qx,qy,z)); maxrise=max(maxrise,E-Emin); Emin=min(Emin,E)
        if t>=nxt: rows.append("%.0f:%.4f"%(t,(E-E0)/E0)); nxt+=3600
    wet=h>0.01; eta=z+h
    print("T4b alpha=%.2f  (E-E0)/E0 hourly: %s ; largest energy RISE above running min: %.3e J (%.2e of E0); final std(eta)=%.2e; eta_max dry0&wet=%.3f"%(a," ".join(rows),maxrise,maxrise/E0,eta[wet].std(),st["eta_max"][(h0==0)&wet].max()))
# T7 and T8 at alpha 0.35 (diagnostic only)
NX=NY=128; z7=np.zeros((NY,NX)); m=np.zeros((NY,NX),bool); m[:,0]=True; h07=np.zeros((NY,NX)); h07[:,0]=1.0
for a in [0.7,0.35]:
    r7=run(z7,np.full((NY,NX),0.03),DX,lambda t:0.0,7200.,h0=h07,surge_mask=m,surge_fn=lambda t:1.0,edges=C,alpha=a)
    hf=r7.h_final[0]; print("T7 alpha=%.2f: final h east wall %.5f, max diff along i %.2e"%(a,hf[-1],np.diff(hf).max()))
z8=np.full((NY,NX),2.0); W=(slice(54,74),slice(20,40)); E_=(slice(54,74),slice(88,108)); z8[W]=0; z8[E_]=0; h08=np.zeros((NY,NX)); h08[W]=1.0
for a in [0.7,0.35]:
    r8=run(z8,np.full((NY,NX),0.03),DX,lambda t:0.0,6*3600.,h0=h08,culverts=[(63,29,63,97,1.0,0.6,0.0)],edges=C,alpha=a)
    e=z8+r8.h_final; print("T8 alpha=%.2f: mean eta west %.5f east %.5f diff %.2e; std west %.1e east %.1e"%(a,e[W].mean(),e[E_].mean(),e[W].mean()-e[E_].mean(),e[W].std(),e[E_].std()))
