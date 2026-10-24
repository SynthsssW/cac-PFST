# Spectral radius of the v1.2 step map at the discrete T3 steady state, mode location, alpha and n sweeps. Solver: commit 192f777.
import sys, numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import floodsolver.solver as S
g=9.81; DX=10.; r=50e-3/3600; S0=0.01; NY=2
def setup(NX, nman=0.03, east="open"):
    X=np.arange(NX)*DX; z=np.broadcast_to(S0*((NX-1)*DX-X),(NY,NX)).copy(); n=np.full((NY,NX),nman)
    ha=(nman*r*(np.arange(NX)+1)*DX/np.sqrt(S0))**0.6
    q=np.zeros(NX+1); q[1:]=r*(np.arange(NX)+1)*DX
    bc={"edges":{"N":"closed","S":"closed","E":east,"W":"closed"},"surge_mask":None,"surge_fn":None,"culverts":[],"Ia":np.zeros((NY,NX)),"f_c":np.zeros((NY,NX))}
    return z,n,ha,q,bc
def F(v,z,n,bc,dt,NX):
    h=np.tile(v[:NX],(NY,1)); qx=np.tile(v[NX:],(NY,1)); qy=np.zeros((NY+1,NX))
    st={"t":0.0,"P_cum":np.zeros((NY,NX)),"h_max":h.copy(),"eta_max":z+h,"budget":dict(rain_volume=0.,infiltrated_volume=0.,outflow_volume=0.,boundary_inflow_volume=0.,clipped_volume=0.)}
    h2,qx2,_,_=S.step(h,qx,qy,z,n,DX,dt,r,{"g":g,"h_min":1e-3},bc,st)
    return np.concatenate([h2[0],qx2[0]])
def rho(NX, nman=0.03, dt=None, east="open"):
    z,n,ha,q,bc=setup(NX,nman,east)
    # relax to the discrete fixed point first (fixed dt), with damping via averaging of iterates
    if dt is None: dt=0.7*DX/np.sqrt(g*ha.max())
    v=np.concatenate([ha,q]); 
    # Newton on F(v)-v for the discrete steady state
    for it in range(30):
        f0=F(v,z,n,bc,dt,NX)-v
        J=np.empty((v.size,v.size)); eps=1e-9
        for k in range(v.size):
            e=np.zeros(v.size); e[k]=eps*max(abs(v[k]),1e-3); J[:,k]=(F(v+e,z,n,bc,dt,NX)-F(v-e,z,n,bc,dt,NX))/(2*e[k])
        A=J-np.eye(v.size)
        keep=np.ones(v.size,bool); keep[NX]=False  # closed west face qx[:,0] is identically 0
        dv=np.zeros(v.size); dv[keep]=np.linalg.lstsq(A[np.ix_(keep,keep)],-f0[keep],rcond=None)[0]
        v=v+dv
        if np.abs(f0).max()<1e-14: break
    lam=np.linalg.eigvals(J[np.ix_(keep,keep)])
    i=np.abs(lam).argmax()
    return np.abs(lam).max(), lam[i], np.abs(F(v,z,n,bc,dt,NX)-v).max(), v[:NX]/ha-1
for NX in [128]:
    rh,l,res,err=rho(NX); print("T3 as specified: spectral radius %.4f (lambda=%s), fixed-point residual %.1e, discrete-vs-analytic depth err i=8..126 max %.3f%%, i=127 %.3f%%"%(rh,np.round(l,4),res,100*np.abs(err[8:127]).max(),100*err[127]))

def rho_full(NX, nman=0.03, alpha=0.7):
    z,n,ha,q,bc=setup(NX,nman)
    hs=(nman*r*(np.arange(NX)+1)*DX/np.sqrt(S0))**0.6
    dt=alpha*DX/np.sqrt(g*hs.max())
    v=np.concatenate([hs,q])
    for it in range(30):
        f0=F(v,z,n,bc,dt,NX)-v
        J=np.empty((v.size,v.size))
        for k in range(v.size):
            e=np.zeros(v.size); e[k]=1e-9*max(abs(v[k]),1e-3); J[:,k]=(F(v+e,z,n,bc,dt,NX)-F(v-e,z,n,bc,dt,NX))/(2*e[k])
        keep=np.ones(v.size,bool); keep[NX]=False
        dv=np.zeros(v.size); dv[keep]=np.linalg.lstsq((J-np.eye(v.size))[np.ix_(keep,keep)],-f0[keep],rcond=None)[0]; v=v+dv
        if np.abs(f0).max()<1e-14: break
    lam,V=np.linalg.eig(J[np.ix_(keep,keep)]); i=np.abs(lam).argmax()
    vec=np.abs(V[:,i]); hpart=vec[:NX]
    return np.abs(lam).max(), hpart.argmax(), hpart[NX//2]/hpart.max(), dt
rh,loc,mid,dt=rho_full(128); print("eigvec of dominant mode: peak at cell i=%d, amplitude at i=64 relative to peak %.2e"%(loc,mid))
print("alpha sweep (n=0.03): ", ", ".join("a=%.2f rho=%.3f"%(a,rho_full(128,alpha=a)[0]) for a in [0.7,0.5,0.3,0.2,0.1]))
print("n sweep (alpha=0.7): ", ", ".join("n=%.3f rho=%.3f"%(nm,rho_full(128,nman=nm)[0]) for nm in [0.03,0.02,0.01,0.005]))
