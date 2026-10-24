# Tracks the velocity Courant number |u|dt/dx during v1.2 T4a filling. Solver: commit 192f777.
import sys, numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import floodsolver.solver as S
DX=10.; g=9.81
N=129; X=(np.arange(N)-64)*DX; R2=X[None,:]**2+X[:,None]**2; z=5e-4*R2; n=np.full((N,N),0.03)
h=np.zeros((N,N)); qx=np.zeros((N,N+1)); qy=np.zeros((N+1,N))
params={"g":g,"h_min":1e-3}
bc={"edges":{"N":"closed","S":"closed","E":"closed","W":"closed"},"surge_mask":None,"surge_fn":None,"culverts":[],"Ia":np.zeros((N,N)),"f_c":np.zeros((N,N))}
st={"t":0.0,"P_cum":np.zeros((N,N)),"h_max":h.copy(),"eta_max":z+h,"budget":dict(rain_volume=0.,infiltrated_volume=0.,outflow_volume=0.,boundary_inflow_volume=0.,clipped_volume=0.)}
t=0.; nxt=0.; r=0.1/3600
while t<3600*2:
    dt=float(np.clip(0.7*DX/np.sqrt(g*max(h.max(),1e-3)),0.01,60)); dt=min(dt,3600-t) if t<3600 else dt
    st["t"]=t; h,qx,qy,st=S.step(h,qx,qy,z,n,DX,dt,r if t<3600 else 0.0,params,bc,st); t+=dt
    if t>=nxt:
        hx=0.5*(h[:,:-1]+h[:,1:]); m=hx>0.01
        u=np.abs(qx[:,1:-1][m])/hx[m]
        c=np.sqrt(g*h.max())
        print("t=%6.0f dt=%5.2f h_max=%6.3f  max|u|=%5.2f  max|u|dt/dx=%5.2f  max|q|=%6.2f  centre h=%.3f"%(t,dt,h.max(),u.max() if u.size else 0,(u.max() if u.size else 0)*dt/DX,np.abs(qx).max(),h[64,64]))
        nxt+= 300 if t<1800 else 900
