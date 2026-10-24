# Runs v1.2 T3 and prints depth/flux vs analytic, showing the face-to-face flux sawtooth. Solver: commit 192f777.
import sys, numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from floodsolver.solver import run
from floodsolver.analytic import tilted_plane_steady_h
NX=NY=128; DX=10.; r=50e-3/3600; S0=0.01
X=np.arange(NX)*DX; z=np.broadcast_to(S0*((NX-1)*DX-X),(NY,NX)).copy()
for t_end in [3600., 4*3600.]:
    res=run(z,np.full((NY,NX),0.03),DX,lambda t:r,t_end,edges={"N":"closed","S":"closed","E":"open","W":"closed"})
    h=res.h_final[0]; ha=tilted_plane_steady_h((np.arange(NX)+1)*DX,r,0.03,S0)
    Q=[q for _,q in res.outflow_series][-5:]
    print("t_end",t_end,"Q last",np.round(Q,3),"target",r*NX*NY*DX**2)
    for i in [0,1,2,4,8,16,32,64,100,120,126,127]: print(i, "%.5f %.5f %.3f"%(h[i],ha[i],h[i]/ha[i]))
    print("qx row0 last faces", res.qx[0,-4:], "expected", r*(np.arange(NX-5,NX-1)+1)*DX)
