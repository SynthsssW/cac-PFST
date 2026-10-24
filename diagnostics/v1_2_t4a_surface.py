# Prints the v1.2 T4a final surface along the centre row. Solver: commit 192f777.
import sys, numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from floodsolver.solver import run
DX=10.; C={"N":"closed","S":"closed","E":"closed","W":"closed"}
N=129; X=(np.arange(N)-64)*DX; z=5e-4*(X[None,:]**2+X[:,None]**2); n=np.full((N,N),0.03)
res=run(z,n,DX,lambda t:0.1/3600 if t<3600 else 0.0,7*3600.,edges=C,rain_breakpoints=[3600.])
h=res.h_final; eta=z+h
np.set_printoptions(linewidth=200)
print("centre row j=64, i=40..88: h"); print(np.round(h[64,40:89],2))
print("eta"); print(np.round(eta[64,40:89],2))
print("qx centre row faces 52..76:", np.round(res.qx[64,52:76],3))
