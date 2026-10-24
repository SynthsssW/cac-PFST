# Checks the v1.2 scheme preserves a lake at rest exactly (well-balanced) for several alpha. Solver: commit 192f777.
import sys, numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from floodsolver.solver import run
DX=10.; C={"N":"closed","S":"closed","E":"closed","W":"closed"}
N=129; X=(np.arange(N)-64)*DX; z=5e-4*(X[None,:]**2+X[:,None]**2); n=np.full((N,N),0.03)
# start from the exact lake at rest (eta = 2.5 m), no rain: an exact discrete steady state
eta0=2.5; h0=np.maximum(eta0-z,0.0)
for a in [0.7,0.5,0.35]:
    res=run(z,n,DX,lambda t:0.0,1800.,h0=h0,edges=C,alpha=a,record_every=300.)
    s=[((z+hs)[hs>0.01]-eta0) for t,hs in res.snapshots]
    print("alpha=%.2f  lake-at-rest max|eta-2.5| at 300s steps:"%a, " ".join("%.1e"%np.abs(x).max() for x in s))
res=run(z,n,DX,lambda t:0.0,1800.,h0=h0,edges=C,alpha=0.7)
d=(z+res.h_final)-eta0; d[res.h_final<=0.01]=np.nan
j,i=np.unravel_index(np.nanargmax(np.abs(d)),d.shape); print("worst cell (%d,%d) r=%.0f m depth %.3f; local eta-2.5 pattern:"%(j,i,np.hypot(j-64,i-64)*10,res.h_final[j,i]))
print(np.round(d[j-2:j+3,i-2:i+3],3))
