# Tracks a 1 cm bump on a lake at rest under v1.2 for several alpha. Solver: commit 192f777.
import sys, numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from floodsolver.solver import run
DX=10.; C={"N":"closed","S":"closed","E":"closed","W":"closed"}
N=129; X=(np.arange(N)-64)*DX; R2=X[None,:]**2+X[:,None]**2; z=5e-4*R2; n=np.full((N,N),0.03)
eta0=2.5; h0=np.maximum(eta0-z,0.0); wet0=h0>0
h0=h0+wet0*0.01*np.exp(-R2/(20.**2))       # 1 cm smooth bump at centre
for a in [0.7,0.5,0.35]:
    res=run(z,n,DX,lambda t:0.0,3600.,h0=h0,edges=C,alpha=a,record_every=300.)
    out=[]
    for t,hs in res.snapshots:
        d=(z+hs)-eta0; core=R2<(50.**2); shore=wet0&~core
        out.append("%.0e/%.0e"%(np.abs(d[core]).max(),np.abs(d[shore&(hs>0.01)]).max()))
    print("alpha=%.2f  max|eta-eta0| core/shore every 300 s:"%a," ".join(out))
