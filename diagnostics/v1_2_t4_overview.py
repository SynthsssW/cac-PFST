# v1.2 T4a/T4b lake flatness and eta_max overview. Solver: commit 192f777.
import sys, numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from floodsolver.solver import run
DX=10.; C={"N":"closed","S":"closed","E":"closed","W":"closed"}
N=129; X=(np.arange(N)-64)*DX; z=5e-4*(X[None,:]**2+X[:,None]**2); n=np.full((N,N),0.03)
# T4a
res=run(z,n,DX,lambda t:0.1/3600 if t<3600 else 0.0,7*3600.,edges=C,rain_breakpoints=[3600.],record_every=3600.)
h=res.h_final; eta=z+h; wet=h>0.01
lake=wet&(z<7.4)
print("T4a: wet cells %d; lake-core eta mean %.4f std %.2e; wet cells with z>8: %d, their h range %.3f..%.3f"%(wet.sum(),eta[lake].mean(),eta[lake].std(),(wet&(z>8)).sum(),h[wet&(z>8)].min() if (wet&(z>8)).any() else 0,h[wet&(z>8)].max() if (wet&(z>8)).any() else 0))
print("    film h on slopes (z>20): max %.4f, cells with h>h_min: %d; clipped %.2e"%(h[z>20].max(),(h[z>20]>1e-3).sum(),res.budget["clipped_volume"]))
for t,hs in res.snapshots: print("    t=%5.0f  lake std %.2e  max h on z>20 %.4f"%(t,(z+hs)[lake].std(),hs[z>20].max()))
# T4b
h0=np.zeros((N,N)); h0[60:70,60:70]=2.0
res=run(z,n,DX,lambda t:0.0,6*3600.,h0=h0,edges=C,record_every=600.)
h=res.h_final; eta=z+h; wet=h>0.01
print("T4b: eta_eq %.4f std %.2e; eta_max(all) %.3f vs init %.3f; eta_max dry0&wet %.3f; std over time:"%(eta[wet].mean(),eta[wet].std(),res.eta_max.max(),(z+h0).max(),res.eta_max[(h0==0)&wet].max()), " ".join("%.0f:%.1e"%(t,(z+s)[s>0.01].std()) for t,s in res.snapshots[::6]))
