# Independent 1D re-implementation of spec v1.2 sections 2.1-2.5 + 4.1 (no solver import); spectral radius for v1.1 and v1.2 ghosts.
# Independent 1D re-implementation of spec §2.1-§2.5 + §4.1 (no import of solver.py)
import numpy as np
g=9.81; dx=10.; r=50e-3/3600; S0=0.01; n=0.03; NX=128; hmin=1e-3
z=S0*((NX-1)*dx-np.arange(NX)*dx)
def F(v,dt,ghost):
    h=v[:NX]; q=v[NX:].copy()            # q: NX+1 faces, q[0] west (closed), q[NX] east (open)
    zg=2*z[-1]-z[-2]; hg=h[-1] if ghost=="v1.2" else 0.0
    ze=np.append(z,zg); he=np.append(h,hg); eta=ze+he
    qn=np.zeros(NX+1)
    for k in range(1,NX+1):              # face k between cell k-1 (a) and k (b)
        a,b=k-1,k
        hf=max(eta[a],eta[b])-max(ze[a],ze[b])
        if hf<hmin: continue
        S=(eta[b]-eta[a])/dx
        qn[k]=(q[k]-g*hf*dt*S)/(1+g*dt*n**2*abs(q[k])/hf**(7/3))
    qn[NX]=max(qn[NX],0.0)
    Vout=dt*dx*dx*0+dt*dx*(np.maximum(qn[1:],0)+np.maximum(-qn[:-1],0)); Vc=h*dx*dx
    sc=np.where(Vout>0,np.minimum(1,Vc/np.where(Vout>0,Vout,1)),1.0)
    for k in range(1,NX+1):
        up=k-1 if qn[k]>0 else (k if k<NX else None)
        if up is not None: qn[k]*=sc[up]
    hn=h+dt*(r-(qn[1:]-qn[:-1])/dx)
    return np.concatenate([hn,qn])
def rho(ghost):
    hs=(n*r*(np.arange(NX)+1)*dx/np.sqrt(S0))**0.6; dt=0.7*dx/np.sqrt(g*hs.max())
    v=np.concatenate([hs,np.r_[0,r*(np.arange(NX)+1)*dx]])
    keep=np.ones(v.size,bool); keep[NX]=False
    for it in range(40):
        f0=F(v,dt,ghost)-v
        J=np.empty((v.size,v.size))
        for k in range(v.size):
            e=np.zeros(v.size); e[k]=1e-9*max(abs(v[k]),1e-3); J[:,k]=(F(v+e,dt,ghost)-F(v-e,dt,ghost))/(2*e[k])
        dv=np.zeros(v.size); dv[keep]=np.linalg.lstsq((J-np.eye(v.size))[np.ix_(keep,keep)],-f0[keep],rcond=None)[0]; v=v+dv
        if np.abs(f0).max()<1e-14: break
    lam=np.linalg.eigvals(J[np.ix_(keep,keep)]); return np.abs(lam).max(), lam[np.abs(lam).argmax()], v[NX-1]/hs[-1]-1
for gh in ["v1.2","v1.1"]:
    rh,l,e=rho(gh); print("%s ghost: spectral radius %.4f (lambda %.4f), outlet depth error %.1f%%"%(gh,rh,l.real,100*e))
