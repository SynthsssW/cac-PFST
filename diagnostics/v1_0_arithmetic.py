# Checks spec v1.0 T3 limiter cap, T4b equilibrium depth and T8 drain time by arithmetic on the spec formulas (no solver).
import numpy as np
g=9.81; dx=10.0; alpha=0.7; nx=128; S0=0.01; n=0.03; r=50e-3/3600
# T3: analytic kinematic h at cell i (x = i*dx), required face flux q_face(i+1/2)=r*(i+1)*dx
i=np.arange(nx); x=i*dx
ha=(n*r*x/np.sqrt(S0))**0.6
dt=alpha*dx/np.sqrt(g*max(ha.max(),1e-3))
q_req=r*(i+1)*dx                     # steady flux out of cell i's east face
q_cap=ha*dx/(4*dt)                   # §2.4 cap using analytic depths
bind=q_req>q_cap
print("T3 dt=%.3f s  h_out=%.4f m  limiter binds for i>=%d (x>=%.0f m), q_req/q_cap at outlet=%.2f"%(dt,ha[-1],i[bind].min(),x[bind].min(),q_req[-1]/q_cap[-1]))
# self-consistent outlet depth if limited: r*L_face = H*sqrt(gH)/(4 alpha)
H=(4*alpha*r*nx*dx/np.sqrt(g))**(2/3)
print("T3 limited outlet depth >= %.4f m vs analytic %.4f m (+%.0f%%)"%(H,ha[-1],100*(H/ha[-1]-1)))
# x-convention sensitivity at i=8 (unlimited region): discrete face balance gives h_i ~ (n r (i+1)dx/sqrtS0)^0.6
print("T3 i=8: analytic(x=i*dx)/discrete(x=(i+1)dx) = %.4f"%((8/9)**0.6))
# T4b: 129 grid bowl, 10x10 block of 2 m -> discrete equilibrium eta
N=129; c=64; X=(np.arange(N)-c)*dx
z=5e-4*(X[None,:]**2+X[:,None]**2)
V=100*2.0*dx*dx
lo,hi=0,10
for _ in range(200):
    e=(lo+hi)/2
    (lo,hi)=(e,hi) if (np.maximum(e-z,0).sum()*dx*dx)<V else (lo,e)
print("T4b equilibrium eta=%.4f m -> final centre depth %.4f m > 2.0"%(e,e-z[c,c]))
# T8 if basins were the two halves of a 128x128 flat grid: time for head 1 m -> 1e-3 m
A=63*128*dx*dx; k=0.6*1.0*np.sqrt(2*g)/A
print("T8 half-grid basins: equalise to 1e-3 m needs %.1f h (run is 6 h)"%((1-np.sqrt(1e-3))/k/3600))
