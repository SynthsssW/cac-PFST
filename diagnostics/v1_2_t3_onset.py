# Locates the onset of the v1.2 T3 instability (outlet cell i=127) and checks the limiter is not binding. Solver: commit 192f777.
import sys, numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
exec(open(__import__("pathlib").Path(__file__).with_name("v1_2_t3_from_steady.py")).read().split("for k in range")[0])
import floodsolver.solver as S
for k in range(60):
    dt=float(np.clip(0.7*DX/np.sqrt(g*max(h.max(),1e-3)),0.01,60))
    h_prev=h
    h,qx,qy,st=S.step(h,qx,qy,z,n,DX,dt,r,params,bc,st)
    # limiter binding? recompute V_out ratio from limited q (scale<1 means V_out==V_cell)
    Vout=dt*DX*(np.maximum(qx[:,1:],0)+np.maximum(-qx[:,:-1],0))
    bind=np.isclose(Vout[0],h_prev[0]*DX*DX,rtol=1e-12)&(Vout[0]>0)
    e=np.abs(h[0]/ha-1)
    if k%5==0 or k<4: print(k,"argmax err i=%d err=%.2e"%(e.argmax(),e.max()),"errs i=0..5",np.round(e[:6],4),"limiter binds at",np.where(bind)[0][:8])
