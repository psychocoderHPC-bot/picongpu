import sys, os, time
import numpy as np
sys.path.insert(0, os.path.expanduser("~/pr24/pr24w53-ref"))
import indfield

def gauss(n): return np.polynomial.legendre.leggauss(n)

def box_field(x_obs, x_src, cells, n):
    xn, wn = gauss(n)
    ext = np.array([1.5*cells[0],1.5*cells[1],1.5*cells[2]])
    lo = np.asarray(x_src)-ext; hi = np.asarray(x_src)+ext
    out = np.zeros(3)
    for axis in range(3):
        for sgn in (-1,1):
            fixed = lo[axis] if sgn<0 else hi[axis]
            df = x_obs[axis]-fixed
            o0=(axis+1)%3; o1=(axis+2)%3
            h0=0.5*(hi[o0]-lo[o0]); m0=0.5*(hi[o0]+lo[o0])
            h1=0.5*(hi[o1]-lo[o1]); m1=0.5*(hi[o1]+lo[o1])
            v0=m0+h0*xn[None,:]; v1=m1+h1*xn[:,None]
            d0=v0-x_obs[o0]; d1=v1-x_obs[o1]
            P=np.sum((h0*h1)*wn[None,:]*wn[:,None]/np.sqrt(d0*d0+d1*d1+df*df))
            out[axis]+=sgn*P
    return out

x_obs=np.array([0.3,-0.2,0.1]); x_src=np.zeros(3); cells=np.array([1.0,1.0,1.0])
ref=box_field(x_obs,x_src,cells,128)
print("box_field vs n (maxerr vs n=128):")
for n in [4,8,16,32,64]:
    print(f"  n={n:3d}  {np.max(np.abs(box_field(x_obs,x_src,cells,n)-ref)):.3e}")

def t1(o,r,u):
    t0=time.time(); e,b=indfield.integrate_point(x_obs,np.zeros(3),u,1,1,1,order=o,refine=r); return time.time()-t0,(e,b)
u0=np.zeros(3)
dt,eb=t1(12,3,u0); print(f"\nindfield order=12 refine=3 stationary: {dt:.3f}s  E={eb[0]}")
dt2,eb2=t1(12,4,u0); print(f"indfield order=12 refine=4 stationary: {dt2:.3f}s  maxerr(12/3 vs 12/4)={np.max(np.abs(np.concatenate([eb[0],eb[1]]-np.concatenate([eb2[0],eb2[1]])))):.3e}")
d=np.array([0.6,0.5,0.62]); d/=np.linalg.norm(d); u10=np.sqrt(99.0)*d
dt3,_=t1(12,3,u10); print(f"indfield order=12 refine=3 moving g10: {dt3:.3f}s")
