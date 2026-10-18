"""Independent 2D scalar Helmholtz FDFD reference for staggered half-plane screens.

No UTD coefficients enter the PDE solver. Complex-coordinate PML, conservative
cell-centred five-point discretization, exact zero-thickness internal screen
conditions. Time exp(+i omega t); outgoing Green -i H_0^(2)(kr)/4.
"""
import argparse, json, time
from pathlib import Path
import numpy as np
from scipy.sparse import diags, kron, eye
from scipy.sparse.linalg import splu


def stretch(t, lower, upper, thickness, k, strength):
    q=np.maximum(lower-t,0)+np.maximum(t-upper,0)
    return 1-1j*strength/k*(q/thickness)**3


def build_axis(lo, hi, h, interior, k, strength):
    n=round((hi-lo)/h)
    assert abs(n*h-hi+lo)<1e-9
    nodes=lo+(np.arange(n)+.5)*h
    faces=lo+np.arange(n+1)*h
    sc=stretch(nodes,*interior,(hi-lo-interior[1]+interior[0])/2,k,strength)
    sf=stretch(faces,*interior,(hi-lo-interior[1]+interior[0])/2,k,strength)
    cl=1/(sc*sf[:-1]*h*h); cr=1/(sc*sf[1:]*h*h)
    A=diags([cl[1:],-cl-cr,cr[:-1]],[-1,0,1],format='csc')
    return nodes,A,cl,cr


def weights(nodes, x):
    h=nodes[1]-nodes[0]; q=(x-nodes[0])/h; i=int(np.floor(q)); a=q-i
    if i<0 or i+1>=len(nodes): raise ValueError('point outside domain')
    return [(i,1-a),(i+1,a)]


def sample(v,x,y,p):
    return sum(wj*wi*v[j,i] for j,wj in weights(y,p[1]) for i,wi in weights(x,p[0]))


def solve(k=8.,h=.05,boundary='soft',screens=2,source=(-1.,-3.),receivers=((4.,0.),),
          bounds=(-4.,7.,-6.,5.), pml_inner=(-2.,5.,-4.,3.),strength=16.,
          screen_specs=((0,0.,1.,-1),(0,2.,2.,-1))):
    """Return unnormalized continuum line-source field and numerical diagnostics."""
    if boundary not in ("soft","hard"): raise ValueError("boundary must be soft or hard")
    if k<=0 or h<=0 or strength<=0: raise ValueError("k, h and PML strength must be positive")
    if not 0<=screens<=len(screen_specs): raise ValueError("unsupported screen count")
    ts=time.time()
    x,Ax,cl,cr=build_axis(*bounds[:2],h,pml_inner[:2],k,strength)
    y,Ay,_,_=build_axis(*bounds[2:],h,pml_inner[2:],k,strength)
    nx,ny=len(x),len(y)
    A=(kron(eye(ny,format='csc'),Ax)+kron(Ay,eye(nx,format='csc'))+k*k*eye(nx*ny,format='csc')).tolil()
    for axis,coordinate,tip,direction in screen_specs[:screens]:
        tangent=y if axis==0 else x
        lo=bounds[0] if axis==0 else bounds[2]
        face=round((coordinate-lo)/h)
        assert abs(lo+face*h-coordinate)<1e-9
        left,right=face-1,face
        rows=np.flatnonzero((tangent-tip)*direction>0)
        for j in rows:
            il=j*nx+left if axis==0 else left*nx+j
            ir=j*nx+right if axis==0 else right*nx+j
            al=A[il,ir]; ar=A[ir,il]
            A[il,ir]=0; A[ir,il]=0
            # Dirichlet at half-cell face requires -2c rather than -c diagonal;
            # homogeneous Neumann removes the corresponding flux altogether.
            sign=-1 if boundary=='soft' else 1
            A[il,il]+=sign*al; A[ir,ir]+=sign*ar
    b=np.zeros(nx*ny,dtype=complex)
    for j,wj in weights(y,source[1]):
        for i,wi in weights(x,source[0]): b[j*nx+i]-=wj*wi/(h*h)
    A=A.tocsc(); setup=time.time()-ts
    lu=splu(A,permc_spec='COLAMD'); u=lu.solve(b)
    residual=np.linalg.norm(A@u-b)/np.linalg.norm(b)
    field=u.reshape(ny,nx)
    vals=[sample(field,x,y,p) for p in receivers]
    return {'k':k,'h':h,'boundary':boundary,'screens':screens,'source':list(source),
            'receivers':[list(p) for p in receivers], 'field':[[float(z.real),float(z.imag)] for z in vals],
            'relative_residual':float(residual),'grid':[nx,ny],'unknowns':nx*ny,
            'setup_seconds':setup,'seconds':time.time()-ts,'bounds':list(bounds),
            'pml_inner':list(pml_inner),'pml_strength':strength,'screen_specs':[list(s) for s in screen_specs[:screens]]}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--k',type=float,default=8);ap.add_argument('--h',type=float,default=.05)
    ap.add_argument('--boundary',choices=['soft','hard'],default='soft');ap.add_argument('--screens',type=int,default=2)
    ap.add_argument('--output');a=ap.parse_args()
    r=solve(k=a.k,h=a.h,boundary=a.boundary,screens=a.screens)
    print(json.dumps(r,indent=2))
    if a.output: Path(a.output).write_text(json.dumps(r,indent=2)+'\n')
if __name__=='__main__': main()
