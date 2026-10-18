"""Exact 2D wedge Green function by separation of variables, n=2 half-plane.

For exterior angle alpha=n*pi, angular eigenfunctions sin(m*phi/n) (soft)
or cos(m*phi/n) (hard) produce a radial J H_2 expansion. This solves the
Helmholtz PDE with -delta source and outgoing exp(-ikr), independently of UTD.
"""
import numpy as np
from scipy.special import jv, hankel2

def green(k,r,phi,r_source,phi_source,boundary='soft',n=2.,terms=None):
    low,high=sorted([r,r_source])
    if abs(high-low)<1e-8: raise ValueError('equal-radius convergence not supported by this simple evaluator')
    if terms is None: terms=int(n*(k*low+70))
    m=np.arange(1,terms+1);order=m/n
    radial=jv(order,k*low)*hankel2(order,k*high)
    angular=(np.sin(order*phi)*np.sin(order*phi_source) if boundary=='soft'
             else np.cos(order*phi)*np.cos(order*phi_source))
    value=-1j/n*np.sum(radial*angular)
    if boundary=='hard': value+=-1j/(2*n)*jv(0,k*low)*hankel2(0,k*high)
    return value
