"""Independent Fresnel Hankel quadrature oracle, no production implementation.
Rebuild: python research/pinhole-independent-oracle.py
SciPy Gauss-Legendre quadrature and scipy.special.j0, checked 512 vs 1024 nodes.
Lengths in mm. References preserve analytic infinite-plane normalization.
"""
import json
from pathlib import Path
import numpy as np
from scipy.special import roots_legendre, j0

rules = {}
for order in (512, 1024):
    nodes, weights = roots_legendre(order)
    rules[order] = ((nodes + 1) / 2, weights / 2)

def density(d, s, nm, r, order):
    t, w = rules[order]
    a, lam, z = d / 2, nm * 1e-6, 50
    beta = np.pi * a*a*(1/s + 1/z) / lam
    q = 2*np.pi*a*r/(lam*z)
    amplitude = np.sum(w * 2*t * np.exp(1j*beta*t*t) * j0(q*t))
    return float(np.pi*a*a/(lam*z)**2 * abs(amplitude)**2)

cases=[]
max_error=0
for d in (.04, .15, .5):
    for s in (200, 300):
        for nm in (360, 550, 830):
            peak_scale = np.pi*(d/2)**2/((nm*1e-6)*50)**2
            for r in (0, .003, .018, .036, .1, .5, 1, 4, 8, 11.313708498984761):
                low, high = density(d,s,nm,r,512), density(d,s,nm,r,1024)
                error=abs(low-high)/peak_scale
                max_error=max(max_error,error)
                if error>1e-11: raise RuntimeError(f'Oracle quadrature failed: {d,s,nm,r,error}')
                cases.append(dict(diameterMm=d,depthMm=s,wavelengthNm=nm,radiusMm=r,density=high,quadratureErrorScaled=error))
output=dict(method='Independent 1024-node Gauss-Legendre Hankel integral using scipy.special.j0; no Zernike expansion or implementation import',normalization='Analytic aperture-area normalization; no finite-crop rescaling',max512To1024ErrorScaled=max_error,cases=cases)
path=Path(__file__).with_name('pinhole-independent-oracle.json')
path.write_text(json.dumps(output,indent=2)+'\n')
print(f'{len(cases)} independent cases; max scaled quadrature difference {max_error:.3e}')
