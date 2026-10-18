"""Independent pure-Gaussian free propagation reference (not UTD/PLT).
Uses scalar Fresnel wave equation quadrature, not covariance shear, to check
complex packet fields including curvature, normalization and Gouy phase.
"""
import json, subprocess
from pathlib import Path
import numpy as np
from scipy.integrate import quad
ROOT=Path(__file__).resolve().parents[1]
cases=[]
for b in [0.,-.1,.06]:
 for s in [1.,7.,13.,40.]:
  cases.append(dict(b=b,s=s,point=[.03,-.06]))
code='''import {createGaussianPacket,propagatePacket,packetField} from './dist/wave-packet.js';
const cases=JSON.parse(process.argv[1]);console.log(JSON.stringify(cases.map(c=>{
const p=createGaussianPacket({wavelengthMm:.05,positionCovariance:[.04,0,0,.09],curvature:[c.b,0,0,c.b*.7],powerW:2.5,phaseRad:.3});
return packetField(propagatePacket(p,c.s),c.point);
})));'''
actual=json.loads(subprocess.check_output(['node','--input-type=module','-e',code,json.dumps(cases)],cwd=ROOT,text=True))
k=2*np.pi/.05
rows=[]
for c,got in zip(cases,actual):
 s=c['s'];f=np.sqrt(2.5)*np.exp(1j*(.3-k*s))
 for x,X,b in zip(c['point'],[.04,.09],[c['b'],c['b']*.7]):
  # Physical scalar paraxial Green function for exp(+i omega t).
  def integrand(y):
   return (2*np.pi*X)**(-.25)*np.exp(-y*y/(4*X))*np.exp(-1j*k*b*y*y/2)*np.exp(-1j*k*(x-y)**2/(2*s))
  limit=12*np.sqrt(X)
  val=quad(lambda y:integrand(y).real,-limit,limit,epsabs=2e-11,epsrel=2e-11,limit=1200)[0]+1j*quad(lambda y:integrand(y).imag,-limit,limit,epsabs=2e-11,epsrel=2e-11,limit=1200)[0]
  f*=np.sqrt(1j*k/(2*np.pi*s))*val
 got=complex(*got);error=abs(got-f)
 rows.append(dict(**c,actual=[got.real,got.imag],reference=[f.real,f.imag],absoluteError=error,relativeError=error/max(abs(f),1e-15)))
assert max(r['relativeError'] for r in rows)<2e-10
report=dict(passed=True,reference='Independent direct 1D×1D scalar Fresnel Green-function integrals using SciPy adaptive quadrature',cases=rows,maxRelativeError=max(r['relativeError'] for r in rows),scope='Pure paraxial Gaussian propagation only. No region-scattering operator, mode weights, UTD/DD or PLT validation claim.')
(ROOT/'research/wave-packet-validation.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='cases'},indent=2))
