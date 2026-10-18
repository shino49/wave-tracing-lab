"""Independent double-precision references for the declared browser PLT approximation.

No JavaScript evaluator is imported. SciPy erfcx evaluates local UTD coefficients;
SciPy erfc gives the exact half-plane/Sommerfeld plane-wave control. These certify
local operator arithmetic, not a Maxwell solution for the finite 3-D scene or an
unbiased solid-angle interpretation of the native auxiliary Keller-angle sampler.
"""
import sys, json, math, cmath
from scipy.special import erfcx, erfc
P=math.pi
EP=cmath.exp(1j*P/4)

def cot_f(beta,n,kL,sign,eps):
    nearest=math.floor((beta+sign*P)/(2*P*n)+.5)
    delta=beta-(2*P*n*nearest-sign*P)
    if abs(delta)<=eps:return 0j
    x=2*kL*math.sin(delta/2)**2
    F=math.sqrt(P*x)*EP*erfcx(EP*math.sqrt(x))
    return sign*F/math.tan(delta/(2*n))

def coefficient(e,boundary,eps):
    if e['n']==1:return 0j
    k,n,pi,po,ro,sb=(e[x] for x in ['k','n','phiI','phiO','ro','sinBeta'])
    d=sum(cot_f(po-pi,n,k*ro*sb*sb,sgn,eps) for sgn in (1,-1))
    s=sum(cot_f(po+pi,n,k*ro*sb*sb,sgn,eps) for sgn in (1,-1))
    return -cmath.exp(-1j*P/4)*(d+(-1 if boundary=='soft' else 1)*s)/(2*n*math.sqrt(2*P*k)*sb)

def evaluate(o):
    out={}
    for bc in ['soft','hard']:
        z=complex(o['directAmplitude'])
        for e in o['edges']:
            z+=coefficient(e,bc,o.get('boundaryEpsilon',2e-6))*cmath.exp(-1j*((e['k']*e['phaseExcess'])%(2*P)))/math.sqrt(e['ro'])
        out[bc]=[z.real,z.imag]
    out['F']=sum(abs(complex(*out[bc]))**2 for bc in ['soft','hard'])/2 if o['boundary']=='unpolarized' else abs(complex(*out[o['boundary']]))**2
    return out

def sommerfeld(case):
    k,r,pi,po,bc=(case[x] for x in ['k','r','phiI','phiO','boundary'])
    R=-1 if bc=='soft' else 1
    def term(beta):
        return .5*cmath.exp(1j*k*r*math.cos(beta))*erfc(-EP*math.sqrt(2*k*r)*math.cos(beta/2))
    z=term(po-pi)+R*term(po+pi)
    return [z.real,z.imag]

if __name__=='__main__':
    data=json.load(sys.stdin)
    json.dump({'operators':[evaluate(o) for o in data.get('operators',[])], 'sommerfeld':[sommerfeld(c) for c in data.get('sommerfeld',[])]},sys.stdout,allow_nan=False)
