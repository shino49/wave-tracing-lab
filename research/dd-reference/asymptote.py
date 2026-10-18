"""Restricted sequential UTD *field* asymptote; never an independent DD truth.
Only separated straight parallel z-edges, perpendicular 2D propagation, no
endpoint, grazing, shadow/reflection transition or caustic configurations.
"""
import math
import numpy as np
from scipy.special import erfcx, hankel2
PI=math.pi
E=np.exp(1j*PI/4)
SOURCE=np.array([-1.,-3.]); E1=np.array([0.,1.]); E2=np.array([2.,2.])

def angle(face,v):
    return float(np.arctan2(face[0]*v[1]-face[1]*v[0],np.dot(face,v))%(2*PI))

def coefficient(k,pi,po,L,boundary):
    terms=[]; xs=[]
    for beta,R in [(po-pi,1),(po+pi,-1 if boundary=='soft' else 1)]:
        for sign in [1,-1]:
            N=math.floor((beta+sign*PI)/(4*PI)+.5)
            delta=beta-(4*PI*N-sign*PI)
            x=2*k*L*np.sin(delta/2)**2; xs.append(float(x))
            if abs(delta)<1e-10: raise ValueError('transition boundary is outside benchmark scope')
            cot=1/np.tan((PI+sign*beta)/4)
            F=np.sqrt(PI*x)*E*erfcx(E*np.sqrt(x))
            terms.append(R*cot*F)
    return -np.exp(-1j*PI/4)*sum(terms)/(4*np.sqrt(2*PI*k)), min(xs)

def evaluate(k=8.,receiver=(4.,0.),boundary='soft',source=SOURCE,planar_second=False,planar_both=False,
             edge1=E1,edge2=E2,face1=(0,-1),face2=(0,-1)):
    p=np.asarray(receiver);source=np.asarray(source);e1=np.asarray(edge1);e2=np.asarray(edge2)
    s=np.linalg.norm(e1-source);d=np.linalg.norm(e2-e1);r=np.linalg.norm(p-e2)
    pi1=angle(face1,source-e1);po1=angle(face1,e2-e1)
    pi2=angle(face2,e1-e2);po2=angle(face2,p-e2)
    L1=d if planar_both else s*d/(s+d); L2=r if (planar_second or planar_both) else d*r/(d+r)
    D1,x1=coefficient(k,pi1,po1,L1,boundary);D2,x2=coefficient(k,pi2,po2,L2,boundary)
    inc=-.25j*hankel2(0,k*s)
    u=inc*D1*D2*np.exp(-1j*k*(d+r))/np.sqrt(d*r)
    return {'field':[float(u.real),float(u.imag)],'L1':L1,'L2':L2,
            'leg_lengths':[s,d,r],'angles':[pi1,po1,pi2,po2],
            'min_transition_arguments':[x1,x2],'spreading_factors':[1/np.sqrt(d),1/np.sqrt(r)],
            'path_length':s+d+r}
