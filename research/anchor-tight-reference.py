#!/usr/bin/env python3
"""Independent exact-Fraction oracle for native tight-helper endpoint outputs.

No shader recentering, f32 arithmetic, padding, or endpoint algorithm is copied.
IEEE-754 f32 bit patterns are reconstructed as exact rational numbers; accepted
outputs must contain exact corner values of A/B, A/B-q, and C+(A/B)*Delta.
"""
from fractions import Fraction as F
from itertools import product
import json,sys

def exact(b):
    m=b&0x7fffff;e=(b>>23)&255
    if e==255:raise ValueError('nonfinite GPU output')
    n=(-1 if b>>31 else 1)*(m+(0x800000 if e else 0));p=e-150 if e else -149
    return F(n<<p) if p>=0 else F(n,1<<-p)

def check(rows):
    fail=[];corners=0;valid=0;unknown=0;point_intervals=0;residual_intervals=0;separate_evaluation_widening=[]
    for i,r in enumerate(rows):
        out=r['output'];name=f"{i}/{r['name']}"
        if out[15]!=0x1a51c021:fail.append({'case':name,'error':'dispatch sentinel missing'});continue
        bad=out[12]!=0
        if r.get('mustValid') and bad:fail.append({'case':name,'error':'unexpected UNKNOWN'});continue
        if r.get('mustInvalid') and not bad:fail.append({'case':name,'error':'unsafe failure accepted'});continue
        if bad:unknown+=1;continue
        valid+=1
        try:
            a,b,d,c=[list(map(exact,r[k])) for k in ['a','b','delta','base']]
            v=list(map(exact,out[:12]))
            if r['op']==1:
                l,h=max(a[0],b[0]),min(a[1],b[1])
                assert l<=h and v[0]==l and v[1]==h,'intersection mismatch'
                continue
            assert not b[0]<=0<=b[1],'accepted denominator contains zero'
            q=v[8]
            for x,y in product(a,b):
                ratio=x/y
                assert v[0]<=ratio<=v[1],'coarse ratio under-enclosure'
                assert v[2]<=ratio<=v[3],'tight ratio under-enclosure'
                assert v[9]<=ratio-q<=v[10],'centered residual under-enclosure'
                corners+=3;residual_intervals+=1
                for z,w in product(d,c):
                    point=w+ratio*z
                    assert v[4]<=point<=v[5],'coarse point under-enclosure'
                    assert v[6]<=point<=v[7],'tight point under-enclosure'
                    corners+=2;point_intervals+=1
            if not (v[0]<=v[2]<=v[3]<=v[1] and v[4]<=v[6]<=v[7]<=v[5]):
                separate_evaluation_widening.append({'case':name,'bits':out[:8],'values':[float(x) for x in v[:8]]})
        except (AssertionError,ValueError,ZeroDivisionError) as e:
            if len(fail)<50:fail.append({'case':name,'error':str(e)})
    return {'failures':fail,'accepted':valid,'unknown':unknown,'exactRationalCornerChecks':corners,'pointCornerInstances':point_intervals,'ratioResidualCornerInstances':residual_intervals,'separateEvaluationWidening':separate_evaluation_widening,'oracle':'Exact Python Fraction from IEEE-754 f32 input/output bits; no host-double numerical oracle'}
print(json.dumps(check(json.load(sys.stdin))))
