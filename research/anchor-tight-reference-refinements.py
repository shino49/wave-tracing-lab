#!/usr/bin/env python3
"""Exact Fraction oracle for true norms and candidate-centered base/delta.

A norm is verified by squared inequalities against the exact extrema of the
input rectangle's squared Euclidean norm, with no approximate square root.
"""
from fractions import Fraction as F
import json,sys

def exact(b):
 m=b&0x7fffff;e=(b>>23)&255
 if e==255:raise ValueError('nonfinite output')
 n=(-1 if b>>31 else 1)*(m+(0x800000 if e else 0));p=e-150 if e else -149
 return F(n<<p) if p>=0 else F(n,1<<-p)

def check(rows):
 failures=[];valid=0;unknown=0;checks=0;norms=0;bases=0
 for i,r in enumerate(rows):
  name=f"{i}/{r['name']}";o=r['output'];bad=o[4]!=0
  if o[7]!=0xac712122:failures.append({'case':name,'error':'missing sentinel'});continue
  if r.get('mustValid') and bad:failures.append({'case':name,'error':'unexpected UNKNOWN'});continue
  if r.get('mustInvalid') and not bad:failures.append({'case':name,'error':'unsafe acceptance'});continue
  if bad:unknown+=1;continue
  valid+=1
  try:
   v=list(map(exact,o[:4]))
   if r['op']==0:
    assert o[5]==1,'accepted norm lost valid marker'
    lo=F(0);hi=F(0)
    for pair in [r['a'][:2],r['b'][:2],r['c'][:2]]:
     l,h=map(exact,pair);assert l<=h,'bad raw interval'
     lo+=0 if l<=0<=h else min(l*l,h*h);hi+=max(l*l,h*h)
    assert v[0]>0 and v[1]>=v[0],'nonpositive or reversed refined norm'
    assert v[0]*v[0]<=lo,'refined norm lower endpoint too high'
    assert v[1]*v[1]>=hi,'refined norm upper endpoint too low'
    assert v[2]>0 and v[3]>=v[2],'bad coarse norm'
    assert v[2]*v[2]<=lo and v[3]*v[3]>=hi,'coarse true norm failed'
    assert v[2]<=v[0]<=v[1]<=v[3],'refinement is not intersection subset'
    checks+=6;norms+=1
   else:
    p,center,error=map(exact,r['a'][:3]);B=F(r['offset'],65536);sgn=-1 if r['sign'] else 1
    for C in [center-error,center+error]:
     delta=p-B-sgn*C;base=B-p+sgn*C
     assert v[0]<=delta<=v[1],'delta under-enclosure'
     assert v[2]<=base<=v[3],'base under-enclosure'
     checks+=2
    bases+=1
  except (AssertionError,ValueError) as e:
   failures.append({'case':name,'error':str(e),'output':o})
 return {'failures':failures[:50],'failureCount':len(failures),'accepted':valid,'unknown':unknown,'normCases':norms,'baseCases':bases,'exactRationalChecks':checks,'oracle':'Python Fraction: exact squared true-norm bounds and affine camera-error endpoint bounds; no approximate sqrt or double oracle'}
print(json.dumps(check(json.load(sys.stdin))))
