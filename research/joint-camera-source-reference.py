#!/usr/bin/env python3
"""Exact independent oracle for the CPU source-to-ideal-affine bridge.

Uses actual binary64 input values as exact rational numbers. The affine ideal
uses exact real arithmetic, not rounded host coefficient construction and not
an emulation of any proposed GPU helper. A separate production-upload bound
will still be necessary. No normalized-direction substitution is used.
"""
from fractions import Fraction as F
import json,sys

def q(x): return F.from_float(x) if isinstance(x,float) else F(x)
u=F(1,2**53);tau=F(1,2**1075)
checks=0;worst=F(0);fail=[];rounding_witness=None;jitter_witness=None
rows=json.load(sys.stdin)
for index,row in enumerate(rows):
 W,H,x,y=map(q,[row['width'],row['height'],row['x'],row['y']]);half=q(row['half']);j0=F(row['word0'],2**32);j1=F(row['word1'],2**32)
 # These algebraic identities use exact rational W/H. Actual CPU division,
 # aspect and all raw component mul/add operations remain on the other side.
 U=(2*(x+j0)/W-1)*(W/H)*half;V=(1-2*(y+j1)/H)*half
 for k in range(3):
  f,r,p=map(q,[row['frame']['forward'][k],row['frame']['right'][k],row['frame']['up'][k]])
  actual=q(row['raw'][k]);A=f-r*(W/H)*half+p*half;B=2*r*half/H;C=-2*p*half/H
  exact=A+B*(x+j0)+C*(y+j1)
  assert exact==f+r*U+p*V
  budget=32*u*(abs(f)+abs(r)*(W/H)*abs(half)+abs(p)*abs(half))+128*tau;error=abs(actual-exact)
  checks+=1
  if error>budget and len(fail)<20:fail.append({'row':index,'axis':k,'error':float(error),'budget':float(budget)})
  if budget:worst=max(worst,error/budget)
  if error and rounding_witness is None:rounding_witness={'row':index,'axis':k,'exactError':str(error),'error':float(error),'removedBudgetDetected':True}
  truncated=A+B*(x+F(row['word0']>>8,2**24))+C*(y+F(row['word1']>>8,2**24))
  if abs(actual-truncated)>budget and jitter_witness is None:jitter_witness={'row':index,'axis':k,'error':float(abs(actual-truncated)),'budget':float(budget),'removedLowEightJitterBitsDetected':True}
# Exact symbolic coefficient bounds used to justify the retained loose 32u.
# t=2(x+j)/W in [0,2]; rounding t then subtracting 1 incurs <=4u.
# u: multiply by rounded aspect and half, using |s|<=1. v has no aspect.
uhat=(1+4*u)*(1+u)**3-1
vhat=(1+4*u)*(1+u)-1
# Raw output adds two rounded products, a rounded inner sum, then outer sum.
# Per r/p term there are at most three additional (1+u) factors; f one.
rbound=(1+uhat)*(1+u)**3-1
pbound=(1+vhat)*(1+u)**3-1
assert rbound<32*u and pbound<32*u and u<32*u
assert 704*u+128*tau<F(1,2**38)
checks+=4
assert rounding_witness and jitter_witness
print(json.dumps({'allPass':not fail,'failures':fail,'exactRationalChecks':checks,'maxSourceBudgetFraction':float(worst),'mutations':[rounding_witness,jitter_witness],'oracle':'Exact Fraction of actual binary64 sampleCamera source inputs and outputs; algebraic affine ideal; no GPU padding emulation','qualifications':['Reference/source bridge evidence only; immutable coefficient upload and native arithmetic require separate validation','Actual normalize(raw) is bitwise equal to sampleCamera.direction in the JS harness','32u64 plus128tau bound assumes nearest-rounded binary64 basic arithmetic with gradual underflow; geometry/hypot contracts unchanged']}))
