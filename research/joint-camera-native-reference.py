#!/usr/bin/env python3
"""Independent exact-integer/Fraction oracle for native camera outputs.

No shader limb multiplication, carry, division, conversion or interval algorithm
is duplicated: Python integers and Fraction directly establish the mathematical
answer. Accepted camera ratios must contain every independently perturbed raw
corner; actual CPU raw is checked separately against uploaded affine state.
"""
from fractions import Fraction as F
from itertools import product
import json,sys

def fp32(b):
 m=b&0x7fffff;e=(b>>23)&255
 if e==255:raise ValueError('nonfinite native result')
 n=(-1 if b>>31 else 1)*(m+(0x800000 if e else 0));p=e-150 if e else -149
 return F(n<<p) if p>=0 else F(n,1<<-p)
def u64(p):return p[0]+(p[1]<<32)
def i64(p):
 n=u64(p);return n-(1<<64) if n>>63 else n
def trunc(n,d):return (-1 if n<0 else 1)*(abs(n)//d)
def q(x):return F.from_float(x) if isinstance(x,float) else F(x)
def check(rows):
 fail=[];total_fail=0;counts={};accepted=0;unknown=0;checks=0;camera_accepted=0;max_source_fraction=F(0)
 def bad(name,error):
  nonlocal total_fail
  total_fail+=1
  if len(fail)<40:fail.append({'case':name,'error':str(error)})
 for number,r in enumerate(rows):
  name=f"{number}/{r['name']}";o=r['output'];op=r['op'];counts[str(op)]=counts.get(str(op),0)+1
  if o[3]!=0xca221122:bad(name,'native sentinel absent');continue
  rejected=bool(o[0] or not o[1])
  if r.get('mustValid') and rejected:bad(name,'unexpected UNKNOWN');continue
  if r.get('mustInvalid') and not rejected:bad(name,'unsafe acceptance');continue
  if rejected:unknown+=1;continue
  accepted+=1
  try:
   a=i64(r['a']);b=i64(r['b']);ua=u64(r['a']);ub=u64(r['b'])
   if op==0:assert u64(o[4:6])==r['a'][0]*r['b'][0],'32-bit product differs from exact integer';checks+=1
   elif op==1:assert i64(o[4:6])==a*r['b'][0],'signed integer product mismatch';checks+=1
   elif op==2:assert i64(o[4:6])==trunc(a*r['b'][0],1<<32),'signed jitter truncation mismatch';checks+=1
   elif op==3:
    lo,hi=map(fp32,o[4:6]);assert lo<=F(a,1<<r['scale'])<=hi,'directed integer conversion fails containment';checks+=1
   elif op==4:assert u64(o[4:6])==(ua<<32)//ub,'long division is not exact floor';checks+=1
   elif op==7:assert i64(o[4:6])==a+b,'signed addition/carry mismatch';checks+=1
   elif op==5:
    c,lo,hi=map(fp32,o[4:7]);assert lo<=hi,'reversed ratio residual'
    for n,d in product([a-r['ne'],a+r['ne']],[b-r['de'],b+r['de']]):
     ratio=F(n,d);assert c+lo<=ratio<=c+hi,'ratio corner under-enclosure';checks+=1
   else:
    camera_accepted+=1;p=r['parameters'];W,H,_,zero=p[24:28];x=r['pixel']%W;y=r['pixel']//W;z=[];err=[]
    for k in range(3):
     A,B,C=[i64(p[j:j+2]) for j in range(k*6,k*6+6,2)]
     expected=A+B*x+C*y+trunc(B*r['word0'],1<<32)+trunc(C*r['word1'],1<<32)
     actual=i64(o[4+k*2:6+k*2]);assert actual==expected,'affine raw integer mismatch';checks+=1;z.append(expected);err.append(0 if zero&(1<<k) else 512)
     lo,hi=map(fp32,o[10+k*2:12+k*2]);assert lo<=F(expected-err[k],1<<44) and F(expected+err[k],1<<44)<=hi,'raw error envelope under-enclosure';checks+=2
     if 'reference' in r:
      ref=r['reference'];f,rv,uv=map(q,[ref['frame']['forward'][k],ref['frame']['right'][k],ref['frame']['up'][k]]);h=q(ref['half']);aspect=q(ref['aspect'])
      coeffs=[f-rv*aspect*h+uv*h,2*rv*aspect*h/W,-2*uv*h/H]
      for integer,ideal in zip([A,B,C],coeffs):assert abs(F(integer,1<<44)-ideal)<=F(1,1<<45),'host coefficient not nearest Q44';checks+=1
      raw=q(ref['raw'][k]);distance=abs(raw-F(expected,1<<44));assert distance<=F(err[k],1<<44),'actual CPU raw escapes fixed camera bound';checks+=1
      if err[k]:max_source_fraction=max(max_source_fraction,distance/F(err[k],1<<44))
      assert lo<=raw<=hi,'actual CPU raw escapes native raw endpoints';checks+=1
    for k in range(3):
     if not z[k] and not err[k]:continue
     for i in range(3):
      if i==k:continue
      index=k*2+i-(1 if i>k else 0);at=16+index*5;c,lo,hi,full_lo,full_hi=map(fp32,o[at:at+5]);assert lo<=hi and full_lo<=full_hi,'reversed cached ratio interval';sgn=-1 if ((r['signMask']>>i)^(r['signMask']>>k))&1 else 1
      for numerator,denominator in product([z[i]-err[i],z[i]+err[i]],[z[k]-err[k],z[k]+err[k]]):
       ratio=sgn*F(numerator,denominator);assert c+lo<=ratio<=c+hi,'cached residual ratio corner under-enclosure';assert full_lo<=ratio<=full_hi,'cached full ratio corner under-enclosure';checks+=2
      if 'reference' in r:
       exact=sgn*q(r['reference']['raw'][i])/q(r['reference']['raw'][k]);assert c+lo<=exact<=c+hi,'actual CPU raw ratio under-enclosure';checks+=1
  except (AssertionError,ValueError,ZeroDivisionError) as e:bad(name,e)
 return {'failureCount':total_fail,'failures':fail,'accepted':accepted,'unknown':unknown,'cameraAccepted':camera_accepted,'casesByOperation':counts,'exactIntegerAndFractionChecks':checks,'maxCPUFixedBudgetFraction':float(max_source_fraction),'oracle':'Exact Python integer/Fraction mathematical target from native input/output bits; not an emulation of shader limb, division, or outward rounding algorithms'}
print(json.dumps(check(json.load(sys.stdin))))
