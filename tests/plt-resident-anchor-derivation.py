"""Exact dyadic checker; normally invoked by plt-resident-anchor-derivation.mjs.

Input is the generated CPU fixture JSON on stdin. Each primary check unit is
one scalar comparison/identity or one three-component origin invariant; exact
reflection identities are counted separately. All values are reconstructed from
binary64 exactly with Fraction, without f32 interval arithmetic.
"""
import json
import sys
from fractions import Fraction as F
u=F(1,2**53); one=F(1); tau=F(1,2**1075)
checks=0
for k in [F(1),F(2),F(4),F(8),F(16)]:
 hi=(1+u)**4/(1-u)**2*(1+32*k*u)-1
 lo=1-(1-u)**4/(1+u)**2*(1-32*k*u)
 assert max(hi,lo)<39*k*u
 checks+=2
qhi=(1+u)**3/((1-u-u*u)*(1-16*u))
qlo=(1-u)**3/((1+u+u*u)*(1+16*u))
assert qhi-1<21*u and 1-qlo<21*u
assert 2*tau/F(2**-100)**2<u*u
checks+=3
for k in [F(1),F(2),F(4),F(8),F(16)]:
 assert (1+32*k*u)/(1-32*u)-1<128*k*u
 assert 1-(1-32*k*u)/(1+32*u)<128*k*u
 checks+=2
paths=json.load(sys.stdin); hits=0;maxres=0.;maxratio=0.;reflection_checks=0
for p in paths:
 C=list(map(F,p['C']));R=list(map(F,p['R']));N2=sum(x*x for x in R);S=[1,1,1];B=[F(0)]*3;E=[F(0)]*3;a=None;pa=None
 for depth,h in enumerate(p['steps']):
  O=list(map(F,h['O']));HP=list(map(F,h['point']));k=h['axis'];plane=F(h['plane']);V=[S[i]*R[i] for i in range(3)];A=[S[i]*C[i]+B[i] for i in range(3)]
  anchor=F(0) if a is None else (pa-A[a])/V[a]
  ideal=[A[i]+V[i]*anchor for i in range(3)]
  e=[O[i]-ideal[i] for i in range(3)]
  assert all(abs(e[i])<=E[i] for i in range(3))
  delta=plane-O[k];param=(plane-A[k])/V[k]-anchor-e[k]/V[k]
  assert param==delta/V[k]
  t=F(h['t']);cap=128*16*u
  assert N2*(1-cap)**2 <= (t/param)**2 <= N2*(1+cap)**2
  newideal=[A[i]+V[i]*(plane-A[k])/V[k] for i in range(3)];newE=[]
  for i in range(3):
   g=V[i]/V[k]*delta
   local=64*16*u*abs(g)+2048*u
   enew=local if i==k else E[i]+abs(V[i]/V[k])*E[k]+local
   err=abs(HP[i]-newideal[i]);assert err<=enew,(p['cfg'],depth,i,float(err),float(enew))
   newE.append(enew);maxres=max(maxres,float(err));maxratio=max(maxratio,float(err/enew));checks+=1
  if h['mirror']:
   B[k]=2*plane-B[k];S[k]*=-1;a=k;pa=plane;E=newE
   A2=[S[i]*C[i]+B[i] for i in range(3)];V2=[S[i]*R[i] for i in range(3)];anchor2=(pa-A2[a])/V2[a]
   assert [A2[i]+V2[i]*anchor2 for i in range(3)]==newideal
   reflection_checks+=1
  checks+=3;hits+=1
print(json.dumps(dict(paths=len(paths),hits=hits,exactChecks=checks,maxActualUnfoldedResidual=maxres,maxResidualBudgetFraction=maxratio,reflectionIdentityChecks=reflection_checks,pythonVersion=sys.version,checkCountConvention='23 constant inequalities plus six primary check units per hit; reflection identities counted separately',allPass=True)))
