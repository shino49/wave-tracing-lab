"""Independent geometry-first check of the predeclared 81-case DD grid.

Uses analytic halfplane/cone contact intervals to cross-check the application
triangle routine. No Helmholtz/BEM values or field-error-based selection.
Run: python research/verify-dd-neighbourhood.py
"""
import subprocess,json,math,itertools,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
K=256
rows=[]
def dot(a,b):return sum(x*y for x,y in zip(a,b))
def sub(a,b):return [x-y for x,y in zip(a,b)]
def norm(a):return math.sqrt(dot(a,a))
def clip(r,p,ro):
 q=sub(p,r['center']);t=dot(q,r['cone']['direction']);a=r['radius'];chi=(dot(q,q)-t*t)/a**2+t*t/(4*a*a)
 half=a*math.sqrt(max(0,1-chi));return dict(chi=chi,half=half,threshold=2*math.sqrt(ro/K),margin=half/(2*math.sqrt(ro/K)))
def analytic_contact(c, near, far=float('inf')):
 ox,oy,_=c['origin'];dx,dy,_=c['direction'];r0=c['radius0'];tau=c['tanAngle'];out=[]
 for screen,(x,tip) in enumerate([(0,1),(8,2)]):
  a=x-ox
  lo=max(near,(a-dy*r0)/(dx+dy*tau),dx*a+dy*(-20-oy))
  hi=min(far,(a+dy*r0)/(dx-dy*tau),dx*a+dy*(tip-oy))
  if lo<=hi:out.append((lo,screen))
 return min(out) if out else None
for sigma,aim1,aim2,y in itertools.product([.0675,.07,.0725],[.09,.10,.11],[.15,.175,.2],[-2.25,-2,-1.75]):
 p=dict(k=K,sigma=sigma,aim1=aim1,aim2=aim2,receiver=[16,y,0]);x=json.loads(subprocess.check_output(['node','research/dd-frustum-check.mjs',json.dumps(p)],cwd=ROOT));r=dict(parameters=p,geometry=x['proof'],checks={})
 if not x['proof'].get('passed'):rows.append(r);continue
 ev=x['nativeCentreEvaluation'];regions=[x['first'],x['second']];cs=[]
 for j,reg in enumerate(regions):
  c=reg['cone'];check=analytic_contact(c,c['near']); cs.append(clip(reg,[0,1,0] if j==0 else [8,2,0],ev['ro1'] if j==0 else ev['ro2']))
  assert check and check[1]==j and abs(check[0]-reg['near'])<2e-6,(p,j,check,reg['near'])
  lo,hi=reg['bounds']['min'],reg['bounds']['max'];assert lo[1]>-20 and lo[2]>-20 and hi[2]<20
  # Check the exact screen-plane interval in each finite forward frustum.
  hit=analytic_contact(c,reg['near'],reg['far']);assert hit and hit[1]==j
  for step in x['ballisticTrace'][j]:
   assert abs((step['to']-step['from'])/(16*2*math.pi/K)-round((step['to']-step['from'])/(16*2*math.pi/K)))<1e-8
   for sx,tip in [(0,1),(8,2)]:
    t=(sx-c['origin'][0])/c['direction'][0]
    if step['from']+1e-8<t<step['from']+(step['to']-step['from'])*1.001:
     yy=c['origin'][1]+t*c['direction'][1];assert yy>tip or yy< -20,(p,j,'ballistic blocked',yy)
   assert reg['near']-step['to']>=(c['radius0']+c['tanAngle']*step['to'])/2
 angles=ev['angles'];trans=[];face=[]
 for i in range(2):
  pi,po=angles[2*i:2*i+2];ro=ev['ro1'] if i==0 else ev['ro2'];face.extend([min(pi,2*math.pi-pi),min(po,2*math.pi-po)])
  for beta in [po-pi,po+pi]:
   for sign in [-1,1]:
    N=math.floor((beta+sign*math.pi)/(4*math.pi)+.5);delta=beta-(4*math.pi*N-sign*math.pi)
    trans.append(2*K*ro*math.sin(delta/2)**2)
 # Explicit segment visibility against both full downward screens, excluding only exact segment endpoints.
 vis=[]
 for a,b in [([-2,0,0],[0,1,0]),([0,1,0],x['second']['center']),(x['first']['center'],[8,2,0]),([8,2,0],[16,y,0]),(x['second']['center'],[16,y,0])]:
  margins=[]
  for sx,tip in [(0,1),(8,2)]:
   t=(sx-a[0])/(b[0]-a[0]);
   if 1e-8<t<1-1e-8:margins.append(a[1]+t*(b[1]-a[1])-tip)
  assert all(m>0 for m in margins),(p,a,b,margins)
  vis.extend(margins)
 r['checks']=dict(clips=cs,minTransitionArgument=min(trans),minFaceAngle=min(face),minKr=K*min(ev['ro1'],ev['ro2']),ballisticAnalyticPassed=True,analyticContactPassed=True,finiteMeshBoundariesClear=True,minSegmentVisibilityMargin=min(vis) if vis else None)
 r['passed']=all(c['margin']>=1 for c in cs) and min(trans)>=10 and min(face)>=.001
 rows.append(r)
summary=dict(scope='Geometry-first fixed Cartesian parameter grid; no BEM values accessed',count=len(rows),geometricPass=sum(r['geometry'].get('passed',False) for r in rows),fullPass=sum(r.get('passed',False) for r in rows),minClippedMargins=[min(r['checks']['clips'][j]['margin'] for r in rows if r['checks']) for j in [0,1]],minTransitionArgument=min(r['checks']['minTransitionArgument'] for r in rows if r['checks']),minFaceAngle=min(r['checks']['minFaceAngle'] for r in rows if r['checks']),minKr=min(r['checks']['minKr'] for r in rows if r['checks']),failures=[r for r in rows if not r.get('passed')])
summary['contactRoutineSHA256']=hashlib.sha256((ROOT/'research/dd-frustum-check.mjs').read_bytes()).hexdigest()
summary['sourceGaussianPhaseCurvatureRangeRad']=[]
phase=[]
for sigma,aim in itertools.product([.0675,.07,.0725],[.09,.10,.11]):
 dx=2/math.hypot(2,1+aim);dy=(1+aim)/math.hypot(2,1+aim)
 tau=3/(2*K*sigma);r0=3*sigma
 t=(2-dy*r0)/(dx+dy*tau);a=r0+tau*t;zR=2*K*sigma*sigma
 phase.append(.5*K*a*a*t/(t*t+zR*zR))
summary['sourceGaussianPhaseCurvatureRangeRad']=[min(phase),max(phase)]
summary['oldCollimatedGaussianPhaseGuardRad']=.25
summary['oldCollimatedGaussianPhaseGuardPassed']=max(phase)<=.25
summary['limitations']=[
 'Finite Cartesian samples, not a proof for every point in the continuous box.',
 'Centre-factor native approximation, not calibrated Gaussian radiance.',
 'No exact outgoing scattering covariance or coupled-aperture certificate.',
 'The old one-region collimated Gaussian curvature guard fails explicitly.',
]
(ROOT/'research/dd-neighbourhood-independent.json').write_text(json.dumps(dict(summary=summary,rows=rows),indent=2)+'\n')
print(json.dumps(summary,indent=2))
