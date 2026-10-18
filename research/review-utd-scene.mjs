// Independent end-to-end CPU scene check against exact Sommerfeld fields.
import fs from 'node:fs';
import assert from 'node:assert/strict';
import {execFileSync} from 'node:child_process';
import {prepareUTDScene,evaluatePoint,builtInSceneOBJ} from '../dist/utd-scene.js';
import {prepareMesh,transformPoint} from '../dist/mesh-geometry.js';
const fixture=JSON.parse(execFileSync('python',['-c',`
import json,numpy as np
from scipy.special import erfc
out=[]
k=2*np.pi/.05
for mode,theta,Z,sigma,sourceDistance in [('forward',0,20,1,5),('forward-oblique',.35,20,1,5),('reflection',0,-70,10,100)]:
 for X in [-1.,-.3,-1e-5,0,1e-5,.3,1.]:
  Y=Z*np.tan(theta); r=np.hypot(X,Z); kt=k*np.cos(theta)
  pi=np.pi/2;po=np.arctan2(-Z,X)%(2*np.pi)
  def B(beta):return .5*np.exp(1j*kt*r*np.cos(beta))*erfc(-np.exp(1j*np.pi/4)*np.sqrt(2*kt*r)*np.cos(beta/2))
  soft=(B(po-pi)-B(po+pi))*np.exp(1j*kt*Z)
  hard=(B(po-pi)+B(po+pi))*np.exp(1j*kt*Z)
  out.append({'mode':mode,'sigma':sigma,'sourceDistance':sourceDistance,'theta':theta,'destination':[X,Y,Z],'soft':[soft.real,soft.imag],'hard':[hard.real,hard.imag]})
print(json.dumps(out))
`],{encoding:'utf8'}));
// A finite plate large enough that the one gathered region contains only the
// relevant physical edge; the clipped saddle is many Fresnel widths from ends.
const plate='v 0 -300 0\nv 300 -300 0\nv 300 300 0\nv 0 300 0\nf 1 2 3\nf 1 3 4\n';
let maxError=0;
const rows=[];
for(const f of fixture){
 const di=[0,Math.sin(f.theta),Math.cos(f.theta)];
 const scene=prepareUTDScene(plate,{sourceOrigin:di.map(v=>-f.sourceDistance*v),incidentDirection:di,sigma0Mm:f.sigma});
 const r=evaluatePoint(scene,f.destination,{details:true});
 assert(r.valid,JSON.stringify(r.invalidReasons));
 const error=Math.max(Math.hypot(r.soft[0]-f.soft[0],r.soft[1]-f.soft[1]),Math.hypot(r.hard[0]-f.hard[0],r.hard[1]-f.hard[1]));
 maxError=Math.max(maxError,error);
 rows.push({mode:f.mode,theta:f.theta,destination:f.destination,edgeCount:r.edgeCount,directWeight:r.directWeight,error});
 assert(error<1e-8,`scene vs exact halfplane ${error}`);
}
let rigidMax=0;
const tr={rotationDeg:[23,31,17],translation:[2,-3,4]},rd={rotationDeg:tr.rotationDeg};
for(const name of ['screen','wedge','box']){
 const obj=builtInSceneOBJ(name),a=prepareUTDScene(obj);
 const b=prepareUTDScene(prepareMesh(obj,tr),{sourceOrigin:transformPoint([0,0,-5],tr),incidentDirection:transformPoint([0,0,1],rd),detectorCenter:transformPoint([0,0,20],tr),detectorU:transformPoint([1,0,0],rd),detectorV:transformPoint([0,1,0],rd)});
 for(const P of [[0,0,20],[2,0,20],[.2,.3,20]]){
  const x=evaluatePoint(a,P),y=evaluatePoint(b,transformPoint(P,tr));
  assert(x.valid===y.valid);
  rigidMax=Math.max(rigidMax,Math.abs(x.f-y.f));
  assert(Math.abs(x.f-y.f)<1e-10);
 }
}
const report={method:'Actual JS scene geometry, cone gathering/clipping, visibility and complex UTD transport versus independent SciPy exact Sommerfeld solution',cpuExecuted:true,gpuExecuted:false,halfPlaneCases:rows,maxHalfPlaneFieldError:maxError,maxRigidTransformPowerError:rigidMax,limitations:['Conditional single-region transfer only','Long edge interior fixture; endpoints and corners excluded','No physical GPU or arbitrary mesh accuracy claim']};
fs.writeFileSync(new URL('./utd-scene-independent-review.json',import.meta.url),JSON.stringify(report,null,2)+'\n');
console.log(JSON.stringify(report,null,2));
