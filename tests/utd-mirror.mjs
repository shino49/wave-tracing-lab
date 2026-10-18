import assert from 'node:assert/strict';
import {prepareMesh,transformMesh,transformPoint,add,sub,scale,dot,distance,normalize,rayOccluded} from '../dist/mesh-geometry.js';
import {prepareUTDScene,builtInSceneOBJ,evaluatePoint,computeReceiverPoint} from '../dist/utd-scene.js';
import {prepareMirrorExperiment,prepareReflectedUTDScene,reflectPoint,reflectDirection,mirrorIntersection,evaluateReflectedPoint} from '../dist/utd-mirror.js';
let count=0;const tests=[];
const test=(name,fn)=>{fn();count++;tests.push(name);};
const near=(a,b,tol=1e-9)=>assert(Math.abs(a-b)<=tol*Math.max(1,Math.abs(a),Math.abs(b)),`${a} != ${b}`);
const vn=(a,b,tol)=>a.forEach((v,i)=>near(v,b[i],tol));
const state={sourceOrigin:[0,0,-20],detectorCenter:[0,0,20],detectorU:[1,0,0],detectorV:[0,1,0],wavelengthMm:.05,sigma0Mm:1,boundary:'soft'};
const make=(order='before',name='screen',extra={})=>prepareMirrorExperiment(builtInSceneOBJ(name),{...state,...extra},{order,distanceMm:10});
const physicalPoint=(s,p)=>s.mirrorTransport.order==='after'?reflectPoint(p,s.mirrorTransport.mirror):p;
test('Image is an involutive isometry for non-axis planes',()=>{
 const mirror={point:[2,-3,7],normal:normalize([.31,.57,-.81])};
 for(const p of [[0,0,0],[2,-1,5],[-20,12,32]]){vn(reflectPoint(reflectPoint(p,mirror),mirror),p);vn(reflectDirection(reflectDirection(p,mirror),mirror),p);}
 near(distance(reflectPoint([1,2,3],mirror),reflectPoint([-4,5,1],mirror)),distance([1,2,3],[-4,5,1]));
 assert.throws(()=>reflectPoint([0,0,0],{point:[0,0,0],normal:[0,0,0]}),/nonzero/);
});
test('Both orders admit default built-ins without changing local model guards',()=>{
 for(const name of ['screen','box','wedge','opaque-double-slit'])for(const order of ['before','after']){
  const s=make(order,name);assert(s.supported,JSON.stringify(s.unsupported));assert(s.mirrorTransport.admission.conservative);assert.equal(s.mirrorTransport.admission.possibleBlockers.length,0);
  assert.equal(s.mesh.triangles.length,prepareMesh(builtInSceneOBJ(name)).triangles.length);
  assert(s.warnings.some(x=>x.includes('Other mirror/direct families are not added')));
 }
});
test('Unit reflection preserves full conditional local power and GO companions',()=>{
 for(const boundary of ['soft','hard','unpolarized'])for(const order of ['before','after']){
  const s=make(order,'screen',{boundary}),base=prepareUTDScene(s.mesh,{...state,boundary});
  for(const x of [-2,-1,-.2,0,.4,1,2]){
   const q=[x,0,20],a=evaluatePoint(base,q),b=evaluateReflectedPoint(s,physicalPoint(s,q),{details:true});
   assert.equal(a.valid,b.valid);near(a.f,b.f,1e-8);near(a.geometricPower,b.geometricPower,1e-8);assert.equal(b.soft.length,2);assert.equal(b.hard.length,2);vn(b.soft,a.soft.map(x=>-x),1e-8);vn(b.hard,a.hard,1e-8);
   assert.equal(a.edgeCount,b.edgeCount);assert.equal(a.directWeight,b.directWeight);assert.match(b.phaseGauge,/not phase-compatible/);
  }
 }
});
test('DR reflects BOTH detector axes and preserves pixel indexing',()=>{
 const s=make('after'),p=s.mirrorTransport.physical;
 vn(p.detectorCenter,[10,0,10]);vn(p.detectorU,[0,0,1]);vn(p.detectorV,[0,1,0]);
 for(const x of [0,3,6])for(const y of [0,2,4]){
  const effective=computeReceiverPoint(s,x,y,7,5),real=add(p.detectorCenter,add(scale(p.detectorU,((x+.5)/7-.5)*6),scale(p.detectorV,(.5-(y+.5)/5)*6)));
  vn(reflectPoint(real,s.mirrorTransport.mirror),effective);near(evaluateReflectedPoint(s,real).f,evaluatePoint(s,effective).f);
 }
});
test('Physical mirror bounce obeys reflection law and unfolded length',()=>{
 for(const order of ['before','after']){
  const s=make(order),t=s.mirrorTransport,p=t.physical,m=t.centralMirrorPoint;
  const a=order==='before'?p.sourceOrigin:s.region.center,b=order==='before'?s.region.center:p.detectorCenter;
  vn(reflectDirection(normalize(sub(m,a)),t.mirror),normalize(sub(b,m)),1e-8);
  const av=order==='before'?s.params.sourceOrigin:a,bv=order==='after'?s.params.detectorCenter:b;
  near(distance(a,m)+distance(m,b),distance(av,bv));assert(mirrorIntersection(av,bv,t.mirror));
 }
});
test('Gaussian waist age is retained; resetting at the mirror is a failing alternative',()=>{
 const s=make('before');near(s.region.distance,20);vn(s.params.sourceOrigin,[0,0,-20]);near(s.incomingCone.radius0,3);
 const reset=prepareUTDScene(s.mesh,{...state,sourceOrigin:s.mirrorTransport.centralMirrorPoint});
 assert(Math.abs(reset.region.distance-s.region.distance)>5);assert(Math.abs(reset.region.radius-s.region.radius)>.05);assert(Math.abs(reset.phaseCurvature-s.phaseCurvature)>.01);
});
test('Local specular companions and their half-weight reflection seams remain coherent',()=>{
 const theta=80*Math.PI/180,d=[-Math.sin(2*theta),0,-Math.cos(2*theta)],tangent=[Math.cos(theta),0,-Math.sin(theta)];
 const mesh=transformMesh(prepareMesh(builtInSceneOBJ('screen')),{rotationDeg:[0,80,0]});
 for(const order of ['before','after'])for(const edge of [-1,1]){
  const center=add(scale(tangent,edge),scale(d,20)),local={...state,detectorCenter:center,detectorU:[d[2],0,-d[0]],detectorWidthMm:1,detectorHeightMm:1};
  const s=prepareMirrorExperiment(mesh,local,{order,distanceMm:10});assert(s.supported);
  const mid=evaluatePoint(s,center);assert(mid.valid);assert.equal(mid.reflectionCount,1);near(mid.geometricPower,.25,1e-8);assert.equal(mid.edgeCount,2);
  const eps=1e-8,left=evaluatePoint(s,add(center,scale(local.detectorU,-eps))),right=evaluatePoint(s,add(center,scale(local.detectorU,eps)));
  vn(mid.soft,mid.soft.map((_,i)=>(left.soft[i]+right.soft[i])/2),2e-7);
  vn(mid.hard,mid.hard.map((_,i)=>(left.hard[i]+right.hard[i])/2),2e-7);
 }
});
test('Finite edge, endpoint and absent higher-order paths remain masked',()=>{
 for(const order of ['before','after']){
  const s=make(order);assert(!evaluateReflectedPoint(s,physicalPoint(s,[1,2.8,20])).valid);
  const box=make(order,'box'),r=evaluateReflectedPoint(box,physicalPoint(box,[0,0,20]));assert(!r.valid);assert(r.invalidReasons.includes('no-first-order-path'));
  const slit=make(order,'opaque-double-slit');assert(!evaluateReflectedPoint(slit,physicalPoint(slit,[0,0,20])).valid);
 }
});
test('Reflection boundary band is symmetric and does not widen beyond its original epsilon',()=>{
 const theta=80*Math.PI/180,d=[-Math.sin(2*theta),0,-Math.cos(2*theta)],tangent=[Math.cos(theta),0,-Math.sin(theta)];
 const mesh=transformMesh(prepareMesh(builtInSceneOBJ('screen')),{rotationDeg:[0,80,0]});
 for(const order of ['before','after'])for(const delta of [-8e-5,-2e-5,0,2e-5,8e-5]){
  const detectorCenter=add(scale(tangent,1+delta),scale(d,20));
  const s=prepareMirrorExperiment(mesh,{...state,detectorCenter,detectorU:[d[2],0,-d[0]],detectorWidthMm:1,detectorHeightMm:1},{order,distanceMm:10});assert(s.supported);
  const r=evaluateReflectedPoint(s,s.mirrorTransport.physical.detectorCenter,{boundaryEpsilon:2e-6});assert(r.valid);
  near(r.geometricPower,Math.abs(delta)<4e-5?.25:delta<0?1:0,1e-8);
  assert.equal(r.reflectionCount,delta>4e-5?0:1);
 }
});
test('Symmetric-border membership cannot bypass a blocker or a true corner mask',()=>{
 const theta=80*Math.PI/180,d=[-Math.sin(2*theta),0,-Math.cos(2*theta)],tangent=[Math.cos(theta),0,-Math.sin(theta)],u=[d[2],0,-d[0]];
 const mesh=transformMesh(prepareMesh(builtInSceneOBJ('screen')),{rotationDeg:[0,80,0]});
 const q=scale(tangent,1+2e-5),dst=add(q,scale(d,20)),center=add(q,scale(d,14)),n=mesh.vertices.length;
 const vertices=[...mesh.vertices,...[-1,1].flatMap(a=>[-1,1].map(b=>add(center,add(scale(u,a*.3),[0,b*.3,0]))))];
 const blockedMesh=prepareMesh({vertices,faces:[...mesh.triangles.map(t=>t.indices),[n,n+2,n+3],[n,n+3,n+1]]});
 const blocked=prepareUTDScene(blockedMesh,{...state,detectorCenter:dst});assert(blocked.supported);
 const blockedResult=evaluatePoint(blocked,dst,{boundaryEpsilon:2e-6});assert.equal(blockedResult.reflectionCount,0);assert(!blockedResult.valid);
 const corner=add(add(tangent,[0,10,0]),scale(d,20)),s=prepareUTDScene(mesh,{...state,detectorCenter:corner,sigma0Mm:5});assert(s.supported);
 const r=evaluatePoint(s,corner,{boundaryEpsilon:2e-6});assert(!r.valid);assert(r.invalidReasons.includes('corner-reflection-boundary'));
});
function addBlocker(mesh,order) {
 const x=order==='before'?-5:5,z=order==='before'?-10:10;
 const n=mesh.vertices.length,vertices=[...mesh.vertices,[x,-2,z-1],[x,2,z-1],[x,2,z+1],[x,-2,z+1]];
 return prepareMesh({vertices,faces:[...mesh.triangles.map(t=>t.indices),[n,n+1,n+2],[n,n+2,n+3]]});
}
test('Blocked physical prefix and suffix are rejected even when unfolded local rays are clear',()=>{
 for(const order of ['before','after']){
  const good=make(order),t=good.mirrorTransport,mesh=addBlocker(good.mesh,order),bad=prepareReflectedUTDScene(mesh,{...t.physical,mirror:{...t.mirror,order}});
  assert(!bad.supported);assert(bad.unsupported.some(x=>x.code==='mirror-extra-contact'));assert(bad.mirrorTransport.admission.possibleBlockers.length>0);
  const local=prepareUTDScene(mesh,state);assert(local.supported,JSON.stringify(local.unsupported));assert(evaluatePoint(local,[0,0,20]).valid);
  const a=order==='before'?t.physical.sourceOrigin:t.centralMirrorPoint,b=order==='before'?t.centralMirrorPoint:t.physical.detectorCenter;
  assert(rayOccluded(mesh,a,normalize(sub(b,a)),{minDistance:1e-7,maxDistance:distance(a,b)-1e-7}));
 }
});
test('Mirror overlap and crossing geometry are rejected',()=>{
 const s=make(),t=s.mirrorTransport;
 for(const [point,code]of [[[0,0,0],'mirror-crossing-geometry'],[[0,0,-3],'mirror-region-overlap']]){
  const m={...t.mirror,point,order:'before'},bad=prepareReflectedUTDScene(s.mesh,{...t.physical,sourceOrigin:reflectPoint(state.sourceOrigin,m),mirror:m});
  assert(!bad.supported);assert(bad.unsupported.some(x=>x.code===code),JSON.stringify(bad.unsupported));
 }
});
test('Wrong halfspace, order, grazing, and projected source support are rejected',()=>{
 const s=make(),t=s.mirrorTransport;
 const bad=prepareReflectedUTDScene(s.mesh,{...t.physical,sourceOrigin:reflectPoint(t.physical.sourceOrigin,t.mirror),mirror:{...t.mirror,order:'before'}});
 assert(bad.unsupported.some(x=>x.code==='mirror-wrong-halfspace'));
 const backward=prepareReflectedUTDScene(s.mesh,{...t.physical,incidentDirection:scale(t.physical.incidentDirection,-1),mirror:{...t.mirror,order:'before'}});
 assert(backward.unsupported.some(x=>x.code==='mirror-grazing-order'));
 const wide=make('before','screen',{detectorWidthMm:40});assert(!wide.supported);assert(wide.unsupported.some(x=>x.code==='mirror-projected-source-side'));
 const g=prepareMirrorExperiment(s.mesh,state,{order:'before',distanceMm:.01});assert(!g.supported);
 assert.throws(()=>prepareMirrorExperiment(s.mesh,state,{order:'before',distanceMm:0}),/positive/);
});
test('Receiver queries cannot escape the certified detector support',()=>{
 for(const order of ['before','after']){
  const s=make(order),p=s.mirrorTransport.physical;
  assert.throws(()=>evaluateReflectedPoint(s,add(p.detectorCenter,scale(p.detectorU,4))),/outside/);
  assert.throws(()=>evaluateReflectedPoint(s,add(p.detectorCenter,reflectDirection([0,0,1],order==='after'?s.mirrorTransport.mirror:{point:[0,0,0],normal:[0,1,0]}))),/outside/);
 }
});
test('Rotating/translating the complete physical arrangement preserves field and admission',()=>{
 const tr={rotationDeg:[23,31,17],translation:[2,-3,4]},point=p=>transformPoint(p,tr),direction=p=>transformPoint(p,{rotationDeg:tr.rotationDeg});
 for(const order of ['before','after']){
  const s=make(order),t=s.mirrorTransport,p=t.physical;
  const changed=prepareReflectedUTDScene(transformMesh(s.mesh,tr),{...p,sourceOrigin:point(p.sourceOrigin),incidentDirection:direction(p.incidentDirection),detectorCenter:point(p.detectorCenter),detectorU:direction(p.detectorU),detectorV:direction(p.detectorV),mirror:{order,point:point(t.mirror.point),normal:direction(t.mirror.normal)}});
  assert(changed.supported,JSON.stringify(changed.unsupported));near(changed.region.distance,s.region.distance,1e-8);
  for(const x of [-2,-1,.4,1,2]){const a=physicalPoint(s,[x,0,20]);near(evaluateReflectedPoint(s,a).f,evaluateReflectedPoint(changed,point(a)).f,1e-7);}
 }
});
test('Uniform scale with wavelength and sigma preserves the conditional field',()=>{
 for(const order of ['before','after'])for(const factor of [.5,2]){
  const s=make(order),t=s.mirrorTransport,p=t.physical,tr={scale:factor};
  const mesh=prepareMesh({vertices:s.mesh.vertices.map(v=>scale(v,factor)),faces:s.mesh.triangles.map(t=>t.indices)});
  const changed=prepareReflectedUTDScene(mesh,{...p,wavelengthMm:p.wavelengthMm*factor,sigma0Mm:p.sigma0Mm*factor,sourceOrigin:scale(p.sourceOrigin,factor),detectorCenter:scale(p.detectorCenter,factor),detectorWidthMm:p.detectorWidthMm*factor,detectorHeightMm:p.detectorHeightMm*factor,mirror:{order,point:scale(t.mirror.point,factor),normal:t.mirror.normal}});
  assert(changed.supported);near(evaluateReflectedPoint(s,p.detectorCenter).f,evaluateReflectedPoint(changed,scale(p.detectorCenter,factor)).f,1e-8);
 }
});
console.log(JSON.stringify({suite:'one-ideal-mirror-one-local-utd-region',passed:count,tests},null,2));
