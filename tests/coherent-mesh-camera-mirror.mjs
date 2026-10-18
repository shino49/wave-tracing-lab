import assert from 'node:assert/strict';
import {validateCoherentCamera,cameraSample,traceCoherentCamera} from '../dist/coherent-mesh-camera.js';
import {prepareCoherentMeshScene} from '../dist/coherent-mesh-core.js';
const close=(a,b,t=2e-11)=>assert(Math.abs(a-b)<=t,`${a} != ${b}`);
const vector=(a,b,t)=>a.forEach((v,i)=>close(v,b[i],t));
const test=(name,fn)=>{fn();console.log('PASS',name);};
const camera={position:[0,0,-2],target:[0,0,0],up:[0,1,0],fovDeg:45};
const materials=[{type:'mirror',reflectance:.8},{type:'mirror',reflectance:.3},{type:'diffuse',reflectance:[{wavelengthNm:360,value:.7},{wavelengthNm:830,value:.7}]},{type:'opaque'}];
function plate(center,u,v,materialIndex,reverse=false){
 const vertices=[[-1,-1],[1,-1],[1,1],[-1,1]].map(([a,b])=>center.map((x,i)=>x+a*u[i]+b*v[i]));
 return {mesh:{vertices,faces:reverse?[[2,1,0],[3,2,0]]:[[0,1,2],[0,2,3]]},materialIndex,surface:'open-plate'};
}
const first=(reverse=false)=>plate([0,0,0],[.6,0,.6],[0,.6,0],0,reverse);
const second=(reverse=false)=>plate([2,0,0],[.6,0,.6],[0,.6,0],1,reverse);
const receiver=(x=2,m=2)=>plate([x,0,0],[0,0,.4],[0,.4,0],m);
const terminal=()=>plate([2,0,3],[.5,0,0],[0,-.5,0],2);
const scene=(entries,ms=materials)=>prepareCoherentMeshScene(entries,ms);
const trace=(s,c=camera)=>traceCoherentCamera(s,c,0,0,1,1);
test('optional camera bounce budget validates strictly and preserves legacy JSON',()=>{
 assert.deepEqual(validateCoherentCamera(camera),camera);
 assert(!JSON.stringify(validateCoherentCamera(camera)).includes('mirrorBounces'));
 for(const n of [0,1,2])assert.equal(validateCoherentCamera({...camera,mirrorBounces:n}).mirrorBounces,n);
 for(const n of [-1,3,1.5,NaN,Infinity,'1',null,undefined])assert.throws(()=>validateCoherentCamera({...camera,mirrorBounces:n}));
});
test('one mirror obeys reflection law and virtual-camera final view',()=>{
 const s=scene([first(),receiver()]),r=trace(s);
 assert.equal(r.status,'diffuse');assert.equal(r.primaryMaterialType,'mirror');assert.equal(r.mirrorBounces,1);close(r.throughput,.8);
 vector(r.point,[2,0,0]);vector(r.normal,[-1,0,0]);vector(r.viewDirection,[-1,0,0]);
 // Camera [0,0,-2] reflected across z=x is [-2,0,0].
 const virtualCamera=[-2,0,0],delta=virtualCamera.map((x,i)=>x-r.point[i]),length=Math.hypot(...delta);
 vector(r.viewDirection,delta.map(x=>x/length));
 assert.equal(cameraSample(s,camera,0,0,1,1).materialIndex,0);
 const farther=trace(s,{...camera,position:[0,0,-5]});close(farther.throughput,r.throughput);vector(farther.point,r.point);
});
test('two mirrors multiply intensity and reverse final segment rather than original camera vector',()=>{
 const r=trace(scene([first(),second(),terminal()]));
 assert.equal(r.status,'diffuse');assert.equal(r.mirrorBounces,2);close(r.throughput,.24);vector(r.point,[2,0,3]);vector(r.viewDirection,[0,0,-1]);
 for(const a of [false,true])for(const b of [false,true]){
  const q=trace(scene([first(a),second(b),terminal()]));assert.equal(q.status,r.status);vector(q.point,r.point);vector(q.viewDirection,r.viewDirection);close(q.throughput,r.throughput);
 }
});
test('mirror chain may revisit the same mesh object',()=>{
 const pieces=[first(),second(),terminal()],vertices=[],faces=[],materialIndices=[];
 for(const p of pieces){const offset=vertices.length;vertices.push(...p.mesh.vertices);faces.push(...p.mesh.faces.map(f=>f.map(i=>i+offset)));materialIndices.push(...p.mesh.faces.map(()=>p.materialIndex));}
 const r=trace(scene([{mesh:{vertices,faces},materialIndices,surface:'open-plate'}]));
 assert.equal(r.status,'diffuse');assert.equal(r.mirrorBounces,2);close(r.throughput,.24);vector(r.point,[2,0,3]);
});
test('nearest blockers stop direct and reflected segments',()=>{
 const before=plate([0,0,-1],[.5,0,0],[0,.5,0],3),direct=trace(scene([first(),receiver(),before]));
 assert.equal(direct.status,'opaque');assert.equal(direct.primaryMaterialType,'opaque');assert.equal(direct.mirrorBounces,0);close(direct.throughput,1);
 const r=trace(scene([first(),receiver(),receiver(1,3)]));assert.equal(r.status,'opaque');vector(r.point,[1,0,0]);assert.equal(r.mirrorBounces,1);close(r.throughput,.8);
});
test('budget exhaustion, absorption, and escape have explicit bounded results',()=>{
 const s=scene([first(),second(),terminal()]);
 for(const n of [0,1]){const r=trace(s,{...camera,mirrorBounces:n});assert.equal(r.status,'mirror-limit');assert.equal(r.mirrorBounces,n);close(r.throughput,n?.8:1);vector(r.point,n?[2,0,0]:[0,0,0]);}
 const third=terminal();third.materialIndex=0;const exhausted=trace(scene([first(),second(),third]));assert.equal(exhausted.status,'mirror-limit');assert.equal(exhausted.mirrorBounces,2);close(exhausted.throughput,.24);
 const ms=materials.map(m=>({...m}));ms[0].reflectance=0;
 const absorbed=trace(scene([first(),receiver()],ms));assert.equal(absorbed.status,'absorbed');assert.equal(absorbed.throughput,0);assert.equal(absorbed.mirrorBounces,0);
 for(const [c,bounces] of [[camera,1],[{...camera,target:[0,0,-3]},0]]){
  const r=trace(scene([first()]),c);assert.equal(r.status,'miss');assert.equal(r.mirrorBounces,bounces);
  for(const k of ['point','normal','viewDirection'])assert.equal(r[k],null);
  for(const k of ['triangleId','materialIndex'])assert.equal(r[k],-1);
  assert.equal(r.primaryMaterialType,bounces?'mirror':null);
 }
});
test('coplanar mirror triangles do not self-hit and near receiver keeps exact endpoint',()=>{
 // Center ray hits the shared diagonal; adjacent coplanar triangles must not trap it.
 const s=scene([first(),receiver(1e-8)]),r=trace(s);
 assert.equal(r.status,'diffuse');assert.equal(r.mirrorBounces,1);close(r.point[0],1e-8,1e-20);close(r.point[2],0,1e-20);
 // Translate to large coordinates: interval scales with floating-point precision,
 // but does not move the physical endpoint or discard an entire object.
 const shift=[1000,500,-1000],entries=[first(),receiver(1e-6)];
 for(const entry of entries)entry.mesh.vertices=entry.mesh.vertices.map(p=>p.map((v,i)=>v+shift[i]));
 const c={...camera,position:camera.position.map((v,i)=>v+shift[i]),target:shift};
 const q=trace(scene(entries),c);assert.equal(q.status,'diffuse');assert.equal(q.mirrorBounces,1);vector(q.point,[1000+1e-6,500,-1000],2e-12);
});
test('grazing mirror never skips a distinct nearby blocker',()=>{
 const mirror=plate([1,0,0],[1,0,0],[0,0,1],0);
 const blocker=plate([1.00001,0,0],[0,1e-5,0],[0,0,1],3);
 const diffuse=plate([2,0,0],[0,1e-5,0],[0,0,1],2);
 const c={position:[0,1e-10,0],target:[1,0,0],up:[0,0,1],fovDeg:45};
 for(const reversed of [false,true]){
  const m=structuredClone(mirror);if(reversed)m.mesh.faces=m.mesh.faces.map(f=>f.reverse());
  const r=trace(scene([m,blocker,diffuse]),c);
  assert.equal(r.status,'opaque');assert.equal(r.mirrorBounces,1);close(r.point[0],1.00001,1e-13);close(r.throughput,.8);
 }
});
console.log('Camera mirror-chain geometry tests passed.');
