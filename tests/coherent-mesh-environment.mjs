import assert from 'node:assert/strict';
import {validateCoherentLighting,environmentAngularRadiance,cosineDirection} from '../dist/coherent-mesh-environment.js';
import {traceCoherentRay,makeCoherentCameraPreset} from '../dist/coherent-mesh-camera.js';
import {prepareCoherentMeshScene} from '../dist/coherent-mesh-core.js';
import {sceneFromCoherentData,sceneToCoherentData,serializeCoherentScene,parseCoherentScene} from '../dist/coherent-mesh-scene.js';
const close=(a,b,t=1e-12)=>assert(Math.abs(a-b)<t,`${a} != ${b}`);
const rho=[{wavelengthNm:360,value:.5},{wavelengthNm:830,value:.5}];
const materials=[{type:'diffuse',reflectance:rho},{type:'opaque'},{type:'mirror',reflectance:.4}];
const plate=(z,m=0,reverse=false)=>({mesh:{vertices:[[-100,-100,z],[100,-100,z],[100,100,z],[-100,100,z]],faces:reverse?[[2,1,0],[3,2,0]]:[[0,1,2],[0,2,3]]},materialIndex:m,surface:'open-plate'});
const env=validateCoherentLighting({environment:{radiance:.3,upper:1,lower:1}}).environment;
assert.equal(validateCoherentLighting(null),null);
assert.equal(validateCoherentLighting(),null);
assert.deepEqual(validateCoherentLighting({}),{environment:null,indirectSamples:0});
assert.equal(env.wavelengthNm,550);
assert.equal(validateCoherentLighting({environment:{radiance:.2,spectrum:{mode:'monochrome'},wavelengthNm:620}}).environment.wavelengthNm,620);
assert(Object.isFrozen(env)&&Object.isFrozen(env.up)&&Object.isFrozen(env.spectrum));
for(const bad of [[],{extra:1},{indirectSamples:3},{indirectSamples:null},{environment:{}},{environment:{radiance:-1}},{environment:{radiance:2}},{environment:{radiance:NaN}},{environment:{radiance:.1,upper:11}},{environment:{radiance:.1,lower:-1}},{environment:{radiance:.1,samples:3}},{environment:{radiance:.1,seed:-1}},{environment:{radiance:.1,up:[0,0,0]}},{environment:{radiance:.1,wavelengthNm:359}},{environment:{radiance:.1,wavelengthNm:831}},{environment:{radiance:.1,spectrum:{mode:'invalid'}}}])assert.throws(()=>validateCoherentLighting(bad));
const scene=prepareCoherentMeshScene([plate(0)],materials),origin=[0,0,0],initial=scene.triangles[0];
for(let i=0;i<32;i++){
 const d=cosineDirection([0,0,1],i,32,791);
 assert.deepEqual(d,cosineDirection([0,0,1],i,32,791));close(Math.hypot(...d),1);assert(d[2]>0);
 const r=traceCoherentRay(scene,origin,d,2,initial);assert.equal(r.status,'miss');close(environmentAngularRadiance(env,r.escapeDirection),.3);
}
assert.notDeepEqual(cosineDirection([0,0,1],0,32,791),cosineDirection([0,0,1],0,32,792));
// Independent randomized realizations reproduce cosine-weighted hemisphere integral.
const two=validateCoherentLighting({environment:{radiance:.2,upper:2,lower:.5}}).environment;
for(const n of [[0,1,0],[0,-1,0],[1,0,0],[Math.sqrt(.75),.5,0]]){
 let mean=0,count=0;for(let seed=0;seed<256;seed++)for(let i=0;i<32;i++){mean+=environmentAngularRadiance(two,cosineDirection(n,i,32,seed));count++;}
 const expected=.2*((2+.5)/2+(2-.5)*n[1]/2);close(mean/count,expected,.003);
}
// Nearest backfaces remain occluders; winding never changes transport visibility.
for(const reverse of [false,true]){const s=prepareCoherentMeshScene([plate(0),plate(1e-8,1,reverse)],materials);const r=traceCoherentRay(s,origin,[0,0,1],2,s.triangles[0]);assert.equal(r.status,'opaque');close(r.point[2],1e-8,1e-20);}
// A reflected infinite ray keeps the outgoing direction and intensity coefficient.
const mirror=prepareCoherentMeshScene([plate(0,2)],materials),r=traceCoherentRay(mirror,[0,0,1],[0,0,-1]);
assert.equal(r.status,'miss');assert.deepEqual(r.escapeDirection,[0,0,1]);close(r.throughput,.4);assert.equal(r.mirrorBounces,1);
assert.equal(traceCoherentRay(mirror,[0,0,1],[0,0,-1],0).status,'mirror-limit');
// Scene import/export snapshots lighting, including default absence and explicit null.
const data=makeCoherentCameraPreset(),legacy=sceneFromCoherentData(data);assert(!('lighting' in legacy));
const input={...data,lighting:{environment:{radiance:.1},indirectSamples:2}},canonical=sceneFromCoherentData(input);
const roundtrip=parseCoherentScene(serializeCoherentScene(canonical));assert.deepEqual(roundtrip,canonical);assert.deepEqual(sceneToCoherentData(roundtrip).lighting,canonical.lighting);
input.lighting.environment.radiance=.9;close(canonical.lighting.environment.radiance,.1);
assert.equal(sceneFromCoherentData({...data,lighting:null}).lighting,null);
console.log('PASS environment schema, cosine sampling, uniform/two-hemisphere oracle, exact launch, backface visibility, mirror escape, and scene roundtrip');
