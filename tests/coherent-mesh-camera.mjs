import assert from 'node:assert/strict';
import {validateCoherentCamera,cameraSample,validateDiffuseReflectance,diffuseReflectanceAt,projectCoherentSurfaceRoutes,makeCoherentCameraPreset} from '../dist/coherent-mesh-camera.js';
import {prepareCoherentMeshScene,compileCoherentSource,sumCoherentRoutes} from '../dist/coherent-mesh-core.js';
import {evaluateMeshTransport} from '../dist/coherent-mesh-transport.js';
const close=(a,b,t=1e-10)=>assert(Math.abs(a-b)<t,`${a} != ${b}`);
const test=(name,fn)=>{fn();console.log('PASS',name);};
const camera={position:[0,0,0],target:[0,0,5],up:[0,1,0],fovDeg:90};
test('camera validation is strict and deeply immutable',()=>{
 const c=validateCoherentCamera(camera);assert(Object.isFrozen(c)&&Object.isFrozen(c.position));
 for(const changes of [{position:[NaN,0,0]},{target:[0,0,0]},{up:[0,0,1]},{fovDeg:180},{fovDeg:'40'}])assert.throws(()=>validateCoherentCamera({...camera,...changes}));
});
test('perspective camera selects nearest hit and retains backface winding',()=>{
 const face=z=>({mesh:{vertices:[[-10,-10,z],[10,-10,z],[0,10,z]],faces:[[0,1,2]]},materialIndex:0});
 const scene=prepareCoherentMeshScene([face(5),face(3)],[{type:'opaque'}]);
 const hit=cameraSample(scene,camera,1,1,3,3);close(hit.distance,3);assert.equal(hit.triangleId,1);assert.deepEqual(hit.normal,[0,0,1]);
 const edge=cameraSample(scene,camera,0,1,3,3);close(edge.point[0],2);close(edge.point[2],3);assert(edge.distance>3);
 assert.equal(cameraSample(scene,{...camera,target:[0,0,-5]},1,1,3,3),null);
});
test('passive spectral reflectance interpolates endpoints and rejects invalid curves',()=>{
 const k=validateDiffuseReflectance([{wavelengthNm:360,value:.1},{wavelengthNm:500,value:.8},{wavelengthNm:830,value:.2}]);
 close(diffuseReflectanceAt(k,360),.1);close(diffuseReflectanceAt(k,430),.45);close(diffuseReflectanceAt(k,830),.2);assert(Object.isFrozen(k[0]));
 for(const bad of [[],[{wavelengthNm:360,value:0},{wavelengthNm:830,value:1.1}],[{wavelengthNm:361,value:0},{wavelengthNm:830,value:1}],[{wavelengthNm:360,value:0},{wavelengthNm:500,value:1},{wavelengthNm:500,value:0},{wavelengthNm:830,value:1}]])assert.throws(()=>validateDiffuseReflectance(bad));
 assert.throws(()=>diffuseReflectanceAt(k,850));
});
test('direct R D RD DR use their final physical leg, preserve phases/counts and scale templates',()=>{
 const p=[0,0,0],src=[0,0,4],q=[Math.sqrt(3),0,1],r=[0,0,-2];
 const routes=[{kind:'direct',vertices:[src,p]},{kind:'reflection',vertices:[src,q,p]},{kind:'diffraction',point:q},{kind:'mixed-diffraction',interactionOrder:'RD',vertices:[src,r,q,p]},{kind:'mixed-diffraction',interactionOrder:'DR',vertices:[src,q,r,p]}].map(r=>({...r,amplitude:2,phaseRadReduced:.4}));
 const transport={routes,geometricPower:8,count:5,templates:routes.map(route=>({route,amplitudeScale:3,interactionPhaseRad:.7}))};
 const result=projectCoherentSurfaceRoutes(transport,p,[0,0,1]);
 assert.equal(result.routes.length,5);assert.equal(result.count,5);close(result.routes[0].amplitude,2);for(const i of [1,2,3])close(result.routes[i].amplitude,Math.sqrt(2));close(result.routes[4].amplitude,0);close(result.geometricPower,6);
 for(let i=0;i<5;i++){assert.equal(result.routes[i].phaseRadReduced,.4);close(result.templates[i].amplitudeScale,1.5*result.routes[i].amplitude);assert.equal(result.templates[i].route,result.routes[i]);assert.equal(routes[i].amplitude,2);}
 assert.equal(result.angularDiagnostics.backfaceCount,1);assert.equal(result.angularDiagnostics.maxAngularSpreadDeg,180);
 const grazing=projectCoherentSurfaceRoutes({routes:[{kind:'diffraction',point:[1,0,0],amplitude:1,phaseRadReduced:0}]},p,[0,0,1]);assert.equal(grazing.angularDiagnostics.grazingCount,1);
});
test('preset is true colored extruded geometry with visible-light Lloyd interference',()=>{
 const p=makeCoherentCameraPreset();p.materials.filter(m=>m.type==='diffuse').forEach(m=>validateDiffuseReflectance(m.reflectance));
 // Mapping receiver materials to blockers permits testing against the original
 // core as well as the extended diffuse-receiver core.
 const scene=prepareCoherentMeshScene(p.entries,p.materials.map(m=>m.type==='diffuse'?{type:'opaque'}:m));
 const compiled=compileCoherentSource(scene,p.source);let sampled=0,visible=0,difference=0;const normals=new Set(),materials=new Set();
 for(let y=0;y<24;y++)for(let x=0;x<36;x++){
  const hit=cameraSample(scene,p.camera,x,y,36,24);if(!hit||hit.materialIndex===0)continue;
  normals.add(hit.normal.map(x=>x.toFixed(3)).join(','));materials.add(hit.materialIndex);sampled++;
  const t=projectCoherentSurfaceRoutes(evaluateMeshTransport(scene,compiled,hit.point),hit.point,hit.normal);if(t.routes.length>=2&&t.geometricPower>0){visible++;difference+=Math.abs(sumCoherentRoutes(t.routes).coherent/t.geometricPower-1);}
 }
 assert(sampled>300);assert(materials.size===4);assert(normals.size>=4);assert(visible>200);assert(difference/visible>.35,`weak wave contrast: ${difference/visible}`);
 console.log(JSON.stringify({sampled,visible,meanRelativeWaveDifference:difference/visible,visibleMaterialCount:materials.size,visibleNormalCount:normals.size}));
});
