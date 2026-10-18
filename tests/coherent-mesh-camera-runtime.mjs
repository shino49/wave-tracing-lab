import assert from 'node:assert/strict';
import {executeCoherentMeshTask,prepareCoherentMeshTask} from '../dist/coherent-mesh-runtime.js';
import {serializeCoherentScene,parseCoherentScene,sceneFromCoherentData} from '../dist/coherent-mesh-scene.js';
import {cameraSample} from '../dist/coherent-mesh-camera.js';
import {prepareCoherentSpatialSources} from '../dist/coherent-mesh-spatial.js';
const flat=r=>[{wavelengthNm:360,value:r},{wavelengthNm:830,value:r}];
const source={position:[0,0,0],power:2,phaseRad:0,wavelengthMm:.00055};
const camera={position:[0,0,5],target:[0,0,10],up:[0,1,0],fovDeg:20};
const scene={schema:'coherent-mesh-scene-v1',units:'mm',name:'Analytic one-sided plate',materials:[{type:'diffuse',reflectance:flat(.6)}],source,camera,detector:{center:[0,0,20],u:[1,0,0],v:[0,1,0],width:1,height:1},meshes:[{vertices:[[-10,-10,10],[-10,10,10],[10,10,10],[10,-10,10]],faces:[[0,1,2],[0,2,3]],materialIndex:0,surface:'open-plate'}]};
const opts={scene,measurement:'camera',resolution:16,pixelQuadrature:1,backend:'cpu',maxReflections:0,diffraction:false,mixedDiffraction:false};
const names=['coherent','incoherent','geometric','crossTerm','coherentXYZ','incoherentXYZ','geometricXYZ'];
function arrays(a,b,rel=1e-11){assert.equal(a.length,b.length);for(let i=0;i<a.length;i++)assert(Math.abs(a[i]-b[i])<=1e-18+Math.max(Math.abs(a[i]),Math.abs(b[i]))*rel,`${i}: ${a[i]} != ${b[i]}`);}
const a=await executeCoherentMeshTask(opts),prepared=prepareCoherentMeshTask(opts);
for(let y=0;y<16;y++)for(let x=0;x<16;x++){const hit=cameraSample(prepared.scene,a.configuration.camera,x,y,16,16),r=Math.hypot(...hit.point),c=10/r,expected=.6*2*c/(4*Math.PI*Math.PI*r*r);assert(Math.abs(a.coherent[y*16+x]-expected)<1e-17);}
assert.equal(a.diagnostics.cameraHitSamples,256);arrays(a.coherent,a.geometric);assert(a.crossTerm.every(x=>x===0));
console.log('PASS direct source inverse-square, incidence cosine, rho/pi radiance, endpoint self-visibility');
for(const [power,rho] of [[4,.6],[2,.3],[2,0]]){const r=await executeCoherentMeshTask({...opts,source:{...source,power},scene:{...scene,materials:[{type:'diffuse',reflectance:flat(rho)}]}});for(const key of names)arrays(r[key],a[key].map(v=>v*power/2*rho/.6));}
const shifted=await executeCoherentMeshTask({...opts,camera:{...camera,position:[0,0,0],fovDeg:2*Math.atan(.5*Math.tan(10*Math.PI/180))*180/Math.PI}});for(const key of names)arrays(shifted[key],a[key]);
console.log('PASS passive reflectance zero/scaling, source linearity, camera distance changes coverage without artificial radiance attenuation');
const redScene={...scene,materials:[{type:'diffuse',reflectance:[{wavelengthNm:360,value:0},{wavelengthNm:450,value:0},{wavelengthNm:650,value:1},{wavelengthNm:830,value:1}]}]};
const lines=await executeCoherentMeshTask({...opts,scene:redScene,spectrum:{mode:'lines',lines:[{wavelengthNm:450,powerFraction:1},{wavelengthNm:650,powerFraction:1}]}}),red=await executeCoherentMeshTask({...opts,scene:redScene,source:{...source,wavelengthMm:.00065}});for(const key of names)arrays(lines[key],red[key].map(v=>v*.5));
console.log('PASS reflectance acts per wavelength before CIE XYZ; blue energy never becomes red');
const blockedScene={...scene,materials:[...scene.materials,{type:'opaque'}],meshes:[...scene.meshes,{vertices:[[-10,-10,2],[-10,10,2],[10,10,2],[10,-10,2]],faces:[[0,1,2],[0,2,3]],materialIndex:1,surface:'open-plate'}]};
const blocked=await executeCoherentMeshTask({...opts,scene:blockedScene});assert(blocked.coherent.every(x=>x===0));assert.equal(blocked.diagnostics.cameraHitSamples,256);
const back=await executeCoherentMeshTask({...opts,scene:{...scene,meshes:[{...scene.meshes[0],faces:[[2,1,0],[3,2,0]]}]}});assert(back.coherent.every(x=>x===0));assert.equal(back.diagnostics.cameraBackfaceSamples,256);
console.log('PASS other triangles cast real source shadows; outward one-sided coating never emits from backside');
const encoded=serializeCoherentScene(scene);assert.equal(serializeCoherentScene(parseCoherentScene(encoded)),encoded);assert.deepEqual(parseCoherentScene(encoded).camera,camera);
const diskSource={...source,spatial:{type:'disk',radiusMm:.05,normal:[0,0,1],samples:2,seed:7391}},disk=await executeCoherentMeshTask({...opts,source:diskSource}),refs=[];for(const node of prepareCoherentSpatialSources(diskSource))refs.push(await executeCoherentMeshTask({...opts,source:node.source}));for(const key of names)arrays(disk[key],refs[0][key].map((v,i)=>(v+refs[1][key][i])/2));
console.log('PASS scene camera/material roundtrip and cached exact hits retain independent source intensity sum');
const waveOpts={preset:'object-imaging',measurement:'camera',resolution:16,pixelQuadrature:1,backend:'cpu',diffraction:false};const wave=await executeCoherentMeshTask(waveOpts),flipped=await executeCoherentMeshTask({...waveOpts,phaseShiftRad:Math.PI});arrays(wave.geometric,flipped.geometric);assert(wave.coherent.some((v,i)=>Math.abs(v-flipped.coherent[i])>1e-8));
const controller=new AbortController();await assert.rejects(executeCoherentMeshTask({...opts,source:diskSource},{signal:controller.signal,preview:p=>{if(p.spatialSamplesIntegrated===2)controller.abort();}}),{name:'AbortError'});
let calls=0,resets=0;const replay=await executeCoherentMeshTask({...opts,source:diskSource,backend:'auto'},{gpuFactory:async()=>({backend:{kind:'test'},destroy(){},async evaluate({sampleCount}){if(++calls===3)throw Error('deliberate camera failure');return {coherent:new Float64Array(sampleCount).fill(33),incoherent:new Float64Array(sampleCount).fill(44)};}}),reset:()=>resets++});assert.equal(resets,1);for(const key of names)arrays(replay[key],disk[key]);assert.equal(replay.diagnostics.cameraHitSamples,disk.diagnostics.cameraHitSamples);
console.log('PASS object mirror phase changes wave image only; camera cancellation and failed-GPU replay clear all channels and counters');
