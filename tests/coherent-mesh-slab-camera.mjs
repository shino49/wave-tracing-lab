import assert from 'node:assert/strict';
import {prepareCoherentMeshScene} from '../dist/coherent-mesh-core.js';
import {traceCoherentRay,traceCoherentCamera} from '../dist/coherent-mesh-camera.js';
import {validateCoherentSlab,fresnelS,refractSlabDirection,slabAbsorptionAt} from '../dist/coherent-mesh-slab.js';
import {integrateCoherentLighting} from '../dist/coherent-mesh-lighting-runtime.js';
import {validateCoherentLighting,cosineDirection} from '../dist/coherent-mesh-environment.js';
import {prepareCoherentSpectrum} from '../dist/coherent-mesh-spectrum.js';
import {cieXYZ,hashSpectralIndex} from '../dist/plt-spectrum.js';
const close=(a,b,t=1e-11)=>assert(Math.abs(a-b)<=t,`${a} != ${b}`),vector=(a,b)=>a.forEach((v,i)=>close(v,b[i]));
const slab=validateCoherentSlab({center:[0,0,0],u:[1,0,0],v:[0,1,0],width:100,height:100,thickness:2,index:1.5,internalReflections:2});
const plate=(z=5,x=0)=>({mesh:{vertices:[[x-20,-20,z],[x+20,-20,z],[x+20,20,z],[x-20,20,z]],faces:[[2,1,0],[3,2,0]]},materialIndex:0,surface:'open-plate'});
const materials=[{type:'diffuse',reflectance:[{wavelengthNm:360,value:1},{wavelengthNm:830,value:1}]}];
const make=(s=slab,entries=[plate()])=>prepareCoherentMeshScene(entries,materials,{maxReflections:0,slab:s});
const trace=(scene,seed=0,d=[0,0,1],o=[0,0,-5])=>traceCoherentRay(scene,o,d,2,null,seed);
const scene=make(),seeds={};
for(let seed=0;seed<10000&&Object.keys(seeds).length<4;seed++){const r=trace(scene,seed);seeds[r.slabEvents[0].branch]=seed;}
assert.equal(Object.keys(seeds).length,4);
const R=.04,T=.96,admitted=R+T*T*(1+R+R*R);
for(const [branch,seed] of Object.entries(seeds)){
 const r=trace(scene,seed);close(r.throughput,admitted);close(r.slabTruncatedPower,1-admitted);
 close(r.glassLength,branch==='R'?0:(branch.length-1)*2);
 assert.equal(r.status,branch==='TT'||branch==='TRRT'?'diffuse':'miss');
 if(r.status==='diffuse')vector(r.point,[0,0,5]);
 close(r.slabEvents[0].power/r.slabEvents[0].probability,admitted);
 assert.deepEqual(trace(scene,seed),r,'deterministic ray seed');
}
console.log('PASS R/TT/TRT/TRRT physical lengths, Fresnel importance weights and omitted power');
for(const budget of [0,1,2]){
 const s=make({...slab,internalReflections:budget});let transmitted=0,reflected=0;const count=30000;
 for(let seed=0;seed<count;seed++){const r=trace(s,seed);if(r.status==='diffuse')transmitted+=r.throughput;else reflected+=r.throughput;}
 const transmission=T*T*(1+(budget===2?R*R:0)),reflection=R+(budget>=1?T*T*R:0);
 close(transmitted/count,transmission,.004);close(reflected/count,reflection,.004);close((transmitted+reflected)/count,transmission+reflection,1e-11);
}
console.log('PASS sampled reflection/transmission expectation and finite-order energy loss');
const direction=[.6,0,.8],oblique=trace(scene,seeds.TT,direction),glassZ=Math.sqrt(1-(.6/1.5)**2);
assert.equal(oblique.status,'diffuse');vector(oblique.point,[8*.6/.8+2*(.6/1.5)/glassZ,0,5]);close(oblique.glassLength,2/glassZ);vector(oblique.viewDirection,direction.map(v=>-v));
const blocker=trace(make(slab,[plate(-2),plate()]),seeds.TT);vector(blocker.point,[0,0,-2]);assert.equal(blocker.slabEvents.length,0);
const off=trace(scene,seeds.TT,[0,0,1],[60,0,-5]);assert.equal(off.status,'miss');assert.equal(off.slabEvents.length,0);
assert.equal(trace(make({...slab,width:2}),seeds.TT,[.6,0,.8],[-2.5,0,-5]).status,'slab-unsupported');
assert.equal(trace(scene,0,[0,0,1],[0,0,0]).status,'slab-unsupported');
assert(fresnelS(1.5,1,.5).tir);assert.equal(refractSlabDirection([Math.sqrt(.75),0,.5],[0,0,1],1.5,1),null);
console.log('PASS oblique Snell shift, exact terminal geometry, finite aperture, visibility, unsupported sides/inside and TIR');
const original=make(null),invisible=make({...slab,index:1});for(let seed=0;seed<10;seed++)assert.deepEqual(trace(invisible,seed,direction),trace(original,seed,direction));
console.log('PASS index-one zero-absorption exact legacy camera result');
// Direct environment uses each selected camera path's Beer power per spectral
// node before XYZ. No source contribution or scalar-color attenuation oracle.
const absorbing={...slab,absorption:[{wavelengthNm:360,value:.1},{wavelengthNm:450,value:.1},{wavelengthNm:650,value:.8},{wavelengthNm:830,value:.8}]};
const skyScene=make(absorbing,[plate(5,100)]),camera={position:[0,0,-5],target:[0,0,0],up:[0,1,0],fovDeg:20};
const source={position:[0,0,-6],power:0,phaseRad:0,wavelengthMm:.00055},options={resolution:1,pixelQuadrature:1,maxReflections:0,diffraction:false,mixedDiffraction:false,coherence:1,spectralSamples:16,spectralSeed:7391};
const lighting=validateCoherentLighting({environment:{radiance:.3,upper:1,lower:1,samples:4,spectrum:{mode:'lines',lines:[{wavelengthNm:450,powerFraction:1},{wavelengthNm:650,powerFraction:1}]}}});
const images=Object.fromEntries(['coherent','incoherent','geometric','crossTerm','coherentXYZ','incoherentXYZ','geometricXYZ'].map(k=>[k,new Float64Array(k.endsWith('XYZ')?3:1)]));
await integrateCoherentLighting({scene:skyScene,source,camera,options,lighting,spectralPlan:prepareCoherentSpectrum(source,options),spatialSources:[],images});
const ray=traceCoherentCamera(skyScene,camera,0,0,1,1,.5,.5,options.spectralSeed);assert.equal(ray.status,'miss');assert(ray.glassLength>0);
const power=nm=>.15*ray.throughput*Math.exp(-slabAbsorptionAt(absorbing,nm)*ray.glassLength);
for(const k of ['coherent','incoherent','geometric'])close(images[k][0],power(450)+power(650));
for(let c=0;c<3;c++)close(images.coherentXYZ[c],power(450)*cieXYZ(450)[c]+power(650)*cieXYZ(650)[c]);
console.log('PASS camera-to-environment wavelength-dependent Beer absorption before XYZ');

const diffuseScene=make(absorbing),diffuseLight=validateCoherentLighting({...lighting,environment:{...lighting.environment,samples:32}});
const diffuseImages=Object.fromEntries(Object.keys(images).map(k=>[k,new Float64Array(images[k].length)]));
await integrateCoherentLighting({scene:diffuseScene,source,camera,options,lighting:diffuseLight,spectralPlan:prepareCoherentSpectrum(source,options),spatialSources:[],images:diffuseImages});
const primary=traceCoherentCamera(diffuseScene,camera,0,0,1,1,.5,.5,options.spectralSeed);assert.equal(primary.status,'diffuse');
const seedFor=(seed,index,domain)=>hashSpectralIndex(seed,index,0,domain)>>>0,seed=seedFor(diffuseLight.environment.seed,0,0x50534b59),terms=[];
for(let k=0;k<32;k++){
 const ray=traceCoherentRay(diffuseScene,primary.point,cosineDirection(primary.normal,k,32,seed),2,diffuseScene.triangles[primary.triangleId],seedFor(seed,k,0x534b534c));
 if(ray.status==='miss')terms.push({weight:.3*primary.throughput*ray.throughput/32,length:primary.glassLength+(ray.glassLength??0)});
}
assert(terms.length>8);assert(new Set(terms.map(t=>t.length)).size>8,'oracle exercises distinct glass lengths');
const diffusePower=nm=>terms.reduce((sum,t)=>sum+.5*t.weight*Math.exp(-slabAbsorptionAt(absorbing,nm)*t.length),0);
close(diffuseImages.coherent[0],diffusePower(450)+diffusePower(650));
for(let c=0;c<3;c++)close(diffuseImages.coherentXYZ[c],diffusePower(450)*cieXYZ(450)[c]+diffusePower(650)*cieXYZ(650)[c]);
console.log('PASS diffuse sky retains distinct escape path lengths for spectral Beer before XYZ');
