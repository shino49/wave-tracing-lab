import assert from 'node:assert/strict';
import {integrateCoherentLighting} from '../dist/coherent-mesh-lighting-runtime.js';
import {prepareCoherentMeshScene} from '../dist/coherent-mesh-core.js';
import {validateCoherentLighting,cosineDirection} from '../dist/coherent-mesh-environment.js';
import {prepareCoherentSpectrum} from '../dist/coherent-mesh-spectrum.js';
import {prepareCoherentSpatialSources} from '../dist/coherent-mesh-spatial.js';
import {traceCoherentCamera,traceCoherentRay} from '../dist/coherent-mesh-camera.js';
import {hashSpectralIndex,cieXYZ} from '../dist/plt-spectrum.js';
const flat=value=>[{wavelengthNm:360,value},{wavelengthNm:830,value}];
const plate=(z,m,reverse=false)=>({mesh:{vertices:[[-1e4,-1e4,z],[1e4,-1e4,z],[1e4,1e4,z],[-1e4,1e4,z]],faces:reverse?[[2,1,0],[3,2,0]]:[[0,1,2],[0,2,3]]},materialIndex:m,surface:'open-plate'});
const materials=[{type:'diffuse',reflectance:flat(.6)},{type:'diffuse',reflectance:flat(.4)}];
const scene=prepareCoherentMeshScene([plate(10,0,true)],materials,{maxReflections:0});
const camera={position:[0,0,9],target:[0,0,10],up:[0,1,0],fovDeg:20};
const source={position:[0,0,5],power:2,phaseRad:0,wavelengthMm:.00055};
const options={resolution:2,pixelQuadrature:1,maxReflections:0,diffraction:false,mixedDiffraction:false,coherence:1,spectralSamples:16,spectralSeed:7391};
const names=['coherent','incoherent','geometric','crossTerm','coherentXYZ','incoherentXYZ','geometricXYZ'];
const makeImages=()=>Object.fromEntries(names.map(k=>[k,new Float64Array(k.endsWith('XYZ')?12:4)]));
const close=(a,b,t=1e-12)=>assert(Math.abs(a-b)<=t,`${a} != ${b}`);
async function run({lighting,src=source,s=scene,runner=null,...extra}={}){const images=makeImages(),diagnostics={};await integrateCoherentLighting({scene:s,source:src,camera,options,lighting:validateCoherentLighting(lighting),runner,spectralPlan:prepareCoherentSpectrum(src,options),spatialSources:prepareCoherentSpatialSources(src),images,diagnostics,...extra});return {images,diagnostics};}
const light={environment:{radiance:.3,upper:1,lower:1,samples:4,spectrum:{mode:'monochrome'}}};
const a=await run({lighting:light});for(const k of ['coherent','incoherent','geometric'])for(const v of a.images[k])close(v,.18);assert(a.images.crossTerm.every(v=>v===0));
for(const src of [{...source,power:0,wavelengthMm:.00065},{...source,power:100,wavelengthMm:.00045,spatial:{type:'disk',radiusMm:.2,normal:[0,0,1],samples:4,seed:3}}]){const b=await run({lighting:light,src});for(const k of names)assert.deepEqual(a.images[k],b.images[k]);}
assert.equal(a.diagnostics.lightingEnvironmentRays,16);assert.equal(a.diagnostics.lightingEnvironmentEscapes,16);
console.log('PASS uniform sky L=rho*L; environment independent of source power/wavelength/disk count and contributes zero cross term');
const mirrorScene=prepareCoherentMeshScene([plate(10,0,true)],[{type:'mirror',reflectance:.4,phaseRad:0}],{maxReflections:0});
const reflected=await run({lighting:light,s:mirrorScene});for(const v of reflected.images.coherent)close(v,.12);
console.log('PASS camera mirror escape samples sky with mirror throughput');
const two=prepareCoherentMeshScene([plate(10,0,true),plate(0,1)],materials,{maxReflections:0});
const indirect={indirectSamples:4},b=await run({lighting:indirect,s:two});
for(let y=0;y<2;y++)for(let x=0;x<2;x++){
 const index=y*2+x,p=traceCoherentCamera(two,camera,x,y,2,2),seed=hashSpectralIndex(7391,index,0,0x494e4452)>>>0;let expected=0;
 for(let k=0;k<4;k++){const h=traceCoherentRay(two,p.point,cosineDirection(p.normal,k,4,seed),2,two.triangles[p.triangleId]);const d=source.position.map((v,i)=>v-h.point[i]),r=Math.hypot(...d),cos=d[2]/r;expected+=.6*.4/Math.PI*source.power/(4*Math.PI*r*r)*cos/4;}
 close(b.images.coherent[index],expected);close(b.images.geometric[index],expected);close(b.images.incoherent[index],expected);
}
assert.equal(b.diagnostics.lightingSecondaryHits,16);assert.equal(b.diagnostics.lightingSourceSamples,16);
console.log('PASS independently evaluated one-diffuse source bounce: no inverse-square along camera leg, rhoPrimary*rhoSecondary/pi, source projected incidence');
let calls=0;const runner={async evaluate({records,sampleCount,routeCount}){calls++;const coherent=new Float64Array(sampleCount),incoherent=new Float64Array(sampleCount);for(let s=0;s<sampleCount;s++){let re=0,im=0;for(let j=0;j<routeCount;j++){const at=(s*routeCount+j)*4,a=records[at],phase=records[at+1];re+=a*Math.cos(phase);im+=a*Math.sin(phase);incoherent[s]+=a*a;}coherent[s]=re*re+im*im;}return {coherent,incoherent};}};
const gpu=await run({lighting:indirect,s:two,runner});assert(calls>0);for(const k of names)for(let i=0;i<b.images[k].length;i++)close(gpu.images[k][i],b.images[k][i],1e-10);
await assert.rejects(run({lighting:indirect,s:two,runner:{async evaluate(){return {coherent:[NaN],incoherent:[0]};}}}),e=>e.coherentGPUFailure===true);
console.log('PASS GPU source-indirect batches used, results match CPU; malformed GPU fields trigger full-attempt fallback marker');
const lines={...light,environment:{...light.environment,spectrum:{mode:'lines',lines:[{wavelengthNm:450,powerFraction:1},{wavelengthNm:650,powerFraction:1}]}}};
const colored=prepareCoherentMeshScene([plate(10,0,true)],[{type:'diffuse',reflectance:[{wavelengthNm:360,value:0},{wavelengthNm:450,value:0},{wavelengthNm:650,value:1},{wavelengthNm:830,value:1}]}]);
const c=await run({lighting:lines,s:colored});for(const v of c.images.coherent)close(v,.15);for(let i=0;i<4;i++)for(let channel=0;channel<3;channel++)close(c.images.coherentXYZ[3*i+channel],.15*cieXYZ(650)[channel]);
let rows=0;await assert.rejects(run({lighting:light,preview(){rows++;},check(){if(rows)throw new Error('cancelled');}}),/cancelled/);assert.equal(rows,1);
console.log('PASS environment reflectance per wavelength before XYZ, complete-row previews and cooperative cancellation');
const sourceLines={...source,spectrum:{mode:'lines',lines:[{wavelengthNm:450,powerFraction:1},{wavelengthNm:650,powerFraction:1}]}};
const spectralBounce=await run({lighting:indirect,s:two,src:sourceLines});
for(let i=0;i<4;i++){close(spectralBounce.images.coherent[i],b.images.coherent[i]);for(let c=0;c<3;c++)close(spectralBounce.images.coherentXYZ[3*i+c],b.images.coherent[i]*(cieXYZ(450)[c]+cieXYZ(650)[c])/2);}
console.log('PASS spectral indirect integrates complete source fields per wavelength then XYZ');
// Independent finite-sample oracle for sky -> colored secondary -> colored
// primary. Only geometry queries and the public directional sampler are shared;
// spectral products and estimator factors are evaluated explicitly here.
const color=(blue,red)=>[{wavelengthNm:360,value:blue},{wavelengthNm:450,value:blue},{wavelengthNm:650,value:red},{wavelengthNm:830,value:red}];
const primaryPlate={mesh:{vertices:[[-2,-2,10],[2,-2,10],[2,2,10],[-2,2,10]],faces:[[2,1,0],[3,2,0]]},materialIndex:0,surface:'open-plate'};
const colors=[[.2,.7],[.8,.1]];
const finiteSecondary={mesh:{vertices:[[-10,-10,0],[10,-10,0],[10,10,0],[-10,10,0]],faces:[[0,1,2],[0,2,3]]},materialIndex:1,surface:'open-plate'};
const coloredPair=(secondaryScale=1)=>prepareCoherentMeshScene([primaryPlate,finiteSecondary],[{type:'diffuse',reflectance:color(...colors[0])},{type:'diffuse',reflectance:color(...colors[1].map(v=>v*secondaryScale))}],{maxReflections:0});
const nestedLight={environment:{radiance:.3,upper:1,lower:1,samples:8,seed:7391,spectrum:{mode:'lines',lines:[{wavelengthNm:450,powerFraction:1},{wavelengthNm:650,powerFraction:1}]}},indirectSamples:4};
const pair=coloredPair(),zeroSource={...source,power:0};
const nested=await run({lighting:nestedLight,s:pair,src:zeroSource});
const directOnly=await run({lighting:{...nestedLight,indirectSamples:0},s:pair,src:zeroSource});
const hash=(index,domain)=>hashSpectralIndex(7391,index,0,domain)>>>0;
let positiveIndirect=0;
for(let y=0;y<2;y++)for(let x=0;x<2;x++){
 const index=y*2+x,p=traceCoherentCamera(pair,camera,x,y,2,2),launch=(h,d)=>traceCoherentRay(pair,h.point,d,2,pair.triangles[h.triangleId]);
 let directAngular=0,indirectAngular=0;
 for(let j=0;j<8;j++){const r=launch(p,cosineDirection(p.normal,j,8,hash(index,0x50534b59)));if(r.status==='miss')directAngular+=.3*r.throughput/8;}
 for(let k=0;k<4;k++){
  const s=launch(p,cosineDirection(p.normal,k,4,hash(index,0x494e4452)));
  if(s.status!=='diffuse'||s.normal.reduce((sum,v,i)=>sum+v*s.viewDirection[i],0)<=0)continue;
  const pairIndex=hash(index,k+0x50414952);let sky=0;
  for(let j=0;j<8;j++){const r=launch(s,cosineDirection(s.normal,j,8,hash(pairIndex,0x53534b59)));if(r.status==='miss')sky+=.3*r.throughput/8;}
  indirectAngular+=p.throughput*s.throughput*sky/4;
 }
 positiveIndirect+=indirectAngular;
 const expectedDirect=directAngular*(.2+.7)/2,expectedIndirect=indirectAngular*(.2*.8+.7*.1)/2;
 close(directOnly.images.coherent[index],expectedDirect);
 for(const key of ['coherent','incoherent','geometric'])close(nested.images[key][index],expectedDirect+expectedIndirect);
 close(nested.images.crossTerm[index],0);
 for(let c=0;c<3;c++){
  const expected=(directAngular*.2+indirectAngular*.2*.8)*cieXYZ(450)[c]/2+(directAngular*.7+indirectAngular*.7*.1)*cieXYZ(650)[c]/2;
  for(const key of ['coherentXYZ','incoherentXYZ','geometricXYZ'])close(nested.images[key][3*index+c],expected);
 }
}
assert(positiveIndirect>0,'fixture must contain visible secondary sky');
assert(directOnly.images.coherent.some(v=>v>0),'fixture must also have nonzero direct sky to detect double counting');
for(const scale of [0,.5]){
 const scaled=await run({lighting:nestedLight,s:coloredPair(scale),src:zeroSource});
 for(const key of names)for(let i=0;i<scaled.images[key].length;i++)close(scaled.images[key][i],directOnly.images[key][i]+scale*(nested.images[key][i]-directOnly.images[key][i]));
}
const independentSourceSpectrum=await run({lighting:nestedLight,s:pair,src:{...zeroSource,wavelengthMm:.00042,spectrum:{mode:'lines',lines:[{wavelengthNm:410,powerFraction:1},{wavelengthNm:730,powerFraction:2}]}}});
for(const key of names)assert.deepEqual(independentSourceSpectrum.images[key],nested.images[key]);
console.log('PASS independent sky-secondary-primary colored spectral-product oracle; direct sky counted once; zero/half secondary rho, zero source power, independent source spectrum');
