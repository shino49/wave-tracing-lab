/** Native Dawn/SwiftShader only. Never launches a browser or claims hardware execution. */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {createHash} from 'node:crypto';
import {create,globals} from 'webgpu';
import {executeCoherentMeshTask,prepareCoherentMeshTask} from '../dist/coherent-mesh-runtime.js';
import {createCoherentMeshSpectralGPURunner} from '../dist/coherent-mesh-spectral-gpu.js';
import {createCoherentMeshGPURunner} from '../dist/coherent-mesh-gpu.js';
import {displaySpectralXYZ} from '../dist/plt-spectrum.js';
import {makeCoherentCameraPreset,traceCoherentCamera} from '../dist/coherent-mesh-camera.js';
import {sceneFromCoherentData} from '../dist/coherent-mesh-scene.js';
Object.assign(globalThis,globals);
assert.equal(process.env.WEBGPU_EXECUTION_BACKEND,'vulkan');
const gpu=create(['backend=vulkan']),mode=process.argv[2]??'fast';
const spectralShader=fs.readFileSync('dist/utd-functions.wgsl','utf8')+'\n'+fs.readFileSync('dist/coherent-mesh-spectral-compute.wgsl','utf8'),monoShader=fs.readFileSync('dist/coherent-mesh-compute.wgsl','utf8');
const prefix='research/environment-',report={mode,scope:'Native Dawn Vulkan SwiftShader software execution. CPU geometry; native WebGPU integration. No browser, physical GPU, or CPU shadow.',exposureEV:18,parity:[],cases:[],hashes:{},startedAt:new Date().toISOString()};
for(const f of ['coherent-mesh-runtime.js','coherent-mesh-lighting-runtime.js','coherent-mesh-environment.js','coherent-mesh-camera.js','coherent-mesh-spectral-gpu.js','coherent-mesh-spectral-compute.wgsl'])report.hashes[f]=createHash('sha256').update(fs.readFileSync('dist/'+f)).digest('hex');
const save=()=>fs.writeFileSync(prefix+mode+'-report.json',JSON.stringify(report,null,2)+'\n');
const preset=makeCoherentCameraPreset(),camera={...preset.camera,mirrorBounces:2},spectrum={mode:'broadband',illuminant:'D65'},base={preset:'object-imaging',measurement:'camera',camera,diffraction:true,mixedDiffraction:false,resolution:16,pixelQuadrature:1,spectralSamples:16,spectralSeed:7391,maxRuntimeMs:600000};
const keys=['coherent','geometric','incoherent','crossTerm','coherentXYZ','geometricXYZ','incoherentXYZ'];
async function run(name,input={},backend='auto'){
 const start=performance.now();let firstCompleteTileMs=null,firstPreviewMs=null,firstLightingRowMs=null,last=-1;
 const r=await executeCoherentMeshTask({...base,...input,backend},{gpuFactory:o=>createCoherentMeshGPURunner({...o,gpu,shaderSource:monoShader}),spectralGpuFactory:o=>createCoherentMeshSpectralGPURunner({...o,gpu,shaderSource:spectralShader}),preview:p=>{if(firstPreviewMs===null)firstPreviewMs=performance.now()-start;if(p.rowCount===1&&p.lightingComplete===false&&firstLightingRowMs===null)firstLightingRowMs=performance.now()-start;if(p.spatialComplete&&firstCompleteTileMs===null){firstCompleteTileMs=performance.now()-start;console.log('FIRST_TILE '+name+' '+firstCompleteTileMs);}},progress:p=>{const step=Math.floor((p.fraction??0)*10);if(step!==last){last=step;console.log(name+' '+step*10+'% '+Math.round(performance.now()-start)+'ms');}}});
 assert(r.complete&&r.spatialComplete&&r.spectralComplete);assert.equal(r.diagnostics.productionCPUShadow,false);if(backend==='auto'){assert.equal(r.backend.kind,'hybrid');assert.equal(r.diagnostics.fallbackReason,null);}r.firstCompleteTileMs=firstCompleteTileMs;r.firstPreviewMs=firstPreviewMs;r.firstLightingRowMs=firstLightingRowMs;return r;
}
function errors(a,b){const out={};for(const k of keys){let s=0,t=0,max=0,peak=0;for(let i=0;i<a[k].length;i++){const d=a[k][i]-b[k][i];s+=d*d;t+=b[k][i]**2;max=Math.max(max,Math.abs(d));peak=Math.max(peak,Math.abs(b[k][i]));}out[k]={relativeL2:Math.sqrt(s/(t||1)),relativeToPeak:max/(peak||1),maxAbsolute:max};}return out;}
function output(name,r){
 for(const k of keys)fs.writeFileSync(prefix+name+'-'+k+'.f64',Buffer.from(r[k].buffer,r[k].byteOffset,r[k].byteLength));
 for(const k of ['coherent','geometric','incoherent','crossTerm']){const rgb=Buffer.alloc(r.width*r.height*3);for(let i=0;i<r.width*r.height;i++){if(k==='crossTerm'){const c=r.crossTerm[i],a=Math.min(255,Math.round(255*Math.sqrt(Math.abs(c)*2**18)));rgb.set(c>=0?[a,0,0]:[0,0,a],3*i);}else rgb.set(displaySpectralXYZ(r[k+'XYZ'].subarray(i*3,i*3+3),18).rgb,i*3);}fs.writeFileSync(prefix+name+'-'+k+'.ppm',Buffer.concat([Buffer.from(`P6\n${r.width} ${r.height}\n255\n`),rgb]));}
 const entry={name,width:r.width,height:r.height,firstPreviewMs:r.firstPreviewMs,firstLightingRowMs:r.firstLightingRowMs,firstCompleteTileMs:r.firstCompleteTileMs,timings:r.timings,backend:r.backend,diagnostics:r.diagnostics,configuration:r.configuration,rawFormat:'Unmodified float64 little-endian row-major; XYZ interleaved.',display:'Fixed +18EV displaySpectralXYZ, no normalization or filtering. Cross term red-positive/blue-negative sqrt(abs(c)*2^18).'};report.cases.push(entry);save();console.log('CASE '+JSON.stringify(entry));
}

const environment={radiance:2e-6,spectrum:{mode:'broadband',illuminant:'D65'},wavelengthNm:550,up:[0,1,0],upper:1,lower:.25,samples:8,seed:7391};
const lighting={environment,indirectSamples:4};
function difference(a,b){let n=0,mean=0,max=0,min=Infinity;for(let i=0;i<a.length;i++){const d=a[i]-b[i];mean+=d;max=Math.max(max,d);min=Math.min(min,d);if(d>0)n++;}return {positivePixels:n,mean:mean/a.length,max,min};}
function masks(baseline,env,indirect){
 const {scene}=prepareCoherentMeshTask({...base,resolution:baseline.width});const groups={},mask=new Uint8Array(baseline.width**2);
 for(let y=0;y<baseline.height;y++)for(let x=0;x<baseline.width;x++){
  const p=y*baseline.width+x,h=traceCoherentCamera(scene,camera,x,y,baseline.width,baseline.height);
  if(h.status!=='diffuse')continue;
  mask[p]=h.materialIndex;const key=`material${h.materialIndex}-`+(h.primaryMaterialType==='mirror'?'reflected':'direct');
  const g=groups[key]??={pixels:0,baseline:0,env:0,indirect:0,envBrightened:0,indirectBrightened:0,darkPixels:0,darkBrightened:0};g.pixels++;g.baseline+=baseline.coherent[p];g.env+=env.coherent[p];g.indirect+=indirect.coherent[p];if(env.coherent[p]>baseline.coherent[p])g.envBrightened++;if(indirect.coherent[p]>env.coherent[p])g.indirectBrightened++;if(baseline.coherent[p]<1e-7){g.darkPixels++;if(env.coherent[p]>baseline.coherent[p])g.darkBrightened++;}
 }
 fs.writeFileSync(prefix+'material-mask.u8',mask);return {groups,environmentDelta:difference(env.coherent,baseline.coherent),indirectDelta:difference(indirect.coherent,env.coherent)};
}
try{
 if(mode==='parity'){
  const reduced=structuredClone(preset);reduced.materials[1].reflectance=[{wavelengthNm:360,value:.31},{wavelengthNm:830,value:.31}];reduced.materials[2].reflectance=[{wavelengthNm:360,value:.23},{wavelengthNm:830,value:.23}];const scene=sceneFromCoherentData(reduced);
  for(const [name,input] of [['mono',{lighting}],['D65',{spectrum,lighting}],['D65-disk2',{spectrum,lighting,source:{...preset.source,spatial:{type:'disk',radiusMm:.06,normal:[0,0,1],samples:2,seed:7391}}}],['independent-mono-sky',{spectrum,lighting:{...lighting,environment:{...environment,spectrum:{mode:'monochrome'},wavelengthNm:610}}}],['rho-primary-secondary',{scene,spectrum,lighting}]]){
   const cpu=await run(name+'-cpu',input,'cpu'),native=await run(name+'-native',input);const e=errors(native,cpu);for(const k of keys)assert(e[k].relativeToPeak<1e-4,JSON.stringify({name,k,...e[k]}));report.parity.push({name,errors:e,cpuMs:cpu.timings.totalMs,nativeMs:native.timings.totalMs,diagnostics:native.diagnostics});save();console.log('PARITY '+name);
  }
 }else if(mode==='fast'){
  const baseline=await run('baseline512',{resolution:512,spectrum,lighting:null});output('baseline512-q1-s16',baseline);
  const env=await run('sky512',{resolution:512,spectrum,lighting:{environment,indirectSamples:0}});output('sky512-q1-s16',env);
  const indirect=await run('diffuse512',{resolution:512,spectrum,lighting});output('diffuse512-q1-s16',indirect);
  report.evidence=masks(baseline,env,indirect);save();
 }else if(mode==='colorbleed'){
  const altered=structuredClone(preset);for(let m=2;m<5;m++)altered.materials[m].reflectance=[{wavelengthNm:360,value:.5},{wavelengthNm:830,value:.5}];const scene=sceneFromCoherentData(altered);
  const opts={resolution:64,spectrum,lighting:{environment:{...environment,samples:16},indirectSamples:8}};
  const colored=await run('colorbleed-original',opts),neutral=await run('colorbleed-neutral-secondary',{...opts,scene});output('colorbleed-original64',colored);output('colorbleed-neutral-secondary64',neutral);
  const world=prepareCoherentMeshTask(base).scene,selected=[];let sum=[0,0,0],absolute=0,affected=0;
  for(let y=0;y<64;y++)for(let x=0;x<64;x++){const p=y*64+x,h=traceCoherentCamera(world,camera,x,y,64,64);if(h.status==='diffuse'&&h.materialIndex===1){selected.push(p);let mag=0;for(let c=0;c<3;c++){const d=colored.coherentXYZ[3*p+c]-neutral.coherentXYZ[3*p+c];sum[c]+=d;absolute+=Math.abs(d);mag+=Math.abs(d);}if(mag>1e-15)affected++;}}
  assert(affected>0&&absolute>1e-12,'Secondary reflectance must change neutral-primary radiance');
  report.colorBleed={unchangedNeutralPrimaryPixels:selected.length,changedBySecondaryReflectance:affected,summedXYZChange:sum,absoluteXYZChange:absolute,scope:'Only colored diffuse reflectance spectra changed to flat0.5; neutral-primary pixels share identical geometry, source and environment direct lighting. Their changed XYZ is caused by secondary diffuse reflectance.'};save();
 }else if(mode==='convergence'){
  const all={};for(const [e,i] of [[8,4],[16,8],[32,8]]){const name=`grid64-e${e}-i${i}`;all[name]=await run(name,{resolution:64,spectrum,lighting:{environment:{...environment,samples:e},indirectSamples:i}});output(name,all[name]);}
  report.convergence={e8i4VsE32i8:errors(all['grid64-e8-i4'],all['grid64-e32-i8']),e16i8VsE32i8:errors(all['grid64-e16-i8'],all['grid64-e32-i8']),scope:'Non-nested fixed-seed stratified sample comparisons, not exact ground truth or a proof of asymptotic convergence; no blur or denoising.'};save();
 }else throw Error('Unknown mode '+mode);
 report.passed=true;
}catch(e){report.passed=false;report.error=e.stack;process.exitCode=1;console.error(e);}finally{report.completedAt=new Date().toISOString();save();}
