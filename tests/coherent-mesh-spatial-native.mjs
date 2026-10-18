/** Native Dawn/SwiftShader evidence, not browser/physical GPU validation.
 * VK_ICD_FILENAMES=/usr/lib/chromium/vk_swiftshader_icd.json WEBGPU_EXECUTION_BACKEND=vulkan node tests/coherent-mesh-spatial-native.mjs
 */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {createHash} from 'node:crypto';
import {create,globals} from 'webgpu';
import {executeCoherentMeshTask} from '../dist/coherent-mesh-runtime.js';
import {createCoherentMeshSpectralGPURunner} from '../dist/coherent-mesh-spectral-gpu.js';
import {createCoherentMeshGPURunner} from '../dist/coherent-mesh-gpu.js';
import {displaySpectralXYZ} from '../dist/plt-spectrum.js';
Object.assign(globalThis,globals);
const backend=process.env.WEBGPU_EXECUTION_BACKEND;assert.equal(backend,'vulkan');
const gpu=create([`backend=${backend}`]);
const shaderSource=fs.readFileSync('dist/utd-functions.wgsl','utf8')+'\n'+fs.readFileSync('dist/coherent-mesh-spectral-compute.wgsl','utf8');
const scene=JSON.parse(fs.readFileSync('dist/examples/coherent-lloyd-d65-color.json','utf8'));
const report={schemaVersion:1,scope:'Native Dawn Vulkan SwiftShader software execution. CPU geometry; GPU API spectral field/XYZ integration; no browser or physical-GPU validation.',browserValidated:false,physicalGPUValidated:false,cpuShadow:false,driver:process.env.VK_ICD_FILENAMES,baseExposureEV:16,sharedExposureEV:0,scene,hashes:{},parity:[],cases:[]};
for(const file of ['dist/coherent-mesh-runtime.js','dist/coherent-mesh-spatial.js','dist/coherent-mesh-scene.js','dist/coherent-mesh-spectral-gpu.js','dist/coherent-mesh-spectral-compute.wgsl','dist/plt-spectrum.js','tests/coherent-mesh-spatial-native.mjs'])report.hashes[file]=createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const save=()=>fs.writeFileSync('research/spatial-native-report.json',JSON.stringify(report,null,2)+'\n');
const source=samples=>({...scene.source,...(samples?{spatial:{type:'disk',radiusMm:.06,normal:[0,0,1],samples,seed:7391}}:{})});
async function run(resolution,samples,backend='auto'){
 let last=-1;const t=performance.now();
 const result=await executeCoherentMeshTask({scene,source:source(samples),resolution,pixelQuadrature:1,spectralSamples:16,spectralSeed:7391,backend,maxRuntimeMs:600000},{spectralGpuFactory:options=>createCoherentMeshSpectralGPURunner({...options,gpu,shaderSource}),progress:p=>{const dec=Math.floor((p.fraction??0)*10);if(dec!==last){last=dec;console.log(`${resolution} disk${samples} ${backend} ${dec*10}% ${(performance.now()-t).toFixed(0)}ms`);}}});
 assert(result.complete&&result.spatialComplete&&result.spectralComplete);assert.equal(result.diagnostics.productionCPUShadow,false);
 if(backend==='auto'){assert.equal(result.backend.kind,'hybrid');assert.equal(result.diagnostics.fallbackReason,null);}
 return result;
}
function compare(a,b){const errors={};for(const key of ['coherent','incoherent','geometric','coherentXYZ','incoherentXYZ','geometricXYZ']){let abs=0,peak=0;for(let i=0;i<a[key].length;i++){abs=Math.max(abs,Math.abs(a[key][i]-b[key][i]));peak=Math.max(peak,Math.abs(b[key][i]));}errors[key]={maxAbsolute:abs,relativeToPeak:abs/peak};assert(abs/peak<1e-4,JSON.stringify({key,...errors[key]}));}return errors;}
function metrics(r){let cross=0,inc=0,colorCross=0,yinc=0,meanY=0;for(let p=0;p<r.coherent.length;p++){cross+=r.crossTerm[p]**2;inc+=r.incoherent[p]**2;colorCross+=(r.coherentXYZ[3*p+1]-r.incoherentXYZ[3*p+1])**2;yinc+=r.incoherentXYZ[3*p+1]**2;meanY+=r.coherentXYZ[3*p+1];}return {normalizedScalarCrossTermRMS:Math.sqrt(cross/inc),normalizedYCrossTermRMS:Math.sqrt(colorCross/yinc),meanY:meanY/r.coherent.length};}
function outputs(name,r){for(const key of ['coherent','incoherent','coherentXYZ','incoherentXYZ'])fs.writeFileSync(`research/spatial-native-${name}-${key}.f64`,Buffer.from(r[key].buffer));for(const key of ['coherent','incoherent']){const rgb=Buffer.alloc(r.width*r.height*3);for(let p=0;p<r.width*r.height;p++)rgb.set(displaySpectralXYZ(r[key+'XYZ'].subarray(3*p,3*p+3),16).rgb,p*3);fs.writeFileSync(`research/spatial-native-${name}-${key}.ppm`,Buffer.concat([Buffer.from(`P6\n${r.width} ${r.height}\n255\n`),rgb]));}const entry={name,width:r.width,height:r.height,metrics:metrics(r),configuration:r.configuration,timings:r.timings,backend:r.backend,diagnostics:r.diagnostics,rawFormat:'little-endian float64 row-major, XYZ interleaved; untouched solver values',display:'Exact imported displaySpectralXYZ at fixed 16 EV; same for every image; no normalization'};report.cases.push(entry);save();console.log('CASE '+JSON.stringify(entry.metrics));}
try{
 if(process.argv.includes('--mono-only')){
  const existing=JSON.parse(fs.readFileSync('research/spatial-native-report.json','utf8'));Object.assign(report,existing);
  const monoShader=fs.readFileSync('dist/coherent-mesh-compute.wgsl','utf8');
  const input={scene,source:{...source(16),spectrum:null},resolution:16,pixelQuadrature:1};
  const cpu=await executeCoherentMeshTask({...input,backend:'cpu'});
  const native=await executeCoherentMeshTask({...input,backend:'auto'},{gpuFactory:options=>createCoherentMeshGPURunner({...options,gpu,shaderSource:monoShader})});
  assert.equal(native.backend.kind,'hybrid');assert.equal(native.diagnostics.fallbackReason,null);assert.equal(native.diagnostics.productionCPUShadow,false);
  report.parity.push({mode:'monochrome',resolution:16,spatialSamples:16,errors:compare(native,cpu),cpuMs:cpu.timings.totalMs,nativeMs:native.timings.totalMs});
  for(const file of Object.keys(report.hashes))report.hashes[file]=createHash('sha256').update(fs.readFileSync(file)).digest('hex');
  report.hashes['dist/coherent-mesh-compute.wgsl']=createHash('sha256').update(monoShader).digest('hex');
  report.hashNote='Final file hashes; runtime post-render changes limited to overview/diagnostic metadata, no numerical path changes.';
 }else{
 for(const samples of [0,16]){const cpu=await run(16,samples,'cpu'),native=await run(16,samples);report.parity.push({resolution:16,spatialSamples:samples||1,errors:compare(native,cpu),cpuMs:cpu.timings.totalMs,nativeMs:native.timings.totalMs});save();}
 for(const samples of [0,16])outputs(samples?'disk16':'point',await run(512,samples));
 const [p,d]=report.cases;report.fringeReduction={scalarRMSFraction:1-d.metrics.normalizedScalarCrossTermRMS/p.metrics.normalizedScalarCrossTermRMS,luminanceRMSFraction:1-d.metrics.normalizedYCrossTermRMS/p.metrics.normalizedYCrossTermRMS,definition:'1 - disk/point normalized RMS(coherent-incoherent). Full 6x1 mm detector; fixed total source power. It is a descriptive integrated interference contrast, not a statistical confidence interval.'};report.passed=true;
 }
}catch(e){report.passed=false;report.error=e.stack;process.exitCode=1;console.error(e);}finally{report.completedAt=new Date().toISOString();save();}
