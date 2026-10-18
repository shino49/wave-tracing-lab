/** Native Dawn/SwiftShader Vulkan evidence; no browser or physical GPU claim. */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {createHash} from 'node:crypto';
import {create,globals} from 'webgpu';
import {executeCoherentMeshTask} from '../dist/coherent-mesh-runtime.js';
import {createCoherentMeshSpectralGPURunner} from '../dist/coherent-mesh-spectral-gpu.js';
import {createCoherentMeshGPURunner} from '../dist/coherent-mesh-gpu.js';
import {displaySpectralXYZ} from '../dist/plt-spectrum.js';
import {makeCoherentCameraPreset} from '../dist/coherent-mesh-camera.js';
Object.assign(globalThis,globals);
assert.equal(process.env.WEBGPU_EXECUTION_BACKEND,'vulkan');
const gpu=create(['backend=vulkan']);
const spectralShader=fs.readFileSync('dist/utd-functions.wgsl','utf8')+'\n'+fs.readFileSync('dist/coherent-mesh-spectral-compute.wgsl','utf8');
const monoShader=fs.readFileSync('dist/coherent-mesh-compute.wgsl','utf8');
const report={scope:'Native Dawn Vulkan SwiftShader software execution. CPU geometry, native WebGPU intensity/XYZ integration. No browser or physical GPU validation.',cpuShadow:false,driver:process.env.VK_ICD_FILENAMES,exposureEV:18,parity:[],cases:[],hashes:{}};
for(const file of ['dist/coherent-mesh-runtime.js','dist/coherent-mesh-camera.js','dist/coherent-mesh-spectral-gpu.js','dist/coherent-mesh-spectral-compute.wgsl','dist/coherent-mesh-compute.wgsl'])report.hashes[file]=createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const save=()=>fs.writeFileSync('research/camera-native-report.json',JSON.stringify(report,null,2)+'\n');
const base={preset:'object-imaging',measurement:'camera',diffraction:true,mixedDiffraction:false,resolution:16,pixelQuadrature:1,spectralSamples:16,spectralSeed:7391,maxRuntimeMs:600000};
const spectral={mode:'broadband',illuminant:'D65'};
async function run(input,backend='auto'){
 const start=performance.now();let firstPreviewMs=null,last=-1;
 const result=await executeCoherentMeshTask({...base,...input,backend},{gpuFactory:o=>createCoherentMeshGPURunner({...o,gpu,shaderSource:monoShader}),spectralGpuFactory:o=>createCoherentMeshSpectralGPURunner({...o,gpu,shaderSource:spectralShader}),preview:p=>{if(firstPreviewMs===null)firstPreviewMs=performance.now()-start;},progress:p=>{const step=Math.floor((p.fraction??0)*10);if(step!==last){last=step;console.log(`${input.resolution??16} ${backend} ${step*10}% ${(performance.now()-start).toFixed(0)}ms`);}}});
 assert(result.complete&&result.spatialComplete&&result.spectralComplete);assert.equal(result.diagnostics.productionCPUShadow,false);
 if(backend==='auto'){assert.equal(result.backend.kind,'hybrid');assert.equal(result.diagnostics.fallbackReason,null);}
 result.firstPreviewMs=firstPreviewMs;return result;
}
const keys=['coherent','geometric','incoherent','crossTerm','coherentXYZ','geometricXYZ','incoherentXYZ'];
function compare(a,b){const errors={};for(const key of keys){let max=0,peak=0;for(let i=0;i<a[key].length;i++){max=Math.max(max,Math.abs(a[key][i]-b[key][i]));peak=Math.max(peak,Math.abs(b[key][i]));}errors[key]={maxAbsolute:max,relativeToPeak:peak?max/peak:0};assert(errors[key].relativeToPeak<1e-4,JSON.stringify({key,...errors[key]}));}return errors;}
function outputs(name,r){
 for(const key of keys)fs.writeFileSync(`research/camera-native-${name}-${key}.f64`,Buffer.from(r[key].buffer,r[key].byteOffset,r[key].byteLength));
 for(const key of ['coherent','geometric','incoherent']){const rgb=Buffer.alloc(r.width*r.height*3);for(let p=0;p<r.width*r.height;p++)rgb.set(displaySpectralXYZ(r[key+'XYZ'].subarray(p*3,p*3+3),report.exposureEV).rgb,p*3);fs.writeFileSync(`research/camera-native-${name}-${key}.ppm`,Buffer.concat([Buffer.from(`P6\n${r.width} ${r.height}\n255\n`),rgb]));}
 const rgb=Buffer.alloc(r.width*r.height*3);let sum2=0,inc2=0,positive=0;for(let p=0;p<r.width*r.height;p++){const c=r.crossTerm[p];sum2+=c*c;inc2+=r.incoherent[p]**2;if(r.coherent[p]>0)positive++;const a=Math.min(255,Math.round(255*Math.sqrt(Math.abs(c)*2**report.exposureEV)));rgb.set(c>=0?[a,0,0]:[0,0,a],p*3);}fs.writeFileSync(`research/camera-native-${name}-crossTerm.ppm`,Buffer.concat([Buffer.from(`P6\n${r.width} ${r.height}\n255\n`),rgb]));
 const entry={name,width:r.width,height:r.height,firstPreviewMs:r.firstPreviewMs,timings:r.timings,backend:r.backend,diagnostics:r.diagnostics,configuration:r.configuration,positivePixels:positive,normalizedCrossTermRMS:Math.sqrt(sum2/inc2),rawFormat:'little-endian float64 row-major, XYZ interleaved; untouched solver outputs',display:`Exact displaySpectralXYZ with fixed +${report.exposureEV} EV, identical coherent/geometric/incoherent exposure; no normalization. Signed cross term uses red positive, blue negative and sqrt(abs(cross)*2^EV), clamped to 1; same fixed EV.`};report.cases.push(entry);save();console.log('CASE '+JSON.stringify(entry));
}
try{
 for(const [name,input] of [['mono',{}],['spectral',{spectrum:spectral}],['endpoint360',{spectrum:{mode:'lines',lines:[{wavelengthNm:360,powerFraction:1}]}}],['endpoint830',{spectrum:{mode:'lines',lines:[{wavelengthNm:830,powerFraction:1}]}}],['spectral-disk2',{spectrum:spectral,source:{...makeCoherentCameraPreset().source,spatial:{type:'disk',radiusMm:.06,normal:[0,0,1],samples:2,seed:7391}}}]]){const cpu=await run(input,'cpu'),native=await run(input);report.parity.push({name,errors:compare(native,cpu),cpuMs:cpu.timings.totalMs,nativeMs:native.timings.totalMs});save();console.log('PARITY '+name);}
 const phase0=await run({spectrum:spectral},'cpu'),phase1=await run({spectrum:spectral,phaseShiftRad:1},'cpu');let waveDifference=0,goDifference=0;for(let i=0;i<phase0.coherent.length;i++){waveDifference=Math.max(waveDifference,Math.abs(phase0.coherent[i]-phase1.coherent[i]));goDifference=Math.max(goDifference,Math.abs(phase0.geometric[i]-phase1.geometric[i]));}assert(waveDifference>1e-10);assert(goDifference<1e-18);report.phaseRegression={phaseShiftRad:1,waveMaxAbsoluteDifference:waveDifference,geometricMaxAbsoluteDifference:goDifference};save();
 outputs('point512',await run({resolution:512,spectrum:spectral}));report.passed=true;
}catch(e){report.passed=false;report.error=e.stack;process.exitCode=1;console.error(e);}finally{report.completedAt=new Date().toISOString();save();}
