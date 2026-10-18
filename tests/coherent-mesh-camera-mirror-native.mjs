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
const prefix='research/camera-mirror-',report={mode,scope:'Native Dawn Vulkan SwiftShader software execution. CPU geometry; native WebGPU integration. No browser, physical GPU, or CPU shadow.',exposureEV:18,parity:[],cases:[],hashes:{},startedAt:new Date().toISOString()};
for(const f of ['coherent-mesh-runtime.js','coherent-mesh-camera.js','coherent-mesh-spectral-gpu.js','coherent-mesh-spectral-compute.wgsl'])report.hashes[f]=createHash('sha256').update(fs.readFileSync('dist/'+f)).digest('hex');
const save=()=>fs.writeFileSync(prefix+mode+'-report.json',JSON.stringify(report,null,2)+'\n');
const preset=makeCoherentCameraPreset(),camera={...preset.camera,mirrorBounces:2},spectrum={mode:'broadband',illuminant:'D65'},base={preset:'object-imaging',measurement:'camera',camera,diffraction:true,mixedDiffraction:false,resolution:16,pixelQuadrature:1,spectralSamples:16,spectralSeed:7391,maxRuntimeMs:600000};
const keys=['coherent','geometric','incoherent','crossTerm','coherentXYZ','geometricXYZ','incoherentXYZ'];
async function run(name,input={},backend='auto'){
 const start=performance.now();let firstCompleteTileMs=null,last=-1;
 const r=await executeCoherentMeshTask({...base,...input,backend},{gpuFactory:o=>createCoherentMeshGPURunner({...o,gpu,shaderSource:monoShader}),spectralGpuFactory:o=>createCoherentMeshSpectralGPURunner({...o,gpu,shaderSource:spectralShader}),preview:p=>{if(p.spatialComplete&&firstCompleteTileMs===null){firstCompleteTileMs=performance.now()-start;console.log('FIRST_TILE '+name+' '+firstCompleteTileMs);}},progress:p=>{const step=Math.floor((p.fraction??0)*10);if(step!==last){last=step;console.log(name+' '+step*10+'% '+Math.round(performance.now()-start)+'ms');}}});
 assert(r.complete&&r.spatialComplete&&r.spectralComplete);assert.equal(r.diagnostics.productionCPUShadow,false);if(backend==='auto'){assert.equal(r.backend.kind,'hybrid');assert.equal(r.diagnostics.fallbackReason,null);}r.firstCompleteTileMs=firstCompleteTileMs;return r;
}
function errors(a,b){const out={};for(const k of keys){let s=0,t=0,max=0,peak=0;for(let i=0;i<a[k].length;i++){const d=a[k][i]-b[k][i];s+=d*d;t+=b[k][i]**2;max=Math.max(max,Math.abs(d));peak=Math.max(peak,Math.abs(b[k][i]));}out[k]={relativeL2:Math.sqrt(s/(t||1)),relativeToPeak:max/(peak||1),maxAbsolute:max};}return out;}
function output(name,r){
 for(const k of keys)fs.writeFileSync(prefix+name+'-'+k+'.f64',Buffer.from(r[k].buffer,r[k].byteOffset,r[k].byteLength));
 for(const k of ['coherent','geometric','incoherent','crossTerm']){const rgb=Buffer.alloc(r.width*r.height*3);for(let i=0;i<r.width*r.height;i++){if(k==='crossTerm'){const c=r.crossTerm[i],a=Math.min(255,Math.round(255*Math.sqrt(Math.abs(c)*2**18)));rgb.set(c>=0?[a,0,0]:[0,0,a],3*i);}else rgb.set(displaySpectralXYZ(r[k+'XYZ'].subarray(i*3,i*3+3),18).rgb,i*3);}fs.writeFileSync(prefix+name+'-'+k+'.ppm',Buffer.concat([Buffer.from(`P6\n${r.width} ${r.height}\n255\n`),rgb]));}
 const entry={name,width:r.width,height:r.height,firstCompleteTileMs:r.firstCompleteTileMs,timings:r.timings,backend:r.backend,diagnostics:r.diagnostics,configuration:r.configuration,rawFormat:'Unmodified float64 little-endian row-major; XYZ interleaved.',display:'Fixed +18EV displaySpectralXYZ, no normalization or filtering. Cross term red-positive/blue-negative sqrt(abs(c)*2^18).'};report.cases.push(entry);save();console.log('CASE '+JSON.stringify(entry));
}
function mirrorMeasure(old,r){const {scene}=prepareCoherentMeshTask({...base,resolution:r.width});let primaryMirror=0,reflectedDiffuse=0,formerlyBlackNowLit=0,reflectedColored=0;const materials={};const mask=Buffer.alloc(r.width*r.height);for(let y=0;y<r.height;y++)for(let x=0;x<r.width;x++){const p=y*r.width+x,t=traceCoherentCamera(scene,camera,x,y,r.width,r.height);if(t.primaryMaterialType==='mirror'){primaryMirror++;if(t.status==='diffuse'){reflectedDiffuse++;mask[p]=t.materialIndex;materials[t.materialIndex]=(materials[t.materialIndex]??0)+1;if(t.materialIndex>=2)reflectedColored++;if(old.coherent[p]===0&&r.coherent[p]>0)formerlyBlackNowLit++;}}}fs.writeFileSync(prefix+'reflected-material-mask.u8',mask);return {primaryMirror,reflectedDiffuse,reflectedColored,formerlyBlackNowLit,materials};}
try{
 if(mode==='parity'||mode==='final-parity'){
  const reduced=structuredClone(preset);reduced.materials[0].reflectance=.37;const scene=sceneFromCoherentData(reduced);
  for(const [name,input] of [['mono',{}],['spectral',{spectrum}],['spectral-disk2',{spectrum,source:{...preset.source,spatial:{type:'disk',radiusMm:.06,normal:[0,0,1],samples:2,seed:7391}}}],['mirror-throughput037',{scene,spectrum}]]){const cpu=await run(name+'-cpu',input,'cpu'),native=await run(name+'-native',input);const e=errors(native,cpu);for(const k of keys)assert(e[k].relativeToPeak<1e-4,JSON.stringify({name,k,...e[k]}));report.parity.push({name,errors:e,cpuMs:cpu.timings.totalMs,nativeMs:native.timings.totalMs,diagnostics:native.diagnostics});save();console.log('PARITY '+name);}
 }else if(mode==='fast'){
  const old=await run('old512',{resolution:512,spectrum,camera:{...camera,mirrorBounces:0}});output('old512-q1-s16',old);
  const improved=await run('mirror512',{resolution:512,spectrum});output('mirror512-q1-s16',improved);report.mirrorEvidence=mirrorMeasure(old,improved);save();console.log('MIRROR_EVIDENCE '+JSON.stringify(report.mirrorEvidence));
 }else if(mode==='final-fast'){const r=await run('final-mirror512',{resolution:512,spectrum});output('final-mirror512-q1-s16',r);report.preFixComparison={};for(const k of keys){const old=fs.readFileSync(prefix+'mirror512-q1-s16-'+k+'.f64'),current=Buffer.from(r[k].buffer,r[k].byteOffset,r[k].byteLength);report.preFixComparison[k]={bitIdentical:old.equals(current)};assert(old.equals(current),'Grazing fix changed '+k);}save();}else if(mode==='quality')output('mirror512-q2-s32',await run('quality512',{resolution:512,spectrum,pixelQuadrature:2,spectralSamples:32}));
 else if(mode==='convergence'){
  const all={};for(const [q,s] of [[1,16],[1,32],[1,128],[2,32],[4,32]]){const name=`grid64-q${q}-s${s}`;all[`${q}-${s}`]=await run(name,{resolution:64,spectrum,pixelQuadrature:q,spectralSamples:s});output(name,all[`${q}-${s}`]);}
  const prepared=prepareCoherentMeshTask(base),coverage={};for(const q of [1,2,8]){const a=new Float64Array(64*64*5);for(let y=0;y<64;y++)for(let x=0;x<64;x++)for(let sy=0;sy<q;sy++)for(let sx=0;sx<q;sx++){const t=traceCoherentCamera(prepared.scene,camera,x,y,64,64,(sx+.5)/q,(sy+.5)/q);if(t.status==='diffuse')a[(y*64+x)*5+t.materialIndex]+=1/(q*q);}coverage[q]=a;}const coverageError=q=>{let s=0,t=0;for(let i=0;i<coverage[q].length;i++){s+=(coverage[q][i]-coverage[8][i])**2;t+=coverage[8][i]**2;}return Math.sqrt(s/t);};report.convergence={geometryCoverageAt64:{q1RelativeL2ToQ8:coverageError(1),q2RelativeL2ToQ8:coverageError(2),scope:'Per-material projected coverage including mirror continuation; independent of spectral sampling and wave shading; finite q8 coverage reference.'},spectralAtFixedQ1:{s16Vs128:errors(all['1-16'],all['1-128']),s32Vs128:errors(all['1-32'],all['1-128'])},spatialAtFixedS32:{q1Vs4:errors(all['1-32'],all['4-32']),q2Vs4:errors(all['2-32'],all['4-32'])},limitations:'Finite deterministic quadrature comparisons, not exact ground truth. Spatial samples also change spectral sampling positions, so spatial comparison includes spectral quadrature error. No blur/denoising.'};save();
 }else throw Error('Unknown mode '+mode);
 report.passed=true;
}catch(e){report.passed=false;report.error=e.stack;process.exitCode=1;console.error(e);}finally{report.completedAt=new Date().toISOString();save();}
