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
import {compileCoherentSource,detectorPoint} from '../dist/coherent-mesh-core.js';
import {evaluateMeshTransport} from '../dist/coherent-mesh-transport.js';
Object.assign(globalThis,globals);
assert.equal(process.env.WEBGPU_EXECUTION_BACKEND,'vulkan');
const gpu=create(['backend=vulkan']),mode=process.argv[2]??'fast';
const spectralShader=fs.readFileSync('dist/utd-functions.wgsl','utf8')+'\n'+fs.readFileSync('dist/coherent-mesh-spectral-compute.wgsl','utf8'),monoShader=fs.readFileSync('dist/coherent-mesh-compute.wgsl','utf8');
const exposureEV=mode==='camera'?18:14;
const prefix='research/slab-mirror-',report={mode,scope:'Native Dawn Vulkan SwiftShader software execution. CPU geometry; native WebGPU integration. No browser, physical GPU, or CPU shadow.',exposureEV,parity:[],cases:[],hashes:{},startedAt:new Date().toISOString()};
for(const f of ['coherent-mesh-slab.js','coherent-mesh-spectral-templates.js','coherent-mesh-runtime.js','coherent-mesh-lighting-runtime.js','coherent-mesh-environment.js','coherent-mesh-camera.js','coherent-mesh-spectral-gpu.js','coherent-mesh-spectral-compute.wgsl'])report.hashes[f]=createHash('sha256').update(fs.readFileSync('dist/'+f)).digest('hex');
for(const f of fs.readdirSync('dist').filter(f=>/^coherent-mesh-.*\.(js|wgsl)$/.test(f)))report.hashes[f]=createHash('sha256').update(fs.readFileSync('dist/'+f)).digest('hex');
const save=()=>fs.writeFileSync(prefix+mode+'-report.json',JSON.stringify(report,null,2)+'\n');
const spectrum={mode:'broadband',illuminant:'D65'},base={measurement:'detector',maxReflections:1,diffraction:false,mixedDiffraction:false,resolution:16,pixelQuadrature:1,spectralSamples:32,spectralSeed:7391,maxRuntimeMs:600000};
const keys=['coherent','geometric','incoherent','crossTerm','coherentXYZ','geometricXYZ','incoherentXYZ'];
async function run(name,input={},backend='auto'){
 const start=performance.now();let firstCompleteTileMs=null,firstPreviewMs=null,firstLightingRowMs=null,last=-1;
 const r=await executeCoherentMeshTask({...base,...input,backend},{gpuFactory:o=>createCoherentMeshGPURunner({...o,gpu,shaderSource:monoShader}),spectralGpuFactory:o=>createCoherentMeshSpectralGPURunner({...o,gpu,shaderSource:spectralShader}),preview:p=>{if(firstPreviewMs===null)firstPreviewMs=performance.now()-start;if(p.rowCount===1&&p.lightingComplete===false&&firstLightingRowMs===null)firstLightingRowMs=performance.now()-start;if(p.spatialComplete&&firstCompleteTileMs===null){firstCompleteTileMs=performance.now()-start;console.log('FIRST_TILE '+name+' '+firstCompleteTileMs);}},progress:p=>{const step=Math.floor((p.fraction??0)*10);if(step!==last){last=step;console.log(name+' '+step*10+'% '+Math.round(performance.now()-start)+'ms');}}});
 assert(r.complete&&r.spatialComplete&&r.spectralComplete);assert.equal(r.diagnostics.productionCPUShadow,false);if(backend==='auto'){assert.equal(r.backend.kind,'hybrid');assert.equal(r.diagnostics.fallbackReason,null);}r.firstCompleteTileMs=firstCompleteTileMs;r.firstPreviewMs=firstPreviewMs;r.firstLightingRowMs=firstLightingRowMs;return r;
}
function errors(a,b){const out={};for(const k of keys){let s=0,t=0,max=0,peak=0;for(let i=0;i<a[k].length;i++){const d=a[k][i]-b[k][i];s+=d*d;t+=b[k][i]**2;max=Math.max(max,Math.abs(d));peak=Math.max(peak,Math.abs(b[k][i]));}out[k]={relativeL2:Math.sqrt(s/(t||1)),relativeToPeak:max/(peak||1),maxAbsolute:max};}return out;}
function output(name,r){
 for(const k of keys)fs.writeFileSync(prefix+name+'-'+k+'.f64',Buffer.from(r[k].buffer,r[k].byteOffset,r[k].byteLength));
 for(const k of ['coherent','geometric','incoherent','crossTerm']){const rgb=Buffer.alloc(r.width*r.height*3);for(let i=0;i<r.width*r.height;i++){if(k==='crossTerm'){const c=r.crossTerm[i],a=Math.min(255,Math.round(255*Math.sqrt(Math.abs(c)*2**18)));rgb.set(c>=0?[a,0,0]:[0,0,a],3*i);}else rgb.set(displaySpectralXYZ(r[k+'XYZ'].subarray(i*3,i*3+3),exposureEV).rgb,i*3);}fs.writeFileSync(prefix+name+'-'+k+'.ppm',Buffer.concat([Buffer.from(`P6\n${r.width} ${r.height}\n255\n`),rgb]));}
 const entry={name,width:r.width,height:r.height,firstPreviewMs:r.firstPreviewMs,firstLightingRowMs:r.firstLightingRowMs,firstCompleteTileMs:r.firstCompleteTileMs,timings:r.timings,backend:r.backend,diagnostics:r.diagnostics,configuration:r.configuration,rawFormat:'Unmodified float64 little-endian row-major; XYZ interleaved.',display:`Fixed +${exposureEV}EV displaySpectralXYZ, no normalization or filtering. Cross term red-positive/blue-negative sqrt(abs(c)*2^18).`};report.cases.push(entry);save();console.log('CASE '+JSON.stringify(entry));
}

const fixture=JSON.parse(fs.readFileSync(process.env.SLAB_MIRROR_FIXTURE??'dist/examples/coherent-slab-mirror.json','utf8'));
const input={scene:fixture,spectrum};
const before=structuredClone(fixture);before.source.position[1]=.1;before.detector.center[1]=.2;before.detector.height=.02;before.slab.center=[0,.125,25];before.slab.height=.04;
function coverage(data,n=16){
 const p=prepareCoherentMeshTask({...base,scene:data,spectrum:{mode:'monochrome'}}),compiled=compileCoherentSource(p.scene,p.source),counts={},examples={};
 for(let y=0;y<n;y++)for(let x=0;x<n;x++){
  const point=detectorPoint(p.detector,x,y,n,n,.5,.5),t=evaluateMeshTransport(p.scene,compiled,point,{diffraction:false});
  for(const r of t.routes){const key=r.interactionOrder??r.kind;counts[key]=(counts[key]??0)+1;if(!examples[key])examples[key]={vertices:r.vertices,glassLength:r.glassLength,excess:r.excess,amplitude:r.amplitude,phaseRadReduced:r.phaseRadReduced};}
 }
 return {counts,examples};
}
function comparison(a,b){let aa=0,bb=0,ab=0,diff=0,max=0;for(let i=0;i<a.length;i++){aa+=a[i]*a[i];bb+=b[i]*b[i];ab+=a[i]*b[i];diff+=(a[i]-b[i])**2;max=Math.max(max,Math.abs(a[i]-b[i]));}return {cosine:ab/Math.sqrt(aa*bb),relativeL2:Math.sqrt(diff/(bb||1)),maxAbsolute:max};}
try{
 report.coverage={after:coverage(fixture),before:coverage(before)};report.fixtureSha256=createHash('sha256').update(JSON.stringify(fixture)).digest('hex');
 assert((report.coverage.after.counts['R-TT']??0)+(report.coverage.after.counts['TT-R']??0)>0,'No mirror/glass routes');if(mode==='parity'){assert(report.coverage.after.counts['TT-R']>0,'No glass-then-mirror routes');assert(report.coverage.before.counts['R-TT']>0,'No mirror-then-glass routes');}
 if(mode==='parity'){
  for(const [name,scene] of [['after',fixture],['before',before]])for(const [band,sp] of [['mono',{mode:'monochrome'}],['D65',spectrum]]){
   const i={scene,spectrum:sp},cpu=await run(name+'-'+band+'-cpu',i,'cpu'),native=await run(name+'-'+band+'-native',i),e=errors(native,cpu);
   for(const k of keys)assert(e[k].maxAbsolute<1e-4*Math.max(...cpu.coherent,...cpu[k],1e-20),JSON.stringify({name,band,k,...e[k]}));
   assert(native.diagnostics.transmittedRoutes>0);report.parity.push({name,band,errors:e,cpuMs:cpu.timings.totalMs,nativeMs:native.timings.totalMs,diagnostics:native.diagnostics});output(name+'-'+band+'-native16',native);save();console.log('PARITY '+name+' '+band);
  }
 }else if(mode==='camera'){
  const glass=JSON.parse(fs.readFileSync('dist/examples/coherent-glass-mirror-imaging.json','utf8')),ci={scene:glass,measurement:'camera',camera:glass.camera,lighting:glass.lighting,maxReflections:1};
  report.cameraFixture=glass;
  for(const [band,sp] of [['mono',{mode:'monochrome'}],['D65',spectrum]]){
   const cpu=await run('camera-'+band+'-cpu',{...ci,spectrum:sp},'cpu'),native=await run('camera-'+band+'-native',{...ci,spectrum:sp}),e=errors(native,cpu);
   for(const k of keys)assert(e[k].maxAbsolute<1e-4*Math.max(...cpu.coherent,...cpu[k],1e-20),JSON.stringify({band,k,...e[k]}));
   assert(native.diagnostics.reflectedTransmittedRoutes>0,'Camera has no new mirror-glass routes');report.parity.push({name:'camera',band,errors:e,diagnostics:native.diagnostics});output('camera-'+band+'-native16',native);
  }
  output('camera-color64',await run('camera-color64',{...ci,spectrum,resolution:64}));
 }else if(mode==='pilot'){output('pilot-mono64',await run('pilot-mono64',{...input,spectrum:{mode:'monochrome'},resolution:64}));output('pilot-color64',await run('pilot-color64',{...input,resolution:64}));}else if(mode==='render'){
  const mono=await run('mono512',{...input,spectrum:{mode:'monochrome'},resolution:512});output('mono512',mono);
  const color=await run('color512',{...input,resolution:512});output('color512',color);
  const clear=structuredClone(fixture);delete clear.slab;const unclad=await run('clear-color512',{...input,scene:clear,resolution:512});output('clear-color512',unclad);
  const phase=structuredClone(fixture);phase.materials.find(m=>m.type==='mirror').phaseRad+=Math.PI;const shifted=await run('phase-pi512',{...input,scene:phase,spectrum:{mode:'monochrome'},resolution:512});output('phase-pi512',shifted);
  report.causal={phase:{crossTerm:comparison(mono.crossTerm,shifted.crossTerm),coherent:comparison(mono.coherent,shifted.coherent),incoherent:comparison(mono.incoherent,shifted.incoherent)},glass:{coherentXYZ:comparison(color.coherentXYZ,unclad.coherentXYZ),coherent:comparison(color.coherent,unclad.coherent)}};
  assert(report.causal.phase.crossTerm.cosine<-.99999);assert(report.causal.phase.incoherent.relativeL2<1e-5);assert(report.causal.glass.coherentXYZ.relativeL2>.01);
 }else throw Error('Unknown mode '+mode);
 report.passed=true;
}catch(e){report.passed=false;report.error=e.stack;process.exitCode=1;console.error(e);}finally{report.completedAt=new Date().toISOString();report.hashesAtEnd=Object.fromEntries(Object.keys(report.hashes).map(f=>[f,createHash('sha256').update(fs.readFileSync('dist/'+f)).digest('hex')]));report.changedDuringRun=Object.keys(report.hashes).filter(f=>report.hashes[f]!==report.hashesAtEnd[f]);save();}
