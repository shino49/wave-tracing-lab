// Native diagnostic-entrypoint regression. Uses synthetic stress only; no user
// adapter data is embedded. An executable backend is mandatory; null is refused.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {prepareUTDScene,builtInSceneOBJ} from '../dist/utd-scene.js';
import {prepareMesh} from '../dist/mesh-geometry.js';
import {packUTDSceneGPU} from '../dist/utd-gpu-pack.js';
import {parityGate} from '../dist/utd-engine.js';
import {makeGPUEngine} from '../dist/utd-gpu-engine.js';
import {collectUTDProbe,selectUTDProbePixels} from '../dist/utd-gpu-diagnostics.js';
import {UTD_PROBE_STRIDE_VEC4,UTD_PROBE_ROWS} from '../dist/utd-probe-reference.js';
const backend=process.env.WEBGPU_EXECUTION_BACKEND;
assert.ok(['vulkan','metal','d3d11','d3d12','opengl','opengles'].includes(backend),'Select an executable WEBGPU_EXECUTION_BACKEND; null is refused');
const {create,globals}=process.env.WEBGPU_MODULE?await import(pathToFileURL(path.resolve(process.env.WEBGPU_MODULE)).href):await import('webgpu');
Object.assign(globalThis,globals);
let gpu=create([`backend=${backend}`]);
const adapter=await gpu.requestAdapter();assert.ok(adapter,'Executable adapter unavailable');
const provenance=Object.fromEntries(['vendor','architecture','device','description','isFallbackAdapter'].map(k=>[k,adapter.info[k]]));
const device=await adapter.requestDevice();
const originalFunctions=fs.readFileSync(new URL('../dist/utd-functions.wgsl',import.meta.url),'utf8');
const originalCompute=fs.readFileSync(new URL('../dist/utd-compute.wgsl',import.meta.url),'utf8');
const source=originalFunctions+'\n'+originalCompute;
const fingerprint=s=>{let h=2166136261;for(let i=0;i<s.length;i++){h^=s.charCodeAt(i);h=Math.imul(h,16777619);}return (h>>>0).toString(16).padStart(8,'0');};
const mesh=prepareMesh(builtInSceneOBJ('screen')),center=mesh.bounds.center;
const scene=prepareUTDScene(mesh,{boundary:'soft',sourceOrigin:[center[0],center[1],center[2]-Math.max(5,Math.hypot(...mesh.bounds.size))],detectorCenter:[center[0],center[1],center[2]+20]});
const width=33,height=33,packed=packUTDSceneGPU(scene,width,height),resources=[];
const make=(size,usage,data)=>{const b=device.createBuffer({size:Math.max(16,Math.ceil(size/16)*16),usage});resources.push(b);if(data)device.queue.writeBuffer(b,0,data);return b;};
const storage=a=>make(a.byteLength,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST,a);
const uniform=make(packed.uniforms.byteLength,GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST,packed.uniforms);
const out=make(width*height*16,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC);
const read=make(width*height*16,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST);
const buffers=[uniform,storage(packed.triangles),storage(packed.edges),storage(packed.adjacency),out,storage(packed.facets)];
async function readOriginal(){
 const enc=device.createCommandEncoder();enc.copyBufferToBuffer(out,0,read,0,width*height*16);device.queue.submit([enc.finish()]);
 await read.mapAsync(GPUMapMode.READ);const data=new Float32Array(read.getMappedRange().slice(0));read.unmap();return data;
}
async function execute(code){
 device.pushErrorScope('validation');
 const module=device.createShaderModule({code}),info=await module.getCompilationInfo();assert.equal(info.messages.filter(m=>m.type==='error').length,0);
 const pipeline=await device.createComputePipelineAsync({layout:'auto',compute:{module,entryPoint:'main'}});
 const group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:buffers.map((buffer,binding)=>({binding,resource:{buffer}}))});
 for(let base=0;base<height;base+=8){
  device.queue.writeBuffer(uniform,0,packed.createUniforms(base,Math.min(8,height-base)));
  const enc=device.createCommandEncoder(),pass=enc.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(Math.ceil(width/8),1);pass.end();device.queue.submit([enc.finish()]);await device.queue.onSubmittedWorkDone();
 }
 const data=await readOriginal(),error=await device.popErrorScope();assert.equal(error,null,error?.message);
 return {width,height,data};
}
const previousNavigator=Object.getOwnPropertyDescriptor(globalThis,'navigator'),previousFetch=globalThis.fetch;
const report={type:'Native optional-trace regression with synthetic negative control',adapter:provenance,mainShaderFingerprint:fingerprint(source)};
try{
 const result=await execute(source);assert.equal(parityGate(scene,result).passed,true);
 const pixels=[[8,8],[16,8],[0,8]],selection={samples:[
  {x:8,y:8,status:'numeric-mismatch',normalizedToleranceRatio:{f:2}},
  {x:16,y:8,status:'numeric-mismatch',normalizedToleranceRatio:{f:3}},
  {x:0,y:8,status:'pass',normalizedToleranceRatio:{f:0}},
 ]};
 assert.deepEqual(selectUTDProbePixels(selection,width,height),pixels);
 const capturedBindings=[];
 const monitoredDevice=new Proxy(device,{get(target,key){
  if(key==='createBindGroup')return desc=>{capturedBindings.push(desc.entries.map(e=>({binding:e.binding,buffer:e.resource.buffer})));return target.createBindGroup(desc);};
  const value=Reflect.get(target,key,target);return typeof value==='function'?value.bind(target):value;
 }});
 const lastBase=Math.floor((height-1)/8)*8,lastUniformBytes=packed.createUniforms(lastBase,height-lastBase);
 const trace=await collectUTDProbe({device:monitoredDevice,source,scene,packed,buffers,result,parity:selection,uniformBytes:lastUniformBytes});
 assert.deepEqual(trace.inputSnapshot.uniformsU32,[...new Uint32Array(lastUniformBytes.buffer,lastUniformBytes.byteOffset,8)]);
 assert.deepEqual(trace.inputSnapshot.uniformsU32.slice(0,4),[33,33,32,1]);
 assert.equal(trace.available,true);assert.equal(trace.allEdgesCaptured,true);assert.equal(trace.capturedEdges,2);
 assert.equal(UTD_PROBE_STRIDE_VEC4,34);assert.equal(UTD_PROBE_ROWS.length,34);assert.equal(trace.rowSchema.length,34);
 assert.deepEqual(trace.selectedPixels,pixels);assert.equal(trace.samples.length,6);
 assert.equal(capturedBindings.length,1);assert.deepEqual(capturedBindings[0].map(e=>e.binding),[0,1,2,3,4,5]);
 for(let i=0;i<6;i++)if(i!==4)assert.equal(capturedBindings[0][i].buffer,buffers[i],`Input binding ${i} must reuse main buffer`);
 assert.notEqual(capturedBindings[0][4].buffer,out,'Diagnostic output must use separate buffer');
 for(const sample of trace.samples){
  assert.equal(sample.gpu.rows.length,34);
  for(const row of sample.gpu.rows){assert.equal(Object.keys(row).length,4);assert.ok(Object.values(row).every(v=>typeof v==='number'&&Number.isFinite(v)),'Every typed trace scalar must be finite');}
  assert.equal(sample.cpuModel.referenceError,undefined);assert.equal(sample.cpuModel.rows.length,34);
  assert.ok(sample.cpuModel.rows.flat().every(v=>v===null||(typeof v==='number'&&Number.isFinite(v))));
  assert.equal(sample.cpuAtGPUInputs.referenceError,undefined);
  for(const key of ['coefficientAtGPUInputs','phaseAtGPUInputs','cisAtGPUInputs','anglesAtGPUInputs','coefficientFromGPUCots','edgeFromGPUCoefficientAndPhase','intrinsicsAtGPUInputs'])assert.ok(sample.cpuAtGPUInputs[key],`Missing CPU-at-GPU-input reference ${key}`);
  assert.equal(sample.cpuAtGPUInputs.branches.length,4);
  const finiteNumbers=value=>{if(typeof value==='number')return Number.isFinite(value);if(value&&typeof value==='object')return Object.values(value).every(finiteNumbers);return true;};
  assert.ok(finiteNumbers(sample.cpuAtGPUInputs),'CPU references at GPU inputs must contain finite numbers');
  assert.equal(sample.traceMatchesMainPower,true,'Default reconstructed trace power must match authoritative main output');
  assert.ok(sample.traceToMainToleranceRatio<=1);
  const [x,y]=sample.pixel,i=(y*width+x)*4;assert.deepEqual(sample.originalGPU,[...result.data.slice(i,i+4)]);
 }
 assert.deepEqual(await readOriginal(),result.data,'Collecting trace must not mutate original GPU output');
 report.baseline={passed:true,pixels,records:trace.samples.length,allSixBindings:true,finiteScalars:6*34*4,
   maximumTraceToMainRatio:Math.max(...trace.samples.map(s=>s.traceToMainToleranceRatio)),mainOutputUnchanged:true};
 // Restore only vulnerable small-angle paths in memory, then inject the same
 // synthetic sine error used by the independent builtin-stress regression.
 const start=originalFunctions.indexOf('fn utd_sinc('),open=originalFunctions.indexOf('{',start);let end=open+1,depth=1;
 for(;end<originalFunctions.length&&depth;end++){if(originalFunctions[end]==='{')depth++;else if(originalFunctions[end]==='}')depth--;}
 assert.ok(start>=0&&depth===0);
 let negative=originalFunctions.slice(0,start)+`fn utd_sinc(x:f32)->f32 { if(abs(x)<0.001){let q=x*x;return 1.0-q/6.0+q*q/120.0;}return sin(x)/x;}`+originalFunctions.slice(end);
 negative=negative.replace('utd_cos_small(h)','cos(h)').replace('abs(t*utd_sinc(t))','abs(sin(t))');
 negative=negative.replace(/\bsin\s*\(/g,'utd_probe_test_sin(')+'\nfn utd_probe_test_sin(x:f32)->f32 {return sin(x)+select(0.0,sign(x)*0.0001,abs(x)>0.000001&&abs(x)<0.1);}\n';
 const failedResult=await execute(negative+'\n'+originalCompute);let expectedError;
 try{parityGate(scene,failedResult);}catch(error){expectedError=error;}
 assert.ok(expectedError?.parity?.numericDisagreements>0,'Negative control must fail numerical parity');
 Object.defineProperty(globalThis,'navigator',{configurable:true,value:{gpu,userAgent:`Node ${process.version}; synthetic native diagnostic test`}});
 globalThis.fetch=async input=>{assert.ok(['./utd-functions.wgsl','./utd-compute.wgsl'].includes(String(input)));return {ok:true,status:200,text:async()=>String(input).endsWith('utd-functions.wgsl')?negative:originalCompute};};
 const engine=await makeGPUEngine(scene,width,height);let actualError;
 try{await engine.run();}catch(error){actualError=error;}finally{engine.destroy();}
 assert.ok(actualError,'Actual engine must retain original failure');
 assert.equal(actualError.message,expectedError.message);assert.equal(actualError.diagnostic.stage,'runtime-parity');
 assert.equal(actualError.diagnostic.parity.samples.length,25);
 assert.deepEqual(actualError.diagnostic.parity,expectedError.parity,'Tracing must preserve all original failed GPU samples and parity results');
 assert.equal(actualError.diagnostic.intermediateTrace.available,true,'Synthetic failure must collect real native trace');
 const failureTrace=actualError.diagnostic.intermediateTrace;
 assert.deepEqual(failureTrace.inputSnapshot.uniformsU32.slice(0,4),[33,33,32,1],'Failure snapshot must retain actual final-batch uniforms');
 assert.equal(failureTrace.samples.length,6);
 for(const s of failureTrace.samples){const [x,y]=s.pixel,i=(y*width+x)*4;assert.deepEqual(s.originalGPU,[...failedResult.data.slice(i,i+4)],'Failure trace must retain authoritative original GPU values');}
 // Do not assert reconstruction agreement in this negative fixture: the custom
 // source mutation and the probe's explicitly expanded arithmetic can differ.
 report.syntheticFailure={passed:true,originalFailurePreserved:true,originalSamples:25,traceAvailable:true,
   selectedPixels:failureTrace.selectedPixels,records:failureTrace.samples.length,
   traceToMainRatios:failureTrace.samples.map(s=>s.traceToMainToleranceRatio),
   note:'Synthetic vulnerable-source stress, not a reproduction of the user adapter. No reconstruction-agreement claim is made for this altered path.'};
 assert.equal(fs.readFileSync(new URL('../dist/utd-functions.wgsl',import.meta.url),'utf8'),originalFunctions);
 assert.equal(fs.readFileSync(new URL('../dist/utd-compute.wgsl',import.meta.url),'utf8'),originalCompute);
 report.passed=true;
 if(process.env.UTD_PROBE_REPORT)fs.writeFileSync(process.env.UTD_PROBE_REPORT,JSON.stringify(report,null,2)+'\n');
 console.log(JSON.stringify(report,null,2));
}finally{
 globalThis.fetch=previousFetch;if(previousNavigator)Object.defineProperty(globalThis,'navigator',previousNavigator);else delete globalThis.navigator;
 for(const b of resources){try{b.destroy();}catch{}}device.destroy();gpu=null;
}
