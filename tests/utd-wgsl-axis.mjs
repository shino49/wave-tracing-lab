// Native regression for exact-axis atan2 handling. The injected fault models an
// allowed unbounded native result at x=0; it is not a claim about every device.
// Set WEBGPU_EXECUTION_BACKEND (e.g. vulkan); driver selection is caller-owned.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {prepareUTDScene,builtInSceneOBJ,evaluatePoint,computeReceiverPoint} from '../dist/utd-scene.js';
import {packUTDSceneGPU} from '../dist/utd-gpu-pack.js';
import {parityGate} from '../dist/utd-engine.js';
const {create,globals}=process.env.WEBGPU_MODULE
  ? await import(pathToFileURL(path.resolve(process.env.WEBGPU_MODULE)).href)
  : await import('webgpu');
Object.assign(globalThis,globals);
const backend=process.env.WEBGPU_EXECUTION_BACKEND;
assert.ok(['vulkan','metal','d3d11','d3d12','opengl','opengles'].includes(backend),'Select an executable WEBGPU_EXECUTION_BACKEND; null is refused.');
let gpu=create([`backend=${backend}`]);
const adapter=await gpu.requestAdapter();assert.ok(adapter,'No executable adapter');
const provenance=Object.fromEntries(['vendor','architecture','device','description','isFallbackAdapter'].map(k=>[k,adapter.info[k]]));
const device=await adapter.requestDevice();
const functions=fs.readFileSync(new URL('../dist/utd-functions.wgsl',import.meta.url),'utf8');
const compute=fs.readFileSync(new URL('../dist/utd-compute.wgsl',import.meta.url),'utf8');
assert.equal((compute.match(/utd_atan2_axes\(/g)||[]).length,2,'Both production wedge-angle calls use the axis guard.');
const fault=`
fn utd_test_axis_fault(y:f32,x:f32)->f32 {
  if(x==0.0&&y!=0.0){return select(0.5*UTD_PI,-0.5*UTD_PI,y>0.0);}
  return atan2(y,x);
}`;
const inject=code=>code.replace(/\batan2\(/g,'utd_test_axis_fault(')+'\n'+fault;
const resources=[];
function buffer(bytes,usage,data){const b=device.createBuffer({size:Math.max(16,Math.ceil(bytes/16)*16),usage});resources.push(b);if(data)device.queue.writeBuffer(b,0,data);return b;}
async function pipeline(code,entryPoint='main'){
 const module=device.createShaderModule({code});const info=await module.getCompilationInfo();
 assert.equal(info.messages.filter(m=>m.type==='error').length,0,JSON.stringify(info.messages));
 return device.createComputePipelineAsync({layout:'auto',compute:{module,entryPoint}});
}
async function readback(out,bytes){const read=buffer(bytes,GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ),enc=device.createCommandEncoder();enc.copyBufferToBuffer(out,0,read,0,bytes);device.queue.submit([enc.finish()]);await read.mapAsync(GPUMapMode.READ);const data=new Float32Array(read.getMappedRange(0,bytes).slice(0));read.unmap();return data;}
const report={execution:'Native Dawn WGSL; adapter provenance is required',adapter:provenance,exactAxisCases:0,ordinaryCases:0,syntheticAxisFailures:0,fields:[]};
try {
 const cases=[];
 for(const zero of [0,-0])for(const sign of [1,-1]){cases.push([sign,zero]);cases.push([zero,sign]);}
 for(const zy of [0,-0])for(const zx of [0,-0])cases.push([zy,zx]);
 const exactCount=cases.length;
 for(const small of [1e-30,1e-12,1e-6,.1,.5,1])for(const sy of [-1,1])for(const sx of [-1,1]){cases.push([sy,sx*small]);cases.push([sy*small,sx]);}
 for(let i=0;i<1024;i++){const theta=-Math.PI+(i+.375)*2*Math.PI/1024;cases.push([Math.sin(theta),Math.cos(theta)]);}
 const inputs=new Float32Array(cases.flat()),input=buffer(inputs.byteLength,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST,inputs),bytes=cases.length*16;
 const unit=`
@group(0) @binding(0) var<storage,read> axisInput:array<vec2f>;
@group(0) @binding(1) var<storage,read_write> axisOutput:array<vec4f>;
@compute @workgroup_size(64) fn axisProbe(@builtin(global_invocation_id) id:vec3u) {
 if(id.x>=${cases.length}u){return;}let v=axisInput[id.x];
 axisOutput[id.x]=vec4f(utd_atan2_axes(v.x,v.y),atan2(v.x,v.y),v);
}`;
 for(const injected of [false,true]){
  const code=functions+'\n'+unit,p=await pipeline(injected?inject(code):code,'axisProbe'),out=buffer(bytes,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC);
  const group=device.createBindGroup({layout:p.getBindGroupLayout(0),entries:[input,out].map((b,binding)=>({binding,resource:{buffer:b}}))});
  const enc=device.createCommandEncoder(),pass=enc.beginComputePass();pass.setPipeline(p);pass.setBindGroup(0,group);pass.dispatchWorkgroups(Math.ceil(cases.length/64));pass.end();device.queue.submit([enc.finish()]);const data=await readback(out,bytes);
  let maxError=0;
  for(let i=0;i<cases.length;i++){
   const y=inputs[2*i],x=inputs[2*i+1],expected=x===0&&y===0?0:Math.atan2(y,x),got=data[4*i],error=Math.abs(got-expected);
   assert.ok(Number.isFinite(got));
   if(i<exactCount)assert.equal(got,Math.fround(expected),`Exact axis ${i}: y=${y}, x=${x}, injected=${injected}`);
   else assert.ok(error<.0016,`Non-axis quadrant ${i}: ${error}`);
   maxError=Math.max(maxError,error);
   if(injected&&x===0&&y!==0&&Math.abs(data[4*i+1]-expected)>3)report.syntheticAxisFailures++;
  }
  report[injected?'injectedAxisMaxError':'nativeAxisMaxError']=maxError;
 }
 report.exactAxisCases=exactCount;report.ordinaryCases=cases.length-exactCount;
 assert.equal(report.syntheticAxisFailures,4,'Negative control must expose all four injected vertical-axis failures.');
 const scene=prepareUTDScene(builtInSceneOBJ('screen'),{boundary:'soft',sourceOrigin:[0,0,-Math.hypot(2,20)]});
 const size=33,packed=packUTDSceneGPU(scene,size,size),uniform=buffer(packed.uniforms.byteLength,GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST,packed.uniforms);
 const store=a=>buffer(a.byteLength,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST,a);
 const tri=store(packed.triangles),edges=store(packed.edges),adj=store(packed.adjacency),facets=store(packed.facets),fieldBytes=size*size*16;
 for(const [label,code,shouldPass]of[
  ['Guarded native',functions+'\n'+compute,true],
  ['Guarded synthetic axis fault',inject(functions+'\n'+compute),true],
  ['Unguarded synthetic axis fault',inject(functions+'\n'+compute.replace(/utd_atan2_axes\(/g,'atan2(')),false],
 ]){
  const p=await pipeline(code),out=buffer(fieldBytes,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC),group=device.createBindGroup({layout:p.getBindGroupLayout(0),entries:[uniform,tri,edges,adj,out,facets].map((b,binding)=>({binding,resource:{buffer:b}}))});
  for(let row=0;row<size;row+=8){device.queue.writeBuffer(uniform,0,packed.createUniforms(row,Math.min(8,size-row)));const enc=device.createCommandEncoder(),pass=enc.beginComputePass();pass.setPipeline(p);pass.setBindGroup(0,group);pass.dispatchWorkgroups(Math.ceil(size/8),1);pass.end();device.queue.submit([enc.finish()]);await device.queue.onSubmittedWorkDone();}
  const data=await readback(out,fieldBytes),result={label,numericalFailures:0,maskMismatches:0,edgeMismatches:0,maxToleranceRatio:0,gatePassed:false};
  try{result.gate=parityGate(scene,{width:size,height:size,data});result.gatePassed=true;}catch(error){result.gateError=error.message;}
  for(let y=0;y<size;y++)for(let x=0;x<size;x++){
   const expected=evaluatePoint(scene,computeReceiverPoint(scene,x,y,size,size),{boundaryEpsilon:2e-6}),i=4*(y*size+x);
   if((data[i+2]>.5)!==expected.valid)result.maskMismatches++;if(data[i+3]!==expected.edgeCount)result.edgeMismatches++;
   if(expected.valid){const ratio=Math.abs(data[i]-expected.f)/(2e-5+.005*Math.abs(expected.f));result.maxToleranceRatio=Math.max(result.maxToleranceRatio,ratio);if(!(ratio<=1))result.numericalFailures++;}
  }
  assert.equal(result.gatePassed,shouldPass,`${label}: parity expectation`);
  if(shouldPass){assert.equal(result.numericalFailures,0);assert.equal(result.maskMismatches,0);assert.equal(result.edgeMismatches,0);}else assert.ok(result.numericalFailures>0,'Unguarded negative control must fail numerical comparison.');
  report.fields.push(result);console.log(`${label}: gate ${result.gatePassed?'PASS':'FAIL (expected negative control)'}; numerical failures ${result.numericalFailures}`);
 }
 report.passed=true;
 if(process.env.WEBGPU_AXIS_REPORT)fs.writeFileSync(process.env.WEBGPU_AXIS_REPORT,JSON.stringify(report,null,2)+'\n');
 console.log(JSON.stringify({...report,fields:report.fields.map(({gate,...r})=>r)},null,2));
} finally {for(const b of resources)b.destroy();device.destroy();gpu=null;}
