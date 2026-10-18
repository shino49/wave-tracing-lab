/** Axis-stage safety controls. Default: host lifecycle mocks only.
 * --execute: additionally dispatch the real records/operator/scatter kernels
 * against a pending-guard lane. No full axis graph or transport benchmark.
 */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import crypto from 'node:crypto';
import {createPLTRandomStates} from '../dist/plt-core.js';
import {preparePLTScene} from '../dist/plt-geometry.js';
import {packResidentScene} from '../dist/plt-resident-geometry.js';
import {createPLTResidentRunner,packResidentParameters,PLT_RESIDENT_LIMITS,PLT_RESIDENT_BUILD_ID} from '../dist/plt-resident-gpu.js';

globalThis.GPUBufferUsage={MAP_READ:1,COPY_SRC:4,COPY_DST:8,UNIFORM:64,STORAGE:128};
globalThis.GPUMapMode={READ:1};
const stages=['rr_init','rr_traverse_step','rr_axis_prepare_step','rr_axis_plane_step','rr_axis_scan_step','rr_axis_finish_step','rr_records_step','rr_operator_step','rr_scatter_step'];
const axisStages=stages.slice(2,6),axisBoundaries=['rr_axis_prepare_step','axis-triangle-loop','rr_axis_finish_step'];
function makeAxisScene(quads=1){
 const entries=Array.from({length:quads},(_,i)=>({mesh:{vertices:[[-8,-8,2+i/4],[8,-8,2+i/4],[8,8,2+i/4],[-8,8,2+i/4]],faces:[[0,1,2],[0,2,3]]},materialIndex:0}));
 const result=preparePLTScene(entries,{materials:[{type:'mirror',albedo:.5,emission:0}]});
 result.camera={position:[0,0,0],target:[0,0,2],up:[0,1,0],fov:20};return result;
}
const scene=makeAxisScene(),options={width:8,height:8,samplesPerPixel:4,maxDepth:2,wave:false};
const submissionNames=['rr_init','rr_traverse_step','rr_axis_prepare_step','axis-triangle-loop','rr_axis_finish_step','rr_records_step','rr_operator_step','rr_scatter_step'];
const submissionOrder=count=>submissionNames.map(name=>name==='axis-triangle-loop'?Array.from({length:count},()=>['rr_axis_plane_step','rr_axis_scan_step']).flat():[name]);
assert.equal(packResidentScene(scene).metadata.axisGuard.eligible,true,'Lifecycle fixture must exercise real triangle-loop scheduling');
const tile={x:0,y:0,width:1,height:1},fresh=()=>({tile,sampleStart:0,samplesPerPixel:1,rngStates:createPLTRandomStates(options)});
const report={buildId:PLT_RESIDENT_BUILD_ID,allPass:false,hostMockOnly:true,actualGPUExecuted:false,hostChecks:[],nativeChecks:[]};
async function test(name,fn){await fn();report.hostChecks.push(name);console.log('PASS',name);}

function mockGPU(config={}){
 const buffers=[],compiled=[],dispatched=[],submissions=[],uniform=new Uint32Array(40),listeners=new Map();
 let initial=new Uint32Array(4),scopes=0,destroyed=0,reads=0,currentStage;
 const device={limits:{maxBufferSize:1e9,maxStorageBufferBindingSize:1e9},lost:new Promise(()=>{}),
  addEventListener:(name,fn)=>listeners.set(name,fn),removeEventListener:name=>listeners.delete(name),
  pushErrorScope(){scopes++;},async popErrorScope(){scopes--;return null;},
  createShaderModule:()=>({getCompilationInfo:async()=>({messages:[]})}),
  async createComputePipelineAsync({compute}){
   compiled.push(compute.entryPoint);assert.equal(compute.constants.RR_AXIS_INTERVAL_ENABLED,config.baseline?0:1);
   if(config.pipelineFailure===compute.entryPoint)throw new Error('mock axis pipeline failure');
   return {...compute,getBindGroupLayout:()=>({})};
  },
  createBindGroup(spec){assert(spec.entries.every(entry=>entry.resource.buffer),'Missing stage buffer');return spec;},
  createBuffer(spec){
   if(config.bufferFailure===spec.label)throw new Error('mock axis state allocation failure');
   const buffer={...spec,destroyed:0,mapped:false,destroy(){this.destroyed++;this.mapped=false;},async mapAsync(){this.mapped=true;},unmap(){this.mapped=false;},
    getMappedRange(_offset,bytes){
     assert(this.mapped&&!this.destroyed);reads++;const data=new ArrayBuffer(bytes),u=new Uint32Array(data),f=new Float32Array(data);
     if(spec.label==='Resident lane readback')u.set([initial[0],initial[1],uniform[8]+uniform[9],2]);
     if(spec.label==='Resident audit readback')for(let sample=0;sample<uniform[9];sample++){
      const at=sample*PLT_RESIDENT_LIMITS.auditVectors*4;u.set([initial[0],uniform[8]+sample,uniform[11],0,initial[1],initial[1],0,2166136261,1,0,0,1],at);
      u[at+81]=0xffffffff;f[at+90]=-1;u[at+1546]=3;
     }
     return data;
    },
   };buffers.push(buffer);return buffer;
  },
  createCommandEncoder(){const commands=[];let copy=false;return {beginComputePass(){let stage;return {setPipeline:p=>{stage=p.entryPoint;},setBindGroup(){},dispatchWorkgroups(){commands.push(stage);},end(){}};},copyBufferToBuffer(){copy=true;},finish:()=>({commands,copy})};},
  queue:{
   writeBuffer(buffer,offset,data){
    assert.equal(destroyed,0,'Upload after destruction');assert(offset+data.byteLength<=buffer.size);
    const words=new Uint32Array(data.buffer??data,data.byteOffset??0,data.byteLength/4);
    if(buffer.label==='Resident parameters')uniform.set(words,offset/4);
    if(buffer.label==='Resident initial prefixes')initial=words.slice();
   },
   submit(commandBuffers){assert.equal(destroyed,0,'Submission after destruction');for(const {commands,copy} of commandBuffers)if(!copy){submissions.push(commands.slice());dispatched.push(...commands);currentStage=commands.length===1?commands[0]:'axis-triangle-loop';}},
   async onSubmittedWorkDone(){config.onStage?.(currentStage);if(config.stallStage===currentStage)await new Promise(()=>{});},
  },
  destroy(){destroyed++;},
 };
 return {gpu:{requestAdapter:async()=>({info:{description:'HOST AXIS LIFECYCLE MOCK; NO GPU EXECUTION'},requestDevice:async()=>device})},scene:config.sceneInput??scene,buffers,compiled,dispatched,submissions,
  get reads(){return reads;},assertDisposed(){assert.equal(scopes,0);assert.equal(destroyed,1);assert.equal(listeners.size,0);for(const b of buffers){assert.equal(b.destroyed,1,b.label);assert.equal(b.mapped,false,b.label);}},
 };
}
const create=(mock,extra={})=>createPLTResidentRunner(mock.scene,mock.scene.camera,options,{gpu:mock.gpu,shaderSource:'HOST MOCK ONLY; NOT WGSL',timeoutMs:1000,recoveryMode:'selective',continuationGuard:'axis-mirror-interval',...extra});
async function withRunner(config,fn,extra={}){const mock=mockGPU(config);let runner;try{runner=await create(mock,extra);await fn(runner,mock);}finally{runner?.destroy();mock.assertDisposed();}}

await test('nine pipelines preserve prepare, all plane/scan pairs, and finish ordering across reuse',()=>withRunner({},async(runner,mock)=>{
 const expected=[];
 for(const guard of ['axis-mirror-interval','replay-unmodelled','audit-only','axis-mirror-interval']){
  const axis=guard==='axis-mirror-interval',order=axis?submissionOrder(scene.triangles.length):stages.filter(name=>!name.startsWith('rr_axis_')).map(name=>[name]);
  const input={...fresh(),guard},before=input.rngStates.slice(),result=await runner.run(input);
  assert.equal(result.candidate,true);assert.equal(result.admitted,false);assert.equal(result.validation,undefined);assert.strictEqual(result.image,result.audit.reference);assert.deepEqual(input.rngStates,before);
  for(const key of ['gpuAxisPrepareSubmitAndWaitMs','gpuAxisTriangleLoopSubmitAndWaitMs','gpuAxisFinishSubmitAndWaitMs']){assert(Number.isFinite(result.timings[key])&&result.timings[key]>=0,key);if(!axis)assert.equal(result.timings[key],0,'Baseline timing must exclude every skipped guard pass');}
  assert.equal(result.dispatches,order.flat().length-1,'Dispatch accounting excludes initialization and includes every submitted triangle pass');
  expected.push(...order);assert.deepEqual(mock.submissions,expected);
 }
 assert.deepEqual(mock.compiled,stages);assert.deepEqual(mock.dispatched,expected.flat());
 const state=mock.buffers.filter(b=>b.label==='Resident staged GPU interval state');assert.equal(state.length,1);assert.equal(state[0].size,PLT_RESIDENT_LIMITS.tilePixels*256);
}));
await test('setup cancellation at all nine pipeline boundaries releases all ownership',async()=>{
 for(const stage of stages){
  const controller=new AbortController(),mock=mockGPU();
  await assert.rejects(create(mock,{signal:controller.signal,onSetupProgress:p=>{if(p.entryPoint===stage)controller.abort();}}),{name:'AbortError'});
  assert.deepEqual(mock.compiled,stages.slice(0,stages.indexOf(stage)));mock.assertDisposed();
 }
});
await test('cancellation after each submission suppresses later submissions and readback',async()=>{
 for(const stage of submissionNames){const controller=new AbortController();await withRunner({onStage:name=>{if(name===stage)controller.abort();}},async(runner,mock)=>{
  const input={...fresh(),signal:controller.signal},before=input.rngStates.slice();await assert.rejects(runner.run(input),{name:'AbortError'});
  assert.deepEqual(mock.submissions,submissionOrder(scene.triangles.length).slice(0,submissionNames.indexOf(stage)+1));assert.equal(mock.reads,0);assert.deepEqual(input.rngStates,before);
  await assert.rejects(runner.run(fresh()),{name:'AbortError'});
 });}
});
await test('timer cancellation after prepare, grouped triangle loop, and finish is observed',async()=>{
 for(const stage of axisBoundaries){const controller=new AbortController();await withRunner({onStage:name=>{if(name===stage)setTimeout(()=>controller.abort(),0);}},async(runner,mock)=>{
  await assert.rejects(runner.run({...fresh(),signal:controller.signal}),{name:'AbortError'});
  assert.deepEqual(mock.submissions,submissionOrder(scene.triangles.length).slice(0,submissionNames.indexOf(stage)+1));assert.equal(mock.reads,0);
 });}
});
await test('stalled prepare, grouped triangle loop, and finish time out and dispose state',async()=>{
 for(const stage of axisBoundaries)await withRunner({stallStage:stage},async(runner,mock)=>{await assert.rejects(runner.run(fresh()),/timeout/);assert.equal(mock.reads,0);},{timeoutMs:20});
});
await test('64-triangle support boundary submits all 128 dependent passes and larger scenes submit none',async()=>{
 for(const quads of [32,33]){const inputScene=makeAxisScene(quads),packed=packResidentScene(inputScene),supported=quads===32;
  assert.equal(packed.metadata.axisGuard.eligible,supported);
  await withRunner({sceneInput:inputScene},async(runner,mock)=>{await runner.run(fresh());assert.deepEqual(mock.submissions,submissionOrder(supported?64:0));});
 }
});
await test('axis pipeline and staged-state allocation failures release prior resources',async()=>{
 for(const config of [...axisStages.map(pipelineFailure=>({pipelineFailure})),{bufferFailure:'Resident staged GPU interval state'}]){
  const mock=mockGPU(config);await assert.rejects(create(mock),/mock axis/);mock.assertDisposed();
 }
});
await test('baseline runners cannot select an uncompiled axis guard and allocate no axis state',async()=>{
 const mock=mockGPU({baseline:true});let runner;
 try{runner=await create(mock,{continuationGuard:'replay-unmodelled'});assert.equal(mock.compiled.length,5);assert(!mock.buffers.some(b=>b.label==='Resident staged GPU interval state'));
  await assert.rejects(runner.run({...fresh(),guard:'axis-mirror-interval'}),/not compiled/);assert.equal(mock.dispatched.length,0);
 }finally{runner?.destroy();mock.assertDisposed();}
});

async function nativePendingLane(){
 const backend=process.env.WEBGPU_EXECUTION_BACKEND;
 assert(['vulkan','metal','d3d11','d3d12','opengl','opengles'].includes(backend),'--execute requires an explicit executable backend');
 const {create,globals}=await import('webgpu');Object.assign(globalThis,globals);let gpu=create([`backend=${backend}`]);
 const adapter=await gpu.requestAdapter();assert(adapter,'Native adapter unavailable');const device=await adapter.requestDevice(),buffers=[];
 report.backend=backend;report.physicalGPUValidated=false;report.hostMockOnly=false;
 const make=(data,usage)=>{const buffer=device.createBuffer({size:data.byteLength,usage});buffers.push(buffer);if(usage&GPUBufferUsage.COPY_DST)device.queue.writeBuffer(buffer,0,data);return buffer;};
 try{
  const names=['utd-functions.wgsl','plt-resident-geometry.wgsl','plt-resident-interval.wgsl','plt-resident-axis-guard.wgsl','plt-resident-rng.wgsl','plt-resident-fsd.wgsl','plt-resident-walker.wgsl'];
  const sources=names.map(name=>fs.readFileSync(new URL('../dist/'+name,import.meta.url),'utf8'));
  report.sourceSHA256=Object.fromEntries(names.map((name,i)=>[name,crypto.createHash('sha256').update(sources[i]).digest('hex')]));
  report.hostSHA256=Object.fromEntries(['plt-resident-gpu.js','plt-resident-geometry.js'].map(name=>[name,crypto.createHash('sha256').update(fs.readFileSync(new URL('../dist/'+name,import.meta.url))).digest('hex')]));
  const module=device.createShaderModule({code:sources.join('\n')});
  assert.deepEqual((await module.getCompilationInfo()).messages.filter(m=>m.type==='error'),[]);
  const packed=packResidentScene(scene),parameters=packResidentParameters(scene.camera,options,packed,fresh());
  new Uint32Array(parameters).set([0,1,PLT_RESIDENT_LIMITS.auditVectors,0],12);new Uint32Array(parameters)[37]=1;new Uint32Array(parameters)[38]=2;
  const laneData=new ArrayBuffer(PLT_RESIDENT_LIMITS.laneBytes),lane=new Uint32Array(laneData),lf=new Float32Array(laneData);
  lane.set([0,123,0,1],0);lane.set([0,0,1,0],4);lane.set([1,0,1,0],8);lane.set([2166136261,123,1,0],16);
  lane.set([1,1,0,0],52);lane.set([1,1,1,1],76);lf[26]=1;lf[34]=1;lf[44+2]=1;lf[43]=1;lf[47]=2;
  const auditData=new ArrayBuffer(PLT_RESIDENT_LIMITS.auditVectors*16);new Uint32Array(auditData)[1243]=0xffffffff;new Float32Array(auditData)[1237]=1;
  const recordData=new Uint32Array(16),weightData=new Uint32Array(192),statusData=new Uint32Array(4);
  const rw=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC;
  const uniform=make(parameters,GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST),geometry=make(packed.data,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST);
  const laneBuffer=make(laneData,rw),auditBuffer=make(auditData,rw),records=make(recordData,rw),weights=make(weightData,rw),status=make(statusData,rw);
  const immutable=[[laneBuffer,laneData,'lane'],[auditBuffer,auditData,'audit'],[records,recordData,'records'],[weights,weightData,'weights']];
  const groupEntries={rr_records_step:[[0,uniform],[1,geometry],[3,laneBuffer],[4,auditBuffer],[5,status],[6,records]],rr_operator_step:[[0,uniform],[3,laneBuffer],[4,auditBuffer],[6,records],[7,weights]],rr_scatter_step:[[0,uniform],[1,geometry],[3,laneBuffer],[4,auditBuffer],[5,status],[7,weights]]};
  device.pushErrorScope('validation');
  for(const name of ['rr_records_step','rr_operator_step','rr_scatter_step']){
   const pipeline=await device.createComputePipelineAsync({layout:'auto',compute:{module,entryPoint:name,constants:{RR_AXIS_INTERVAL_ENABLED:1}}});
   const group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:groupEntries[name].map(([binding,buffer])=>({binding,resource:{buffer}}))});
   const encoder=device.createCommandEncoder(),pass=encoder.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(1);pass.end();device.queue.submit([encoder.finish()]);await device.queue.onSubmittedWorkDone();
   const expectedStatus=new Uint32Array([name==='rr_scatter_step'?1:0,0,0,0]);
   for(const [buffer,expected,label] of [...immutable,[status,expectedStatus,'status']]){
    const readback=device.createBuffer({size:expected.byteLength,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});buffers.push(readback);
    const copy=device.createCommandEncoder();copy.copyBufferToBuffer(buffer,0,readback,0,expected.byteLength);device.queue.submit([copy.finish()]);await readback.mapAsync(GPUMapMode.READ);
    const actual=new Uint8Array(readback.getMappedRange()).slice();readback.unmap();const bytes=new Uint8Array(expected.buffer??expected,expected.byteOffset??0,expected.byteLength);
    assert.deepEqual(actual,bytes,`${name} modified pending ${label}`);
   }
   report.nativeChecks.push(`${name}: pending lane is read-only`);console.log('PASS native pending-lane guard',name);
  }
  assert.equal(await device.popErrorScope(),null);report.actualGPUExecuted=true;
 }finally{for(const buffer of buffers)buffer.destroy();device.destroy();gpu=null;}
}
if(process.argv.includes('--execute')){
 try{await nativePendingLane();report.allPass=true;}
 catch(error){report.failure={name:error.name,message:error.message};throw error;}
 finally{fs.writeFileSync(new URL('../research/plt-resident-axis-safety-validation.json',import.meta.url),JSON.stringify(report,null,2)+'\n');}
}else report.allPass=true;
console.log(JSON.stringify(report,null,2));
