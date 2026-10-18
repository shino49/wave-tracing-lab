/** Timer-driven selective replay regression. The CPU replay is real; the GPU
 * lifecycle is mocked. This is not native shader or GPU performance evidence.
 */
import assert from 'node:assert/strict';
import {createPLTScene,createPLTRandomStates} from '../dist/plt-core.js';
import {createPLTResidentRunner,PLT_RESIDENT_LIMITS} from '../dist/plt-resident-gpu.js';

globalThis.GPUBufferUsage={MAP_READ:1,COPY_SRC:4,COPY_DST:8,UNIFORM:64,STORAGE:128};
globalThis.GPUMapMode={READ:1};
const laneWords=PLT_RESIDENT_LIMITS.laneBytes/4,ambiguous=2|2048,pixels=1024;

function mockPausedGPU(onResume){
 const buffers=[],reads=[],uniform=new Uint32Array(40),listeners=new Map();
 let initial,lanes,status=new Uint32Array(4),resumes=0,scopes=0,destroyed=0;
 const simulate=name=>{
  if(name==='rr_init'){
   lanes=new Uint32Array(pixels*laneWords);
   for(let lane=0;lane<pixels;lane++)lanes.set([initial[lane*4],initial[lane*4+1],uniform[8],0],lane*laneWords);
  }
  if(name!=='rr_scatter_step')return;
  for(let lane=uniform[12];lane<uniform[12]+uniform[13];lane++){
   const at=lane*laneWords;
   if(lanes[at+3]===0){
    // Only a paused-path certificate is needed: cancellation must happen
    // before the final path audit can be read or decoded.
    lanes[at+17]=lanes[at+1];lanes[at+2]++;lanes[at+3]=3;lanes[at+11]=ambiguous;
   }
   if(lanes[at+3]!==2)status[0]++;
   status[1]|=lanes[at+11];
  }
 };
 const device={
  limits:{maxBufferSize:1e9,maxStorageBufferBindingSize:1e9},lost:new Promise(()=>{}),
  addEventListener:(name,fn)=>listeners.set(name,fn),removeEventListener:name=>listeners.delete(name),
  pushErrorScope(){scopes++;},async popErrorScope(){scopes--;return null;},
  createShaderModule:()=>({getCompilationInfo:async()=>({messages:[]})}),
  createComputePipelineAsync:async({compute})=>({...compute,getBindGroupLayout:()=>({})}),
  createBindGroup:spec=>spec,
  createBuffer(spec){
   const buffer={...spec,destroyed:0,mapped:false,
    destroy(){this.destroyed++;this.mapped=false;},async mapAsync(){this.mapped=true;},unmap(){this.mapped=false;},
    getMappedRange(_offset,bytes){
     assert.ok(this.mapped&&!this.destroyed);reads.push(spec.label);
     const data=spec.label==='Resident status readback'?status.buffer:spec.label==='Resident lane readback'?lanes.buffer:null;
     assert.ok(data,`Unexpected final readback before cancellation: ${spec.label}`);
     assert.equal(data.byteLength,bytes);return data.slice(0);
    },
   };buffers.push(buffer);return buffer;
  },
  createCommandEncoder(){
   let name;return {
    beginComputePass(){return {setPipeline:p=>{name=p.entryPoint;},setBindGroup(){},dispatchWorkgroups(){},end(){}};},
    copyBufferToBuffer(){},finish:()=>name,
   };
  },
  queue:{
   writeBuffer(buffer,offset,data){
    const bytes=new Uint8Array(data.buffer??data,data.byteOffset??0,data.byteLength),words=new Uint32Array(bytes.slice().buffer);
    if(buffer.label==='Resident parameters')uniform.set(words,offset/4);
    if(buffer.label==='Resident initial prefixes')initial=words;
    if(buffer.label==='Resident status')status.set(words,offset/4);
    if(buffer.label==='Resident persistent lanes'){
     assert.equal(offset%PLT_RESIDENT_LIMITS.laneBytes,0);assert.equal(words.length,laneWords);
     lanes.set(words,offset/4);resumes++;onResume(resumes);
    }
   },
   submit:commands=>commands.forEach(simulate),onSubmittedWorkDone:async()=>{},
  },
  destroy(){destroyed++;},
 };
 return {
  gpu:{requestAdapter:async()=>({info:{description:'HOST REPLAY RESPONSIVENESS MOCK; NO GPU EXECUTION'},requestDevice:async()=>device})},
  reads,get resumes(){return resumes;},
  assertDisposed(){
   assert.equal(scopes,0);assert.equal(destroyed,1);assert.equal(listeners.size,0);
   for(const buffer of buffers){assert.equal(buffer.destroyed,1,buffer.label);assert.equal(buffer.mapped,false,buffer.label);}
  },
 };
}

const scene=createPLTScene({preset:'gallery'}),options={width:32,height:32,samplesPerPixel:1,maxDepth:6,wave:true};
const controller=new AbortController(),rngStates=createPLTRandomStates(options),original=rngStates.slice();
let scheduledAt,observed,abortTimer,runner;
const mock=mockPausedGPU(count=>{
 if(count!==1)return;
 scheduledAt=performance.now();
 // Calling abort synchronously here cannot detect a microtask-only replay
 // loop. The cancellation must arrive as a later event-loop task.
 abortTimer=setTimeout(()=>{observed={resumes:mock.resumes,elapsedMs:performance.now()-scheduledAt};controller.abort();},0);
});
try{
 runner=await createPLTResidentRunner(scene,scene.camera,options,{gpu:mock.gpu,shaderSource:'HOST MOCK ONLY; NOT WGSL',recoveryMode:'selective'});
 await assert.rejects(runner.run({tile:{x:0,y:0,width:32,height:32},rngStates,signal:controller.signal}),{name:'AbortError'});
 assert.ok(observed,'The scheduled cancellation must actually execute');
 assert.ok(observed.resumes>=1&&observed.resumes<pixels,`Timer starvation: ${observed.resumes}/${pixels} paths replayed before cancellation (${observed.elapsedMs.toFixed(1)} ms)`);
 assert.equal(mock.resumes,observed.resumes,'No further lane may resume after cancellation');
 assert.ok(!mock.reads.includes('Resident audit readback'),'Cancelled replay must not read a final audit or publish a candidate');
 assert.deepEqual(rngStates,original,'Cancelled replay must not mutate the caller RNG prefix');
 await assert.rejects(runner.run({tile:{x:0,y:0,width:32,height:32},rngStates}),{name:'AbortError'});
 console.log(JSON.stringify({allPass:true,hostMockOnly:true,actualGPUExecuted:false,pausedPaths:pixels,replayedBeforeTimer:observed.resumes,timerDelayMs:observed.elapsedMs}));
}finally{
 clearTimeout(abortTimer);runner?.destroy();mock.assertDisposed();
}
