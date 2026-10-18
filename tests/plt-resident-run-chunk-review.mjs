/** Independent host scheduling/lifecycle review of per-run lane chunk overrides.
 * Host mocks only: no shader execution or numerical GPU correctness is claimed.
 */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import crypto from 'node:crypto';
import {createPLTRandomStates} from '../dist/plt-core.js';
import {preparePLTScene} from '../dist/plt-geometry.js';
import {createPLTResidentRunner,PLT_RESIDENT_LIMITS,PLT_RESIDENT_BUILD_ID} from '../dist/plt-resident-gpu.js';

globalThis.GPUBufferUsage={MAP_READ:1,COPY_SRC:4,COPY_DST:8,UNIFORM:64,STORAGE:128};
globalThis.GPUMapMode={READ:1};
const hash=value=>crypto.createHash('sha256').update(value).digest('hex');
const delay=(ms=0)=>new Promise(resolve=>setTimeout(resolve,ms));
const deferred=()=>{let resolve;const promise=new Promise(r=>{resolve=r;});return {promise,resolve};};
const scene=preparePLTScene([{mesh:{vertices:[[-8,-8,2],[8,-8,2],[8,8,2],[-8,8,2]],faces:[[0,1,2],[0,2,3]]},materialIndex:0}],{materials:[{type:'mirror',albedo:.5,emission:0}]});
const camera={position:[0,0,0],target:[0,0,2],up:[0,1,0],fov:20};
const options={width:32,height:32,samplesPerPixel:4,maxDepth:2,wave:false,nee:false};
const tile={x:3,y:2,width:17,height:17};
const sourceURL=new URL('../dist/plt-resident-gpu.js',import.meta.url);
const source=fs.readFileSync(sourceURL,'utf8');
const report={buildId:PLT_RESIDENT_BUILD_ID,allPass:false,hostMockOnly:true,actualGPUExecuted:false,fullCPUShadowRequired:true,sourceSHA256:hash(source),testSHA256:hash(fs.readFileSync(new URL(import.meta.url))),checks:[],mutations:[]};
const save=()=>fs.writeFileSync(new URL('../research/plt-resident-run-chunk-review-validation.json',import.meta.url),JSON.stringify(report,null,2)+'\n');
const aliases={rr_init_camera:'rr_init',rr_camera_init_step:'rr_anchor_init_step',rr_camera_prepare_step:'rr_axis_prepare_step',rr_camera_plane_step:'rr_axis_plane_step',rr_camera_point_step:'rr_anchor_point_step',rr_camera_finish_step:'rr_axis_finish_step'};
const logical=name=>aliases[name]??name;
function stageSequence(guard){
 const anchor=['axis-mirror-camera','axis-mirror-anchor'].includes(guard),axis=guard.startsWith('axis-');
 return ['rr_traverse_step',...(axis?[...(anchor?['rr_anchor_init_step']:[]),'rr_axis_prepare_step','rr_axis_plane_step','rr_axis_scan_step','rr_axis_plane_step','rr_axis_scan_step',...(anchor?['rr_anchor_point_step']:[]),'rr_axis_finish_step']:[]),'rr_records_step','rr_operator_step','rr_scatter_step'];
}
function expectedGroups(mode,guard){
 const sequence=stageSequence(guard);if(mode==='round')return [sequence];
 const axis=guard.startsWith('axis-'),anchor=['axis-mirror-camera','axis-mirror-anchor'].includes(guard);
 if(!axis)return sequence.map(name=>[name]);
 if(mode==='batched')return [[sequence[0]],sequence.slice(1,-3),...sequence.slice(-3).map(name=>[name])];
 return [['rr_traverse_step'],...(anchor?[['rr_anchor_init_step']]:[]),['rr_axis_prepare_step'],['rr_axis_plane_step','rr_axis_scan_step','rr_axis_plane_step','rr_axis_scan_step'],...(anchor?[['rr_anchor_point_step']]:[]),['rr_axis_finish_step'],['rr_records_step'],['rr_operator_step'],['rr_scatter_step']];
}
function mockGPU(config={}){
 const lost=deferred(),buffers=[],submissions=[],writes=[],events=[],listeners=new Map();
 let uniform=new Uint32Array(40),initial=new Uint32Array(4),outstanding=false,current=null,scopes=0,destroyed=0,readCount=0;
 const device={limits:{maxBufferSize:1e9,maxStorageBufferBindingSize:1e9},lost:lost.promise,
  addEventListener:(name,fn)=>listeners.set(name,fn),removeEventListener:name=>listeners.delete(name),
  pushErrorScope(){scopes++;},async popErrorScope(){assert(scopes>0);scopes--;return null;},
  createShaderModule:()=>({getCompilationInfo:async()=>({messages:[]})}),
  async createComputePipelineAsync({compute}){return {entryPoint:logical(compute.entryPoint),getBindGroupLayout:()=>({entryPoint:logical(compute.entryPoint)})};},
  createBindGroup(spec){return spec;},
  createBuffer(spec){const buffer={...spec,destroyed:0,mapped:false,destroy(){this.destroyed++;this.mapped=false;},async mapAsync(){assert.equal(destroyed,0);this.mapped=true;},unmap(){assert(this.mapped);this.mapped=false;},getMappedRange(_offset,bytes){
   assert(this.mapped&&!this.destroyed);readCount++;const data=new ArrayBuffer(bytes),u=new Uint32Array(data),f=new Float32Array(data),pixels=uniform[6]*uniform[7],samples=uniform[9];
   if(spec.label==='Resident lane readback')for(let lane=0;lane<pixels;lane++)u.set([initial[4*lane],initial[4*lane+1],uniform[8]+samples,2],lane*PLT_RESIDENT_LIMITS.laneBytes/4);
   if(spec.label==='Resident audit readback')for(let lane=0;lane<pixels;lane++)for(let sample=0;sample<samples;sample++){
    const at=(lane*samples+sample)*PLT_RESIDENT_LIMITS.auditVectors*4;
    u.set([initial[4*lane],uniform[8]+sample,uniform[11],0,initial[4*lane+1],initial[4*lane+1],0,2166136261,1,0,0,1],at);
    u[at+81]=0xffffffff;f[at+90]=-1;u[at+1545]=5;u[at+1546]=3;
   }return data;
  }};buffers.push(buffer);return buffer;},
  createCommandEncoder(){const passes=[];let open=null,copies=0;return {
   beginComputePass(){assert.equal(open,null,'No overlapping passes');const data={stage:null,group:null,count:0,ended:false};passes.push(data);open=data;return {
    setPipeline(p){data.stage=p.entryPoint;},setBindGroup(index,group){assert.equal(index,0);data.group=group;},
    dispatchWorkgroups(count){assert.equal(data.count,0,'Exactly one dispatch per preserved compute pass');assert.equal(data.group.layout.entryPoint,data.stage,'Pipeline-specific bind group');assert(Number.isInteger(count)&&count>0);data.count=count;},
    end(){assert.equal(open,data);assert(data.count);data.ended=true;open=null;},
   };},copyBufferToBuffer(){assert.equal(open,null);copies++;},finish(){assert.equal(open,null);assert(passes.every(p=>p.ended));return {passes,copies};},
  };},
  queue:{
   writeBuffer(buffer,offset,data){assert.equal(destroyed,0);assert(offset+data.byteLength<=buffer.size);const words=new Uint32Array(data.buffer??data,data.byteOffset??0,data.byteLength/4);writes.push({label:buffer.label,offset,words:[...words]});
    if(buffer.label==='Resident parameters'){assert(!outstanding,'Uniform changed before preceding chunk completed');uniform.set(words,offset/4);events.push({kind:'uniform',offset,words:[...words]});}
    if(buffer.label==='Resident initial prefixes')initial=words.slice();
   },
   submit(commandBuffers){assert.equal(destroyed,0);assert.equal(commandBuffers.length,1);for(const cb of commandBuffers){assert(!outstanding,'Submission before prior compute completion');if(cb.passes.length){outstanding=true;current={index:submissions.length,uniform:[...uniform],passes:cb.passes.map(p=>({stage:p.stage,count:p.count})),stageNames:cb.passes.map(p=>p.stage)};submissions.push(current);events.push({kind:'submit',index:current.index});config.onSubmit?.(current);}else{assert(cb.copies>0);events.push({kind:'readback'});}}},
   async onSubmittedWorkDone(){const captured=current;config.onWork?.(captured);if(config.holdIndex===captured.index)await config.hold;outstanding=false;events.push({kind:'done',index:captured.index});},
  },destroy(){destroyed++;},
 };
 return {gpu:{requestAdapter:async()=>({info:{description:'Independent host mock, no GPU execution'},requestDevice:async()=>device})},submissions,writes,events,
  get reads(){return readCount;},lose(){lost.resolve({message:'independent mock loss'});},
  assertDisposed(){assert.equal(scopes,0);assert.equal(destroyed,1);assert.equal(listeners.size,0);for(const b of buffers){assert.equal(b.destroyed,1,b.label);assert(!b.mapped,b.label);}},
 };
}
const fresh=extra=>({tile,sampleStart:0,samplesPerPixel:1,rngStates:createPLTRandomStates(options),axisSubmission:'round',...extra});
async function withRunner(config,extra,fn,factory=createPLTResidentRunner){
 const mock=mockGPU(config);let runner;
 try{runner=await factory(scene,camera,options,{gpu:mock.gpu,shaderSource:'HOST MOCK ONLY',timeoutMs:1000,recoveryMode:'selective',continuationGuard:'axis-mirror-camera',...extra});await fn(runner,mock);}
 finally{runner?.destroy();await delay();mock.assertDisposed();}
}
function verifySchedule(submissions,input,size,guard='axis-mirror-camera'){
 const pixels=input.tile.width*input.tile.height,mode=input.axisSubmission??'staged',groups=expectedGroups(mode,guard),chunks=Math.ceil(pixels/size);
 assert.equal(submissions.length,1+chunks*groups.length);assert.deepEqual(submissions[0].stageNames,['rr_init']);assert.equal(submissions[0].passes[0].count,Math.ceil(pixels/32));
 for(let chunk=0;chunk<chunks;chunk++)for(let group=0;group<groups.length;group++){
  const s=submissions[1+chunk*groups.length+group],count=Math.min(size,pixels-chunk*size);
  assert.deepEqual(s.uniform.slice(12,16),[chunk*size,count,PLT_RESIDENT_LIMITS.auditVectors,0]);
  assert.deepEqual(s.stageNames,groups[group]);assert(s.passes.every(p=>p.count===Math.ceil(count/32)));
 }
 return chunks*stageSequence(guard).length;
}
function verifyBarriers(events){let pending=null;for(const e of events){if(e.kind==='submit'){assert.equal(pending,null);pending=e.index;}else if(e.kind==='done'){assert.equal(pending,e.index);pending=null;}else if(e.kind==='uniform'||e.kind==='readback')assert.equal(pending,null);}assert.equal(pending,null);}
async function test(name,fn){await fn();report.checks.push(name);console.log('PASS independent host',name);}
async function runAndCheck(runner,mock,extra,size){
 const input=fresh(extra),before=input.rngStates.slice(),start=mock.submissions.length,result=await runner.run(input);
 assert(result.candidate&&!result.admitted);assert.equal(result.recovery.laneChunkSize,size);assert.equal(result.rounds,1);
 assert.equal(result.dispatches,verifySchedule(mock.submissions.slice(start),input,size,input.guard));assert.deepEqual(input.rngStates,before);verifyBarriers(mock.events);return result;
}
try{
 await test('per-run overrides reset to constructor default and preserve partial multi-workgroup chunks',()=>withRunner({}, {},async(runner,mock)=>{
  assert.equal(runner.diagnostic.dispatchLanes,32);let canonical;
  for(const size of [32,128,17,64,1,undefined,128]){const result=await runAndCheck(runner,mock,size===undefined?{}:{laneChunkSize:size},size??32);if(canonical){for(const key of ['audit','image','pixelIndices','finalRngStates'])assert.deepEqual(result[key],canonical[key]);}else canonical=result;assert.equal(runner.diagnostic.dispatchLanes,32);}
 }));
 await test('nonstandard constructor default is restored after explicit overrides',()=>withRunner({}, {dispatchLanes:17},async(runner,mock)=>{
  for(const [size,expected] of [[undefined,17],[128,128],[undefined,17],[32,32],[undefined,17]])await runAndCheck(runner,mock,size===undefined?{}:{laneChunkSize:size},expected);
  assert.equal(runner.diagnostic.dispatchLanes,17);
 }));
 await test('all established submission modes and guard choices retain exact dependencies',()=>withRunner({}, {},async(runner,mock)=>{
  for(const axisSubmission of ['staged','batched','round'])for(const guard of ['axis-mirror-camera','axis-mirror-anchor','axis-mirror-unit','axis-mirror-interval','replay-unmodelled','audit-only'])await runAndCheck(runner,mock,{laneChunkSize:128,axisSubmission,guard},128);
 }));
 await test('repeated sample windows and full tile preserve chunk boundaries',()=>withRunner({}, {},async(runner,mock)=>{
  for(const [sampleStart,samplesPerPixel,laneChunkSize] of [[0,4,128],[2,1,31],[1,3,127],[0,1,33]])await runAndCheck(runner,mock,{sampleStart,samplesPerPixel,laneChunkSize},laneChunkSize);
  await runAndCheck(runner,mock,{tile:{x:0,y:0,width:32,height:32},laneChunkSize:128},128);
 }));
 await test('invalid overrides fail before any batch write or command and dispose once',async()=>{
  for(const laneChunkSize of [0,-1,129,1.5,NaN,Infinity,-Infinity,'128',null,true,{},[]])await withRunner({}, {},async(runner,mock)=>{
   const beforeWrites=mock.writes.length,input=fresh({laneChunkSize}),before=input.rngStates.slice();await assert.rejects(runner.run(input),/chunk|dispatch/i);
   assert.equal(mock.writes.length,beforeWrites);assert.equal(mock.submissions.length,0);assert.equal(mock.reads,0);assert.deepEqual(input.rngStates,before);
  });
 });
 await test('128-lane cancellation after first round chunk blocks later chunks and readback',async()=>{
  const controller=new AbortController();await withRunner({onWork:s=>{if(s.index===1)controller.abort();}}, {},async(runner,mock)=>{
   const input=fresh({laneChunkSize:128,signal:controller.signal}),before=input.rngStates.slice();await assert.rejects(runner.run(input),{name:'AbortError'});
   assert.equal(mock.submissions.length,2);assert.equal(mock.reads,0);assert.deepEqual(input.rngStates,before);
  });
 });
 await test('128-lane queued cancellation cannot retract work but cannot return a candidate',async()=>{
  const controller=new AbortController();await withRunner({onSubmit:s=>{if(s.index===1)controller.abort();}}, {},async(runner,mock)=>{
   await assert.rejects(runner.run(fresh({laneChunkSize:128,signal:controller.signal})),{name:'AbortError'});assert.equal(mock.submissions.length,2);assert.equal(mock.reads,0);
  });
 });
 await test('128-lane in-flight loss disposal abort and timeout preserve resource ownership',async()=>{
  for(const kind of ['loss','dispose','abort','timeout']){const held=deferred(),entered=deferred(),controller=new AbortController();await withRunner({holdIndex:1,hold:held.promise,onWork:s=>{if(s.index===1)entered.resolve();}}, {timeoutMs:kind==='timeout'?30:1000},async(runner,mock)=>{
   const input=fresh({laneChunkSize:128,signal:controller.signal}),before=input.rngStates.slice(),pending=runner.run(input),rejected=assert.rejects(pending,kind==='loss'?/independent mock loss/:kind==='timeout'?/timeout/:{name:'AbortError'});
   await entered.promise;if(kind==='loss')mock.lose();if(kind==='dispose'){runner.destroy();runner.destroy();}if(kind==='abort')controller.abort();await rejected;held.resolve();await delay();assert.equal(mock.submissions.length,2);assert.equal(mock.reads,0);assert.deepEqual(input.rngStates,before);
  });}
 });
 await test('a concurrent run cannot override the active chunk or damage its completion',async()=>{
  const entered=deferred(),held=deferred();await withRunner({holdIndex:1,hold:held.promise,onWork:s=>{if(s.index===1)entered.resolve();}}, {},async(runner,mock)=>{
   const input=fresh({laneChunkSize:128}),pending=runner.run(input);await entered.promise;const count=mock.writes.length;await assert.rejects(runner.run(fresh({laneChunkSize:1})),/sequential/);assert.equal(mock.writes.length,count);held.resolve();const result=await pending;assert.equal(result.recovery.laneChunkSize,128);assert.equal(result.dispatches,verifySchedule(mock.submissions,input,128));verifyBarriers(mock.events);
  });
 });
 await test('source mutations are detected by independent range order and ownership oracles',async()=>{
  const roundStart=source.indexOf('    const dispatchRound=async count=>{'),roundEnd=source.indexOf('    let rounds=0,dispatches=0',roundStart);assert(roundStart>0&&roundEnd>roundStart);
  const alterRound=change=>source.slice(0,roundStart)+change(source.slice(roundStart,roundEnd))+source.slice(roundEnd);
  const mutations=[
   {name:'ignore-effective-override',changed:source.replace('first+=laneChunkSize','first+=dispatchLanes').replace('Math.min(laneChunkSize,pixels-first)','Math.min(dispatchLanes,pixels-first)'),extra:{laneChunkSize:128},size:128},
   {name:'unbounded-final-chunk',changed:source.replace('Math.min(laneChunkSize,pixels-first)','laneChunkSize'),extra:{laneChunkSize:128},size:128},
   {name:'overlapping-chunk-ranges',changed:source.replace('first+=laneChunkSize','first+=laneChunkSize-1'),extra:{laneChunkSize:128},size:128},
   {name:'gapped-chunk-ranges',changed:source.replace('first+=laneChunkSize','first+=laneChunkSize+1'),extra:{laneChunkSize:128},size:128},
   {name:'override-changes-omitted-default',changed:source.replace('laneChunkSize=dispatchLanes','laneChunkSize=128'),extra:{},size:32},
   {name:'metadata-reports-default',changed:source.replace('axisSubmission,laneChunkSize,paths:','axisSubmission,laneChunkSize:dispatchLanes,paths:'),extra:{laneChunkSize:128},size:128},
   {name:'omit-camera-point-dispatch',changed:alterRound(s=>s.replace("if(isAnchorGuard(guard))encode('rr_anchor_point_step');",'/* mutation: point omitted */')),extra:{laneChunkSize:128},size:128},
   {name:'scan-before-plane',changed:alterRound(s=>s.replace("encode('rr_axis_plane_step');encode('rr_axis_scan_step');","encode('rr_axis_scan_step');encode('rr_axis_plane_step');")),extra:{laneChunkSize:128},size:128},
   {name:'omit-chunk-completion-wait',changed:alterRound(s=>s.replace("await wait(device.queue.onSubmittedWorkDone(),'Resident lane-round timeout',signal);",'/* mutation: fence omitted */')),extra:{laneChunkSize:128},size:128},
  ];
  for(const mutation of mutations){assert.notEqual(mutation.changed,source,mutation.name+' must be effective');const absolute=mutation.changed.replace(/(['"])(\.\/[a-z0-9-]+\.js)\1/g,(_,quote,path)=>quote+new URL(path,sourceURL).href+quote),factory=(await import('data:text/javascript;base64,'+Buffer.from(absolute+'\n// '+mutation.name).toString('base64'))).createPLTResidentRunner;
   await withRunner({}, {},async(runner,mock)=>{await assert.rejects(runAndCheck(runner,mock,mutation.extra,mutation.size),{name:'AssertionError'},mutation.name);},factory);
   report.mutations.push({name:mutation.name,rejected:true,oracle:'actual encoded host ranges, dependency order, metadata or completion ownership'});
  }
 });
 assert.equal(hash(fs.readFileSync(sourceURL)),report.sourceSHA256,'Runtime source changed during independent host review');
 assert.equal(hash(fs.readFileSync(new URL(import.meta.url))),report.testSHA256,'Review test changed during execution');
 report.allPass=true;
}catch(error){report.failure={name:error.name,message:error.message,stack:error.stack};throw error;}
finally{save();}
console.log(JSON.stringify({allPass:report.allPass,checks:report.checks.length,mutations:report.mutations.length,hostMockOnly:true,actualGPUExecuted:false}));
