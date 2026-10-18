/** Focused camera-unit provenance, pending-guard, and submission lifecycle safety.
 * Default: host lifecycle mocks plus genuine Dawn/Tint null-backend compilation.
 * --host-only: mocks only. --execute: actual selected native backend dispatch.
 * WEBGPU_EXECUTION_BACKEND=vulkan VK_ICD_FILENAMES=/usr/lib/chromium/vk_swiftshader_icd.json
 * No physical-GPU/browser validation or performance claim is made.
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
const execute=process.argv.includes('--execute'),hostOnly=process.argv.includes('--host-only');
assert(!(execute&&hostOnly),'Choose host-only or native execution');
const stages=['rr_init','rr_traverse_step','rr_axis_prepare_step','rr_axis_plane_step','rr_axis_scan_step','rr_axis_finish_step','rr_records_step','rr_operator_step','rr_scatter_step'];
const baseStages=stages.filter(name=>!name.startsWith('rr_axis_'));
const delay=(ms=0)=>new Promise(resolve=>setTimeout(resolve,ms));
const deferred=()=>{let resolve;const promise=new Promise(r=>resolve=r);return {promise,resolve};};
const never=()=>new Promise(()=>{});
function makeScene(quads=1){
 const scene=preparePLTScene(Array.from({length:quads},(_,i)=>({mesh:{vertices:[[-8,-8,2+i/4],[8,-8,2+i/4],[8,8,2+i/4],[-8,8,2+i/4]],faces:[[0,1,2],[0,2,3]]},materialIndex:0})),{materials:[{type:'mirror',albedo:.5,emission:0}]});
 scene.camera={position:[0,0,0],target:[0,0,2],up:[0,1,0],fov:20};return scene;
}
const scene=makeScene(),options={width:8,height:8,samplesPerPixel:4,maxDepth:2,wave:false,nee:false};
const tile={x:0,y:0,width:1,height:1},fresh=()=>({tile,sampleStart:0,samplesPerPixel:1,rngStates:createPLTRandomStates(options)});
const pairs=n=>Array.from({length:n},()=>['rr_axis_plane_step','rr_axis_scan_step']).flat();
const groupName=commands=>commands.length===1?commands[0]:commands[0]==='rr_axis_prepare_step'?'axis-guard-group':'axis-triangle-loop';
function expectedSubmissions(mode,n,axis=true){
 const guard=mode==='batched'?[['rr_axis_prepare_step',...pairs(n),'rr_axis_finish_step']]:[['rr_axis_prepare_step'],pairs(n),['rr_axis_finish_step']];
 return [['rr_init'],['rr_traverse_step'],...(axis?guard:[]),['rr_records_step'],['rr_operator_step'],['rr_scatter_step']];
}
const report={createdAt:new Date().toISOString(),buildId:PLT_RESIDENT_BUILD_ID,executionRequested:execute,actualGPUExecuted:false,allPass:false,status:'running',hostMockOnly:hostOnly,fullCPUShadowMandatory:true,physicalGPUValidated:false,browserValidated:false,hardwarePerformanceClaim:false,hostChecks:[],nativeChecks:[],mutations:[]};
const reportURL=new URL('../research/plt-resident-unit-safety'+(execute?'':'-compile')+'-validation.json',import.meta.url);
const save=()=>fs.writeFileSync(reportURL,JSON.stringify(report,null,2)+'\n');
async function test(name,fn){await fn();report.hostChecks.push(name);console.log('PASS host',name);}

function mockGPU(config={}){
 const buffers=[],compiled=[],submissions=[],parameterSnapshots=[],events=[],listeners=new Map(),lost=deferred();
 let initial=new Uint32Array(4),uniform=new Uint32Array(40),scopes=0,destroyed=0,reads=0,currentStage,encoders=0,passCount=0;
 const device={limits:{maxBufferSize:1e9,maxStorageBufferBindingSize:1e9},lost:lost.promise,
  addEventListener:(name,fn)=>listeners.set(name,fn),removeEventListener:name=>listeners.delete(name),
  pushErrorScope(){scopes++;},async popErrorScope(){assert(scopes>0);scopes--;return null;},
  createShaderModule:()=>({getCompilationInfo:async()=>({messages:[]})}),
  async createComputePipelineAsync({compute}){compiled.push(compute.entryPoint);assert.equal(compute.constants.RR_AXIS_INTERVAL_ENABLED,config.baseline?0:1);if(config.pipelineFailure===compute.entryPoint)throw new Error('mock pipeline failure');return {...compute,getBindGroupLayout:()=>({})};},
  createBindGroup(spec){assert(spec.entries.every(e=>e.resource.buffer));return spec;},
  createBuffer(spec){if(config.bufferFailure===spec.label)throw new Error('mock allocation failure');const buffer={...spec,destroyed:0,mapped:false,destroy(){this.destroyed++;this.mapped=false;},async mapAsync(){assert.equal(destroyed,0);this.mapped=true;},unmap(){this.mapped=false;},getMappedRange(_offset,bytes){assert(this.mapped&&!this.destroyed);reads++;const data=new ArrayBuffer(bytes),u=new Uint32Array(data),f=new Float32Array(data),pixels=uniform[6]*uniform[7],samples=uniform[9];
   if(spec.label==='Resident lane readback')for(let lane=0;lane<pixels;lane++)u.set([initial[lane*4],initial[lane*4+1],uniform[8]+samples,2],lane*PLT_RESIDENT_LIMITS.laneBytes/4);
   if(spec.label==='Resident audit readback')for(let lane=0;lane<pixels;lane++)for(let sample=0;sample<samples;sample++){const at=(lane*samples+sample)*PLT_RESIDENT_LIMITS.auditVectors*4;u.set([initial[lane*4],uniform[8]+sample,uniform[11],0,initial[lane*4+1],initial[lane*4+1],0,2166136261,1,0,0,1],at);u[at+81]=0xffffffff;f[at+90]=-1;u[at+1546]=3;}
   return data;
  }};buffers.push(buffer);return buffer;},
  createCommandEncoder(){const id=++encoders,commands=[],passes=[];let open=false,copy=false;return {beginComputePass(){assert(!open,'Dependent dispatches require closed preceding passes');open=true;let stage,dispatched=false;const pass={id:++passCount,stage:null,ended:false};passes.push(pass);return {setPipeline:p=>{stage=p.entryPoint;},setBindGroup(){},dispatchWorkgroups(count){assert(count>=1);assert(!dispatched,'One dependency stage per compute pass');dispatched=true;commands.push(stage);pass.stage=stage;},end(){assert(open&&dispatched);pass.ended=true;open=false;}};},copyBufferToBuffer(){assert(!open);copy=true;},finish(){assert(!open);assert(passes.every(p=>p.ended));return {id,commands,passes,copy};}};},
  queue:{writeBuffer(buffer,offset,data){assert.equal(destroyed,0,'Upload after destruction');assert(offset+data.byteLength<=buffer.size);const words=new Uint32Array(data.buffer??data,data.byteOffset??0,data.byteLength/4);if(buffer.label==='Resident parameters'){uniform.set(words,offset/4);if(offset===0)parameterSnapshots.push(uniform.slice());}if(buffer.label==='Resident initial prefixes')initial=words.slice();},
   submit(commandBuffers){assert.equal(destroyed,0,'Submission after destruction');assert.equal(commandBuffers.length,1,'Each group must be one command buffer');for(const cb of commandBuffers)if(!cb.copy){submissions.push(cb.commands.slice());currentStage=groupName(cb.commands);events.push({kind:'submit',name:currentStage,id:cb.id,passes:cb.passes.map(p=>p.stage)});config.onSubmit?.(currentStage);}else events.push({kind:'readback'});},
   async onSubmittedWorkDone(){const stage=currentStage;events.push({kind:'wait',name:stage});config.onWork?.(stage);if(config.stallStage===stage)await never();if(config.holdStage===stage)await config.hold;events.push({kind:'done',name:stage});},
  },destroy(){destroyed++;},
 };
 return {gpu:{requestAdapter:async()=>({info:{description:'HOST UNIT SAFETY MOCK; NO GPU EXECUTION'},requestDevice:async()=>device})},scene:config.sceneInput??scene,buffers,compiled,submissions,events,parameterSnapshots,
  get reads(){return reads;},lose:()=>lost.resolve({message:'mock device lost'}),assertDisposed(){assert.equal(scopes,0);assert.equal(destroyed,1);assert.equal(listeners.size,0);for(const b of buffers){assert.equal(b.destroyed,1,b.label);assert.equal(b.mapped,false,b.label);}},
 };
}
const create=(mock,extra={},factory=createPLTResidentRunner)=>factory(mock.scene,mock.scene.camera,options,{gpu:mock.gpu,shaderSource:'HOST MOCK ONLY; NOT WGSL',timeoutMs:1000,recoveryMode:'selective',continuationGuard:'axis-mirror-unit',...extra});
async function withRunner(config,fn,extra={},factory){const mock=mockGPU(config);let runner;try{runner=await create(mock,extra,factory);await fn(runner,mock);}finally{runner?.destroy();await delay();mock.assertDisposed();}}
function assertBarrierEvents(mock){
 let outstanding=false;
 for(const event of mock.events){if(event.kind==='submit'){assert(!outstanding,'Later submit before preceding completion');outstanding=true;}else if(event.kind==='done'){assert(outstanding);outstanding=false;}else if(event.kind==='readback')assert(!outstanding,'Readback before group completion');}
 assert(!outstanding);
}
function assertCandidate(result,input,before){assert.equal(result.candidate,true);assert.equal(result.admitted,false);assert.equal(result.validation,undefined);assert.strictEqual(result.image,result.audit.reference);assert.deepEqual(input.rngStates,before,'Failed or unadmitted candidate cannot mutate input RNG');}

try{
await test('staged default and batched reuse identical nine pipelines with exact dependent pass barriers',()=>withRunner({},async(runner,mock)=>{
 const cases=[['axis-mirror-unit',undefined,4],['axis-mirror-unit','batched',4],['axis-mirror-interval','batched',2],['axis-mirror-unit','staged',4],['replay-unmodelled','batched',1],['audit-only','staged',0],['axis-mirror-unit','batched',4]];
 const expected=[];
 for(const [guard,axisSubmission,config] of cases){const input={...fresh(),guard,...(axisSubmission?{axisSubmission}:{})},before=input.rngStates.slice(),result=await runner.run(input),mode=axisSubmission??'staged',axis=guard.startsWith('axis-');assertCandidate(result,input,before);const order=expectedSubmissions(mode,scene.triangles.length,axis);expected.push(...order);assert.deepEqual(mock.submissions,expected);assert.equal(result.dispatches,order.flat().length-1);assert.equal(result.recovery.axisSubmission,mode);assert.equal(mock.parameterSnapshots.at(-1)[38],config);
  for(const key of ['gpuAxisPrepareSubmitAndWaitMs','gpuAxisTriangleLoopSubmitAndWaitMs','gpuAxisFinishSubmitAndWaitMs','gpuAxisGuardSubmitAndWaitMs']){assert(Number.isFinite(result.timings[key])&&result.timings[key]>=0);if(!axis||(mode==='batched'&&key!=='gpuAxisGuardSubmitAndWaitMs')||(mode==='staged'&&key==='gpuAxisGuardSubmitAndWaitMs'))assert.equal(result.timings[key],0,key);}
 }
 assert.deepEqual(mock.compiled,stages);assertBarrierEvents(mock);const state=mock.buffers.filter(b=>b.label==='Resident staged GPU interval state');assert.equal(state.length,1);assert.equal(state[0].size,PLT_RESIDENT_LIMITS.tilePixels*256);
}));
await test('both modes bound 64 triangle pairs exactly and unsupported scenes cannot encode any pair',async()=>{
 for(const quads of [1,32,33])for(const mode of ['staged','batched']){const sceneInput=makeScene(quads),eligible=quads<=32;assert.equal(packResidentScene(sceneInput).metadata.axisGuard.eligible,eligible);await withRunner({sceneInput},async(runner,mock)=>{const result=await runner.run({...fresh(),axisSubmission:mode});const order=expectedSubmissions(mode,eligible?2*quads:0);assert.deepEqual(mock.submissions,order);assert.equal(result.dispatches,order.flat().length-1);assertBarrierEvents(mock);});}
});
await test('each bounded lane group completes guard before transport and before next group',()=>withRunner({},async(runner,mock)=>{
 for(const mode of ['staged','batched']){const input={...fresh(),tile:{x:0,y:0,width:8,height:8},axisSubmission:mode},before=input.rngStates.slice(),start=mock.submissions.length,result=await runner.run(input);assertCandidate(result,input,before);const one=expectedSubmissions(mode,2).slice(1);assert.deepEqual(mock.submissions.slice(start),[['rr_init'],...one,...one]);assert.equal(result.dispatches,2*one.flat().length);}
 assertBarrierEvents(mock);
}));
await test('synchronous cancellation at all boundaries and timer cancellation at yielded boundaries blocks later work',async()=>{
 for(const mode of ['staged','batched']){const order=expectedSubmissions(mode,2);for(const [index,commands] of order.entries())for(const timing of index===0?['sync']:['sync','timer']){const controller=new AbortController(),stage=groupName(commands);await withRunner({onWork:name=>{if(name===stage){if(timing==='sync')controller.abort();else setTimeout(()=>controller.abort(),0);}}},async(runner,mock)=>{const input={...fresh(),axisSubmission:mode,signal:controller.signal},before=input.rngStates.slice();await assert.rejects(runner.run(input),{name:'AbortError'});assert.deepEqual(mock.submissions,order.slice(0,index+1));assert.equal(mock.reads,0);assert.deepEqual(input.rngStates,before);await assert.rejects(runner.run(fresh()),{name:'AbortError'});});}}
});
await test('cancellation after queued batched group may not retract its passes but blocks transport and candidate',async()=>{
 const controller=new AbortController();await withRunner({onSubmit:name=>{if(name==='axis-guard-group')controller.abort();}},async(runner,mock)=>{const input={...fresh(),axisSubmission:'batched',signal:controller.signal},before=input.rngStates.slice();await assert.rejects(runner.run(input),{name:'AbortError'});assert.deepEqual(mock.submissions,expectedSubmissions('batched',2).slice(0,3));assert.equal(mock.reads,0);assert.deepEqual(input.rngStates,before);});
});
await test('in-flight group cancellation, device loss, explicit disposal, and timeout release all ownership once',async()=>{
 for(const mode of ['staged','batched'])for(const method of ['cancel','lost','destroy','timeout']){const entered=deferred(),held=deferred(),controller=new AbortController(),stage=mode==='batched'?'axis-guard-group':'axis-triangle-loop';await withRunner({holdStage:stage,hold:held.promise,onWork:name=>{if(name===stage)entered.resolve();}},async(runner,mock)=>{const input={...fresh(),axisSubmission:mode,signal:controller.signal},before=input.rngStates.slice(),pending=runner.run(input);const rejection=assert.rejects(pending,method==='lost'?/mock device lost/:method==='timeout'?/timeout/:{name:'AbortError'});await entered.promise;if(method==='cancel')controller.abort();if(method==='lost')mock.lose();if(method==='destroy'){runner.destroy();runner.destroy();}await rejection;held.resolve();await delay();const order=expectedSubmissions(mode,2),index=order.findIndex(c=>groupName(c)===stage);assert.deepEqual(mock.submissions,order.slice(0,index+1));assert.equal(mock.reads,0);assert.deepEqual(input.rngStates,before);},{timeoutMs:method==='timeout'?30:1000});}
});
await test('staged unit prepare and finish timeouts withhold readback and release state',async()=>{
 for(const stage of ['rr_axis_prepare_step','rr_axis_finish_step'])await withRunner({stallStage:stage},async(runner,mock)=>{await assert.rejects(runner.run(fresh()),/timeout/);const order=expectedSubmissions('staged',2),index=order.findIndex(c=>groupName(c)===stage);assert.deepEqual(mock.submissions,order.slice(0,index+1));assert.equal(mock.reads,0);},{timeoutMs:30});
});
await test('unit setup cancellation and pipeline/state failures release exactly once',async()=>{
 for(const stage of stages){const controller=new AbortController(),mock=mockGPU();await assert.rejects(create(mock,{signal:controller.signal,onSetupProgress:p=>{if(p.entryPoint===stage)controller.abort();}}),{name:'AbortError'});assert.deepEqual(mock.compiled,stages.slice(0,stages.indexOf(stage)));mock.assertDisposed();}
 for(const config of [...stages.slice(2,6).map(pipelineFailure=>({pipelineFailure})),{bufferFailure:'Resident staged GPU interval state'}]){const mock=mockGPU(config);await assert.rejects(create(mock),/mock/);mock.assertDisposed();}
});
await test('generic runners remain five pipelines with no interval allocation or batched guard work',async()=>{
 await withRunner({baseline:true},async(runner,mock)=>{for(const mode of ['staged','batched']){const result=await runner.run({...fresh(),axisSubmission:mode});assert.equal(result.admitted,false);}assert.deepEqual(mock.compiled,baseStages);assert.deepEqual(mock.submissions,[...baseStages,...baseStages].map(s=>[s]));assert(!mock.buffers.some(b=>b.label==='Resident staged GPU interval state'));},{continuationGuard:'replay-unmodelled'});
 for(const guard of ['axis-mirror-interval','axis-mirror-unit'])await withRunner({baseline:true},async(runner,mock)=>{await assert.rejects(runner.run({...fresh(),guard,axisSubmission:'batched'}),/not compiled/);assert.equal(mock.submissions.length,0);},{continuationGuard:'replay-unmodelled'});
 await withRunner({},async(runner,mock)=>{await assert.rejects(runner.run({...fresh(),axisSubmission:'unchecked'}),/invalid axis submission mode/);assert.equal(mock.submissions.length,0);});
});
await test('missing batched finish mutation is caught by actual encoded stage ordering',async()=>{
 const url=new URL('../dist/plt-resident-gpu.js',import.meta.url),source=fs.readFileSync(url,'utf8'),mutated=source.replace("encode('rr_axis_finish_step');","/* deliberately omit finish */");assert.notEqual(mutated,source);const absolute=mutated.replace(/(['"])(\.\/[a-z0-9-]+\.js)\1/g,(_,quote,path)=>quote+new URL(path,url).href+quote),factory=(await import('data:text/javascript;base64,'+Buffer.from(absolute).toString('base64'))).createPLTResidentRunner;
 await withRunner({},async(runner,mock)=>{await runner.run({...fresh(),axisSubmission:'batched'});assert.throws(()=>assert.deepEqual(mock.submissions,expectedSubmissions('batched',2)),{code:'ERR_ASSERTION'});assert(!mock.submissions.flat().includes('rr_axis_finish_step'));},{},factory);
 report.mutations.push({name:'omit-batched-finish',rejected:true,oracle:'actual command-buffer pass order'});
});

const files=['utd-functions.wgsl','plt-resident-geometry.wgsl','plt-resident-interval.wgsl','plt-resident-axis-guard.wgsl','plt-resident-rng.wgsl','plt-resident-fsd.wgsl','plt-resident-walker.wgsl'];
const sources=files.map(name=>fs.readFileSync(new URL('../dist/'+name,import.meta.url),'utf8')),source=sources.join('\n');
report.sourceSHA256=Object.fromEntries(files.map((name,i)=>[name,crypto.createHash('sha256').update(sources[i]).digest('hex')]));
report.testSHA256=crypto.createHash('sha256').update(fs.readFileSync(new URL(import.meta.url))).digest('hex');
report.hostSHA256=Object.fromEntries(['plt-resident-gpu.js','plt-resident-geometry.js'].map(name=>[name,crypto.createHash('sha256').update(fs.readFileSync(new URL('../dist/'+name,import.meta.url))).digest('hex')]));
if(!hostOnly)await nativeChecks(source);
report.allPass=true;report.status='passed';
}catch(error){report.status='failed';report.failure={name:error.name,message:error.message,stack:error.stack};throw error;}finally{save();console.log(JSON.stringify(report,null,2));}

async function nativeChecks(source){
 const backend=execute?process.env.WEBGPU_EXECUTION_BACKEND:'null';assert(!execute||['vulkan','metal','d3d11','d3d12','opengl','opengles'].includes(backend),'--execute needs explicit executable backend');
 const {create,globals}=await import('webgpu');Object.assign(globalThis,globals);let gpu=create([`backend=${backend}`]);const adapter=await gpu.requestAdapter();assert(adapter,'Native adapter unavailable');const device=await adapter.requestDevice(),buffers=[];
 report.backend=backend;report.adapter={vendor:adapter.info.vendor,device:adapter.info.device,description:adapter.info.description};report.compilations=[];
 const harness=`
@compute @workgroup_size(1) fn safety_initialize(){var s=rr_lanes[0];rr_rng_words[0]=0x80000000u;rr_rng_words[1]=0x80000000u;rri_initialize(&s);rr_lanes[0]=s;}
@compute @workgroup_size(1) fn safety_finish(){var s=rr_lanes[0];rg_flags=s.pending.w;rr_rng_count=s.trace.w;rr_rng_hash=s.progress.x;rr_finish(&s,0u,7u);rr_lanes[0]=s;}
`;
 const entries={safety_initialize:[0,1,3],safety_finish:[0,3,4],rr_axis_prepare_step:[0,1,3,8],rr_axis_finish_step:[0,1,3,4,8],rr_records_step:[0,1,3,4,5,6],rr_operator_step:[0,3,4,6,7],rr_scatter_step:[0,1,3,4,5,7]};
 const compile=async(src,names=Object.keys(entries),label='baseline')=>{const module=device.createShaderModule({code:src+harness});assert.deepEqual((await module.getCompilationInfo()).messages.filter(m=>m.type==='error'),[]);const pipelines={};for(const name of names){const started=performance.now();pipelines[name]=await device.createComputePipelineAsync({layout:'auto',compute:{module,entryPoint:name,constants:{RR_AXIS_INTERVAL_ENABLED:1}}});report.compilations.push({label,entryPoint:name,wallMs:performance.now()-started});console.log('PASS native compile',label,name);}return pipelines;};
 const rw=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC;
 const make=(size,usage)=>{const b=device.createBuffer({size,usage});buffers.push(b);return b;};
 const packed=packResidentScene(scene),params=packResidentParameters(scene.camera,options,packed,fresh());new Uint32Array(params).set([0,1,PLT_RESIDENT_LIMITS.auditVectors,0],12);new Uint32Array(params)[37]=1;new Uint32Array(params)[38]=4;
 const sizes={0:160,1:packed.data.byteLength,3:PLT_RESIDENT_LIMITS.laneBytes,4:PLT_RESIDENT_LIMITS.auditVectors*16,5:16,6:64,7:768,8:256},resources={};
 try{
  device.pushErrorScope('validation');const pipelines=await compile(source);
  if(!execute){assert.equal(await device.popErrorScope(),null);return;}
  for(const [binding,size]of Object.entries(sizes))resources[binding]=make(size,binding==='0'?GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST:rw);
  const reset=(lane,customParams=params,audit=new ArrayBuffer(sizes[4]),weights=new ArrayBuffer(sizes[7]))=>{for(const [binding,b]of Object.entries(resources))device.queue.writeBuffer(b,0,binding==='0'?customParams:binding==='1'?packed.data:binding==='3'?lane:binding==='4'?audit:binding==='7'?weights:new ArrayBuffer(sizes[binding]));};
  const dispatch=async(name,pipeline=pipelines[name])=>{const group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:entries[name].map(binding=>({binding,resource:{buffer:resources[binding]}}))}),encoder=device.createCommandEncoder(),pass=encoder.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(1);pass.end();device.queue.submit([encoder.finish()]);await device.queue.onSubmittedWorkDone();report.actualGPUExecuted=true;report.nativeDispatches=(report.nativeDispatches??0)+1;};
  const read=async(binding)=>{const size=sizes[binding],b=make(size,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST),encoder=device.createCommandEncoder();encoder.copyBufferToBuffer(resources[binding],0,b,0,size);device.queue.submit([encoder.finish()]);await b.mapAsync(GPUMapMode.READ);const out=b.getMappedRange().slice(0);b.unmap();return out;};
  function laneFixture({meta=2,pending=1}={}){const b=new ArrayBuffer(sizes[3]),u=new Uint32Array(b),f=new Float32Array(b);u.set([0,123,0,1],0);u.set([0,0,1,0],4);u.set([1,1,1,0],8);u.set([2166136261,123,1,0],16);f.set(new Float32Array(params).slice(16,19),20);f[26]=1;f[34]=1;u.set([1,0,0,0],52);f[43]=1;f[46]=1;f[47]=2;u.set([0,1,1,pending],76);u.set([meta,3,0,0],108);return b;}
  async function check(name,fn){await fn();report.nativeChecks.push(name);console.log('PASS native',name);}
  await check('camera initialization grants private unit marker only on successful camera interval and clears stale markers',async()=>{
   for(const kind of ['unit','generic','wave','disabled','unsupported-scene','invalid-camera']){const p=params.slice(0),pu=new Uint32Array(p),pf=new Float32Array(p);if(kind==='generic')pu[38]=2;if(kind==='wave')pu[11]=1;if(kind==='disabled')pu[38]=0;if(kind==='invalid-camera')pf.fill(0,20,32);reset(laneFixture({meta:0xffffffff}),p);if(kind==='unsupported-scene')device.queue.writeBuffer(resources[1],120,new Uint32Array([0]));await dispatch('safety_initialize');const lane=new Uint32Array(await read(3));assert.equal(lane[108],kind==='unit'?2:kind==='generic'?1:0,kind);}
  });
  await check('unit prepare rejects absent, generic, and forged marker enums; generic prepare keeps full normalization',async()=>{
   for(const meta of [0,1,2,3,0xffffffff]){reset(laneFixture({meta}));await dispatch('rr_axis_prepare_step');const state=new Uint32Array(await read(8));assert.equal(state[0],meta===2?1:0,'marker '+meta);if(meta!==2)assert(state[1]&1,'missing provenance must be unsupported');}
   const p=params.slice(0);new Uint32Array(p)[38]=2;const lane=laneFixture({meta:0});new Float32Array(lane)[26]=2;reset(lane,p);await dispatch('rr_axis_prepare_step');assert.equal(new Uint32Array(await read(8))[0],1,'generic nonunit ray must use full normalization');
  });
  await check('prepare without all triangle scans cannot finish successfully or retain unit proof',async()=>{
   reset(laneFixture());await dispatch('rr_axis_prepare_step');await dispatch('rr_axis_finish_step');const lane=new Uint32Array(await read(3)),audit=new Uint32Array(await read(4));assert.equal(lane[3],3);assert.equal(lane[108],0);assert.equal(lane[79],0);assert(lane[11]&268435456);assert.equal(audit[1545],0);assert.equal(new Float32Array(lane.buffer)[49],0);
  });
  await check('failed/replayed finish revokes proof and never adds partial contribution',async()=>{
   for(const flags of [2,134217730,268435458]){const lane=laneFixture({pending:0}),u=new Uint32Array(lane),f=new Float32Array(lane);u[11]=flags;f[48]=19;f[49]=7;reset(lane);await dispatch('safety_finish');const result=new Uint32Array(await read(3)),audit=new Uint32Array(await read(4));assert.equal(result[108],0);assert.equal(result[3],3);assert.equal(new Float32Array(result.buffer)[49],7);assert.equal(audit[1545],0);}
  });
  const pending=laneFixture(),audit=new ArrayBuffer(sizes[4]),au=new Uint32Array(audit),af=new Float32Array(audit);au[1243]=0xffffffff;af[1237]=1;
  await check('missing finish leaves lane pending: records, operator, and scatter add no contribution or state',async()=>{
   for(const name of ['rr_records_step','rr_operator_step','rr_scatter_step']){reset(pending,params,audit);await dispatch(name);for(const [binding,expected]of [[3,pending],[4,audit],[6,new ArrayBuffer(sizes[6])],[7,new ArrayBuffer(sizes[7])]])assert.deepEqual(new Uint8Array(await read(binding)),new Uint8Array(expected),name+' binding '+binding);assert.deepEqual([...new Uint32Array(await read(5))],[name==='rr_scatter_step'?1:0,0,0,0]);}
  });
  await check('null and FSD continuations revoke camera-unit provenance',async()=>{
   for(const aperture of [0,1]){const lane=laneFixture({pending:0});new Uint32Array(lane)[76]=aperture;reset(lane);await dispatch('rr_scatter_step');const result=new Uint32Array(await read(3));assert.equal(result[108],0,aperture?'FSD':'null');if(!aperture)assert.equal(result[5],1);}
  });
  const proofBegin=source.indexOf('fn rr_axis_prepare_step('),proofEnd=source.indexOf('\n@compute',proofBegin);assert(proofBegin>=0&&proofEnd>proofBegin);const proofSource=source.slice(proofBegin,proofEnd),proofChanged=proofSource.replace('select(0u,1u,s.intervalMeta.x==2u)','1u');assert.notEqual(proofChanged,proofSource);const bypass=source.slice(0,proofBegin)+proofChanged+source.slice(proofEnd);const proofPipelines=await compile(bypass,['rr_axis_prepare_step'],'bypass-private-proof');reset(laneFixture({meta:1}));await dispatch('rr_axis_prepare_step',proofPipelines.rr_axis_prepare_step);assert.equal(new Uint32Array(await read(8))[0],1,'Mutation must actually accept forbidden generic provenance');report.mutations.push({name:'bypass-private-unit-provenance',rejected:true,oracle:'native missing/generic-marker rejection',witnessMarker:1});
  const begin=source.indexOf('fn rr_operator_step('),end=source.indexOf('fn rr_scatter_step(',begin);assert(begin>=0&&end>begin);const original=source.slice(begin,end),altered=original.replace('s.vertexMeta.w!=0u||','');assert.notEqual(altered,original);const barrierSource=source.slice(0,begin)+altered+source.slice(end),barrierPipelines=await compile(barrierSource,['rr_operator_step'],'remove-pending-barrier');reset(pending,params,audit);await dispatch('rr_operator_step',barrierPipelines.rr_operator_step);assert.notDeepEqual(new Uint8Array(await read(4)),new Uint8Array(audit),'Mutation must actually change a pending audit');report.mutations.push({name:'remove-operator-pending-barrier',rejected:true,oracle:'native pending audit immutability'});
  assert.equal(await device.popErrorScope(),null);report.actualGPUExecuted=true;
 }finally{for(const b of buffers)b.destroy();device.destroy();gpu=null;}
}
