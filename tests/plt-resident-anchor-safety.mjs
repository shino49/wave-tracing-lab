/** Focused anchored-ray provenance, pending-guard, and submission lifecycle safety.
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
import {createPLTResidentRunner,packResidentParameters,PLT_RESIDENT_LIMITS,PLT_RESIDENT_BUILD_ID,isResidentReplayable} from '../dist/plt-resident-gpu.js';

globalThis.GPUBufferUsage={MAP_READ:1,COPY_SRC:4,COPY_DST:8,UNIFORM:64,STORAGE:128};
globalThis.GPUMapMode={READ:1};
const execute=process.argv.includes('--execute'),hostOnly=process.argv.includes('--host-only');
assert(!(execute&&hostOnly),'Choose host-only or native execution');
const stages=['rr_init','rr_traverse_step','rr_anchor_init_step','rr_axis_prepare_step','rr_axis_plane_step','rr_axis_scan_step','rr_anchor_point_step','rr_axis_finish_step','rr_records_step','rr_operator_step','rr_scatter_step'];
const anchorAliases={rr_init:'rr_init_anchored',rr_axis_prepare_step:'rr_anchor_prepare_step',rr_axis_plane_step:'rr_anchor_plane_step',rr_axis_finish_step:'rr_anchor_finish_step'};
const logicalName=name=>Object.entries(anchorAliases).find(([,actual])=>actual===name)?.[0]??name;
const baseStages=stages.filter(name=>!name.startsWith('rr_axis_')&&!name.startsWith('rr_anchor_'));
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
const groupName=commands=>commands.length===1?commands[0]:['rr_axis_prepare_step','rr_anchor_init_step'].includes(commands[0])?'axis-guard-group':'axis-triangle-loop';
function expectedSubmissions(mode,n,axis=true,anchor=true){
 const guard=mode==='batched'?[[...(anchor?['rr_anchor_init_step']:[]),'rr_axis_prepare_step',...pairs(n),...(anchor?['rr_anchor_point_step']:[]),'rr_axis_finish_step']]:[...(anchor?[['rr_anchor_init_step']]:[]),['rr_axis_prepare_step'],pairs(n),...(anchor?[['rr_anchor_point_step']]:[]),['rr_axis_finish_step']];
 return [['rr_init'],['rr_traverse_step'],...(axis?guard:[]),['rr_records_step'],['rr_operator_step'],['rr_scatter_step']];
}
const report={createdAt:new Date().toISOString(),buildId:PLT_RESIDENT_BUILD_ID,executionRequested:execute,actualGPUExecuted:false,allPass:false,status:'running',hostMockOnly:hostOnly,fullCPUShadowMandatory:true,physicalGPUValidated:false,browserValidated:false,hardwarePerformanceClaim:false,hostChecks:[],nativeChecks:[],mutations:[]};
const reportURL=new URL('../research/plt-resident-anchor-safety'+(execute?'':'-compile')+'-validation.json',import.meta.url);
const save=()=>fs.writeFileSync(reportURL,JSON.stringify(report,null,2)+'\n');
async function test(name,fn){await fn();report.hostChecks.push(name);console.log('PASS host',name);}

function mockGPU(config={}){
 const buffers=[],compiled=[],submissions=[],parameterSnapshots=[],events=[],listeners=new Map(),lost=deferred();
 let initial=new Uint32Array(4),uniform=new Uint32Array(40),scopes=0,destroyed=0,reads=0,currentStage,encoders=0,passCount=0;
 const device={limits:{maxBufferSize:1e9,maxStorageBufferBindingSize:1e9},lost:lost.promise,
  addEventListener:(name,fn)=>listeners.set(name,fn),removeEventListener:name=>listeners.delete(name),
  pushErrorScope(){scopes++;},async popErrorScope(){assert(scopes>0);scopes--;return null;},
  createShaderModule:()=>({getCompilationInfo:async()=>({messages:[]})}),
  async createComputePipelineAsync({compute}){compiled.push(compute.entryPoint);assert.equal(compute.constants.RR_AXIS_INTERVAL_ENABLED,config.baseline?0:1);assert.equal(compute.constants.RR_ANCHOR_ENABLED,config.baseline||config.axisOnly?0:1);if(config.pipelineFailure===logicalName(compute.entryPoint))throw new Error('mock pipeline failure');return {...compute,entryPoint:logicalName(compute.entryPoint),getBindGroupLayout:()=>({entryPoint:logicalName(compute.entryPoint)})};},
  createBindGroup(spec){assert(spec.entries.every(e=>e.resource.buffer));const anchored=!config.baseline&&!config.axisOnly,expected=anchored&&['rr_init','rr_anchor_init_step','rr_anchor_point_step','rr_axis_prepare_step','rr_axis_plane_step','rr_axis_finish_step'].includes(spec.layout.entryPoint);assert.equal(spec.entries.some(e=>e.binding===9),expected,'anchor state binding '+spec.layout.entryPoint);return spec;},
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
const create=(mock,extra={},factory=createPLTResidentRunner)=>factory(mock.scene,mock.scene.camera,options,{gpu:mock.gpu,shaderSource:'HOST MOCK ONLY; NOT WGSL',timeoutMs:1000,recoveryMode:'selective',continuationGuard:'axis-mirror-anchor',...extra});
async function withRunner(config,fn,extra={},factory){const mock=mockGPU(config);let runner;try{runner=await create(mock,extra,factory);await fn(runner,mock);}finally{runner?.destroy();await delay();mock.assertDisposed();}}
function assertBarrierEvents(mock){
 let outstanding=false;
 for(const event of mock.events){if(event.kind==='submit'){assert(!outstanding,'Later submit before preceding completion');outstanding=true;}else if(event.kind==='done'){assert(outstanding);outstanding=false;}else if(event.kind==='readback')assert(!outstanding,'Readback before group completion');}
 assert(!outstanding);
}
function assertCandidate(result,input,before){assert.equal(result.candidate,true);assert.equal(result.admitted,false);assert.equal(result.validation,undefined);assert.strictEqual(result.image,result.audit.reference);assert.deepEqual(input.rngStates,before,'Failed or unadmitted candidate cannot mutate input RNG');}

try{
await test('staged default and batched reuse identical eleven pipelines with exact dependent pass barriers',()=>withRunner({},async(runner,mock)=>{
 const cases=[['axis-mirror-anchor',undefined,6],['axis-mirror-anchor','batched',6],['axis-mirror-unit','batched',4],['axis-mirror-interval','staged',2],['replay-unmodelled','batched',1],['audit-only','staged',0],['axis-mirror-anchor','batched',6]];
 const expected=[];
 for(const [guard,axisSubmission,config] of cases){const input={...fresh(),guard,...(axisSubmission?{axisSubmission}:{})},before=input.rngStates.slice(),result=await runner.run(input),mode=axisSubmission??'staged',axis=guard.startsWith('axis-');assertCandidate(result,input,before);const order=expectedSubmissions(mode,scene.triangles.length,axis,guard==='axis-mirror-anchor');expected.push(...order);assert.deepEqual(mock.submissions,expected);assert.equal(result.dispatches,order.flat().length-1);assert.equal(result.recovery.axisSubmission,mode);assert.equal(mock.parameterSnapshots.at(-1)[38],config);
  for(const key of ['gpuAnchorInitSubmitAndWaitMs','gpuAnchorPointSubmitAndWaitMs']){assert(Number.isFinite(result.timings[key])&&result.timings[key]>=0);if(guard!=='axis-mirror-anchor'||mode==='batched')assert.equal(result.timings[key],0,key);}
  for(const key of ['gpuAxisPrepareSubmitAndWaitMs','gpuAxisTriangleLoopSubmitAndWaitMs','gpuAxisFinishSubmitAndWaitMs','gpuAxisGuardSubmitAndWaitMs']){assert(Number.isFinite(result.timings[key])&&result.timings[key]>=0);if(!axis||(mode==='batched'&&key!=='gpuAxisGuardSubmitAndWaitMs')||(mode==='staged'&&key==='gpuAxisGuardSubmitAndWaitMs'))assert.equal(result.timings[key],0,key);}
 }
 assert.deepEqual(mock.compiled,stages.map(name=>anchorAliases[name]??name));assertBarrierEvents(mock);for(const label of ['Resident staged GPU interval state','Resident anchored GPU proof state']){const state=mock.buffers.filter(b=>b.label===label);assert.equal(state.length,1);assert.equal(state[0].size,PLT_RESIDENT_LIMITS.tilePixels*256);}
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
 for(const mode of ['staged','batched'])for(const stage of mode==='batched'?['axis-guard-group']:['rr_anchor_init_step','rr_axis_prepare_step','axis-triangle-loop','rr_anchor_point_step','rr_axis_finish_step'])for(const method of ['cancel','lost','destroy','timeout']){const entered=deferred(),held=deferred(),controller=new AbortController();await withRunner({holdStage:stage,hold:held.promise,onWork:name=>{if(name===stage)entered.resolve();}},async(runner,mock)=>{const input={...fresh(),axisSubmission:mode,signal:controller.signal},before=input.rngStates.slice(),pending=runner.run(input);const rejection=assert.rejects(pending,method==='lost'?/mock device lost/:method==='timeout'?/timeout/:{name:'AbortError'});await entered.promise;if(method==='cancel')controller.abort();if(method==='lost')mock.lose();if(method==='destroy'){runner.destroy();runner.destroy();}await rejection;held.resolve();await delay();const order=expectedSubmissions(mode,2),index=order.findIndex(c=>groupName(c)===stage);assert.deepEqual(mock.submissions,order.slice(0,index+1));assert.equal(mock.reads,0);assert.deepEqual(input.rngStates,before);},{timeoutMs:method==='timeout'?30:1000});}
});
await test('staged anchor init, point, prepare, and finish timeouts withhold readback and release state',async()=>{
 for(const stage of ['rr_anchor_init_step','rr_axis_prepare_step','rr_anchor_point_step','rr_axis_finish_step'])await withRunner({stallStage:stage},async(runner,mock)=>{await assert.rejects(runner.run(fresh()),/timeout/);const order=expectedSubmissions('staged',2),index=order.findIndex(c=>groupName(c)===stage);assert.deepEqual(mock.submissions,order.slice(0,index+1));assert.equal(mock.reads,0);},{timeoutMs:30});
});
await test('anchor setup cancellation and pipeline/state failures release exactly once',async()=>{
 for(const stage of stages){const controller=new AbortController(),mock=mockGPU();await assert.rejects(create(mock,{signal:controller.signal,onSetupProgress:p=>{if(p.stageName===stage)controller.abort();}}),{name:'AbortError'});assert.deepEqual(mock.compiled,stages.slice(0,stages.indexOf(stage)).map(name=>anchorAliases[name]??name));mock.assertDisposed();}
 for(const config of [...stages.slice(2,8).map(pipelineFailure=>({pipelineFailure})),{bufferFailure:'Resident staged GPU interval state'},{bufferFailure:'Resident anchored GPU proof state'}]){const mock=mockGPU(config);await assert.rejects(create(mock),/mock/);mock.assertDisposed();}
});
await test('generic runners remain five pipelines with no interval allocation or batched guard work',async()=>{
 await withRunner({baseline:true},async(runner,mock)=>{for(const mode of ['staged','batched']){const result=await runner.run({...fresh(),axisSubmission:mode});assert.equal(result.admitted,false);}assert.deepEqual(mock.compiled,baseStages);assert.deepEqual(mock.submissions,[...baseStages,...baseStages].map(s=>[s]));assert(!mock.buffers.some(b=>['Resident staged GPU interval state','Resident anchored GPU proof state'].includes(b.label)));},{continuationGuard:'replay-unmodelled'});
 for(const guard of ['axis-mirror-interval','axis-mirror-unit','axis-mirror-anchor'])await withRunner({baseline:true},async(runner,mock)=>{await assert.rejects(runner.run({...fresh(),guard,axisSubmission:'batched'}),/not compiled/);assert.equal(mock.submissions.length,0);},{continuationGuard:'replay-unmodelled'});
 await withRunner({},async(runner,mock)=>{await assert.rejects(runner.run({...fresh(),axisSubmission:'unchecked'}),/invalid axis submission mode/);assert.equal(mock.submissions.length,0);});
});
await test('legacy axis runners retain nine pipelines and cannot opt into uncompiled anchor state',async()=>{
 for(const guard of ['axis-mirror-interval','axis-mirror-unit'])await withRunner({axisOnly:true},async(runner,mock)=>{await runner.run({...fresh(),guard});assert.deepEqual(mock.compiled,stages.filter(name=>!name.startsWith('rr_anchor_')));assert(!mock.buffers.some(b=>b.label==='Resident anchored GPU proof state'));await assert.rejects(runner.run({...fresh(),guard:'axis-mirror-anchor'}),/not compiled/);},{continuationGuard:guard});
});
await test('repeated sample windows reset init and preserve input RNG across runner reuse',()=>withRunner({},async(runner,mock)=>{
 for(const [sampleStart,samplesPerPixel]of [[0,1],[1,3],[0,4],[2,1]]){const input={...fresh(),sampleStart,samplesPerPixel},before=input.rngStates.slice(),start=mock.submissions.length,result=await runner.run(input);assertCandidate(result,input,before);assert.deepEqual(mock.submissions.slice(start),expectedSubmissions('staged',2));assert.equal(mock.parameterSnapshots.at(-1)[8],sampleStart);assert.equal(mock.parameterSnapshots.at(-1)[9],samplesPerPixel);assert.equal(mock.parameterSnapshots.at(-1)[38],6);}
}));
await test('omitted anchor init, point, or finish mutations are caught by encoded stage ordering',async()=>{
 for(const omitted of ['rr_anchor_init_step','rr_anchor_point_step','rr_axis_finish_step']){
 const url=new URL('../dist/plt-resident-gpu.js',import.meta.url),source=fs.readFileSync(url,'utf8'),mutated=source.replace(`encode('${omitted}');`,";/* deliberately omit stage */");assert.notEqual(mutated,source);const absolute=mutated.replace(/(['"])(\.\/[a-z0-9-]+\.js)\1/g,(_,quote,path)=>quote+new URL(path,url).href+quote),factory=(await import('data:text/javascript;base64,'+Buffer.from(absolute).toString('base64'))).createPLTResidentRunner;
 await withRunner({},async(runner,mock)=>{await runner.run({...fresh(),axisSubmission:'batched'});assert.throws(()=>assert.deepEqual(mock.submissions,expectedSubmissions('batched',2)),{code:'ERR_ASSERTION'});assert(!mock.submissions.flat().includes(omitted));},{},factory);
 report.mutations.push({name:'omit-batched-'+omitted,rejected:true,oracle:'actual command-buffer pass order'});
 }
});

const files=['utd-functions.wgsl','plt-resident-geometry.wgsl','plt-resident-interval.wgsl','plt-resident-axis-guard.wgsl','plt-resident-rng.wgsl','plt-resident-fsd.wgsl','plt-resident-walker.wgsl'];
const sources=files.map(name=>fs.readFileSync(new URL('../dist/'+name,import.meta.url),'utf8')),source=sources.join('\n');
report.sourceSHA256=Object.fromEntries(files.map((name,i)=>[name,crypto.createHash('sha256').update(sources[i]).digest('hex')]));
report.testSHA256=crypto.createHash('sha256').update(fs.readFileSync(new URL(import.meta.url))).digest('hex');
report.hostSHA256=Object.fromEntries(['plt-resident-gpu.js','plt-resident-geometry.js'].map(name=>[name,crypto.createHash('sha256').update(fs.readFileSync(new URL('../dist/'+name,import.meta.url))).digest('hex')]));
const {WgslReflect}=await import('wgsl_reflect/wgsl_reflect.module.js'),reflected=new WgslReflect(source);
report.abi={};for(const [name,size]of [['RRLane',448],['RRAState',204],['RRAnchorState',176]]){const struct=reflected.structs.find(s=>s.name===name);assert.equal(struct.size,size,name+' ABI');report.abi[name]={bytes:struct.size,members:Object.fromEntries(struct.members.map(m=>[m.name,m.offset]))};}
assert.equal(PLT_RESIDENT_LIMITS.auditVectors,387,'Existing audit ABI remains unchanged');
if(!hostOnly)await nativeChecks(source);
report.allPass=true;report.status='passed';
}catch(error){report.status='failed';report.failure={name:error.name,message:error.message,stack:error.stack};throw error;}finally{save();console.log(JSON.stringify(report,null,2));}

async function nativeChecks(source){
 const backend=execute?process.env.WEBGPU_EXECUTION_BACKEND:'null';assert(!execute||['vulkan','metal','d3d11','d3d12','opengl','opengles'].includes(backend),'--execute needs explicit executable backend');
 const {create,globals}=await import('webgpu');Object.assign(globalThis,globals);let gpu=create([`backend=${backend}`]);const adapter=await gpu.requestAdapter();assert(adapter,'Native adapter unavailable');const device=await adapter.requestDevice(),buffers=[];
 report.backend=backend;report.adapter={vendor:adapter.info.vendor,device:adapter.info.device,description:adapter.info.description};report.compilations=[];
 const harness=`
@compute @workgroup_size(1) fn safety_finish(){var s=rr_lanes[0];rg_flags=s.pending.w;rr_rng_count=s.trace.w;rr_rng_hash=s.progress.x;rr_finish(&s,0u,7u);rr_lanes[0]=s;}
`;
 const entries={rr_init:[0,2,3,9],rr_traverse_step:[0,1,3,4,7],rr_anchor_init_step:[0,3,9],rr_axis_prepare_step:[0,1,3,8,9],rr_axis_plane_step:[0,1,3,8,9],rr_axis_scan_step:[0,1,3,8],rr_anchor_point_step:[0,1,3,8,9],rr_axis_finish_step:[0,1,3,4,8,9],rr_records_step:[0,1,3,4,5,6],rr_operator_step:[0,3,4,6,7],rr_scatter_step:[0,1,3,4,5,7],safety_finish:[0,3,4]};
 const compile=async(src,names=Object.keys(entries),label='baseline',anchor=1,axis=1)=>{const module=device.createShaderModule({code:src+harness});assert.deepEqual((await module.getCompilationInfo()).messages.filter(m=>m.type==='error'),[]);const pipelines={};for(const name of names){const started=performance.now();pipelines[name]=await device.createComputePipelineAsync({layout:'auto',compute:{module,entryPoint:anchor?(anchorAliases[name]??name):name,constants:{RR_AXIS_INTERVAL_ENABLED:axis,RR_ANCHOR_ENABLED:anchor}}});report.compilations.push({label,logicalStage:name,entryPoint:anchor?(anchorAliases[name]??name):name,wallMs:performance.now()-started});console.log('PASS native compile',label,name);}return pipelines;};
 const rw=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC;
 const make=(size,usage)=>{const b=device.createBuffer({size,usage});buffers.push(b);return b;};
 const packed=packResidentScene(scene),params=packResidentParameters(scene.camera,options,packed,fresh());new Uint32Array(params).set([0,1,PLT_RESIDENT_LIMITS.auditVectors,0],12);new Uint32Array(params)[37]=1;new Uint32Array(params)[38]=6;
 const sizes={0:160,1:packed.data.byteLength,2:16,3:PLT_RESIDENT_LIMITS.laneBytes,4:PLT_RESIDENT_LIMITS.auditVectors*16*4,5:16,6:64,7:768,8:256,9:256},resources={};
 const anchorOffsets=report.abi.RRAnchorState.members,proofAt=anchorOffsets.proofMeta/4,checkedAt=anchorOffsets.checked/4,resultAt=anchorOffsets.result/4;
 try{
  device.pushErrorScope('validation');const pipelines=await compile(source);const legacyNames=['rr_init','rr_axis_prepare_step','rr_axis_plane_step','rr_axis_finish_step'],legacyPipelines=await compile(source,legacyNames,'legacy-no-anchor-bindings',0);
  if(!execute){assert.equal(await device.popErrorScope(),null);return;}
  for(const [binding,size]of Object.entries(sizes))resources[binding]=make(size,binding==='0'?GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST:rw);
  const write=(binding,data)=>device.queue.writeBuffer(resources[binding],0,data);
  const reset=(custom={})=>{for(const [binding]of Object.entries(resources))write(binding,custom[binding]??(binding==='0'?params:binding==='1'?packed.data:binding==='2'?new Uint32Array([0,123,0,0]):new ArrayBuffer(sizes[binding])));};
  const dispatch=async(name,pipeline=pipelines[name])=>{const group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:entries[name].map(binding=>({binding,resource:{buffer:resources[binding]}}))}),encoder=device.createCommandEncoder(),pass=encoder.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(1);pass.end();device.queue.submit([encoder.finish()]);await device.queue.onSubmittedWorkDone();report.actualGPUExecuted=true;report.nativeDispatches=(report.nativeDispatches??0)+1;};
  const read=async(binding)=>{const size=sizes[binding],b=make(size,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST),encoder=device.createCommandEncoder();encoder.copyBufferToBuffer(resources[binding],0,b,0,size);device.queue.submit([encoder.finish()]);await b.mapAsync(GPUMapMode.READ);const out=b.getMappedRange().slice(0);b.unmap();return out;};
  const snapshot=async()=>Object.fromEntries(await Promise.all([3,4,5,6,7,8,9].map(async binding=>[binding,await read(binding)])));
  const scan=async()=>{for(let i=0;i<packed.metadata.triangles;i++){await dispatch('rr_axis_plane_step');await dispatch('rr_axis_scan_step');}};
  const u32=b=>new Uint32Array(b),f32=b=>new Float32Array(b),bytes=b=>new Uint8Array(b);
  async function check(name,fn){await fn();report.nativeChecks.push(name);console.log('PASS native',name);}
  function rejected(lane,proof){assert.equal(u32(lane)[3],3,'failed sample must pause for whole CPU replay');assert.equal(u32(lane)[108],0,'reject must revoke lane provenance');assert.equal(u32(lane)[79],0,'finish must clear pending barrier');assert(u32(lane)[11]&268435456,'reject must mark axis uncertainty');assert.equal(u32(proof)[proofAt],0,'failed geometry proof cannot remain valid');assert.equal(u32(proof)[checkedAt],0,'checked marker must be consumed');assert.equal(f32(lane)[49],0,'failed sample cannot contribute');}
  await check('legacy initialization and axis pipelines require no hidden anchor binding',async()=>{for(const name of legacyNames){const pipeline=legacyPipelines[name];device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:entries[name].filter(binding=>binding!==9).map(binding=>({binding,resource:{buffer:resources[binding]}}))});}});
  let pending,initialized,prepared,scanned,pointed,finished;
  await check('actual camera traversal and separate raw reconstruction keep transport RNG and audit unchanged',async()=>{
   reset();await dispatch('rr_init');await dispatch('rr_traverse_step');pending=await snapshot();assert.equal(u32(pending[3])[108],3);assert.equal(u32(pending[3])[79],1);assert.equal(u32(pending[3])[3],1);assert.equal(u32(pending[3])[15],2,'camera consumed exactly two words');
   await dispatch('rr_anchor_init_step');initialized=await snapshot();for(const binding of [3,4,5,6,7,8])assert.deepEqual(bytes(initialized[binding]),bytes(pending[binding]),'raw reconstruction changed binding '+binding);
   assert.deepEqual([...u32(initialized[9]).slice(proofAt,proofAt+4)],[1,0,123,0]);assert.equal(u32(initialized[9])[checkedAt],0);
   await dispatch('rr_axis_prepare_step');prepared=await snapshot();assert.equal(u32(prepared[8])[0],1);
   await scan();scanned=await snapshot();await dispatch('rr_anchor_point_step');pointed=await snapshot();assert.equal(u32(pointed[9])[resultAt],1,'positive primary fixture must certify');assert.equal(u32(pointed[9])[checkedAt],1);assert.equal(u32(pointed[9])[11],4,'point pass pre-reflects z sign');
   assert.deepEqual(bytes(pointed[3]),bytes(pending[3]),'point proof must leave pending transport intact');await dispatch('rr_axis_finish_step');finished=await snapshot();assert.equal(u32(finished[3])[3],1);assert.equal(u32(finished[3])[108],3);assert.equal(u32(finished[3])[79],0);assert.equal(u32(finished[9])[proofAt+3],1,'finish counts certified hit once');assert.equal(u32(finished[9])[checkedAt],0);
   await dispatch('rr_axis_finish_step');assert.deepEqual(bytes(await read(9)),bytes(finished[9]),'repeated finish must not count a second hit');
  });
  await check('anchor-only stages are byte-for-byte no-ops in legacy continuation modes',async()=>{
   for(const mode of [0,1,2,4])for(const stage of ['rr_anchor_init_step','rr_anchor_point_step']){const p=params.slice(0);u32(p)[38]=mode;reset({...pointed,0:p});await dispatch(stage);for(const binding of [3,4,5,6,7,8,9])assert.deepEqual(bytes(await read(binding)),bytes(pointed[binding]),stage+' legacy '+mode+' binding '+binding);}
  });
  await check('rr_init clears old proof ownership and checked state on every repeated sample window',async()=>{
   for(const sampleStart of [0,1,3,0]){const p=params.slice(0);u32(p)[8]=sampleStart;const poison=new Uint32Array(sizes[9]/4).fill(0xffffffff);reset({0:p,9:poison});await dispatch('rr_init');const a=u32(await read(9)),lane=u32(await read(3));assert.deepEqual([...a.slice(proofAt,proofAt+4)],[0,0,0,0]);assert.equal(a[checkedAt],0);assert.equal(lane[2],sampleStart);assert.equal(lane[108],0);}
  });
  await check('successive samples rebuild raw proof from the next GPU prefix without inheriting reflected state',async()=>{
   const p=params.slice(0);u32(p)[9]=3;reset({0:p});await dispatch('rr_init');let prefix=123;report.sameRunSamples=[];
   for(let sample=0;sample<3;sample++){await dispatch('rr_traverse_step');const before=await read(3);assert.equal(u32(before)[2],sample);assert.equal(u32(before)[17],prefix);assert.equal(u32(before)[15],2);await dispatch('rr_anchor_init_step');assert.deepEqual(bytes(await read(3)),bytes(before));const a=u32(await read(9));assert.deepEqual([...a.slice(proofAt,proofAt+4)],[1,sample,prefix,0]);assert.equal(a[11],0);assert.deepEqual([...a.slice(8,11)],[0,0,0]);assert.deepEqual([...a.slice(12,15)],[0,0,0]);assert.equal(a[checkedAt],0);
    await dispatch('rr_axis_prepare_step');await scan();await dispatch('rr_anchor_point_step');await dispatch('rr_axis_finish_step');await dispatch('rr_scatter_step');await dispatch('rr_traverse_step');await dispatch('rr_anchor_init_step');await dispatch('rr_axis_prepare_step');await scan();await dispatch('rr_anchor_point_step');await dispatch('rr_axis_finish_step');const completed=u32(await read(3));prefix=(prefix+2*0x6D2B79F5)>>>0;assert.equal(completed[1],prefix);assert.equal(completed[2],sample+1);report.sameRunSamples.push({sample,status:completed[3],flags:completed[11],reason:completed[111]});if(completed[3]===3){assert(isResidentReplayable(completed[11]),'sample must be safely replayable');assert.equal(completed[108],0);completed[3]=sample===2?2:0;completed[11]=0;write(3,completed);}else{assert.equal(completed[3],sample===2?2:0);assert.equal(completed[11],0);}
   }
  });
  await check('minimal CPU replay resume cannot reuse prior sample proof before anchor initialization',async()=>{
   const p=params.slice(0);u32(p)[9]=2;reset({...pointed,0:p});const failed=pointed[3].slice(0);u32(failed)[11]=268435458;write(3,failed);await dispatch('safety_finish');const resumed=await read(3);assert.equal(u32(resumed)[3],3);u32(resumed)[1]=345;u32(resumed)[3]=0;u32(resumed)[11]=0;write(3,resumed);await dispatch('rr_traverse_step');const lane=await read(3);assert.equal(u32(lane)[2],1);assert.equal(u32(lane)[17],345);await dispatch('rr_axis_prepare_step');assert.equal(u32(await read(8))[0],0,'stale pre-replay proof must fail');await dispatch('rr_anchor_init_step');assert.deepEqual(bytes(await read(3)),bytes(lane));assert.deepEqual([...u32(await read(9)).slice(proofAt,proofAt+4)],[1,1,345,0]);await dispatch('rr_axis_prepare_step');assert.equal(u32(await read(8))[0],1);
  });
  await check('missing init, stale sample/prefix/depth, absent validity, and bad lane marker are rejected by prepare',async()=>{
   for(const field of ['missing-init','sample','prefix','depth','proof-valid','anchor-valid','lane-marker']){reset(initialized);const a=initialized[9].slice(0),lane=initialized[3].slice(0);if(field==='missing-init')u32(a).fill(0);else if(field==='sample')u32(a)[proofAt+1]++;else if(field==='prefix')u32(a)[proofAt+2]++;else if(field==='depth')u32(a)[proofAt+3]++;else if(field==='proof-valid')u32(a)[proofAt]=0;else if(field==='anchor-valid')u32(a)[17]=0;else u32(lane)[108]=2;write(9,a);write(3,lane);await dispatch('rr_axis_prepare_step');const out=u32(await read(8));assert.equal(out[0],0,field);assert(out[1]&1,field);}
  });
  await check('ownership is rechecked independently at plane, point, and finish boundaries',async()=>{
   for(const offset of [proofAt+1,proofAt+2,proofAt+3])for(const stage of ['rr_axis_plane_step','rr_anchor_point_step','rr_axis_finish_step']){const base=stage==='rr_axis_plane_step'?prepared:stage==='rr_anchor_point_step'?scanned:pointed;reset(base);const a=base[9].slice(0);u32(a)[offset]++;write(9,a);await dispatch(stage);if(stage==='rr_axis_plane_step'){const out=u32(await read(8));assert.equal(out[29],2,'stale plane proof must be unknown');assert(out[30]&1);}else if(stage==='rr_anchor_point_step'){const out=u32(await read(9));assert.equal(out[resultAt],0);assert(out[resultAt+1]&1);assert.equal(out[checkedAt],1);}else rejected(await read(3),await read(9));}
  });
  await check('missing point and unchecked stale accepted result cannot cross finish barrier',async()=>{
   for(const marker of [0,2,0xffffffff]){reset(scanned);const a=scanned[9].slice(0);u32(a).set(u32(pointed[9]).slice(resultAt,resultAt+16),resultAt);u32(a)[checkedAt]=marker;write(9,a);await dispatch('rr_axis_finish_step');rejected(await read(3),await read(9));}
   reset(pointed);await dispatch('rr_axis_prepare_step');assert.equal(u32(await read(9))[checkedAt],0,'prepare clears stale checked marker');await scan();await dispatch('rr_axis_finish_step');rejected(await read(3),await read(9));
  });
  await check('point with missing or partial full scan rejects and cannot retain a reusable proof',async()=>{
   for(const count of [0,packed.metadata.triangles-1]){reset(prepared);for(let i=0;i<count;i++){await dispatch('rr_axis_plane_step');await dispatch('rr_axis_scan_step');}await dispatch('rr_anchor_point_step');assert.equal(u32(await read(9))[resultAt],0);await dispatch('rr_axis_finish_step');rejected(await read(3),await read(9));}
  });
  await check('pending anchor proof leaves records, operator, scatter, and contribution state immutable',async()=>{
   const lane=pending[3].slice(0),audit=pending[4].slice(0);u32(lane)[8]=1;u32(lane)[77]=1;u32(audit)[1243]=0xffffffff;f32(audit)[1237]=1;
   for(const stage of ['rr_records_step','rr_operator_step','rr_scatter_step']){reset({...pending,3:lane,4:audit});await dispatch(stage);for(const [binding,expected]of [[3,lane],[4,audit],[6,pending[6]],[7,pending[7]],[9,pending[9]]])assert.deepEqual(bytes(await read(binding)),bytes(expected),stage+' binding '+binding);assert.deepEqual([...u32(await read(5))],[stage==='rr_scatter_step'?1:0,0,0,0]);}
  });
  await check('replayed finish revokes lane proof and failed sample contribution, then sample init cannot inherit it',async()=>{
   for(const flags of [2,134217730,268435458]){reset(pointed);const lane=pointed[3].slice(0);u32(lane)[11]=flags;f32(lane)[48]=19;f32(lane)[49]=7;write(3,lane);await dispatch('safety_finish');const out=await read(3);assert.equal(u32(out)[108],0);assert.equal(u32(out)[3],3);assert.equal(f32(out)[49],7);assert.equal(u32(await read(4))[1545],0);await dispatch('rr_init');const a=u32(await read(9));assert.deepEqual([...a.slice(proofAt,proofAt+4)],[0,0,0,0]);assert.equal(a[checkedAt],0);}
  });
  await check('successful finish requires matching actual axis mirror scattering before continuation',async()=>{
   reset(finished);await dispatch('rr_scatter_step');const lane=await read(3),a=await read(9);assert.equal(u32(lane)[4],1);assert.equal(u32(lane)[108],3);assert.equal(u32(a)[proofAt+3],u32(lane)[4]);assert.equal(f32(lane)[26],-f32(pending[3])[26]);assert.equal(f32(lane)[24],f32(pending[3])[24]);assert.equal(f32(lane)[25],f32(pending[3])[25]);
   await dispatch('rr_traverse_step');const continued=await snapshot();assert.equal(u32(continued[3])[79],1);await dispatch('rr_anchor_init_step');assert.deepEqual(bytes(await read(9)),bytes(continued[9]),'later-depth init cannot replace anchor');await dispatch('rr_axis_prepare_step');await scan();await dispatch('rr_anchor_point_step');await dispatch('rr_axis_finish_step');assert.equal(u32(await read(3))[3],2,'reflected ray should certify escape');
   for(const kind of ['non-axis-reflection','null','FSD']){reset(finished);const altered=finished[3].slice(0);if(kind==='non-axis-reflection')f32(altered)[64]=.125;else{u32(altered)[53]=0;u32(altered)[76]=kind==='FSD'?1:0;}write(3,altered);await dispatch('rr_scatter_step');assert.equal(u32(await read(3))[108],0,kind);}
  });
  // Real mutations are compiled and dispatched. A positive witness must violate
  // the corresponding assertion, not merely differ from an expected string.
  for(const [name,clause,offset]of [['sample','state.proofMeta.y==s.header.z&&',proofAt+1],['prefix','state.proofMeta.z==s.progress.y&&',proofAt+2],['depth','state.proofMeta.w==s.counters.x&&',proofAt+3]]){
   const mutant=source.replace(clause,'');assert.notEqual(mutant,source);const mp=await compile(mutant,['rr_axis_prepare_step'],'remove-'+name+'-ownership');reset(initialized);const a=initialized[9].slice(0);u32(a)[offset]++;write(9,a);await dispatch('rr_axis_prepare_step',mp.rr_axis_prepare_step);assert.equal(u32(await read(8))[0],1,'ownership mutation must actually admit stale '+name);report.mutations.push({name:'remove-'+name+'-ownership',rejected:true,oracle:'native stale proof preparation'});
  }
  const unchecked=source.replace('anchorState.checked!=1u||','');assert.notEqual(unchecked,source);const cp=await compile(unchecked,['rr_axis_finish_step'],'remove-checked-barrier');reset(pointed);const uncheckedState=pointed[9].slice(0);u32(uncheckedState)[checkedAt]=0;write(9,uncheckedState);await dispatch('rr_axis_finish_step',cp.rr_axis_finish_step);assert.equal(u32(await read(3))[108],3);assert.equal(u32(await read(3))[79],0);report.mutations.push({name:'remove-point-checked-barrier',rejected:true,oracle:'native unchecked accepted result crosses finish'});
  const begin=source.indexOf('fn rr_operator_step('),end=source.indexOf('fn rr_scatter_step(',begin);assert(begin>=0&&end>begin);const original=source.slice(begin,end),altered=original.replace('s.vertexMeta.w!=0u||','');assert.notEqual(altered,original);const bypass=source.slice(0,begin)+altered+source.slice(end),bp=await compile(bypass,['rr_operator_step'],'remove-pending-barrier');const lane=pending[3].slice(0),audit=pending[4].slice(0);u32(lane)[8]=1;u32(lane)[77]=1;u32(audit)[1243]=0xffffffff;f32(audit)[1237]=1;reset({...pending,3:lane,4:audit});await dispatch('rr_operator_step',bp.rr_operator_step);assert.notDeepEqual(bytes(await read(4)),bytes(audit));report.mutations.push({name:'remove-operator-pending-barrier',rejected:true,oracle:'native pending audit immutability'});
  assert.equal(await device.popErrorScope(),null);report.actualGPUExecuted=true;
 }finally{for(const b of buffers)b.destroy();device.destroy();gpu=null;}
}
