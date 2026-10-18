/** Whole-round scheduler host ordering, uniform ownership, and lifecycle safety.
 * Host mocks only; no shader compilation or actual GPU execution is claimed.
 * Native scheduling parity and mutations have a separate executable suite.
 */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import crypto from 'node:crypto';
import {createPLTRandomStates} from '../dist/plt-core.js';
import {preparePLTScene} from '../dist/plt-geometry.js';
import {createPLTResidentRunner,PLT_RESIDENT_LIMITS,PLT_RESIDENT_BUILD_ID} from '../dist/plt-resident-gpu.js';

globalThis.GPUBufferUsage={MAP_READ:1,COPY_SRC:4,COPY_DST:8,UNIFORM:64,STORAGE:128};
globalThis.GPUMapMode={READ:1};
const execute=false,hostOnly=true;
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
const groupName=commands=>commands.length===1?commands[0]:commands[0]==='rr_traverse_step'?'lane-round':['rr_axis_prepare_step','rr_anchor_init_step'].includes(commands[0])?'axis-guard-group':'axis-triangle-loop';
function expectedSubmissions(mode,n,axis=true,anchor=true){
 const guard=mode==='batched'?[[...(anchor?['rr_anchor_init_step']:[]),'rr_axis_prepare_step',...pairs(n),...(anchor?['rr_anchor_point_step']:[]),'rr_axis_finish_step']]:[...(anchor?[['rr_anchor_init_step']]:[]),['rr_axis_prepare_step'],pairs(n),...(anchor?[['rr_anchor_point_step']]:[]),['rr_axis_finish_step']];
 const all=[['rr_traverse_step'],...(axis?guard:[]),['rr_records_step'],['rr_operator_step'],['rr_scatter_step']];return mode==='round'?[['rr_init'],all.flat()]:[['rr_init'],...all];
}
const report={createdAt:new Date().toISOString(),buildId:PLT_RESIDENT_BUILD_ID,executionRequested:execute,actualGPUExecuted:false,allPass:false,status:'running',hostMockOnly:hostOnly,fullCPUShadowMandatory:true,physicalGPUValidated:false,browserValidated:false,hardwarePerformanceClaim:false,hostChecks:[],nativeChecks:[],mutations:[]};
const reportURL=new URL('../research/plt-resident-round-scheduling-host'+(execute?'':'-compile')+'-validation.json',import.meta.url);
const save=()=>fs.writeFileSync(reportURL,JSON.stringify(report,null,2)+'\n');
async function test(name,fn){await fn();report.hostChecks.push(name);console.log('PASS host',name);}

function mockGPU(config={}){
 const buffers=[],compiled=[],submissions=[],parameterSnapshots=[],events=[],listeners=new Map(),lost=deferred();
 let outstanding=false;let initial=new Uint32Array(4),uniform=new Uint32Array(40),scopes=0,destroyed=0,reads=0,currentStage,encoders=0,passCount=0;
 const device={limits:{maxBufferSize:1e9,maxStorageBufferBindingSize:1e9},lost:lost.promise,
  addEventListener:(name,fn)=>listeners.set(name,fn),removeEventListener:name=>listeners.delete(name),
  pushErrorScope(){scopes++;},async popErrorScope(){assert(scopes>0);scopes--;return null;},
  createShaderModule:()=>({getCompilationInfo:async()=>({messages:[]})}),
  async createComputePipelineAsync({compute}){compiled.push(compute.entryPoint);assert.equal(compute.constants.RR_AXIS_INTERVAL_ENABLED,config.baseline?0:1);assert.equal(compute.constants.RR_ANCHOR_ENABLED,config.baseline||config.axisOnly?0:1);if('RR_TIGHT_ENABLED'in compute.constants)assert.equal(compute.constants.RR_TIGHT_ENABLED,config.tight?1:0);if(config.pipelineFailure===logicalName(compute.entryPoint))throw new Error('mock pipeline failure');return {...compute,entryPoint:logicalName(compute.entryPoint),getBindGroupLayout:()=>({entryPoint:logicalName(compute.entryPoint)})};},
  createBindGroup(spec){assert(spec.entries.every(e=>e.resource.buffer));const anchored=!config.baseline&&!config.axisOnly,expected=anchored&&['rr_init','rr_anchor_init_step','rr_anchor_point_step','rr_axis_prepare_step','rr_axis_plane_step','rr_axis_finish_step'].includes(spec.layout.entryPoint);assert.equal(spec.entries.some(e=>e.binding===9),expected,'anchor state binding '+spec.layout.entryPoint);return spec;},
  createBuffer(spec){if(config.bufferFailure===spec.label)throw new Error('mock allocation failure');const buffer={...spec,destroyed:0,mapped:false,destroy(){this.destroyed++;this.mapped=false;},async mapAsync(){assert.equal(destroyed,0);this.mapped=true;},unmap(){this.mapped=false;},getMappedRange(_offset,bytes){assert(this.mapped&&!this.destroyed);reads++;const data=new ArrayBuffer(bytes),u=new Uint32Array(data),f=new Float32Array(data),pixels=uniform[6]*uniform[7],samples=uniform[9];
   if(spec.label==='Resident lane readback')for(let lane=0;lane<pixels;lane++)u.set([initial[lane*4],initial[lane*4+1],uniform[8]+samples,2],lane*PLT_RESIDENT_LIMITS.laneBytes/4);
   if(spec.label==='Resident audit readback')for(let lane=0;lane<pixels;lane++)for(let sample=0;sample<samples;sample++){const at=(lane*samples+sample)*PLT_RESIDENT_LIMITS.auditVectors*4;u.set([initial[lane*4],uniform[8]+sample,uniform[11],0,initial[lane*4+1],initial[lane*4+1],0,2166136261,1,0,0,1],at);u[at+81]=0xffffffff;f[at+90]=-1;u[at+1546]=3;}
   return data;
  }};buffers.push(buffer);return buffer;},
  createCommandEncoder(){const id=++encoders,commands=[],passes=[];let open=false,copy=false;return {beginComputePass(){assert(!open,'Dependent dispatches require closed preceding passes');open=true;let stage,dispatched=false;const pass={id:++passCount,stage:null,ended:false};passes.push(pass);return {setPipeline:p=>{stage=p.entryPoint;},setBindGroup(){},dispatchWorkgroups(count){assert(count>=1);assert(!dispatched,'One dependency stage per compute pass');dispatched=true;commands.push(stage);pass.stage=stage;},end(){assert(open&&dispatched);pass.ended=true;open=false;}};},copyBufferToBuffer(){assert(!open);copy=true;},finish(){assert(!open);assert(passes.every(p=>p.ended));return {id,commands,passes,copy};}};},
  queue:{writeBuffer(buffer,offset,data){assert.equal(destroyed,0,'Upload after destruction');assert(offset+data.byteLength<=buffer.size);const words=new Uint32Array(data.buffer??data,data.byteOffset??0,data.byteLength/4);if(buffer.label==='Resident parameters'){assert(!outstanding,'Shared dispatch uniform rewritten while prior chunk is outstanding');events.push({kind:'uniform',offset,words:[...words]});uniform.set(words,offset/4);if(offset===0)parameterSnapshots.push(uniform.slice());}if(buffer.label==='Resident initial prefixes')initial=words.slice();},
   submit(commandBuffers){assert.equal(destroyed,0,'Submission after destruction');assert.equal(commandBuffers.length,1,'Each group must be one command buffer');for(const cb of commandBuffers)if(!cb.copy){assert(!outstanding,'Second chunk submitted before awaiting prior chunk');outstanding=true;submissions.push(cb.commands.slice());currentStage=groupName(cb.commands);events.push({kind:'submit',name:currentStage,id:cb.id,passes:cb.passes.map(p=>p.stage),uniform:[...uniform]});config.onSubmit?.(currentStage);}else events.push({kind:'readback'});},
   async onSubmittedWorkDone(){const stage=currentStage;events.push({kind:'wait',name:stage});config.onWork?.(stage);if(config.stallStage===stage)await never();if(config.holdStage===stage)await config.hold;outstanding=false;events.push({kind:'done',name:stage});},
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
await test('opt-in round exactly preserves staged and batched pass order and candidate-only admission',()=>withRunner({},async(runner,mock)=>{
 for(const guard of ['axis-mirror-anchor','axis-mirror-unit','axis-mirror-interval','replay-unmodelled','audit-only'])for(const mode of [undefined,'staged','batched','round']){
  const input={...fresh(),guard,...(mode?{axisSubmission:mode}:{})},before=input.rngStates.slice(),start=mock.submissions.length,result=await runner.run(input),actual=mode??'staged',axis=guard.startsWith('axis-');
  assertCandidate(result,input,before);const expected=expectedSubmissions(actual,2,axis,guard==='axis-mirror-anchor');assert.deepEqual(mock.submissions.slice(start),expected);assert.equal(result.dispatches,expected.flat().length-1);assert.equal(result.recovery.axisSubmission,actual);
  assert(result.timings.gpuRoundSubmitAndWaitMs>=0);if(actual!=='round')assert.equal(result.timings.gpuRoundSubmitAndWaitMs,0);
  if(actual==='round')for(const key of ['gpuTraversalSubmitAndWaitMs','gpuRecordsSubmitAndWaitMs','gpuOperatorSubmitAndWaitMs','gpuScatterSubmitAndWaitMs','gpuAxisGuardSubmitAndWaitMs','gpuAxisTriangleLoopSubmitAndWaitMs','gpuAnchorInitSubmitAndWaitMs','gpuAnchorPointSubmitAndWaitMs'])assert.equal(result.timings[key],0,'Grouped cost cannot be misreported as separately measured '+key);
 }
 assertBarrierEvents(mock);
}));
await test('every bounded chunk owns one unchanged uniform across all dependent passes',async()=>{
 for(const dispatchLanes of [1,17,32,65,128])await withRunner({},async(runner,mock)=>{
  const input={...fresh(),tile:{x:0,y:0,width:8,height:8},axisSubmission:'round'},result=await runner.run(input),chunks=Math.ceil(64/dispatchLanes),events=mock.events.filter(e=>e.kind==='submit');
  assert.equal(events.length,1+chunks);assert.deepEqual(events[0].passes,['rr_init']);
  for(let chunk=0;chunk<chunks;chunk++){const e=events[chunk+1];assert.deepEqual(e.passes,expectedSubmissions('round',2)[1]);assert.deepEqual(e.uniform.slice(12,16),[chunk*dispatchLanes,Math.min(dispatchLanes,64-chunk*dispatchLanes),PLT_RESIDENT_LIMITS.auditVectors,0]);}
  assert.equal(result.dispatches,chunks*expectedSubmissions('round',2)[1].length);assertBarrierEvents(mock);
 },{dispatchLanes});
});
await test('round grouping retains the 64-triangle ceiling and never scans unsupported geometry',async()=>{
 for(const quads of [1,32,33])await withRunner({sceneInput:makeScene(quads)},async(runner,mock)=>{const result=await runner.run({...fresh(),axisSubmission:'round'});assert.deepEqual(mock.submissions,expectedSubmissions('round',quads<=32?quads*2:0));assert.equal(result.dispatches,8+(quads<=32?quads*4:0));});
});
await test('one whole submitted chunk cannot be retracted but cancellation suppresses later chunks and readbacks',async()=>{
 for(const timing of ['submit','sync-wait','timer-wait']){const controller=new AbortController(),abort=name=>{if(name==='lane-round'){if(timing==='timer-wait')setTimeout(()=>controller.abort(),0);else controller.abort();}};
  await withRunner({onSubmit:timing==='submit'?abort:undefined,onWork:timing==='submit'?undefined:abort},async(runner,mock)=>{const input={...fresh(),tile:{x:0,y:0,width:8,height:8},axisSubmission:'round',signal:controller.signal},before=input.rngStates.slice();await assert.rejects(runner.run(input),{name:'AbortError'});assert.deepEqual(mock.submissions,expectedSubmissions('round',2));assert.equal(mock.reads,0);assert.deepEqual(input.rngStates,before);await assert.rejects(runner.run(fresh()),{name:'AbortError'});});
 }
});
await test('in-flight round cancellation, loss, disposal, and timeout close ownership exactly once',async()=>{
 for(const method of ['cancel','owner-cancel','lost','destroy','timeout']){const entered=deferred(),held=deferred(),controller=new AbortController(),owner=new AbortController();
  await withRunner({holdStage:'lane-round',hold:held.promise,onWork:name=>{if(name==='lane-round')entered.resolve();}},async(runner,mock)=>{const input={...fresh(),axisSubmission:'round',signal:controller.signal},before=input.rngStates.slice(),pending=runner.run(input),rejected=assert.rejects(pending,method==='lost'?/mock device lost/:method==='timeout'?/lane-round timeout/:{name:'AbortError'});await entered.promise;if(method==='cancel')controller.abort();if(method==='owner-cancel')owner.abort();if(method==='lost')mock.lose();if(method==='destroy'){runner.destroy();runner.destroy();}await rejected;held.resolve();await delay();assert.deepEqual(mock.submissions,expectedSubmissions('round',2));assert.equal(mock.reads,0);assert.deepEqual(input.rngStates,before);},{signal:owner.signal,timeoutMs:method==='timeout'?30:1000});
 }
});
await test('round submission remains opt-in for legacy five-pipeline and nine-pipeline runners',async()=>{
 for(const [guard,config]of [['replay-unmodelled',{baseline:true}],['axis-mirror-unit',{axisOnly:true}],['axis-mirror-interval',{axisOnly:true}]])await withRunner(config,async(runner,mock)=>{const result=await runner.run({...fresh(),axisSubmission:'round'});assertCandidate(result,fresh(),fresh().rngStates);assert.deepEqual(mock.submissions,expectedSubmissions('round',2,guard.startsWith('axis-'),false));assert.equal(mock.compiled.length,guard==='replay-unmodelled'?5:9);assert(!mock.buffers.some(b=>b.label==='Resident anchored GPU proof state'));},{continuationGuard:guard});
 await withRunner({},async(runner,mock)=>{await assert.rejects(runner.run({...fresh(),axisSubmission:'unchecked'}),/invalid axis submission mode/);assert.equal(mock.submissions.length,0);});
});
await test('round scheduling cannot enable selective recovery or change strict configuration',()=>withRunner({},async(runner,mock)=>{
 const result=await runner.run({...fresh(),axisSubmission:'round'});assert.equal(result.recovery.mode,'strict');assert.equal(result.recovery.paths,0);assert.equal(result.admitted,false);assert.equal(mock.parameterSnapshots.at(-1)[37],0);assert.equal(mock.parameterSnapshots.at(-1)[38],0);assert.deepEqual(mock.submissions,expectedSubmissions('round',2));
},{recoveryMode:'strict'}));
await test('tight guard uses the same bounded round order only when explicitly compiled',async()=>{
 await withRunner({tight:true},async(runner,mock)=>{for(const mode of ['staged','batched','round']){const start=mock.submissions.length,result=await runner.run({...fresh(),guard:'axis-mirror-tight',axisSubmission:mode});assert.equal(result.recovery.continuationGuard,'axis-mirror-tight');assert.deepEqual(mock.submissions.slice(start),expectedSubmissions(mode,2));assert.equal(mock.parameterSnapshots.at(-1)[38],8);}assertBarrierEvents(mock);},{continuationGuard:'axis-mirror-tight'});
 await withRunner({},async(runner,mock)=>{await assert.rejects(runner.run({...fresh(),guard:'axis-mirror-tight',axisSubmission:'round'}),/tight guard was not compiled/);assert.equal(mock.submissions.length,0);});
});
await test('runner reuse resets initialization across repeated sample windows without changing caller RNG',()=>withRunner({},async(runner,mock)=>{
 for(const [sampleStart,samplesPerPixel]of [[0,1],[1,3],[0,4],[2,1]]){const input={...fresh(),sampleStart,samplesPerPixel,axisSubmission:'round'},before=input.rngStates.slice(),start=mock.submissions.length,result=await runner.run(input);assertCandidate(result,input,before);assert.deepEqual(mock.submissions.slice(start),expectedSubmissions('round',2));assert.equal(mock.parameterSnapshots.at(-1)[8],sampleStart);assert.equal(mock.parameterSnapshots.at(-1)[9],samplesPerPixel);}
}));
await test('actual host command-buffer omission and dependency reorder mutations fail the independent ordering oracle',async()=>{
 const url=new URL('../dist/plt-resident-gpu.js',import.meta.url),source=fs.readFileSync(url,'utf8'),start=source.indexOf('    const dispatchRound=async count=>{'),end=source.indexOf('    let rounds=0,dispatches=0',start),segment=source.slice(start,end);assert(start>0&&end>start);
 const mutations=[...expectedSubmissions('round',2)[1].filter((n,i,a)=>a.indexOf(n)===i).map(name=>({name:'omit-'+name,change:s=>s.replace(`encode('${name}');`,";/* deliberately omit dependency */")})),
  {name:'scan-before-plane',change:s=>s.replace("encode('rr_axis_plane_step');encode('rr_axis_scan_step');","encode('rr_axis_scan_step');encode('rr_axis_plane_step');")},
  {name:'scatter-before-operator',change:s=>s.replace("encode('rr_operator_step');encode('rr_scatter_step');","encode('rr_scatter_step');encode('rr_operator_step');")},
  {name:'finish-before-point',change:s=>s.replace(/(if\([^;\n]+\)encode\('rr_anchor_point_step'\);)\n      encode\('rr_axis_finish_step'\);/,"encode('rr_axis_finish_step');\n      $1")},
 ];
 for(const mutation of mutations){const altered=mutation.change(segment);assert.notEqual(altered,segment,mutation.name);const changed=source.slice(0,start)+altered+source.slice(end),absolute=changed.replace(/(['"])(\.\/[a-z0-9-]+\.js)\1/g,(_,quote,path)=>quote+new URL(path,url).href+quote),factory=(await import('data:text/javascript;base64,'+Buffer.from(absolute).toString('base64'))).createPLTResidentRunner;
  await withRunner({},async(runner,mock)=>{await runner.run({...fresh(),axisSubmission:'round'});assert.throws(()=>assert.deepEqual(mock.submissions,expectedSubmissions('round',2)),{code:'ERR_ASSERTION'});},{},factory);report.mutations.push({name:mutation.name,rejected:true,oracle:'actual command-buffer pass order'});
 }
});
report.hostSHA256=crypto.createHash('sha256').update(fs.readFileSync(new URL('../dist/plt-resident-gpu.js',import.meta.url))).digest('hex');report.testSHA256=crypto.createHash('sha256').update(fs.readFileSync(new URL(import.meta.url))).digest('hex');report.allPass=true;report.status='passed';
}catch(error){report.status='failed';report.failure={name:error.name,message:error.message,stack:error.stack};throw error;}finally{save();console.log(JSON.stringify(report,null,2));}
