/** Isolated native profiler. No production feature/default/shader changes.
 * WEBGPU_EXECUTION_BACKEND=vulkan VK_ICD_FILENAMES=/usr/lib/chromium/vk_swiftshader_icd.json node tests/plt-resident-stage-profile.mjs --execute
 * Timestamp readback happens after the measured renderer run. Each stage retains
 * its own production compute pass and every result needs the unchanged shadow.
 */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {create,globals} from 'webgpu';
import {createPLTRandomStates} from '../dist/plt-core.js';
import {createPLTResidentRunner,PLT_RESIDENT_BUILD_ID} from '../dist/plt-resident-gpu.js';
import {buildResidentReferenceBatch} from '../dist/plt-resident-reference.js';
import {fixtures,options,hash,sourceProvenance,validateCandidate,reuseNativeDevice} from './plt-resident-camera-transport.mjs';
Object.assign(globalThis,globals);
const execute=process.argv.includes('--execute'),backend=execute?process.env.WEBGPU_EXECUTION_BACKEND:'null';
if(execute)assert(['vulkan','metal','d3d11','d3d12','opengl','opengles'].includes(backend));
const names=['two-mirrors-emission','four-scale-shifted-zxy','half-scale-three-mirrors-negative-permutation','eight-scale-two-mirrors'];
const selected=process.env.PLT_PROFILE_CASES?.split(',')??names;assert(selected.every(n=>names.includes(n)));
const queryCount=4096,measuredRepeats=4,warmups=2,provenance=sourceProvenance();
const report={buildId:PLT_RESIDENT_BUILD_ID,baselineCommit:'0b6afb30408dd757330eaa86cc340e84345aa203',allPass:false,status:'preparing',executionRequested:execute,actualGPUExecuted:false,backend,nodeVersion:process.version,vulkanICD:process.env.VK_ICD_FILENAMES??null,physicalGPUValidated:false,browserValidated:false,hardwarePerformanceClaim:false,descriptiveOnly:true,fullCPUShadowRequired:true,queryReadbackOutsideRenderer:true,sourceSHA256:provenance.sourceSHA256,hostSHA256:provenance.hostSHA256,testSHA256:hash(fs.readFileSync(new URL(import.meta.url))),fixtureHelperSHA256:hash(fs.readFileSync(new URL('./plt-resident-camera-transport.mjs',import.meta.url))),timestampQueryCount:queryCount,warmupsPerTreatment:warmups,measuredRepeatsPerTreatment:measuredRepeats,scenarios:[],setup:[],limitations:['Native Dawn/Vulkan SwiftShader timestamps measure software-backend command intervals, not physical GPU hardware.','Timestamp writes perturb execution; paired timestamp-off runs measure the direction/size of instrumentation perturbation.','All stage passes, barriers, round command buffers, and safety contracts remain unchanged.','Shared native device/pipeline cache is test-only; uncached first compilation and subsequent cache hits are separate.','Component medians are not additive; full CPU shadow construction and post-render validation are separate.','The 64-pixel historical workload cannot distinguish dispatchLanes=64 from 128. A separately labelled 256-pixel square workload exercises both bounds without changing scene/domain.']};
const suffix=process.env.PLT_PROFILE_CASES?'-'+selected.join('--'):'';
const output=new URL(`../research/plt-resident-stage-profile${suffix}-${execute?'validation':'compile-validation'}.json`,import.meta.url);
const save=()=>fs.writeFileSync(output,JSON.stringify(report,null,2)+'\n');
const forward=(target,key)=>{const value=Reflect.get(target,key,target);return typeof value==='function'?value.bind(target):value;};
const sum=a=>a.reduce((n,v)=>n+v,0);
const stats=a=>{assert(a.length&&a.every(Number.isFinite));const s=[...a].sort((a,b)=>a-b);return {count:s.length,min:s[0],median:(s[(s.length-1)>>1]+s[s.length>>1])/2,max:s.at(-1),mean:sum(s)/s.length};};
function sameTransport(a,b){for(const key of ['audit','image','pixelIndices','finalRngStates'])assert.deepEqual(a[key],b[key],key);assert.deepEqual(a.recovery.keys,b.recovery.keys);assert.equal(a.recovery.paths,b.recovery.paths);assert.equal(a.rounds,b.rounds);}
let native=create([`backend=${backend}`]),timestampSupported=false;
const featureGPU={async requestAdapter(options){const a=await native.requestAdapter(options);assert(a);timestampSupported=a.features.has('timestamp-query');report.adapter={...Object.fromEntries(['vendor','architecture','device','description','isFallbackAdapter'].map(k=>[k,a.info[k]])),features:[...a.features]};report.timestampSupported=timestampSupported;return new Proxy(a,{get(t,k){if(k==='requestDevice')return async()=>{const began=performance.now(),d=await t.requestDevice({requiredFeatures:timestampSupported?['timestamp-query']:[]});report.nativeDeviceRequestWallMs=performance.now()-began;report.enabledDeviceFeatures=[...d.features];return d;};return forward(t,k);}});}};
const cache=reuseNativeDevice(featureGPU);report.nativeDeviceReuse=cache.stats;
function instrumentGPU(gpu){
 let current=null,device=null,querySet=null,resolveBuffer=null,readback=null;
 const pipelines=new WeakMap(),commands=new WeakMap(),compilation=[];
 const profile={compilation,
  start(timestamp){assert.equal(current,null);current={timestamp:timestamp&&timestampSupported,passes:[],encoders:[],submits:[],waits:[],nextQuery:0};},
  async finish(){const p=current;assert(p);current=null;let raw=[];const started=performance.now();
   if(p.timestamp&&p.nextQuery){device.pushErrorScope('validation');const enc=device.createCommandEncoder({label:'Profiler query readback outside renderer'});enc.resolveQuerySet(querySet,0,p.nextQuery,resolveBuffer,0);enc.copyBufferToBuffer(resolveBuffer,0,readback,0,p.nextQuery*8);device.queue.submit([enc.finish()]);await readback.mapAsync(GPUMapMode.READ,0,p.nextQuery*8);raw=[...new BigUint64Array(readback.getMappedRange(0,p.nextQuery*8).slice(0))];readback.unmap();const error=await device.popErrorScope();assert.equal(error,null,error?.message);}
   const queryReadbackWallMs=performance.now()-started;
   const perStage={};for(const pass of p.passes){const stage=perStage[pass.entryPoint]??={passes:0,workgroups:0,timestampMs:0};stage.passes++;stage.workgroups+=pass.workgroups;
    if(p.timestamp){const begin=raw[pass.query],end=raw[pass.query+1];assert(begin>0n&&end>=begin,'Timestamp query must produce nonnegative real values');pass.timestampBeginNs=String(begin);pass.timestampEndNs=String(end);pass.timestampMs=Number(end-begin)/1e6;stage.timestampMs+=pass.timestampMs;}}
   let spanMs=0,passMs=0;for(const enc of p.encoders){if(p.timestamp&&enc.passes.length){spanMs+=Number(raw[enc.passes.at(-1).query+1]-raw[enc.passes[0].query])/1e6;passMs+=sum(enc.passes.map(x=>x.timestampMs));}}
   const totals={encoderCount:p.encoders.length,computeEncoderCount:p.encoders.filter(e=>e.passes.length).length,readbackEncoderCount:p.encoders.filter(e=>e.copyCount&&!e.passes.length).length,passes:p.passes.length,encodeWallMs:sum(p.encoders.map(e=>e.encodeWallMs)),queueSubmitCpuMs:sum(p.submits.map(e=>e.cpuMs)),queueWaitWallMs:sum(p.waits.map(e=>e.wallMs)),queueWaitCount:p.waits.length,queryReadbackWallMs,timestampPassTotalMs:p.timestamp?passMs:null,timestampCommandSpansMs:p.timestamp?spanMs:null,timestampBetweenPassGapsMs:p.timestamp?spanMs-passMs:null};
   return {...p,nextQuery:undefined,perStage,totals};},
  destroy(){for(const resource of [querySet,resolveBuffer,readback])resource?.destroy();querySet=resolveBuffer=readback=null;},
  gpu:{async requestAdapter(options){const a=await gpu.requestAdapter(options);return new Proxy(a,{get(t,k){if(k==='requestDevice')return async()=>{device=await t.requestDevice();
    if(timestampSupported){querySet=device.createQuerySet({type:'timestamp',count:queryCount});resolveBuffer=device.createBuffer({size:queryCount*8,usage:GPUBufferUsage.QUERY_RESOLVE|GPUBufferUsage.COPY_SRC});readback=device.createBuffer({size:queryCount*8,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});}
    const queue=new Proxy(device.queue,{get(q,key){if(key==='submit')return buffers=>{const began=performance.now();q.submit(buffers);if(current)current.submits.push({cpuMs:performance.now()-began,encoderIds:buffers.map(b=>commands.get(b)?.id??null)});};if(key==='onSubmittedWorkDone')return ()=>{const began=performance.now(),captured=current;return q.onSubmittedWorkDone().then(value=>{captured?.waits.push({wallMs:performance.now()-began});return value;});};return forward(q,key);}});
    return new Proxy(device,{get(d,key){if(key==='queue')return queue;
     if(key==='createComputePipelineAsync')return async spec=>{const began=performance.now(),pipeline=await d.createComputePipelineAsync(spec);compilation.push({entryPoint:spec.compute.entryPoint,wallMs:performance.now()-began});pipelines.set(pipeline,spec.compute.entryPoint);return pipeline;};
     if(key==='createCommandEncoder')return descriptor=>{const began=performance.now(),enc=d.createCommandEncoder(descriptor),captured=current,record={id:captured?.encoders.length??-1,passes:[],copyCount:0};captured?.encoders.push(record);
      return new Proxy(enc,{get(e,name){if(name==='beginComputePass')return spec=>{const passRecord={entryPoint:null,workgroups:0,query:null};if(captured?.timestamp){assert(captured.nextQuery+2<=queryCount,'Profiler query capacity');passRecord.query=captured.nextQuery;captured.nextQuery+=2;spec={...spec,timestampWrites:{querySet,beginningOfPassWriteIndex:passRecord.query,endOfPassWriteIndex:passRecord.query+1}};}const pass=e.beginComputePass(spec);record.passes.push(passRecord);captured?.passes.push(passRecord);return new Proxy(pass,{get(p,k){if(k==='setPipeline')return pipeline=>{passRecord.entryPoint=pipelines.get(pipeline);assert(passRecord.entryPoint);p.setPipeline(pipeline);};if(k==='dispatchWorkgroups')return (...args)=>{passRecord.workgroups+=args[0];return p.dispatchWorkgroups(...args);};return forward(p,k);}});};if(name==='copyBufferToBuffer')return (...args)=>{record.copyCount++;return e.copyBufferToBuffer(...args);};if(name==='finish')return (...args)=>{const buffer=e.finish(...args);record.encodeWallMs=performance.now()-began;commands.set(buffer,record);return buffer;};return forward(e,name);}});};return forward(d,key);}});
   };return forward(t,k);}});}}
 };return profile;
}
report.status='compiling';save();
try{
 for(const width of process.argv.includes('--historical-only')?[8]:[8,16])for(const name of selected){
  // All four original scenes at both fixed workload sizes; same square aspect.
  const f=fixtures.find(f=>f.name===name),o={...options,width,height:width,samplesPerPixel:2},tile={x:0,y:0,width,height:width},initial=createPLTRandomStates(o),cpu=initial.slice(),shadowStart=performance.now();
  const reference=await buildResidentReferenceBatch(f.scene,f.camera,o,{tile,sampleStart:0,samplesPerPixel:2,rngStates:cpu}),b={seed:o.seed,options:o,tile,sampleStart:0,samplesPerPixel:2,start:initial,end:cpu,reference};
  const scenario={name,width,height:width,pathsPerRun:width*width*2,sceneTriangles:f.scene.triangles.length,cpuFullShadowConstructionMs:performance.now()-shadowStart,runs:[],summary:{}};report.scenarios.push(scenario);const canonical=new Map();
  for(const dispatchLanes of [32,64,128]){
   const profiler=instrumentGPU(cache.gpu),setupStart=performance.now(),runner=await createPLTResidentRunner(f.scene,f.camera,o,{gpu:profiler.gpu,shaderSource:provenance.sources.join('\n'),timeoutMs:120000,dispatchLanes,recoveryMode:'selective',continuationGuard:'axis-mirror-camera',onSetupProgress:step=>console.log('COMPILE',name,width,dispatchLanes,step.entryPoint)});
   report.setup.push({name,width,dispatchLanes,wallMs:performance.now()-setupStart,compilation:profiler.compilation});save();
   try{if(execute){
    // Four balanced pairs per chunk: timestamp instrumentation off/on. This
    // isolates instrumentation, while chunk treatment order is explicitly fixed.
    for(let iteration=-warmups;iteration<measuredRepeats;iteration++)for(const timestamps of (iteration%2?[true,false]:[false,true])){
     const mode={id:'camera-round',guard:'axis-mirror-camera',axisSubmission:'round'},rng=initial.slice();profiler.start(timestamps);const began=performance.now();
     const progress=[];const candidate=await runner.run({tile,sampleStart:0,samplesPerPixel:2,wave:false,rngStates:rng,guard:mode.guard,axisSubmission:'round',onProgress:value=>progress.push({...value})}),observedRunWallMs=performance.now()-began,profile=await profiler.finish();report.actualGPUExecuted=true;report.status='measuring';assert.deepEqual(rng,initial,'Caller RNG immutable');
     const validation=validateCandidate(f,b,candidate,mode);const existing=canonical.get(mode.guard);if(existing)sameTransport(existing,candidate);else canonical.set(mode.guard,candidate);
     const result={progress,phase:iteration<0?'warmup':'measured',iteration,dispatchLanes,timestamps:profile.timestamp,observedRunWallMs,...validation,exactAcrossChunkAuditRngReplayParity:true,profile};scenario.runs.push(result);save();
     console.log('PASS',name,width,dispatchLanes,timestamps,iteration,observedRunWallMs.toFixed(3),profile.totals.timestampPassTotalMs?.toFixed(3));
    }
   }}finally{profiler.destroy();runner.destroy();}
   if(!execute)break;
  }
  if(execute)for(const lanes of [32,64,128])for(const timestamps of [false,true]){const runs=scenario.runs.filter(r=>r.phase==='measured'&&r.dispatchLanes===lanes&&r.timestamps===timestamps);if(!runs.length)continue;scenario.summary[`${lanes}/${timestamps?'timestamp-on':'timestamp-off'}`]={count:runs.length,dispatches:runs[0].dispatches,rounds:runs[0].rounds,replayPaths:runs[0].replayPaths,retainedMirrorPaths:runs[0].retainedMirrorPaths,rendererWallMs:stats(runs.map(r=>r.timings.gpuResidentTotalWallMs)),observedRunWallMs:stats(runs.map(r=>r.observedRunWallMs)),host:Object.fromEntries(['gpuRoundSubmitAndWaitMs','gpuStatusReadbackWallMs','cpuSelectiveReplayMs','replayReadbackWallMs','gpuFinalReadbackDecodeMs'].map(k=>[k,stats(runs.map(r=>r.timings[k]))])),instrumentation:Object.fromEntries(Object.keys(runs[0].profile.totals).filter(k=>runs[0].profile.totals[k]!==null).map(k=>[k,stats(runs.map(r=>r.profile.totals[k]))])),perStage:timestamps?Object.fromEntries(Object.keys(runs[0].profile.perStage).map(stage=>[stage,{passes:runs[0].profile.perStage[stage].passes,workgroups:runs[0].profile.perStage[stage].workgroups,timestampMs:stats(runs.map(r=>r.profile.perStage[stage].timestampMs))}])):null};}save();
  if(!execute)break;
 }
 assert.deepEqual(sourceProvenance().sourceSHA256,report.sourceSHA256,'Shaders changed while profiling');assert.deepEqual(sourceProvenance().hostSHA256,report.hostSHA256,'Host changed while profiling');assert.equal(hash(fs.readFileSync(new URL(import.meta.url))),report.testSHA256);report.allPass=true;report.status='passed';
}catch(error){report.status='failed';report.failure={name:error.name,message:error.message,stack:error.stack};throw error;}
finally{try{cache.destroy();}finally{native=null;save();}}
console.log(JSON.stringify({allPass:report.allPass,actualGPUExecuted:report.actualGPUExecuted,timestampSupported:report.timestampSupported,scenarios:report.scenarios.length,output:output.pathname}));
