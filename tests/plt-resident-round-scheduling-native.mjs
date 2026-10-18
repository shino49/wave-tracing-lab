/** Whole-round queue grouping: actual native complete-shadow parity and executable
 * command-buffer omission/reorder mutations. --timing also runs warmed probes,
 * which must be run only when all other native workloads are idle.
 * WEBGPU_EXECUTION_BACKEND=vulkan VK_ICD_FILENAMES=/usr/lib/chromium/vk_swiftshader_icd.json
 * node tests/plt-resident-round-scheduling-native.mjs --execute [--timing]
 * Null mode compiles only. SwiftShader is not physical GPU/browser evidence.
 */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import crypto from 'node:crypto';
import {create,globals} from 'webgpu';
import {preparePLTScene} from '../dist/plt-geometry.js';
import {createPLTRandomStates} from '../dist/plt-core.js';
import {createPLTResidentRunner,PLT_RESIDENT_BUILD_ID} from '../dist/plt-resident-gpu.js';
import {buildResidentReferenceBatch,compareResidentAudit} from '../dist/plt-resident-reference.js';
import {admitResidentWithRecovery} from '../dist/plt-resident-recovery.js';
Object.assign(globalThis,globals);
const execute=process.argv.includes('--execute'),timing=process.argv.includes('--timing'),tight=process.argv.includes('--tight'),backend=execute?process.env.WEBGPU_EXECUTION_BACKEND:'null';
assert(!timing||execute);if(execute)assert(['vulkan','metal','d3d11','d3d12','opengl','opengles'].includes(backend),'Explicit executable backend required');
function quad(axis,position,lo,hi,materialIndex){const a=(axis+1)%3,b=(axis+2)%3,point=(u,v)=>{const p=[0,0,0];p[axis]=position;p[a]=u;p[b]=v;return p;};return {mesh:{vertices:[point(lo,lo),point(hi,lo),point(hi,hi),point(lo,hi)],faces:[[0,1,2],[0,2,3]]},materialIndex};}
const scene=preparePLTScene([quad(0,2,-8,8,0),quad(2,3,-8,8,1),quad(2,-2,-16,16,2)],{materials:[{type:'mirror',albedo:.5,emission:0},{type:'mirror',albedo:.75,emission:0},{type:'diffuse',albedo:0,emission:8,twoSidedEmission:true}]});
scene.emitters=scene.triangles.filter(t=>t.materialIndex===2);scene.emitterArea=scene.emitters.reduce((n,t)=>n+t.area,0);
const camera={position:[0,0,0],target:[4,1,4],up:[0,1,0],fov:10},options={width:8,height:8,samplesPerPixel:4,maxDepth:6,wave:false,nee:false,seed:7391},tile={x:0,y:0,width:8,height:8};
const files=['utd-functions.wgsl','plt-resident-geometry.wgsl','plt-resident-interval.wgsl','plt-resident-axis-guard.wgsl','plt-resident-rng.wgsl','plt-resident-fsd.wgsl','plt-resident-walker.wgsl'];
const sources=files.map(f=>fs.readFileSync(new URL('../dist/'+f,import.meta.url),'utf8')),hash=x=>crypto.createHash('sha256').update(x).digest('hex');
const report={buildId:PLT_RESIDENT_BUILD_ID,executionRequested:execute,actualGPUExecuted:false,allPass:false,status:'preparing',backend,vulkanICD:process.env.VK_ICD_FILENAMES??null,nodeVersion:process.version,fullCPUShadowMandatory:true,tightGuardRequested:tight,descriptiveOnly:true,noStatisticalSpeedClaim:true,physicalGPUValidated:false,browserValidated:false,hardwarePerformanceClaim:false,scene:'unit-scale two-mirror emission',pathsPerTimingRun:128,sourceSHA256:Object.fromEntries(files.map((f,i)=>[f,hash(sources[i])])),hostSHA256:hash(fs.readFileSync(new URL('../dist/plt-resident-gpu.js',import.meta.url))),testSHA256:hash(fs.readFileSync(new URL(import.meta.url))),cases:[],mutations:[],warmups:[],measurements:[]};
const save=()=>fs.writeFileSync(new URL(`../research/plt-resident-round-scheduling-${tight?'tight-':''}${timing?'timing':execute?'native':'compile'}-validation.json`,import.meta.url),JSON.stringify(report,null,2)+'\n');
const aliases={rr_init_anchored:'rr_init',rr_anchor_prepare_step:'rr_axis_prepare_step',rr_anchor_plane_step:'rr_axis_plane_step',rr_anchor_finish_step:'rr_axis_finish_step'};
let mutate=null,mutationName=null,roundCommands=[];
// Capture actual pipeline/group/dispatch intents and encode them unchanged on
// the native device. Mutations alter only real command-buffer stage order.
// This avoids recompiling unchanged WGSL for every negative scheduling witness.
function instrument(gpu){return {requestAdapter:async descriptor=>{const adapter=await gpu.requestAdapter(descriptor);return {info:adapter.info,requestDevice:async()=>{
 const device=await adapter.requestDevice(),names=new WeakMap();
 return new Proxy(device,{get(target,key){
  if(key==='createComputePipelineAsync')return async spec=>{const pipeline=await target.createComputePipelineAsync(spec);names.set(pipeline,aliases[spec.compute.entryPoint]??spec.compute.entryPoint);return pipeline;};
  if(key==='createCommandEncoder')return descriptor=>{const operations=[];return {
   beginComputePass(spec){const op={kind:'compute',spec};operations.push(op);return {setPipeline:p=>{op.pipeline=p;op.name=names.get(p);},setBindGroup:(index,group)=>{assert.equal(index,0);op.group=group;},dispatchWorkgroups:count=>{op.count=count;},end:()=>{op.ended=true;}};},
   copyBufferToBuffer(...args){operations.push({kind:'copy',args});},
   finish(){const isRound=operations[0]?.name==='rr_traverse_step'&&operations.length>1;let encoded=operations;if(isRound){encoded=mutate?mutate([...operations]):operations;roundCommands.push(encoded.map(op=>op.name));}const encoder=target.createCommandEncoder(descriptor);for(const op of encoded){if(op.kind==='copy')encoder.copyBufferToBuffer(...op.args);else{assert(op.ended);const pass=encoder.beginComputePass(op.spec);pass.setPipeline(op.pipeline);pass.setBindGroup(0,op.group);pass.dispatchWorkgroups(op.count);pass.end();}}return encoder.finish();},
  };};
  const value=Reflect.get(target,key,target);return typeof value==='function'?value.bind(target):value;
 }});
 }};}};}
const exact=(a,b)=>{for(const key of ['audit','image','finalRngStates','pixelIndices'])assert.deepEqual(a[key],b[key],key);assert.deepEqual(a.recovery.keys,b.recovery.keys);assert.equal(a.recovery.paths,b.recovery.paths);assert.equal(a.rounds,b.rounds);assert.equal(a.dispatches,b.dispatches);};
const references=new Map(),canonicalByCase=new Map();
async function referenceFor(wave,sampleStart,samplesPerPixel){const key=[wave,sampleStart,samplesPerPixel].join('/');if(references.has(key))return references.get(key);const initial=createPLTRandomStates(options),rng=initial.slice();if(sampleStart)await buildResidentReferenceBatch(scene,camera,{...options,wave},{tile,sampleStart:0,samplesPerPixel:sampleStart,rngStates:rng});const before=rng.slice(),began=performance.now(),reference=await buildResidentReferenceBatch(scene,camera,{...options,wave},{tile,sampleStart,samplesPerPixel,rngStates:rng}),entry={initial:before,reference,final:rng,cpuFullShadowConstructionMs:performance.now()-began};references.set(key,entry);return entry;}
let gpu=create([`backend=${backend}`]),runner;
try{
 const start=performance.now();runner=await createPLTResidentRunner(scene,camera,options,{gpu:instrument(gpu),shaderSource:sources.join('\n'),timeoutMs:120000,recoveryMode:'selective',continuationGuard:tight?'axis-mirror-tight':'axis-mirror-anchor',onSetupProgress:step=>console.log('COMPILE',step.entryPoint)});report.setupWallMs=performance.now()-start;report.adapter=runner.diagnostic.adapter;
 async function run(guard,axisSubmission,{wave=false,sampleStart=0,samplesPerPixel=2,phase,comparison,pairIndex,orderIndex}={}){
  const ref=await referenceFor(wave,sampleStart,samplesPerPixel),rng=ref.initial.slice();roundCommands=[];const began=performance.now(),candidate=await runner.run({tile,sampleStart,samplesPerPixel,wave,rngStates:rng,guard,axisSubmission}),wallMs=performance.now()-began;report.actualGPUExecuted=true;assert.deepEqual(rng,ref.initial);assert(candidate.candidate&&!candidate.admitted);assert.strictEqual(candidate.image,candidate.audit.reference);
  const auditStart=performance.now(),audit=compareResidentAudit(ref.reference,candidate.audit,{maxMismatches:4096}),rawShadowAuditMs=performance.now()-auditStart;assert(audit.ok,JSON.stringify(audit));const admissionStart=performance.now(),admitted=admitResidentWithRecovery(ref.reference,candidate),admissionAuditWallMs=performance.now()-admissionStart;assert(admitted.validation.passed);assert.equal(admitted.recovery.numericGeometry.count,0);assert.deepEqual(candidate.pixelIndices,ref.reference.audit.pixelIndices);for(let i=0;i<candidate.pixelIndices.length;i++)assert.equal(candidate.finalRngStates[i],ref.final[candidate.pixelIndices[i]]);
  if(!wave)for(const p of candidate.audit.paths)assert.equal(p.sampleValue??p.value,3);
  if(!mutationName){const key=[guard,wave,sampleStart,samplesPerPixel].join('/'),canonical=canonicalByCase.get(key);if(canonical)exact(candidate,canonical);else canonicalByCase.set(key,candidate);}
  const entry={guard,axisSubmission,mutation:mutationName,wave,sampleStart,samplesPerPixel,paths:candidate.audit.paths.length,replayedPaths:candidate.recovery.paths,retainedPaths:candidate.audit.paths.length-candidate.recovery.paths,rounds:candidate.rounds,dispatches:candidate.dispatches,roundSubmissions:roundCommands.length,roundPassCounts:[...new Set(roundCommands.map(x=>x.length))],wallMs,cpuFullShadowConstructionMs:ref.cpuFullShadowConstructionMs,rawShadowAuditMs,admissionAuditWallMs,timings:candidate.timings,numericGeometryReplacements:0};
  if(phase)report[phase==='warmup'?'warmups':'measurements'].push({...entry,phase,comparison,pairIndex,orderIndex});else report.cases.push(entry);save();console.log('PASS',guard,axisSubmission,wave?'wave':'ray',sampleStart,samplesPerPixel,candidate.recovery.paths);return candidate;
 }
 if(execute){
  report.status='executing';
  for(const guard of ['axis-mirror-anchor','axis-mirror-unit','axis-mirror-interval','replay-unmodelled','audit-only',...(tight?['axis-mirror-tight']:[])]){const staged=await run(guard,'staged'),batched=await run(guard,'batched'),round=await run(guard,'round');exact(staged,batched);exact(staged,round);}
  for(const guard of ['axis-mirror-anchor',...(tight?['axis-mirror-tight']:[])])for(const params of [{sampleStart:2,samplesPerPixel:2},{samplesPerPixel:4},{wave:true,samplesPerPixel:1}]){const staged=await run(guard,'staged',params),round=await run(guard,'round',params);exact(staged,round);}
  if(tight){
   report.modeReuseSequence=[];
   for(const [guard,sampleStart]of [['axis-mirror-tight',0],['axis-mirror-anchor',2],['axis-mirror-tight',2],['axis-mirror-anchor',0],['axis-mirror-unit',0],['axis-mirror-tight',0]]){await run(guard,'round',{sampleStart,samplesPerPixel:2});report.modeReuseSequence.push({guard,sampleStart,samplesPerPixel:2,exactCanonicalParity:true});}
  }
  report.exactAllGroupingAuditRngOwnershipReplayParity=true;
  if(timing){
   const modes={'unit-staged':{guard:'axis-mirror-unit',axisSubmission:'staged'},'anchor-staged':{guard:'axis-mirror-anchor',axisSubmission:'staged'},'anchor-batched':{guard:'axis-mirror-anchor',axisSubmission:'batched'},'anchor-round':{guard:'axis-mirror-anchor',axisSubmission:'round'}};
   const comparisons=[['anchor-staged','anchor-round'],['anchor-batched','anchor-round']];
   for(const [a,b]of comparisons){const comparison=a+'-versus-'+b;for(const [orderIndex,id]of [a,b,b,a].entries())await run(modes[id].guard,modes[id].axisSubmission,{phase:'warmup',comparison,pairIndex:-1,orderIndex});for(let pairIndex=0;pairIndex<8;pairIndex++)for(const [orderIndex,id]of (pairIndex%2?[b,a]:[a,b]).entries())await run(modes[id].guard,modes[id].axisSubmission,{phase:'measured',comparison,pairIndex,orderIndex});}
   const stats=values=>{const sorted=[...values].sort((a,b)=>a-b);return {count:sorted.length,min:sorted[0],median:(sorted[(sorted.length-1)>>1]+sorted[sorted.length>>1])/2,max:sorted.at(-1),mean:sorted.reduce((a,b)=>a+b,0)/sorted.length};};
   report.timingSummary=comparisons.map(([a,b])=>{const comparison=a+'-versus-'+b,runs=report.measurements.filter(r=>r.comparison===comparison);return {comparison,byMode:Object.fromEntries([a,b].map(id=>[id,Object.fromEntries(['gpuResidentTotalWallMs','gpuRoundSubmitAndWaitMs','gpuStatusReadbackWallMs','cpuSelectiveReplayMs','replayReadbackWallMs','gpuFinalReadbackDecodeMs'].map(key=>[key,stats(runs.filter(r=>r.axisSubmission===modes[id].axisSubmission).map(r=>r.timings[key]))]))])),pairedBMinusARunMs:stats(Array.from({length:8},(_,pair)=>{const pairRuns=runs.filter(r=>r.pairIndex===pair);return pairRuns.find(r=>r.axisSubmission===modes[b].axisSubmission).timings.gpuResidentTotalWallMs-pairRuns.find(r=>r.axisSubmission===modes[a].axisSubmission).timings.gpuResidentTotalWallMs;}))};});
  }
  const mutations=[
   {name:'omit-anchor-init',change:ops=>ops.filter(op=>op.name!=='rr_anchor_init_step')},
   {name:'omit-point-check',change:ops=>ops.filter(op=>op.name!=='rr_anchor_point_step')},
   {name:'omit-last-triangle-scan',change:ops=>{const index=ops.findLastIndex(op=>op.name==='rr_axis_scan_step');return ops.filter((_,i)=>i!==index);}},
   {name:'finish-before-point',change:ops=>{const a=ops.findIndex(op=>op.name==='rr_anchor_point_step'),b=ops.findIndex(op=>op.name==='rr_axis_finish_step');[ops[a],ops[b]]=[ops[b],ops[a]];return ops;}},
   {name:'scan-before-plane',change:ops=>{for(let i=0;i<ops.length-1;i++)if(ops[i].name==='rr_axis_plane_step'&&ops[i+1].name==='rr_axis_scan_step'){[ops[i],ops[i+1]]=[ops[i+1],ops[i]];i++;}return ops;}},
  ];
  for(const guard of ['axis-mirror-anchor',...(tight?['axis-mirror-tight']:[])])for(const mutation of mutations){mutate=mutation.change;mutationName=mutation.name;const candidate=await run(guard,'round',{samplesPerPixel:1});assert.equal(candidate.recovery.paths,candidate.audit.paths.length,'Mutant must retain no unchecked partial GPU paths');assert(candidate.recovery.keys.every(key=>key.flags&268435456));report.mutations.push({name:mutation.name,guard,actualGPUExecuted:true,allPathsReplayed:true,replayedPaths:candidate.recovery.paths,fullShadowPassed:true,numericGeometryReplacements:0});mutate=null;mutationName=null;save();}
  // A missing finish cannot release pending transport. Observe two actual
  // completed GPU rounds, then cancel the deliberately non-terminating run.
  mutate=ops=>ops.filter(op=>op.name!=='rr_axis_finish_step');const controller=new AbortController(),ref=await referenceFor(false,0,1),rng=ref.initial.slice(),progress=[];await assert.rejects(runner.run({tile,samplesPerPixel:1,rngStates:rng,guard:tight?'axis-mirror-tight':'axis-mirror-anchor',axisSubmission:'round',signal:controller.signal,onProgress:p=>{progress.push(p);assert.equal(p.active,64);if(p.round===2)controller.abort();}}),{name:'AbortError'});assert.equal(progress.length,2);assert.deepEqual(rng,ref.initial);report.mutations.push({name:'omit-finish-barrier',guard:tight?'axis-mirror-tight':'axis-mirror-anchor',actualGPUExecuted:true,allLanesRemainPending:true,observedRounds:2,cancelled:true,noCandidateReturned:true,inputRngUnchanged:true});mutate=null;
 }
 report.allPass=true;report.status='passed';
}catch(error){report.status='failed';report.failure={name:error.name,message:error.message,stack:error.stack};throw error;}finally{runner?.destroy();gpu=null;save();}
console.log(JSON.stringify({allPass:report.allPass,actualGPUExecuted:report.actualGPUExecuted,cases:report.cases.length,mutations:report.mutations,timingSummary:report.timingSummary},null,2));
