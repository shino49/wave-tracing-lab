/** Descriptive warmed anchor/baseline and submission-grouping paired measurements.
 * Run only with other native validation workloads idle.
 * No statistical speed claim. Full independent CPU shadow on every measured run.
 * WEBGPU_EXECUTION_BACKEND=vulkan VK_ICD_FILENAMES=/usr/lib/chromium/vk_swiftshader_icd.json \
 * node tests/plt-resident-anchor-submit-timing.mjs --execute
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
const execute=process.argv.includes('--execute'),backend=execute?process.env.WEBGPU_EXECUTION_BACKEND:'null';
if(execute)assert(['vulkan','metal','d3d11','d3d12','opengl','opengles'].includes(backend),'An explicit executable backend is required');
function quad(axis,position,lo,hi,materialIndex){const a=(axis+1)%3,b=(axis+2)%3,point=(u,v)=>{const p=[0,0,0];p[axis]=position;p[a]=u;p[b]=v;return p;};return {mesh:{vertices:[point(lo,lo),point(hi,lo),point(hi,hi),point(lo,hi)],faces:[[0,1,2],[0,2,3]]},materialIndex};}
const scene=preparePLTScene([quad(0,2,-8,8,0),quad(2,3,-8,8,1),quad(2,-2,-16,16,2)],{materials:[{type:'mirror',albedo:.5,emission:0},{type:'mirror',albedo:.75,emission:0},{type:'diffuse',albedo:0,emission:8,twoSidedEmission:true}]});
scene.emitters=scene.triangles.filter(t=>t.materialIndex===2);scene.emitterArea=scene.emitters.reduce((n,t)=>n+t.area,0);
const camera={position:[0,0,0],target:[4,1,4],up:[0,1,0],fov:10},options={width:8,height:8,samplesPerPixel:2,maxDepth:6,wave:false,nee:false,seed:7391},tile={x:0,y:0,width:8,height:8};
const modes={
 'unit-staged':{guard:'axis-mirror-unit',axisSubmission:'staged'},
 'anchor-staged':{guard:'axis-mirror-anchor',axisSubmission:'staged'},
 'anchor-batched':{guard:'axis-mirror-anchor',axisSubmission:'batched'},
};
const comparisons=[{id:'unit-versus-anchor',a:'unit-staged',b:'anchor-staged'},{id:'anchor-submission',a:'anchor-staged',b:'anchor-batched'}];
const initial=createPLTRandomStates(options),cpuRng=initial.slice(),cpuStart=performance.now();
const reference=await buildResidentReferenceBatch(scene,camera,options,{tile,sampleStart:0,samplesPerPixel:2,rngStates:cpuRng});
const cpuFullShadowConstructionMs=performance.now()-cpuStart;
for(const p of reference.audit.paths){assert.equal(p.value,3);assert.equal(p.scatters.length,2);assert.equal(p.rng.count,2);}
const files=['utd-functions.wgsl','plt-resident-geometry.wgsl','plt-resident-interval.wgsl','plt-resident-axis-guard.wgsl','plt-resident-rng.wgsl','plt-resident-fsd.wgsl','plt-resident-walker.wgsl'];
const sources=files.map(f=>fs.readFileSync(new URL('../dist/'+f,import.meta.url),'utf8'));
const hash=x=>crypto.createHash('sha256').update(x).digest('hex');
const report={buildId:PLT_RESIDENT_BUILD_ID,executionRequested:execute,actualGPUExecuted:false,allPass:false,status:'compiling',backend,vulkanICD:process.env.VK_ICD_FILENAMES??null,nodeVersion:process.version,descriptiveOnly:true,noStatisticalSpeedClaim:true,physicalGPUValidated:false,browserValidated:false,hardwarePerformanceClaim:false,fullCPUShadowRequired:true,scene:'unit-scale two-mirror emission',comparisonScope:'same shared anchor-compiled runner; unit mode skips anchor passes',limitations:['Unit mode is run through the shared anchor-compiled runner and entrypoint aliases, with anchor passes skipped. This is not a separately compiled historical v19 runner setup or steady-state comparison.','Full fixture transport independently pins unit-mode replay and retained-mirror counts to the preserved v19 report.','Descriptive software-adapter wall times only; no physical-GPU or statistical speed claim.'],modes,comparisons,pathsPerRun:128,cpuFullShadowConstructionMs,sourceSHA256:Object.fromEntries(files.map((f,i)=>[f,hash(sources[i])])),hostSHA256:Object.fromEntries(['plt-resident-gpu.js','plt-resident-geometry.js','plt-resident-reference.js','plt-resident-recovery.js','plt-core.js'].map(f=>[f,hash(fs.readFileSync(new URL('../dist/'+f,import.meta.url)))])),testSHA256:hash(fs.readFileSync(new URL(import.meta.url))),warmups:[],measurements:[]};
const save=()=>fs.writeFileSync(new URL(`../research/plt-resident-anchor-submit-timing-${execute?'validation':'compile-validation'}.json`,import.meta.url),JSON.stringify(report,null,2)+'\n');
save();let gpu=create([`backend=${backend}`]);
try{
 const setup=performance.now(),runner=await createPLTResidentRunner(scene,camera,options,{gpu,shaderSource:sources.join('\n'),timeoutMs:120000,recoveryMode:'selective',continuationGuard:'axis-mirror-anchor',onSetupProgress:step=>{if(execute)console.log(JSON.stringify({setup:'warmed-submission-probe',...step}));}});
 report.setupWallMs=performance.now()-setup;report.adapter=runner.diagnostic.adapter;
 try{if(execute){
  const canonicalByGuard=new Map();
  const exactParity=(a,b)=>{assert.deepEqual(a.audit,b.audit);assert.deepEqual(a.image,b.image);assert.deepEqual(a.finalRngStates,b.finalRngStates);assert.deepEqual(a.pixelIndices,b.pixelIndices);assert.deepEqual(a.recovery.keys,b.recovery.keys);assert.equal(a.recovery.paths,b.recovery.paths);assert.equal(a.rounds,b.rounds);assert.equal(a.dispatches,b.dispatches);};
  const run=async(modeId,phase,comparison,pairIndex,orderIndex)=>{
   const {guard,axisSubmission}=modes[modeId],rng=initial.slice(),began=performance.now(),candidate=await runner.run({tile,sampleStart:0,samplesPerPixel:2,wave:false,rngStates:rng,guard,axisSubmission}),observedRunWallMs=performance.now()-began;
   report.actualGPUExecuted=true;report.status='measuring';assert.deepEqual(rng,initial);assert.equal(candidate.recovery.axisSubmission,axisSubmission);assert.equal(candidate.recovery.continuationGuard,guard);
   assert.equal(candidate.candidate,true);assert.equal(candidate.admitted,false);assert.strictEqual(candidate.image,candidate.audit.reference);
   const auditStart=performance.now(),rawAudit=compareResidentAudit(reference,candidate.audit,{maxMismatches:4096}),rawShadowAuditMs=performance.now()-auditStart;assert(rawAudit.ok);
   const admissionStart=performance.now(),admitted=admitResidentWithRecovery(reference,candidate),admissionAuditWallMs=performance.now()-admissionStart;assert(admitted.validation.passed);assert.equal(admitted.recovery.numericGeometry.count,0);
   for(const p of candidate.audit.paths)assert.equal(p.sampleValue??p.value,3);
   for(const value of candidate.image)assert.equal(value,3);
   for(let i=0;i<candidate.pixelIndices.length;i++)assert.equal(candidate.finalRngStates[i],cpuRng[candidate.pixelIndices[i]]);
   assert.deepEqual(candidate.pixelIndices,reference.audit.pixelIndices);assert.equal(candidate.audit.paths.length,128);
   const canonical=canonicalByGuard.get(guard);if(canonical)exactParity(candidate,canonical);else canonicalByGuard.set(guard,candidate);
   const result={phase,comparison,pairIndex,orderIndex,modeId,guard,axisSubmission,observedRunWallMs,replayPaths:candidate.recovery.paths,retainedMirrorPaths:128-candidate.recovery.paths,rawShadowAuditMs,admissionAuditWallMs,numericGeometryReplacements:0,exactWithinGuardAuditRngOwnershipParity:true,timings:admitted.timings,rawAudit};
   report[phase==='warmup'?'warmups':'measurements'].push(result);save();return result;
  };
  // Independent comparisons each get ABBA warmups and eight alternating pairs.
  for(const comparison of comparisons){
   for(const [i,mode] of [comparison.a,comparison.b,comparison.b,comparison.a].entries())await run(mode,'warmup',comparison.id,-1,i);
   for(let pair=0;pair<8;pair++){
    const order=pair%2?[comparison.b,comparison.a]:[comparison.a,comparison.b];
    for(const [i,mode] of order.entries())await run(mode,'measured',comparison.id,pair,i);
   }
  }
  const stats=values=>{const sorted=[...values].sort((a,b)=>a-b);return {count:values.length,min:sorted[0],median:(sorted[(sorted.length-1)>>1]+sorted[sorted.length>>1])/2,max:sorted.at(-1),mean:values.reduce((n,v)=>n+v,0)/values.length};};
  const costSummary=runs=>Object.fromEntries([
   ['rendererRunWallMs',r=>r.timings.gpuResidentTotalWallMs],['observedRunWallMs',r=>r.observedRunWallMs],
   ['statusReadbackWallMs',r=>r.timings.gpuStatusReadbackWallMs],['cpuSelectiveReplayMs',r=>r.timings.cpuSelectiveReplayMs],
   ['replayReadbackWallMs',r=>r.timings.replayReadbackWallMs],['finalReadbackDecodeMs',r=>r.timings.gpuFinalReadbackDecodeMs],
   ['rawShadowAuditMs',r=>r.rawShadowAuditMs],['admissionAuditWallMs',r=>r.admissionAuditWallMs],
  ].map(([name,value])=>[name,stats(runs.map(value))]));
  report.summary={measuredPairs:16,measuredPathRuns:4096,warmupPathRuns:1024,exactWithinGuardAuditRngOwnershipParity:true,numericGeometryReplacements:0,
   byGuard:Object.fromEntries([...canonicalByGuard].map(([guard,c])=>[guard,{replayPathsPerRun:c.recovery.paths,retainedMirrorPathsPerRun:128-c.recovery.paths}])),
   comparisons:comparisons.map(c=>({id:c.id,a:c.a,b:c.b,measuredPairs:8,byMode:Object.fromEntries([c.a,c.b].map(mode=>[mode,costSummary(report.measurements.filter(r=>r.comparison===c.id&&r.modeId===mode))])),
    pairedBMinusARunMs:stats(Array.from({length:8},(_,pair)=>{const runs=report.measurements.filter(r=>r.comparison===c.id&&r.pairIndex===pair);return runs.find(r=>r.modeId===c.b).timings.gpuResidentTotalWallMs-runs.find(r=>r.modeId===c.a).timings.gpuResidentTotalWallMs;})),
   })),
  };

 }}finally{runner.destroy();}
 report.allPass=true;report.status='passed';
}catch(error){report.status='failed';report.failure={name:error.name,message:error.message,stack:error.stack};throw error;}
finally{gpu=null;save();}
console.log(JSON.stringify({allPass:report.allPass,actualGPUExecuted:report.actualGPUExecuted,summary:report.summary}));
