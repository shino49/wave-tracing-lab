/** Descriptive warmed submission-grouping measurements on one exact mirror scene.
 * No statistical speed claim. Full independent CPU shadow on every measured run.
 * WEBGPU_EXECUTION_BACKEND=vulkan VK_ICD_FILENAMES=/usr/lib/chromium/vk_swiftshader_icd.json \
 * node tests/plt-resident-unit-submit-timing.mjs --execute
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
const quad=(z,size,materialIndex)=>({mesh:{vertices:[[-size,-size,z],[size,-size,z],[size,size,z],[-size,size,z]],faces:[[0,1,2],[0,2,3]]},materialIndex});
const scene=preparePLTScene([quad(2,8,0),quad(-2,16,1)],{materials:[{type:'mirror',albedo:.5,emission:0},{type:'diffuse',albedo:0,emission:8,twoSidedEmission:true}]});
scene.emitters=scene.triangles.filter(t=>t.materialIndex===1);scene.emitterArea=scene.emitters.reduce((n,t)=>n+t.area,0);
const camera={position:[0,0,0],target:[0,0,2],up:[0,1,0],fov:20},options={width:8,height:8,samplesPerPixel:2,maxDepth:6,wave:false,nee:false,seed:7391},tile={x:0,y:0,width:8,height:8};
const initial=createPLTRandomStates(options),cpuRng=initial.slice(),cpuStart=performance.now();
const reference=await buildResidentReferenceBatch(scene,camera,options,{tile,sampleStart:0,samplesPerPixel:2,rngStates:cpuRng});
const cpuFullShadowConstructionMs=performance.now()-cpuStart;
for(const p of reference.audit.paths){assert.equal(p.value,4);assert.equal(p.scatters.length,1);assert.equal(p.rng.count,2);}
const files=['utd-functions.wgsl','plt-resident-geometry.wgsl','plt-resident-interval.wgsl','plt-resident-axis-guard.wgsl','plt-resident-rng.wgsl','plt-resident-fsd.wgsl','plt-resident-walker.wgsl'];
const sources=files.map(f=>fs.readFileSync(new URL('../dist/'+f,import.meta.url),'utf8'));
const hash=x=>crypto.createHash('sha256').update(x).digest('hex');
const report={buildId:PLT_RESIDENT_BUILD_ID,executionRequested:execute,actualGPUExecuted:false,allPass:false,status:'compiling',backend,vulkanICD:process.env.VK_ICD_FILENAMES??null,nodeVersion:process.version,descriptiveOnly:true,noStatisticalSpeedClaim:true,physicalGPUValidated:false,browserValidated:false,fullCPUShadowRequired:true,scene:'unit one-mirror emission',pathsPerRun:128,cpuFullShadowConstructionMs,sourceSHA256:Object.fromEntries(files.map((f,i)=>[f,hash(sources[i])])),hostSHA256:Object.fromEntries(['plt-resident-gpu.js','plt-resident-geometry.js','plt-resident-reference.js','plt-resident-recovery.js','plt-core.js'].map(f=>[f,hash(fs.readFileSync(new URL('../dist/'+f,import.meta.url)))])),testSHA256:hash(fs.readFileSync(new URL(import.meta.url))),warmups:[],measurements:[]};
const save=()=>fs.writeFileSync(new URL(`../research/plt-resident-unit-submit-timing-${execute?'validation':'compile-validation'}.json`,import.meta.url),JSON.stringify(report,null,2)+'\n');
save();let gpu=create([`backend=${backend}`]);
try{
 const setup=performance.now(),runner=await createPLTResidentRunner(scene,camera,options,{gpu,shaderSource:sources.join('\n'),timeoutMs:120000,recoveryMode:'selective',continuationGuard:'axis-mirror-unit',onSetupProgress:step=>{if(execute)console.log(JSON.stringify({setup:'warmed-submission-probe',...step}));}});
 report.setupWallMs=performance.now()-setup;report.adapter=runner.diagnostic.adapter;
 try{if(execute){
  let canonical;
  const run=async(axisSubmission,phase,pairIndex,orderIndex)=>{
   const rng=initial.slice(),began=performance.now(),candidate=await runner.run({tile,sampleStart:0,samplesPerPixel:2,wave:false,rngStates:rng,guard:'axis-mirror-unit',axisSubmission}),observedRunWallMs=performance.now()-began;
   report.actualGPUExecuted=true;report.status='measuring';assert.deepEqual(rng,initial);assert.equal(candidate.recovery.axisSubmission,axisSubmission);
   const auditStart=performance.now(),rawAudit=compareResidentAudit(reference,candidate.audit,{maxMismatches:4096}),rawShadowAuditMs=performance.now()-auditStart;assert(rawAudit.ok);
   const admissionStart=performance.now(),admitted=admitResidentWithRecovery(reference,candidate),admissionAuditWallMs=performance.now()-admissionStart;assert(admitted.validation.passed);assert.equal(admitted.recovery.numericGeometry.count,0);
   for(const p of candidate.audit.paths)assert.equal(p.sampleValue??p.value,4);
   for(let i=0;i<candidate.pixelIndices.length;i++)assert.equal(candidate.finalRngStates[i],cpuRng[candidate.pixelIndices[i]]);
   if(canonical){assert.deepEqual(candidate.audit,canonical.audit);assert.deepEqual(candidate.finalRngStates,canonical.finalRngStates);assert.deepEqual(candidate.pixelIndices,canonical.pixelIndices);assert.deepEqual(candidate.recovery.keys,canonical.recovery.keys);assert.equal(candidate.rounds,canonical.rounds);assert.equal(candidate.dispatches,canonical.dispatches);}else canonical=candidate;
   const result={phase,pairIndex,orderIndex,axisSubmission,observedRunWallMs,replayPaths:candidate.recovery.paths,retainedMirrorPaths:128-candidate.recovery.paths,rawShadowAuditMs,admissionAuditWallMs,numericGeometryReplacements:0,exactAuditRngOwnershipParity:true,timings:admitted.timings,rawAudit};
   report[phase==='warmup'?'warmups':'measurements'].push(result);save();return result;
  };
  // Both modes receive two warmup runs; measured order alternates each pair.
  for(const [i,mode] of ['staged','batched','batched','staged'].entries())await run(mode,'warmup',-1,i);
  for(let pair=0;pair<8;pair++){
   const order=pair%2?['batched','staged']:['staged','batched'];
   for(const [i,mode] of order.entries())await run(mode,'measured',pair,i);
  }
  const stats=values=>{const sorted=[...values].sort((a,b)=>a-b);return {count:values.length,min:sorted[0],median:(sorted[(sorted.length-1)>>1]+sorted[sorted.length>>1])/2,max:sorted.at(-1),mean:values.reduce((n,v)=>n+v,0)/values.length};};
  report.summary={measuredPairs:8,measuredPathRuns:2048,warmupPathRuns:512,replayPathsPerRun:canonical.recovery.paths,retainedMirrorPathsPerRun:128-canonical.recovery.paths,exactAuditRngOwnershipParity:true,numericGeometryReplacements:0,bySubmission:Object.fromEntries(['staged','batched'].map(mode=>{const runs=report.measurements.filter(r=>r.axisSubmission===mode);return [mode,{rendererRunWallMs:stats(runs.map(r=>r.timings.gpuResidentTotalWallMs)),observedRunWallMs:stats(runs.map(r=>r.observedRunWallMs)),statusReadbackWallMs:stats(runs.map(r=>r.timings.gpuStatusReadbackWallMs)),cpuSelectiveReplayMs:stats(runs.map(r=>r.timings.cpuSelectiveReplayMs)),replayReadbackWallMs:stats(runs.map(r=>r.timings.replayReadbackWallMs)),finalReadbackDecodeMs:stats(runs.map(r=>r.timings.gpuFinalReadbackDecodeMs)),rawShadowAuditMs:stats(runs.map(r=>r.rawShadowAuditMs)),admissionAuditWallMs:stats(runs.map(r=>r.admissionAuditWallMs))}];})),pairedBatchedMinusStagedRunMs:stats(Array.from({length:8},(_,pair)=>{const runs=report.measurements.filter(r=>r.pairIndex===pair);return runs.find(r=>r.axisSubmission==='batched').timings.gpuResidentTotalWallMs-runs.find(r=>r.axisSubmission==='staged').timings.gpuResidentTotalWallMs;}))};
 }}finally{runner.destroy();}
 report.allPass=true;report.status='passed';
}catch(error){report.status='failed';report.failure={name:error.name,message:error.message,stack:error.stack};throw error;}
finally{gpu=null;save();}
console.log(JSON.stringify({allPass:report.allPass,actualGPUExecuted:report.actualGPUExecuted,summary:report.summary}));
