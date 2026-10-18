/** v21 paired warmed renderer measurements. Run with other native work idle.
 * CPU planning: node tests/plt-resident-tight-submit-timing.mjs --cpu-only
 * Native: WEBGPU_EXECUTION_BACKEND=vulkan VK_ICD_FILENAMES=/usr/lib/chromium/vk_swiftshader_icd.json node tests/plt-resident-tight-submit-timing.mjs --execute
 * Full independent CPU shadow on every run. Descriptive software wall times only.
 */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {create,globals} from 'webgpu';
import {createPLTRandomStates} from '../dist/plt-core.js';
import {createPLTResidentRunner,PLT_RESIDENT_BUILD_ID} from '../dist/plt-resident-gpu.js';
import {buildResidentReferenceBatch} from '../dist/plt-resident-reference.js';
import {fixtures,options,modes,hash,pinnedBaseline,readBaseline,sourceProvenance,assertEquivalent,validateCandidate} from './plt-resident-tight-transport.mjs';
Object.assign(globalThis,globals);
const execute=process.argv.includes('--execute'),cpuOnly=process.argv.includes('--cpu-only');assert(!(execute&&cpuOnly),'Choose CPU planning or native execution');
const backend=execute?process.env.WEBGPU_EXECUTION_BACKEND:'null';if(execute)assert(['vulkan','metal','d3d11','d3d12','opengl','opengles'].includes(backend),'--execute requires an explicit executable backend');
const baseline=readBaseline(),provenance=sourceProvenance();
const timingBaselineBytes=fs.readFileSync(new URL('../research/plt-resident-anchor-submit-timing-validation.json',import.meta.url));
assert.equal(hash(timingBaselineBytes),'8696fef8cd2a072578be22e405712050e27d510cdc27303343c4ac06dcd72f06','Pinned v20 warmed report must remain unchanged');
const baseNames=['two-mirrors-emission','four-scale-shifted-zxy','half-scale-three-mirrors-negative-permutation','eight-scale-two-mirrors'];
const selected=process.env.PLT_TIGHT_TIMING_CASE?fixtures.filter(f=>f.name===process.env.PLT_TIGHT_TIMING_CASE):fixtures.filter(f=>baseNames.includes(f.name));assert(selected.length,'Unknown PLT_TIGHT_TIMING_CASE');
assert(selected.every(f=>f.mustRetain),'Timing corpus must use existing designed mirror fixtures');
const comparisons=[
 {id:'anchor-scheduling-only',a:'anchor-batched',b:'anchor-round'},
 {id:'tight-scheduling-only',a:'tight-batched',b:'tight-round'},
 {id:'tightening-only-batched',a:'anchor-batched',b:'tight-batched'},
 {id:'combined-versus-v20-mode',a:'anchor-batched',b:'tight-round'},
];
// Balanced four-treatment orders: each mode occupies every position per cycle.
// The comparison pair always shares one fixture, one immutable RNG prefix and one quartet.
const balancedOrders=[[0,1,3,2],[1,2,0,3],[2,3,1,0],[3,0,2,1]].map(order=>order.map(i=>modes[i].id));
const report={buildId:PLT_RESIDENT_BUILD_ID,executionRequested:execute,actualGPUExecuted:false,allPass:false,status:'preparing',backend,vulkanICD:process.env.VK_ICD_FILENAMES??null,nodeVersion:process.version,descriptiveOnly:true,noStatisticalSpeedClaim:true,physicalGPUValidated:false,browserValidated:false,hardwarePerformanceClaim:false,fullCPUShadowRequired:true,baseline:pinnedBaseline,preservedTimingBaseline:{file:'plt-resident-anchor-submit-timing-validation.json',sha256:hash(timingBaselineBytes)},comparisonScope:'Same shared tight-compiled runner per fixture and workload; v20 anchor mode keeps its exact guard and historical first-batch retention counts',limitations:['Shared tight-compiled aliases are used for anchor and tight modes; this is not a separately compiled historical v20 runner setup comparison.','This bounded checkpoint measures four pre-existing mirror fixtures at 128 paths per run. No larger-path-load measurements or performance claim are included.','CPU shadow is constructed independently per fixture/workload and audited on every measured or warmup run. Renderer time excludes CPU shadow construction and subsequent audits; it includes selective replay and readbacks, all separately reported.','Renderer dispatch/submit/wait counters are host wall times, not GPU timestamp queries.','Descriptive native software-adapter timing only; no physical-GPU, browser or statistical speed claim.'],modes,comparisons,balancedOrders,measuredQuartetsPerScenario:8,warmupsPerMode:2,selectedFixtures:selected.map(f=>f.name),fullTimingSuite:!process.env.PLT_TIGHT_TIMING_CASE,sourceSHA256:provenance.sourceSHA256,hostSHA256:provenance.hostSHA256,testSHA256:hash(fs.readFileSync(new URL(import.meta.url))),transportHarnessSHA256:hash(fs.readFileSync(new URL('./plt-resident-tight-transport.mjs',import.meta.url))),scenarios:[]};
const save=()=>fs.writeFileSync(new URL(`../research/plt-resident-tight-submit-timing-${process.env.PLT_TIGHT_TIMING_CASE?process.env.PLT_TIGHT_TIMING_CASE+'-':''}${cpuOnly?'cpu-validation':execute?'validation':'compile-validation'}.json`,import.meta.url),JSON.stringify(report,null,2)+'\n');
if(execute){
 const bytes=fs.readFileSync(new URL('../research/plt-resident-tight-validation.json',import.meta.url)),transport=JSON.parse(bytes);
 assert(transport.allPass&&transport.actualGPUExecuted&&transport.fullFixtureSuite,'Timing requires completed full-corpus native transport correctness');
 assert.deepEqual(transport.sourceSHA256,provenance.sourceSHA256,'Timing sources differ from validated transport');assert.deepEqual(transport.hostSHA256,provenance.hostSHA256,'Timing host differs from validated transport');
 report.transportEvidence={file:'plt-resident-tight-validation.json',sha256:hash(bytes),totalReplayReduction:transport.totalReplayReduction};
}
report.largeWorkloadSelection={executed:false,skipped:true,reason:'Checkpoint bounded to four representative fixture identities at the same 128-path workload; no larger-workload performance claim'};
const scenarios=selected.map(f=>({f,width:8,height:8,samplesPerPixel:2,pathMultiplier:1}));
const preparations=[];
for(const s of scenarios){
 const o={...options,width:s.width,height:s.height,samplesPerPixel:s.samplesPerPixel},tile={x:0,y:0,width:s.width,height:s.height},initial=createPLTRandomStates(o),cpuRng=initial.slice(),began=performance.now();
 const reference=await buildResidentReferenceBatch(s.f.scene,s.f.camera,o,{tile,sampleStart:0,samplesPerPixel:s.samplesPerPixel,rngStates:cpuRng}),cpuFullShadowConstructionMs=performance.now()-began;
 for(const p of reference.audit.paths){assert.equal(p.value,s.f.expected);assert.equal(p.scatters.length,s.f.mirrors);assert.equal(p.rng.count,2);}
 assert.equal(reference.audit.paths.length,128*s.pathMultiplier);
 const record={id:`${s.f.name}/${s.pathMultiplier}x`,name:s.f.name,pathMultiplier:s.pathMultiplier,width:s.width,height:s.height,samplesPerPixel:s.samplesPerPixel,pathsPerRun:reference.audit.paths.length,expectedRadiance:s.f.expected,expectedMirrorBounces:s.f.mirrors,cpuFullShadowConstructionMs,setupWallMs:null,warmups:[],measurements:[]};
 report.scenarios.push(record);preparations.push({...s,options:o,tile,initial,b:{seed:o.seed,options:o,sampleStart:0,samplesPerPixel:s.samplesPerPixel,start:initial,end:cpuRng,reference},record});
}
if(cpuOnly){report.allPass=true;report.status='cpu-planning-passed';report.summary={scenarios:preparations.length,totalReferencePaths:preparations.reduce((n,p)=>n+p.record.pathsPerRun,0),baseFixtureCount:selected.length,allGeometriesFromPinnedCorpus:true};save();console.log(JSON.stringify({allPass:true,cpuOnly:true,summary:report.summary}));process.exit(0);}
const stats=values=>{assert(values.length);const sorted=[...values].sort((a,b)=>a-b);return {count:values.length,min:sorted[0],median:(sorted[(sorted.length-1)>>1]+sorted[sorted.length>>1])/2,max:sorted.at(-1),mean:values.reduce((n,v)=>n+v,0)/values.length};};
const costFields=[['rendererRunWallMs',r=>r.timings.gpuResidentTotalWallMs],['observedRunWallMs',r=>r.observedRunWallMs],['initSubmitWaitWallMs',r=>r.timings.gpuInitSubmitAndWaitMs],['roundSubmitWaitWallMs',r=>r.timings.gpuRoundSubmitAndWaitMs??0],['axisGuardSubmitWaitWallMs',r=>r.timings.gpuAxisGuardSubmitAndWaitMs??0],['statusReadbackWallMs',r=>r.timings.gpuStatusReadbackWallMs],['cpuSelectiveReplayMs',r=>r.timings.cpuSelectiveReplayMs],['replayReadbackWallMs',r=>r.timings.replayReadbackWallMs],['finalReadbackDecodeMs',r=>r.timings.gpuFinalReadbackDecodeMs],['rawShadowAuditMs',r=>r.rawShadowAuditMs],['admissionAuditWallMs',r=>r.admissionAuditWallMs]];
const costSummary=runs=>Object.fromEntries(costFields.map(([name,value])=>[name,stats(runs.map(value))]));
report.status='compiling';save();let gpu=create([`backend=${backend}`]);
try{
 for(const p of execute?preparations:preparations.slice(0,1)){
  const {f,b,record}=p,setup=performance.now(),runner=await createPLTResidentRunner(f.scene,f.camera,p.options,{gpu,shaderSource:provenance.sources.join('\n'),timeoutMs:120000,recoveryMode:'selective',continuationGuard:'axis-mirror-tight',onSetupProgress:step=>{if(execute)console.log(JSON.stringify({setup:record.id,...step}));}});
  record.setupWallMs=performance.now()-setup;record.adapter=runner.diagnostic.adapter;
  try{if(execute){
   const canonicalByGuard=new Map();
   const run=async(modeId,phase,pairIndex,orderIndex)=>{
    const mode=modes.find(m=>m.id===modeId),rng=p.initial.slice(),began=performance.now(),candidate=await runner.run({tile:p.tile,sampleStart:0,samplesPerPixel:p.samplesPerPixel,wave:false,rngStates:rng,guard:mode.guard,axisSubmission:mode.axisSubmission}),observedRunWallMs=performance.now()-began;
    report.actualGPUExecuted=true;report.status='measuring';assert.deepEqual(rng,p.initial,'Transactional timing RNG mutated');
    const validation=validateCandidate(f,b,candidate,mode,{pixelCount:p.width*p.height}),canonical=canonicalByGuard.get(mode.guard);
    if(canonical)assertEquivalent(canonical,candidate,`${record.id}/${modeId}: exact warmed within-guard parity`);else canonicalByGuard.set(mode.guard,candidate);
    if(mode.guard==='axis-mirror-anchor'&&p.pathMultiplier===1){
     const historical=baseline.cases.find(c=>c.name===f.name)?.batches.find(h=>h.seed===options.seed&&h.sampleStart===0&&h.samplesPerPixel===2)?.['anchor-batched'];assert(historical,'Missing v20 first-batch checkpoint');
     assert.equal(validation.replayPaths,historical.replayPaths,'Pinned anchor first-batch replay count');assert.equal(validation.retainedMirrorPaths,historical.retainedMirrorPaths,'Pinned anchor first-batch retention count');
    }
    const result={phase,pairIndex,orderIndex,modeId,guard:mode.guard,axisSubmission:mode.axisSubmission,observedRunWallMs,...validation,exactWithinGuardAuditRngOwnershipParity:true};record[phase==='warmup'?'warmups':'measurements'].push(result);save();return result;
   };
   for(let repeat=0;repeat<2;repeat++)for(const [index,id] of (repeat?[...modes].reverse():modes).map(m=>m.id).entries())await run(id,'warmup',-1-repeat,index);
   for(let pair=0;pair<8;pair++)for(const [index,id] of balancedOrders[pair%balancedOrders.length].entries())await run(id,'measured',pair,index);
   record.summary={measuredQuartets:8,measuredRuns:record.measurements.length,measuredPathRuns:record.measurements.length*record.pathsPerRun,warmupPathRuns:record.warmups.length*record.pathsPerRun,exactWithinGuardAuditRngOwnershipParity:true,numericGeometryReplacements:0,byMode:Object.fromEntries(modes.map(({id})=>{const runs=record.measurements.filter(r=>r.modeId===id);return [id,{replayPathsPerRun:runs[0].replayPaths,retainedMirrorPathsPerRun:runs[0].retainedMirrorPaths,...costSummary(runs)}];})),comparisons:comparisons.map(c=>{
    const pairs=Array.from({length:8},(_,pairIndex)=>{const runs=record.measurements.filter(r=>r.pairIndex===pairIndex),a=runs.find(r=>r.modeId===c.a),b=runs.find(r=>r.modeId===c.b);return {pairIndex,aOrder:a.orderIndex,bOrder:b.orderIndex,aRunWallMs:a.timings.gpuResidentTotalWallMs,bRunWallMs:b.timings.gpuResidentTotalWallMs,bMinusARunWallMs:b.timings.gpuResidentTotalWallMs-a.timings.gpuResidentTotalWallMs,bOverARunWallRatio:b.timings.gpuResidentTotalWallMs/a.timings.gpuResidentTotalWallMs,replayReduction:a.replayPaths-b.replayPaths};});
    return {...c,pairs,pairedBMinusARunWallMs:stats(pairs.map(p=>p.bMinusARunWallMs)),pairedBOverARunWallRatio:stats(pairs.map(p=>p.bOverARunWallRatio))};
   })};save();console.log(JSON.stringify({scenario:record.id,pathsPerRun:record.pathsPerRun,byMode:Object.fromEntries(Object.entries(record.summary.byMode).map(([id,m])=>[id,{replay:m.replayPathsPerRun,retained:m.retainedMirrorPathsPerRun,medianRendererRunMs:m.rendererRunWallMs.median}])),pairedComparisons:record.summary.comparisons.map(c=>({id:c.id,medianBMinusAMs:c.pairedBMinusARunWallMs.median}))}));
  }}finally{runner.destroy();}
 }
 if(execute)report.summary={scenarioCount:report.scenarios.length,baseFixtureCount:selected.length,largerFixtureCount:0,measuredRuns:report.scenarios.reduce((n,s)=>n+s.measurements.length,0),measuredPathRuns:report.scenarios.reduce((n,s)=>n+s.summary.measuredPathRuns,0),warmupPathRuns:report.scenarios.reduce((n,s)=>n+s.summary.warmupPathRuns,0),cpuFullShadowConstructionMs:report.scenarios.reduce((n,s)=>n+s.cpuFullShadowConstructionMs,0),sharedSetupWallMs:report.scenarios.reduce((n,s)=>n+s.setupWallMs,0),exactWithinGuardAuditRngOwnershipParity:true,numericGeometryReplacements:0,allGeometriesFromPinnedCorpus:true};
 readBaseline();assert.equal(hash(fs.readFileSync(new URL('../research/plt-resident-anchor-submit-timing-validation.json',import.meta.url))),report.preservedTimingBaseline.sha256);report.allPass=true;report.status='passed';
}catch(error){report.allPass=false;report.status='failed';report.failure={name:error.name,message:error.message,stack:error.stack};throw error;}
finally{gpu=null;save();}
console.log(JSON.stringify({allPass:report.allPass,actualGPUExecuted:report.actualGPUExecuted,summary:report.summary}));
