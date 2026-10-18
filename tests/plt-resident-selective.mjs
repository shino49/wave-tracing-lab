/** Actual native partial-replay experiment. No physical-GPU performance claim. */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import crypto from 'node:crypto';
import {create,globals} from 'webgpu';
import {createPLTScene,createPLTRandomStates} from '../dist/plt-core.js';
import {createPLTResidentRunner,PLT_RESIDENT_BUILD_ID} from '../dist/plt-resident-gpu.js';
import {buildResidentReferenceBatch,compareResidentAudit} from '../dist/plt-resident-reference.js';
import {admitResidentWithRecovery} from '../dist/plt-resident-recovery.js';
import {executePLTTask} from '../dist/plt-runtime.js';
Object.assign(globalThis,globals);
const execute=process.argv.includes('--execute'),backend=execute?process.env.WEBGPU_EXECUTION_BACKEND:'null';
if(execute)assert(['vulkan','metal','d3d11','d3d12','opengl','opengles'].includes(backend));
const filenames=['utd-functions.wgsl','plt-resident-geometry.wgsl','plt-resident-interval.wgsl','plt-resident-axis-guard.wgsl','plt-resident-rng.wgsl','plt-resident-fsd.wgsl','plt-resident-walker.wgsl'],sources=filenames.map(f=>fs.readFileSync(new URL('../dist/'+f,import.meta.url),'utf8'));
let gpu=create([`backend=${backend}`]);const cases=[];
const report={buildId:PLT_RESIDENT_BUILD_ID,actualGPUExecuted:execute,backend,hardwarePerformanceClaim:false,fullCPUShadowRequired:true,hostSHA256:Object.fromEntries(['plt-resident-gpu.js','plt-resident-recovery.js','plt-runtime.js','plt-resident-reference.js','plt-core.js'].map(f=>[f,crypto.createHash('sha256').update(fs.readFileSync(new URL('../dist/'+f,import.meta.url))).digest('hex')])),sourceSHA256:Object.fromEntries(filenames.map((f,i)=>[f,crypto.createHash('sha256').update(sources[i]).digest('hex')])),cases};
try{
 const scene=createPLTScene({preset:'gallery'}),o={width:8,height:8,samplesPerPixel:3,maxDepth:6,wave:true};
 const setup=performance.now(),runner=await createPLTResidentRunner(scene,scene.camera,o,{gpu,shaderSource:sources.join('\n'),timeoutMs:120000,recoveryMode:'selective',continuationGuard:'audit-only'});report.setupWallMs=performance.now()-setup;report.adapter=runner.diagnostic.adapter;
 try{if(execute){
  for(const wave of [true,false]){
   const rng=createPLTRandomStates(o),tile=wave?{x:0,y:0,width:8,height:8}:{x:1,y:0,width:7,height:8};
   for(const [sampleStart,samplesPerPixel] of [[0,2],[2,1]]){
    const original=rng.slice(),referenceRng=rng.slice(),start=performance.now();
    const reference=await buildResidentReferenceBatch(scene,scene.camera,{...o,wave},{tile,sampleStart,samplesPerPixel,rngStates:referenceRng});const cpuFullShadowMs=performance.now()-start;
    const result=await runner.run({tile,sampleStart,samplesPerPixel,wave,rngStates:rng});assert.deepEqual(rng,original);
    const initialGate=compareResidentAudit(reference,result.audit,{maxMismatches:4096}),admitted=admitResidentWithRecovery(reference,result),gate=admitted.validation;assert.equal(gate.passed,true);assert.strictEqual(admitted.image,admitted.audit.reference);
    cases.push({wave,tile,sampleStart,samplesPerPixel,cpuFullShadowMs,recovery:admitted.recovery,timings:admitted.timings,initialGate,gate});
    console.log(JSON.stringify({wave,sampleStart,samplesPerPixel,gpuPredicatePaths:result.recovery.paths,auditGeometryPaths:admitted.recovery.numericGeometry.count,passed:gate.passed}));
    for(let i=0;i<result.pixelIndices.length;i++)assert.equal(result.finalRngStates[i],referenceRng[result.pixelIndices[i]],'canonical later-sample RNG');
    rng.set(referenceRng);
   }
  }
  const request={scene:{preset:'gallery'},backend:'resident-recovery',compare:true,options:o,progressive:{tileSize:8,batchSamples:2,maxRuntimeMs:120000,previewIntervalMs:0}},previews=[];
  const cpu=await executePLTTask({...request,backend:'cpu'});
  const runtime=await executePLTTask(request,{preview:p=>previews.push(p),gpuRunner:input=>runner.run({...input,wave:input.options.wave})});
  assert.equal(runtime.results.length,2);assert.equal(runtime.progressive.completedBatches,4);
  for(let mode=0;mode<2;mode++){const actual=runtime.results[mode],expected=cpu.results[mode];assert(actual.validation.passed);assert.equal(actual.validation.fullCPUShadow,true);assert.equal(actual.completedSamplesPerPixel,3);assert(actual.backend.includes('逐路径 CPU'));for(let i=0;i<actual.image.length;i++)assert(Math.abs(actual.image[i]-expected.image[i])<=2e-5+.003*Math.abs(expected.image[i]));}
  assert(previews.length>=4);assert(previews.at(-1).results.every(r=>r.completedSamplesPerPixel===3));
  report.runtime={passed:true,totalMs:runtime.totalMs,cpuTotalMs:cpu.totalMs,previews:previews.length,completedBatches:runtime.progressive.completedBatches,results:runtime.results.map(r=>({wave:r.wave,backend:r.backend,validation:r.validation,timings:r.timings}))};
 }}finally{runner.destroy();}
 if(execute){
  const guardedSetup=performance.now(),guarded=await createPLTResidentRunner(scene,scene.camera,o,{gpu,shaderSource:sources.join('\n'),timeoutMs:120000,recoveryMode:'selective'});
  report.guarded={setupWallMs:performance.now()-guardedSetup,continuationGuard:'replay-unmodelled',batches:[]};
  try{for(const wave of [true,false]){
   const rng=createPLTRandomStates(o),tile=wave?{x:0,y:0,width:8,height:8}:{x:1,y:0,width:7,height:8};
   for(const [sampleStart,samplesPerPixel] of [[0,2],[2,1]]){
    const before=rng.slice(),cpuRng=rng.slice(),start=performance.now(),reference=await buildResidentReferenceBatch(scene,scene.camera,{...o,wave},{tile,sampleStart,samplesPerPixel,rngStates:cpuRng}),cpuFullShadowMs=performance.now()-start;
    const candidate=await guarded.run({tile,sampleStart,samplesPerPixel,wave,rngStates:rng});assert.deepEqual(rng,before);
    const admitted=admitResidentWithRecovery(reference,candidate);assert(admitted.validation.passed);assert.equal(admitted.recovery.numericGeometry.count,0,'guarded fixtures must not rely on shadow-discovered numerical replacement');
    for(let i=0;i<candidate.pixelIndices.length;i++)assert.equal(candidate.finalRngStates[i],cpuRng[candidate.pixelIndices[i]]);
    rng.set(cpuRng);
    const coverageGuardPaths=candidate.recovery.keys.filter(e=>e.reasons.includes('unmodelledContinuation')).length;
    report.guarded.batches.push({wave,tile,sampleStart,samplesPerPixel,cpuFullShadowMs,coverageGuardPaths,predicatePaths:candidate.recovery.paths-coverageGuardPaths,recovery:admitted.recovery,timings:admitted.timings,gate:admitted.validation});
    if(!wave&&sampleStart===2){const known=candidate.recovery.keys.find(e=>e.key.pixel===39&&e.key.sample===2);assert(known?.reasons.includes('unmodelledContinuation'));report.guarded.knownDriftPreempted=true;}
    console.log(JSON.stringify({guarded:true,wave,sampleStart,paths:admitted.validation.checked.paths,coverageGuardPaths,predicatePaths:candidate.recovery.paths-coverageGuardPaths,numericalReplacements:admitted.recovery.numericGeometry.count}));
   }
  }
   const request={scene:{preset:'gallery'},backend:'resident-recovery',compare:true,options:o,progressive:{tileSize:8,batchSamples:2,maxRuntimeMs:120000,previewIntervalMs:0}},previews=[];
   const cpu=await executePLTTask({...request,backend:'cpu'}),runtime=await executePLTTask(request,{preview:p=>previews.push(p),gpuRunner:input=>guarded.run({...input,wave:input.options.wave})});
   for(let mode=0;mode<2;mode++){const actual=runtime.results[mode],expected=cpu.results[mode];assert(actual.validation.passed);assert.equal(actual.validation.recovery.auditGeometryPaths,0);assert.equal(actual.validation.recovery.continuationGuard,'replay-unmodelled');assert.equal(actual.completedSamplesPerPixel,3);for(let i=0;i<actual.image.length;i++)assert(Math.abs(actual.image[i]-expected.image[i])<=2e-5+.003*Math.abs(expected.image[i]));}
   assert(previews.length>=4);report.guarded.runtime={passed:true,totalMs:runtime.totalMs,cpuTotalMs:cpu.totalMs,previews:previews.length,completedBatches:runtime.progressive.completedBatches,results:runtime.results.map(r=>({wave:r.wave,backend:r.backend,validation:r.validation,timings:r.timings}))};
  }finally{guarded.destroy();}
 }
 report.allPass=true;
}finally{gpu=null;fs.writeFileSync(new URL('../research/plt-resident-selective-'+(execute?'validation':'compile-validation')+'.json',import.meta.url),JSON.stringify(report,null,2)+'\n');}

console.log(JSON.stringify({allPass:report.allPass,actualGPUExecuted:execute,cases:cases.length,directPaths:cases.reduce((n,c)=>n+c.gate.checked.paths,0),runtimePaths:report.runtime?.results.reduce((n,r)=>n+r.validation.auditCounts.paths,0)??0,guardedDirectPaths:report.guarded?.batches.reduce((n,c)=>n+c.gate.checked.paths,0)??0,guardedRuntimePaths:report.guarded?.runtime?.results.reduce((n,r)=>n+r.validation.auditCounts.paths,0)??0}));
