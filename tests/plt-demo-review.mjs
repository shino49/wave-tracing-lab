/** Independent host review. GPU outputs are deliberate synthetic fixtures here;
 * this suite makes no browser, hardware-GPU, or numerical-accuracy claim. */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {fileURLToPath,pathToFileURL} from 'node:url';
import {createPLTScene,createPLTRandomStates,buildPLTBatch,materializePLTBatchReference,replayPLT} from '../dist/plt-core.js';
import {executePLTDemoTask} from '../dist/plt-demo-runtime.js';
import {packPLTGPU,pltEventFiniteGate,pltPixelFiniteGate} from '../dist/plt-gpu.js';
import {installPLTDemoWorker} from '../dist/plt-demo-worker.js';
const root=fileURLToPath(new URL('../',import.meta.url));
let passed=0,failed=0;
const test=async(name,body)=>{try{await body();passed++;console.log(`PASS ${name}`);}catch(error){failed++;console.error(`FAIL ${name}\n${error.stack}`);}};
const request={backend:'auto',compare:true,utdAblation:false,scene:{preset:'gallery'},options:{width:8,height:8,samplesPerPixel:3,maxDepth:4,wavelengthMm:.2,seed:7391},progressive:{tileSize:4,batchSamples:2,previewIntervalMs:0,maxRuntimeMs:10000}};
const synthetic=(job,value=1)=>({width:job.width,height:job.height,image:new Float32Array(job.pixelIndices.length).fill(value),eventPowers:new Float32Array(job.events.length).fill(1),validation:{passed:true,mode:'finite',parityChecked:false,events:{count:job.events.length},pixels:{count:job.pixelIndices.length}}});
const close=(a,b)=>{assert.equal(a.length,b.length);for(let i=0;i<a.length;i++)assert(Math.abs(a[i]-b[i])<=5e-13*Math.max(1,Math.abs(a[i]),Math.abs(b[i])),`pixel ${i}: ${a[i]} != ${b[i]}`);};
const deferred=()=>{let resolve,reject;const promise=new Promise((a,b)=>{resolve=a;reject=b;});return {promise,resolve,reject};};
const tick=()=>new Promise(resolve=>setTimeout(resolve,0));

await test('instrumented normal GPU path never executes CPU UTD coefficients or CPU reference replay',async()=>{
 const temporary=fs.mkdtempSync(path.join(os.tmpdir(),'plt-demo-review-'));
 try{
  fs.writeFileSync(path.join(temporary,'package.json'),'{"type":"module"}');
  for(const file of fs.readdirSync(path.join(root,'dist')).filter(name=>name.endsWith('.js')))fs.copyFileSync(path.join(root,'dist',file),path.join(temporary,file));
  const patch=(file,from,to)=>{const p=path.join(temporary,file),source=fs.readFileSync(p,'utf8');assert(source.includes(from),`instrumentation anchor missing: ${file}`);fs.writeFileSync(p,source.replace(from,to));};
  patch('plt-fsd.js','export function evaluateOperator(operator) {','export function evaluateOperator(operator) { throw new Error("REVIEW CPU UTD EVALUATION");');
  patch('utd-math.js',"export function coefficient(k, n, phi_i, phi_o, L, sinBeta = 1, boundary = 'soft', boundaryEps = BOUNDARY_EPS) {", "export function coefficient(k, n, phi_i, phi_o, L, sinBeta = 1, boundary = 'soft', boundaryEps = BOUNDARY_EPS) { throw new Error('REVIEW CPU UTD COEFFICIENT');");
  patch('plt-core.js','export function replayPLT(job,eventPowers=null){','export function replayPLT(job,eventPowers=null){ if(eventPowers===null) throw new Error("REVIEW CPU REFERENCE REPLAY"); if(globalThis.__pltReviewControlRemaining!==undefined && --globalThis.__pltReviewControlRemaining===0) throw new Error("REVIEW CONTROL FAILURE");');
  const {executePLTDemoTask:instrumented}=await import(pathToFileURL(path.join(temporary,'plt-demo-runtime.js')));
  let events=0;const normal=await instrumented(request,{gpuRunner:job=>{events+=job.events.length;assert.equal(job.reference,undefined);assert(job.events.every(e=>e.referenceF===undefined));return synthetic(job,5);}});
  assert(events>0);assert.equal(normal.execution.cpuOperatorEvaluations,0);assert.equal(normal.execution.cpuReferenceImages,0);assert.equal(normal.execution.cpuBatches,0);assert(normal.results.every(r=>r.image.every(v=>v===5)));
  const control=await instrumented({...request,utdAblation:true},{gpuRunner:job=>synthetic(job,5)});assert.equal(control.execution.cpuOperatorEvaluations,0);assert.equal(control.execution.cpuReferenceImages,0);assert(control.execution.cpuControlImages>0);assert(control.results[0].utdOffImage.every(Number.isFinite));
  const committed=[];globalThis.__pltReviewControlRemaining=3;
  try{await assert.rejects(instrumented({...request,utdAblation:true},{gpuRunner:job=>synthetic(job,5),preview:frame=>committed.push(frame)}),/REVIEW CONTROL FAILURE/);}
  finally{delete globalThis.__pltReviewControlRemaining;}
  assert.equal(committed.length,4);
  const prior=committed.at(-1);assert.equal(prior.results[0].sampleCounts.filter(n=>n===2).length,32);assert.equal(prior.results[1].sampleCounts.filter(n=>n===2).length,32);
  for(const frame of committed)for(const r of frame.results)for(let i=0;i<r.image.length;i++)if(r.sampleCounts[i]===0){assert.equal(r.image[i],0);if(r.utdOffImage)assert.equal(r.utdOffImage[i],0);}
  // Positive controls show the traps actually detect the forbidden operations.
  await assert.rejects(instrumented({...request,backend:'cpu'}),/REVIEW CPU/);
  await assert.rejects(instrumented({...request,debugValidation:true}),/REVIEW CPU/);
 }finally{fs.rmSync(temporary,{recursive:true,force:true});}
});

await test('deferred and strict tapes preserve complete path/term/RNG identity across all public presets',async()=>{
 for(const preset of ['gallery','prism','baffles','double-edge','micro-gallery','slitbench-demo'])for(const wave of [true,false]){
  const scene=createPLTScene({preset}),options={...request.options,wave,samplesPerPixel:4},strictRNG=createPLTRandomStates(options),lazyRNG=strictRNG.slice();
  for(const sampleStart of [0,2]){
   const args={tile:{x:0,y:0,width:8,height:8},sampleStart,samplesPerPixel:2};
   const strict=await buildPLTBatch(scene,scene.camera,options,{...args,rngStates:strictRNG}),lazy=await buildPLTBatch(scene,scene.camera,options,{...args,rngStates:lazyRNG,deferOperatorEvaluation:true});
   assert.deepEqual(lazyRNG,strictRNG);assert.deepEqual(lazy.terms,strict.terms);assert.deepEqual(lazy.paths,strict.paths);assert.deepEqual(lazy.events,strict.events.map(({referenceF,...event})=>event));
   assert.deepEqual(materializePLTBatchReference(lazy),strict.reference);assert.deepEqual(lazy.stats,strict.stats);
  }
 }
});

await test('every fallback position retains accepted GPU tiles without duplicate samples or UTD-off ownership drift',async()=>{
 const base=await executePLTDemoTask({...request,backend:'cpu',utdAblation:true});
 for(const failAt of [1,2,5,8,12,16]){
  const expected=base.results.map(r=>r.image.slice());let calls=0,accepted=0;
  const result=await executePLTDemoTask({...request,utdAblation:true},{gpuRunner:job=>{
   const call=++calls;if(call===failAt)throw new Error(`REVIEW fail batch ${call}`);
   const value=100+call,reply=synthetic(job,value),mode=job.wave?0:1,original=materializePLTBatchReference(job);
   for(let i=0;i<original.length;i++)expected[mode][job.pixelIndices[i]]+=(value-original[i])*job.samplesPerPixel/request.options.samplesPerPixel;
   accepted++;return reply;
  }});
  assert.equal(calls,failAt);assert.equal(result.execution.gpuBatches,accepted);assert.equal(result.execution.cpuBatches,result.progressive.totalBatches-accepted);assert.equal(result.execution.cpuReferenceImages,result.execution.cpuBatches);
  for(let m=0;m<2;m++){close(result.results[m].image,expected[m]);assert(result.results[m].sampleCounts.every(n=>n===request.options.samplesPerPixel));}
  assert.deepEqual(result.results[0].utdOffImage,base.results[0].utdOffImage);assert.equal(result.results[1].utdOffImage,undefined);
 }
});

await test('runner result buffers and previously published preview buffers cannot modify retained camera sums',async()=>{
 const previews=[],outputs=[];let calls=0;
 const result=await executePLTDemoTask(request,{gpuRunner:job=>{for(const old of outputs){old.image.fill(999);old.eventPowers.fill(999);}const result=synthetic(job,3);outputs.push(result);calls++;return result;},preview:p=>{previews.push(structuredClone(p));for(const r of p.results){r.image.fill(777);r.sampleCounts.fill(0);}}});
 assert.equal(calls,result.progressive.totalBatches);assert(result.results.every(r=>r.image.every(v=>v===3)&&r.sampleCounts.every(n=>n===3)));assert(previews.length>1);
 assert(previews[0].results[0].image.some(v=>v===3));assert(previews[0].results[0].image.every(v=>v===0||v===3));
});

await test('finite gate checks all unused event lanes and exact edge counts before any camera acceptance',()=>{
 const job={width:8,height:8,samplesPerPixel:1,events:[{boundary:'soft',directAmplitude:0,edges:[]},{boundary:'hard',directAmplitude:1,edges:[]}],terms:[{pixel:0,weight:1,eventIds:[]}]};
 const packed=packPLTGPU(job,{validationMode:'finite'}),valid=new Float32Array(16);valid.set([1,0,1,0,1,1,1,0],8);
 assert.equal(pltEventFiniteGate(packed,valid).count,2);
 for(let lane=0;lane<8;lane++){const bad=valid.slice();bad[8+lane]=NaN;assert.throws(()=>pltEventFiniteGate(packed,bad),e=>e.name==='PLTGPUFiniteError'&&e.validation.invalidEventRecords===1);}
 for(const count of [-1,.5,1,128]){const bad=valid.slice();bad[15]=count;assert.throws(()=>pltEventFiniteGate(packed,bad),e=>e.validation.mismatchedEdgeCounts===1);}
 assert.throws(()=>pltPixelFiniteGate(packed,new Float32Array(63)),/exact Float32Array/);
 assert.throws(()=>pltPixelFiniteGate(packed,new Float32Array(64).fill(-1)),e=>e.name==='PLTGPUFiniteError'&&e.validation.negative===64);
});

await test('late fulfillment and late rejection after cancellation publish no stale frame or unhandled rejection',async()=>{
 for(const lateFailure of [false,true]){
  const entered=deferred(),gate=deferred(),ac=new AbortController(),frames=[],unhandled=[];
  const handler=error=>unhandled.push(error);process.on('unhandledRejection',handler);
  try{
   const work=executePLTDemoTask(request,{signal:ac.signal,preview:p=>frames.push(p),gpuRunner:async job=>{entered.resolve();await gate.promise;return synthetic(job,9);}});
   await entered.promise;ac.abort();await assert.rejects(work,{name:'AbortError'});if(lateFailure)gate.reject(new Error('late rejected fixture'));else gate.resolve();await tick();await tick();
   assert.equal(frames.length,0);assert.deepEqual(unhandled,[]);
  }finally{process.removeListener('unhandledRejection',handler);}
 }
});

await test('superseded worker generations never send late progress, preview, result, or failure',async()=>{
 const sent=[],runs=new Map(),scope={postMessage:message=>sent.push(message)};
 const uninstall=installPLTDemoWorker(scope,(input,hooks)=>{const gate=deferred();runs.set(input.id,{...gate,hooks});return gate.promise;});
 try{
  const old=scope.onmessage({data:{id:1}}),latest=scope.onmessage({data:{id:2}});assert(runs.get(1).hooks.signal.aborted);
  runs.get(1).hooks.progress({text:'stale',fraction:1});runs.get(1).hooks.preview({stale:true});runs.get(1).resolve({stale:true});await old;
  runs.get(2).hooks.progress({text:'current',fraction:.5});runs.get(2).resolve({current:true});await latest;
  assert.deepEqual(sent.map(m=>[m.id,m.type]),[[2,'progress'],[2,'result']]);
  const cancelled=scope.onmessage({data:{id:3}});scope.onmessage({data:{type:'cancel',id:3}});runs.get(3).reject(new Error('late failure'));await cancelled;assert.equal(sent.length,2);
 }finally{uninstall();}
});

console.log(JSON.stringify({suite:'plt-demo-independent-review',passed,failed,browserValidated:false,physicalGPUValidated:false}));
if(failed)process.exitCode=1;
