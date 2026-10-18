/** No browser or physical-GPU claim: real CPU traversal plus explicitly injected
 * CPU emulation of GPU outputs tests orchestration. Native WGSL is a separate suite. */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {createPLTScene,createPLTRandomStates,buildPLTBatch,materializePLTBatchReference,replayPLT} from '../dist/plt-core.js';
import {evaluateOperator,buildFSDOperator,evaluateFSD} from '../dist/plt-fsd.js';
import {executePLTTask} from '../dist/plt-runtime.js';
import {executePLTDemoTask} from '../dist/plt-demo-runtime.js';
const tests=[],failures=[],options={width:9,height:8,samplesPerPixel:3,maxDepth:4,seed:7391};
const request={backend:'cpu',scene:{preset:'gallery'},options,progressive:{tileSize:5,batchSamples:2,previewIntervalMs:0}};
const run=(change={},hooks={})=>executePLTDemoTask({...request,...change,options:{...options,...change.options},progressive:{...request.progressive,...change.progressive}},hooks);
const close=(a,b,tolerance=2e-13)=>{assert.equal(a.length,b.length);for(let i=0;i<a.length;i++)assert(Math.abs(a[i]-b[i])<=tolerance*Math.max(1,Math.abs(a[i]),Math.abs(b[i])),`${i}: ${a[i]} != ${b[i]}`);};
const test=async(name,body)=>{try{await body();tests.push(name);console.log(`PASS ${name}`);}catch(error){failures.push({name,error:error.stack});console.error(`FAIL ${name}: ${error.stack}`);}};
const gpuFixture=job=>{assert.equal(job.reference,undefined);assert(job.events.every(e=>e.referenceF===undefined));const eventPowers=Float32Array.from(job.events,e=>evaluateOperator(e).F);return {width:job.width,height:job.height,image:Float32Array.from(replayPLT(job,eventPowers)),eventPowers,validation:{passed:true,mode:'finite',parityChecked:false,events:{count:job.events.length},pixels:{count:job.width*job.height}},timings:{cpuReferenceProvided:false}};};
let cpu;
await test('deferred tape preserves native geometry, sampled paths, RNG and terms without CPU F or image',async()=>{
 const scene=createPLTScene({preset:'gallery'}),a=createPLTRandomStates(options),b=a.slice(),config={tile:{x:0,y:0,width:5,height:5},sampleStart:0,samplesPerPixel:2};
 const strict=await buildPLTBatch(scene,scene.camera,options,{...config,rngStates:a}),deferred=await buildPLTBatch(scene,scene.camera,options,{...config,rngStates:b,deferOperatorEvaluation:true});
 assert.deepEqual(a,b);assert.deepEqual(strict.terms,deferred.terms);assert.deepEqual(strict.paths,deferred.paths);assert.deepEqual(strict.events.map(({referenceF,...record})=>record),deferred.events);assert.equal(deferred.reference,undefined);assert.equal(deferred.timings.cpuOperatorEvaluations,0);assert.equal(deferred.timings.cpuReferenceImages,0);
 const before=b.slice(),ref=materializePLTBatchReference(deferred);assert.deepEqual(ref,strict.reference);assert.deepEqual(b,before);assert.equal(deferred.timings.cpuOperatorEvaluations,deferred.events.length);assert.equal(deferred.timings.cpuReferenceImages,1);assert.equal(materializePLTBatchReference(deferred),ref);assert.equal(deferred.timings.cpuReferenceImages,1);
 await assert.rejects(buildPLTBatch(scene,scene.camera,options,{...config,rngStates:b,deferOperatorEvaluation:true,audit:{}}),/unaudited/);
 await assert.rejects(buildPLTBatch(scene,scene.camera,{...options,spectralMode:'visible'},{...config,rngStates:b,deferOperatorEvaluation:true}),/monochrome/);
});
await test('pure CPU demo uses each tape once and reproduces strict CPU images and path statistics',async()=>{
 let gpuCalls=0;cpu=await run({}, {gpuRunner:()=>{gpuCalls++;throw new Error('CPU mode called GPU');}});assert.equal(gpuCalls,0);const strict=await executePLTTask(request);
 cpu.results.forEach((r,i)=>{close(r.image,strict.results[i].image);assert.deepEqual(r.stats,strict.results[i].stats);assert.equal(r.execution.gpuBatches,0);assert.equal(r.execution.cpuOperatorEvaluations,r.events);assert.equal(r.execution.cpuReferenceImages,r.execution.cpuBatches);assert.equal(r.validation.parityChecked,false);assert(r.sampleCounts.every(n=>n===options.samplesPerPixel));});assert.equal(cpu.execution.cpuFullShadow,false);
});
await test('zero emission remains exactly black with no fabricated or normalized light',async()=>{
 const result=await run({scene:{preset:'gallery',emission:0}});for(const r of result.results){assert(r.image.every(v=>v===0));assert.equal(r.measurement.mean,0);assert.equal(r.measurement.max,0);}
});
await test('GPU-only accepted batches retain no CPU operator evaluations or reference images',async()=>{
 let calls=0;const result=await run({backend:'auto'},{gpuRunner:async(job,hook)=>{calls++;assert.equal(hook.validationMode,'finite');return gpuFixture(job);}});
 assert.equal(calls,result.progressive.totalBatches);assert.equal(result.execution.cpuOperatorEvaluations,0);assert.equal(result.execution.cpuReferenceImages,0);assert.equal(result.execution.cpuBatches,0);assert.equal(result.execution.gpuBatches,calls);assert.equal(result.parityChecked,false);assert.equal(result.gpuFailure,null);for(const r of result.results){assert.match(r.backend,/WebGPU/);assert.equal(r.validation.parityChecked,false);assert(r.image.every(Number.isFinite));}
});
await test('later GPU rejection preserves committed GPU tiles and evaluates failed/current/future tapes once',async()=>{
 const expected=[new Float64Array(options.width*options.height),new Float64Array(options.width*options.height)],previews=[];let calls=0;
 const result=await run({backend:'auto'},{preview:p=>previews.push(p),gpuRunner:async job=>{
  calls++;const index=job.wave?0:1;if(calls===3)throw new Error('synthetic device loss');
  const r=gpuFixture(job);r.image=Float32Array.from(r.image,v=>v*1.25+.125);for(let i=0;i<job.pixelIndices.length;i++){const id=job.pixelIndices[i];expected[index][id]+=r.image[i]*job.samplesPerPixel/options.samplesPerPixel;}
  return r;
 }});
 assert.equal(calls,3);assert.equal(result.execution.gpuBatches,2);assert.equal(result.execution.cpuReferenceImages,result.execution.cpuBatches);assert.equal(result.warnings.length,1);assert.equal(result.results.reduce((n,r)=>n+r.validation.rejectedGPUBatches,0),1);
 // First accepted tile was sample range [0,2). Compare against the same tile
 // computed on CPU, then add only its GPU-vs-CPU delta to the all-CPU image.
 const scene=createPLTScene({preset:'gallery'});for(let mode=0;mode<2;mode++){const rng=createPLTRandomStates(options),job=await buildPLTBatch(scene,scene.camera,{...options,wave:mode===0},{tile:{x:0,y:0,width:5,height:5},sampleStart:0,samplesPerPixel:2,rngStates:rng});const want=cpu.results[mode].image.slice();for(let i=0;i<job.pixelIndices.length;i++){const id=job.pixelIndices[i];want[id]+=expected[mode][id]-job.reference[i]*2/options.samplesPerPixel;}close(result.results[mode].image,want);assert.match(result.results[mode].backend,/混合/);}
 assert(previews.length>2);const saved=previews[0].results[0].image.slice();result.results[0].image.fill(0);assert.deepEqual(previews[0].results[0].image,saved);
});
await test('all invalid GPU contracts fall back once without corrupting counts or camera output',async()=>{
 const bad=[r=>r.image[0]=NaN,r=>r.image[0]=-1,r=>r.image=new Float32Array(1),r=>r.width++,r=>delete r.validation,r=>r.validation.parityChecked=true,r=>r.validation.events.count++,r=>r.eventPowers=new Float32Array(r.eventPowers.length+1),r=>{if(r.eventPowers.length)r.eventPowers[0]=Infinity;else throw new Error('invalid event');}];
 for(const mutate of bad){let calls=0;const result=await run({backend:'auto'},{gpuRunner:async job=>{calls++;const r=gpuFixture(job);mutate(r);return r;}});assert.equal(calls,1);assert.equal(result.execution.gpuBatches,0);result.results.forEach((r,i)=>{close(r.image,cpu.results[i].image);assert(r.sampleCounts.every(n=>n===options.samplesPerPixel));assert.equal(r.execution.cpuOperatorEvaluations,r.events);});}
});
await test('submitted runner mutations cannot alter original lazy-fallback tape',async()=>{
 const result=await run({backend:'auto'},{gpuRunner:async job=>{job.terms.length=0;job.events.length=0;job.pixelIndices.fill(0);throw new Error('mutated injected runner');}});result.results.forEach((r,i)=>close(r.image,cpu.results[i].image));
});
await test('finite-only mode accepts finite mismatches without claiming CPU parity',async()=>{
 const result=await run({backend:'auto',compare:false},{gpuRunner:async job=>{const r=gpuFixture(job);r.image.fill(7);return r;}});close(result.results[0].image,new Float64Array(options.width*options.height).fill(7));assert.equal(result.results[0].validation.parityChecked,false);assert.equal(result.execution.cpuReferenceImages,0);
});
await test('explicit DEBUG runs unchanged strict admission with CPU references',async()=>{
 let calls=0;const result=await run({debugValidation:true,backend:'auto'},{gpuRunner:async job=>{calls++;assert(job.reference instanceof Float64Array);assert(job.events.every(e=>Number.isFinite(e.referenceF)));return {image:Float32Array.from(job.reference),validation:{passed:true},label:'injected strict reference'}}});assert.equal(result.executionMode,'debug-strict');assert(calls>0);assert(result.results.every(r=>r.validation.passed));
});
await test('uncommitted GPU batches cannot publish and cancellation rejects stale output',async()=>{
 let entered,release;const started=new Promise(r=>entered=r),gate=new Promise(r=>release=r),controller=new AbortController(),previews=[];
 const pending=run({backend:'auto'},{signal:controller.signal,preview:p=>previews.push(p),gpuRunner:async job=>{entered();await gate;return gpuFixture(job);}});await started;assert.equal(previews.length,0);controller.abort();await assert.rejects(pending,{name:'AbortError'});release();await new Promise(r=>setTimeout(r,10));assert.equal(previews.length,0);
 const ac=new AbortController();await assert.rejects(run({}, {signal:ac.signal,preview:()=>ac.abort()}),{name:'AbortError'});
});
await test('runtime cap interrupts an ignored GPU promise and pre-cancel prevents all work',async()=>{
 const start=performance.now();await assert.rejects(run({backend:'auto',progressive:{maxRuntimeMs:50}},{gpuRunner:()=>new Promise(()=>{})}),{name:'TimeoutError'});assert(performance.now()-start<1500);
 const controller=new AbortController();controller.abort();let calls=0;await assert.rejects(run({backend:'auto'},{signal:controller.signal,gpuRunner:()=>calls++}),{name:'AbortError'});assert.equal(calls,0);
});
await test('irregular tiles and sample batches preserve sample identity and physical camera',async()=>{
 for(const [tileSize,batchSamples]of [[3,1],[7,3],[32,2]]){const result=await run({progressive:{tileSize,batchSamples}});result.results.forEach((r,i)=>{close(r.image,cpu.results[i].image);assert.deepEqual(r.stats,cpu.results[i].stats);assert(r.sampleCounts.every(n=>n===options.samplesPerPixel));});}
});
await test('same-path direct-only control uses no CPU UTD reference and commits with wave samples',async()=>{
 const expected=new Float64Array(options.width*options.height),previews=[];
 const result=await run({backend:'auto',utdAblation:true},{preview:p=>previews.push(p),gpuRunner:async job=>{if(job.wave){const control=replayPLT(job,Float64Array.from(job.events,e=>e.directAmplitude**2));for(let i=0;i<control.length;i++)expected[job.pixelIndices[i]]+=control[i]*job.samplesPerPixel/options.samplesPerPixel;}return gpuFixture(job);}});
 const wave=result.results.find(r=>r.wave),ray=result.results.find(r=>!r.wave);close(wave.utdOffImage,expected);assert.equal(ray.utdOffImage,undefined);assert.equal(result.execution.cpuOperatorEvaluations,0);assert.equal(result.execution.cpuReferenceImages,0);assert.equal(result.execution.cpuControlImages,wave.execution.gpuBatches);assert.equal(wave.execution.cpuControlImages,wave.execution.gpuBatches);assert.equal(ray.execution.cpuControlImages,0);
 for(const p of previews){const w=p.results.find(r=>r.wave);assert.equal(w.utdOffImage.length,w.image.length);for(let i=0;i<w.image.length;i++){assert(Number.isFinite(w.utdOffImage[i]));if(!w.sampleCounts[i])assert.equal(w.utdOffImage[i],0);}}
 const base=await run({utdAblation:true});let calls=0;const fallback=await run({backend:'auto',utdAblation:true},{gpuRunner:async job=>{if(++calls===3)throw new Error('late control test loss');return gpuFixture(job);}});close(fallback.results[0].utdOffImage,base.results[0].utdOffImage);assert.equal(fallback.execution.cpuControlImages,base.execution.cpuControlImages);
});
await test('bounded 204-triangle demo scene executes actual paths and respects zero-source control',async()=>{
 const r=await run({scene:{preset:'slitbench-demo',emission:0},utdAblation:true,options:{samplesPerPixel:1,maxDepth:2}});assert.equal(r.scene.triangles,204);assert(r.results.every(result=>result.image.every(v=>v===0)));assert(r.results[0].utdOffImage.every(v=>v===0));
});
await test('normal visible spectrum and unsupported backend reject rather than silently switch physics',async()=>{
 await assert.rejects(run({options:{spectralMode:'visible'}}),/monochrome/);await assert.rejects(run({backend:'resident-shadow'}),/Unsupported demo backend/);await assert.rejects(run({debugValidation:'yes'}),/boolean/);
});
let nativeExecution=null;
if(process.argv.includes('--execute'))await test('actual native WGSL reference-free gallery and 3D demo equal strict GPU images without shadow',async()=>{
 const {create,globals}=await import('webgpu');Object.assign(globalThis,globals);const {createPLTGPUBatchRunner}=await import('../dist/plt-gpu.js');
 const source=['utd-functions.wgsl','plt-compute.wgsl'].map(name=>fs.readFileSync(new URL(`../dist/${name}`,import.meta.url),'utf8')).join('\n'),gpu=create(['backend=vulkan']);nativeExecution=[];
 for(const [preset,width,height,samplesPerPixel]of [['gallery',8,8,2],['slitbench-demo',16,12,4],['slitbench-demo',32,24,2]]){
  const finite=createPLTGPUBatchRunner({gpu,shaderSource:source,validationMode:'finite'}),strict=createPLTGPUBatchRunner({gpu,shaderSource:source});
  const input={backend:'auto',compare:true,scene:{preset},options:{width,height,samplesPerPixel,maxDepth:2,wavelengthMm:.2,seed:7391},progressive:{tileSize:8,batchSamples:1,previewIntervalMs:0}};
  try{const result=await executePLTDemoTask({...input,utdAblation:true},{gpuRunner:job=>finite.run(job)}),reference=await executePLTTask(input,{gpuRunner:job=>strict.run(job)});assert.equal(result.execution.cpuOperatorEvaluations,0);assert.equal(result.execution.cpuReferenceImages,0);assert.equal(result.execution.cpuBatches,0);assert(reference.results.every(r=>r.validation.passed));result.results.forEach((r,i)=>assert.deepEqual(r.image,reference.results[i].image));nativeExecution.push({preset,width,height,samplesPerPixel,triangles:result.scene.triangles,backend:'Dawn Vulkan software execution',physicalGPUValidated:false,cpuOperatorEvaluations:result.execution.cpuOperatorEvaluations,cpuReferenceImages:result.execution.cpuReferenceImages,gpuBatches:result.execution.gpuBatches,utdControlImages:result.execution.cpuControlImages,imagesEqualStrictGPU:true,events:result.results.map(r=>r.events),totalMs:result.totalMs});}finally{finite.destroy();strict.destroy();}
 }
});
const report={suite:'plt-demo-runtime',generatedAt:new Date().toISOString(),nativeExecution,passed:tests.length,failed:failures.length,tests,failures,browserValidated:false,physicalGPUValidated:false,scope:'Real native CPU path construction and explicit CPU-emulated GPU-output hooks; not hardware execution or physical accuracy validation'};
fs.writeFileSync(new URL('../research/plt-demo-runtime-validation.json',import.meta.url),JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify({passed:tests.length,failed:failures.length}));if(failures.length)process.exitCode=1;
