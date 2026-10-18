/** Actual CPU paths and deterministic accumulation tests. GPU runner tests below
 * are injected contracts; they do not claim device/WGSL execution. */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {executePLTTask,validatePLTProgressiveOptions} from '../dist/plt-runtime.js';
import {createPLTScene,orbitCamera,buildPLTJob,buildPLTBatch,createPLTRandomStates,validatePLTOptions,PLT_PROGRESSIVE_LIMITS,PLT_LIMITS} from '../dist/plt-core.js';
const report={generatedAt:new Date().toISOString(),scope:'Bounded CPU path tapes, sample-prefix continuation, verified-only previews, cancellation and stale worker suppression. Injected GPU tests are contract tests, not device execution.',checks:[],failures:[]};
const test=async(name,fn)=>{const start=performance.now();try{const evidence=await fn();report.checks.push({name,ms:performance.now()-start,evidence});console.log('PASS',name);}catch(error){report.failures.push({name,error:error.stack});console.error('FAIL',name,error);}};
const options={width:8,height:8,samplesPerPixel:5,maxDepth:4,seed:7391};
const request={options,backend:'cpu',scene:{preset:'gallery'},compare:true,progressive:{tileSize:4,batchSamples:2,previewIntervalMs:0}};
const run=(changes={},hooks={})=>executePLTTask({...request,...changes,options:{...options,...changes.options},progressive:changes.progressive===false?false:{...request.progressive,...changes.progressive}},hooks);
let canonical;
await test('CPU progressive image and path statistics exactly match the legacy full-frame calculation in both modes',async()=>{
 canonical=await run({progressive:false});const result=await run();for(const [i,mode] of result.results.entries()){assert.deepEqual(mode.image,canonical.results[i].image);assert.deepEqual(mode.stats,canonical.results[i].stats);assert.equal(mode.events,canonical.results[i].events);assert.equal(mode.terms,canonical.results[i].terms);assert(mode.sampleCounts.every(n=>n===options.samplesPerPixel));assert.equal(mode.completedSamplesPerPixel,options.samplesPerPixel);assert.equal(mode.partial,false);}return {modes:2,bitIdentical:true,options};
});
await test('Tile sizes and sample batches preserve the exact per-global-pixel v11 RNG sequence and CPU summation',async()=>{
 const tested=[];for(const tileSize of [1,3,8,32])for(const batchSamples of [1,2,4]){const result=await run({progressive:{tileSize,batchSamples,previewIntervalMs:5000}});result.results.forEach((r,i)=>assert.deepEqual(r.image,canonical.results[i].image));tested.push({tileSize,batchSamples,batches:result.progressive.completedBatches});}return tested;
});
await test('Non-divisible tiles use full-frame camera geometry, not tile-sized sensor footprints',async()=>{
 const o={...options,width:13,height:11,samplesPerPixel:3},scene=createPLTScene(),camera=orbitCamera(scene.camera),legacy=await buildPLTJob(scene,camera,o),result=await run({compare:false,options:o,progressive:{tileSize:5,batchSamples:2}});assert.deepEqual(result.results[0].image,legacy.reference);return {frame:[13,11],tileSize:5,exact:true};
});
await test('Second sample batch advances persistent RNG and contributes new paths',async()=>{
 const scene=createPLTScene(),camera=orbitCamera(scene.camera),o={...options,samplesPerPixel:2},rngStates=createPLTRandomStates(o),initial=rngStates.slice(),tile={x:0,y:0,width:8,height:8};
 const first=await buildPLTBatch(scene,camera,o,{tile,rngStates,sampleStart:0,samplesPerPixel:1}),statesAfterFirst=rngStates.slice(),second=await buildPLTBatch(scene,camera,o,{tile,rngStates,sampleStart:1,samplesPerPixel:1});
 assert.notDeepEqual(initial,statesAfterFirst);assert.notDeepEqual(statesAfterFirst,rngStates);assert.notDeepEqual(first.reference,second.reference);assert(second.paths.every(p=>p.sample===1));return {firstMean:first.reference.reduce((s,x)=>s+x,0)/64,secondMean:second.reference.reduce((s,x)=>s+x,0)/64};
});
await test('Preview snapshots contain honest committed counts, immutable images, and only complete shared passes',async()=>{
 const previews=[];await run({}, {preview:p=>{previews.push(p);for(const mode of p.results){assert.equal(mode.partial,true);assert.equal(mode.minSamplesPerPixel,Math.min(...mode.sampleCounts));assert.equal(mode.maxSamplesPerPixel,Math.max(...mode.sampleCounts));assert.equal(mode.completedSamplesPerPixel,mode.minSamplesPerPixel);assert(mode.image.every((v,i)=>Number.isFinite(v)&&v>=0&&(mode.sampleCounts[i]>0||v===0)));}assert.equal(p.completedSamplesPerPixel,Math.min(...p.results.map(r=>r.minSamplesPerPixel)));}});
 assert.equal(previews[0].completedSamplesPerPixel,0);assert(previews[0].results[0].sampleCounts.some(n=>n===0));assert(previews[0].results[1].sampleCounts.every(n=>n===0));assert.equal(previews.at(-1).completedSamplesPerPixel,5);assert.equal(previews[0].results[0].sampleCounts.filter(n=>n===2).length,16);assert.deepEqual(previews.slice(0,8).map(p=>p.batch.wave),[true,false,true,false,true,false,true,false]);return {previews:previews.length,firstFullPassBatch:previews.find(p=>p.completedSamplesPerPixel>0).batch.index,finalSharedSPP:5};
});
await test('No preview is emitted before the GPU batch admission resolves',async()=>{
 let entered,release;const started=new Promise(r=>entered=r),gate=new Promise(r=>release=r),previews=[];
 const work=run({compare:false,options:{samplesPerPixel:1},backend:'auto',progressive:{tileSize:8}},{preview:p=>previews.push(p),gpuRunner:async job=>{entered();await gate;return {image:Float32Array.from(job.reference),validation:{passed:true}};}});
 await started;assert.equal(previews.length,0);release();const result=await work;assert.equal(previews.length,1);assert(result.results[0].validation.passed);return {previewsBeforeAdmission:0,previewsAfterAdmission:1};
});
await test('All invalid GPU camera contracts fail closed and disable later GPU attempts',async()=>{
 const outcomes=[];for(const payload of [j=>({image:Float32Array.from(j.reference),validation:{passed:false}}),j=>({image:new Float32Array(j.reference.length-1),validation:{passed:true}}),j=>({image:new Float32Array(j.reference.length).fill(NaN),validation:{passed:true}}),j=>({image:new Float32Array(j.reference.length).fill(-1),validation:{passed:true}})]){
  let calls=0;const result=await run({backend:'auto'},{gpuRunner:async job=>{calls++;return payload(job);}});assert.equal(calls,1);assert.equal(result.warnings.length,1);result.results.forEach((r,i)=>{assert.deepEqual(r.image,canonical.results[i].image);assert.equal(r.validation.passed,false);assert.equal(r.backend,'CPU 双精度');});outcomes.push({calls,warnings:result.warnings.length});}return outcomes;
});
await test('A later GPU rejection restores exact CPU sums for previously accepted tiles',async()=>{
 let calls=0;const previews=[];const result=await run({backend:'auto'},{preview:p=>previews.push(p),gpuRunner:async job=>{if(++calls===3)throw Object.assign(new Error('synthetic late loss'),{diagnostic:{stage:'every-event',array:new Float64Array(10000)}});return {image:Float32Array.from(job.reference,v=>v*1.00001),validation:{passed:true},label:'test GPU'};}});
 assert.equal(calls,3);assert.equal(result.warnings.length,1);result.results.forEach((r,i)=>{assert.deepEqual(r.image,canonical.results[i].image);assert.equal(r.validation.passed,false);assert.equal(r.backend,'CPU 双精度');});assert(previews.slice(2).every(p=>p.results.every(r=>r.backend==='CPU 双精度')));return {gpuCalls:calls,finalCPUExact:true};
});
await test('Aggregated validation retains exhaustive counts and bounded summaries, not per-event error tapes',async()=>{
 let totalEvents=0,totalPixels=0;const result=await run({backend:'auto'},{gpuRunner:async job=>{totalEvents+=job.events.length;totalPixels+=job.reference.length;return {image:Float32Array.from(job.reference),validation:{passed:true,events:{count:job.events.length,absoluteErrors:new Float64Array(job.events.length),toleranceRatios:new Float64Array(job.events.length),maxAbsoluteError:.000001,maxToleranceRatio:.25},pixels:{count:job.reference.length,absoluteErrors:new Float64Array(job.reference.length),toleranceRatios:new Float64Array(job.reference.length),maxAbsoluteError:.000002,maxToleranceRatio:.5}}};}});
 assert.equal(result.results.reduce((s,r)=>s+r.validation.events.count,0),totalEvents);assert.equal(result.results.reduce((s,r)=>s+r.validation.pixels.count,0),totalPixels);for(const r of result.results){const json=JSON.stringify(r.validation);assert(!json.includes('absoluteErrors'));assert(!json.includes('toleranceRatios'));assert(json.length<1500);assert.equal(r.validation.events.maxToleranceRatio,.25);}return {totalEvents,totalPixels,retainedValidationBytes:result.results.reduce((s,r)=>s+JSON.stringify(r.validation).length,0)};
});
await test('Cancellation at preview rejects without a later preview or finalized image',async()=>{
 const ac=new AbortController();let previews=0;await assert.rejects(run({}, {signal:ac.signal,preview:()=>{previews++;ac.abort();}}),{name:'AbortError'});assert.equal(previews,1);return {previews};
});
await test('Cancellation while awaiting GPU cannot publish the returned stale image',async()=>{
 const ac=new AbortController();let previews=0;await assert.rejects(run({backend:'auto'},{signal:ac.signal,preview:()=>previews++,gpuRunner:async job=>{ac.abort();return {image:job.reference,validation:{passed:true}};}}),{name:'AbortError'});assert.equal(previews,0);return {previews};
});
await test('Runtime cap rejects incomplete work without a final result',async()=>{
 await assert.rejects(run({options:{width:32,height:24,samplesPerPixel:512},progressive:{maxRuntimeMs:1}}),{name:'TimeoutError'});return {capMs:1};
});
await test('Runtime cap also interrupts a GPU promise that ignores its cancellation signal',async()=>{
 let gpuEntered=false;const start=performance.now();await assert.rejects(run({backend:'auto',progressive:{maxRuntimeMs:100}},{gpuRunner:()=>{gpuEntered=true;return new Promise(()=>{});}}),{name:'TimeoutError'});assert(gpuEntered);assert(performance.now()-start<1000);return {capMs:100,elapsedMs:performance.now()-start,gpuEntered};
});
await test('Worker replacement and cancellation suppress stale result and preview messages',async()=>{
 const oldSelf=globalThis.self,messages=[];globalThis.self={postMessage:m=>messages.push(m)};
 try{await import('../dist/plt-worker.js');const long=self.onmessage({data:{...request,id:1,options:{...options,width:24,height:24,samplesPerPixel:8}}});const short=self.onmessage({data:{...request,id:2,options:{...options,samplesPerPixel:1}}});await Promise.all([long,short]);assert(!messages.some(m=>m.id===1&&['preview','result'].includes(m.type)));assert(messages.some(m=>m.id===2&&m.type==='result'));messages.length=0;const cancelled=self.onmessage({data:{...request,id:3}});await self.onmessage({data:{type:'cancel',id:3}});await cancelled;assert(!messages.some(m=>m.id===3&&['preview','result','error'].includes(m.type)));return {staleFinals:0,cancelledFinals:0};}finally{globalThis.self=oldSelf;}
});
await test('Legacy full-frame cap stays unchanged; progressive-only frame and work budgets remain bounded',async()=>{
 assert.throws(()=>validatePLTOptions({...options,width:256,height:192}));assert.equal(PLT_LIMITS.maxSamples,512);assert.equal(validatePLTOptions({...options,width:256,height:192},{progressive:true}).width,256);
 for(const invalid of [{width:257},{height:193},{samplesPerPixel:513}])assert.throws(()=>validatePLTOptions({...options,...invalid},{progressive:true}));for(const invalid of [{tileSize:33},{batchSamples:5},{maxRuntimeMs:600001},{previewIntervalMs:5001}])assert.throws(()=>validatePLTProgressiveOptions(invalid));return PLT_PROGRESSIVE_LIMITS;
});
await test('Useful 256×192 frame executes both physical modes through bounded 32×32 tapes and complete sample passes',async()=>{
 const r=await run({options:{width:256,height:192,samplesPerPixel:2,maxDepth:3},progressive:{tileSize:32,batchSamples:1,maxRuntimeMs:120000,previewIntervalMs:5000}}),p=r.progressive;
 assert.equal(p.completedSamplesPerPixel,2);assert.equal(p.completedBatches,192);assert(p.peakTape.tilePixels<=1024);assert(p.peakTape.batchPaths<=1024);assert(p.peakTape.events<=PLT_PROGRESSIVE_LIMITS.maxEvents);assert(p.peakTape.terms<=PLT_PROGRESSIVE_LIMITS.maxTerms);assert(p.peakTape.edgeRecords<=PLT_PROGRESSIVE_LIMITS.maxEdgeRecords);assert(p.peakTape.retainedPaths<=24);assert(p.peakTape.retainedEvidence<=16);assert(p.firstPreviewMs>0);assert(p.firstFullPassMs>=p.firstPreviewMs);assert(r.results.every(x=>x.sampleCounts.every(n=>n===2)&&x.image.every(v=>Number.isFinite(v)&&v>=0)));return {options:r.options,totalMs:r.totalMs,progressive:p,modes:r.results.map(({wave,events,terms,measurement})=>({wave,events,terms,measurement}))};
});
fs.writeFileSync(new URL('../research/plt-progressive-validation.json',import.meta.url),JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify({passed:report.checks.length,failed:report.failures.length}));if(report.failures.length)process.exitCode=1;
