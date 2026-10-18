/** Independent review of work-reusing progression. Real source/scheduler/CPU
 * checks are separated from labelled orchestration mocks. No hardware claim. */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {createHash} from 'node:crypto';
import {planPinholeContributions} from '../dist/pinhole-progress.js';
import {createPinholeScene,buildPinholeSourceLayers} from '../dist/pinhole-scene.js';
import {spectralQuadrature} from '../dist/pinhole-optics.js';
import {createPinholeCPURunner} from '../dist/pinhole-cpu.js';
import {executePinholeTask} from '../dist/pinhole-runtime.js';
import {installPinholeWorker} from '../dist/pinhole-worker.js';

const tests=[],failures=[],evidence={schemaVersion:1,startedAt:new Date().toISOString(),nativeGPUExecuted:false,browserExecuted:false};
const production=['pinhole-progress.js','pinhole-runtime.js','pinhole-cpu.js','pinhole-gpu.js','pinhole-worker.js'];
const hashes=()=>Object.fromEntries(production.map(name=>[name,createHash('sha256').update(fs.readFileSync(new URL(`../dist/${name}`,import.meta.url))).digest('hex')]));
const initialHashes=hashes();
async function test(name,fn){try{await fn();tests.push(name);console.log('PASS',name);}catch(error){failures.push({name,error:String(error.stack).replace(/data:text\/javascript,[^\s)]+/g,'<mock-runtime-module>')});console.error('FAIL',name,error.message);}}
const pause=(ms=0)=>new Promise(resolve=>setTimeout(resolve,ms));
const deferred=()=>{let resolve;const promise=new Promise(r=>resolve=r);return {promise,resolve};};
function relative(actual,expected,tolerance=1e-11){assert.equal(actual.length,expected.length);let peak=0,error=0;for(let i=0;i<actual.length;i++){assert(Number.isFinite(actual[i]));peak=Math.max(peak,Math.abs(expected[i]));error=Math.max(error,Math.abs(actual[i]-expected[i]));}assert(error<=Math.max(1e-30,peak)*tolerance,`relative error ${error/peak}`);return error/Math.max(peak,1e-30);}
function direct(source,n,kernel,m){const output=new Float64Array(n*n);for(let y=0;y<n;y++)for(let x=0;x<n;x++)for(let sy=0;sy<n;sy++)for(let sx=0;sx<n;sx++)output[y*n+x]+=source[sy*n+sx]*kernel[((y-sy+m)%m)*m+(x-sx+m)%m];return output;}

await test('scheduler retains each input exactly once, bounded checkpoints and independently counted support',async()=>{
 const tasks=[{depth:0,basis:0,sourceY:1,values:new Float32Array([1,0,0,0])},{depth:1,basis:0,sourceY:4,values:new Float32Array([2,3,0,0])},{depth:2,basis:0,sourceY:2,values:new Float32Array([1,1,0,0])},{depth:0,basis:1,sourceY:3,values:new Float32Array([0,0,1,2])}];
 const before=structuredClone(tasks),plan=await planPinholeContributions(tasks),active=new Set(),seen=new Set();
 for(const t of tasks)for(let p=0;p<t.values.length;p++)if(t.values[p]>0)active.add(`${t.basis}:${p}`);
 assert.equal(new Set(plan.tasks.map(t=>t.originalIndex)).size,tasks.length);assert.equal(plan.tasks.length,tasks.length);assert.deepEqual(tasks,before);
 let cumulative=0,previousY=0,firstFull=0;for(const [i,t]of plan.tasks.entries()){
  assert.strictEqual(t.values,tasks[t.originalIndex].values);cumulative+=t.sourceY;
  for(let p=0;p<t.values.length;p++)if(t.values[p]>0)seen.add(`${t.basis}:${p}`);
  assert.equal(t.representedSourceFraction,seen.size/active.size);assert.equal(t.sourcePowerFraction,cumulative/10);assert(t.sourcePowerFraction>=previousY);previousY=t.sourcePowerFraction;
  if(!firstFull&&seen.size===active.size)firstFull=i+1;
 }
 assert.equal(plan.firstFullSourceSupport,firstFull);assert.equal(plan.activeSourcePairs,active.size);assert.equal(plan.totalSourceY,10);
 assert(plan.checkpoints.length<=2);assert.equal(new Set(plan.checkpoints).size,plan.checkpoints.length);assert(plan.checkpoints.every(x=>x>=firstFull&&x<tasks.length));
 assert.deepEqual((await planPinholeContributions([])).tasks,[]);assert.deepEqual((await planPinholeContributions([tasks[0]])).checkpoints,[]);
 const c=new AbortController();c.abort();await assert.rejects(planPinholeContributions(tasks,{signal:c.signal}),e=>e.name==='AbortError');
});

await test('real native512 first preview represents every active source pair without renormalizing its light',async()=>{
 const start=performance.now(),source=await buildPinholeSourceLayers(createPinholeScene(),{resolution:512,depthCount:8,diameterMm:.2});
 const colors=source.basisNames.map((_,b)=>spectralQuadrature({basisNames:source.basisNames}).reduce((sum,s)=>sum+s.basisXYZ[b][1],0)),tasks=[];
 for(let depth=0;depth<source.layers.length;depth++)for(let basis=0;basis<source.basisNames.length;basis++){const values=source.layers[depth].basis[basis],sum=values.reduce((s,v)=>s+v,0);if(sum>0)tasks.push({depth,basis,values,sourceY:sum*colors[basis]});}
 const plan=await planPinholeContributions(tasks),first=plan.checkpoints[0],selected=plan.tasks.slice(0,first);let active=0,represented=0,totalY=0,retainedY=0,minFraction=1;
 for(let b=0;b<source.basisNames.length;b++)for(let p=0;p<512**2;p++){
  const total=tasks.filter(t=>t.basis===b).reduce((sum,t)=>sum+t.values[p],0),partial=selected.filter(t=>t.basis===b).reduce((sum,t)=>sum+t.values[p],0);
  totalY+=total*colors[b];retainedY+=partial*colors[b];if(total>0){active++;if(partial>0)represented++;minFraction=Math.min(minFraction,partial/total);}
 }
 assert.equal(tasks.length,13);assert.equal(first,5);assert.equal(active,35419);assert.equal(represented,active);assert(minFraction>.35);assert(retainedY<totalY);assert(Math.abs(selected.at(-1).sourcePowerFraction-retainedY/totalY)<1e-12);
 evidence.nativeSource={resolution:512,tasks:tasks.length,checkpoints:plan.checkpoints,activeSourcePairs:active,representedSourcePairs:represented,firstSourceYFraction:retainedY/totalY,minimumRetainedDepthMassFraction:minFraction,elapsedMs:performance.now()-start};
});

await test('real CPU nondestructive snapshots match direct sums, preserve native amplitude and final results',async()=>{
 const n=16,m=32,a=Float64Array.from({length:n*n},(_,i)=>(i*17%31)/31),b=Float64Array.from(a,x=>x*.25),kernel=new Float64Array(m*m);kernel[0]=.7;kernel[2*m+1]=.2;kernel[(m-3)*m+m-2]=.1;
 const kernels=Array.from({length:6},(_,channel)=>({values:kernel,channel,weight:channel+1})),progressive=createPinholeCPURunner({width:n}),plain=createPinholeCPURunner({width:n});
 try{
  await progressive.convolveAccumulate({source:a,kernels});const snapshot=await progressive.snapshotResults(),expected=direct(a,n,kernel,m);
  for(let c=0;c<6;c++)relative(snapshot.channels[c],Float64Array.from(expected,x=>x*(c+1)));
  snapshot.channels.forEach(channel=>channel.fill(NaN));await progressive.convolveAccumulate({source:b,kernels});
  const second=await progressive.snapshotResults(),final=await progressive.readResults();for(let c=0;c<6;c++)relative(second.channels[c],final.channels[c],0);
  await plain.convolveAccumulate({source:a,kernels});await plain.convolveAccumulate({source:b,kernels});const baseline=await plain.readResults();
  for(let c=0;c<6;c++){relative(final.channels[c],baseline.channels[c],0);relative(final.channels[c],Float64Array.from(expected,x=>1.25*x*(c+1)));}
  assert.equal(final.stats.convolutions,12);assert.equal(final.stats.transforms,baseline.stats.transforms);assert.equal(final.stats.snapshotTransforms,12);assert.equal(final.stats.snapshotScratchComplexBytes,2*m*m*8);
  await assert.rejects(progressive.snapshotResults(),/closed/);evidence.cpuSnapshot={exactFinalParity:true,stats:final.stats};
 }finally{progressive.destroy();plain.destroy();}
});

await test('real CPU runtime checkpoints preserve final parity, unscaled values and increasing partial irradiance',async()=>{
 const request={resolution:16,spectralMode:'narrowband',depthLayers:2,backend:'cpu'},previews=[];
 const baseline=await executePinholeTask(request),result=await executePinholeTask(request,{preview:frame=>previews.push(frame)});
 assert.equal(result.stages.length,1);assert.deepEqual(result.diagnostics.progressiveWork.resolutions,[16]);assert(previews.length>0&&previews.length<=2);
 for(const key of ['waveXYZ','geometricXYZ']){relative(result[key],baseline[key],0);let previous=new Float64Array(result[key].length);for(const frame of [...previews,result]){const tolerance=Math.max(...frame[key])*1e-10+1e-25;for(let i=0;i<previous.length;i++)assert(frame[key][i]+tolerance>=previous[i]);previous=frame[key];}}
 for(const frame of previews){assert.equal(frame.width,16);assert.equal(frame.complete,false);assert.equal(frame.progressive.representedSourceFraction,1);assert(frame.progressive.sourcePowerFraction<1);assert.match(frame.diagnostics.previewMeaning,/unscaled/);assert.equal(frame.diagnostics.cpuShadow,false);}
 assert.equal(result.diagnostics.convolutionStats.convolutions,baseline.diagnostics.convolutionStats.convolutions);assert.equal(result.diagnostics.convolutionStats.transforms,baseline.diagnostics.convolutionStats.transforms);
 evidence.realRuntime={resolution:16,checkpoints:previews.map(p=>p.progressive),exactFinalParity:true,extraSnapshotTransforms:result.diagnostics.convolutionStats.snapshotTransforms};
});

// Physics imports below are deliberately replaced with simple impulse inputs.
// These cases prove native-grid orchestration, not optical correctness or GPU execution.
globalThis.__pinholeProgressiveReview=new Map();let fixtureIndex=0;
async function fixture(){
 const id=++fixtureIndex,log={sceneWidths:[],kernelWidths:[],kernelTasks:[],gpu:[],cpu:[],events:[]};
 const makeRunner=(width,kind,{failAt,readMutation,holdRead}={})=>{
  const record={kind,width,calls:[],reads:0,destroyed:0},channels=Array.from({length:6},()=>new Float64Array(width**2));log[kind].push(record);
  const read=async()=>{record.reads++;if(holdRead)await holdRead;const result={channels:channels.map(a=>a.slice()),backend:kind==='gpu'?{kind:'webgpu',label:'MOCK ONLY'}:'MOCK CPU',validation:kind==='gpu'?{passed:true}:{finite:true},stats:{mock:true,calls:record.calls.length}};readMutation?.(result);return result;};
  return {record,async convolveAccumulate({source,kernels}){record.calls.push(source[0]);log.events.push(`${kind}:convolve:${record.calls.length}`);if(record.calls.length===failAt)throw new Error('MOCK contribution failure');for(const kernel of kernels)for(let p=0;p<source.length;p++)channels[kernel.channel][p]+=source[p]*kernel.values[0];},readResults:read,snapshotResults:read,destroy(){record.destroyed++;}};
 };
 const f={
  createPinholeScene:()=>({}),
  async buildPinholeSourceLayers(_,o){log.sceneWidths.push(o.resolution);return {layers:[0,1].map(d=>({basis:[0,1].map(b=>new Float32Array(o.resolution**2).fill(1+2*d+b))})),depthsMm:[220,270],basisNames:['neutral','red'],overview:{objects:[{label:'MOCK'}]},metadata:{mock:true}};},
  async generateKernels(o){log.kernelWidths.push(o.resolution);return {...o,layers:[0,1].map(()=>({basis:[0,1].map(()=>({colorXYZ:[1,1,1]}))})),throughput:1,pixelAreaMm2:(8/o.resolution)**2,pixelWidthMm:8/o.resolution,areaMm2:1,spectralSamples:1,spectralStepNm:5,radialStepMm:1,radialCount:2,normalization:'MOCK'};},
  async buildPaddedKernel(k,d,b,{kind}){log.kernelTasks.push([d,b,kind]);return {channels:[0,1,2].map(c=>{const v=new Float32Array(4*k.resolution**2);v[0]=1+c+(kind==='geometric'?3:0);return v;})};},
  renderPointDetector:()=>{throw new Error('MOCK point not supported');},
  createPinholeCPURunner:({width})=>makeRunner(width,'cpu')
 };
 globalThis.__pinholeProgressiveReview.set(id,f);
 const moduleURL=names=>'data:text/javascript,'+encodeURIComponent(names.map(name=>name==='PINHOLE_OPTICS_SCOPE'?`export const ${name}='MOCK ONLY';`:`export const ${name}=(...args)=>globalThis.__pinholeProgressiveReview.get(${id}).${name}(...args);`).join('\n'));
 let source=fs.readFileSync(new URL('../dist/pinhole-runtime.js',import.meta.url),'utf8');
 for(const [file,names]of [['scene',['createPinholeScene','buildPinholeSourceLayers']],['optics',['generateKernels','buildPaddedKernel','renderPointDetector','PINHOLE_OPTICS_SCOPE']],['cpu',['createPinholeCPURunner']]])source=source.replace(`'./pinhole-${file}.js'`,JSON.stringify(moduleURL(names)));
 source=source.replace("'./pinhole-progress.js'",JSON.stringify(new URL('../dist/pinhole-progress.js',import.meta.url).href));
 return {execute:(await import('data:text/javascript,'+encodeURIComponent(source)+`#${id}`)).executePinholeTask,log,makeRunner};
}

await test('MOCK native512 GPU snapshots do not repeat work, add gain, or expose mutable state',async()=>{
 const {execute,log,makeRunner}=await fixture(),previews=[];
 const result=await execute({resolution:512},{gpuFactory:({width})=>makeRunner(width,'gpu'),preview:frame=>{
  previews.push(structuredClone(frame));frame.waveXYZ.fill(NaN);frame.geometricXYZ.fill(-1);frame.configuration.diameterMm=99;frame.sceneOverview.objects[0].label='mutated';frame.backend.kind='mutated';frame.timings.totalMs=-1;frame.progressive.sourcePowerFraction=99;
 }});
 assert.deepEqual(log.sceneWidths,[512]);assert.deepEqual(log.kernelWidths,[512]);assert.equal(log.gpu.length,1);assert.equal(log.cpu.length,0);assert.deepEqual(log.gpu[0].calls,[4,3,2,1]);assert.equal(log.gpu[0].reads,3);assert.equal(log.gpu[0].destroyed,1);assert.equal(log.kernelTasks.length,8);
 assert.deepEqual(previews.map(p=>p.progressive.completedContributions),[2,3]);assert.deepEqual(previews.map(p=>p.waveXYZ[1]),[14,18]);assert.equal(result.waveXYZ[1],20);assert.equal(result.geometricXYZ[1],50);
 assert.equal(result.configuration.diameterMm,.2);assert.equal(result.sceneOverview.objects[0].label,'MOCK');assert.equal(result.backend.kind,'webgpu');assert.equal(result.progressive.sourcePowerFraction,1);assert.equal(result.stages.length,1);
 assert.equal(result.diagnostics.progressiveWork.extraGPUConvolutionFFTs,0);assert.equal(result.diagnostics.cpuReferenceImages,0);
});

await test('MOCK GPU failure after a published preview resets before fresh CPU contributions',async()=>{
 const {execute,log,makeRunner}=await fixture(),publications=[];
 const result=await execute({resolution:16},{gpuFactory:({width})=>makeRunner(width,'gpu',{failAt:3}),preview:frame=>publications.push(`preview:${frame.progressive.attempt}:${frame.progressive.completedContributions}`),reset:value=>{publications.push(`reset:${value.attempt}`);log.events.push('reset');}});
 assert.deepEqual(publications,['preview:0:2','reset:1','preview:1:2','preview:1:3']);assert(log.events.indexOf('reset')<log.events.indexOf('cpu:convolve:1'));
 assert.deepEqual(log.cpu[0].calls,[4,3,2,1]);assert.equal(log.gpu[0].destroyed,1);assert.equal(log.cpu[0].destroyed,1);assert.equal(result.waveXYZ[1],20);assert.equal(result.backend.kind,'cpu');assert.equal(result.progressive.attempt,1);
 assert.equal(log.sceneWidths.length,1);assert.equal(log.kernelWidths.length,1);assert.equal(result.diagnostics.fallback.attemptedGPU,true);
});

await test('MOCK invalid snapshot data is withheld and triggers a reset before CPU replacement',async()=>{
 for(const mutate of [r=>r.channels.pop(),r=>r.channels[0][0]=NaN,r=>r.channels[0][0]=-1,r=>r.validation.passed=false]){
  const {execute,log,makeRunner}=await fixture(),previews=[],resets=[];
  const result=await execute({resolution:16},{gpuFactory:({width})=>makeRunner(width,'gpu',{readMutation:mutate}),preview:frame=>previews.push(frame),reset:value=>resets.push(value)});
  assert.equal(resets.length,1);assert(previews.every(p=>p.backend.kind==='cpu'&&p.progressive.attempt===1));assert.equal(result.waveXYZ[1],20);assert.equal(log.gpu[0].destroyed,1);
 }
});

await test('MOCK cancellation from preview rejects the task without further contributions or fallback',async()=>{
 const {execute,log,makeRunner}=await fixture(),controller=new AbortController();let count=0,resets=0;
 await assert.rejects(execute({resolution:16},{signal:controller.signal,gpuFactory:({width})=>makeRunner(width,'gpu'),preview:()=>{count++;controller.abort();},reset:()=>resets++}),e=>e.name==='AbortError');
 assert.equal(count,1);assert.equal(log.gpu[0].calls.length,2);assert.equal(log.gpu[0].destroyed,1);assert.equal(log.cpu.length,0);assert.equal(resets,0);
});

await test('MOCK cancelled uncooperative snapshot cannot publish a late preview or result',async()=>{
 const {execute,log,makeRunner}=await fixture(),controller=new AbortController(),held=deferred();let previews=0,resets=0;
 const run=execute({resolution:16},{signal:controller.signal,gpuFactory:({width})=>makeRunner(width,'gpu',{holdRead:held.promise}),preview:()=>previews++,reset:()=>resets++});
 while(!log.gpu[0]?.reads)await pause();controller.abort();await assert.rejects(run,e=>e.name==='AbortError');held.resolve();await pause();assert.equal(previews,0);assert.equal(resets,0);assert.equal(log.cpu.length,0);assert.equal(log.gpu[0].destroyed,1);
});

await test('worker suppresses stale resets as well as stale previews and final results',async()=>{
 const runs=[],messages=[],scope={postMessage:m=>messages.push(m)},dispose=installPinholeWorker(scope,(_,hooks)=>{const d=deferred();runs.push({...d,hooks});return d.promise;});
 try{const old=scope.onmessage({data:{id:1}}),fresh=scope.onmessage({data:{id:2}});runs[0].hooks.reset({attempt:1});runs[0].hooks.preview({complete:false});runs[0].resolve({complete:true});await old;assert.equal(messages.length,0);runs[1].hooks.reset({attempt:1});runs[1].hooks.preview({complete:false});runs[1].resolve({complete:true});await fresh;assert.deepEqual(messages.map(m=>[m.id,m.type]),[[2,'reset'],[2,'preview'],[2,'result']]);}finally{dispose();}
});

await test('production review targets remained unchanged during execution',()=>assert.deepEqual(hashes(),initialHashes));
evidence.passed=failures.length===0;evidence.tests=tests;evidence.failures=failures;evidence.hashes=hashes();
fs.writeFileSync(new URL('../research/pinhole-progressive-review.json',import.meta.url),JSON.stringify(evidence,null,2)+'\n');
console.log('REVIEW_EVIDENCE',JSON.stringify(evidence));console.log(`${tests.length} passed, ${failures.length} failed`);if(failures.length)process.exitCode=1;
