/** Real Float64 snapshot/lifecycle checks, plus explicitly labelled fault injection. */
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import {createPinholeCPURunner} from '../dist/pinhole-cpu.js';

let passed=0;const failures=[];
async function test(name,fn){try{await fn();passed++;console.log('PASS',name);}catch(error){failures.push({name,error:String(error.stack)});console.error('FAIL',name,error);}}
function contribution(n,seed){
 const m=2*n,source=Float64Array.from({length:n*n},(_,i)=>((i*17+seed*7)%37)/37),kernels=[];
 for(let c=0;c<6;c++){const values=new Float64Array(m*m);values[0]=.5+.03*c;values[(seed%m)*m+(c+1)%m]=.17;values[(m-1)*m+m-2]=.09;kernels.push({values,channel:c,weight:.3+.1*seed});}
 return {source,kernels};
}
function direct(n,{source,kernels}){
 const m=2*n,channels=Array.from({length:6},()=>new Float64Array(n*n));
 for(const {values,channel,weight} of kernels)for(let y=0;y<n;y++)for(let x=0;x<n;x++)for(let sy=0;sy<n;sy++)for(let sx=0;sx<n;sx++)channels[channel][y*n+x]+=source[sy*n+sx]*values[((y-sy+m)%m)*m+(x-sx+m)%m]*weight;
 return channels;
}
function close(a,b){assert.equal(a.length,b.length);for(let c=0;c<a.length;c++){let error=0,peak=0;for(let i=0;i<a[c].length;i++){assert(Number.isFinite(a[c][i]));error=Math.max(error,Math.abs(a[c][i]-b[c][i]));peak=Math.max(peak,Math.abs(b[c][i]));}assert(error<=Math.max(1e-20,peak*1e-12),`channel ${c}: error ${error}, peak ${peak}`);}}

await test('real repeated snapshots preserve exact final buffers and useful-transform counters',async()=>{
 for(const n of [8,16,32]){
  const progressive=createPinholeCPURunner({width:n}),baseline=createPinholeCPURunner({width:n});
  try{
   const inputs=[1,2,3].map(seed=>contribution(n,seed));
   await progressive.convolveAccumulate(inputs[0]);await baseline.convolveAccumulate(inputs[0]);
   const a=await progressive.snapshotResults(),b=await progressive.snapshotResults();
   assert.deepEqual(a.channels,b.channels);if(n<=16)close(a.channels,direct(n,inputs[0]));
   assert.equal(a.stats.transforms,7);assert.equal(a.stats.snapshotTransforms,6);assert.equal(b.stats.transforms,7);assert.equal(b.stats.snapshotTransforms,12);assert.equal(b.stats.snapshots,2);assert(b.stats.snapshotMs>=a.stats.snapshotMs);assert.equal(b.stats.fftMs,a.stats.fftMs);
   assert.notEqual(a.channels[0].buffer,b.channels[0].buffer);for(let c=1;c<6;c++)assert.notEqual(a.channels[0].buffer,a.channels[c].buffer);
   a.channels[0].fill(NaN);a.channels[1].fill(-Infinity);b.channels[2].fill(1e99);
   for(const input of inputs.slice(1)){await progressive.convolveAccumulate(input);await baseline.convolveAccumulate(input);}
   const actual=await progressive.readResults(),expected=await baseline.readResults();
   assert.deepEqual(actual.channels,expected.channels);assert.equal(actual.stats.transforms,27);assert.equal(actual.stats.transforms,expected.stats.transforms);assert.equal(actual.stats.snapshotTransforms,12);assert.equal(expected.stats.snapshotTransforms,0);assert.equal(expected.stats.snapshots,0);assert.equal(expected.stats.snapshotMs,0);assert.equal(actual.stats.convolutions,18);
   if(n<=16){const summed=direct(n,inputs[0]);for(const input of inputs.slice(1)){const addition=direct(n,input);for(let c=0;c<6;c++)for(let i=0;i<n*n;i++)summed[c][i]+=addition[c][i];}close(actual.channels,summed);}
   await assert.rejects(progressive.snapshotResults(),/closed/);await assert.rejects(progressive.convolveAccumulate(inputs[0]),/closed/);await assert.rejects(progressive.readResults(),/closed/);
  }finally{progressive.destroy();baseline.destroy();}
 }
});

await test('real native 512 snapshot allocates one 16 MiB scratch, not six spectrum clones',async()=>{
 const Native=globalThis.Float64Array,complexLength=2*1024*1024,allocations=[];
 class CountedFloat64Array extends Native{constructor(...args){super(...args);allocations.push(this.length);}}
 let runner;
 try{
  globalThis.Float64Array=CountedFloat64Array;runner=createPinholeCPURunner({width:512});assert.equal(allocations.filter(n=>n===complexLength).length,6);
  const result=await runner.snapshotResults();assert.equal(allocations.filter(n=>n===complexLength).length,7);assert.equal(allocations.filter(n=>n===512*512).length,6);
  assert.equal(result.stats.snapshotScratchComplexBytes,16*1024**2);assert.equal(result.stats.estimatedSnapshotPeakComplexBytes,112*1024**2);assert.equal(result.stats.estimatedPeakComplexBytes,128*1024**2);assert.equal(result.stats.transforms,0);assert.equal(result.stats.snapshotTransforms,6);assert.equal(result.stats.snapshots,1);assert(result.stats.snapshotMs>0);assert(result.channels.every(a=>a.every(v=>v===0)));
  console.log('SNAPSHOT_512_EVIDENCE',JSON.stringify({snapshotMs:result.stats.snapshotMs,scratchMiB:result.stats.snapshotScratchComplexBytes/1024**2,complexAllocations:allocations.filter(n=>n===complexLength).length}));
 }finally{runner?.destroy();globalThis.Float64Array=Native;}
});

await test('real simultaneous snapshot/read/convolution calls reject without unlocking active work',async()=>{
 for(const active of ['snapshotResults','readResults','convolveAccumulate']){
  const runner=createPinholeCPURunner({width:16}),input=contribution(16,1);
  try{
   await runner.convolveAccumulate(input);const pending=active==='convolveAccumulate'?runner[active](input):runner[active]();
   // Issue competitors before yielding, so the overlap is deterministic.
   const snapshots=runner.snapshotResults(),final=runner.readResults(),convolution=runner.convolveAccumulate(input);
   await Promise.all([pending,assert.rejects(snapshots,/sequential/),assert.rejects(final,/sequential/),assert.rejects(convolution,/sequential/)]);
   if(active!=='readResults'){const snapshot=await runner.snapshotResults();assert.equal(snapshot.channels.length,6);await runner.readResults();}else await assert.rejects(runner.snapshotResults(),/closed/);
  }finally{runner.destroy();}
 }
 const runner=createPinholeCPURunner({width:8});try{const a=runner.snapshotResults(),b=runner.readResults(),c=runner.convolveAccumulate(contribution(8,1));await Promise.all([a,assert.rejects(b,/sequential/),assert.rejects(c,/sequential/)]);await runner.readResults();}finally{runner.destroy();}
});

await test('real cancellation before a snapshot and invalid signals permanently poison the runner',async()=>{
 for(const badSignal of [{aborted:true,reason:new Error('pre-aborted snapshot')},{}]){
  const controller=new AbortController();controller.abort(badSignal.reason);const local=badSignal.reason?controller.signal:badSignal,runner=createPinholeCPURunner({width:8});
  await assert.rejects(runner.snapshotResults({signal:local}),badSignal.reason?/pre-aborted snapshot/:/Invalid CPU cancellation signal/);await assert.rejects(runner.readResults(),/closed/);await assert.rejects(runner.convolveAccumulate(contribution(8,1)),/closed/);
 }
 const controller=new AbortController(),runner=createPinholeCPURunner({width:8,signal:controller.signal});controller.abort(new Error('runner aborted'));await assert.rejects(runner.snapshotResults(),/runner aborted/);await assert.rejects(runner.snapshotResults(),/closed/);
});

await test('real in-flight snapshots observe local AND construction signals and explicit destroy',async()=>{
 for(const mode of ['local','construction','destroy']){
  const construction=new AbortController(),local=new AbortController(),runner=createPinholeCPURunner({width:16,signal:construction.signal});await runner.convolveAccumulate(contribution(16,1));
  const descriptor=Object.getOwnPropertyDescriptor(globalThis,'performance');let tick=0,timer;
  try{
   // Force the existing cooperative FFT yield deterministically, without mocking FFT arithmetic.
   Object.defineProperty(globalThis,'performance',{configurable:true,value:{now:()=>tick+=13}});
   timer=setTimeout(()=>{if(mode==='destroy')runner.destroy();else(mode==='local'?local:construction).abort(new Error(`${mode} interrupted snapshot`));},0);
   await assert.rejects(runner.snapshotResults({signal:local.signal}),mode==='destroy'?/closed/:new RegExp(`${mode} interrupted snapshot`));
  }finally{clearTimeout(timer);Object.defineProperty(globalThis,'performance',descriptor);}
  await assert.rejects(runner.readResults(),/closed/);await assert.rejects(runner.snapshotResults(),/closed/);
 }
});

await test('real invalid accumulation and overflow cannot leak a later snapshot',async()=>{
 const runner=createPinholeCPURunner({width:8});await assert.rejects(runner.convolveAccumulate({source:new Float64Array(64).fill(-1),kernels:[]}),/Invalid CPU convolution source/);await assert.rejects(runner.snapshotResults(),/closed/);
 const overflow=createPinholeCPURunner({width:8}),kernel=new Float64Array(256).fill(Number.MAX_VALUE);try{await overflow.convolveAccumulate({source:new Float64Array(64).fill(Number.MAX_VALUE),kernels:[{values:kernel,channel:0}]});await assert.rejects(overflow.snapshotResults(),/Nonfinite/);await assert.rejects(overflow.readResults(),/closed/);}finally{overflow.destroy();}
});

const cpuSource=await fs.readFile(new URL('../dist/pinhole-cpu.js',import.meta.url),'utf8');
const injectionPoint='await transform(data,true,localSignal,snapshot);ensure(localSignal);';
assert(cpuSource.includes(injectionPoint));
async function mutatedRunner(code){const mutated=cpuSource.replace(injectionPoint,`${injectionPoint}\n${code}`);assert.notEqual(mutated,cpuSource);const {createPinholeCPURunner:factory}=await import(`data:text/javascript;base64,${Buffer.from(mutated).toString('base64')}`);return factory({width:8});}
await test('FAULT INJECTION nonfinite real/imaginary, significant negative/imaginary outputs reject atomically',async()=>{
 for(const [code,pattern] of [
  ['if(c===0)data[0]=NaN;',/Nonfinite/],['if(c===0)data[1]=NaN;',/Nonfinite/],['if(c===0)data[0]=Infinity;',/Nonfinite/],['if(c===0)data[1]=Infinity;',/Nonfinite/],
  ['if(c===0)data[0]=-1;',/negative/],['if(c===0)data[1]=1;',/imaginary/],
  ['if(c===0)data[0]=-2.0001e-10;if(c===1)data[0]=1;',/negative/],['if(c===0)data[1]=2.0001e-10;if(c===1)data[0]=1;',/imaginary/],
  ['if(c===0)data[0]=-1e-3;if(c===3)data[0]=1e20;',/negative/],
 ]){
  const runner=await mutatedRunner(code);try{await assert.rejects(runner.snapshotResults(),pattern);await assert.rejects(runner.readResults(),/closed/);}finally{runner.destroy();}
 }
});
await test('FAULT INJECTION bounded signed roundoff is retained using unchanged runtime XYZ-group tolerance',async()=>{
 for(const code of ['if(c===0){data[0]=-2e-10;data[1]=2e-10;}if(c===1)data[0]=1;','if(c===0){data[0]=-1e-20;data[1]=1e-20;}']){
  const runner=await mutatedRunner(code);try{const snapshot=await runner.snapshotResults();assert(snapshot.channels[0][0]<0);assert.equal(snapshot.validation.finite,true);assert.equal(snapshot.validation.roundoff[0].tolerance,snapshot.validation.roundoff[0].peak?2e-10:1e-20);assert.deepEqual((await runner.readResults()).channels,snapshot.channels);}finally{runner.destroy();}
 }
});

console.log(JSON.stringify({suite:'pinhole-cpu-snapshot',passed,failed:failures.length,failures}));
if(failures.length)process.exitCode=1;
