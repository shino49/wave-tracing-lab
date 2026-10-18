/** Engine lifecycle tests. The WebGPU API below is a host-side control-flow mock:
 * no WGSL is executed, no graphics driver is used, and this is not GPU parity QA.
 * The data supplied to readback is an independently generated CPU fixture. */
import assert from 'node:assert/strict';
import {prepareUTDScene,builtInSceneOBJ} from '../dist/utd-scene.js';
import {makeCPUReference,createUTDEngine,parityGate} from '../dist/utd-engine.js';
import {makeGPUEngine} from '../dist/utd-gpu-engine.js';
import {mockGPU,cpuFixture} from './helpers/utd-mock-gpu.mjs';

const tests=[],failures=[];
const test=async(name,fn)=>{try{await fn();tests.push(name);}catch(error){failures.push({name,error:error.stack});}};
const deferred=()=>{let resolve,reject;const promise=new Promise((a,b)=>{resolve=a;reject=b;});return {promise,resolve,reject};};
const pause=()=>new Promise(r=>setTimeout(r,0));
const abortError=error=>error?.name==='AbortError';
const scene=prepareUTDScene(builtInSceneOBJ('screen'));

async function withMock(options,fn){const mock=mockGPU(options);let engine;try{engine=await makeGPUEngine(scene,options?.fixture?.width??9,options?.fixture?.height??9,{onLost:options?.onLost});await fn(engine,mock);}finally{engine?.destroy();}if(engine)mock.assertDestroyed();return mock;}

await test('CPU validates dimensions before allocating',async()=>{
  for(const [w,h]of [[0,1],[1,0],[98,1],[1,98],[1.5,3],[NaN,3],[Infinity,2]])assert.throws(()=>makeCPUReference(scene,w,h),/安全范围/);
});
await test('CPU returns honest label, monotonic progress, finite data and validity mask',async()=>{
  const engine=makeCPUReference(scene,9,9),progress=[];try{const result=await engine.run({onProgress:p=>progress.push(p)});
    assert.equal(result.backend,'CPU 参考');assert.equal(result.parity,null);assert.equal(progress.length,9);assert.equal(progress.at(-1),1);assert(progress.every((v,i)=>i===0||v>progress[i-1]));
    assert(result.valid>0&&result.invalid>0);assert.equal(result.valid+result.invalid,81);assert(result.data.every(Number.isFinite));
    let valid=0;for(let i=0;i<81;i++){assert([0,1].includes(result.data[i*4+2]));if(result.data[i*4+2])valid++;else{assert.equal(result.data[i*4],0);assert.equal(result.data[i*4+1],0);}}assert.equal(valid,result.valid);
  }finally{engine.destroy();}
});
await test('CPU pre-abort and row-yield cancellation reject rather than return partial data',async()=>{
  const a=new AbortController();a.abort();const pre=makeCPUReference(scene,3,3);await assert.rejects(pre.run({signal:a.signal}),abortError);pre.destroy();
  const controller=new AbortController(),engine=makeCPUReference(scene,3,3);await assert.rejects(engine.run({signal:controller.signal,onProgress:()=>controller.abort()}),abortError);engine.destroy();
});
await test('CPU disposal during final-row yield never returns a completed image',async()=>{
  const engine=makeCPUReference(scene,3,1);await assert.rejects(engine.run({onProgress:()=>engine.destroy()}),abortError);
});
await test('CPU destroyed engine cannot be restarted',async()=>{const engine=makeCPUReference(scene,3,3);engine.destroy();engine.destroy();await assert.rejects(engine.run(),/关闭/);});
await test('Missing WebGPU falls back with truthful reason; explicit CPU never touches GPU',async()=>{
  Object.defineProperty(globalThis,'navigator',{value:{},configurable:true});const auto=await createUTDEngine(scene,3,3);assert.equal(auto.engine.type,'CPU 参考');assert.match(auto.fallbackReason,/WebGPU/);auto.engine.destroy();
  Object.defineProperty(globalThis,'navigator',{value:{get gpu(){throw new Error('CPU must not inspect GPU');}},configurable:true});const cpu=await createUTDEngine(scene,3,3,{backend:'cpu'});assert.equal(cpu.fallbackReason,null);cpu.engine.destroy();
});
await test('Parity gate rejects wrong values, mismatched masks, and all-invalid fixture',async()=>{
  assert(parityGate(scene,cpuFixture()).passed);
  const f=cpuFixture();const first=Array.from({length:81},(_,i)=>i).find(i=>f.data[i*4+2]&&[0,2,4,6,8].includes(i%9)&&[0,2,4,6,8].includes(Math.floor(i/9)));
  const values=cpuFixture();values.data[first*4]+=10;assert.throws(()=>parityGate(scene,values),/未通过/);
  const masks=cpuFixture();masks.data[first*4+2]=0;assert.throws(()=>parityGate(scene,masks),/有效区域不一致/);
  const zero=cpuFixture();zero.data.fill(0);assert.throws(()=>parityGate(scene,zero),/有效区域不一致|没有有效/);
});
await test('Parity failure retains all 25 samples and unchanged tolerance',async()=>{
  const f=cpuFixture();for(let i=0;i<81;i++)if(f.data[i*4+2])f.data[i*4]+=10;
  let caught;try{parityGate(scene,f);}catch(error){caught=error;}
  assert(caught);assert.equal(caught.parity.passed,false);assert.equal(caught.parity.samples.length,25);
  assert.deepEqual(caught.parity.tolerance,{absolute:2e-5,relative:.005,boundaryEpsilon:2e-6});
  assert(caught.parity.numericDisagreements>1,'must continue after the first mismatch');
  for(const sample of caught.parity.samples){assert.equal(sample.gpu.f,f.data[(sample.y*9+sample.x)*4]);assert.equal(sample.position.length,3);}
});
await test('Mock GPU successful host flow uses bundled shaders, chunked submits and unmaps',async()=>{
  await withMock({},async(engine,mock)=>{const progress=[];const result=await engine.run({onProgress:p=>progress.push(p)});assert.equal(result.backend,'WebGPU');assert(result.parity.passed);assert.deepEqual(progress,[8/9,1]);assert.equal(mock.submits,3);assert.equal(mock.scopes,0);assert.equal(mock.mappedReads,1);assert.equal(mock.unmaps,1);assert.deepEqual(mock.fetches,['./utd-functions.wgsl','./utd-compute.wgsl']);engine.destroy();});
});
await test('Mock GPU compile errors, pipeline rejection and shader-fetch failure destroy device',async()=>{
  for(const options of [{compileMessages:[{type:'error',lineNum:2,linePos:3,message:'bad shader'}]},{pipelineError:'compile failed'},{fetchError:true},{maxBufferSize:159}]){
    const mock=mockGPU(options);await assert.rejects(makeGPUEngine(scene,9,9));mock.assertDestroyed();
  }
});
await test('Mock GPU adapter failure gets labelled CPU fallback',async()=>{mockGPU({noAdapter:true});const result=await createUTDEngine(scene,3,3);assert.equal(result.engine.type,'CPU 参考');assert.match(result.fallbackReason,/适配器/);result.engine.destroy();});
await test('Mock GPU pre-abort submits nothing and disposal is idempotent',async()=>{
  await withMock({},async(engine,mock)=>{const c=new AbortController();c.abort();await assert.rejects(engine.run({signal:c.signal}),abortError);assert.equal(mock.submits,0);engine.destroy();engine.destroy();await assert.rejects(engine.run(),abortError);});
});
await test('Mock GPU abort during queued batch suppresses readback',async()=>{
  const done=deferred(),entered=deferred();await withMock({workWait:done.promise,onWork:()=>entered.resolve()},async(engine,mock)=>{const c=new AbortController(),run=engine.run({signal:c.signal});await entered.promise;c.abort();done.resolve();await assert.rejects(run,abortError);assert.equal(mock.mappedReads,0);assert.equal(mock.scopes,0);});
});
await test('Mock GPU cancellation during mapping unmaps but never reads stale bytes',async()=>{
  const done=deferred(),entered=deferred();await withMock({mapWait:done.promise,onMap:()=>entered.resolve()},async(engine,mock)=>{const c=new AbortController(),run=engine.run({signal:c.signal});await entered.promise;c.abort();done.resolve();await assert.rejects(run,abortError);assert.equal(mock.unmaps,1);assert.equal(mock.mappedReads,0);});
});
await test('Mock GPU destroyed during mapping fails as cancellation without stale buffer access',async()=>{
  const done=deferred(),entered=deferred();await withMock({mapWait:done.promise,onMap:()=>entered.resolve()},async(engine,mock)=>{const run=engine.run();await entered.promise;engine.destroy();done.resolve();await assert.rejects(run,abortError);assert.equal(mock.mappedReads,0);});
});
await test('Mock GPU device loss and uncaptured validation error stop future output',async()=>{
  for(const kind of ['lose','uncaptured']){const notices=[];await withMock({onLost:s=>notices.push(s)},async(engine,mock)=>{mock[kind]('device failed');await pause();await assert.rejects(engine.run(),/device failed/);assert.deepEqual(notices,['device failed']);assert.equal(mock.submits,0);});}
});
await test('Mock GPU device loss while work pending rejects and never maps',async()=>{
  const done=deferred(),entered=deferred();await withMock({workWait:done.promise,onWork:()=>entered.resolve()},async(engine,mock)=>{const run=engine.run();await entered.promise;mock.lose('lost in batch');await pause();done.resolve();await assert.rejects(run,/lost in batch/);assert.equal(mock.mappedReads,0);});
});
await test('Mock GPU readback exceptions always unmap, mapping rejection does not read',async()=>{
  await withMock({rangeError:'bad range'},async(engine,mock)=>{await assert.rejects(engine.run(),/bad range/);assert.equal(mock.unmaps,1);assert.equal(mock.mappedReads,1);});
  await withMock({mapError:'map failed'},async(engine,mock)=>{await assert.rejects(engine.run(),/map failed/);assert.equal(mock.mappedReads,0);});
});
await test('Mock GPU parity rejection unmaps first and returns no image',async()=>{
  const fixture=cpuFixture();for(let i=0;i<81;i++)if(fixture.data[i*4+2])fixture.data[i*4]+=10;
  await withMock({fixture},async(engine,mock)=>{await assert.rejects(engine.run(),/数值核对未通过/);assert.equal(mock.unmaps,2,'main read and optional diagnostic read are both unmapped');});
});
await test('Optional trace compilation failure preserves original parity error and raw samples',async()=>{
  const fixture=cpuFixture();for(let i=0;i<81;i++)if(fixture.data[i*4+2])fixture.data[i*4]+=10;
  await withMock({fixture,traceCompileMessages:[{type:'error',lineNum:1,linePos:1,message:'trace compile failed'}]},async(engine,mock)=>{
    let error;try{await engine.run();}catch(e){error=e;}
    assert.match(error.message,/数值核对未通过/);assert.equal(error.diagnostic.parity.samples.length,25);
    assert.equal(error.diagnostic.intermediateTrace.available,false);assert.match(error.diagnostic.intermediateTrace.error,/trace compile failed/);
    assert.equal(mock.scopes,0);assert.equal(mock.unmaps,1);
  });
});
await test('Cancellation while optional trace compiles cannot publish stale diagnostics',async()=>{
  const fixture=cpuFixture();for(let i=0;i<81;i++)if(fixture.data[i*4+2])fixture.data[i*4]+=10;
  const entered=deferred(),done=deferred();
  await withMock({fixture,tracePipelineWait:done.promise,onTrace:()=>entered.resolve()},async(engine,mock)=>{
    const controller=new AbortController(),run=engine.run({signal:controller.signal});await entered.promise;controller.abort();done.resolve();
    await assert.rejects(run,abortError);assert.equal(mock.scopes,0);assert.equal(mock.unmaps,1);
  });
});
await test('Mock GPU full image rejects non-finite power beyond sparse parity points',async()=>{
  for(const lane of [0,1]){const fixture=cpuFixture(),i=Array.from({length:81},(_,i)=>i).find(i=>fixture.data[i*4+2]&&i%9===1);assert.notEqual(i,undefined);fixture.data[i*4+lane]=NaN;await withMock({fixture},async(engine)=>{await assert.rejects(engine.run(),/无效数值/);});}
});
await test('Mock GPU full image rejects malformed validity masks outside parity points',async()=>{
  for(const value of [NaN,Infinity,.75]){const fixture=cpuFixture(),i=Array.from({length:81},(_,i)=>i).find(i=>fixture.data[i*4+2]&&i%9===1);fixture.data[i*4+2]=value;await withMock({fixture},async(engine)=>{await assert.rejects(engine.run(),/有效|数值|mask|valid/i);});}
});
await test('Mock GPU full image rejects negative geometric power outside parity points',async()=>{
  const fixture=cpuFixture(),i=Array.from({length:81},(_,i)=>i).find(i=>fixture.data[i*4+2]&&i%9===1);fixture.data[i*4+1]=-10;await withMock({fixture},async(engine)=>{await assert.rejects(engine.run(),/数值|power/i);});
});
await test('Mock GPU batch submission or completion errors balance validation scopes',async()=>{
  for(const options of [{submitError:'submit failed'},{workError:'queue failed'}])await withMock(options,async(engine,mock)=>{await assert.rejects(engine.run(),/failed/);assert.equal(mock.scopes,0,'validation scope must be popped even when the batch throws');});
});
console.log(JSON.stringify({suite:'utd-lifecycle',passed:tests.length,failed:failures.length,browserValidated:false,gpuExecutionValidated:false,coverage:'CPU engine and mocked WebGPU lifecycle/control flow only',tests,failures},null,2));
if(failures.length)process.exitCode=1;
