/** Opt-in mono finite-safety admission. Native Dawn/Tint compilation always.
 * Actual native software dispatch/readback (not browser/hardware validation):
 * VK_ICD_FILENAMES=/usr/lib/chromium/vk_swiftshader_icd.json WEBGPU_EXECUTION_BACKEND=vulkan node tests/plt-demo-gpu.mjs --execute
 */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {create,globals} from 'webgpu';
import {coefficient} from '../dist/utd-math.js';
import {packPLTGPU,pltEventParityGate,pltPixelParityGate,pltEventFiniteGate,pltPixelFiniteGate,
  makePLTGPUEngine,renderPLTGPU,createPLTGPUBatchRunner,PLT_GPU_LIMITS,PLT_GPU_SCOPE,PLT_GPU_FINITE_SCOPE} from '../dist/plt-gpu.js';
Object.assign(globalThis,globals);
const source=['utd-functions.wgsl','plt-compute.wgsl'].map(name=>fs.readFileSync(new URL(`../dist/${name}`,import.meta.url),'utf8')).join('\n');
const finite={validationMode:'finite'},execute=process.argv.includes('--execute'),tests=[],failures=[];
const report={suite:'plt-demo-gpu',browserValidated:false,physicalGPUValidated:false,executionRequested:execute,
  scope:'Mono CPU-constructed event tapes; optional GPU finite-safety admission, with explicit strict parity controls. Finite admission does not establish numerical or physical accuracy.',tests,failures,nativeCases:[]};
const test=async(name,body)=>{try{await body();tests.push(name);console.log(`PASS ${name}`);}catch(error){failures.push({name,error:error.stack,diagnostic:error.diagnostic});console.error(`FAIL ${name}: ${error.stack}`);}};
const deferred=()=>{let resolve,reject;const promise=new Promise((a,b)=>{resolve=a;reject=b;});return{resolve,reject,promise};};
const delay=()=>new Promise(resolve=>setTimeout(resolve,0));
// CPU evaluation belongs solely to the strict control fixture in this test.
function evaluate(op){
  let sr=op.directAmplitude,si=0,hr=sr,hi=0;
  for(const edge of op.edges){const k=edge.k??op.k,L=edge.ro*edge.sinBeta**2,p=-k*edge.phaseExcess,c=Math.cos(p)/Math.sqrt(edge.ro),s=Math.sin(p)/Math.sqrt(edge.ro);
    for(const boundary of ['soft','hard']){const d=coefficient(k,edge.n,edge.phiI,edge.phiO,L,edge.sinBeta,boundary,2e-6),re=d[0]*c-d[1]*s,im=d[0]*s+d[1]*c;
      if(boundary==='soft'){sr+=re;si+=im;}else{hr+=re;hi+=im;}}
  }
  const sp=sr*sr+si*si,hp=hr*hr+hi*hi,F=op.boundary==='soft'?sp:op.boundary==='hard'?hp:(sp+hp)/2;
  return[sr,si,hr,hi,F,sp,hp,op.edges.length];
}
function fixture({width=7,height=5,samplesPerPixel=4,eventCount=23,maxEdges=5}={}){
  let seed=971;const random=()=>{seed=(Math.imul(seed,1664525)+1013904223)>>>0;return seed/2**32;};
  const events=Array.from({length:eventCount},(_,i)=>{const op={k:90,boundary:['soft','hard','unpolarized'][i%3],directAmplitude:i%3/2,
    edges:Array.from({length:i%(maxEdges+1)},()=>({k:20+random()*90,n:1+random(),phiI:.2+random()*2,phiO:.2+random()*2,ro:.3+random()*3,sinBeta:.2+random()*.8,phaseExcess:random()}))};
    return{...op,referenceF:evaluate(op)[4]};});
  const terms=[];for(let pixel=0;pixel<width*height;pixel++)for(let sample=0;sample<samplesPerPixel;sample++)terms.push({pixel,weight:random()*2,eventIds:Array.from({length:eventCount?sample%7:0},()=>Math.floor(random()*eventCount))});
  terms.reverse();const reference=new Float64Array(width*height);for(const term of terms)reference[term.pixel]+=term.weight*term.eventIds.reduce((value,id)=>value*events[id].referenceF,1)/samplesPerPixel;
  return{width,height,samplesPerPixel,events,terms,reference};
}
const withoutReferences=job=>{const {reference,referenceXYZ,...out}=structuredClone(job);out.events=out.events.map(({referenceF,...event})=>event);return out;};
const rawEvents=job=>Float32Array.from(job.events.flatMap(item=>evaluate(item.operator??item))),small=fixture();
function mockGPU({job=small,eventFixture=rawEvents(job),pixelFixture=Float32Array.from(job.reference),compileError,pipelineError,bufferError,scopeError,queueError,mapError,rangeError,
  limit=1e9,workWait,onWork,mapWait,onMap,deviceWait,adapterWait}={}){
  const lost=deferred(),listeners={},buffers=[],uploads=[];let scopes=0,destroyed=false,reads=0,unmaps=0,adapters=0,devices=0,pipelines=0,submits=0;
  const device={limits:{maxBufferSize:limit,maxStorageBufferBindingSize:limit},lost:lost.promise,
    addEventListener:(name,fn)=>{listeners[name]=fn;},removeEventListener:name=>{delete listeners[name];},
    pushErrorScope(){scopes++;},async popErrorScope(){scopes--;return scopeError?{message:scopeError}:null;},
    createShaderModule:()=>({getCompilationInfo:async()=>({messages:compileError?[{type:'error',lineNum:1,linePos:1,message:compileError}]:[]})}),
    async createComputePipelineAsync(){pipelines++;if(pipelineError)throw new Error(pipelineError);return{getBindGroupLayout:()=>({})};},
    createBuffer(options){if(bufferError)throw new Error(bufferError);const data=options.label==='PLT event readback'?eventFixture:pixelFixture;
      const b={...options,destroyed:false,destroy(){this.destroyed=true;},async mapAsync(){onMap?.();if(mapWait)await mapWait;if(mapError)throw new Error(mapError);},getMappedRange(){reads++;if(rangeError)throw new Error(rangeError);return data.buffer.slice(data.byteOffset,data.byteOffset+data.byteLength);},unmap(){unmaps++;}};buffers.push(b);return b;},
    createBindGroup:()=>({}),createCommandEncoder:()=>({beginComputePass:()=>({setPipeline(){},setBindGroup(){},dispatchWorkgroups(){},end(){}}),copyBufferToBuffer(){},finish:()=>({})}),
    queue:{writeBuffer(buffer,offset,data){uploads.push({label:buffer.label,data:new Uint8Array(data.buffer??data,data.byteOffset??0,data.byteLength).slice()});},submit(){submits++;},async onSubmittedWorkDone(){onWork?.();if(workWait)await workWait;if(queueError)throw new Error(queueError);}},destroy(){destroyed=true;}};
  return{gpu:{requestAdapter:async()=>{adapters++;if(adapterWait)await adapterWait;return{info:{description:'CONTROL-FLOW MOCK ONLY'},async requestDevice(){devices++;if(deviceWait)await deviceWait;return device;}};}},
    lose:message=>lost.resolve({message}),uncaptured:message=>listeners.uncapturederror?.({error:{message}}),uploads,
    get counts(){return{adapters,devices,pipelines,reads,unmaps,submits};},get liveBuffers(){return buffers.filter(b=>!b.destroyed).length;},get deviceDestroyed(){return destroyed;},
    check(){assert.equal(scopes,0);assert(destroyed);assert(buffers.every(b=>b.destroyed));}};
}
async function withMock(options={},body){const mock=mockGPU(options);let engine;try{engine=await makePLTGPUEngine(withoutReferences(options.job??small),{gpu:mock.gpu,shaderSource:source,...finite});await body(engine,mock);}finally{engine?.destroy();}mock.check();}
await test('finite accepts missing references and never reads reference getters; strict defaults still reject',()=>{
  const job=withoutReferences(small),packed=packPLTGPU(job,finite);assert.equal(packed.reference,null);assert.equal(packed.eventReference,null);assert.equal(packed.validationMode,'finite');
  assert.throws(()=>packPLTGPU(job),/reference/);assert.throws(()=>packPLTGPU({...job,reference:small.reference}),/referenceF/);
  for(const name of ['reference','referenceXYZ'])Object.defineProperty(job,name,{get(){throw new Error('CPU reference must not be read');}});
  for(const event of job.events)Object.defineProperty(event,'referenceF',{get(){throw new Error('CPU local field must not be read');}});
  assert.equal(packPLTGPU(job,finite).eventCount,small.events.length);
  const wrapped=withoutReferences(small);wrapped.events=wrapped.events.map(operator=>({operator}));assert.equal(packPLTGPU(wrapped,finite).edgeCount,packed.edgeCount);
  assert.equal(packPLTGPU(small).reference.length,small.reference.length);
});
await test('finite and strict packing use identical transport buffers and immutable mutation snapshots',()=>{
  const job=fixture(),strict=packPLTGPU(job),packed=packPLTGPU(job,finite);
  for(const name of ['eventData','edgeData','termData','ranges','eventEdgeCounts'])assert.deepEqual(packed[name],strict[name]);
  const snapshot=Object.fromEntries(['eventData','edgeData','termData','ranges','eventEdgeCounts'].map(name=>[name,packed[name].slice(0)])),uniforms=packed.createUniforms(2,3);
  job.events[1].edges[0].k=0;job.events[1].edges.length=0;job.events[0].directAmplitude=99;job.terms[0].weight=99;job.terms[0].eventIds.push(999);job.terms.reverse();job.events.length=0;job.width=999;job.height=999;job.samplesPerPixel=999;
  for(const name of Object.keys(snapshot))assert.deepEqual(packed[name],snapshot[name]);assert.deepEqual(packed.createUniforms(2,3),uniforms);
});
await test('mode validation and mono-only finite restriction fail before adapter allocation',async()=>{
  for(const validationMode of [null,'','fast','FINITE',true])assert.throws(()=>packPLTGPU(small,{validationMode}),/validationMode/);
  const spectral={...withoutReferences(small),spectralMode:'visible'};assert.throws(()=>packPLTGPU(spectral,finite),/mono jobs only/);
  let adapters=0;const gpu={requestAdapter:async()=>{adapters++;}};await assert.rejects(renderPLTGPU(spectral,{gpu,...finite}),/mono jobs only/);assert.equal(adapters,0);
});
await test('finite retains exact counts, domains, term shape and packed-f32 phase constraints',()=>{
  const mutations=[j=>j.width=0,j=>j.height=Infinity,j=>j.samplesPerPixel=0,j=>j.events={},j=>j.terms={},j=>j.events[0]=null,
    j=>j.events[0].edges={},j=>j.events[0].boundary='metal',j=>j.events[0].boundaryEpsilon=0,j=>j.events[0].directAmplitude=NaN,j=>j.events[0].directAmplitude=-1,
    j=>j.events[1].edges[0]=null,j=>j.events[1].edges[0].k=0,j=>j.events[1].edges[0].n=3,j=>j.events[1].edges[0].phiI=-1,j=>j.events[1].edges[0].phiO=8,
    j=>j.events[1].edges[0].ro=0,j=>j.events[1].edges[0].sinBeta=0,j=>j.events[1].edges[0].phaseExcess=1e8,
    j=>Object.assign(j.events[1].edges[0],{k:1e-6,ro:1e-8,sinBeta:1e-5}),j=>j.events[1].edges=Array(129).fill(j.events[1].edges[0]),
    j=>j.terms[0]=null,j=>j.terms[0].pixel=j.width*j.height,j=>j.terms[0].weight=Infinity,j=>j.terms[0].weight=-1,j=>j.terms[0].eventIds={},
    j=>j.terms[0].eventIds=Array(7).fill(0),j=>j.terms[0].eventIds=[j.events.length]];
  for(const mutate of mutations){const job=withoutReferences(small);mutate(job);assert.throws(()=>packPLTGPU(job,finite),RangeError);}
  // The f64 product is admissible but f32 operand quantization pushes it over.
  const packedPhase=withoutReferences(small);Object.assign(packedPhase.events[1].edges[0],{k:99999.99,phaseExcess:99999.999/99999.99});assert.throws(()=>packPLTGPU(packedPhase,finite),/packed local phase/);
  const oversized=withoutReferences(small),edge=oversized.events[1].edges[0];oversized.events=Array(16385).fill({...oversized.events[0],edges:Array(128).fill(edge)});assert.throws(()=>packPLTGPU(oversized,finite),/total edge count/);
  oversized.events=[];oversized.terms=Array(65537).fill({pixel:0,weight:1,eventIds:[]});assert.throws(()=>packPLTGPU(oversized,finite),/pixel 0 term count/);
  oversized.terms=Array(PLT_GPU_LIMITS.terms+1);assert.throws(()=>packPLTGPU(oversized,finite),/term count/);
  oversized.terms=[];oversized.events=Array(PLT_GPU_LIMITS.events+1);assert.throws(()=>packPLTGPU(oversized,finite),/event count/);
});
await test('finite gates admit wrong-but-finite powers and pixels while strict parity rejects both',()=>{
  const packed=packPLTGPU(small),events=rawEvents(small),pixels=Float32Array.from(small.reference);events[events.length-4]+=12;pixels[pixels.length-1]+=12;
  assert(pltEventFiniteGate(packed,events).passed);assert(pltPixelFiniteGate(packed,pixels).passed);
  assert.throws(()=>pltEventParityGate(packed,events),e=>e.name==='PLTGPUParityError');assert.throws(()=>pltPixelParityGate(packed,pixels),e=>e.name==='PLTGPUParityError');
});
await test('every field of every event is checked, with nonnegative powers and exact edge counts',()=>{
  const packed=packPLTGPU(withoutReferences(small),finite),valid=rawEvents(small),base=valid.length-8;
  for(let lane=0;lane<8;lane++)for(const value of [NaN,Infinity,-Infinity]){const data=valid.slice();data[base+lane]=value;assert.throws(()=>pltEventFiniteGate(packed,data),e=>e.name==='PLTGPUFiniteError'&&e.validation.failures.at(-1).index===packed.eventCount-1);}
  for(const lane of [4,5,6]){const data=valid.slice();data[base+lane]=-1;assert.throws(()=>pltEventFiniteGate(packed,data),e=>e.validation.negative===1);}
  for(const value of [-1,.5,128]){const data=valid.slice();data[base+7]=value;assert.throws(()=>pltEventFiniteGate(packed,data),e=>e.validation.mismatchedEdgeCounts===1);}
  const signedFields=valid.slice();for(let lane=0;lane<4;lane++)signedFields[base+lane]=-99;assert(pltEventFiniteGate(packed,signedFields).passed);
  const allBad=valid.slice();for(let i=0;i<packed.eventCount;i++)allBad[i*8]=NaN;
  assert.throws(()=>pltEventFiniteGate(packed,allBad),e=>e.validation.invalidEventRecords===packed.eventCount&&e.validation.failures.length===packed.eventCount);
  for(const data of [new Float64Array(valid),[...valid],valid.subarray(1),new Float32Array(valid.length+8)])assert.throws(()=>pltEventFiniteGate(packed,data),/exact stride-8 Float32Array/);
});
await test('every pixel is checked; wrong types and short/long buffers cannot be admitted',()=>{
  const packed=packPLTGPU(withoutReferences(small),finite),valid=Float32Array.from(small.reference);
  for(const value of [NaN,Infinity,-Infinity,-1]){const data=valid.slice();data[data.length-1]=value;assert.throws(()=>pltPixelFiniteGate(packed,data),e=>e.name==='PLTGPUFiniteError'&&e.validation.failures.at(-1).index===packed.pixels-1);}
  for(const data of [new Float64Array(valid),[...valid],valid.subarray(1),new Float32Array(valid.length+1)])assert.throws(()=>pltPixelFiniteGate(packed,data),/exact Float32Array/);
  assert.throws(()=>pltPixelFiniteGate(packed,new Float32Array(packed.pixels).fill(NaN)),e=>e.validation.nonfinite===packed.pixels&&e.validation.failures.length===packed.pixels);
  const empty=packPLTGPU({width:2,height:1,samplesPerPixel:1,events:[],terms:[]},finite);assert(pltEventFiniteGate(empty,new Float32Array()).passed);assert(pltPixelFiniteGate(empty,new Float32Array(2)).passed);
});
await test('finite engine snapshots before first await and returns honest safety-only metadata',async()=>{
  const job=withoutReferences(small),packed=packPLTGPU(job,finite),resume=deferred(),mock=mockGPU({adapterWait:resume.promise});
  const promise=makePLTGPUEngine(job,{gpu:mock.gpu,shaderSource:source,...finite});job.events.length=0;job.terms.length=0;job.samplesPerPixel=99;job.width=99;resume.resolve();const engine=await promise;
  try{const progress=[],result=await engine.run({onProgress:p=>progress.push(p)});assert.equal(result.width,small.width);assert.equal(result.image.length,small.reference.length);assert.equal(result.eventPowers.length,small.events.length);
    assert.deepEqual(result.validation,{passed:true,mode:'finite',parityChecked:false,coverage:'every-event-record-and-pixel-finite-safety',events:result.validation.events,pixels:result.validation.pixels});
    assert.equal(result.timings.cpuReferenceProvided,false);assert.equal(result.backend,'WebGPU hybrid');assert.equal(result.label,PLT_GPU_FINITE_SCOPE);assert(!result.label.includes('every-event and every-pixel CPU verification'));assert.equal(progress.at(-1),1);
    assert.equal(mock.counts.reads,2);assert.equal(mock.counts.unmaps,2);assert.deepEqual(mock.uploads.find(x=>x.label==='PLT events').data,new Uint8Array(packed.eventData));
  }finally{engine.destroy();}mock.check();
});
await test('engine finite mismatch is admitted; default strict engine still rejects',async()=>{
  const pixels=Float32Array.from(small.reference);pixels[pixels.length-1]+=8;await withMock({pixelFixture:pixels},async engine=>assert((await engine.run()).validation.passed));
  const events=rawEvents(small);events[events.length-4]+=8;await withMock({eventFixture:events},async engine=>assert((await engine.run()).validation.passed));
  const mock=mockGPU({pixelFixture:pixels});await assert.rejects(renderPLTGPU(small,{gpu:mock.gpu,shaderSource:source}),e=>e.name==='PLTGPUParityError');mock.check();
  const good=mockGPU();const result=await renderPLTGPU(small,{gpu:good.gpu,shaderSource:source});assert.equal(result.label,PLT_GPU_SCOPE);assert.equal(result.timings.cpuReferenceProvided,true);assert.equal(result.validation.coverage,'every-event-and-every-pixel');assert.equal(result.validation.mode,undefined);good.check();
});
await test('corrupt event output stops before camera dispatch and skips parity-only diagnostics',async()=>{
  for(const mutate of [a=>a[a.length-8]=NaN,a=>a[a.length-4]=-1,a=>a[a.length-1]+=1]){const eventFixture=rawEvents(small);mutate(eventFixture);await withMock({eventFixture},async(engine,mock)=>{
    await assert.rejects(engine.run(),e=>e.name==='PLTGPUFiniteError'&&e.diagnostic.failureTrace===undefined);assert.equal(mock.counts.reads,1);assert.equal(mock.counts.submits,2);await assert.rejects(engine.run(),e=>e.name==='AbortError');});}
  await withMock({eventFixture:rawEvents(small).slice(8)},async engine=>assert.rejects(engine.run(),/exact stride-8/));
  const pixelFixture=Float32Array.from(small.reference);pixelFixture[pixelFixture.length-1]=NaN;await withMock({pixelFixture},async engine=>assert.rejects(engine.run(),e=>e.name==='PLTGPUFiniteError'&&e.diagnostic.validation.events.passed));
  await withMock({pixelFixture:pixelFixture.slice(1)},async engine=>assert.rejects(engine.run(),/exact Float32Array/));
});
await test('finite setup, compilation, allocation and dispatch/readback failures free resources',async()=>{
  await assert.rejects(renderPLTGPU(withoutReferences(small),{gpu:null,...finite}),/unavailable/);
  await assert.rejects(renderPLTGPU(withoutReferences(small),{gpu:{requestAdapter:async()=>null},...finite}),/adapter/);
  for(const options of [{compileError:'compile failed'},{pipelineError:'pipeline failed'},{bufferError:'buffer failed'},{scopeError:'scope failed'},{limit:16}]){const mock=mockGPU(options);await assert.rejects(makePLTGPUEngine(withoutReferences(small),{gpu:mock.gpu,shaderSource:source,...finite}));mock.check();}
  for(const options of [{queueError:'queue failed'},{mapError:'map failed'},{rangeError:'range failed'}])await withMock(options,async engine=>assert.rejects(engine.run(),/failed/));
});
await test('finite pre-cancel, in-flight work/map cancel, explicit destroy and progress cancel suppress output',async()=>{
  const before=new AbortController();before.abort();await assert.rejects(renderPLTGPU(withoutReferences(small),{gpu:null,signal:before.signal,...finite}),e=>e.name==='AbortError');
  for(const phase of ['work','map']){const entered=deferred(),resume=deferred(),controller=new AbortController(),options=phase==='work'?{workWait:resume.promise,onWork:entered.resolve}:{mapWait:resume.promise,onMap:entered.resolve};
    await withMock(options,async(engine,mock)=>{const pending=engine.run({signal:controller.signal});await entered.promise;controller.abort();await assert.rejects(pending,e=>e.name==='AbortError');resume.resolve();assert.equal(mock.counts.reads,0);});}
  await withMock({},async engine=>{engine.destroy();await assert.rejects(engine.run(),e=>e.name==='AbortError');});
  await withMock({},async engine=>assert.rejects(engine.run({onProgress:()=>engine.destroy()}),e=>e.name==='AbortError'));
});
await test('finite late devices after setup abort/timeout are destroyed',async()=>{
  for(const reason of ['abort','timeout']){const resume=deferred(),controller=new AbortController(),mock=mockGPU({deviceWait:resume.promise});
    const pending=makePLTGPUEngine(withoutReferences(small),{gpu:mock.gpu,shaderSource:source,...finite,signal:controller.signal,timeoutMs:reason==='timeout'?5:30000});await delay();if(reason==='abort')controller.abort();await assert.rejects(pending);resume.resolve();await delay();mock.check();}
});
await test('finite device loss and uncaptured errors interrupt pending work; queue/map timeouts clean up',async()=>{
  for(const reason of ['lose','uncaptured']){const entered=deferred(),resume=deferred();await withMock({workWait:resume.promise,onWork:entered.resolve},async(engine,mock)=>{const pending=engine.run();await entered.promise;mock[reason]('lost finite device');await assert.rejects(pending,/lost finite device/);resume.resolve();assert.equal(mock.counts.reads,0);});}
  for(const phase of ['work','map']){const resume=deferred(),mock=mockGPU(phase==='work'?{workWait:resume.promise}:{mapWait:resume.promise});let engine;
    try{engine=await makePLTGPUEngine(withoutReferences(small),{gpu:mock.gpu,shaderSource:source,...finite,timeoutMs:5});await assert.rejects(engine.run(),/timed out/);}finally{engine?.destroy();resume.resolve();}mock.check();}
});
await test('finite completed engines rerun; concurrent runs remain rejected',async()=>{
  const entered=deferred(),resume=deferred();await withMock({workWait:resume.promise,onWork:entered.resolve},async engine=>{const pending=engine.run();await entered.promise;await assert.rejects(engine.run(),/already in progress/);resume.resolve();assert((await pending).validation.passed);assert((await engine.run()).validation.passed);});
});
await test('finite batch runner reuses device/pipelines and releases buffers after every tile',async()=>{
  const mock=mockGPU(),runner=createPLTGPUBatchRunner({gpu:mock.gpu,shaderSource:source,...finite});
  try{for(let i=0;i<4;i++){const result=await runner.run(withoutReferences(small));assert.equal(result.validation.mode,'finite');assert.equal(mock.liveBuffers,0);assert.equal(mock.deviceDestroyed,false);}}
  finally{runner.destroy();}assert.equal(mock.counts.adapters,1);assert.equal(mock.counts.devices,1);assert.equal(mock.counts.pipelines,2);mock.check();await assert.rejects(runner.run(withoutReferences(small)),e=>e.name==='AbortError');
});
await test('finite batch runner is poisoned by corruption, cancellation, concurrency stop or idle device loss',async()=>{
  {const pixelFixture=Float32Array.from(small.reference);pixelFixture[0]=-1;const mock=mockGPU({pixelFixture}),runner=createPLTGPUBatchRunner({gpu:mock.gpu,shaderSource:source,...finite});await assert.rejects(runner.run(withoutReferences(small)),/finite-safety/);await assert.rejects(runner.run(withoutReferences(small)),e=>e.name==='AbortError');mock.check();}
  {const controller=new AbortController(),mock=mockGPU(),runner=createPLTGPUBatchRunner({gpu:mock.gpu,shaderSource:source,...finite,signal:controller.signal});await runner.run(withoutReferences(small));controller.abort();await assert.rejects(runner.run(withoutReferences(small)),e=>e.name==='AbortError');mock.check();}
  {const pending=deferred(),entered=deferred(),mock=mockGPU({workWait:pending.promise,onWork:entered.resolve}),runner=createPLTGPUBatchRunner({gpu:mock.gpu,shaderSource:source,...finite});const first=runner.run(withoutReferences(small));await entered.promise;await assert.rejects(runner.run(withoutReferences(small)),/sequential/);runner.destroy();await assert.rejects(first,e=>e.name==='AbortError');pending.resolve();mock.check();}
  {const mock=mockGPU(),runner=createPLTGPUBatchRunner({gpu:mock.gpu,shaderSource:source,...finite});await runner.run(withoutReferences(small));mock.lose('lost between finite tiles');await delay();await assert.rejects(runner.run(withoutReferences(small)),/lost between finite tiles/);mock.check();}
  {const before=new AbortController();before.abort();let adapters=0;const runner=createPLTGPUBatchRunner({gpu:{requestAdapter:async()=>{adapters++;}},signal:before.signal,...finite});await assert.rejects(runner.run(withoutReferences(small)),e=>e.name==='AbortError');assert.equal(adapters,0);}
});
await test('native Dawn/Tint compiles the unchanged event and camera WGSL entry points',async()=>{
  let gpu=create(['backend=null']);const adapter=await gpu.requestAdapter(),device=await adapter.requestDevice();
  try{device.pushErrorScope('validation');const module=device.createShaderModule({code:source});assert.deepEqual((await module.getCompilationInfo()).messages.filter(x=>x.type==='error'),[]);assert.equal(await device.popErrorScope(),null);
    for(const entryPoint of ['plt_events','plt_camera'])await device.createComputePipelineAsync({layout:'auto',compute:{module,entryPoint}});report.nativeCompilePassed=true;
  }finally{device.destroy();gpu=null;}
});
if(execute){
  const backend=process.env.WEBGPU_EXECUTION_BACKEND;assert(['vulkan','metal','d3d11','d3d12','opengl','opengles'].includes(backend),'Explicit native executable backend required; null is refused');
  let gpu=create([`backend=${backend}`]);report.requestedBackend=backend;report.execution='Actual native WGSL dispatch and mapped readback; no browser or physical GPU claim';report.driverSelection={VK_ICD_FILENAMES:process.env.VK_ICD_FILENAMES??null};
  try{
    const cases=[['mixed boundaries and six-event factors',fixture({samplesPerPixel:7})],['multiple event and camera batches',fixture({width:48,height:40,eventCount:4103})],
      ['128-edge local events',fixture({eventCount:129,maxEdges:128})],['direct-only no events',fixture({eventCount:0})],
      ['empty terms and zero-valued pixels',{width:3,height:2,samplesPerPixel:1,events:[],terms:[],reference:new Float64Array(6)}]];
    for(const [name,job] of cases)await test(`actual native WGSL finite and strict outputs match: ${name}`,async()=>{
      const strict=await renderPLTGPU(job,{gpu,shaderSource:source}),demo=await renderPLTGPU(withoutReferences(job),{gpu,shaderSource:source,...finite});
      assert.deepEqual(demo.image,strict.image);assert.deepEqual(demo.eventPowers,strict.eventPowers);assert.equal(demo.validation.events.count,job.events.length);assert.equal(demo.validation.pixels.count,job.width*job.height);assert.equal(demo.validation.parityChecked,false);assert.equal(demo.timings.cpuReferenceProvided,false);
      report.adapter=demo.adapter;report.nativeCases.push({name,eventCount:job.events.length,pixels:job.width*job.height,strictParityPassed:strict.validation.passed,finiteValidation:demo.validation,timings:demo.timings});
    });
    await test('actual native finite runner reuses its device across distinct reference-free jobs',async()=>{
      const runner=createPLTGPUBatchRunner({gpu,shaderSource:source,...finite});try{for(const [,job] of cases)assert.equal((await runner.run(withoutReferences(job))).validation.mode,'finite');}finally{runner.destroy();}
    });
    await test('actual native finite accepts a wrong finite camera pixel while strict mode rejects it',async()=>{
      const changed=source.replace('camera[pixel] = sum / f32(plt.counts.z);','camera[pixel] = sum / f32(plt.counts.z); if (pixel + 1u == plt.counts.y) { camera[pixel] += 1.0; }');assert.notEqual(changed,source);
      const demo=await renderPLTGPU(withoutReferences(small),{gpu,shaderSource:changed,...finite});assert(demo.validation.passed);assert(demo.image.at(-1)>small.reference.at(-1)+.9);
      await assert.rejects(renderPLTGPU(small,{gpu,shaderSource:changed}),e=>e.name==='PLTGPUParityError');
    });
    await test('actual native finite accepts wrong unused event power while strict mode rejects it',async()=>{
      const job=fixture();job.terms=job.terms.map(term=>({...term,eventIds:[]}));job.reference.fill(0);for(const term of job.terms)job.reference[term.pixel]+=term.weight/job.samplesPerPixel;
      const changed=source.replace('eventResults[index].powers = vec4f(selected,sp,hp,f32(event.edgeRange.y));','eventResults[index].powers = vec4f(selected,sp,hp,f32(event.edgeRange.y)); if (index + 1u == plt.counts.x) { eventResults[index].powers.x += 1.0; }');assert.notEqual(changed,source);
      const demo=await renderPLTGPU(withoutReferences(job),{gpu,shaderSource:changed,...finite});assert(demo.validation.passed);assert(demo.eventPowers.at(-1)>job.events.at(-1).referenceF+.9);
      await assert.rejects(renderPLTGPU(job,{gpu,shaderSource:changed}),e=>e.name==='PLTGPUParityError');
    });
    await test('actual native finite still rejects corrupt unused event edge counts and negative pixels',async()=>{
      const corrupt=source.replace('eventResults[index].powers = vec4f(selected,sp,hp,f32(event.edgeRange.y));','eventResults[index].powers = vec4f(selected,sp,hp,f32(event.edgeRange.y) + 1.0);');assert.notEqual(corrupt,source);
      await assert.rejects(renderPLTGPU(withoutReferences(small),{gpu,shaderSource:corrupt,...finite}),e=>e.name==='PLTGPUFiniteError'&&e.validation.mismatchedEdgeCounts===small.events.length);
      const negative=source.replace('camera[pixel] = sum / f32(plt.counts.z);','camera[pixel] = sum / f32(plt.counts.z); if (pixel + 1u == plt.counts.y) { camera[pixel] = -1.0; }');assert.notEqual(negative,source);
      await assert.rejects(renderPLTGPU(withoutReferences(small),{gpu,shaderSource:negative,...finite}),e=>e.name==='PLTGPUFiniteError'&&e.validation.negative===1);
    });
    await test('actual native finite cancellation destroys the engine and withholds stale image',async()=>{
      const controller=new AbortController(),engine=await makePLTGPUEngine(withoutReferences(small),{gpu,shaderSource:source,...finite});
      try{await assert.rejects(engine.run({signal:controller.signal,onProgress:()=>controller.abort()}),e=>e.name==='AbortError');await assert.rejects(engine.run(),e=>e.name==='AbortError');}finally{engine.destroy();}
    });
  }finally{gpu=null;}
}
report.passed=failures.length===0;report.completedAt=new Date().toISOString();
if(process.env.PLT_DEMO_GPU_REPORT)fs.writeFileSync(process.env.PLT_DEMO_GPU_REPORT,JSON.stringify(report,(_,value)=>ArrayBuffer.isView(value)?Array.from(value):value,2)+'\n');
console.log(`${tests.length} passed, ${failures.length} failed`);if(failures.length)process.exitCode=1;
