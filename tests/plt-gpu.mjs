/** Native Dawn/Tint compilation always; --execute adds actual dispatch/readback.
 * VK_ICD_FILENAMES=/usr/lib/chromium/vk_swiftshader_icd.json WEBGPU_EXECUTION_BACKEND=vulkan node tests/plt-gpu.mjs --execute
 * Native software execution is not browser or physical-GPU validation.
 */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {createHash} from 'node:crypto';
import {create,globals} from 'webgpu';
import {coefficient} from '../dist/utd-math.js';
import {packPLTGPU,pltEventParityGate,pltPixelParityGate,makePLTGPUEngine,renderPLTGPU,createPLTGPUBatchRunner,PLT_GPU_LIMITS,PLT_GPU_TOLERANCE} from '../dist/plt-gpu.js';
Object.assign(globalThis,globals);
const names=['utd-functions.wgsl','plt-compute.wgsl'],read=name=>fs.readFileSync(new URL(`../dist/${name}`,import.meta.url),'utf8');
const source=names.map(read).join('\n'),hash=value=>createHash('sha256').update(value).digest('hex');
const tests=[],failures=[],execute=process.argv.includes('--execute');
const report={suite:'plt-gpu',schemaVersion:1,browserValidated:false,physicalGPUValidated:false,executionRequested:execute,
  scope:'Backend agreement for CPU-constructed local UTD events and stochastic camera terms; not independent physical validation or GPU geometry traversal',
  sourceHashes:Object.fromEntries([...names,'plt-gpu.js','utd-math.js'].map(name=>[name,hash(read(name))])),
  testSHA256:hash(fs.readFileSync(new URL('./plt-gpu.mjs',import.meta.url))),tolerance:PLT_GPU_TOLERANCE,limits:PLT_GPU_LIMITS,tests,failures,nativeCases:[],negativeControls:[]};
const test=async(name,fn)=>{try{await fn();tests.push(name);console.log(`PASS ${name}`);}catch(error){failures.push({name,error:error.stack,validation:error.validation,diagnostic:error.diagnostic});console.error(`FAIL ${name}: ${error.stack}`);}};
const delay=()=>new Promise(resolve=>setTimeout(resolve,0));
const deferred=()=>{let resolve,reject;const promise=new Promise((a,b)=>{resolve=a;reject=b;});return{resolve,reject,promise};};
function evaluate(op){
  let sr=op.directAmplitude,si=0,hr=sr,hi=0;
  for(const edge of op.edges){const k=edge.k??op.k,L=edge.ro*edge.sinBeta**2,p=-k*edge.phaseExcess,c=Math.cos(p)/Math.sqrt(edge.ro),s=Math.sin(p)/Math.sqrt(edge.ro);
    for(const boundary of ['soft','hard']){const d=coefficient(k,edge.n,edge.phiI,edge.phiO,L,edge.sinBeta,boundary,2e-6),re=d[0]*c-d[1]*s,im=d[0]*s+d[1]*c;if(boundary==='soft'){sr+=re;si+=im;}else{hr+=re;hi+=im;}}
  }
  const sp=sr*sr+si*si,hp=hr*hr+hi*hi,F=op.boundary==='soft'?sp:op.boundary==='hard'?hp:(sp+hp)/2;
  return[sr,si,hr,hi,F,sp,hp,op.edges.length];
}
function jobFor({width=9,height=7,samplesPerPixel=4,eventCount=57,maxEdges=7,depth=6,zero=false}={}){
  let seed=18341;const random=()=>{seed=(Math.imul(seed,1664525)+1013904223)>>>0;return seed/2**32;};
  const events=Array.from({length:eventCount},(_,i)=>{
    const op={k:20+random()*180,boundary:['soft','hard','unpolarized'][i%3],directAmplitude:zero?0:i%4===0?.5:i%4===1?0:1,
      edges:Array.from({length:zero?0:i%(maxEdges+1)},(_,j)=>({k:20+random()*180,n:1+random(),phiI:.1+random()*2.5,phiO:.1+random()*2.5,
        ro:.3+random()*5,sinBeta:.1+random()*.9,phaseExcess:random()*2,edgeId:`edge-${j}`}))};
    return{...op,referenceF:evaluate(op)[4]};
  });
  const terms=[];for(let pixel=0;pixel<width*height;pixel++)for(let j=0;j<samplesPerPixel;j++){
    const ids=Array.from({length:eventCount?j%(depth+1):0},()=>Math.floor(random()*eventCount));
    terms.push({pixel,weight:zero?0:random()*3,eventIds:ids});
  }
  // Deliberately unsorted input; the host must group without changing semantics.
  terms.reverse();const reference=new Float64Array(width*height);
  for(const term of terms)reference[term.pixel]+=term.weight*term.eventIds.reduce((value,id)=>value*events[id].referenceF,1)/samplesPerPixel;
  return{width,height,samplesPerPixel,events,terms,reference};
}
const small=jobFor(),rawEvents=job=>Float32Array.from(job.events.flatMap(event=>evaluate(event.operator??event)));
function mockGPU({job=small,eventFixture=rawEvents(job),pixelFixture=Float32Array.from(job.reference),compileError,pipelineError,bufferError,submitError,queueError,mapError,rangeError,
  validationError,limit=1e9,workWait,onWork,mapWait,onMap,deviceWait}={}){
  const lost=deferred(),listeners={},buffers=[];let scopes=0,destroyed=false,submitted=0,reads=0,unmaps=0;
  const device={limits:{maxBufferSize:limit,maxStorageBufferBindingSize:limit},lost:lost.promise,
    addEventListener:(name,fn)=>{listeners[name]=fn;},removeEventListener:name=>{delete listeners[name];},
    pushErrorScope(){scopes++;},async popErrorScope(){scopes--;return validationError?{message:validationError}:null;},
    createShaderModule:()=>({getCompilationInfo:async()=>({messages:compileError?[{type:'error',lineNum:1,linePos:1,message:compileError}]:[]})}),
    async createComputePipelineAsync(){if(pipelineError)throw new Error(pipelineError);return{getBindGroupLayout:()=>({})};},
    createBuffer(options){if(bufferError)throw new Error(bufferError);const fixture=options.label==='PLT event readback'?eventFixture:pixelFixture;
      const b={...options,destroyed:false,destroy(){this.destroyed=true;},async mapAsync(){onMap?.();if(mapWait)await mapWait;if(mapError)throw new Error(mapError);},getMappedRange(){reads++;if(rangeError)throw new Error(rangeError);return fixture.buffer.slice(0);},unmap(){unmaps++;}};buffers.push(b);return b;},
    createBindGroup:()=>({}),createCommandEncoder:()=>({beginComputePass:()=>({setPipeline(){},setBindGroup(){},dispatchWorkgroups(){},end(){}}),copyBufferToBuffer(){},finish:()=>({})}),
    queue:{writeBuffer(){},submit(){submitted++;if(submitError)throw new Error(submitError);},async onSubmittedWorkDone(){onWork?.();if(workWait)await workWait;if(queueError)throw new Error(queueError);}},
    destroy(){destroyed=true;}};
  return{gpu:{requestAdapter:async()=>({info:{description:'CONTROL-FLOW MOCK ONLY'},async requestDevice(){if(deviceWait)await deviceWait;return device;}})},
    lose:reason=>lost.resolve({message:reason}),uncaptured:reason=>listeners.uncapturederror?.({error:{message:reason}}),
    get liveBuffers(){return buffers.filter(b=>!b.destroyed).length;},get deviceDestroyed(){return destroyed;},get submitted(){return submitted;},get reads(){return reads;},get unmaps(){return unmaps;},
    check(){assert.equal(scopes,0);assert(destroyed);assert(buffers.every(b=>b.destroyed));}};
}
async function withMock(options,body){const mock=mockGPU(options);let engine;try{engine=await makePLTGPUEngine(options?.job??small,{gpu:mock.gpu,shaderSource:source});await body(engine,mock);}finally{engine?.destroy();}mock.check();}
await test('packing snapshots, sorts terms, preserves six local event factors and empty pixels',()=>{
  const job=jobFor(),packed=packPLTGPU(job);assert.equal(packed.pixels,63);assert.equal(packed.termCount,252);
  for(let i=0;i<packed.pixels;i++)assert.equal(packed.ranges[2*i+1],4);
  const previous=packed.createUniforms();job.events.length=0;job.samplesPerPixel=99;job.reference.fill(99);
  assert.deepEqual(packed.createUniforms(),previous);assert(packed.reference.every(x=>x!==99));
  const empty=packPLTGPU({width:2,height:1,samplesPerPixel:1,events:[],terms:[],reference:new Float64Array(2)});assert.equal(empty.eventCount,0);
});
await test('strict bounded malformed input rejection, including total records and packed phase',()=>{
  const mutations=[j=>j.width=0,j=>j.reference=new Float32Array(j.reference),j=>j.reference[0]=NaN,j=>j.reference[0]=-1,
    j=>j.events[0].boundary='metal',j=>j.events[0].boundaryEpsilon=1e-14,j=>j.events[0].directAmplitude=2,j=>delete j.events[0].referenceF,
    j=>j.events[1].edges[0].n=3,j=>j.events[1].edges[0].sinBeta=0,j=>j.events[1].edges[0].phiI=-1,
    j=>j.events[1].edges[0].phaseExcess=1e8,j=>j.events[0].edges=Array(129).fill(j.events[1].edges[0]),
    j=>j.terms[0].eventIds=Array(7).fill(0),j=>j.terms[0].eventIds=[j.events.length],j=>j.terms[0].weight=-1,j=>j.terms[0].pixel=j.width*j.height];
  for(const mutate of mutations){const job=jobFor();mutate(job);assert.throws(()=>packPLTGPU(job),RangeError);}
  const over=jobFor(),op={...over.events[0],edges:Array(128).fill(over.events[1].edges[0])};
  over.events=Array(16385).fill(op);assert.throws(()=>packPLTGPU(over),/total edge count/);
  over.events=[];over.terms=Array(65537).fill({pixel:0,weight:1,eventIds:[]});assert.throws(()=>packPLTGPU(over),/pixel 0 term count/);
  over.terms=Array(PLT_GPU_LIMITS.terms+1);assert.throws(()=>packPLTGPU(over),/term count/);
  over.terms=[];over.events=Array(PLT_GPU_LIMITS.events+1);assert.throws(()=>packPLTGPU(over),/event count/);
});
await test('exhaustive event and pixel gates retain errors for every item and reject last pixel',()=>{
  const packed=packPLTGPU(small),events=rawEvents(small),pixels=Float32Array.from(small.reference);
  assert.equal(pltEventParityGate(packed,events).absoluteErrors.length,small.events.length);
  assert.equal(pltPixelParityGate(packed,pixels).absoluteErrors.length,pixels.length);
  for(const lane of [0,events.length-4]){const corrupt=events.slice();corrupt[lane]=NaN;assert.throws(()=>pltEventParityGate(packed,corrupt),e=>e.name==='PLTGPUParityError');}
  for(const value of [NaN,Infinity,-1,1e10]){const corrupt=pixels.slice();corrupt[corrupt.length-1]=value;assert.throws(()=>pltPixelParityGate(packed,corrupt),e=>e.name==='PLTGPUParityError'&&e.validation.failures.at(-1).index===pixels.length-1);}
});
await test('mock successful result is GPU labelled, keeps exhaustive validation, and frees resources',async()=>{
  await withMock({},async(engine,mock)=>{const progress=[],result=await engine.run({onProgress:p=>progress.push(p)});assert.equal(result.backend,'WebGPU hybrid');assert.equal(result.validation.coverage,'every-event-and-every-pixel');assert.equal(progress.at(-1),1);assert.equal(mock.reads,2);assert.equal(mock.unmaps,2);});
});
await test('no GPU, no adapter, compile, pipeline, buffer and validation failures reject cleanly',async()=>{
  await assert.rejects(renderPLTGPU(small,{gpu:null}),/unavailable/);await assert.rejects(renderPLTGPU(small,{gpu:{requestAdapter:async()=>null}}),/adapter/);
  for(const options of [{compileError:'compile failed'},{pipelineError:'pipeline failed'},{bufferError:'buffer failed'},{validationError:'validation failed'},{limit:16}]){const mock=mockGPU(options);await assert.rejects(makePLTGPUEngine(small,{gpu:mock.gpu,shaderSource:source}));mock.check();}
});
await test('dispatch, map, range and parity failures destroy all buffers and withhold images',async()=>{
  for(const options of [{submitError:'submit failed'},{queueError:'queue failed'},{mapError:'map failed'},{rangeError:'range failed'}])await withMock(options,async engine=>assert.rejects(engine.run(),/failed/));
  const pixels=Float32Array.from(small.reference);pixels[pixels.length-1]=999;
  await withMock({pixelFixture:pixels},async engine=>assert.rejects(engine.run(),e=>e.name==='PLTGPUParityError'&&e.diagnostic.validation.events.passed));
  const events=rawEvents(small);events[4]=999;await withMock({eventFixture:events},async(engine,mock)=>{await assert.rejects(engine.run(),e=>e.name==='PLTGPUParityError'&&e.validation.failures[0].actual===999&&e.diagnostic.failureTrace.status==='unavailable');assert.equal(mock.reads,2);}); // Original event read + best-effort probe read; this mock has no probe fixture.
});
await test('pre-cancel, in-flight cancel, destroy and progress callback cancellation suppress stale output',async()=>{
  const controller=new AbortController();controller.abort();await assert.rejects(renderPLTGPU(small,{gpu:null,signal:controller.signal}),e=>e.name==='AbortError');
  for(const phase of ['work','map']){const entered=deferred(),resume=deferred(),abort=new AbortController();const options=phase==='work'?{workWait:resume.promise,onWork:entered.resolve}:{mapWait:resume.promise,onMap:entered.resolve};
    await withMock(options,async(engine,mock)=>{const promise=engine.run({signal:abort.signal});await entered.promise;abort.abort();await assert.rejects(promise,e=>e.name==='AbortError');resume.resolve();assert.equal(mock.reads,0);});}
  await withMock({},async engine=>{await assert.rejects(engine.run({onProgress:()=>engine.destroy()}),e=>e.name==='AbortError');await assert.rejects(engine.run(),e=>e.name==='AbortError');});
});
await test('late requested device after initialization cancel or timeout is destroyed',async()=>{
  for(const method of ['abort','timeout']){const resume=deferred(),controller=new AbortController(),mock=mockGPU({deviceWait:resume.promise});
    const promise=makePLTGPUEngine(small,{gpu:mock.gpu,shaderSource:source,signal:controller.signal,timeoutMs:method==='timeout'?5:30000});await delay();if(method==='abort')controller.abort();await assert.rejects(promise);resume.resolve();await delay();mock.check();}
});
await test('device loss and uncaptured errors cancel pending queue work without needing its resolution',async()=>{
  for(const method of ['lose','uncaptured']){const entered=deferred(),resume=deferred();await withMock({workWait:resume.promise,onWork:entered.resolve},async(engine,mock)=>{
    const promise=engine.run();await entered.promise;mock[method]('lost test device');await assert.rejects(promise,/lost test device/);assert.equal(mock.reads,0);resume.resolve();});}
});
await test('concurrent runs rejected; repeated completed runs remain valid until destroy',async()=>{
  const entered=deferred(),resume=deferred();await withMock({workWait:resume.promise,onWork:entered.resolve},async engine=>{const promise=engine.run();await entered.promise;await assert.rejects(engine.run(),/already in progress/);resume.resolve();assert((await promise).validation.passed);assert((await engine.run()).validation.passed);});
});
await test('adapter/device rejection, fetch failure and queue/readback timeouts clean up',async()=>{
  await assert.rejects(renderPLTGPU(small,{gpu:{requestAdapter:async()=>{throw new Error('adapter failed');}}}),/adapter failed/);
  await assert.rejects(renderPLTGPU(small,{gpu:{requestAdapter:async()=>({requestDevice:async()=>{throw new Error('device failed');}})}}),/device failed/);
  const fetchBefore=globalThis.fetch,mock=mockGPU();globalThis.fetch=async()=>({ok:false,status:503});
  try{await assert.rejects(makePLTGPUEngine(small,{gpu:mock.gpu}),/Cannot load/);}finally{globalThis.fetch=fetchBefore;}mock.check();
  for(const phase of ['work','map']){const pending=deferred(),options=phase==='work'?{workWait:pending.promise}:{mapWait:pending.promise},mock=mockGPU(options);let engine;
    try{engine=await makePLTGPUEngine(small,{gpu:mock.gpu,shaderSource:source,timeoutMs:5});await assert.rejects(engine.run(),/timed out/);}finally{engine?.destroy();pending.resolve();}mock.check();}
  await withMock({},async engine=>assert.rejects(engine.run({onProgress:()=>{throw new Error('progress failed');}}),/progress failed/));
});
await test('tile runner reuses exactly one device and pipeline pair while releasing every batch buffer',async()=>{
  const mock=mockGPU(),counts={adapters:0,devices:0,modules:0,pipelines:0},raw=mock.gpu.requestAdapter;
  mock.gpu.requestAdapter=async(...args)=>{counts.adapters++;const adapter=await raw(...args),request=adapter.requestDevice;
    adapter.requestDevice=async()=>{counts.devices++;const device=await request(),module=device.createShaderModule,pipeline=device.createComputePipelineAsync;
      device.createShaderModule=(...args)=>{counts.modules++;return module(...args);};device.createComputePipelineAsync=(...args)=>{counts.pipelines++;return pipeline(...args);};return device;};return adapter;};
  const runner=createPLTGPUBatchRunner({gpu:mock.gpu,shaderSource:source});
  try{for(let i=0;i<5;i++){const r=await runner.run(small);assert(r.validation.passed);assert.equal(mock.liveBuffers,0);assert.equal(mock.deviceDestroyed,false);}}
  finally{runner.destroy();}assert.deepEqual(counts,{adapters:1,devices:1,modules:1,pipelines:2});mock.check();await assert.rejects(runner.run(small),e=>e.name==='AbortError');
});
await test('tile runner constructor cancellation guards setup and closes an idle reused device',async()=>{
 const before=new AbortController();before.abort();let adapters=0;const stopped=createPLTGPUBatchRunner({signal:before.signal,gpu:{requestAdapter:async()=>{adapters++;throw new Error('Must not request adapter');}}});await assert.rejects(stopped.run(small),e=>e.name==='AbortError');assert.equal(adapters,0);
 const mock=mockGPU(),parent=new AbortController(),runner=createPLTGPUBatchRunner({gpu:mock.gpu,shaderSource:source,signal:parent.signal});await runner.run(small);parent.abort();await assert.rejects(runner.run(small),e=>e.name==='AbortError');mock.check();
});
await test('tile runner destroys a resolved device when cancellation wins the adoption microtask',async()=>{
 const mock=mockGPU(),raw=mock.gpu.requestAdapter;let runner;
 mock.gpu.requestAdapter=async(...args)=>{const adapter=await raw(...args),request=adapter.requestDevice;adapter.requestDevice=()=>{const pending=request();queueMicrotask(()=>runner.destroy());return pending;};return adapter;};
 runner=createPLTGPUBatchRunner({gpu:mock.gpu,shaderSource:source});await assert.rejects(runner.run(small),e=>e.name==='AbortError');await delay();mock.check();
});
await test('tile runner rejects concurrency, poisons on failed gate, handles between-batch device loss',async()=>{
  {const pending=deferred(),mock=mockGPU({workWait:pending.promise}),runner=createPLTGPUBatchRunner({gpu:mock.gpu,shaderSource:source});
   const first=runner.run(small);await delay();await assert.rejects(runner.run(small),/sequential/);runner.destroy();await assert.rejects(first,e=>e.name==='AbortError');pending.resolve();await delay();mock.check();}
  {const bad=Float32Array.from(small.reference);bad[bad.length-1]+=2;const mock=mockGPU({pixelFixture:bad}),runner=createPLTGPUBatchRunner({gpu:mock.gpu,shaderSource:source});
   await assert.rejects(runner.run(small),/parity failed/);await assert.rejects(runner.run(small),e=>e.name==='AbortError');mock.check();}
  {const mock=mockGPU(),runner=createPLTGPUBatchRunner({gpu:mock.gpu,shaderSource:source});await runner.run(small);mock.lose('lost between tiles');await delay();await assert.rejects(runner.run(small),/lost between tiles/);mock.check();}
});
await test('native Dawn/Tint compiles both passes; invalid WGSL negative control rejected',async()=>{
  let gpu=create(['backend=null']);const adapter=await gpu.requestAdapter(),device=await adapter.requestDevice();
  try{device.pushErrorScope('validation');const module=device.createShaderModule({code:source});const info=await module.getCompilationInfo();assert.deepEqual(info.messages.filter(x=>x.type==='error'),[]);assert.equal(await device.popErrorScope(),null);
    for(const entryPoint of ['plt_events','plt_camera'])await device.createComputePipelineAsync({layout:'auto',compute:{module,entryPoint}});
    device.pushErrorScope('validation');const bad=device.createShaderModule({code:'not a shader'});assert((await bad.getCompilationInfo()).messages.some(x=>x.type==='error'));assert(await device.popErrorScope());report.nativeCompilePassed=true;
  }finally{device.destroy();gpu=null;}
});
if(execute){
  const backend=process.env.WEBGPU_EXECUTION_BACKEND;assert(['vulkan','metal','d3d11','d3d12','opengl','opengles'].includes(backend),'Explicit executable backend required; null is refused');
  let gpu=create([`backend=${backend}`]);report.requestedBackend=backend;report.execution='Actual native Dawn/Tint WGSL compilation, dispatch and mapped readback';report.driverSelection={VK_ICD_FILENAMES:process.env.VK_ICD_FILENAMES??null};
  try{
    const cases=[['mixed scalar boundaries and unordered path terms',jobFor()],['multi-batch event and image grid',jobFor({width:96,height:72,samplesPerPixel:8,eventCount:4103,maxEdges:12})],
      ['six-event path depth and 128-edge coherent local sums',jobFor({width:17,height:13,samplesPerPixel:7,eventCount:129,maxEdges:128})],
      ['all zero transport',jobFor({zero:true})],['direct-only transport without events',jobFor({eventCount:0})],
      ['empty image pixels and empty event/term buffers',{width:5,height:3,samplesPerPixel:1,events:[],terms:[],reference:new Float64Array(15)}]];
    for(const [name,job] of cases)await test(`actual WGSL: ${name}`,async()=>{
      const result=await renderPLTGPU(job,{gpu,shaderSource:source});assert(result.validation.passed);assert(result.image.every(x=>Number.isFinite(x)&&x>=0));assert.equal(result.validation.events.count,job.events.length);assert.equal(result.validation.pixels.count,job.reference.length);
      report.adapter=result.adapter;report.nativeCases.push({name,width:job.width,height:job.height,samplesPerPixel:job.samplesPerPixel,eventCount:job.events.length,termCount:job.terms.length,
        validation:result.validation,timings:result.timings});
    });
    await test('actual WGSL: reused tile runner executes different jobs with exhaustive admission',async()=>{
      const runner=createPLTGPUBatchRunner({gpu,shaderSource:source});try{for(const [name,job] of cases){const result=await runner.run(job);assert(result.validation.passed);assert.equal(result.validation.events.count,job.events.length);assert.equal(result.validation.pixels.count,job.width*job.height);}}
      finally{runner.destroy();}
    });
    await test('actual WGSL: wedge seams, flat wedge, skew and large bounded phase',async()=>{
      const events=[];
      for(const boundary of ['soft','hard','unpolarized'])for(const n of [1,1.5,2])for(const theta of [0,.00001,1,1024,65536,99999]){
        const op={k:256,boundary,directAmplitude:.5,boundaryEpsilon:2e-6,edges:[{k:256,n,phiI:.3,phiO:.3+Math.PI,ro:8,sinBeta:theta===0?.001:.707,phaseExcess:theta/256}]};
        events.push({...op,referenceF:evaluate(op)[4]});
      }
      for(const offset of [-1e-5,-1e-7,0,1e-7,1e-5]){const op={k:256,boundary:'unpolarized',directAmplitude:.5,edges:[{k:256,n:2,phiI:.3,phiO:.3+Math.PI+offset,ro:3,sinBeta:1,phaseExcess:.2}]};events.push({...op,referenceF:evaluate(op)[4]});}
      const job={width:events.length,height:1,samplesPerPixel:1,events,terms:events.map((_,pixel)=>({pixel,weight:1,eventIds:[pixel]})),reference:Float64Array.from(events,e=>e.referenceF)};
      // n=1's exterior sector stops at pi; a flat wedge contributes zero at any admissible angle.
      for(const op of events)if(op.edges[0].n===1){op.edges[0].phiO=2;op.referenceF=evaluate(op)[4];}job.reference=Float64Array.from(events,e=>e.referenceF);
      const result=await renderPLTGPU(job,{gpu,shaderSource:source});assert(result.validation.passed);report.nativeCases.push({name:'wedge seams, flat wedge, skew and large bounded phase',validation:result.validation,timings:result.timings});
    });
    const {createPLTScene,buildPLTJob}=await import('../dist/plt-core.js');
    report.sceneReferenceHashes=Object.fromEntries(['plt-core.js','plt-fsd.js','plt-geometry.js','mesh-geometry.js'].map(name=>[name,hash(read(name))]));
    for(const [preset,boundary,wave] of [['gallery','soft',true],['gallery','hard',true],['baffles','unpolarized',true],['prism','unpolarized',true],['gallery','unpolarized',false]])await test(`actual scene GPU replay: ${preset}, ${boundary}, wave=${wave}`,async()=>{
      const scene=createPLTScene({preset}),job=await buildPLTJob(scene,scene.camera,{width:16,height:12,samplesPerPixel:8,maxDepth:6,boundary,wave},{yieldEvery:10000});
      const result=await renderPLTGPU(job,{gpu,shaderSource:source});assert(result.validation.passed);assert(job.terms.length>0);if(wave)assert(job.stats.twoFSDContributions>0);
      report.nativeCases.push({name:`actual 3D ${preset} ${boundary} wave=${wave}`,width:job.width,height:job.height,samplesPerPixel:job.samplesPerPixel,eventCount:job.events.length,termCount:job.terms.length,
        sceneSummary:job.sceneSummary,pathStats:job.stats,validation:result.validation,timings:result.timings});
    });
    await test('actual WGSL: progressive irregular tiles and sample batches preserve final exhaustive pixel parity',async()=>{
      const {executePLTTask}=await import('../dist/plt-runtime.js');
      const request={backend:'auto',scene:{preset:'prism'},options:{width:13,height:11,samplesPerPixel:3,maxDepth:4,wavelengthMm:.025,seed:7391},progressive:{tileSize:5,batchSamples:2}};
      const runner=createPLTGPUBatchRunner({gpu,shaderSource:source});let previews=0;
      try{const result=await executePLTTask(request,{gpuRunner:(job,opts)=>runner.run(job,opts),preview:()=>previews++});
        const cpu=await executePLTTask({...request,backend:'cpu',progressive:false});
        for(let i=0;i<result.results.length;i++){const r=result.results[i];assert(r.validation.passed);assert.equal(r.validation.acceptedBatches,18);assert(r.sampleCounts.every(n=>n===3));pltPixelParityGate({reference:cpu.results[i].image,pixels:r.image.length},Float32Array.from(r.image));}
        assert(previews>0);report.nativeCases.push({name:'progressive irregular tiles, two sample batches',options:request.options,progressive:result.progressive,validation:result.results.map(r=>r.validation)});
      }finally{runner.destroy();}
    });
    await test('actual native engine cancellation, explicit destruction and bad compilation withhold output',async()=>{
      const controller=new AbortController(),engine=await makePLTGPUEngine(small,{gpu,shaderSource:source});
      try{await assert.rejects(engine.run({signal:controller.signal,onProgress:()=>controller.abort()}),e=>e.name==='AbortError');await assert.rejects(engine.run(),e=>e.name==='AbortError');}finally{engine.destroy();}
      const destroyed=await makePLTGPUEngine(small,{gpu,shaderSource:source});destroyed.destroy();await assert.rejects(destroyed.run(),e=>e.name==='AbortError');
      await assert.rejects(renderPLTGPU(small,{gpu,shaderSource:source+'\nthis is invalid WGSL'}),/WGSL/);
      report.negativeControls.push({name:'native cancellation, destruction, malformed shader',passed:true});
    });
    await test('actual WGSL mutation: last camera pixel wrong is rejected with no image',async()=>{
      const modified=source.replace('camera[pixel] = sum / f32(plt.counts.z);','camera[pixel] = sum / f32(plt.counts.z); if (pixel + 1u == plt.counts.y) { camera[pixel] += 1.0; }');assert.notEqual(modified,source);
      await assert.rejects(renderPLTGPU(small,{gpu,shaderSource:modified}),error=>{assert.equal(error.name,'PLTGPUParityError');assert.equal(error.validation.failures.at(-1).index,small.reference.length-1);report.negativeControls.push({name:'last-pixel shader mutation',validation:error.validation});return true;});
    });
    await test('actual WGSL mutation: event unused by any camera term is still rejected',async()=>{
      const job=jobFor({eventCount:10});job.terms=job.terms.map(t=>({...t,eventIds:[]}));job.reference.fill(0);for(const term of job.terms)job.reference[term.pixel]+=term.weight/job.samplesPerPixel;
      const modified=source.replace('eventResults[index].powers = vec4f(selected,sp,hp,f32(event.edgeRange.y));','eventResults[index].powers = vec4f(selected,sp,hp,f32(event.edgeRange.y)); if (index + 1u == plt.counts.x) { eventResults[index].powers.x += 1.0; }');assert.notEqual(modified,source);
      await assert.rejects(renderPLTGPU(job,{gpu,shaderSource:modified}),error=>{assert.equal(error.name,'PLTGPUParityError');assert.equal(error.validation.failures.at(-1).index,9);report.negativeControls.push({name:'unused-event shader mutation',validation:error.validation});return true;});
    });
  }finally{gpu=null;}
}
report.passed=failures.length===0;report.completedAt=new Date().toISOString();
const path=process.env.PLT_GPU_REPORT;if(path)fs.writeFileSync(path,JSON.stringify(report,(_,value)=>ArrayBuffer.isView(value)?Array.from(value):value,2)+'\n');
console.log(`${tests.length} passed, ${failures.length} failed`);if(failures.length)process.exitCode=1;
