/** Native compile always; --execute adds actual Dawn/Tint + SwiftShader dispatch.
 * VK_ICD_FILENAMES=/usr/lib/chromium/vk_swiftshader_icd.json WEBGPU_EXECUTION_BACKEND=vulkan node tests/plt-failure-trace.mjs --execute
 * Native software execution is not browser, physical-device, or physical-accuracy validation.
 * Negative-control shader mutations exist only in memory; the shared erfcx evaluator has its own numerical regression.
 */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {execFileSync} from 'node:child_process';
import {createHash} from 'node:crypto';
import {create,globals} from 'webgpu';
import {coefficient} from '../dist/utd-math.js';
import {packPLTGPU,makePLTGPUEngine,renderPLTGPU,PLT_GPU_TOLERANCE} from '../dist/plt-gpu.js';
import {PLT_TRACE_SCHEMA,PLT_TRACE_MAX_EVENTS,PLT_TRACE_MAX_EDGES,PLT_TRACE_EVENT_FLOATS,PLT_FAILURE_TRACE_WGSL,selectPLTTraceEvents,describePLTFailureTrace} from '../dist/plt-failure-trace.js';
import {createPLTScene,createPLTRandomStates,buildPLTBatch} from '../dist/plt-core.js';
import {executePLTTask} from '../dist/plt-runtime.js';
Object.assign(globalThis,globals);
const root=new URL('../',import.meta.url),read=name=>fs.readFileSync(new URL(`../dist/${name}`,import.meta.url),'utf8'),hash=value=>createHash('sha256').update(value).digest('hex');
const source=['utd-functions.wgsl','plt-compute.wgsl'].map(read).join('\n'),execute=process.argv.includes('--execute'),tests=[],failures=[];
const report={suite:'plt-failure-trace',schemaVersion:1,executionRequested:execute,browserValidated:false,physicalGPUValidated:false,
 scope:'Failure evidence, immutable original event readback, bounded resource lifecycle and canonical CPU fallback; no rendering-algorithm or physical-accuracy claim',
 sourceHashes:Object.fromEntries(['utd-functions.wgsl','plt-compute.wgsl','plt-gpu.js','plt-failure-trace.js','plt-runtime.js'].map(name=>[name,hash(read(name))])),
 testSHA256:hash(fs.readFileSync(new URL('./plt-failure-trace.mjs',import.meta.url))),tolerance:PLT_GPU_TOLERANCE,tests,failures,nativeCases:[]};
const test=async(name,body)=>{try{await body();tests.push(name);console.log(`PASS ${name}`);}catch(error){failures.push({name,error:error.stack});console.error(`FAIL ${name}: ${error.stack}`);}};
const defer=()=>{let resolve;const promise=new Promise(r=>resolve=r);return {promise,resolve};},tick=()=>new Promise(resolve=>setTimeout(resolve,0));
const words=array=>Array.from(new Uint32Array(array.buffer,array.byteOffset,array.length)),clone=value=>JSON.parse(JSON.stringify(value));
const powerStatement='eventResults[index].powers = vec4f(selected,sp,hp,f32(event.edgeRange.y));';
function mutateMain(extra){assert.equal(source.split(powerStatement).length,2);return source.replace(powerStatement,`${powerStatement} ${extra}`);}
const plusNine=mutateMain('if (index < 9u) { eventResults[index].powers.x += 1.0; }');
function packedCPU(packed,index,limit=Infinity){
 const u=new Uint32Array(packed.eventData),f=new Float32Array(packed.eventData),start=u[8*index],count=u[8*index+1],boundary=u[8*index+2];let sr=f[8*index+4],si=0,hr=sr,hi=0;
 for(let j=0;j<Math.min(count,limit);j++){
  const [k,n,pi,po,ro,sb,d,L]=packed.edgeData.subarray((start+j)*8,(start+j+1)*8),theta=-k*d,c=Math.cos(theta)/Math.sqrt(ro),s=Math.sin(theta)/Math.sqrt(ro);
  const soft=coefficient(k,n,pi,po,L,sb,'soft',2e-6),hard=coefficient(k,n,pi,po,L,sb,'hard',2e-6);
  sr+=soft[0]*c-soft[1]*s;si+=soft[0]*s+soft[1]*c;hr+=hard[0]*c-hard[1]*s;hi+=hard[0]*s+hard[1]*c;
 }const sp=sr*sr+si*si,hp=hr*hr+hi*hi;return [sr,si,hr,hi,boundary===0?sp:boundary===1?hp:(sp+hp)/2,sp,hp,count];
}
function synthetic(){
 const events=[0,1,2,5,7].map((count,index)=>({k:113,boundary:['soft','hard','unpolarized'][index%3],directAmplitude:.5,
  edges:Array.from({length:count},(_,j)=>({k:113+13*j,n:1.3+.06*j,phiI:.21+.03*j,phiO:1.1+.08*j,ro:.7+.13*j,sinBeta:.6+.03*j,phaseExcess:.002*(j+1)})),referenceF:0}));
 const job={width:events.length,height:1,samplesPerPixel:1,events,terms:[],reference:new Float64Array(events.length)},packed=packPLTGPU(job);events.forEach((event,i)=>event.referenceF=packedCPU(packed,i)[4]);return job;
}
function syntheticTrace(packed,ids){const out=new Float32Array(ids.length*PLT_TRACE_EVENT_FLOATS);ids.forEach((id,i)=>out.set(packedCPU(packed,id),i*PLT_TRACE_EVENT_FLOATS));return out;}
function checkTrace(trace,packed,original){
 assert.equal(trace.schema,PLT_TRACE_SCHEMA);assert.equal(trace.status,'captured');assert(trace.events.length<=8);assert.equal(trace.limits.edgesPerEvent,2);
 for(const event of trace.events){const index=event.index,raw=original.subarray(index*8,index*8+8);assert.deepEqual(event.originalGPUWords,words(raw));assert.deepEqual(event.cpuAtPackedInputs,packedCPU(packed,index));
  assert.equal(event.capturedEdges,Math.min(event.totalEdges,2));assert.equal(event.omittedEdges,event.totalEdges-event.capturedEdges);assert.equal(event.referenceF,packed.eventReference[index]);assert.equal(event.reconstructedGPU.length,8);
  for(const edge of event.edges){assert.equal(edge.gpu.length,96);assert.equal(edge.gpuWords.length,96);assert.equal(edge.cpuAtPackedInputs.length,96);assert.deepEqual(edge.cpuAtPackedInputs.slice(0,8),edge.packedInput);}
 }assert.deepEqual(clone(trace),trace);
}
function mockGPU(job,{compileFailure=false,mapFailure=false,probeWait=null,onProbe=()=>{}}={}){
 const packed=packPLTGPU(job),events=Float32Array.from(job.events.flatMap((_,i)=>packedCPU(packed,i)));events[3*8+4]+=1;
 const probe=syntheticTrace(packed,[3]),buffers=[],codes=[],entryPoints=[];let scopes=0,reads=0,destroyed=false;const lost=defer();
 const device={limits:{maxBufferSize:1e9,maxStorageBufferBindingSize:1e9},lost:lost.promise,addEventListener(){},removeEventListener(){},pushErrorScope(){scopes++;},async popErrorScope(){scopes--;return null;},
  createShaderModule({code,label}){codes.push(code);const diagnostic=label==='PLT failure-only observational rerun';return {async getCompilationInfo(){if(diagnostic){onProbe();if(probeWait)await probeWait;}return {messages:diagnostic&&compileFailure?[{type:'error',message:'injected probe compile failure'}]:[]};}};},
  async createComputePipelineAsync({compute}){entryPoints.push(compute.entryPoint);return {getBindGroupLayout:()=>({})};},
  createBuffer(options){const b={...options,destroyed:false,destroy(){this.destroyed=true;},async mapAsync(){if(mapFailure&&options.label==='PLT diagnostic readback')throw new Error('injected probe map failure');},getMappedRange(){reads++;const data=options.label==='PLT event readback'?events:options.label==='PLT diagnostic readback'?probe:Float32Array.from(job.reference);return data.buffer.slice(0);},unmap(){}};buffers.push(b);return b;},
  createBindGroup:()=>({}),createCommandEncoder:()=>({beginComputePass:()=>({setPipeline(){},setBindGroup(){},dispatchWorkgroups(){},end(){}}),copyBufferToBuffer(){},finish:()=>({})}),queue:{writeBuffer(){},submit(){},async onSubmittedWorkDone(){}},destroy(){destroyed=true;}};
 return {gpu:{requestAdapter:async()=>({info:{description:'CONTROL-FLOW MOCK ONLY'},requestDevice:async()=>device})},codes,entryPoints,events,check(){assert.equal(scopes,0);assert(destroyed);assert(buffers.every(b=>b.destroyed));},get reads(){return reads;}};
}
// Observe native source and mapped original readback without changing either.
function observeGPU(gpu){
 const codes=[],readbacks=[],bind=(object,overrides)=>new Proxy(object,{get(target,key){if(key in overrides)return overrides[key];const value=Reflect.get(target,key,target);return typeof value==='function'?value.bind(target):value;}});
 return {codes,readbacks,gpu:bind(gpu,{requestAdapter:async(...args)=>{const adapter=await gpu.requestAdapter(...args);return bind(adapter,{requestDevice:async(...options)=>{
  const device=await adapter.requestDevice(...options);return bind(device,{createShaderModule:options=>{codes.push(options.code);return device.createShaderModule(options);},createBuffer:options=>{
   const buffer=device.createBuffer(options);if(options.label==='PLT event readback'){const get=buffer.getMappedRange.bind(buffer);buffer.getMappedRange=(...args)=>{const range=get(...args);readbacks.push(new Float32Array(range.slice(0)));return range;};}return buffer;
  }});
 }});}})};
}
await test('all production WGSL outside the approved erfcx evaluator matches HEAD',()=>{
 const git=(...args)=>execFileSync('git',args,{cwd:root,encoding:'utf8'}),names=git('ls-files','dist/*.wgsl').trim().split('\n').filter(Boolean);assert(names.length>=2);report.head=git('rev-parse','HEAD').trim();report.unchangedShaderHashes={};
 for(const name of names){const current=fs.readFileSync(new URL(`../${name}`,import.meta.url)),baseline=execFileSync('git',['show',`HEAD:${name}`],{cwd:root});if(name==='dist/utd-functions.wgsl'){
  const strip=text=>text.replace(/fn utd_erfcx_scaled\(t:f32\)->vec2f \{[\s\S]*?\n\}/,'').replace('// The scalar erfcx evaluator is generated beside its coefficients above.','').replace('// Same degree-40 Chebyshev polynomial and coefficients as the f64 reference.','').replace('// Separate scalar streams and literal indices avoid vector/dynamic-array','').replace('// lowering in this numerically sensitive recurrence; no change to UTD physics.','').replace(/\n+/g,'\n');
  assert.equal(strip(current.toString()),strip(baseline.toString()),'unrelated UTD functions unchanged');
  const coefficients=text=>text.match(/const UTD_ERFCX_CHEBYSHEV[\s\S]*?\n\);/)[0];assert.equal(coefficients(current.toString()),coefficients(baseline.toString()),'Chebyshev coefficients unchanged');
 }else assert.equal(hash(current),hash(baseline),name);report.unchangedShaderHashes[name]=hash(current);}
});
await test('failed-event selection is ordered, unique, bounded and rejects invalid indices',()=>{
 assert.equal(PLT_TRACE_MAX_EVENTS,8);assert.equal(PLT_TRACE_MAX_EDGES,2);assert.equal(PLT_TRACE_EVENT_FLOATS,208);
 const failures=[-1,NaN,1.5,90,3,3,2,1,0,4,5,6,7,8,9].map(index=>({index}));assert.deepEqual(selectPLTTraceEvents({failures},10),[3,2,1,0,4,5,6,7]);assert.deepEqual(selectPLTTraceEvents({},10),[]);
});
await test('descriptor reconstructs all edges beyond capture cap and zero-edge direct fields',()=>{
 const job=synthetic(),packed=packPLTGPU(job),ids=[0,3,4],original=Float32Array.from(job.events.flatMap((_,i)=>packedCPU(packed,i))),output=syntheticTrace(packed,ids),trace=describePLTFailureTrace(packed,original,ids,output);
 checkTrace(trace,packed,original);assert.deepEqual(trace.events[0].cpuAtPackedInputs,[.5,0,.5,0,.25,.25,.25,0]);for(const event of trace.events.slice(1)){assert(event.totalEdges>2);assert.notEqual(event.cpuAtPackedInputs[4],packedCPU(packed,event.index,2)[4]);}
 assert.throws(()=>describePLTFailureTrace(packed,original,ids,new Float32Array(1)),/readback length/);
});
await test('NaN and infinities retain JSON-safe tags and exact original f32 words',()=>{
 const job=synthetic(),packed=packPLTGPU(job),raw=Float32Array.from(job.events.flatMap((_,i)=>packedCPU(packed,i))),bits=new Uint32Array(raw.buffer);bits[0]=0x7fc00001;bits[1]=0x7f800000;bits[2]=0xff800000;
 const out=syntheticTrace(packed,[0]);out[0]=NaN;const trace=describePLTFailureTrace(packed,raw,[0],out),event=clone(trace).events[0];assert.deepEqual(event.originalGPU.slice(0,3),['NaN','+Infinity','-Infinity']);assert.deepEqual(event.originalGPUWords.slice(0,3),[0x7fc00001,0x7f800000,0xff800000]);assert.equal(event.reconstructedGPU[0],'NaN');assert.equal(event.reconstruction.absoluteErrors[0],'NaN');assert.equal(event.reconstruction.bitExactAllLanes,false);report.nonfiniteSerialization=event;
});
await test('probe compile/map failures preserve original parity rejection and release every resource',async()=>{
 for(const kind of ['compileFailure','mapFailure']){const job=synthetic(),mock=mockGPU(job,{[kind]:true});let error;try{await renderPLTGPU(job,{gpu:mock.gpu,shaderSource:source});assert.fail('unverified image escaped');}catch(caught){error=caught;}
  assert.equal(error.name,'PLTGPUParityError');assert.match(error.message,/every-event CPU parity failed/);assert.equal(error.validation,error.parity);assert.equal(error.validation.failures[0].index,3);assert.equal(error.validation.failures[0].actual,mock.events[3*8+4]);assert.equal(error.diagnostic.failureTrace.status,'unavailable');assert.match(error.diagnostic.failureTrace.error,/injected probe/);const retained=clone(error.diagnostic.failureTrace).events[0];assert.deepEqual(retained.originalGPUWords,words(mock.events.subarray(24,32)));assert.deepEqual(retained.cpuAtPackedInputs,packedCPU(packPLTGPU(job),3));assert.equal(retained.reconstructedGPU,null);assert.equal(retained.edges[0].gpu,null);assert.equal(retained.edges[0].cpuAtPackedInputs.length,96);assert.equal(error.diagnostic.validation.rejected,error.validation);assert.equal(mock.codes[0],source);assert.equal(mock.codes[1],source+'\n'+PLT_FAILURE_TRACE_WGSL);mock.check();
 }
});
await test('cancellation during probe compilation withholds image and drains late resource scopes',async()=>{
 const ready=defer(),resume=defer(),job=synthetic(),mock=mockGPU(job,{probeWait:resume.promise,onProbe:ready.resolve}),controller=new AbortController();const engine=await makePLTGPUEngine(job,{gpu:mock.gpu,shaderSource:source}),pending=engine.run({signal:controller.signal});await ready.promise;controller.abort();await assert.rejects(pending,error=>error.name==='AbortError');resume.resolve();await tick();await tick();engine.destroy();mock.check();assert.equal(mock.reads,1);
});
await test('native Dawn/Tint compiles original event/camera and appended diagnostic entry points',async()=>{
 let gpu=create(['backend=null']);const adapter=await gpu.requestAdapter(),device=await adapter.requestDevice();try{for(const [code,entryPoints] of [[source,['plt_events','plt_camera']],[source+'\n'+PLT_FAILURE_TRACE_WGSL,['plt_failure_trace']]]){
  device.pushErrorScope('validation');const module=device.createShaderModule({code});assert.deepEqual((await module.getCompilationInfo()).messages.filter(m=>m.type==='error'),[]);assert.equal(await device.popErrorScope(),null);for(const entryPoint of entryPoints)await device.createComputePipelineAsync({layout:'auto',compute:{module,entryPoint}});
 }report.nativeCompilePassed=true;}finally{device.destroy();gpu=null;}
});
let exactJob;
if(execute){
 const backend=process.env.WEBGPU_EXECUTION_BACKEND??'vulkan';assert(['vulkan','metal','d3d11','d3d12','opengl','opengles'].includes(backend),'Executable backend required; null is refused');let gpu=create([`backend=${backend}`]);report.requestedBackend=backend;report.driverSelection={VK_ICD_FILENAMES:process.env.VK_ICD_FILENAMES??null};report.execution='Actual native Dawn/Tint compilation, dispatch and mapped readback';
 try{
  const options={width:48,height:36,samplesPerPixel:64,maxDepth:4,seed:7391,spectralMode:'visible',spectralSource:'D65',spectralMaterial:'red'},scene=createPLTScene({preset:'micro-gallery',...options}),tile={x:0,y:0,width:32,height:32};
  await test('exact reported first batch executes unchanged shaders without a diagnostic rerun',async()=>{
   exactJob=await buildPLTBatch(scene,scene.camera,options,{tile,sampleStart:0,samplesPerPixel:1,rngStates:createPLTRandomStates(options),yieldEvery:1024});const observed=observeGPU(gpu),result=await renderPLTGPU(exactJob,{gpu:observed.gpu,shaderSource:source});assert(result.validation.passed);assert.deepEqual(observed.codes,[source]);assert.equal(result.failureTrace,undefined);assert.equal(result.diagnostic,undefined);assert.equal(exactJob.events.length,429);assert.equal(exactJob.terms.length,594);report.adapter=result.adapter;
   report.fixture={preset:'micro-gallery',options,tile,sampleStart:0,batchSamples:1,eventCount:exactJob.events.length,termCount:exactJob.terms.length,packedEventSHA256:hash(Buffer.from(packPLTGPU(exactJob).eventData)),validation:result.validation};
  });
  await test('nine main-kernel power mutations retain original rejection and cap trace at eight',async()=>{
   assert(exactJob,'first-batch fixture missing');const packed=packPLTGPU(exactJob),observed=observeGPU(gpu);let error;try{await renderPLTGPU(exactJob,{gpu:observed.gpu,shaderSource:plusNine});assert.fail('mutated image escaped');}catch(caught){error=caught;}
   assert.equal(error.name,'PLTGPUParityError');assert.match(error.message,/every-event CPU parity failed/);assert.equal(error.validation,error.parity);assert.deepEqual(error.validation.failures.map(f=>f.index),[0,1,2,3,4,5,6,7,8]);assert.equal(observed.readbacks.length,1);
   const original=observed.readbacks[0].subarray(0,packed.eventCount*8),trace=error.diagnostic.failureTrace;checkTrace(trace,packed,original);assert.equal(trace.events.length,8);assert.equal(error.diagnostic.validation.events,null);assert.equal(error.diagnostic.validation.pixels,null);assert.equal(error.diagnostic.validation.rejected,error.validation);assert.deepEqual(observed.codes,[plusNine,plusNine+'\n'+PLT_FAILURE_TRACE_WGSL]);
   for(const event of trace.events){assert.equal(event.reconstruction.bitExactSelectedPower,false);assert.equal(event.reconstruction.selectedPowerWithinOriginalGate,false);assert(Math.abs(event.originalGPU[4]-event.reconstructedGPU[4]-1)<1e-5);const failure=error.validation.failures.find(f=>f.index===event.index);assert.equal(failure.actual,event.originalGPU[4]);assert.equal(failure.expected,event.referenceF);}
   report.nativeCases.push({name:'reported first batch; first nine selected powers +1',validation:error.validation,trace,sourcePrefixUnchanged:true});
  });
  await test('unchanged main/probe arithmetic reproduces original raw lanes and all-edge fields',async()=>{
   const job=synthetic();job.events.forEach(event=>event.referenceF+=1);const packed=packPLTGPU(job),observed=observeGPU(gpu);let error;try{await renderPLTGPU(job,{gpu:observed.gpu,shaderSource:source});assert.fail('changed references passed');}catch(caught){error=caught;}
   assert.equal(error.name,'PLTGPUParityError');assert.deepEqual(observed.codes,[source,source+'\n'+PLT_FAILURE_TRACE_WGSL]);const trace=error.diagnostic.failureTrace;checkTrace(trace,packed,observed.readbacks[0].subarray(0,packed.eventCount*8));assert.equal(trace.events.length,5);
   for(const event of trace.events){assert(event.reconstruction.selectedPowerWithinOriginalGate);for(let lane=0;lane<8;lane++)assert(event.reconstruction.absoluteErrors[lane]<=2e-6+3e-5*Math.abs(event.originalGPU[lane]),`event ${event.index} lane ${lane}`);}
   assert(trace.events.some(event=>event.totalEdges>2&&event.capturedEdges===2));assert.equal(trace.events[0].totalEdges,0);report.nativeCases.push({name:'unmodified source reconstruction; zero through seven edges',trace,sourcePrefixUnchanged:true,bitExactEvents:trace.events.filter(event=>event.reconstruction.bitExactAllLanes).length});
  });
  await test('native NaN mutation preserves tagged original readback and unsigned raw words',async()=>{
   const job=synthetic(),modified=mutateMain('if (index == 0u) { eventResults[index].powers.x = bitcast<f32>(0x7fc00001u + index); }');let error;try{await renderPLTGPU(job,{gpu,shaderSource:modified});assert.fail('NaN image escaped');}catch(caught){error=caught;}
   assert.equal(error.name,'PLTGPUParityError');const trace=clone(error.diagnostic.failureTrace),event=trace.events[0];assert.equal(event.index,0);assert.equal(event.originalGPU[4],'NaN');assert.equal((event.originalGPUWords[4]&0x7f800000)>>>0,0x7f800000);assert.notEqual(event.originalGPUWords[4]&0x7fffff,0);assert.equal(event.reconstruction.selectedPowerWithinOriginalGate,false);assert.equal(event.reconstructedGPU[4],.25);report.nativeCases.push({name:'native NaN original power',trace});
  });
  await test('second native GPU batch failure preserves deep JSON trace and replaces entire XYZ with canonical CPU',async()=>{
   const request={backend:'auto',compare:false,scene:{preset:'micro-gallery'},options:{width:8,height:8,samplesPerPixel:2,maxDepth:4,seed:7391,spectralMode:'visible',spectralSource:'D65',spectralMaterial:'red'},progressive:{tileSize:8,batchSamples:1,previewIntervalMs:0}};let calls=0,capturedError,acceptedXYZ;const previews=[];
   const result=await executePLTTask(request,{preview:p=>previews.push(p),gpuRunner:async(job,hook)=>{calls++;try{const value=await renderPLTGPU(job,{...hook,gpu,shaderSource:calls===2?plusNine:source});acceptedXYZ=value.xyz.slice();return value;}catch(error){capturedError=error;throw error;}}}),cpu=await executePLTTask({...request,backend:'cpu'});
   assert.equal(calls,2);assert(acceptedXYZ instanceof Float32Array);assert.equal(capturedError.name,'PLTGPUParityError');assert.equal(result.results.length,1);const actual=result.results[0],expected=cpu.results[0];assert.deepEqual(actual.image,expected.image);assert.deepEqual(actual.xyz,expected.xyz);assert.equal(actual.backend,'CPU 双精度');assert.equal(actual.validation.passed,false);assert.equal(actual.validation.acceptedBatches,1);assert.equal(actual.validation.rejectedBatches,1);assert(actual.sampleCounts.every(n=>n===2));
   const saved=clone(result),trace=saved.results[0].validation.diagnostic.failureTrace;assert.deepEqual(trace,clone(capturedError.diagnostic.failureTrace));assert.equal(trace.status,'captured');assert(trace.events.some(event=>event.edges.length));const deep=trace.events.find(event=>event.edges.length).edges[0];assert.equal(deep.gpu.length,96);assert.equal(deep.cpuAtPackedInputs.length,96);assert.equal(deep.packedInputWords.length,8);assert.deepEqual(actual.xyz,previews.at(-1).results[0].xyz);assert(result.warnings.some(w=>w.includes('parity failed')));
   report.nativeCases.push({name:'second batch rejection resets complete XYZ and preserves deep trace',gpuCalls:calls,canonicalCPUExact:true,acceptedBatches:actual.validation.acceptedBatches,rejectedBatches:actual.validation.rejectedBatches,traceBytes:JSON.stringify(trace).length,deepEdgeLanes:deep.gpu.length,validation:saved.results[0].validation});
  });
 }finally{gpu=null;}
}
report.passed=failures.length===0;report.passCount=tests.length;report.failCount=failures.length;report.completedAt=new Date().toISOString();const path=process.env.PLT_TRACE_REPORT??new URL('../research/plt-failure-trace-validation.json',import.meta.url);fs.writeFileSync(path,JSON.stringify(report,(_,value)=>ArrayBuffer.isView(value)?Array.from(value):typeof value==='number'&&!Number.isFinite(value)?String(value):value,2)+'\n');console.log(`${tests.length} passed, ${failures.length} failed`);if(failures.length)process.exitCode=1;
