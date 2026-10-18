/** Host lifecycle mocks ONLY: no Dawn device, shader compilation, or GPU math.
 * Fixed synthetic readback bytes exercise resource ownership and fault paths.
 */
import assert from 'node:assert/strict';
import {createPLTScene,createPLTRandomStates} from '../dist/plt-core.js';
import {createPLTResidentRunner,packResidentParameters,PLT_RESIDENT_LIMITS} from '../dist/plt-resident-gpu.js';
// The mock implements only the WebGPU flags the host uses; importing this suite
// must never initialize a native/software GPU while other precision tests run.
globalThis.GPUBufferUsage={MAP_READ:1,COPY_SRC:4,COPY_DST:8,UNIFORM:64,STORAGE:128};
globalThis.GPUMapMode={READ:1};
const STAGES=['rr_init','rr_traverse_step','rr_records_step','rr_operator_step','rr_scatter_step'];
const scene=createPLTScene(),options={width:8,height:8,samplesPerPixel:4,maxDepth:2,wave:false},tile={x:0,y:0,width:1,height:1};
const fresh=()=>({tile,sampleStart:0,samplesPerPixel:1,rngStates:createPLTRandomStates(options)});
const delay=(ms=0)=>new Promise(resolve=>setTimeout(resolve,ms));
const deferred=()=>{let resolve,reject;const promise=new Promise((a,b)=>{resolve=a;reject=b;});return {promise,resolve,reject};};
const never=()=>new Promise(()=>{});
const shaderSource='host-control-flow-mock-only';
let passed=0;const failures=[];
async function test(name,fn){try{await fn();passed++;console.log('PASS',name);}catch(error){failures.push({name,error:error.stack});console.error('FAIL',name,error.stack);}}
function mockGPU(config={}){
 const lost=deferred(),listeners=new Map(),buffers=[],compiled=[],dispatched=[],parameterSnapshots=[];
 let scopes=0,popCount=0,deviceDestroyCount=0,adapterRequests=0,deviceRequests=0,reads=0,mapCalls=0,unmaps=0,submitCount=0,currentStage=null,uniform=new Uint32Array(40),initial=new Uint32Array(4);
 const fixture=(label,bytes)=>{
  const buffer=new ArrayBuffer(bytes),u=new Uint32Array(buffer),f=new Float32Array(buffer),pixels=uniform[6]*uniform[7],samples=uniform[9];
  if(label==='Resident status readback')u.set([config.statusActive??0,config.statusFlags??0,config.recordCount??0,0]);
  if(label==='Resident lane readback')for(let lane=0;lane<pixels;lane++){const at=lane*PLT_RESIDENT_LIMITS.laneBytes/4;u[at]=initial[lane*4];u[at+1]=initial[lane*4+1];u[at+2]=config.badPrefix?0:uniform[8]+samples;u[at+3]=2;}
  if(label==='Resident audit readback')for(let lane=0;lane<pixels;lane++)for(let sample=0;sample<samples;sample++){
   const at=(lane*samples+sample)*PLT_RESIDENT_LIMITS.auditVectors*4;
   u.set([initial[lane*4],uniform[8]+sample,uniform[11],0],at);u[at+4]=initial[lane*4+1];u[at+5]=initial[lane*4+1];u[at+6]=2;u[at+7]=2166136261;u[at+8]=1;u[at+11]=1;u[at+81]=0xffffffff;f[at+90]=-1;if(config.recordCount===1&&lane===0&&sample===0){u[at+9]=1;u[at+1232]=1;u[at+1243]=0;}
  }
  if(label==='Resident edge readback'&&config.recordCount===1)u.set([0,0,42,0xffffffff]);
  return buffer;
 };
 const device={limits:{maxBufferSize:config.limit??1e9,maxStorageBufferBindingSize:config.storageLimit??1e9},lost:lost.promise,
  addEventListener:(name,fn)=>listeners.set(name,fn),removeEventListener:name=>listeners.delete(name),
  pushErrorScope(){scopes++;config.onPush?.(scopes);},
  async popErrorScope(){scopes--;assert.ok(scopes>=0,'Unbalanced error scope pop');popCount++;if(config.popWait)await config.popWait;if(config.popError)throw new Error('mock pop failed');return config.validationError&&(!config.validationPop||config.validationPop===popCount)?{message:'mock validation failed'}:null;},
  createShaderModule(){if(config.moduleError)throw new Error('mock module failed');return {async getCompilationInfo(){config.onCompile?.();if(config.compileWait)await config.compileWait;return {messages:config.compileError?[{type:'error',lineNum:1,linePos:1,message:'mock compile failed'}]:[]};}};},
  async createComputePipelineAsync({compute:{entryPoint}}){compiled.push(entryPoint);if(config.pipelineWait)await config.pipelineWait;if(config.pipelineError===true||config.pipelineError===entryPoint)throw new Error('mock pipeline failed');return {entryPoint,getBindGroupLayout:()=>({entryPoint})};},
  createBuffer(spec){if(config.bufferError===true||config.bufferError===spec.label)throw new Error('mock buffer failed');const b={...spec,destroyCount:0,mapped:false,
   destroy(){this.destroyCount++;this.mapped=false;},
   async mapAsync(mode,offset,bytes){assert.equal(mode,GPUMapMode.READ);assert.equal(offset,0);assert.ok(bytes>0&&bytes<=this.size);mapCalls++;config.onMap?.(this.label);
    if(config.mapWait&&(!config.mapLabel||config.mapLabel===this.label))await config.mapWait;
    if(this.destroyCount)throw Object.assign(new Error('destroyed map'),{name:'AbortError'});
    if(config.mapError&&(!config.mapLabel||config.mapLabel===this.label))throw new Error('mock map failed');this.mapped=true;
   },
   getMappedRange(offset,bytes){assert.equal(offset,0);assert.ok(this.mapped&&!this.destroyCount,'Readback after destroy/unmap');reads++;if(config.rangeError&&(!config.mapLabel||config.mapLabel===this.label))throw new Error('mock range failed');return config.shortRead?new ArrayBuffer(0):fixture(this.label,bytes);},
   unmap(){assert.ok(this.mapped);this.mapped=false;unmaps++;},
  };buffers.push(b);return b;},
  createBindGroup(spec){if(config.bindError)throw new Error('mock bind failed');return spec;},
  createCommandEncoder(){if(config.encoderError)throw new Error('mock encoder failed');const commands=[];return {
   beginComputePass(){let pipeline;return {setPipeline:p=>pipeline=p,setBindGroup(){},dispatchWorkgroups(count){assert.ok(count>=1);commands.push({stage:pipeline.entryPoint,count});},end(){}};},
   copyBufferToBuffer(source,sourceOffset,target,targetOffset,bytes){if(config.copyError)throw new Error('mock copy failed');assert.equal(sourceOffset,0);assert.equal(targetOffset,0);assert.ok(bytes<=source.size&&bytes<=target.size);commands.push({copy:true});},finish:()=>({commands}),
  };},
  queue:{
   writeBuffer(buffer,offset,data){if(config.writeError)throw new Error('mock write failed');assert.ok(!deviceDestroyCount,'Write after device destruction');assert.ok(offset+data.byteLength<=buffer.size);if(buffer.label==='Resident parameters'){const words=new Uint32Array(data.buffer??data,data.byteOffset??0,data.byteLength/4);uniform.set(words,offset/4);if(offset===0&&data.byteLength===160)parameterSnapshots.push(uniform.slice());}if(buffer.label==='Resident initial prefixes')initial=new Uint32Array(data).slice();},
   submit(commands){submitCount++;if(config.submitError)throw new Error('mock submit failed');for(const command of commands.flatMap(x=>x.commands)){if(command.stage){currentStage=command.stage;dispatched.push(currentStage);config.onDispatch?.(currentStage);}}},
   async onSubmittedWorkDone(){const stage=currentStage;config.onWork?.(stage);if(config.workWait)await config.workWait;if(config.queueError)throw new Error('mock queue failed');},
  },
  destroy(){deviceDestroyCount++;},
 };
 const adapter={info:{description:'HOST LIFECYCLE MOCK; NO GPU EXECUTION'},requestDevice(){deviceRequests++;config.onDevice?.();const pending=(async()=>{if(config.deviceWait)await config.deviceWait;if(config.deviceError)throw new Error('mock device request failed');return device;})();config.afterDeviceRequest?.();return pending;}};
 const mock={parameterSnapshots,gpu:{async requestAdapter(){adapterRequests++;config.onAdapter?.();if(config.adapterWait)await config.adapterWait;if(config.adapterError)throw new Error('mock adapter failed');return config.noAdapter?null:adapter;}},device,buffers,compiled,dispatched,
  lose:message=>lost.resolve({message}),uncaptured:message=>listeners.get('uncapturederror')?.({error:{message}}),
  get deviceDestroyCount(){return deviceDestroyCount;},get adapterRequests(){return adapterRequests;},get deviceRequests(){return deviceRequests;},get reads(){return reads;},get mapCalls(){return mapCalls;},get unmaps(){return unmaps;},get submitCount(){return submitCount;},
  check({destroyed=deviceRequests>0&&!config.deviceError}={}){assert.equal(scopes,0,'Leaked error scopes');assert.equal(deviceDestroyCount,destroyed?1:0,'Device must be disposed exactly once');assert.equal(listeners.size,0,'Device listener leak');for(const b of buffers){assert.equal(b.destroyCount,1,`Buffer ${b.label} must be destroyed exactly once`);assert.equal(b.mapped,false,'Mapped buffer leak');}},
 };
 return mock;
}
const create=(mock,extra={})=>createPLTResidentRunner(scene,scene.camera,options,{gpu:mock.gpu,shaderSource,timeoutMs:500,...extra});
async function withRunner(config,fn,extra={}){const mock=mockGPU(config);let runner;try{runner=await create(mock,extra);await fn(runner,mock);}finally{runner?.destroy();}await delay();mock.check();}
await test('successful candidates stay unadmitted; one scene/device/five pipelines reused',async()=>{
 await withRunner({},async(runner,mock)=>{for(let i=0;i<2;i++){const result=await runner.run(fresh());assert.equal(result.candidate,true);assert.equal(result.admitted,false);assert.equal(result.validation,undefined);assert.strictEqual(result.image,result.audit.reference);assert.equal(result.audit.paths.length,1);assert.equal(result.timings.cpuReferenceProvided,false);assert.equal(mock.deviceDestroyCount,0);}assert.deepEqual(mock.compiled,STAGES);assert.equal(mock.adapterRequests,1);assert.equal(mock.deviceRequests,1);assert.deepEqual(mock.dispatched,[...STAGES,...STAGES]);assert.ok(mock.buffers.every(b=>b.destroyCount===0));});
});
await test('camera is deeply snapshotted before async setup and reused immutable across batches',async()=>{
 const camera=structuredClone(scene.camera),original=structuredClone(camera),held=deferred(),entered=deferred(),mock=mockGPU({adapterWait:held.promise,onAdapter:entered.resolve});let runner;
 const pending=createPLTResidentRunner(scene,camera,options,{gpu:mock.gpu,shaderSource,timeoutMs:500});
 try{await entered.promise;camera.position[0]+=100;camera.target[1]-=99;camera.up[2]=.5;camera.fov=77;held.resolve();runner=await pending;
  await runner.run(fresh());const expected=new Uint32Array(packResidentParameters(original,options,runner.packedScene,fresh()));assert.deepEqual(mock.parameterSnapshots[0],expected);
  camera.position.fill(NaN);camera.target=[NaN,NaN,NaN];camera.up.fill(NaN);camera.fov=NaN;
  await runner.run({...fresh(),sampleStart:2});assert.deepEqual(mock.parameterSnapshots[1].slice(16,36),expected.slice(16,36));
 }finally{runner?.destroy();}mock.check();
});
await test('no GPU and pre-cancel request no adapter/device',async()=>{
 await assert.rejects(createPLTResidentRunner(scene,scene.camera,options,{gpu:null,shaderSource}),/unavailable/);
 const controller=new AbortController();controller.abort();const mock=mockGPU();await assert.rejects(create(mock,{signal:controller.signal}),{name:'AbortError'});assert.equal(mock.adapterRequests,0);mock.check({destroyed:false});
});
await test('adapter failure, absence and timeout do not create a device',async()=>{
 for(const config of [{noAdapter:true},{adapterError:true},{adapterWait:never()}]){const mock=mockGPU(config);await assert.rejects(create(mock,{timeoutMs:10}));assert.equal(mock.deviceRequests,0);mock.check({destroyed:false});}
});
await test('late adapter after cancellation never requests a device',async()=>{
 const held=deferred(),entered=deferred(),controller=new AbortController(),mock=mockGPU({adapterWait:held.promise,onAdapter:entered.resolve});const pending=create(mock,{signal:controller.signal});await entered.promise;controller.abort();await assert.rejects(pending,{name:'AbortError'});held.resolve();await delay();assert.equal(mock.deviceRequests,0);mock.check({destroyed:false});
});
await test('device request rejection cleans setup',async()=>{const mock=mockGPU({deviceError:true});await assert.rejects(create(mock),/device request failed/);mock.check({destroyed:false});});
await test('late device after cancellation or timeout disposed exactly once',async()=>{
 for(const mode of ['cancel','timeout']){const held=deferred(),entered=deferred(),controller=new AbortController(),mock=mockGPU({deviceWait:held.promise,onDevice:entered.resolve});const pending=create(mock,{signal:controller.signal,timeoutMs:mode==='timeout'?10:500});await entered.promise;if(mode==='cancel')controller.abort();await assert.rejects(pending);held.resolve();await delay();mock.check();}
});
await test('device adoption microtask cancellation disposes exactly once',async()=>{const controller=new AbortController(),mock=mockGPU({afterDeviceRequest:()=>queueMicrotask(()=>controller.abort())});await assert.rejects(create(mock,{signal:controller.signal}),{name:'AbortError'});await delay();mock.check();});
await test('module, compilation, pipeline, scope and buffer failures release all ownership',async()=>{
 for(const config of [{moduleError:true},{compileError:true},{pipelineError:true},{compileWait:never()},{pipelineWait:never()},{popError:true},{validationError:true},{bufferError:'Resident bounded edge records'},{bindError:true},{writeError:true},{limit:16},{storageLimit:16}]){const mock=mockGPU(config);await assert.rejects(create(mock,{timeoutMs:10}));await delay();mock.check();}
});
await test('setup progress cancellation at each pipeline never proceeds to later pipelines',async()=>{
 for(const entry of STAGES){const controller=new AbortController(),mock=mockGPU();await assert.rejects(create(mock,{signal:controller.signal,onSetupProgress:p=>{if(p.entryPoint===entry)controller.abort();}}),{name:'AbortError'});assert.deepEqual(mock.compiled,STAGES.slice(0,STAGES.indexOf(entry)));mock.check();}
});
await test('setup callback errors drain scopes and release device',async()=>{const mock=mockGPU();await assert.rejects(create(mock,{onSetupProgress:()=>{throw new Error('callback failed');}}),/callback failed/);mock.check();});
await test('dispatch, queue, encoder, copy, map, range and malformed readback failures withhold candidate',async()=>{
 for(const config of [{submitError:true},{queueError:true},{encoderError:true},{copyError:true},{mapError:true},{rangeError:true},{recordCount:1,mapError:true,mapLabel:'Resident edge readback'},{recordCount:1,rangeError:true,mapLabel:'Resident edge readback'},{shortRead:true},{badPrefix:true}])await withRunner(config,async runner=>{await assert.rejects(runner.run(fresh()));await assert.rejects(runner.run(fresh()),{name:'AbortError'});});
});
await test('queue/readback/scoped-operation timeouts free all resources',async()=>{
 for(const config of [{workWait:never()},{mapWait:never()}])await withRunner(config,async runner=>assert.rejects(runner.run(fresh()),/timeout/),{timeoutMs:10});
 const mock=mockGPU({popWait:never()});await assert.rejects(create(mock,{timeoutMs:10}),/timeout/);mock.check();
});
await test('cancellation immediately after each of the five dispatches blocks every later pass',async()=>{
 for(const stage of STAGES){const controller=new AbortController();await withRunner({onWork:entry=>{if(entry===stage)controller.abort();}},async(runner,mock)=>{await assert.rejects(runner.run({...fresh(),signal:controller.signal}),{name:'AbortError'});assert.deepEqual(mock.dispatched,STAGES.slice(0,STAGES.indexOf(stage)+1));assert.equal(mock.reads,0);});}
});
await test('cancellation during inter-pass yield submits no later stage',async()=>{
 for(const stage of STAGES.slice(1,-1)){const controller=new AbortController();await withRunner({onWork:entry=>{if(entry===stage)setTimeout(()=>controller.abort(),0);}},async(runner,mock)=>{await assert.rejects(runner.run({...fresh(),signal:controller.signal}),{name:'AbortError'});assert.deepEqual(mock.dispatched,STAGES.slice(0,STAGES.indexOf(stage)+1));});}
});
await test('cancellation in each readback suppresses late mapping and candidate publication',async()=>{
 for(const label of ['Resident status readback','Resident lane readback','Resident audit readback','Resident edge readback']){const held=deferred(),entered=deferred(),controller=new AbortController();await withRunner({recordCount:label==='Resident edge readback'?1:0,mapWait:held.promise,mapLabel:label,onMap:current=>{if(current===label)entered.resolve();}},async(runner,mock)=>{const pending=runner.run({...fresh(),signal:controller.signal});await entered.promise;const reads=mock.reads;controller.abort();await assert.rejects(pending,{name:'AbortError'});held.resolve();await delay();assert.equal(mock.reads,reads);});}
});
await test('device loss and uncaptured errors interrupt pending dispatch without its completion',async()=>{
 for(const method of ['lose','uncaptured']){const held=deferred(),entered=deferred();await withRunner({workWait:held.promise,onWork:entered.resolve},async(runner,mock)=>{const pending=runner.run(fresh());await entered.promise;mock[method]('mock lost device');await assert.rejects(pending,/mock lost device/);held.resolve();await delay();assert.equal(mock.reads,0);await assert.rejects(runner.run(fresh()),/mock lost device/);});}
});
await test('concurrent batch rejects without poisoning existing work',async()=>{
 const held=deferred(),entered=deferred();await withRunner({workWait:held.promise,onWork:entered.resolve},async runner=>{const first=runner.run(fresh());await entered.promise;await assert.rejects(runner.run(fresh()),/sequential/);held.resolve();const result=await first;assert.equal(result.candidate,true);});
});
await test('explicit repeated destroy and destroyed runner reuse stay closed',async()=>{await withRunner({},async(runner,mock)=>{runner.destroy();runner.destroy();await assert.rejects(runner.run(fresh()),{name:'AbortError'});assert.equal(mock.deviceDestroyCount,1);});});
await test('owner cancellation closes a successfully reused idle session',async()=>{const controller=new AbortController();await withRunner({},async(runner,mock)=>{await runner.run(fresh());controller.abort();await assert.rejects(runner.run(fresh()),{name:'AbortError'});assert.equal(mock.deviceDestroyCount,1);},{signal:controller.signal});});
await test('onProgress error or cancellation never returns a candidate',async()=>{
 await withRunner({},async runner=>assert.rejects(runner.run({...fresh(),onProgress:()=>{throw new Error('progress failed');}}),/progress failed/));
 const controller=new AbortController();await withRunner({},async runner=>assert.rejects(runner.run({...fresh(),signal:controller.signal,onProgress:()=>controller.abort()}),{name:'AbortError'}));
});
await test('GPU diagnostic flags and record overflow fail closed before audit readback',async()=>{
 for(const config of [{statusFlags:2},{recordCount:131073}])await withRunner(config,async(runner,mock)=>{await assert.rejects(runner.run(fresh()));assert.ok(mock.reads<=2);});
});
console.log(JSON.stringify({allPass:failures.length===0,passed,failed:failures.length,hostMockOnly:true,actualGPUExecuted:false,failures},null,2));if(failures.length)process.exitCode=1;
