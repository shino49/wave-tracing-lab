/** Bounded, explicitly invoked, experimental hardware test. Runs inside the
 * dedicated worker. It never modifies the stable renderer or uploads a CPU tape.
 */
import {createPLTRandomStates,randomGenerator} from './plt-core.js';
import {createPLTResidentRunner,PLT_RESIDENT_BUILD_ID} from './plt-resident-gpu.js';
import {buildResidentReferenceBatch,residentPathKey,residentHashWords,RESIDENT_AUDIT_TOLERANCES} from './plt-resident-reference.js';
import {admitResidentWithRecovery} from './plt-resident-recovery.js';
import {RESIDENT_HARDWARE_TEST_PRESETS,createResidentHardwareTestFixture} from './plt-resident-test-presets.js';
export {RESIDENT_HARDWARE_TEST_PRESETS};
export const RESIDENT_HARDWARE_TEST_BUILD_ID='plt-resident-hardware-test-2026-10-13.2';
export const RESIDENT_HARDWARE_TEST_SCHEMA_VERSION=2;
export const RESIDENT_HARDWARE_TEST_MAX_TOTAL_MS=240000;
const now=()=>globalThis.performance?.now?.()??Date.now();
const assert=(condition,message)=>{if(!condition)throw new Error(`Resident hardware test: ${message}`);};
const boundedString=value=>typeof value==='string'?value.slice(0,2000):null;
const abortError=()=>Object.assign(new Error('Resident hardware test cancelled; no image committed'),{name:'AbortError'});
const timeoutError=()=>Object.assign(new Error('Resident hardware test exceeded its total deadline; no image committed'),{name:'TimeoutError'});

export function validateResidentHardwareTestRequest(request={}){
 if(!request||typeof request!=='object'||Array.isArray(request)||![Object.prototype,null].includes(Object.getPrototypeOf(request)))throw new TypeError('Hardware-test request must be a plain object');
 if(Object.keys(request).some(key=>!['preset','size','laneChunkSize'].includes(key)))throw new RangeError('Only preset, size and laneChunkSize are accepted; custom scenes, wave and spectral modes are unavailable');
 const {preset='two-mirrors-emission',size=16,laneChunkSize=128}=request;
 if(!RESIDENT_HARDWARE_TEST_PRESETS.some(p=>p.id===preset))throw new RangeError('Choose one of the four resident hardware-test presets');
 if(![8,16,32].includes(size))throw new RangeError('Hardware-test size must be 8, 16 or 32 square pixels');
 if(![32,128].includes(laneChunkSize))throw new RangeError('Hardware-test laneChunkSize must be 32 or 128');
 return {preset,size,laneChunkSize,width:size,height:size,samplesPerPixel:2,maxDepth:6,seed:7391,wave:false,spectral:false,nee:false,mode:'ray-only-monochrome',spectralMode:'monochrome',wavelengthMm:.025,boundary:'unpolarized',guard:'axis-mirror-camera',axisSubmission:'round',recoveryMode:'selective',sampleStart:0,tile:{x:0,y:0,width:size,height:size}};
}
function environmentSummary(){
 const n=globalThis.navigator;
 return {userAgent:boundedString(n?.userAgent),platform:boundedString(n?.userAgentData?.platform??n?.platform),language:boundedString(n?.language),hardwareConcurrency:Number.isFinite(n?.hardwareConcurrency)?n.hardwareConcurrency:null,deviceMemoryGiB:Number.isFinite(n?.deviceMemory)?n.deviceMemory:null,isSecureContext:typeof globalThis.isSecureContext==='boolean'?globalThis.isSecureContext:null,worker:typeof WorkerGlobalScope!=='undefined'&&globalThis instanceof WorkerGlobalScope,browserVersion:boundedString(n?.userAgent)??'unknown',osVersion:'unknown',driverVersion:'unknown'};
}
function adapterSummary(adapter){
 const out={vendor:null,architecture:null,device:null,description:null,isFallbackAdapter:null};
 for(const key of Object.keys(out))try{const value=adapter.info?.[key]??(key==='isFallbackAdapter'?adapter.isFallbackAdapter:null);if(typeof value==='string'||typeof value==='boolean')out[key]=typeof value==='string'?boundedString(value):value;}catch{}
 return out;
}
const forward=(target,key)=>{const value=Reflect.get(target,key,target);return typeof value==='function'?value.bind(target):value;};
/** Transparent observation of genuine API calls; native objects passed to
 * pipelines/bind groups/queues are unwrapped. No cached or fake reference work.
 */
function observeGPU(gpu,state,compilation,emit,check){
 if(!gpu)return undefined;
 return {async requestAdapter(options){
  emit('adapter','Requesting a WebGPU adapter');const adapter=await gpu.requestAdapter(options);check();
  if(!adapter)return null;state.adapterAvailable=true;state.adapter=adapterSummary(adapter);
  return new Proxy(adapter,{get(target,key){
   if(key!=='requestDevice')return forward(target,key);
   return async descriptor=>{
    emit('device','Creating a WebGPU device');const device=await target.requestDevice(descriptor);state.deviceCreated=true;
    // The core owns disposal, including a device resolving after cancellation.
    const modules=new WeakMap(),computeBuffers=new WeakSet();let pendingCompute=0;
    const queue=new Proxy(device.queue,{get(q,k){
     if(k==='submit')return commands=>{const list=Array.from(commands);q.submit(list);const count=list.filter(c=>computeBuffers.has(c)).length;state.computeSubmissions+=count;pendingCompute+=count;};
     if(k==='onSubmittedWorkDone')return async()=>{const pending=pendingCompute;await q.onSubmittedWorkDone();if(pending){state.completedComputeSubmissions+=pending;pendingCompute=Math.max(0,pendingCompute-pending);}};
     return forward(q,k);
    }});
    return new Proxy(device,{get(d,k){
     if(k==='queue')return queue;
     if(k==='createCommandEncoder')return descriptor=>{
      const encoder=d.createCommandEncoder(descriptor);let compute=false;
      return new Proxy(encoder,{get(e,property){
       if(property==='beginComputePass')return descriptor=>{compute=true;return e.beginComputePass(descriptor);};
       if(property==='finish')return descriptor=>{const buffer=e.finish(descriptor);if(compute)computeBuffers.add(buffer);return buffer;};
       return forward(e,property);
      }});
     };
     if(k==='createShaderModule')return descriptor=>{
      emit('shader-compilation','Compiling the resident WGSL module');const started=now(),module=d.createShaderModule(descriptor);compilation.apiWallMs+=now()-started;compilation.moduleCount++;
      const wrapper=new Proxy(module,{get(m,property){
       if(property==='getCompilationInfo')return async()=>{const began=now();try{return await m.getCompilationInfo();}finally{compilation.apiWallMs+=now()-began;}};
       return forward(m,property);
      }});modules.set(wrapper,module);return wrapper;
     };
     if(k==='createComputePipelineAsync')return async descriptor=>{
      const began=now();try{return await d.createComputePipelineAsync({...descriptor,compute:{...descriptor.compute,module:modules.get(descriptor.compute.module)??descriptor.compute.module}});}finally{compilation.apiWallMs+=now()-began;compilation.pipelineCount++;}
     };
     return forward(d,k);
    }});
   };
  }});
 }};
}
function failureSummary(error,stage){
 return {stage,name:boundedString(error?.name)??'Error',code:boundedString(error?.code),message:boundedString(error?.message)??'Unknown resident test failure',mismatchCount:Number.isSafeInteger(error?.audit?.mismatchCount)?error.audit.mismatchCount:null,diagnosticFlags:Number.isSafeInteger(error?.diagnostic?.flags)?error.diagnostic.flags:null};
}
function histogram(values){const out={};for(const value of values){const key=String(value);out[key]=(out[key]??0)+1;}return out;}
function rngEqual(a,b){return a&&b&&['start','final','count','hash'].every(k=>a[k]===b[k])&&a.words.length===b.words.length&&a.words.every((word,i)=>word===b.words[i]);}
function validateDyadicPath(path,fixture){
 assert(path&&path.scatters.every(s=>s.type==='mirror')&&path.localEvents.length===0,'CPU oracle permits mirror-only paths without local operators');
 const mirrors=path.scatters.length;
 assert(path.vertices.length===mirrors+1&&path.termination.reason==='absorbed'&&path.termination.depth===mirrors,'CPU oracle requires complete terminal surface history');
 let throughput=1;const materials=[];
 for(let i=0;i<path.vertices.length;i++){
  const vertex=path.vertices[i],triangle=fixture.scene.triangles[vertex.triangleId];
  assert(vertex.type==='surface'&&triangle?.id===vertex.triangleId,'CPU oracle invalid surface ownership');
  const material=fixture.scene.materials[triangle.materialIndex];materials.push(triangle.materialIndex);
  if(i<mirrors){assert(material.type==='mirror'&&material.emission===0&&path.scatters[i].weight===material.albedo&&path.scatters[i].depth===i,'CPU oracle mirror history/material mismatch');throughput*=material.albedo;}
  else{
   assert(material.type==='diffuse'&&material.albedo===0,'CPU oracle terminal surface must absorb');
   const expected=throughput*material.emission;
   assert(path.value===expected&&path.contributions.length===(expected>0?1:0),'CPU oracle dyadic emitted power mismatch');
   if(expected>0)assert(path.contributions[0].type==='emission'&&path.contributions[0].value===expected&&path.contributions[0].weight===material.emission&&path.contributions[0].depth===mirrors,'CPU oracle contribution mismatch');
  }
 }
 if(fixture.oracle==='dyadic-mirror-panels-v1'){
  assert(fixture.expectedPowers.includes(path.value)&&mirrors<=2,'Structured oracle power/mirror domain');
  if(path.value===0)assert(mirrors===0&&materials[0]===4,'Structured black path must hit the baffle directly');
  else assert((mirrors===1&&materials[0]===1||mirrors===2&&materials[0]===0&&materials[1]===1)&&[2,3].includes(materials.at(-1)),'Structured mirror/emitter route mismatch');
 }else assert(path.value===fixture.expectedPower&&mirrors===fixture.mirrors,'CPU constant dyadic fixture mismatch');
}
function sameHistogram(actual,expected){return Object.keys(actual).length===Object.keys(expected).length&&Object.entries(expected).every(([key,value])=>actual[key]===value);}
function validateReference(reference,initial,final,config,fixture){
 const audit=reference.audit,pixels=config.size**2;
 assert(audit.complete&&audit.flags===0&&audit.paths.length===pixels*2,'complete independent CPU shadow is mandatory');
 assert(audit.pixelIndices.length===pixels&&audit.reference.length===pixels,'CPU shadow must cover the complete image');
 const paths=new Map(audit.paths.map(path=>[residentPathKey(path.key),path]));assert(paths.size===pixels*2,'duplicate CPU path ownership');
 for(let pixel=0;pixel<pixels;pixel++){
  assert(audit.pixelIndices[pixel]===pixel,'CPU analytic full image ownership mismatch');
  const rng=randomGenerator(initial[pixel]);let pixelSum=0;
  for(let sample=0;sample<2;sample++){
   const path=paths.get(`0:${pixel}:${sample}`),start=rng.getState(),words=[Math.floor(rng()*2**32),Math.floor(rng()*2**32)];
   validateDyadicPath(path,fixture);pixelSum+=path.value;
   assert(path.rng.start===start&&path.rng.count===2&&path.rng.words.length===2&&path.rng.words.every((word,i)=>word===words[i])&&path.rng.final===rng.getState(),'CPU canonical RNG mismatch');
  }
  assert(audit.reference[pixel]===pixelSum/2,'CPU image is not its exact two-sample dyadic mean');
  assert(final[pixel]===rng.getState(),'CPU complete RNG prefix mismatch');
 }
 const pathPowerHistogram=histogram(audit.paths.map(p=>p.value)),mirrorCountHistogram=histogram(audit.paths.map(p=>p.scatters.length));
 if(fixture.oracle==='dyadic-mirror-panels-v1'&&config.size===32){
  assert(sameHistogram(pathPowerHistogram,{'0':181,'0.75':511,'1.5':424,'3':423,'6':509}),'Frozen structured 32-square path-power histogram changed');
  assert(sameHistogram(mirrorCountHistogram,{'0':181,'1':933,'2':934}),'Frozen structured 32-square mirror histogram changed');
 }
 return {pathPowerHistogram,mirrorCountHistogram};
}
function comparisonFor(reference,admitted,explicitKeys,numericKeys,config){
 const image=values=>({width:config.size,height:config.size,values:Array.from(values)}),cpu=image(reference.audit.reference),gpuAssisted=image(admitted.image);
 const absoluteDifference=image(cpu.values.map((value,i)=>Math.abs(value-gpuAssisted.values[i]))),replayMask=image(new Uint8Array(config.size**2));
 const recoveredKeys=new Set([...explicitKeys,...numericKeys]),byMirrorCount={};
 for(const path of reference.audit.paths){
  const key=residentPathKey(path.key),mirrors=String(path.scatters.length),recovered=recoveredKeys.has(key);
  const stats=byMirrorCount[mirrors]??={paths:0,retainedPaths:0,explicitReplayPaths:0,auditNumericalRecoveryPaths:0,recoveredPaths:0};
  stats.paths++;stats.retainedPaths+=!recovered;stats.explicitReplayPaths+=explicitKeys.has(key);stats.auditNumericalRecoveryPaths+=numericKeys.has(key);stats.recoveredPaths+=recovered;
  if(recovered)replayMask.values[path.key.pixel]++;
 }
 assert(replayMask.values.every(value=>value>=0&&value<=2)&&replayMask.values.reduce((sum,value)=>sum+value,0)===recoveredKeys.size,'Comparison replay mask must preserve unique sample ownership');
 return {comparison:{cpu,gpuAssisted,absoluteDifference,replayMask},diagnostics:{maxAbsoluteDifference:Math.max(...absoluteDifference.values),replayMaskHistogram:{0:0,1:0,2:0,...histogram(replayMask.values)},byMirrorCount}};
}
function validateCandidate(candidate,reference,initial,input,final,config,fixture){
 const pixels=config.size**2;
 assert(candidate?.candidate===true&&candidate.admitted===false,'runner must return an unadmitted candidate');
 assert(candidate.image===candidate.audit?.reference,'image must be the audited owned buffer');
 assert(candidate.recovery?.mode==='selective'&&candidate.recovery.continuationGuard===config.guard&&candidate.recovery.axisSubmission==='round'&&candidate.recovery.laneChunkSize===config.laneChunkSize&&candidate.recovery.completeCPUPathReplay===true,'fixed guarded selective transport required');
 assert(input.length===initial.length&&input.every((word,i)=>word===initial[i]),'runner changed uncommitted input RNG');
 const expected=new Map(reference.audit.paths.map(p=>[residentPathKey(p.key),p]));
 assert(candidate.audit?.complete&&candidate.audit.flags===0&&candidate.audit.paths.length===pixels*2,'candidate omitted complete paths');
 const keys=new Set(candidate.audit.paths.map(p=>residentPathKey(p.key)));assert(keys.size===pixels*2,'duplicate candidate path ownership');
 const replay=candidate.recovery.keys;
 assert(Array.isArray(replay)&&replay.length===candidate.recovery.paths&&new Set(replay.map(p=>residentPathKey(p.key))).size===replay.length&&replay.every(p=>expected.has(residentPathKey(p.key))&&Array.isArray(p.reasons)&&p.reasons.length),'invalid explicit replay ownership');
 const replayKeys=new Set(replay.map(p=>residentPathKey(p.key)));
 for(const path of candidate.audit.paths){const key=residentPathKey(path.key),cpu=expected.get(key);assert(cpu&&rngEqual(path.rng,cpu.rng),'candidate canonical RNG/owner mismatch');assert((path.sampleValue??path.value)===cpu.value&&path.contributions.length===cpu.contributions.length&&path.contributions.every((c,i)=>c.type===cpu.contributions[i].type&&c.value===cpu.contributions[i].value&&c.weight===cpu.contributions[i].weight),'candidate analytic power mismatch');assert(path.scatters.length===cpu.scatters.length&&path.scatters.every((scatter,i)=>scatter.type==='mirror'&&scatter.weight===cpu.scatters[i].weight&&scatter.depth===cpu.scatters[i].depth)&&path.vertices.length===cpu.vertices.length&&path.vertices.every((v,i)=>v.triangleId===cpu.vertices[i].triangleId)&&path.termination.reason===cpu.termination.reason&&path.termination.depth===cpu.termination.depth,'candidate mirror/terminal history mismatch');if(!replayKeys.has(key))assert(path.intervalGuard?.unitProvenance===true&&path.intervalGuard?.anchorProvenance===true&&path.intervalGuard?.cameraProvenance===true&&path.intervalGuard.queries>=cpu.vertices.length,'retained path lacks camera guard evidence');}
 assert(candidate.pixelIndices?.length===pixels&&candidate.image?.length===pixels&&candidate.finalRngStates?.length===pixels,'candidate incomplete image or RNG prefix');
 for(let pixel=0;pixel<pixels;pixel++)assert(candidate.pixelIndices[pixel]===pixel&&candidate.image[pixel]===reference.audit.reference[pixel]&&candidate.finalRngStates[pixel]===final[pixel],'candidate image/RNG ownership mismatch');
}

/** Test-only dependency injection stays outside the serializable request. Never
 * pass a CPU reference to runnerFactory: it receives only immutable scene,
 * camera, fixed options and genuine device/setup controls.
 */
export async function executeResidentHardwareTest(request={}, {signal,progress=()=>{},gpu=globalThis.navigator?.gpu,runnerFactory=createPLTResidentRunner,shaderSource,totalTimeoutMs=RESIDENT_HARDWARE_TEST_MAX_TOTAL_MS}={}){
 const config=validateResidentHardwareTestRequest(request);
 if(!Number.isFinite(totalTimeoutMs)||totalTimeoutMs<1||totalTimeoutMs>RESIDENT_HARDWARE_TEST_MAX_TOTAL_MS)throw new RangeError('Total timeout must be between 1 and 240000 ms');
 if(typeof runnerFactory!=='function'||typeof progress!=='function')throw new TypeError('Invalid runtime callbacks');
 const started=now(),controller=new AbortController();let runner=null,timer,removeAbort,stage='cpu-shadow',stopped=false;
 const stop=reason=>{if(!controller.signal.aborted)controller.abort(reason);try{runner?.destroy();}catch{}};
 const check=()=>{if(signal?.aborted)stop(signal.reason?.name==='TimeoutError'?signal.reason:abortError());if(now()-started>=totalTimeoutMs)stop(timeoutError());if(controller.signal.aborted)throw controller.signal.reason??abortError();};
 const emit=(next,message,details={})=>{check();stage=next;try{progress({stage:next,message,elapsedMs:now()-started,...details});}catch{}check();};
 const interrupted=new Promise((_,reject)=>{controller.signal.addEventListener('abort',()=>reject(controller.signal.reason??abortError()),{once:true});});interrupted.catch(()=>{});
 const wait=promise=>Promise.race([promise,interrupted]);
 const onAbort=()=>stop(signal.reason?.name==='TimeoutError'?signal.reason:abortError());
 if(signal){signal.addEventListener('abort',onAbort,{once:true});removeAbort=()=>signal.removeEventListener('abort',onAbort);}
 timer=setTimeout(()=>stop(timeoutError()),totalTimeoutMs);
 const gpuState={apiAvailable:!!gpu,adapterAvailable:false,deviceCreated:false,executionAttempted:false,executed:false,fullRenderCompleted:false,candidateCompleted:false,candidateAdmitted:false,computeSubmissions:0,completedComputeSubmissions:0,adapter:adapterSummary(null),deviceLost:false,injectedRunner:runnerFactory!==createPLTResidentRunner};
 const compilation={apiWallMs:0,moduleCount:0,pipelineCount:0};
 const timings={coldSetupMs:0,compilationMs:null,cpuFullShadowMs:0,rendererIncludingReadbackReplayMs:null,admissionMs:null,endToEndMs:0};
 const report={schemaVersion:RESIDENT_HARDWARE_TEST_SCHEMA_VERSION,build:{testRuntime:RESIDENT_HARDWARE_TEST_BUILD_ID,resident:PLT_RESIDENT_BUILD_ID,shaderFingerprintFNV1a:null},requested:config,status:'cpu-only-fallback',gpu:gpuState,timings,environment:environmentSummary(),compilation:{...compilation,apiWallMs:null,measurement:'Wall time in createShaderModule/getCompilationInfo/createComputePipelineAsync, nested within coldSetupMs; not isolated GPU compiler time'},cpuFullShadow:{required:true,complete:false,paths:0,pixels:0},validation:{passed:false,completeImage:false,exactRng:false,expectedPower:null,admissionPassed:false,tolerance:{...RESIDENT_AUDIT_TOLERANCES}},recovery:{explicitReplayPaths:null,auditNumericalRecoveryPaths:null,totalRecoveredPaths:null,recoveryFraction:null,retainedPaths:null,explicitReplayKeys:null,auditNumericalRecoveryKeys:null,explicitReplayReasons:{},auditNumericalRecoveryReasons:{}},image:null,comparison:{cpu:null,gpuAssisted:null,absoluteDifference:null,replayMask:null},diagnostics:{maxAbsoluteDifference:null,replayMaskHistogram:null,pathPowerHistogram:null,mirrorCountHistogram:null,byMirrorCount:null},failure:null,limitations:['Single cold, complete-image correctness test with mandatory full CPU shadow.','Renderer wall time includes GPU work, host scheduling, readbacks and selective CPU replay; no timestamp-query GPU-only timing.','No warmed paired benchmark or GPU-versus-CPU speedup claim.','WebGPU may use software; adapter identity is reported when exposed. A physical GPU is not established by this test.','No wave, spectral, custom-scene or unaudited fast mode.'],physicalGPUValidated:false};
 try{
  check();const fixture=createResidentHardwareTestFixture(config.preset),options={width:config.size,height:config.size,samplesPerPixel:2,maxDepth:6,seed:7391,wave:false,nee:false,spectralMode:'monochrome',wavelengthMm:.025,boundary:'unpolarized'};
  const initial=createPLTRandomStates(options),cpuRng=initial.slice();emit('cpu-shadow','Building the mandatory complete CPU shadow');
  const cpuStart=now(),reference=await wait(buildResidentReferenceBatch(fixture.scene,fixture.camera,options,{tile:config.tile,sampleStart:0,samplesPerPixel:2,rngStates:cpuRng,signal:controller.signal,check,yieldEvery:16}));check();
  const oracleDiagnostics=validateReference(reference,initial,cpuRng,config,fixture);timings.cpuFullShadowMs=now()-cpuStart;
  report.cpuFullShadow={required:true,complete:true,paths:reference.audit.paths.length,pixels:config.size**2,analyticPower:fixture.expectedPower,analyticOracle:fixture.oracle??'constant-dyadic-mirror-v1',exactCanonicalRng:true};
  report.validation={...report.validation,passed:true,completeImage:true,exactRng:true,expectedPower:fixture.expectedPower,analyticOracle:fixture.oracle??'constant-dyadic-mirror-v1',analyticOraclePassed:true,initialRngFingerprintFNV1a:residentHashWords(initial).toString(16).padStart(8,'0'),finalRngFingerprintFNV1a:residentHashWords(cpuRng).toString(16).padStart(8,'0'),validatedPixels:config.size**2,validatedPaths:reference.audit.paths.length,imageSource:'cpu-full-shadow'};
  report.image={width:config.size,height:config.size,values:Array.from(reference.audit.reference),source:'cpu-full-shadow',validated:true};
  report.comparison.cpu={width:config.size,height:config.size,values:Array.from(reference.audit.reference)};report.diagnostics={...report.diagnostics,...oracleDiagnostics};
  let candidate=null,admitted=null;
  try{
   emit('setup','Setting up a fresh guarded resident GPU runner');const setupStart=now();
   try{
    const pending=Promise.resolve(runnerFactory(fixture.scene,fixture.camera,options,{gpu:observeGPU(gpu,gpuState,compilation,emit,check),shaderSource,signal:controller.signal,timeoutMs:Math.min(120000,totalTimeoutMs),recoveryMode:'selective',continuationGuard:'axis-mirror-camera',onLost:()=>{gpuState.deviceLost=true;},onSetupProgress:step=>emit('pipeline-compilation',`Compiling ${step.entryPoint??step.stageName??'pipeline'}`,{entryPoint:step.entryPoint??null})}));
    pending.then(value=>{if(stopped||controller.signal.aborted)try{value?.destroy();}catch{}},()=>{});
    runner=await wait(pending);check();
   }finally{timings.coldSetupMs=now()-setupStart;timings.compilationMs=compilation.moduleCount?compilation.apiWallMs:null;report.compilation={...report.compilation,...compilation,apiWallMs:timings.compilationMs};}
   report.build.resident=runner.diagnostic?.buildId??PLT_RESIDENT_BUILD_ID;report.build.shaderFingerprintFNV1a=runner.diagnostic?.shaderFingerprintFNV1a??null;
   emit('renderer','Rendering one complete tile with guarded GPU paths and selective CPU replay');gpuState.executionAttempted=true;
   const renderStart=now(),input=initial.slice();
   try{candidate=await wait(runner.run({tile:config.tile,sampleStart:0,samplesPerPixel:2,wave:false,rngStates:input,signal:controller.signal,guard:'axis-mirror-camera',axisSubmission:'round',laneChunkSize:config.laneChunkSize,onProgress:p=>emit('renderer',`Resident round ${p.round}; ${p.active} active lanes`,{round:p.round,active:p.active})}));check();gpuState.candidateCompleted=true;gpuState.fullRenderCompleted=gpuState.completedComputeSubmissions>0&&!gpuState.injectedRunner;}
   finally{timings.rendererIncludingReadbackReplayMs=now()-renderStart;gpuState.executed=gpuState.completedComputeSubmissions>0&&!gpuState.injectedRunner;}
   emit('admission','Checking complete path ownership, canonical RNG and the unchanged full-shadow admission');const admissionStart=now();
   try{
    validateCandidate(candidate,reference,initial,input,cpuRng,config,fixture);admitted=admitResidentWithRecovery(reference,candidate);check();
    assert(admitted.admitted&&admitted.validation.passed&&admitted.image===admitted.audit.reference,'strict final admission failed');
    assert(admitted.image.length===config.size**2&&admitted.image.every((value,i)=>value===reference.audit.reference[i]),'admitted full image failed analytic power');
   }finally{timings.admissionMs=now()-admissionStart;}
   gpuState.candidateAdmitted=true;
   const explicitKeys=new Set(candidate.recovery.keys.map(p=>residentPathKey(p.key))),numericKeys=new Set(admitted.recovery.numericGeometry.keys),recoveredKeys=new Set([...explicitKeys,...numericKeys]);
   assert(recoveredKeys.size<=reference.audit.paths.length&&[...recoveredKeys].every(key=>reference.audit.paths.some(path=>residentPathKey(path.key)===key)),'Recovery keys must belong to the exact complete sample domain');
   report.recovery={explicitReplayKeys:[...explicitKeys],auditNumericalRecoveryKeys:[...numericKeys],explicitReplayPaths:explicitKeys.size,auditNumericalRecoveryPaths:numericKeys.size,totalRecoveredPaths:recoveredKeys.size,recoveryFraction:recoveredKeys.size/reference.audit.paths.length,retainedPaths:reference.audit.paths.length-recoveredKeys.size,explicitReplayReasons:histogram(candidate.recovery.keys.flatMap(k=>k.reasons)),auditNumericalRecoveryReasons:histogram(admitted.recovery.numericGeometry.reasons.map(r=>r.reason))};
   report.validation={...report.validation,admissionPassed:true,rawAuditPassed:admitted.recovery.numericGeometry.initialAudit.passed,admissionMismatchCount:admitted.validation.mismatchCount,initialAuditMismatchCount:admitted.recovery.numericGeometry.initialMismatchCount,checked:{...admitted.validation.checked},maxError:{...admitted.validation.maxError}};
   report.renderer={rounds:candidate.rounds,dispatches:candidate.dispatches,timings:Object.fromEntries(Object.entries(admitted.timings??{}).filter(([,value])=>typeof value==='boolean'||Number.isFinite(value)))};
   if(gpuState.executed){report.status='gpu-validated';report.image={...report.image,values:Array.from(admitted.image),source:'gpu-with-selective-recovery'};report.validation.imageSource=report.image.source;const compared=comparisonFor(reference,admitted,explicitKeys,numericKeys,config);report.comparison=compared.comparison;report.diagnostics={...report.diagnostics,...compared.diagnostics};}
   else report.failure={stage:'execution-evidence',name:'UnverifiedGPUExecution',code:null,message:'Candidate admitted, but no genuine completed GPU compute submission was observed; returning the CPU shadow.',mismatchCount:null,diagnosticFlags:null};
  }catch(error){check();report.failure=failureSummary(error,stage);gpuState.executed=gpuState.completedComputeSubmissions>0&&!gpuState.injectedRunner;}
  check();emit('complete',report.status==='gpu-validated'?'Complete GPU image passed full-shadow validation':'CPU-only fallback; see GPU failure details');
  return report;
 }finally{
  stopped=true;clearTimeout(timer);removeAbort?.();try{runner?.destroy();}catch{}
  timings.endToEndMs=now()-started;
 }
}
