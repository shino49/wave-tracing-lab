/** Public opt-in wrapper checks. Native execution is separately --execute.
 * Mocks verify host flow only; genuine native runs remain full-shadow audited.
 */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import crypto from 'node:crypto';
import {create,globals} from 'webgpu';
import {fixtures,reuseNativeDevice} from './plt-resident-tight-transport.mjs';
import {createPLTRandomStates,randomGenerator} from '../dist/plt-core.js';
import {buildResidentReferenceBatch} from '../dist/plt-resident-reference.js';
import {executeResidentHardwareTest,validateResidentHardwareTestRequest,RESIDENT_HARDWARE_TEST_PRESETS} from '../dist/plt-resident-test-runtime.js';
import {createResidentHardwareTestFixture} from '../dist/plt-resident-test-presets.js';
import {installResidentHardwareTestWorker} from '../dist/plt-resident-test-worker.js';
const delay=ms=>new Promise(resolve=>setTimeout(resolve,ms));
const report={allPass:false,actualGPUExecuted:false,physicalGPUValidated:false,hostChecks:[],nativeCases:[]};
const save=()=>fs.writeFileSync(new URL('../research/plt-resident-test-runtime-v25-'+(process.argv.includes('--execute')?'':'host-')+'validation.json',import.meta.url),JSON.stringify(report,null,2)+'\n');
async function check(name,action){await action();report.hostChecks.push(name);}
await check('Exact preset geometry, camera, materials and analytic/RNG corpus at every supported size',async()=>{
 for(const preset of RESIDENT_HARDWARE_TEST_PRESETS.filter(p=>p.expectedPower!==null)){
  const f=createResidentHardwareTestFixture(preset.id),baseline=fixtures.find(x=>x.name===preset.id);
  assert.deepEqual(f.scene,baseline.scene);assert.deepEqual(f.camera,baseline.camera);assert.equal(f.expectedPower,baseline.expected);
  for(const size of [8,16,32]){
   const options={width:size,height:size,samplesPerPixel:2,maxDepth:6,seed:7391,wave:false,nee:false};
   const initial=createPLTRandomStates(options),rng=initial.slice(),reference=await buildResidentReferenceBatch(f.scene,f.camera,options,{tile:{x:0,y:0,width:size,height:size},samplesPerPixel:2,sampleStart:0,rngStates:rng});
   assert.equal(reference.audit.paths.length,size*size*2);assert(reference.audit.reference.every(x=>x===f.expectedPower));
   for(let pixel=0;pixel<initial.length;pixel++){const r=randomGenerator(initial[pixel]);for(let sample=0;sample<2;sample++){const p=reference.audit.paths[pixel*2+sample],start=r.getState(),words=[Math.floor(r()*2**32),Math.floor(r()*2**32)];assert.deepEqual(p.key,{wave:false,pixel,sample});assert.equal(p.value,f.expectedPower);assert.equal(p.scatters.filter(s=>s.type==='mirror').length,f.mirrors);assert.equal(p.rng.start,start);assert.deepEqual(p.rng.words,words);assert.equal(p.rng.final,r.getState());}assert.equal(rng[pixel],r.getState());}
  }
 }
});
await check('Strict bounded request schema rejects custom, wave and spectral inputs before runner work',async()=>{
 assert.equal(validateResidentHardwareTestRequest({}).size,16);assert.equal(validateResidentHardwareTestRequest({}).laneChunkSize,128);
 const bad=[null,[],7,'x',{wave:true},{spectral:true},{spectralMode:'visible'},{scene:{}},{camera:{}},{materials:[]},{preset:'one-mirror-emission'},{size:64},{size:'8'},{laneChunkSize:64},{seed:1},{nee:true},{maxDepth:4},{width:8},{totalTimeoutMs:999999}];
 for(const request of bad){let called=false;await assert.rejects(executeResidentHardwareTest(request,{runnerFactory:()=>{called=true;}}));assert.equal(called,false);}
 await assert.rejects(executeResidentHardwareTest({},{totalTimeoutMs:240001}));
});
await check('No GPU, no adapter, device failure and compilation failure retain only complete CPU image',async()=>{
 const cases=[undefined,{requestAdapter:async()=>null},{requestAdapter:async()=>({info:{vendor:'mock'},requestDevice:async()=>{throw new Error('device denied');}})}];
 for(const gpu of cases){const r=await executeResidentHardwareTest({size:8},{gpu});assert.equal(r.status,'cpu-only-fallback');assert.equal(r.gpu.executed,false);assert.equal(r.gpu.candidateAdmitted,false);assert.equal(r.validation.admissionPassed,false);assert(r.validation.completeImage&&r.validation.exactRng&&r.image.values.every(x=>x===3));assert(r.failure);assert.equal(r.timings.compilationMs,null);}
 let destroyed=0;
 const gpu={requestAdapter:async()=>({info:{vendor:'mock'},requestDevice:async()=>({lost:new Promise(()=>{}),addEventListener(){},removeEventListener(){},queue:{},destroy(){destroyed++;},pushErrorScope(){},popErrorScope:async()=>null,createShaderModule:()=>({getCompilationInfo:async()=>({messages:[{type:'error',lineNum:1,linePos:1,message:'compile failed'}]})})})})};
 const r=await executeResidentHardwareTest({size:8},{gpu,shaderSource:'invalid'});assert.equal(r.status,'cpu-only-fallback');assert.equal(r.gpu.deviceCreated,true);assert.equal(r.gpu.executed,false);assert.match(r.failure.message,/compile failed/);assert(r.timings.compilationMs>=0);assert.equal(destroyed,1);
});
function factory({mutate,runWait=0,deviceLoss=false}={},state={destroyed:0}){
 return async(scene,camera,options,controls)=>({diagnostic:{buildId:'host-test-only'},destroy(){state.destroyed++;},async run(batch){
  if(runWait)await delay(runWait);
  if(deviceLoss){controls.onLost('mock loss');throw new Error('mock device lost');}
  const rng=batch.rngStates.slice(),job=await buildResidentReferenceBatch(scene,camera,options,{...batch,rngStates:rng});
  const audit=job.audit;for(const p of audit.paths)p.intervalGuard={unitProvenance:true,anchorProvenance:true,cameraProvenance:true,queries:p.scatters.length+1};
  const candidate={candidate:true,admitted:false,audit,image:audit.reference,pixelIndices:audit.pixelIndices,finalRngStates:rng,recovery:{mode:'selective',continuationGuard:'axis-mirror-camera',axisSubmission:'round',laneChunkSize:batch.laneChunkSize,paths:0,keys:[],completeCPUPathReplay:true},timings:{},rounds:1,dispatches:1};
  mutate?.(candidate,batch);return candidate;
 }});
}
await check('Injected success is admitted but cannot claim genuine GPU execution; no shadow is supplied to factory',async()=>{
 const state={destroyed:0};const r=await executeResidentHardwareTest({size:8},{runnerFactory:factory({},state)});
 assert.equal(r.gpu.candidateCompleted,true);assert.equal(r.gpu.candidateAdmitted,true);assert.equal(r.validation.admissionPassed,true);assert.equal(r.gpu.executed,false);assert.equal(r.gpu.fullRenderCompleted,false);assert.equal(r.status,'cpu-only-fallback');assert.equal(r.image.source,'cpu-full-shadow');assert.equal(r.recovery.totalRecoveredPaths,0);assert.equal(state.destroyed,1);
});
await check('Altered image, RNG input/output, ownership and audit are rejected',async()=>{
 for(const mutate of [c=>c.image[0]=4,c=>c.finalRngStates[0]++,c=>c.pixelIndices[0]=1,c=>c.audit.paths[0].rng.words[0]++,c=>c.audit.paths.pop(),(c,b)=>b.rngStates[0]++,c=>{c.recovery.paths=2;c.recovery.keys=[{key:c.audit.paths[0].key,reasons:['x']},{key:c.audit.paths[0].key,reasons:['x']}];},c=>{c.audit.paths[0].scatters.pop();},c=>{c.audit.paths[0].intervalGuard.cameraProvenance=false;}]){
  const r=await executeResidentHardwareTest({size:8},{runnerFactory:factory({mutate})});assert.equal(r.gpu.candidateAdmitted,false);assert.equal(r.validation.admissionPassed,false);assert.equal(r.status,'cpu-only-fallback');assert(r.image.values.every(x=>x===3));assert(r.failure);
 }
});
await check('Device loss fallback is separate from admission',async()=>{const r=await executeResidentHardwareTest({size:8},{runnerFactory:factory({deviceLoss:true})});assert.equal(r.gpu.deviceLost,true);assert.equal(r.gpu.candidateAdmitted,false);assert.match(r.failure.message,/lost/);});
await check('Pre-abort, setup abort, late runner, run abort and total deadline withhold results and destroy resources',async()=>{
 const pre=new AbortController();pre.abort();await assert.rejects(executeResidentHardwareTest({size:8},{signal:pre.signal}),{name:'AbortError'});
 const controller=new AbortController(),late={destroyed:0};let began;
 const pending=executeResidentHardwareTest({size:8},{signal:controller.signal,runnerFactory:async()=>{began=true;await delay(25);return {destroy(){late.destroyed++;}};}});while(!began)await delay(1);controller.abort();await assert.rejects(pending,{name:'AbortError'});await delay(30);assert.equal(late.destroyed,1);
 const runController=new AbortController(),running={destroyed:0};let runStarted=false;
 const run=executeResidentHardwareTest({size:8},{signal:runController.signal,runnerFactory:factory({runWait:30},running),progress:p=>{if(p.stage==='renderer'){runStarted=true;setTimeout(()=>runController.abort(),1);}}});await assert.rejects(run,{name:'AbortError'});assert(runStarted&&running.destroyed>=1);await delay(35);
 const deadline={destroyed:0};await assert.rejects(executeResidentHardwareTest({size:8},{totalTimeoutMs:20,runnerFactory:async()=>{await delay(35);return {destroy(){deadline.destroyed++;}};}}),{name:'TimeoutError'});await delay(40);assert.equal(deadline.destroyed,1);
});
await check('Worker is inert on load, rejects malformed envelopes, suppresses stale/aborted runs and supports repeated runs',async()=>{
 const posts=[],calls=[],scope={postMessage:m=>posts.push(m)};
 const dispose=installResidentHardwareTestWorker(scope,async(request,{signal,progress})=>{calls.push(request);progress({stage:'test'});await delay(request.size===8?25:2);return {aborted:signal.aborted,size:request.size};});
 assert.equal(calls.length,0);await scope.onmessage({data:{}});await scope.onmessage({data:{id:1,type:'mystery',request:{size:8}}});assert.equal(calls.length,0);
 const first=scope.onmessage({data:{id:2,request:{size:8}}});const second=scope.onmessage({data:{id:3,request:{size:16}}});await Promise.all([first,second]);assert.equal(posts.filter(p=>p.type==='result').length,1);assert.equal(posts.find(p=>p.type==='result').id,3);
 const cancelled=scope.onmessage({data:{id:4,request:{size:8}}});await scope.onmessage({data:{type:'cancel',id:4}});await cancelled;assert(!posts.some(p=>p.id===4&&p.type==='result'));
 await scope.onmessage({data:{id:5,request:{size:16}}});assert(posts.some(p=>p.id===5&&p.type==='result'));dispose();assert.equal(scope.onmessage,null);
});
await check('Structured scene fixed dyadic oracle, frozen histograms, schema2 CPU-only comparison at every size',async()=>{
 const expected={8:{power:{0:7,'.75':34,'1.5':27,3:27,6:33},mirrors:{0:7,1:60,2:61}},16:{power:{0:46,'.75':127,'1.5':105,3:107,6:127},mirrors:{0:46,1:232,2:234}},32:{power:{0:181,'.75':511,'1.5':424,3:423,6:509},mirrors:{0:181,1:933,2:934}}};
 const f=createResidentHardwareTestFixture('structured-mirror-panels');assert.equal(f.scene.triangles.length,14);assert.equal(f.scene.materials.length,5);assert.deepEqual(f.camera,{position:[0,0,0],target:[4,1,4],up:[0,1,0],fov:10});
 for(const size of [8,16,32]){
  const r=await executeResidentHardwareTest({preset:f.id,size},{gpu:null});const power={...expected[size].power};power['0.75']=power['.75'];delete power['.75'];
  assert.equal(r.schemaVersion,2);assert.equal(r.validation.analyticOraclePassed,true);assert.equal(r.validation.expectedPower,null);assert.deepEqual(r.diagnostics.pathPowerHistogram,power);assert.deepEqual(r.diagnostics.mirrorCountHistogram,expected[size].mirrors);
  assert.deepEqual(r.image.values,r.comparison.cpu.values);assert.equal(r.comparison.gpuAssisted,null);assert.equal(r.comparison.absoluteDifference,null);assert.equal(r.comparison.replayMask,null);assert.equal(r.diagnostics.maxAbsoluteDifference,null);assert.equal(r.diagnostics.byMirrorCount,null);assert(new Set(r.image.values).size>4);
  const injected=await executeResidentHardwareTest({preset:f.id,size},{runnerFactory:factory()});assert.equal(injected.validation.admissionPassed,true);assert.equal(injected.comparison.gpuAssisted,null);assert.equal(injected.comparison.replayMask,null);
 }
});
function checkComparison(r){
 const {comparison:c,recovery:re,diagnostics:d}=r,n=r.requested.size**2;
 for(const image of Object.values(c)){assert.equal(image.values.length,n);assert(image.values.every(Number.isFinite));}
 assert.deepEqual(r.image.values,c.gpuAssisted.values);assert.deepEqual(c.cpu.values,c.gpuAssisted.values);
 const keys=new Set([...re.explicitReplayKeys,...re.auditNumericalRecoveryKeys]),mask=new Array(n).fill(0);
 for(const key of keys){const [wave,pixel,sample]=key.split(':').map(Number);assert.equal(wave,0);assert(pixel>=0&&pixel<n&&[0,1].includes(sample));mask[pixel]++;}
 assert.deepEqual(c.replayMask.values,mask);assert.equal(keys.size,re.totalRecoveredPaths);assert(c.absoluteDifference.values.every(x=>x===0));assert.equal(d.maxAbsoluteDifference,0);
 const histogram={0:0,1:0,2:0};for(const value of mask)histogram[value]++;assert.deepEqual(d.replayMaskHistogram,histogram);
 assert.equal(Object.values(d.byMirrorCount).reduce((sum,x)=>sum+x.paths,0),n*2);assert.equal(Object.values(d.byMirrorCount).reduce((sum,x)=>sum+x.recoveredPaths,0),keys.size);assert.equal(Object.values(d.byMirrorCount).reduce((sum,x)=>sum+x.retainedPaths,0),re.retainedPaths);
}
if(process.argv.includes('--execute')){
 Object.assign(globalThis,globals);const backend=process.env.WEBGPU_EXECUTION_BACKEND??'vulkan';
 const native=create([`backend=${backend}`]),cache=reuseNativeDevice(native);
 const shaderFiles=['utd-functions.wgsl','plt-resident-geometry.wgsl','plt-resident-interval.wgsl','plt-resident-axis-guard.wgsl','plt-resident-rng.wgsl','plt-resident-fsd.wgsl','plt-resident-walker.wgsl'];
 const shaderSource=shaderFiles.map(f=>fs.readFileSync(new URL('../dist/'+f,import.meta.url),'utf8')).join('\n');
 report.nativeDeviceReuse=cache.stats;report.nativeAdapterIsSoftwareOnly=null;report.nativeTimingClaims=false;report.nativeEnvironment={backend,configuredICD:process.env.VK_ICD_FILENAMES??null};
 try{
  for(const request of [{preset:'structured-mirror-panels',size:32,laneChunkSize:128},{preset:'structured-mirror-panels',size:32,laneChunkSize:32},{preset:'structured-mirror-panels',size:8,laneChunkSize:128},{preset:'structured-mirror-panels',size:16,laneChunkSize:128},...RESIDENT_HARDWARE_TEST_PRESETS.filter(p=>p.expectedPower!==null).map(p=>({preset:p.id,size:16,laneChunkSize:128})),{preset:RESIDENT_HARDWARE_TEST_PRESETS[0].id,size:16,laneChunkSize:32},{preset:RESIDENT_HARDWARE_TEST_PRESETS[0].id,size:32,laneChunkSize:128}]){
   console.log('Native request',JSON.stringify(request));
   const r=await executeResidentHardwareTest(request,{gpu:cache.gpu,shaderSource,progress:p=>{if(p.stage==='pipeline-compilation'||p.stage==='complete')console.log(JSON.stringify(p));}});
   r.testHarness={sharedNativeDeviceAndPipelineCache:true,freshNativeDeviceAndPipelines:report.nativeCases.length===0,performanceEvidence:false};report.nativeCases.push(r);report.nativeAdapterIsSoftwareOnly=report.nativeCases.every(c=>c.gpu.adapter.isFallbackAdapter===true&&/swiftshader|llvmpipe|software/i.test([c.gpu.adapter.architecture,c.gpu.adapter.description].join(' ')));save();assert.equal(r.status,'gpu-validated',JSON.stringify(r.failure));assert(r.gpu.executed&&r.gpu.fullRenderCompleted&&r.gpu.candidateAdmitted);assert(r.validation.admissionPassed&&r.validation.completeImage&&r.validation.exactRng);assert.equal(r.recovery.auditNumericalRecoveryPaths,0);report.actualGPUExecuted=true;checkComparison(r);
  }
  for(const [a,b] of [[0,1],[4,7]]){assert.deepEqual(report.nativeCases[a].image,report.nativeCases[b].image,'chunk-size complete image parity');assert.deepEqual(report.nativeCases[a].comparison,report.nativeCases[b].comparison,'chunk-size complete comparison/mask parity');assert.deepEqual(report.nativeCases[a].recovery,report.nativeCases[b].recovery,'chunk-size exact recovered path parity');for(const key of ['initialRngFingerprintFNV1a','finalRngFingerprintFNV1a'])assert.equal(report.nativeCases[a].validation[key],report.nativeCases[b].validation[key]);}
 }finally{cache.destroy();save();}
}
report.sourceSHA256=Object.fromEntries(['plt-resident-test-runtime.js','plt-resident-test-worker.js','plt-resident-test-presets.js'].map(file=>[file,crypto.createHash('sha256').update(fs.readFileSync(new URL('../dist/'+file,import.meta.url))).digest('hex')]));
report.allPass=true;save();console.log(JSON.stringify({allPass:true,hostChecks:report.hostChecks.length,nativeCases:report.nativeCases.length,actualGPUExecuted:report.actualGPUExecuted}));
