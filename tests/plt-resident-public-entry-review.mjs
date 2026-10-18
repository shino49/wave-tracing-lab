/** Independent public wrapper review. CPU reference and injected failures only;
 * this is deliberately not native, browser or physical-hardware evidence.
 */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import crypto from 'node:crypto';
import {execFileSync} from 'node:child_process';
import {createPLTRandomStates} from '../dist/plt-core.js';
import {buildResidentReferenceBatch} from '../dist/plt-resident-reference.js';
import {createResidentHardwareTestFixture} from '../dist/plt-resident-test-presets.js';
import {executeResidentHardwareTest,validateResidentHardwareTestRequest,RESIDENT_HARDWARE_TEST_PRESETS} from '../dist/plt-resident-test-runtime.js';
import {installResidentHardwareTestWorker} from '../dist/plt-resident-test-worker.js';
import {fixtures} from './plt-resident-tight-transport.mjs';
import {Window} from 'happy-dom';
import {structuredOracle} from './plt-resident-structured-oracle.mjs';

const baseline='b42a330b7bb9ed824a810f2fe6ccc2dd23c65197',checks=[];
const delay=()=>new Promise(r=>setTimeout(r,0));
const deferred=()=>{let resolve,reject;const promise=new Promise((a,b)=>{resolve=a;reject=b;});return {promise,resolve,reject};};
async function test(name,fn){await fn();checks.push(name);console.log('PASS public independent',name);}
const request={preset:'two-mirrors-emission',size:8,laneChunkSize:128};
const prepared=new Map();
async function candidateFor(r=request){
 const c=validateResidentHardwareTestRequest(r),key=JSON.stringify(c);
 if(!prepared.has(key)){
  const f=createResidentHardwareTestFixture(c.preset),options={width:c.size,height:c.size,samplesPerPixel:2,maxDepth:6,seed:7391,wave:false,nee:false},rng=createPLTRandomStates(options);
  const reference=await buildResidentReferenceBatch(f.scene,f.camera,options,{tile:c.tile,sampleStart:0,samplesPerPixel:2,rngStates:rng});
  const audit=reference.audit;for(const p of audit.paths){p.sampleValue=p.value;p.intervalGuard={unitProvenance:true,anchorProvenance:true,cameraProvenance:true,queries:p.vertices.length};}
  prepared.set(key,{candidate:true,admitted:false,audit,image:audit.reference,pixelIndices:audit.pixelIndices,finalRngStates:rng,recovery:{mode:'selective',continuationGuard:'axis-mirror-camera',axisSubmission:'round',laneChunkSize:c.laneChunkSize,paths:0,keys:[],completeCPUPathReplay:true},rounds:1,dispatches:1,timings:{}});
 }
 return structuredClone(prepared.get(key));
}
async function executeInjected(mutate=()=>{},r=request){
 const candidate=await candidateFor(r),originalImage=Array.from(candidate.image);let destroyed=0,runs=0;
 const report=await executeResidentHardwareTest(r,{gpu:null,runnerFactory:async(scene,camera,options,controls)=>{
  assert.equal(options.wave,false);assert.equal(options.samplesPerPixel,2);assert.equal(controls.recoveryMode,'selective');assert.equal(controls.continuationGuard,'axis-mirror-camera');
  for(const value of [scene,camera,options,controls])assert(!('reference' in value)&&!('audit' in value));
  return {diagnostic:{},destroy(){destroyed++;},async run(input){runs++;mutate(candidate,input);return candidate;}};
 }});
 assert.equal(runs,1);assert(destroyed>=1);assert.equal(report.gpu.executed,false);assert.equal(report.physicalGPUValidated,false);assert.equal(report.status,'cpu-only-fallback');assert.equal(report.image.source,'cpu-full-shadow');
 assert(report.cpuFullShadow.complete&&report.validation.passed&&report.validation.exactRng);
 assert.deepEqual(report.image.values,originalImage);
 return report;
}

await test('stable production source and resident math remain byte-identical',()=>{
 const tracked=execFileSync('git',['ls-tree','-r','--name-only',baseline,'dist'],{encoding:'utf8'}).trim().split('\n');
 for(const file of tracked.filter(f=>f!=='dist/tracer.html')){
  const old=execFileSync('git',['show',`${baseline}:${file}`]);assert.deepEqual(fs.readFileSync(new URL('../'+file,import.meta.url)),old,file);
 }
 const tracer=fs.readFileSync(new URL('../dist/tracer.html',import.meta.url),'utf8'),old=execFileSync('git',['show',`${baseline}:dist/tracer.html`],{encoding:'utf8'});
 assert.equal(tracer.replace(/<a[^>]+href="resident-test\.html"[^>]*>[\s\S]*?<\/a>/g,''),old,'Stable tracer changes must be only the opt-in link');
});
await test('published fixtures exactly match frozen scene and camera values',()=>{
 for(const p of RESIDENT_HARDWARE_TEST_PRESETS.filter(p=>p.expectedPower!==null)){const actual=createResidentHardwareTestFixture(p.id),expected=fixtures.find(f=>f.name===p.id);assert(expected);assert.deepEqual(actual.scene,expected.scene);assert.deepEqual(actual.camera,expected.camera);assert.equal(actual.expectedPower,expected.expected);assert.equal(actual.mirrors,expected.mirrors);}
});
await test('unsupported requests reject before factory invocation',async()=>{
 let calls=0;
 for(const value of [null,[],3,'x',{wave:true},{spectral:true},{scene:{}},{camera:{}},{samplesPerPixel:3},{maxDepth:7},{seed:0},{sampleStart:1},{tile:{}},{size:'16'},{size:0},{size:64},{size:NaN},{size:Infinity},{laneChunkSize:'128'},{laneChunkSize:64},{laneChunkSize:0},{preset:'unknown'},{preset:'unsupported-tilted-mirror'}])await assert.rejects(executeResidentHardwareTest(value,{runnerFactory(){calls++;}}));
 assert.equal(calls,0);const c=validateResidentHardwareTestRequest();assert.equal(c.size,16);assert.equal(c.laneChunkSize,128);assert.equal(c.samplesPerPixel,2);assert.equal(c.wave,false);assert.equal(c.spectral,false);
});
await test('no WebGPU still returns analytic complete CPU shadow without hardware claim',async()=>{
 const r=await executeResidentHardwareTest(request,{gpu:null});assert.equal(r.status,'cpu-only-fallback');assert.equal(r.gpu.apiAvailable,false);assert.equal(r.gpu.executed,false);assert.equal(r.gpu.adapterAvailable,false);assert(r.failure);assert(r.image.values.every(v=>v===3));assert(r.validation.passed&&r.cpuFullShadow.complete);assert.equal(r.gpu.adapter.description,null);
});
await test('an admitted injected CPU tape cannot manufacture actual GPU execution',async()=>{
 const r=await executeInjected();assert.equal(r.validation.admissionPassed,true);assert.equal(r.recovery.auditNumericalRecoveryPaths,0);assert.equal(r.recovery.explicitReplayPaths,0);assert.equal(r.recovery.retainedPaths,128);assert.equal(r.failure.name,'UnverifiedGPUExecution');
});
await test('corrupt ownership, power, contribution, image, RNG and replay candidates retain only CPU output',async()=>{
 const mutants=[
  c=>{c.image=c.image.slice();},
  c=>{c.audit.paths[1].key={...c.audit.paths[0].key};},
  c=>{c.audit.paths[0].key.sample=2;},
  c=>{c.audit.paths[0].rng.words[0]^=1;},
  c=>{c.finalRngStates[0]^=1;},
  (c,input)=>{input.rngStates[0]^=1;},
  c=>{c.audit.paths[0].sampleValue=2;},
  c=>{c.audit.paths[0].contributions[0].value=2;},
  c=>{c.image[0]=NaN;},
  c=>{c.audit.paths.pop();},
  c=>{c.pixelIndices[0]=1;},
  c=>{c.recovery.keys=[{key:{...c.audit.paths[0].key},reasons:['fake']},{key:{...c.audit.paths[0].key},reasons:['fake']}];c.recovery.paths=2;},
  c=>{c.recovery.keys=[{key:{wave:false,pixel:999,sample:0},reasons:['fake']}];c.recovery.paths=1;},
  c=>{c.audit.paths[0].vertices[0].origin[0]+=1;},
 ];
 for(const mutate of mutants){const r=await executeInjected(mutate);assert.equal(r.validation.admissionPassed,false);assert(r.failure);}
});
await test('bounded geometry drift is separately counted and replay overlap cannot double-count',async()=>{
 const r=await executeInjected(c=>{c.audit.paths[0].vertices[0].origin[0]+=.0001;});assert.equal(r.validation.admissionPassed,true);assert.equal(r.recovery.auditNumericalRecoveryPaths,1);assert.equal(r.recovery.explicitReplayPaths,0);assert.equal(r.recovery.retainedPaths,127);
 const overlap=await executeInjected(c=>{const path=c.audit.paths[0];path.vertices[0].origin[0]+=.0001;c.recovery.keys=[{key:{...path.key},reasons:['axisIntervalUncertain']}];c.recovery.paths=1;});
 if(overlap.validation.admissionPassed)assert.equal(overlap.recovery.retainedPaths,127,'A path replaced twice is still one CPU-owned path');
});
await test('pre-abort never calls the runner and late setup is destroyed without result',async()=>{
 const pre=new AbortController();pre.abort();let calls=0;await assert.rejects(executeResidentHardwareTest(request,{signal:pre.signal,runnerFactory(){calls++;}}),{name:'AbortError'});assert.equal(calls,0);
 const entered=deferred(),held=deferred(),controller=new AbortController();let destroyed=0;
 const pending=executeResidentHardwareTest(request,{gpu:null,signal:controller.signal,runnerFactory(){entered.resolve();return held.promise;}});await entered.promise;controller.abort();await assert.rejects(pending,{name:'AbortError'});held.resolve({destroy(){destroyed++;}});await delay();assert.equal(destroyed,1);
});
await test('abort and total timeout cannot admit an in-flight candidate',async()=>{
 for(const timeout of [false,true]){
  const entered=deferred(),held=deferred(),controller=new AbortController();let destroyed=0;
  const pending=executeResidentHardwareTest(request,{gpu:null,signal:controller.signal,totalTimeoutMs:timeout?200:5000,runnerFactory:async()=>({diagnostic:{},destroy(){destroyed++;},run(){entered.resolve();return held.promise;}})});
  await entered.promise;if(!timeout)controller.abort();await assert.rejects(pending,{name:timeout?'TimeoutError':'AbortError'});held.resolve(await candidateFor());await delay();assert(destroyed>=1);
 }
});
await test('diagnostics and exception payloads remain bounded and do not include full path tapes',async()=>{
 const r=await executeResidentHardwareTest(request,{gpu:null,runnerFactory(){throw Object.assign(new Error('x'.repeat(1000000)),{huge:'x'.repeat(1000000),audit:{mismatchCount:123,paths:new Array(99999).fill('secret')}});}});
 assert(r.failure.message.length<=2000);assert.equal(r.failure.mismatchCount,123);assert(!JSON.stringify(r).includes('secret'));assert(JSON.stringify(r).length<25000);assert(!('paths' in r.image));
});
await test('structured schema2 CPU and injected results preserve independent oracle and withhold nonCPU views',async()=>{
 for(const size of [8,16,32]){
  const r=await executeResidentHardwareTest({preset:'structured-mirror-panels',size,laneChunkSize:128},{gpu:null}),oracle=structuredOracle(size);
  assert.equal(r.schemaVersion,2);assert.deepEqual(r.image.values,oracle.image);assert.deepEqual(r.comparison.cpu.values,oracle.image);assert.equal(r.validation.expectedPower,null);assert.equal(r.validation.analyticOraclePassed,true);assert.equal(r.validation.analyticOracle,'dyadic-mirror-panels-v1');assert.deepEqual(r.diagnostics.pathPowerHistogram,oracle.powers);assert.deepEqual(r.diagnostics.mirrorCountHistogram,oracle.mirrors);
  for(const field of ['gpuAssisted','absoluteDifference','replayMask'])assert.equal(r.comparison[field],null);assert.equal(r.diagnostics.maxAbsoluteDifference,null);assert.equal(r.diagnostics.byMirrorCount,null);
 }
 const r=await executeInjected(()=>{},{preset:'structured-mirror-panels',size:8,laneChunkSize:128});assert.equal(r.validation.admissionPassed,true);assert.equal(r.gpu.executed,false);for(const field of ['gpuAssisted','absoluteDifference','replayMask'])assert.equal(r.comparison[field],null);
});
await test('structured path powers cannot be uniformized, moved between owners or detached from mirror history',async()=>{
 for(const mutate of [c=>c.image.fill(3),c=>{const p=c.audit.paths.find(p=>p.value===0);p.sampleValue=3;p.value=3;},c=>{const p=c.audit.paths.find(p=>p.scatters.length===1);p.scatters.push(structuredClone(p.scatters[0]));},c=>{const p=c.audit.paths.find(p=>p.value>0);p.vertices.at(-1).triangleId=12;},c=>{const a=c.audit.paths.find(p=>p.value===.75),b=c.audit.paths.find(p=>p.value===6);[a.key,b.key]=[b.key,a.key];}]){
  const r=await executeInjected(mutate,{preset:'structured-mirror-panels',size:8,laneChunkSize:128});assert.equal(r.validation.admissionPassed,false);assert.equal(r.status,'cpu-only-fallback');for(const field of ['gpuAssisted','absoluteDifference','replayMask'])assert.equal(r.comparison[field],null);
 }
});
await test('structured numerical recovery preserves mixed-power pixels and deduplicates exact sample ownership',async()=>{
 const oracle=structuredOracle(8),pixel=oracle.image.findIndex(v=>v===3.75);assert(pixel>=0);
 for(const overlap of [false,true]){
  const r=await executeInjected(c=>{const a=c.audit.paths.find(p=>p.key.pixel===pixel&&p.key.sample===0),b=c.audit.paths.find(p=>p.key.pixel===pixel&&p.key.sample===1);assert.notEqual(a.value,b.value);a.vertices[0].origin[0]+=.0001;c.recovery.keys=[{key:{...(overlap?a:b).key},reasons:['axisIntervalUncertain']}];c.recovery.paths=1;},{preset:'structured-mirror-panels',size:8,laneChunkSize:128});
  assert.equal(r.validation.admissionPassed,true);assert.equal(r.recovery.explicitReplayPaths,1);assert.equal(r.recovery.auditNumericalRecoveryPaths,1);assert.equal(r.recovery.totalRecoveredPaths,overlap?1:2);assert.equal(r.image.values[pixel],3.75);assert.equal(r.recovery.retainedPaths,128-(overlap?1:2));assert.deepEqual(r.recovery.auditNumericalRecoveryKeys,[`0:${pixel}:0`]);assert.deepEqual(r.recovery.explicitReplayKeys,[`0:${pixel}:${overlap?0:1}`]);
 }
});
await test('worker ignores stale completion/progress and malformed envelopes cannot start or cancel work',async()=>{
 const calls=[],posts=[],scope={postMessage:m=>posts.push(m)},dispose=installResidentHardwareTestWorker(scope,(r,c)=>{const d=deferred();calls.push({r,c,d});return d.promise;});
 scope.onmessage({data:{id:1,request}});assert.equal(calls.length,1);
 for(const data of [{},{type:'unexpected',id:9,request},{id:0,request},{id:2,request:{wave:true}}])await scope.onmessage({data});
 assert.equal(calls.length,1);assert.equal(calls[0].c.signal.aborted,false,'Malformed message cancelled legitimate work');
 scope.onmessage({data:{id:2,request}});assert.equal(calls.length,2);assert.equal(calls[0].c.signal.aborted,true);const before=posts.length;
 calls[0].c.progress({message:'stale'});calls[0].d.resolve({stale:true});await delay();assert.equal(posts.length,before);
 calls[1].c.progress({message:'current'});calls[1].d.resolve({current:true});await delay();assert.equal(posts.at(-1).type,'result');assert.equal(posts.at(-1).id,2);assert(posts.at(-1).result.current);
 dispose();assert.equal(scope.onmessage,null);
});
const window=new Window({url:'https://independent.invalid/resident-test.html'}),workers=[],paint=[];
window.document.write(fs.readFileSync(new URL('../dist/resident-test.html',import.meta.url),'utf8'));
Object.assign(globalThis,{window,document:window.document});
const $=id=>document.getElementById(id);
window.HTMLCanvasElement.prototype.getContext=function(){const canvasId=this.id;return {clearRect(){},createImageData(w,h){return {data:new Uint8ClampedArray(w*h*4)};},putImageData(data){paint.push({canvasId,data});}};};
class MockWorker{
 constructor(){this.terminated=false;workers.push(this);}
 postMessage(data){if(data.type!=='cancel')this.request=data;}
 terminate(){this.terminated=true;}
 emit(data){this.onmessage?.({data:{id:this.request.id,...data}});}
}
globalThis.Worker=MockWorker;
await import('../dist/plt-resident-test-app.js');
const config8=()=>{$('residentPreset').value='two-mirrors-emission';$('residentDimension').value='8';$('residentDimension').dispatchEvent(new window.Event('change'));};
const start=()=>{$('residentRun').click();return workers.at(-1);};
const blank=()=>{for(const id of ['residentEmpty','residentGpuEmpty','residentDifferenceEmpty','residentReplayEmpty'])assert.equal($(id).hidden,false);assert.equal($('residentRetained').textContent,'—');};
const cpuReport=await executeResidentHardwareTest(request,{gpu:null});
await test('UI is inert, defaults stay opt-in and repeated ordinary clicks cannot spawn concurrent workers',()=>{
 assert.equal(workers.length,0);assert.equal($('residentPreset').value,'structured-mirror-panels');assert.equal($('residentDimension').value,'32');assert.equal($('residentLanes').value,'128');config8();const w=start(),count=workers.length;$('residentRun').click();assert.equal(workers.length,count);assert.equal(w.terminated,false);w.emit({type:'result',result:structuredClone(cpuReport)});assert.equal(w.terminated,true);assert.equal($('residentEmpty').hidden,true);assert.equal($('residentBackend').textContent,'CPU 回退图像');assert.equal($('residentDownload').disabled,false);assert.equal(JSON.parse($('residentReport').textContent).image.valuesOmitted,true);for(const id of ['residentGpuEmpty','residentDifferenceEmpty','residentReplayEmpty'])assert.equal($(id).hidden,false);
});
await test('UI controls, cancel, stale progress/error/result, and pagehide clear ownership',()=>{
 for(const id of ['residentPreset','residentDimension','residentLanes']){
  config8();const old=start();$(id).dispatchEvent(new window.Event('change'));blank();const text=$('residentStatus').textContent,count=paint.length;
  old.emit({type:'progress',progress:{message:'STALE'}});old.emit({type:'result',result:structuredClone(cpuReport)});old.emit({type:'error',error:'STALE'});old.onerror?.({message:'STALE'});old.onmessageerror?.();assert.equal($('residentStatus').textContent,text);assert.equal(paint.length,count);
 }
 config8();const w=start();$('residentCancel').click();blank();const text=$('residentStatus').textContent;w.emit({type:'result',result:structuredClone(cpuReport)});assert.equal($('residentStatus').textContent,text);assert.equal($('residentDownload').disabled,true);
 const next=start();next.emit({type:'result',result:structuredClone(cpuReport)});assert.equal($('residentEmpty').hidden,true);window.dispatchEvent(new window.Event('pagehide'));blank();assert.equal($('residentDownload').disabled,true);
});
let admittedMock;
await test('UI rejects contradictory claimed GPU execution and fixed-mode metadata',async()=>{
 const base=await executeInjected();base.status='gpu-validated';base.image.source='gpu-with-selective-recovery';base.validation.imageSource=base.image.source;
 Object.assign(base.gpu,{apiAvailable:true,adapterAvailable:true,deviceCreated:true,executionAttempted:true,executed:true,candidateCompleted:true,candidateAdmitted:true,fullRenderCompleted:true,injectedRunner:false,computeSubmissions:2,completedComputeSubmissions:2});
 base.comparison={cpu:{width:8,height:8,values:[...base.image.values]},gpuAssisted:{width:8,height:8,values:[...base.image.values]},absoluteDifference:{width:8,height:8,values:new Array(64).fill(0)},replayMask:{width:8,height:8,values:new Array(64).fill(0)}};
 Object.assign(base.diagnostics,{maxAbsoluteDifference:0,replayMaskHistogram:{0:64,1:0,2:0},byMirrorCount:{2:{paths:128,retainedPaths:128,explicitReplayPaths:0,auditNumericalRecoveryPaths:0,recoveredPaths:0}}});
 admittedMock=structuredClone(base);
 config8();const good=start();good.emit({type:'result',result:structuredClone(base)});for(const id of ['residentEmpty','residentGpuEmpty','residentDifferenceEmpty','residentReplayEmpty'])assert.equal($(id).hidden,true,'Unmutated mock schema2 positive control must pass');
 for(const mutate of [r=>r.gpu.completedComputeSubmissions=0,r=>r.gpu.completedComputeSubmissions=3,r=>r.gpu.computeSubmissions=-1,r=>r.gpu.injectedRunner=true,r=>r.gpu.candidateCompleted=false,r=>r.requested.spectralMode='visible',r=>r.requested.wavelengthMm=1,r=>r.requested.boundary='soft',r=>r.recovery.totalRecoveredPaths=1,r=>r.image.values[0]=4]){
  config8();const w=start(),bad=structuredClone(base);mutate(bad);w.emit({type:'result',result:bad});blank();assert.match($('residentStatus').textContent,/结果被拒绝/);assert.equal(JSON.parse($('residentReport').textContent).noImageDisplayed,true);
 }
});
await test('UI binds replay union to exact pixel ownership and rejects histogram-preserving map corruption',()=>{
 const base=structuredClone(admittedMock);
 Object.assign(base.recovery,{explicitReplayKeys:['0:0:0','0:1:0'],auditNumericalRecoveryKeys:['0:0:1','0:1:0'],explicitReplayPaths:2,auditNumericalRecoveryPaths:2,totalRecoveredPaths:3,retainedPaths:125,recoveryFraction:3/128});
 Object.assign(base.validation,{rawAuditPassed:false,initialAuditMismatchCount:2});
 base.comparison.replayMask.values[0]=2;base.comparison.replayMask.values[1]=1;
 Object.assign(base.diagnostics,{replayMaskHistogram:{0:62,1:1,2:1},byMirrorCount:{2:{paths:128,retainedPaths:125,explicitReplayPaths:2,auditNumericalRecoveryPaths:2,recoveredPaths:3}}});
 config8();const good=start();good.emit({type:'result',result:structuredClone(base)});assert.equal($('residentReplayEmpty').hidden,true,'Overlapping selective/numerical union is valid and counted once');
 const mutants=[r=>{[r.comparison.replayMask.values[0],r.comparison.replayMask.values[2]]=[r.comparison.replayMask.values[2],r.comparison.replayMask.values[0]];},r=>r.recovery.explicitReplayKeys[1]='0:0:0',r=>r.recovery.auditNumericalRecoveryKeys[1]='0:64:0',r=>r.recovery.auditNumericalRecoveryKeys[1]='0:1:2',r=>r.recovery.auditNumericalRecoveryKeys[1]='1:1:0',r=>r.recovery.auditNumericalRecoveryKeys[1]='0:01:0',r=>r.recovery.explicitReplayKeys=null,r=>r.comparison.absoluteDifference.values[0]=1e-6,r=>r.diagnostics.maxAbsoluteDifference=1e-6,r=>r.diagnostics.byMirrorCount[2].retainedPaths--,r=>r.comparison.replayMask.values[0]=-1,r=>r.comparison.replayMask.values[0]=3];
 for(const mutate of mutants){config8();const w=start(),bad=structuredClone(base);mutate(bad);w.emit({type:'result',result:bad});blank();assert.match($('residentStatus').textContent,/结果被拒绝/);}
});
await test('UI never displays fabricated GPU, difference or replay views on CPU fallback',()=>{
 for(const mutate of [r=>r.comparison.gpuAssisted=structuredClone(r.comparison.cpu),r=>r.comparison.absoluteDifference={width:8,height:8,values:new Array(64).fill(0)},r=>r.comparison.replayMask={width:8,height:8,values:new Array(64).fill(0)},r=>r.diagnostics.maxAbsoluteDifference=0,r=>r.diagnostics.replayMaskHistogram={0:64,1:0,2:0},r=>r.diagnostics.byMirrorCount={}]){
  config8();const w=start(),bad=structuredClone(cpuReport);mutate(bad);w.emit({type:'result',result:bad});blank();assert.match($('residentStatus').textContent,/结果被拒绝/);
 }
});
const nativeEvidence=JSON.parse(fs.readFileSync(new URL('../research/plt-resident-test-runtime-v25-validation.json',import.meta.url),'utf8'));
await test('saved structured native comparisons agree with independently derived powers and replay ownership',()=>{
 assert(nativeEvidence.allPass&&nativeEvidence.actualGPUExecuted);for(const [file,sha] of Object.entries(nativeEvidence.sourceSHA256))assert.equal(crypto.createHash('sha256').update(fs.readFileSync(new URL('../dist/'+file,import.meta.url))).digest('hex'),sha,file);
 const cases=nativeEvidence.nativeCases.filter(r=>r.requested.preset==='structured-mirror-panels');assert.equal(cases.length,4);
 for(const r of cases){
  const oracle=structuredOracle(r.requested.size),keys=new Set([...r.recovery.explicitReplayKeys,...r.recovery.auditNumericalRecoveryKeys]),mask=new Array(oracle.image.length).fill(0),byMirror={};
  assert.deepEqual(r.comparison.cpu.values,oracle.image);assert.deepEqual(r.comparison.gpuAssisted.values,oracle.image);assert.deepEqual(r.image.values,oracle.image);assert(r.comparison.absoluteDifference.values.every(v=>v===0));assert.equal(r.diagnostics.maxAbsoluteDifference,0);
  for(const p of oracle.paths){const key=`0:${p.pixel}:${p.sample}`,group=byMirror[p.mirrors]??={paths:0,retainedPaths:0,explicitReplayPaths:0,auditNumericalRecoveryPaths:0,recoveredPaths:0};const recovered=keys.has(key);group.paths++;group.retainedPaths+=!recovered;group.explicitReplayPaths+=r.recovery.explicitReplayKeys.includes(key);group.auditNumericalRecoveryPaths+=r.recovery.auditNumericalRecoveryKeys.includes(key);group.recoveredPaths+=recovered;if(recovered)mask[p.pixel]++;}
  assert.deepEqual(r.comparison.replayMask.values,mask);assert.deepEqual(r.diagnostics.byMirrorCount,byMirror);assert.equal(mask.reduce((a,b)=>a+b,0),r.recovery.totalRecoveredPaths);assert.equal(r.physicalGPUValidated,false);
 }
 const primary=cases.find(r=>r.requested.size===32&&r.requested.laneChunkSize===128),control=cases.find(r=>r.requested.size===32&&r.requested.laneChunkSize===32);assert.deepEqual(primary.comparison,control.comparison);assert.deepEqual(primary.recovery,control.recovery);assert.equal(primary.validation.finalRngFingerprintFNV1a,control.validation.finalRngFingerprintFNV1a);
});
await test('actual structured report paints equal CPU/GPU exposure, raw zero difference and exact categorical mask',()=>{
 const r=nativeEvidence.nativeCases.find(r=>r.requested.preset==='structured-mirror-panels'&&r.requested.size===32&&r.requested.laneChunkSize===128);$('residentPreset').value=r.requested.preset;$('residentDimension').value='32';$('residentLanes').value='128';$('residentDimension').dispatchEvent(new window.Event('change'));
 const w=start(),first=paint.length;w.emit({type:'result',result:structuredClone(r)});assert.equal(paint.length-first,4);const frames=Object.fromEntries(paint.slice(first).map(p=>[p.canvasId,p.data.data]));assert.deepEqual(frames.residentCanvas,frames.residentGpuCanvas);
 for(let pixel=0;pixel<1024;pixel++){const gray=Math.round(255*Math.pow(-Math.expm1(-r.comparison.cpu.values[pixel]*.25),1/2.2));assert.deepEqual([...frames.residentCanvas.slice(4*pixel,4*pixel+4)],[gray,gray,gray,255]);assert.deepEqual([...frames.residentDifferenceCanvas.slice(4*pixel,4*pixel+4)],[0,0,0,255]);assert.deepEqual([...frames.residentReplayCanvas.slice(4*pixel,4*pixel+4)],[...[[16,27,25],[227,186,117],[186,139,212]][r.comparison.replayMask.values[pixel]],255]);}
 const exported=JSON.parse($('residentReport').textContent);for(const view of Object.values(exported.comparison)){assert.equal(view.valuesOmitted,true);assert.equal(view.values,undefined);}assert.deepEqual(exported.recovery.explicitReplayKeys,r.recovery.explicitReplayKeys);assert.equal(exported.diagnostics.maxAbsoluteDifference,0);
 $('residentCancel').click();blank();assert.equal($('residentDownload').disabled,true);
});
console.log(JSON.stringify({allPass:true,checks:checks.length,actualGPUExecuted:false,domMockOnly:true,browserValidated:false,physicalGPUValidated:false}));
await window.happyDOM.close();
