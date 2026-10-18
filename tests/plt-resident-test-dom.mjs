/** Independent UI lifecycle/contract tests using happy-dom and a Worker double.
 * CPU fallback fixtures execute the real bounded runtime. GPU records are mocks,
 * never hardware evidence. This is not browser visual QA. */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {createHash} from 'node:crypto';
import {executeResidentHardwareTest} from '../dist/plt-resident-test-runtime.js';
const {Window} = await import(process.env.WAVE_TEST_DOM || 'happy-dom');
const win = new Window({url:'https://test.local/resident-test.html',settings:{disableJavaScriptEvaluation:true,disableJavaScriptFileLoading:true,disableCSSFileLoading:true}});
win.document.write(fs.readFileSync(new URL('../dist/resident-test.html',import.meta.url),'utf8'));
const $ = id => win.document.getElementById(id);
const workers = [], downloads = [], blobs = new Map(), livePixels = new WeakMap(), paintCalls = [], contexts = new WeakMap();
let mode = 'manual', blobSerial = 0, networkRequests = 0;
win.HTMLCanvasElement.prototype.getContext = function () {
  if (!contexts.has(this)) {
    const canvas = this;
    contexts.set(this,{createImageData:(width,height)=>({width,height,data:new Uint8ClampedArray(width*height*4)}),putImageData:image=>{paintCalls.push(image);livePixels.set(canvas,image);},clearRect:()=>livePixels.delete(canvas)});
  }
  return contexts.get(this);
};
Object.assign(globalThis,{window:win,document:win.document});
Object.defineProperty(globalThis,'navigator',{configurable:true,value:{}});
globalThis.fetch = async () => { networkRequests++; throw new Error('Unexpected UI fetch'); };
URL.createObjectURL = blob => { const url=`blob:test-${++blobSerial}`; blobs.set(url,blob); return url; };
URL.revokeObjectURL = url => blobs.delete(url);
win.HTMLAnchorElement.prototype.click = function () { downloads.push({href:this.href,name:this.download,blob:blobs.get(this.href)}); };
class MockWorker {
  constructor(url,options) {
    if (mode === 'construct-error') throw new Error('mock constructor failure');
    this.url=String(url);this.options=options;this.mode=mode;this.messages=[];this.terminated=false;this.terminateCalls=0;
    workers.push(this);
  }
  postMessage(message) {
    this.messages.push(message);
    if (message.type === 'cancel') return;
    this.request=message;
    if (this.mode === 'post-error') throw new Error('mock post failure');
  }
  terminate() { this.terminated=true; this.terminateCalls++; }
  // Deliberately deliver after cancellation, including saved handler invocations.
  emit(data) { this.onmessage?.({data:{id:this.request.id,...data}}); }
  error(message='mock worker failure') { this.onerror?.({message}); }
  messageError() { this.onmessageerror?.({}); }
}
globalThis.Worker = MockWorker;
await import('../dist/plt-resident-test-app.js');
const wait = ms => new Promise(resolve=>setTimeout(resolve,ms));
const last = () => workers.at(-1);
const click = id => $(id).dispatchEvent(new win.MouseEvent('click',{bubbles:true}));
const change = (id,value,event='change') => { $(id).value=String(value); $(id).dispatchEvent(new win.Event(event,{bubbles:true})); };
const fresh = () => { mode='manual'; click('residentReset'); };
const start = () => { click('residentRun'); return last(); };
const assertIdle = () => { assert.equal($('residentRun').disabled,false);assert.equal($('residentCancel').hidden,true);assert.equal($('residentProgress').hidden,true); };
const viewIds = [['residentCanvas','residentEmpty'],['residentGpuCanvas','residentGpuEmpty'],['residentDifferenceCanvas','residentDifferenceEmpty'],['residentReplayCanvas','residentReplayEmpty']];
const assertNoImage = () => { for(const [canvas,empty] of viewIds){assert.equal($(empty).hidden,false);assert.equal(livePixels.has($(canvas)),false);} };
const assertAllViews = () => { for(const [canvas,empty] of viewIds){assert.equal($(empty).hidden,true);assert.equal(livePixels.has($(canvas)),true);} };
const assertCPUOnly = () => { assert.equal($('residentEmpty').hidden,true);for(const [canvas,empty] of viewIds.slice(1)){assert.equal($(empty).hidden,false);assert.equal(livePixels.has($(canvas)),false);assert.match($(empty).textContent,/不可用/);} };
const assertCleared = () => { assertNoImage();assert.equal($('residentDownload').disabled,true);assert.equal($('residentReport').textContent,'尚无结果'); for (const id of ['residentAdmission','residentExecuted','residentAdapter','residentReplay','residentRetained','residentTotalTime']) assert.equal($(id).textContent,'—'); };
const fixtures = new Map();
async function cpu(config = {preset:'structured-mirror-panels',size:32,laneChunkSize:128}) {
  const key=JSON.stringify(config);
  if (!fixtures.has(key)) fixtures.set(key,await executeResidentHardwareTest(config,{gpu:undefined}));
  return structuredClone(fixtures.get(key));
}
async function gpu(config) {
  const result=await cpu(config),paths=result.requested.size**2*2;
  Object.assign(result,{status:'gpu-validated',failure:null});
  Object.assign(result.gpu,{apiAvailable:true,adapterAvailable:true,deviceCreated:true,executionAttempted:true,executed:true,computeSubmissions:8,completedComputeSubmissions:8,injectedRunner:false,candidateCompleted:true,candidateAdmitted:true,fullRenderCompleted:true,adapter:{vendor:'mock-vendor',architecture:'mock-arch',device:'mock-device',description:'Mock software adapter',isFallbackAdapter:true}});
  Object.assign(result.validation,{admissionPassed:true,imageSource:'gpu-with-selective-recovery'});
  Object.assign(result.image,{source:'gpu-with-selective-recovery'});
  Object.assign(result.timings,{rendererIncludingReadbackReplayMs:123,admissionMs:1});
  Object.assign(result.recovery,{explicitReplayPaths:10,auditNumericalRecoveryPaths:2,totalRecoveredPaths:11,retainedPaths:paths-11,recoveryFraction:11/paths,explicitReplayReasons:{cameraInterval:10},auditNumericalRecoveryReasons:{distance:2}});
  result.recovery.explicitReplayKeys=Array.from({length:10},(_,n)=>`0:${Math.floor(n/2)}:${n%2}`);result.recovery.auditNumericalRecoveryKeys=['0:4:1','0:5:0'];
  const size=result.requested.size,pixels=size*size;
  result.comparison.gpuAssisted=structuredClone(result.comparison.cpu);
  result.comparison.absoluteDifference={width:size,height:size,values:Array(pixels).fill(0)};
  const mask=Array(pixels).fill(0);for(let n=0;n<11;n++) mask[Math.floor(n/2)]++;
  result.comparison.replayMask={width:size,height:size,values:mask};
  const counts={'0':0,'1':0,'2':0};for(const value of mask)counts[value]++;
  const byMirror=Object.fromEntries(Object.entries(result.diagnostics.mirrorCountHistogram).map(([key,count])=>[key,{paths:count,retainedPaths:count,explicitReplayPaths:0,auditNumericalRecoveryPaths:0,recoveredPaths:0}]));
  const first=Object.values(byMirror).find(group=>group.paths>=11);Object.assign(first,{retainedPaths:first.paths-11,explicitReplayPaths:10,auditNumericalRecoveryPaths:2,recoveredPaths:11});
  Object.assign(result.diagnostics,{maxAbsoluteDifference:0,replayMaskHistogram:counts,byMirrorCount:byMirror});
  return result;
}
const cases=[];
async function test(name,body) { fresh();try { await body();cases.push(name);console.log('PASS',name); } catch(error) { console.error('FAIL',name);throw error; } }

assert.equal(workers.length,0,'page load must not auto-run');
assert.equal(networkRequests,0);assertCleared();
assert.equal($('residentDimension').value,'32');assert.equal($('residentLanes').value,'128');
assert.equal(win.document.querySelector('a[aria-current="page"]').getAttribute('href'),'resident-test.html');
const stable=fs.readFileSync(new URL('../dist/tracer.html',import.meta.url),'utf8');
assert(stable.includes('<a href="resident-test.html">GPU 驻留实验测试</a>'));
assert(stable.includes('value="monochrome" selected'));

await test('no automatic start, explicit default bounded request',async()=>{
  const w=start();assert.match(w.url,/plt-resident-test-worker\.js$/);assert.equal(w.options.type,'module');
  assert.deepEqual(w.request.request,{preset:'structured-mirror-panels',size:32,laneChunkSize:128});assert(Number.isSafeInteger(w.request.id)&&w.request.id>0);
  assert.equal($('residentRun').disabled,true);assert.equal($('residentCancel').hidden,false);assert.equal($('residentProgress').hasAttribute('value'),false);
  w.emit({type:'progress',progress:{stage:'pipeline-compilation',message:'正在编译',entryPoint:'camera_guard',elapsedMs:12345}});
  assert.match($('residentStatus').textContent,/正在编译.*camera_guard.*12.3/);assertNoImage();
});
await test('all preset, square size and lane controls reach exact request',async()=>{
  for (const preset of ['structured-mirror-panels','two-mirrors-emission','four-scale-shifted-zxy','half-scale-three-mirrors-negative-permutation']) for(const size of [8,16,32]) for(const lane of [32,128]) {
    change('residentPreset',preset);change('residentDimension',size);change('residentLanes',lane);
    const w=start();assert.deepEqual(w.request.request,{preset,size,laneChunkSize:lane});click('residentCancel');assertCleared();
  }
});
await test('real no-WebGPU CPU fallback renders with truthful CPU labels',async()=>{
  const w=start();w.emit({type:'result',result:await cpu(w.request.request)});
  assertIdle();assert.equal(w.terminated,true);assert.equal($('residentEmpty').hidden,true);
  assert.match($('residentBackend').textContent,/CPU 回退/);assert.match($('residentAdmission').textContent,/CPU 参考/);
  assert.match($('residentExecuted').textContent,/无已完成/);assert.match($('residentRetained').textContent,/0%/);
  assert.match($('residentReplay').textContent,/无候选准入/);assert.match($('residentRenderTime').textContent,/未单独测量/);
  assert.equal($('residentDownload').disabled,false);assert.equal(livePixels.get($('residentCanvas')).data.length,32*32*4);
});
await test('all three real frozen CPU fixtures pass UI validation',async()=>{
  for (const preset of ['two-mirrors-emission','four-scale-shifted-zxy','half-scale-three-mirrors-negative-permutation']) {
    change('residentPreset',preset);change('residentDimension','8');const w=start();w.emit({type:'result',result:await cpu(w.request.request)});
    assert.equal($('residentEmpty').hidden,true);assert.match($('residentMeasurement').textContent,new RegExp(preset.includes('half')?'0.75':'3'));assertIdle();
  }
});
await test('mock GPU execution requires strict admission and reports replay union',async()=>{
  const w=start();w.emit({type:'result',result:await gpu(w.request.request)});assertIdle();
  assert.equal($('residentEmpty').hidden,true);assert.match($('residentExecuted').textContent,/已完成执行.*已准入/);
  assert.match($('residentAdapter').textContent,/mock-vendor.*软件回退/);assert.match($('residentReplay').textContent,/去重 11 \/ 2048/);
  assert.match($('residentRetained').textContent,/99.5%/);assert.match($('residentWarnings').textContent,/物理 GPU/);
});
await test('completed compute then CPU fallback cannot be presented as admitted GPU',async()=>{
  const w=start(),result=await cpu(w.request.request);
  Object.assign(result.gpu,{apiAvailable:true,adapterAvailable:true,deviceCreated:true,executionAttempted:true,executed:true,computeSubmissions:4,completedComputeSubmissions:2,fullRenderCompleted:false,candidateAdmitted:false});
  w.emit({type:'result',result});assert.match($('residentBackend').textContent,/CPU 回退/);assert.match($('residentExecuted').textContent,/曾完成计算提交.*未采用/);
});
await test('export contains all bounded metadata but no image values or DOM-injected content',async()=>{
  const w=start(),result=await gpu(w.request.request);result.extraDiagnostics={bounded:['entry',17,null],unknownAdapterField:false};
  w.emit({type:'result',result});$('residentReport').textContent='tampered UI';click('residentDownload');
  const download=downloads.at(-1),report=JSON.parse(await download.blob.text());
  assert.equal(download.name,'resident-ray-local-test.json');assert.equal(report.image.values,undefined);assert.equal(report.image.valuesOmitted,true);
  assert.deepEqual(report.extraDiagnostics,result.extraDiagnostics);assert.deepEqual(report.environment,result.environment);assert.deepEqual(report.build,result.build);
  assert.equal(report.physicalGPUValidated,false);assert.equal(download.blob.type,'application/json');assert.equal(networkRequests,0);
});
await test('parameter input/change invalidate image, report, export and active worker',async()=>{
  for (const [id,value,event] of [['residentPreset','four-scale-shifted-zxy','change'],['residentDimension','8','input'],['residentLanes','32','change']]) {
    fresh();const w=start();w.emit({type:'result',result:await cpu(w.request.request)});change(id,value,event);assertCleared();assertIdle();
    const pending=start();change(id,id==='residentPreset'?'two-mirrors-emission':id==='residentDimension'?'16':'128',event);assertCleared();assertIdle();
    assert.equal(pending.messages.at(-1).type,'cancel');
    pending.emit({type:'progress',progress:{message:'stale progress'}});pending.emit({type:'result',result:await cpu(pending.request.request)});pending.error('stale error');pending.messageError();assertCleared();
    assert.doesNotMatch($('residentStatus').textContent,/stale/);
  }
});
await test('cancel immediately invalidates then terminates after bounded cleanup grace',async()=>{
  const w=start();click('residentCancel');assertCleared();assertIdle();assert.equal(w.messages.at(-1).type,'cancel');
  w.emit({type:'result',result:await cpu(w.request.request)});assertCleared();await wait(300);assert.equal(w.terminated,true);
});
await test('repeated programmatic starts retire old workers before starting current',async()=>{
  const first=start();$('residentRun').disabled=false;const second=start();assert.equal(first.terminated,true);assert.notEqual(first.request.id,second.request.id);
  first.emit({type:'result',result:await cpu(first.request.request)});first.error();assertNoImage();assert.equal($('residentRun').disabled,true);
  second.emit({type:'result',result:await cpu(second.request.request)});assertIdle();assert.equal($('residentEmpty').hidden,true);
});
await test('new run flushes a previously cancelled worker grace timer',async()=>{
  const first=start();click('residentCancel');const second=start();assert.equal(first.terminated,true);
  assert.equal(workers.filter(w=>!w.terminated).length,1);second.emit({type:'result',result:await cpu(second.request.request)});assertIdle();
});
await test('wrong generation is ignored, final result retires late progress',async()=>{
  const w=start();w.emit({id:w.request.id+1,type:'result',result:await cpu(w.request.request)});assertNoImage();
  w.emit({type:'result',result:await cpu(w.request.request)});const message=$('residentStatus').textContent;
  w.emit({type:'progress',progress:{message:'late progress'}});w.error('late error');assert.equal($('residentStatus').textContent,message);
});
await test('reset restores frozen defaults and clears all old results',async()=>{
  change('residentPreset','four-scale-shifted-zxy');change('residentDimension','8');change('residentLanes','32');
  const w=start();w.emit({type:'result',result:await cpu(w.request.request)});click('residentReset');assertCleared();assertIdle();
  assert.equal($('residentPreset').value,'structured-mirror-panels');assert.equal($('residentDimension').value,'32');assert.equal($('residentLanes').value,'128');
});
for (const failure of ['construct-error','post-error','worker-error','message-error','runtime-error','unknown-message']) await test(`${failure} releases UI and exposes only bounded error metadata`,async()=>{
  mode=failure;const count=workers.length;click('residentRun');const w=last();
  if(failure==='worker-error')w.error();if(failure==='message-error')w.messageError();
  if(failure==='runtime-error')w.emit({type:'error',name:'GPUDeviceLostError',error:'device lost',diagnostic:{reason:'unknown',flags:2}});
  if(failure==='unknown-message')w.emit({type:'preview',image:[3]});
  assertIdle();assertNoImage();assert.equal($('residentDownload').disabled,false);
  const report=JSON.parse($('residentReport').textContent);assert.equal(report.status,'error');assert.equal(report.noImageDisplayed,true);
  if(workers.length>count)assert.equal(w.terminated,true);
});
await test('structured CPU image has genuine spatial variation and exact frozen path histogram',async()=>{
  const w=start(),result=await cpu(w.request.request);w.emit({type:'result',result});assertCPUOnly();
  assert(new Set(result.image.values).size>2);assert(result.image.values.includes(0));assert(result.image.values.some(value=>value>0));
  assert.deepEqual(result.diagnostics.pathPowerHistogram,{'0':181,'0.75':511,'1.5':424,'3':423,'6':509});
  assert.deepEqual(result.diagnostics.mirrorCountHistogram,{'0':181,'1':933,'2':934});
  assert.match($('residentMirrorHistogram').textContent,/0 次 181 条.*1 次 933 条.*2 次 934 条/);
  assert.match($('residentDifferenceMax').textContent,/不可用/);assert.match($('residentReplayHistogram').textContent,/不可用/);
});
await test('structured real CPU fallback supports 8, 16 and 32 square sizes',async()=>{
  for(const size of [8,16,32]){change('residentDimension',size);const w=start();w.emit({type:'result',result:await cpu(w.request.request)});assertCPUOnly();assert.equal($('residentCanvas').width,size);}
});
await test('four views use matched fixed exposure, fixed difference scale and exact mask colours',async()=>{
  const w=start(),result=await gpu(w.request.request);w.emit({type:'result',result});assertAllViews();
  const cpuPixels=livePixels.get($('residentCanvas')).data,gpuPixels=livePixels.get($('residentGpuCanvas')).data,diff=livePixels.get($('residentDifferenceCanvas')).data,mask=livePixels.get($('residentReplayCanvas')).data;
  assert.deepEqual(gpuPixels,cpuPixels);
  for(let i=0;i<result.image.values.length;i++){
    const expected=Math.round(255*Math.pow(-Math.expm1(-result.comparison.cpu.values[i]*.25),1/2.2));
    assert.deepEqual(Array.from(cpuPixels.slice(i*4,i*4+4)),[expected,expected,expected,255]);
    assert.deepEqual(Array.from(diff.slice(i*4,i*4+4)),[0,0,0,255]);
    const colour=[[16,27,25],[227,186,117],[186,139,212]][result.comparison.replayMask.values[i]];
    assert.deepEqual(Array.from(mask.slice(i*4,i*4+4)),[...colour,255]);
  }
  assert.equal($('residentDifferenceMax').textContent,'最大绝对差值：0');assert.match($('residentReplayHistogram').textContent,/1\/2：1 像素.*2\/2：5 像素/);
});
await test('JSON removes all four comparison pixel arrays but keeps numeric diagnostics and path ownership',async()=>{
  const w=start(),result=await gpu(w.request.request);w.emit({type:'result',result});click('residentDownload');
  const exported=JSON.parse(await downloads.at(-1).blob.text());
  for(const key of ['cpu','gpuAssisted','absoluteDifference','replayMask']){assert.equal(exported.comparison[key].values,undefined);assert.equal(exported.comparison[key].valuesOmitted,true);assert.equal(exported.comparison[key].width,32);}
  assert.deepEqual(exported.diagnostics,result.diagnostics);assert.deepEqual(exported.recovery.explicitReplayKeys,result.recovery.explicitReplayKeys);assert.deepEqual(exported.recovery.auditNumericalRecoveryKeys,result.recovery.auditNumericalRecoveryKeys);
});
await test('CPU-only JSON preserves unavailable comparisons as null',async()=>{
  const w=start();w.emit({type:'result',result:await cpu(w.request.request)});click('residentDownload');const exported=JSON.parse(await downloads.at(-1).blob.text());
  assert.equal(exported.comparison.cpu.values,undefined);for(const key of ['gpuAssisted','absoluteDifference','replayMask'])assert.equal(exported.comparison[key],null);
});
await test('cancel, reset, every parameter and failure clear all four previously painted canvases',async()=>{
  for(const action of ['cancel','reset','preset','size','lanes','error']){
    fresh();let w=start();w.emit({type:'result',result:await gpu(w.request.request)});assertAllViews();
    if(action==='cancel')click('residentCancel');if(action==='reset')click('residentReset');
    if(action==='preset')change('residentPreset','two-mirrors-emission');if(action==='size')change('residentDimension','8');if(action==='lanes')change('residentLanes','32');
    if(action==='error'){w=start();w.emit({type:'error',error:'failed new render'});assertNoImage();}else assertCleared();
    assertIdle();
  }
});
await test('late cancelled GPU comparison cannot repaint any of the four views',async()=>{
  const old=start(),oldResult=await gpu(old.request.request);click('residentCancel');const next=start();old.emit({type:'result',result:oldResult});assertNoImage();
  next.emit({type:'result',result:await cpu(next.request.request)});assertCPUOnly();old.emit({type:'result',result:oldResult});assertCPUOnly();
});

const mutations = [
 ['schema1',r=>r.schemaVersion=1],['missing comparison',r=>delete r.comparison],['extra comparison image',r=>r.comparison.extra={values:[1]}],['missing CPU reference',r=>r.comparison.cpu=null],['CPU width mismatch',r=>r.comparison.cpu.width=8],['CPU comparison NaN',r=>r.comparison.cpu.values[0]=NaN],['CPU comparison power mismatch',r=>r.comparison.cpu.values[0]=123],['CPU reference changed',r=>r.comparison.cpu.values[0]=r.comparison.cpu.values[0]===0?6:0],
 ['false oracle proof',r=>r.validation.analyticOraclePassed=false],['wrong analytic model',r=>r.validation.analyticOracle='constant-dyadic-mirror-v1'],['wrong shadow analytic model',r=>r.cpuFullShadow.analyticOracle='constant-dyadic-mirror-v1'],['wrong expected power',r=>r.validation.expectedPower=3],['wrong shadow expected power',r=>r.cpuFullShadow.analyticPower=3],['path power count',r=>r.diagnostics.pathPowerHistogram['6']--],['frozen mirror count',r=>{r.diagnostics.mirrorCountHistogram['0']--;r.diagnostics.mirrorCountHistogram['1']++;}],
 ['CPU-only fake GPU comparison',r=>r.comparison.gpuAssisted=structuredClone(r.comparison.cpu)],['CPU-only fake zero difference',r=>r.comparison.absoluteDifference={width:32,height:32,values:Array(1024).fill(0)}],['CPU-only fake zero replay mask',r=>r.comparison.replayMask={width:32,height:32,values:Array(1024).fill(0)}],['CPU-only fake maximum difference',r=>r.diagnostics.maxAbsoluteDifference=0],['CPU-only fake coverage histogram',r=>r.diagnostics.replayMaskHistogram={'0':1024,'1':0,'2':0}],
 ['schema',r=>r.schemaVersion=99],['preset',r=>r.requested.preset='gallery'],['size',r=>r.requested.size=8],['lanes',r=>r.requested.laneChunkSize=32],
 ['wave mode',r=>r.requested.wave=true],['spectral mode',r=>r.requested.spectral=true],['seed',r=>r.requested.seed=123],['guard',r=>r.requested.guard='none'],['tile',r=>r.requested.tile.width=1],
 ['width',r=>r.image.width=8],['height',r=>r.image.height=8],['missing values',r=>delete r.image.values],['short values',r=>r.image.values.pop()],['NaN pixel',r=>r.image.values[0]=NaN],['infinite pixel',r=>r.image.values[0]=Infinity],['negative pixel',r=>r.image.values[0]=-1],['wrong power',r=>r.image.values[0]=2.9],
 ['unvalidated image',r=>r.image.validated=false],['missing image',r=>r.image=null],['failed validation',r=>r.validation.passed=false],['incomplete image',r=>r.validation.completeImage=false],['wrong RNG',r=>r.validation.exactRng=false],['wrong validated count',r=>r.validation.validatedPaths--],['wrong image source',r=>r.image.source='gpu-with-selective-recovery'],
 ['missing shadow',r=>delete r.cpuFullShadow],['disabled shadow',r=>r.cpuFullShadow.required=false],['incomplete shadow',r=>r.cpuFullShadow.complete=false],['wrong shadow RNG',r=>r.cpuFullShadow.exactCanonicalRng=false],['wrong shadow counts',r=>r.cpuFullShadow.paths--],
 ['physical GPU claim',r=>r.physicalGPUValidated=true],['invalid time',r=>r.timings.endToEndMs=NaN],['negative time',r=>r.timings.cpuFullShadowMs=-1],['missing time',r=>delete r.timings.coldSetupMs],
 ['fabricated executed',r=>r.gpu.executed=true],['unknown status',r=>r.status='fast-mode'],['binary diagnostics',r=>r.binary=new Uint8Array(2)],['infinite diagnostics',r=>r.diagnostic=Infinity],['oversized text',r=>r.diagnostic='x'.repeat(70000)],['circular diagnostics',r=>r.cycle=r]
];
for (const [name,mutate] of mutations) await test(`reject malformed result: ${name}`,async()=>{
  const w=start(),result=await cpu(w.request.request),before=paintCalls.length;mutate(result);w.emit({type:'result',result});
  assertIdle();assertNoImage();assert.equal(paintCalls.length,before);assert.match($('residentStatus').textContent,/结果被拒绝/);
  assert.equal(JSON.parse($('residentReport').textContent).status,'error');
});
for(const [name,mutate] of [
 ['consistent forged difference',r=>{r.comparison.gpuAssisted.values[0]+=.1;r.image.values[0]=r.comparison.gpuAssisted.values[0];r.comparison.absoluteDifference.values[0]=Math.abs(r.image.values[0]-r.comparison.cpu.values[0]);r.diagnostics.maxAbsoluteDifference=r.comparison.absoluteDifference.values[0];}],
 ['GPU comparison missing',r=>r.comparison.gpuAssisted=null],['GPU comparison wrong width',r=>r.comparison.gpuAssisted.width=8],['GPU comparison changed',r=>r.comparison.gpuAssisted.values[0]+=.1],['difference map missing',r=>r.comparison.absoluteDifference=null],['difference NaN',r=>r.comparison.absoluteDifference.values[0]=NaN],['difference map tampering',r=>r.comparison.absoluteDifference.values[0]=1e-8],['difference max tampering',r=>r.diagnostics.maxAbsoluteDifference=1e-8],
 ['mask missing',r=>r.comparison.replayMask=null],['mask sample count out of bounds',r=>r.comparison.replayMask.values[0]=3],['mask fractional count',r=>r.comparison.replayMask.values[0]=.5],['mask sum changed',r=>r.comparison.replayMask.values[0]=0],['mask relocated but histogram unchanged',r=>{const mask=r.comparison.replayMask.values;[mask[0],mask[100]]=[mask[100],mask[0]];}],['mask histogram tampering',r=>r.diagnostics.replayMaskHistogram['0']--],
 ['mirror breakdown missing',r=>r.diagnostics.byMirrorCount=null],['mirror breakdown path count',r=>r.diagnostics.byMirrorCount['0'].paths--],['mirror breakdown recovery count',r=>r.diagnostics.byMirrorCount['0'].recoveredPaths--],
 ['missing explicit ownership',r=>r.recovery.explicitReplayKeys=null],['missing numeric ownership',r=>r.recovery.auditNumericalRecoveryKeys=null],['duplicate explicit ownership',r=>r.recovery.explicitReplayKeys[1]=r.recovery.explicitReplayKeys[0]],['duplicate numeric ownership',r=>r.recovery.auditNumericalRecoveryKeys[1]=r.recovery.auditNumericalRecoveryKeys[0]],['ownership out of pixel bounds',r=>r.recovery.explicitReplayKeys[0]='0:1024:0'],['ownership out of sample bounds',r=>r.recovery.explicitReplayKeys[0]='0:0:2'],['ownership wrong wave flag',r=>r.recovery.explicitReplayKeys[0]='1:0:0'],['noncanonical ownership',r=>r.recovery.explicitReplayKeys[0]='0:00:0'],['ownership union changed',r=>r.recovery.auditNumericalRecoveryKeys[1]='0:3:0'],
 ['not executed',r=>r.gpu.executed=false],['zero completed submissions',r=>r.gpu.completedComputeSubmissions=0],['injected runner',r=>r.gpu.injectedRunner=true],['unfinished candidate',r=>r.gpu.candidateCompleted=false],['overcount submissions',r=>r.gpu.completedComputeSubmissions=99],['negative submissions',r=>r.gpu.computeSubmissions=-1],['missing renderer time',r=>r.timings.rendererIncludingReadbackReplayMs=null],['not admitted',r=>r.gpu.candidateAdmitted=false],['not complete',r=>r.gpu.fullRenderCompleted=false],['admission false',r=>r.validation.admissionPassed=false],['replay counts',r=>r.recovery.retainedPaths=9999],['replay union',r=>r.recovery.totalRecoveredPaths=2],['replay fraction',r=>r.recovery.recoveryFraction=.75]
]) await test(`reject malformed GPU proof: ${name}`,async()=>{
  const w=start(),result=await gpu(w.request.request),before=paintCalls.length;mutate(result);w.emit({type:'result',result});assertIdle();assertNoImage();assert.equal(paintCalls.length,before);
});
await test('unsupported DOM values never launch a worker',async()=>{
  change('residentDimension','999');const count=workers.length;click('residentRun');assert.equal(workers.length,count);assertIdle();assertNoImage();
});
await test('oversized runtime error diagnostics are safely omitted in error export',async()=>{
  const w=start();w.emit({type:'error',error:'bounded error',diagnostic:{large:'x'.repeat(100000)}});
  const report=JSON.parse($('residentReport').textContent);assert(report.error.diagnosticOmitted);assertNoImage();assertIdle();
});
await test('export creation failure remains local and leaves checked image intact',async()=>{
  const w=start();w.emit({type:'result',result:await cpu(w.request.request)});const old=URL.createObjectURL;
  URL.createObjectURL=()=>{throw new Error('mock download failure');};click('residentDownload');URL.createObjectURL=old;
  assert.match($('residentStatus').textContent,/本地下载失败/);assert.equal($('residentEmpty').hidden,true);
});
await test('pagehide terminates pending work and revokes downloads',async()=>{
  const w=start();w.emit({type:'result',result:await cpu(w.request.request)});click('residentDownload');assert(blobs.size>0);
  const pending=start();win.dispatchEvent(new win.Event('pagehide'));assert.equal(pending.terminated,true);assert.equal(blobs.size,0);assertCleared();assertIdle();
  pending.emit({type:'result',result:await cpu(pending.request.request)});assertCleared();
});
// Bridge saved actual SwiftShader execution reports unchanged through the same
// UI message handler. These remain software-native evidence, not browser/hardware QA.
const nativeURL = new URL('../research/plt-resident-test-runtime-v25-validation.json',import.meta.url);
let nativeReportsBridged = 0;
if (fs.existsSync(nativeURL) && JSON.parse(fs.readFileSync(nativeURL,'utf8')).nativeCases?.every(report=>report.schemaVersion===2)) {
  const artifact = JSON.parse(fs.readFileSync(nativeURL,'utf8'));
  assert.equal(artifact.allPass,true);assert.equal(artifact.actualGPUExecuted,true);assert.equal(artifact.physicalGPUValidated,false);
  assert.equal(artifact.nativeAdapterIsSoftwareOnly,true);assert.equal(artifact.nativeCases.length,9);
  for (const [name,hash] of Object.entries(artifact.sourceSHA256)) assert.equal(createHash('sha256').update(fs.readFileSync(new URL('../dist/'+name,import.meta.url))).digest('hex'),hash,'saved native report must match current runtime source');
  for (const result of artifact.nativeCases) await test(`saved native report integration: ${result.requested.preset}/${result.requested.size}/${result.requested.laneChunkSize}`,async()=>{
    const {preset,size,laneChunkSize}=result.requested;
    change('residentPreset',preset);change('residentDimension',size);change('residentLanes',laneChunkSize);
    const w=start();w.emit({type:'result',result:structuredClone(result)});assertIdle();
    assertAllViews();assert.match($('residentBackend').textContent,/GPU 辅助/);assert.match($('residentAdmission').textContent,/通过完整 CPU 核验/);
    assert.equal($('residentCanvas').width,size);assert.equal($('residentCanvas').height,size);assert.equal(livePixels.get($('residentCanvas')).data.length,size*size*4);
    assert($('residentReplay').textContent.includes(`去重 ${result.recovery.totalRecoveredPaths} / ${size*size*2}`));
    assert.equal($('residentRetained').textContent,`${(100*result.recovery.retainedPaths/(size*size*2)).toFixed(1)}%`);
    click('residentDownload');const exported=JSON.parse(await downloads.at(-1).blob.text());
    assert.deepEqual(exported.requested,result.requested);assert.deepEqual(exported.gpu,result.gpu);assert.deepEqual(exported.recovery,result.recovery);assert.deepEqual(exported.timings,result.timings);
    assert.equal(exported.image.values,undefined);assert.equal(exported.image.width,size);assert.equal(exported.image.valuesOmitted,true);
    assert.equal(exported.physicalGPUValidated,false);
    for(const key of ['cpu','gpuAssisted','absoluteDifference','replayMask']){assert.equal(exported.comparison[key].values,undefined);assert.equal(exported.comparison[key].valuesOmitted,true);assert.equal(exported.comparison[key].width,size);}
    assert.deepEqual(exported.diagnostics,result.diagnostics);
    nativeReportsBridged++;
  });
} else console.log('Native report integration skipped: no current schema2 execution artifact.');

assert.equal(networkRequests,0,'UI must not auto-upload or fetch');
console.log(JSON.stringify({passed:true,cases:cases.length,cpuFixtures:fixtures.size,nativeReportsBridged,domMock:true,browserVisualQA:false,physicalGPUValidated:false}));
await win.happyDOM.close();
