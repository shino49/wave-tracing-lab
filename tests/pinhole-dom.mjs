/** Happy-dom + Canvas spies + module-Worker doubles. Includes real 512² point
 * and small 3-D CPU fixtures. No browser layout or physical GPU claim. */
import fs from 'node:fs';
import assert from 'node:assert/strict';
import {executePinholeTask} from '../dist/pinhole-runtime.js';
import {installPinholeWorker} from '../dist/pinhole-worker.js';
import {displaySpectralXYZ, linearToSRGB} from '../dist/plt-spectrum.js';
const {Window} = await import(process.env.WAVE_TEST_DOM || 'happy-dom');
const win = new Window({url:'https://test.local/pinhole.html',settings:{disableJavaScriptEvaluation:true,disableCSSFileLoading:true,disableJavaScriptFileLoading:true}});
win.document.write(fs.readFileSync(new URL('../dist/pinhole.html', import.meta.url), 'utf8'));
const contexts = new WeakMap(), pixels = new WeakMap(), draws = [], calls = [], workers = [], downloadCalls = [], blobs = new Map(), revoked = [], passed = [], failures = [];
win.HTMLCanvasElement.prototype.getContext = function () {
  if (!contexts.has(this)) {
    const canvas = this;
    contexts.set(this, new Proxy({createImageData:(width,height)=>({width,height,data:new Uint8ClampedArray(width*height*4)}),putImageData: image => {pixels.set(canvas,image);draws.push({canvas,image});},clearRect:()=>pixels.delete(canvas)}, {get:(t,k)=>k in t?t[k]:(...args)=>{calls.push({canvas,method:k});}}));
  }
  return contexts.get(this);
};
Object.assign(globalThis,{window:win,document:win.document});
Object.defineProperty(globalThis,'navigator',{value:{},configurable:true});
let networkRequests = 0, workerMode = 'manual', blobId = 0, actualCPUExecutions = 0;
globalThis.fetch = async () => {networkRequests++; throw new Error('No network needed for pinhole CPU fixtures');};
URL.createObjectURL = blob => {const url=`blob:pinhole-${++blobId}`;blobs.set(url,blob);return url;};
URL.revokeObjectURL = url => {blobs.delete(url);revoked.push(url);};
win.HTMLAnchorElement.prototype.click = function () {downloadCalls.push({url:this.href,name:this.download,blob:blobs.get(this.href)});};
class WorkerDouble {
  constructor(url, options) {if(workerMode==='factory-error')throw new Error('Worker construction denied');this.url=String(url);this.options=options;this.mode=workerMode;this.messages=[];this.terminated=false;workers.push(this);}
  terminate(){this.terminated=true;}
  postMessage(request){this.messages.push(request);if(request.type==='cancel')return;this.request=structuredClone(request);if(this.mode==='post-error')throw new Error('Worker post failed');if(!['execute','real-timeout'].includes(this.mode))return;queueMicrotask(async()=>{try{actualCPUExecutions++;const result=await executePinholeTask(this.mode==='real-timeout'?{...request,maxRuntimeMs:1}:request,{preview:p=>{if(!this.terminated)this.emit({type:'preview',...p});},progress:p=>{if(!this.terminated)this.emit({type:'progress',...p});},reset:p=>{if(!this.terminated)this.emit({type:'reset',...p});}});this.result=result;if(!this.terminated)this.emit({type:'result',result});}catch(error){if(!this.terminated)this.emit({type:'error',error:error.message,name:error.name,diagnostics:error.diagnostics,timings:error.timings});}});}
  emit(data){this.onmessage?.({data:{id:this.request.id,...data}});}
  error(){this.onerror?.({message:'device or worker lost',preventDefault(){}});}
  messageError(){this.onmessageerror?.({});}
}
globalThis.Worker = WorkerDouble;
const {boundedPinholeReport} = await import('../dist/pinhole-app.js');
const $=id=>document.getElementById(id), last=()=>workers.at(-1);
function change(id,value){$(id).value=String(value);$(id).dispatchEvent(new win.Event('input',{bubbles:true}));}
function fresh(){workerMode='manual';$('phReset').click();}
function point(){change('phMode','point');change('phCompute','cpu');}
function start(){workerMode='manual';$('phRun').click();return last();}
function idle(){assert.equal($('phRun').disabled,false);assert.equal($('phCancel').hidden,true);assert.equal($('phProgress').hidden,true);}
function emptyImages(){for(const k of ['Wave','Geometric']){assert.equal(pixels.has($(`ph${k}Canvas`)),false,k);assert.equal($(`ph${k}Canvas`).width,512);assert.equal($(`ph${k}Canvas`).height,512);assert.equal($(`ph${k}PartialWrap`).hidden,true);assert.equal($(`ph${k}Empty`).hidden,false);}assert.equal(pixels.has($('phDifferenceCanvas')),false);}
const report=()=>JSON.parse($('phReport').textContent);
async function until(fn){for(let i=0;i<4000;i++){if(fn())return;await new Promise(r=>setTimeout(r,5));}throw new Error('DOM fixture timeout');}
async function test(name, fn){fresh();try{await fn();passed.push(name);console.log('PASS',name);}catch(error){failures.push({name,error:error.stack});console.error('FAIL',name,error.stack);}}
let reference;
function deliver(result=reference){point();const w=start();w.emit({type:'result',result:structuredClone(result)});return w;}
function preview(completed=5,{attempt=0,total=13,power=.774,represented=1,n=512}={}){return {width:n,height:n,waveXYZ:new Float32Array(n*n*3).fill(1e-5),geometricXYZ:new Float32Array(n*n*3).fill(5e-6),complete:false,stageResolution:n,requestedResolution:512,progressive:{kind:'contribution-accumulation',completedContributions:completed,totalContributions:total,sourcePowerFraction:power,representedSourceFraction:represented,attempt}};}
function completedScene(w,attempt=0){return {...structuredClone(reference),configuration:{...reference.configuration,...w.request},progressive:{kind:'contribution-accumulation',completedContributions:13,totalContributions:13,sourcePowerFraction:1,representedSourceFraction:1,attempt}};}

await test('Initial page is explicit-start, accessible-labelled and native 512² with new navigation',()=>{
  assert.equal(workers.length,0);idle();emptyImages();assert.equal(networkRequests,0);
  for(const e of document.querySelectorAll('input,select'))assert(e.labels.length,e.id);
  for(const e of document.querySelectorAll('canvas'))assert(e.getAttribute('aria-label'));
  const ids=[...document.querySelectorAll('[id]')].map(e=>e.id);assert.equal(ids.length,new Set(ids).size);
  assert($('phStatus').closest('[aria-live]'));assert.equal($('phWavelength').disabled,true);
  assert.equal(document.querySelector('[aria-current="page"]').getAttribute('href'),'pinhole.html');
  for(const page of ['index','tracer','wave-demo'])assert(fs.readFileSync(new URL(`../dist/${page}.html`,import.meta.url),'utf8').includes('href="pinhole.html"'));
  assert(calls.some(c=>c.canvas===$('phRigCanvas')&&c.method==='fill'),'Initial apparatus projects actual mesh polygons');
});
await test('Default request fixes 512 resolution, physical aperture and independent depth layers',()=>{
  const w=start();assert.match(w.url,/pinhole-worker\.js$/);assert.deepEqual(w.options,{type:'module'});
  assert.deepEqual({...w.request,id:0},{id:0,mode:'scene',diameterMm:.2,wavelengthNm:550,spectralMode:'broadband',resolution:512,depthLayers:8,backend:'auto'});
  const count=workers.length;$('phRun').click();assert.equal(workers.length,count);$('phCancel').click();assert(w.terminated);idle();emptyImages();
});
await test('Actual 512² CPU point execution computes one final stage and finishes both native canvases',async()=>{
  point();assert.equal($('phSpectrum').value,'narrowband');assert.equal(Number($('phDiameter').value),.08);assert.equal($('phWavelength').value,'550');workerMode='execute';$('phRun').click();const w=last();await until(()=>!$('phRun').disabled);reference=structuredClone(w.result);
  assert(reference);assert.match($('phStatus').textContent,/计算完成/);idle();assert(w.terminated);assert.equal($('phDownload').disabled,false);
  assert.equal(reference.width,512);assert.equal(reference.configuration.sensorWidthMm,8);assert.equal(reference.configuration.detectorDistanceMm,50);
  assert.match($('phBackend').textContent,/CPU/);assert.deepEqual(reference.stages.map(s=>s.resolution),[512]);assert.equal(draws.filter(d=>d.canvas.id==='phWaveCanvas'&&d.image.width<512).length,0);
  assert.equal(report().diagnostics.cpuShadow,false);assert.equal(report().diagnostics.cpuReferenceImages,0);assert.equal(report().stages.length,1);
  assert.equal(report().diagnostics.spectral.samples,1);assert.match($('phSpectralCount').textContent,/1 个波长/);assert.equal(networkRequests,0);
});
await test('Both detector images exactly use shared CIE XYZ/sRGB mapping of actual arrays',()=>{
  deliver();for(const [name,key]of [['Wave','waveXYZ'],['Geometric','geometricXYZ']]){const canvas=$(`ph${name}Canvas`),p=pixels.get(canvas),xyz=reference[key];assert.equal(p.width,512);assert.equal(p.height,512);for(let i=0;i<xyz.length;i+=3){const rgb=displaySpectralXYZ([xyz[i],xyz[i+1],xyz[i+2]],16).rgb;assert.deepEqual([...p.data.slice(i/3*4,i/3*4+4)],[...rgb,255]);}}
  assert.equal(calls.filter(c=>['phWaveCanvas','phGeometricCanvas'].includes(c.canvas.id)).length,0,'No hand-drawn fringe, image proxy or canvas upscale');
});
await test('Exposure is shared display-only, preserves raw record and does not repeat transport',()=>{
  deliver();const count=workers.length,before=report(),old=pixels.get($('phWaveCanvas')).data.slice();change('phExposure',-3);assert.equal(workers.length,count);assert.deepEqual(report(),before);assert.notDeepEqual(pixels.get($('phWaveCanvas')).data,old);
  for(const [name,key]of [['Wave','waveXYZ'],['Geometric','geometricXYZ']]){const p=pixels.get($(`ph${name}Canvas`));for(const i of [0,12345,131328,154112,262143]){const rgb=displaySpectralXYZ(Array.from(reference[key].slice(i*3,i*3+3)),13).rgb;assert.deepEqual([...p.data.slice(i*4,i*4+4)],[...rgb,255]);}}
});
await test('Difference is actual CIE Y subtraction with same exposure and no independent normalization',()=>{
  deliver();const p=pixels.get($('phDifferenceCanvas'));for(let i=0;i<512*512;i++){const d=Math.abs(reference.waveXYZ[3*i+1]-reference.geometricXYZ[3*i+1]),v=Math.round(255*linearToSRGB(-Math.expm1(-65536*d)));assert.deepEqual([...p.data.slice(i*4,i*4+4)],[v,v,v,255]);}
});
await test('Incomplete contribution snapshots use the native512 canvases and remain visibly distinct from completion',()=>{
  const w=start(),p=preview();w.emit({type:'preview',...p});for(const k of ['Wave','Geometric']){const canvas=$(`ph${k}Canvas`);assert.equal(canvas.width,512);assert.equal(canvas.height,512);assert.equal(pixels.get(canvas).width,512);assert.equal(canvas.dataset.complete,'false');assert.equal($(`ph${k}PreviewCanvas`),null);assert.equal($(`ph${k}PartialWrap`).hidden,false);assert.match($(`ph${k}PartialWrap`).textContent,/INCOMPLETE.*5\/13.*100.0%.*77.4%/);assert.match($(`ph${k}Resolution`).textContent,/未完成.*缺少源光/);}
  assert.equal($('phDownload').disabled,true);assert.equal(pixels.has($('phDifferenceCanvas')),false);assert.match($('phMeasurement').textContent,/曝光与最终图相同.*缺源光贡献.*不表示噪声或采样收敛/);
  w.emit({type:'preview',...preview(9,{power:.91})});assert.match($('phWavePartialWrap').textContent,/9\/13/);$('phCancel').click();emptyImages();idle();
});
await test('Incomplete XYZ uses identical exposure without scaling by represented source power',()=>{
  const w=start(),p=preview();w.emit({type:'preview',...p});const count=workers.length;
  for(const exposure of [0,-2]){change('phExposure',exposure);for(const [name,key]of [['Wave','waveXYZ'],['Geometric','geometricXYZ']]){const image=pixels.get($(`ph${name}Canvas`)),rgb=displaySpectralXYZ(Array.from(p[key].slice(0,3)),16+exposure).rgb;assert.deepEqual([...image.data.slice(0,4)],[...rgb,255]);assert.equal($(`ph${name}PartialWrap`).hidden,false);}}
  assert.equal(workers.length,count);assert.equal($('phDownload').disabled,true);assert.equal(pixels.has($('phDifferenceCanvas')),false);
});
await test('Malformed arrays, dimensions, fractions and contribution metadata reject atomically',()=>{
  const mutations=[p=>p.waveXYZ[0]=NaN,p=>p.geometricXYZ[0]=-1,p=>p.waveXYZ=p.waveXYZ.slice(1),p=>p.requestedResolution=256,p=>p.complete=true,p=>p.stageResolution=64,p=>p.progressive=null,p=>p.progressive.kind='sampling',p=>p.progressive.completedContributions=-1,p=>p.progressive.completedContributions=14,p=>p.progressive.totalContributions=0,p=>p.progressive.sourcePowerFraction=NaN,p=>p.progressive.sourcePowerFraction=1.01,p=>p.progressive.representedSourceFraction=-.1,p=>p.progressive.attempt=1,p=>p.progressive.attempt=.5];
  for(const mutate of mutations){fresh();const w=start(),p=preview();mutate(p);w.emit({type:'preview',...p});emptyImages();idle();assert.match($('phStatus').textContent,/渐进帧被拒绝/);assert.equal(report().complete,false);}
  for(const n of [128,256]){fresh();const w=start();w.emit({type:'preview',...preview(5,{n})});emptyImages();idle();assert.match($('phStatus').textContent,/原生 512/);}
});
await test('Regressing counts or fractions and changing contribution totals cannot replace a partial frame',()=>{
  for(const next of [preview(4),preview(9,{power:.7}),preview(9,{represented:.9}),preview(9,{total:14})]){fresh();const w=start();w.emit({type:'preview',...preview()});w.emit({type:'preview',...next});emptyImages();idle();assert.match($('phStatus').textContent,/渐进帧被拒绝/);}
});
await test('Full source support or a fully accumulated snapshot is never mislabeled as a final image',()=>{
  const w=start();w.emit({type:'preview',...preview(13,{power:1,represented:1})});assert.equal($('phWavePartialWrap').hidden,false);assert.equal($('phWaveCanvas').dataset.complete,'false');assert.equal($('phDownload').disabled,true);assert.equal(pixels.has($('phDifferenceCanvas')),false);assert.equal($('phRun').disabled,true);
  w.emit({type:'progress',attempt:0,fraction:1});assert(Number($('phProgress').value)<1);w.emit({type:'result',result:completedScene(w)});idle();assert.equal($('phWavePartialWrap').hidden,true);assert.equal($('phWaveCanvas').dataset.complete,'true');assert.equal($('phDownload').disabled,false);assert.equal(report().progressive.completedContributions,13);
});
await test('GPU retry resets all native pixels before CPU accumulation and ignores stale attempt messages',()=>{
  const w=start();w.emit({type:'preview',...preview()});w.emit({type:'progress',attempt:0,fraction:.7,message:'GPU'});w.emit({type:'reset',attempt:1,reason:'device lost'});emptyImages();assert.equal($('phRun').disabled,true);assert.equal($('phProgress').hidden,false);assert.equal(Number($('phProgress').value),0);assert.match($('phStatus').textContent,/清除部分图像.*CPU/);assert.equal($('phDownload').disabled,true);
  const drawCount=draws.length,statusBefore=$('phStatus').textContent;for(const message of [{type:'preview',...preview(9,{power:.9})},{type:'progress',attempt:0,fraction:1,message:'stale GPU'},{type:'result',result:completedScene(w,0)},{type:'reset',attempt:0}])w.emit(message);assert.equal(draws.length,drawCount);assert.equal($('phStatus').textContent,statusBefore);emptyImages();
  w.emit({type:'preview',...preview(5,{attempt:1,power:.774})});assert.equal($('phWavePartialWrap').hidden,false);const currentPixels=pixels.get($('phWaveCanvas'));w.emit({type:'reset',attempt:1});assert.equal(pixels.get($('phWaveCanvas')),currentPixels);w.emit({type:'progress',attempt:1,fraction:.5,message:'CPU accumulation'});assert.equal(Number($('phProgress').value),.5);w.emit({type:'result',result:completedScene(w,1)});idle();assert.equal($('phWavePartialWrap').hidden,true);assert.equal(report().progressive.attempt,1);assert.match($('phStatus').textContent,/计算完成/);
});
await test('Skipped reset attempts and final results without complete matching contributions fail safely',()=>{
  for(const message of [{type:'reset',attempt:2},{type:'progress',attempt:1,fraction:.5}]){fresh();const w=start();w.emit({type:'preview',...preview()});w.emit(message);idle();emptyImages();assert.equal(report().complete,false);}
  for(const mutate of [r=>r.progressive.completedContributions=12,r=>r.progressive.sourcePowerFraction=.9,r=>r.progressive.representedSourceFraction=.9,r=>delete r.progressive]){fresh();const w=start();w.emit({type:'preview',...preview()});const r=completedScene(w);mutate(r);w.emit({type:'result',result:r});idle();emptyImages();assert.match($('phStatus').textContent,/结果被拒绝/);}
});
await test('Final small frames and mismatched physical configuration cannot be labelled as native512',()=>{
  for(const mutate of [r=>{r.width=256;r.height=256;},r=>r.waveXYZ[0]=NaN,r=>r.configuration.diameterMm=.3,r=>r.configuration.sensorWidthMm=10,r=>r.configuration.detectorDistanceMm=60,r=>r.complete=false,r=>r.totalMs=NaN]){fresh();point();const w=start(),r=structuredClone(reference);mutate(r);w.emit({type:'result',result:r});emptyImages();idle();assert.match($('phStatus').textContent,/结果被拒绝/);assert.equal(report().complete,false);}
});
await test('Parameter edits cancel in-flight work, invalidate results and feed exact next values',()=>{
  for(const [id,value,key]of [['phDiameter',.045,'diameterMm'],['phMode','scene','mode'],['phSpectrum','broadband','spectralMode'],['phWavelength',630,'wavelengthNm'],['phCompute','auto','backend'],['phLayers',16,'depthLayers']]){fresh();deliver();change(id,value);emptyImages();assert.equal($('phDownload').disabled,true);const w=start();assert.equal(w.request[key],value);assert.equal(w.request.resolution,512);$('phCancel').click();}
});
await test('Transport bounds are checked independently of tampered HTML input limits',()=>{
  $('phDiameter').max='1';change('phDiameter',.8);const count=workers.length;$('phRun').click();assert.equal(workers.length,count);idle();emptyImages();assert.match($('phStatus').textContent,/0.04–0.50/);$('phDiameter').max='.5';
});
await test('Apparatus-only orbit preserves completed camera pixels, report and worker count',()=>{
  deliver();const count=workers.length,reportBefore=report(),p=pixels.get($('phWaveCanvas'));change('phOrbit',25);assert.equal(workers.length,count);assert.equal(pixels.get($('phWaveCanvas')),p);assert.deepEqual(report(),reportBefore);assert.equal($('phOrbitValue').textContent,'25°');
});
await test('Late messages after cancellation cannot repaint, update progress or publish a report',()=>{
  point();const w=start();$('phCancel').click();const before=$('phStatus').textContent,count=draws.length;for(const message of [{type:'progress',fraction:1,message:'stale'},{type:'preview',...preview()},{type:'reset',attempt:1},{type:'result',result:reference},{type:'error',error:'stale'}])w.emit(message);w.error();w.messageError();assert.equal($('phStatus').textContent,before);assert.equal(draws.length,count);emptyImages();idle();assert.equal($('phDownload').disabled,true);
});
await test('Superseded worker success cannot overwrite a newer active request',()=>{
  point();const old=start();change('phDiameter',.25);assert(old.terminated);change('phDiameter',reference.configuration.diameterMm);const current=start();old.emit({type:'result',result:reference});assert.equal($('phRun').disabled,true);emptyImages();current.emit({type:'result',result:structuredClone(reference)});assert.match($('phStatus').textContent,/计算完成/);idle();
});
await test('Worker creation, posting, runtime, device and deserialization errors produce bounded local failure records',()=>{
  for(const mode of ['factory-error','post-error','runtime','device','deserialize']){fresh();if(['factory-error','post-error'].includes(mode)){workerMode=mode;$('phRun').click();}else{const w=start();if(mode==='runtime')w.emit({type:'error',error:'GPU device lost',diagnostics:{stage:'compute',buffer:new Float32Array(100)}});if(mode==='device')w.error();if(mode==='deserialize')w.messageError();}idle();emptyImages();assert.equal(report().complete,false);assert.equal($('phDownload').disabled,false);assert(report().error.length<3000);}
});
await test('A real runtime timeout releases the UI and publishes only an incomplete failure record',async()=>{
  point();workerMode='real-timeout';$('phRun').click();const w=last();await until(()=>!$('phRun').disabled);assert(w.terminated);idle();emptyImages();assert.equal(report().complete,false);assert.match($('phStatus').textContent,/计算未完成/);assert.equal($('phDownload').disabled,false);assert.match(report().error,/runtime limit|cancelled|timeout/i);
});
await test('An unsolicited AbortError is a visible worker error, while user-cancelled work remains silent',async()=>{
  const messages=[],scope={postMessage:m=>messages.push(m)};const dispose=installPinholeWorker(scope,async()=>{throw Object.assign(new Error('Internal abort without owner cancellation'),{name:'AbortError'});});await scope.onmessage({data:{id:91}});assert.equal(messages.length,1);assert.equal(messages[0].type,'error');assert.equal(messages[0].name,'AbortError');dispose();
  point();const w=start();w.emit(messages[0]&&{...messages[0],id:w.request.id});idle();emptyImages();assert.equal(report().complete,false);assert.match($('phStatus').textContent,/Internal abort/);
});
await test('Native diagnostic download excludes images, stays local, and pagehide revokes its URL',async()=>{
  deliver();const expected=report(),count=downloadCalls.length;$('phDownload').click();assert.equal(downloadCalls.length,count+1);const file=downloadCalls.at(-1);assert.equal(file.name,'pinhole-camera-local-run.json');assert.equal(file.blob.type,'application/json');assert.deepEqual(JSON.parse(await file.blob.text()),expected);assert(!('waveXYZ'in expected));assert(JSON.stringify(expected).length<64000);window.dispatchEvent(new win.Event('pagehide'));assert(revoked.includes(file.url));emptyImages();idle();assert.equal($('phDownload').disabled,true);assert.equal(networkRequests,0);
});
await test('Sanitizer caps strings, nested structures and lists and excludes typed buffers',()=>{
  const bounded=boundedPinholeReport({s:'x'.repeat(10000),a:Array.from({length:100},(_,i)=>i),waveXYZ:new Float64Array(3),buffer:new ArrayBuffer(8),deeper:{a:{b:{c:{d:{e:{f:{g:1}}}}}}}});assert.equal(bounded.s.length,1024);assert.equal(bounded.a.length,24);assert.equal(bounded.waveXYZ,undefined);assert.equal(bounded.buffer,undefined);assert(JSON.stringify(bounded).includes('depth limit'));
});
await test('Small real 3-D CPU fixture verifies scene, fixed physical units and no network or shadow image',async()=>{
  actualCPUExecutions++;const messages=[],scope={postMessage:m=>messages.push(m)},dispose=installPinholeWorker(scope);await scope.onmessage({data:{id:101,mode:'scene',spectralMode:'narrowband',resolution:16,depthLayers:2,backend:'cpu'}});const r=messages.find(m=>m.type==='result')?.result;assert(r,'Actual worker adapter produced a CPU result');assert(messages.every(m=>m.id===101));dispose();assert.equal(r.width,16);assert.equal(r.configuration.sensorWidthMm,8);assert.equal(r.configuration.detectorDistanceMm,50);assert.equal(r.sceneOverview.objects.length,5);assert(r.waveXYZ.some(v=>v>0));assert(r.geometricXYZ.some(v=>v>0));assert.equal(r.diagnostics.cpuShadow,false);assert.equal(r.diagnostics.cpuReferenceImages,0);assert.equal(networkRequests,0);
});
await test('Worker installer gates cancel, supersession, malformed IDs, errors and disposal',async()=>{
  const messages=[],jobs=[],scope={postMessage:m=>messages.push(m)},execute=(request,callbacks)=>new Promise((resolve,reject)=>jobs.push({request,callbacks,resolve,reject}));const dispose=installPinholeWorker(scope,execute);
  await scope.onmessage({data:{id:0}});await scope.onmessage({data:[]});assert.equal(jobs.length,0);
  const first=scope.onmessage({data:{id:1}});jobs[0].callbacks.progress({message:'first',fraction:0,id:99,type:'result'});assert.equal(messages.at(-1).id,1);assert.equal(messages.at(-1).type,'progress');jobs[0].callbacks.reset({attempt:1,reason:'GPU failure',id:99,type:'result'});assert.equal(messages.at(-1).type,'reset');assert.equal(messages.at(-1).id,1);assert.equal(messages.at(-1).attempt,1);
  const second=scope.onmessage({data:{id:2}});assert(jobs[0].callbacks.signal.aborted);jobs[0].callbacks.preview(preview());jobs[0].callbacks.reset({attempt:1,reason:'stale reset'});jobs[0].resolve(reference);await first;assert.equal(messages.filter(m=>m.type==='result').length,0);
  await scope.onmessage({data:{type:'cancel',id:1}});assert.equal(jobs[1].callbacks.signal.aborted,false);await scope.onmessage({data:{type:'cancel',id:2}});assert(jobs[1].callbacks.signal.aborted);jobs[1].resolve(reference);await second;assert.equal(messages.filter(m=>m.type==='result').length,0);
  const third=scope.onmessage({data:{id:3}});jobs[2].reject(Object.assign(new Error('GPU lost'),{diagnostics:{array:new Float32Array(500),label:'x'.repeat(5000)}}));await third;assert.equal(messages.at(-1).type,'error');assert.equal(messages.at(-1).diagnostics.array,undefined);assert.equal(messages.at(-1).diagnostics.label.length,1000);
  const fourth=scope.onmessage({data:{id:4}});dispose();assert(jobs[3].callbacks.signal.aborted);jobs[3].resolve(reference);await fourth;assert.equal(scope.onmessage,null);assert.equal(messages.at(-1).id,3);
});
const result={generatedAt:new Date().toISOString(),passed:passed.length,failed:failures.length,tests:passed,failures,actualCPUExecutions,networkRequests,cameraImageWrites:draws.length,limitations:['Happy-dom, Canvas spies and Worker doubles; no visual browser or accessibility-tree inspection.','One actual 512² CPU point fixture and a small actual 3-D CPU fixture; no browser WebGPU or physical-GPU assertion.']};
fs.writeFileSync(new URL('../research/pinhole-dom-validation.json',import.meta.url),JSON.stringify(result,null,2)+'\n');
console.log(JSON.stringify({passed:passed.length,failed:failures.length,actualCPUExecutions,networkRequests}));await win.happyDOM.close();if(failures.length)process.exitCode=1;
