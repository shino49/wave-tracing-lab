/** Happy-dom + Canvas spies + Worker double, with actual native CPU fixtures.
 * Does not establish browser layout, browser WebGPU, or physical-device support. */
import fs from 'node:fs';
import assert from 'node:assert/strict';
import {executePLTDemoTask} from '../dist/plt-demo-runtime.js';
import {PLT_DEMO_DEFAULTS} from '../dist/plt-demo-scene.js';
import {installPLTDemoWorker} from '../dist/plt-demo-worker.js';
const {Window}=await import(process.env.WAVE_TEST_DOM||'happy-dom');
const win=new Window({url:'https://test.local/wave-demo.html',settings:{disableJavaScriptEvaluation:true,disableCSSFileLoading:true,disableJavaScriptFileLoading:true}});
win.document.write(fs.readFileSync(new URL('../dist/wave-demo.html',import.meta.url),'utf8'));
const canvases=new WeakMap(),pixels=new WeakMap(),draws=[],calls=[],workers=[],downloads=[],blobs=new Map(),revoked=[],tests=[],failures=[];
win.HTMLCanvasElement.prototype.getContext=function(){if(!canvases.has(this)){const canvas=this;canvases.set(this,new Proxy({createImageData:(width,height)=>({width,height,data:new Uint8ClampedArray(width*height*4)}),putImageData:image=>{pixels.set(canvas,image);draws.push({canvas,image});},clearRect:()=>pixels.delete(canvas)},{get:(t,k)=>k in t?t[k]:(...args)=>calls.push({canvas,method:k,args})}));}return canvases.get(this);};
Object.assign(globalThis,{window:win,document:win.document});Object.defineProperty(globalThis,'navigator',{value:{},configurable:true});
let networkRequests=0,mode='manual',blobId=0,executions=0;
globalThis.fetch=async()=>{networkRequests++;throw new Error('No network should be requested for local CPU demo');};
URL.createObjectURL=blob=>{const url=`blob:demo-${++blobId}`;blobs.set(url,blob);return url;};
URL.revokeObjectURL=url=>{blobs.delete(url);revoked.push(url);};
win.HTMLAnchorElement.prototype.click=function(){downloads.push({url:this.href,name:this.download,blob:blobs.get(this.href)});};
const cache=new Map();
async function fixture(request){const key=JSON.stringify({...request,id:0});if(!cache.has(key)){executions++;cache.set(key,await executePLTDemoTask(request));}return structuredClone(cache.get(key));}
class WorkerDouble{
 constructor(url,options){if(mode==='factory-error')throw new Error('construction failed');this.url=String(url);this.options=options;this.mode=mode;this.terminated=false;this.messages=[];workers.push(this);}
 terminate(){this.terminated=true;}
 postMessage(request){this.messages.push(request);if(request.type==='cancel')return;this.request=structuredClone(request);if(this.mode==='post-error')throw new Error('postMessage failed');if(this.mode!=='execute')return;queueMicrotask(async()=>{try{const result=await fixture(request);if(!this.terminated)this.emit({type:'result',result});}catch(error){if(!this.terminated)this.emit({type:'error',error:error.message});}});}
 emit(data){this.onmessage?.({data:{id:this.request.id,...data}});}
 error(){this.onerror?.({message:'worker failed'});}
 messageError(){this.onmessageerror?.({});}
}
globalThis.Worker=WorkerDouble;
await import('../dist/plt-demo-app.js');
const $=id=>document.getElementById(id),last=()=>workers.at(-1),pause=()=>new Promise(r=>setTimeout(r,0));
const until=async f=>{for(let i=0;i<3000;i++){if(f())return;await new Promise(r=>setTimeout(r,5));}throw new Error('DOM flow timeout');};
function change(id,value){if($(id).type==='checkbox')$(id).checked=value;else $(id).value=String(value);$(id).dispatchEvent(new win.Event('input',{bubbles:true}));}
function fresh(){mode='manual';$('demoReset').click();}
function low(){change('demoResolution',32);change('demoSamples',4);}
function start(){mode='manual';$('demoRun').click();return last();}
function idle(){assert.equal($('demoRun').disabled,false);assert.equal($('demoCancel').hidden,true);assert.equal($('demoProgress').hidden,true);}
function empty(){for(const key of ['Wave','Ray','Difference']){assert.equal(pixels.has($(`demo${key}Canvas`)),false,key);assert.equal($(`demo${key}Empty`).hidden,false,key);}for(const id of ['demoGeometry','demoEvents','demoTime','demoValidation'])assert.equal($(id).textContent,'—',id);assert.equal($('demoDownload').disabled,true);assert.equal($('demoReport').textContent,'尚无结果');}
const report=()=>JSON.parse($('demoReport').textContent);
let reference;
function deliver(value=reference){low();const w=start();w.emit({type:'result',result:structuredClone(value)});return w;}
function preview({samples=1,extra=false}={}){const n=reference.options.width*reference.options.height;return {partial:true,width:reference.options.width,height:reference.options.height,targetSamplesPerPixel:reference.options.samplesPerPixel,completedSamplesPerPixel:samples,results:reference.results.map(r=>({...r,image:r.image.slice(),...(r.utdOffImage?{utdOffImage:r.utdOffImage.slice()}:{}),sampleCounts:Uint32Array.from({length:n},(_,i)=>samples+Number(extra&&i===0)),partial:true,completedSamplesPerPixel:samples,minSamplesPerPixel:samples,maxSamplesPerPixel:samples+Number(extra)}))};}
async function test(name,fn){fresh();try{await fn();tests.push(name);console.log('PASS',name);}catch(error){failures.push({name,error:error.stack});console.error('FAIL',name,error.message);}}
await test('Initial page is manual, labelled, monochrome, with all existing apps linked',()=>{
 assert.equal(workers.length,0);idle();empty();assert.equal(networkRequests,0);const ids=[...document.querySelectorAll('[id]')].map(e=>e.id);assert.equal(ids.length,new Set(ids).size);
 for(const e of document.querySelectorAll('input,select'))assert(e.labels.length,e.id);for(const e of document.querySelectorAll('canvas'))assert(e.getAttribute('aria-label'));assert($('demoStatus').closest('[aria-live]'));
 for(const href of ['wave-demo.html','tracer.html','index.html','mesh.html','coherent.html','resident-test.html'])assert(document.querySelector(`a[href="${href}"]`),href);
 assert.equal(document.querySelector('[aria-current="page"]').getAttribute('href'),'wave-demo.html');assert.equal($('demoDebug').checked,false);assert.equal($('demoExposure').value,'0');assert.equal($('demoWavelength').value,String(PLT_DEMO_DEFAULTS.options.wavelengthMm));
 for(const page of ['index','tracer']){const html=fs.readFileSync(new URL(`../dist/${page}.html`,import.meta.url),'utf8');assert(html.includes('href="wave-demo.html"'));}
});
await test('Default request uses measured native preset, progressive limits and no full parity',()=>{
 const w=start();assert.match(w.url,/plt-demo-worker\.js$/);assert.deepEqual(w.options,{type:'module'});assert.deepEqual(w.request.scene,PLT_DEMO_DEFAULTS.scene);assert.deepEqual(w.request.options,PLT_DEMO_DEFAULTS.options);assert.equal(w.request.utdAblation,true);assert.equal(w.request.debugValidation,false);assert.equal(w.request.compare,true);assert.equal(w.request.progressive.maxRuntimeMs,PLT_DEMO_DEFAULTS.progressive?.maxRuntimeMs??120000);assert.equal($('demoRun').disabled,true);assert.equal($('demoCancel').hidden,false);
 const n=workers.length;$('demoRun').click();assert.equal(workers.length,n);$('demoCancel').click();idle();empty();assert(w.terminated);
});
await test('Real 32×24 CPU demo fixture drives completion and preserves finite safety evidence',async()=>{
 low();mode='execute';$('demoRun').click();const w=last();await until(()=>!$('demoRun').disabled);assert.match($('demoStatus').textContent,/计算完成/);idle();assert(w.terminated);reference=await fixture(w.request);assert.equal(reference.executionMode,'demo');assert(reference.results[0].utdOffImage);assert.equal($('demoDownload').disabled,false);assert.match($('demoBackend').textContent,/CPU/);assert.match($('demoValidation').textContent,/无完整 parity/);assert.equal(report().results[0].validation.parityChecked,false);assert.equal(networkRequests,0);
});
await test('Both native camera canvases are exactly fixed HDR mappings of the actual arrays',()=>{
 deliver();for(const r of reference.results){const canvas=$(r.wave?'demoWaveCanvas':'demoRayCanvas'),p=pixels.get(canvas);assert.equal(canvas.width,r.width);assert.equal(canvas.height,r.height);for(let i=0;i<r.image.length;i++){const v=Math.round(255*Math.min(1,Math.log1p(4*r.image[i])/Math.log1p(32))**(1/2.2));assert.deepEqual([...p.data.slice(i*4,i*4+4)],[v,v,v,255]);}}assert.equal(calls.length,0,'No geometry proxy, painted fringe, or posteffect drawing');
});
await test('Exposure changes only common display mapping, preserves arrays and does not rerun transport',()=>{
 deliver();const n=workers.length,before=report(),initial=pixels.get($('demoWaveCanvas')).data.slice();change('demoExposure',2);assert.equal(workers.length,n);assert.deepEqual(report(),before);assert.notDeepEqual(pixels.get($('demoWaveCanvas')).data,initial);for(const r of reference.results){const p=pixels.get($(r.wave?'demoWaveCanvas':'demoRayCanvas'));for(let i=0;i<r.image.length;i++)assert.equal(p.data[i*4],Math.round(255*Math.min(1,Math.log1p(16*r.image[i])/Math.log1p(32))**(1/2.2)));}
});
await test('Same-path UTD-off selector displays computed control values without new sampling',()=>{
 deliver();const n=workers.length;change('demoReference','utd-off');assert.equal(workers.length,n);assert.match($('demoReferenceTitle').textContent,/UTD Off/);assert.match($('demoReferenceNote').textContent,/直接项振幅/);const p=pixels.get($('demoRayCanvas')),control=reference.results.find(r=>r.wave).utdOffImage;for(let i=0;i<control.length;i++)assert.equal(p.data[i*4],Math.round(255*Math.min(1,Math.log1p(4*control[i])/Math.log1p(32))**(1/2.2)));assert.equal($('demoDownload').disabled,false);
});
await test('Absolute and signed differences use direct array subtraction and fixed shared scale',()=>{
 deliver();const wave=reference.results.find(r=>r.wave),ray=reference.results.find(r=>!r.wave);
 for(const mode of ['absolute-utd','signed-utd','absolute-ray','signed-ray']){change('demoDifferenceMode',mode);const p=pixels.get($('demoDifferenceCanvas')),other=mode.endsWith('utd')?wave.utdOffImage:ray.image;for(let i=0;i<other.length;i++){const d=wave.image[i]-other[i],v=Math.round(255*Math.min(1,Math.log1p(4*Math.abs(d))/Math.log1p(32))**(1/2.2)),want=mode.startsWith('absolute')?[v,v,v,255]:d>=0?[Math.round(v*.4),v,Math.round(v*.9),255]:[v,Math.round(v*.6),Math.round(v*.3),255];assert.deepEqual([...p.data.slice(i*4,i*4+4)],want);}assert.match($('demoDifferenceNote').textContent,mode.endsWith('utd')?/同一批波动路径/:/光束足迹和采样策略/);}
});
await test('Progressive previews show per-mode samples and forbid premature export or difference',()=>{
 low();const w=start(),frame=preview({extra:true});w.emit({type:'preview',...frame});assert.equal($('demoWaveEmpty').hidden,true);assert.match($('demoWaveSamples').textContent,/1–2.*计算中/);assert.equal($('demoDownload').disabled,true);assert.equal($('demoDifferenceEmpty').hidden,false);assert.equal($('demoRun').disabled,true);change('demoReference','utd-off');assert(pixels.has($('demoRayCanvas')));assert.equal($('demoDownload').disabled,true);$('demoCancel').click();empty();idle();
});
await test('Same-path comparison count follows Wave even when the geometric tile is behind',()=>{
 low();change('demoReference','utd-off');const w=start(),f=preview();const wave=f.results.find(r=>r.wave);wave.sampleCounts.fill(2);wave.completedSamplesPerPixel=2;wave.minSamplesPerPixel=2;wave.maxSamplesPerPixel=2;w.emit({type:'preview',...f});assert.match($('demoRaySamples').textContent,/^2 \/ 4 spp/);change('demoReference','geometric');assert.match($('demoRaySamples').textContent,/^1 \/ 4 spp/);$('demoCancel').click();empty();
});
await test('Malformed previews including missing ablation and backwards counts reject atomically',()=>{
 const mutations=[f=>f.results[0].image[0]=NaN,f=>f.results[0].sampleCounts[0]=5,f=>f.results[1].image=f.results[1].image.slice(1),f=>f.completedSamplesPerPixel=3,f=>f.targetSamplesPerPixel=99,f=>delete f.results[0].utdOffImage,f=>f.results[0].utdOffImage[0]=-1,f=>f.results[0].utdOffImage=new Float64Array(1),f=>{f.results[0].sampleCounts[0]=0;f.results[0].image[0]=0;f.results[0].utdOffImage[0]=1;}];
 for(const mutate of mutations){fresh();low();const w=start(),f=preview();mutate(f);w.emit({type:'preview',...f});empty();idle();assert.match($('demoStatus').textContent,/渐进帧被拒绝/);}
 fresh();low();const w=start();w.emit({type:'preview',...preview({samples:2})});w.emit({type:'preview',...preview({samples:1})});empty();assert.match($('demoStatus').textContent,/倒退/);
});
await test('Transport controls invalidate images and their exact values feed the next request',()=>{
 for(const [id,value,read]of [['demoWavelength',.1,r=>r.options.wavelengthMm],['demoYaw',15,r=>r.camera.yaw],['demoResolution',64,r=>r.options.width],['demoSamples',32,r=>r.options.samplesPerPixel],['demoCompute','cpu',r=>r.backend],['demoDebug',true,r=>r.debugValidation]]){fresh();deliver();change(id,value);empty();idle();const w=start();assert.equal(read(w.request),value,id);$('demoCancel').click();}
});
await test('Wavelength input is independently bounded by the scene limits',()=>{
 low();$('demoWavelength').max='.5';change('demoWavelength',.3);const n=workers.length;$('demoRun').click();assert.equal(workers.length,n);empty();idle();assert.match($('demoStatus').textContent,/波长或相机角度无效/);
});
await test('Late progress, previews, result, errors and deserialization after cancel cannot repaint',()=>{
 low();const w=start();$('demoCancel').click();const before=$('demoStatus').textContent,n=draws.length;for(const m of [{type:'progress',fraction:1,text:'stale'},{type:'preview',...preview()},{type:'result',result:reference},{type:'error',error:'stale'}])w.emit(m);w.error();w.messageError();assert.equal($('demoStatus').textContent,before);assert.equal(draws.length,n);empty();idle();
});
await test('Parameter change supersedes active generation and newer run ignores older success',()=>{
 low();const old=start();change('demoYaw',15);assert(old.terminated);empty();change('demoYaw',0);const current=start();old.emit({type:'result',result:reference});assert.equal($('demoRun').disabled,true);current.emit({type:'result',result:structuredClone(reference)});assert.equal($('demoDownload').disabled,false);idle();
});
await test('Reset and pagehide clear active previews and all export state',()=>{
 for(const action of ['reset','pagehide']){fresh();low();const w=start();w.emit({type:'preview',...preview()});if(action==='reset')$('demoReset').click();else window.dispatchEvent(new win.Event('pagehide'));assert(w.terminated);idle();empty();w.emit({type:'result',result:reference});empty();}
});
await test('Worker failure paths release the UI and never retain a success image',()=>{
 for(const kind of ['factory-error','post-error','message','error','returned']){fresh();deliver();if(kind==='factory-error'||kind==='post-error'){mode=kind;$('demoRun').click();}else{const w=start();if(kind==='message')w.messageError();if(kind==='error')w.error();if(kind==='returned')w.emit({type:'error',error:'runtime cap reached'});}empty();idle();assert.match($('demoStatus').textContent,/无法|未生成|出错/);}
});
await test('Nonfinite and unknown progress messages are rejected rather than keeping busy forever',()=>{
 for(const m of [{type:'progress',fraction:NaN,text:'bad'},{type:'progress',fraction:2,text:'bad'},{type:'unexpected'}]){fresh();low();const w=start();w.emit(m);idle();empty();assert.match($('demoStatus').textContent,/拒绝/);}
});
await test('Incomplete, nonfinite, wrong-mode or mismatched final results reject before painting',()=>{
 const mutations=[r=>r.results[0].image[0]=NaN,r=>r.results[1].image=r.results[1].image.slice(1),r=>r.results.pop(),r=>r.results[0].sampleCounts[0]=0,r=>r.results[0].partial=true,r=>r.progressive.completedSamplesPerPixel=0,r=>r.options.width=64,r=>r.options.wavelengthMm=.1,r=>r.scene.preset='gallery',r=>r.executionMode='debug-strict',r=>r.results[0].validation.parityChecked=true,r=>r.results[0].validation.passed=false,r=>r.results[0].validation.mode='parity',r=>delete r.results[0].execution,r=>delete r.results[0].utdOffImage,r=>r.results[0].utdOffImage[0]=Infinity];
 for(const mutate of mutations){fresh();const r=structuredClone(reference);mutate(r);const n=draws.length;deliver(r);empty();idle();assert.equal(draws.length,n);assert.match($('demoStatus').textContent,/结果被拒绝/);}
});
await test('Oversized, cyclic and nonfinite diagnostic metadata cannot create downloadable success',()=>{
 for(const mutate of [r=>r.metadata={huge:'x'.repeat(65537)},r=>r.metadata={bad:Infinity},r=>r.metadata=r,r=>r.metadata={huge:Array(4097).fill(0)}]){fresh();const r=structuredClone(reference);mutate(r);const n=draws.length;deliver(r);empty();assert.equal(draws.length,n);assert.match($('demoStatus').textContent,/拒绝/);}
});
await test('CPU, GPU-assisted and mixed labels preserve no-parity distinction without claiming physical GPU',()=>{
 for(const label of ['CPU 原生 3D UTD','CPU 遍历与采样 + WebGPU UTD 与相机','混合：已完成 GPU 分块 + CPU 后续分块']){fresh();const r=structuredClone(reference);for(const result of r.results)result.backend=label;deliver(r);assert.equal($('demoBackend').textContent,label);assert.match($('demoValidation').textContent,/无完整 parity/);assert.match($('demoWarnings').textContent,/不推断适配器是物理 GPU/);assert.equal($('demoDownload').disabled,false);}
});
await test('Explicit DEBUG requests strict execution, disables ablation and labels CPU-only honestly',async()=>{
 low();change('demoDebug',true);mode='execute';$('demoRun').click();const w=last();assert.equal(w.request.debugValidation,true);assert.equal(w.request.utdAblation,false);await until(()=>!$('demoRun').disabled);assert.match($('demoStatus').textContent,/计算完成/);assert.equal(report().executionMode,'debug-strict');assert.match($('demoValidation').textContent,/无 GPU parity/);assert.equal($('demoReference').value,'geometric');assert.equal([...$('demoReference').options].find(o=>o.value==='utd-off').disabled,true);assert.equal($('demoDifferenceMode').value,'absolute-ray');
});
await test('Download only exports bounded current JSON metadata and pagehide revokes pending URLs',async()=>{
 const n=downloads.length;$('demoDownload').click();assert.equal(downloads.length,n);deliver();const expected=report();$('demoDownload').click();assert.equal(downloads.length,n+1);const file=downloads.at(-1);assert.equal(file.name,'native-wave-demo-local-run.json');assert.equal(file.blob.type,'application/json');assert.deepEqual(JSON.parse(await file.blob.text()),expected);assert(expected.results.every(r=>!r.image&&!r.utdOffImage&&!r.sampleCounts));assert.equal(expected.display.sharedExposureAtCompletionEV,0);window.dispatchEvent(new win.Event('pagehide'));assert(revoked.includes(file.url));empty();$('demoDownload').click();assert.equal(downloads.length,n+1);
});
await test('Worker installer cancels superseded and aborted work, ignores malformed IDs and disposes',async()=>{
 const messages=[],jobs=[],scope={postMessage:message=>messages.push(message)},execute=(request,callbacks)=>new Promise((resolve,reject)=>jobs.push({request,callbacks,resolve,reject}));
 const dispose=installPLTDemoWorker(scope,execute);
 await scope.onmessage({data:{id:0}});assert.equal(jobs.length,0);
 const first=scope.onmessage({data:{id:1,options:{}}});jobs[0].callbacks.progress({text:'first',fraction:0});assert.equal(messages.at(-1).id,1);
 const second=scope.onmessage({data:{id:2,options:{}}});assert(jobs[0].callbacks.signal.aborted);jobs[0].callbacks.preview({partial:true});jobs[0].resolve({old:true});await first;assert.equal(messages.filter(m=>m.type==='result').length,0);
 await scope.onmessage({data:{type:'cancel',id:1}});assert.equal(jobs[1].callbacks.signal.aborted,false);await scope.onmessage({data:{type:'cancel',id:2}});assert(jobs[1].callbacks.signal.aborted);jobs[1].resolve({stale:true});await second;assert.equal(messages.filter(m=>m.type==='result').length,0);
 const third=scope.onmessage({data:{id:3,options:{}}});jobs[2].resolve({ok:true});await third;assert.equal(messages.at(-1).result.ok,true);
 const fourth=scope.onmessage({data:{id:4,options:{}}});dispose();assert(jobs[3].callbacks.signal.aborted);jobs[3].resolve({late:true});await fourth;assert.equal(scope.onmessage,null);assert.equal(messages.at(-1).id,3);
});
const result={generatedAt:new Date().toISOString(),passed:tests.length,failed:failures.length,tests,failures,actualCpuFixtures:cache.size,actualCpuExecutions:executions,networkRequests,canvasImageWrites:draws.length,limitations:['Happy-dom, Canvas spies and module Worker doubles; no visual browser or accessibility-tree QA.','Actual CPU runtime fixtures exercised. Injected UI GPU labels are contract checks, not browser WebGPU or physical-device evidence.']};
fs.writeFileSync(new URL('../research/plt-demo-dom-validation.json',import.meta.url),JSON.stringify(result,null,2)+'\n');console.log(JSON.stringify({passed:tests.length,failed:failures.length,actualCpuExecutions:executions,networkRequests}));await win.happyDOM.close();if(failures.length)process.exitCode=1;
