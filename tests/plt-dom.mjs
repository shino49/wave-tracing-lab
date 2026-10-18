/** Browser UI control-flow tests: happy-dom + Canvas spies + a module-Worker double.
 * Real CPU runtime fixtures are computed. No browser visual QA or device GPU runs. */
import fs from 'node:fs';
import assert from 'node:assert/strict';
import {executePLTTask} from '../dist/plt-runtime.js';
const {Window}=await import(process.env.WAVE_TEST_DOM||'happy-dom');
const win=new Window({url:'https://test.local/tracer.html',settings:{disableJavaScriptEvaluation:true,disableCSSFileLoading:true,disableJavaScriptFileLoading:true}});
win.document.write(fs.readFileSync(new URL('../dist/tracer.html',import.meta.url),'utf8'));
const calls=[],images=[],contexts=new WeakMap(),livePixels=new WeakMap();
win.HTMLCanvasElement.prototype.getContext=function(){if(!contexts.has(this)){const canvas=this;contexts.set(this,new Proxy({createImageData:(width,height)=>({width,height,data:new Uint8ClampedArray(width*height*4)}),putImageData:image=>{images.push({canvas,image});livePixels.set(canvas,image);},clearRect:(...args)=>{calls.push({canvas,method:'clearRect',args});livePixels.delete(canvas);},measureText:s=>({width:String(s).length*7})},{get:(t,k)=>k in t?t[k]:(...args)=>calls.push({canvas,method:k,args}),set:(t,k,v)=>(t[k]=v,true)}));}return contexts.get(this);};
Object.assign(globalThis,{window:win,document:win.document});Object.defineProperty(globalThis,'navigator',{value:{},configurable:true});
let networkRequests=0;globalThis.fetch=async()=>{networkRequests++;throw new Error('Unexpected network in local OBJ/CPU flow');};
const $=id=>document.getElementById(id),tests=[],failures=[],workers=[],cache=new Map(),downloads=[],blobs=new Map();
let plan='manual',cpuExecutions=0,blobSerial=0;
URL.createObjectURL=blob=>{const url=`blob:test-${++blobSerial}`;blobs.set(url,blob);return url;};URL.revokeObjectURL=url=>blobs.delete(url);
win.HTMLAnchorElement.prototype.click=function(){downloads.push({href:this.href,download:this.download,blob:blobs.get(this.href)});};
const pause=()=>new Promise(r=>setTimeout(r,0));
const until=async fn=>{for(let i=0;i<3000;i++){if(fn())return;await new Promise(r=>setTimeout(r,5));}throw new Error('Timed out waiting for PLT DOM flow');};
const deferred=()=>{let resolve,reject;const promise=new Promise((a,b)=>{resolve=a;reject=b;});return {resolve,reject,promise};};
const change=(id,value)=>{$(id).value=String(value);$(id).dispatchEvent(new win.Event('input',{bubbles:true}));};
const upload=file=>{Object.defineProperty($('pltFile'),'files',{value:[file],configurable:true});$('pltFile').dispatchEvent(new win.Event('change',{bubbles:true}));};
const OBJ='v -1 -1 0\nv 1 -1 0\nv 0 1 0\nf 1 2 3\n';
async function fixture(request){const key=JSON.stringify({...request,id:0});if(!cache.has(key)){cpuExecutions++;cache.set(key,await executePLTTask(request));}return structuredClone(cache.get(key));}
class MockWorker{
 constructor(url,options){if(plan==='factory-error')throw new Error('mock Worker construction failed');this.url=String(url);this.options=options;this.mode=plan;this.terminated=false;this.terminateCalls=0;this.messages=[];workers.push(this);}
 terminate(){this.terminated=true;this.terminateCalls++;}
 postMessage(request){this.request=request;this.messages.push(request);if(this.mode==='post-error')throw new Error('mock postMessage failed');if(this.mode==='manual')return;queueMicrotask(async()=>{try{this.emit({type:'progress',text:'Actual CPU fixture executing',fraction:.25});const result=await fixture(request);if(!this.terminated)this.emit({type:'result',result});}catch(error){if(!this.terminated)this.emit({type:'error',error:error.message});}});}
 // Deliberately deliver even after termination: old queued responses must be ignored.
 emit(data){this.onmessage?.({data:{id:this.request.id,...data}});}
 error(message='mock Worker crashed'){this.onerror?.({message});}
 messageError(){this.onmessageerror?.({});}
}
globalThis.Worker=MockWorker;
await import('../dist/plt-app.js');
const lastWorker=()=>workers.at(-1);
const assertIdle=()=>{assert.equal($('pltRun').disabled,false);assert.equal($('pltCancel').hidden,true);assert.equal($('pltProgress').hidden,true);};
const assertEmpty=()=>{for(const mode of ['Wave','Ray']){assert.equal($(`plt${mode}Empty`).hidden,false);assert.equal(livePixels.has($(`plt${mode}Canvas`)),false);}for(const id of ['pltGeometry','pltEvents','pltMulti','pltParity'])assert.equal($(id).textContent,'—',id);assert.equal($('pltDownload').disabled,true);assert.equal($('pltReport').textContent,'尚无结果');assert.match($('pltMeasurement').textContent,/相机不是几何预览/);};
const start=()=>{plan='manual';$('pltRun').click();return lastWorker();};
const reset=()=>{$('pltReset').click();plan='manual';};
const lowSettings=()=>{change('pltResolution',32);change('pltSamples',4);change('pltDepth',2);};
const report=()=>JSON.parse($('pltReport').textContent);
let reference;
const deliver=(result=reference)=>{const w=start();w.emit({type:'result',result:structuredClone(result)});return w;};
const test=async(name,fn)=>{console.log('TEST:',name);try{await fn();tests.push(name);console.log('PASS');}catch(error){failures.push({name,error:error.stack});console.error('FAIL:',error.message);}finally{reset();}};
await test('Initial page has labelled empty native canvases, no auto-run and old modes remain linked',async()=>{
 assert.equal(workers.length,0);assertIdle();assertEmpty();assert.equal(networkRequests,0);const ids=Array.from(document.querySelectorAll('[id]'),e=>e.id);assert.equal(ids.length,new Set(ids).size);for(const e of document.querySelectorAll('input,select'))assert(e.labels?.length,e.id);for(const c of document.querySelectorAll('canvas'))assert(c.getAttribute('aria-label'));assert($('pltStatus').closest('[aria-live]'));
 for(const href of ['index.html','mesh.html','coherent.html'])assert(document.querySelector(`.experiment-nav a[href="${href}"]`),href);
});
await test('Run, progress and completion drive a real 32×24 CPU runtime through a module Worker request',async()=>{
 lowSettings();plan='execute';$('pltRun').click();const worker=lastWorker();assert.equal($('pltRun').disabled,true);assert.equal($('pltCancel').hidden,false);assert.equal($('pltProgress').hidden,false);assert.equal($('pltDownload').disabled,true);assert.match(worker.url,/plt-worker\.js$/);assert.deepEqual(worker.options,{type:'module'});assert.equal(worker.request.options.width,32);assert.equal(worker.request.options.height,24);assert.equal(worker.request.options.samplesPerPixel,4);assert.equal(worker.request.compare,true);await until(()=>$('pltRun').disabled===false);assert.match($('pltStatus').textContent,/计算完成/);assertIdle();assert.equal(worker.terminated,true);assert.equal(worker.messages.length,1);assert.equal($('pltWaveEmpty').hidden,true);assert.equal($('pltRayEmpty').hidden,true);assert.equal($('pltDownload').disabled,false);assert.match($('pltBackend').textContent,/CPU/);assert.match($('pltWarnings').textContent,/WebGPU/);assert.equal($('pltParity').textContent,'CPU');assert.equal(networkRequests,0);reference=await fixture(worker.request);assert.equal(report().results.length,2);
});
function previewFrame({samples=1,extra=false}={}){
 const width=reference.options.width,height=reference.options.height,n=width*height;
 return {partial:true,width,height,completedSamplesPerPixel:samples,targetSamplesPerPixel:reference.options.samplesPerPixel,results:reference.results.map(r=>{
  const counts=new Uint32Array(n).fill(samples);if(extra)counts[0]++;
  return {wave:r.wave,width,height,image:r.image.slice(),sampleCounts:counts,completedSamplesPerPixel:samples,minSamplesPerPixel:samples,maxSamplesPerPixel:samples+Number(extra),partial:true,backend:r.backend};}),elapsedMs:12,fraction:.25};
}
await test('Verified progressive frames display real arrays and honest partial counts without enabling final evidence',()=>{
 lowSettings();const w=start(),frame=previewFrame({extra:true});w.emit({type:'preview',...frame});assert.equal($('pltRun').disabled,true);assert.equal($('pltWaveEmpty').hidden,true);assert.equal($('pltRayEmpty').hidden,true);assert.equal($('pltDownload').disabled,true);assert.match($('pltWaveSamples').textContent,/1–2.*未完成/);assert.match($('pltMeasurement').textContent,/完成 1 轮/);assert.match($('pltReport').textContent,/尚未产生最终/);
 const before=livePixels.get($('pltWaveCanvas')).data.slice();change('pltExposure',2);assert.notDeepEqual(livePixels.get($('pltWaveCanvas')).data,before);assert.equal($('pltDownload').disabled,true);
 $('pltCancel').click();assertEmpty();const count=images.length;w.emit({type:'preview',...frame});assert.equal(images.length,count);assert.equal($('pltWaveSamples').textContent,'尚未采样');
});
await test('Malformed or falsely complete progressive samples are rejected and clear preview output',()=>{
 for(const mutate of [f=>f.results[0].sampleCounts[0]=9,f=>f.results[1].image[1]=NaN,f=>f.results[1].sampleCounts[1]=0,f=>f.completedSamplesPerPixel=4,f=>f.width+=1]){
  reset();lowSettings();const w=start(),frame=previewFrame();mutate(frame);w.emit({type:'preview',...frame});assertEmpty();assertIdle();assert.match($('pltStatus').textContent,/渐进帧被拒绝/);
 }
});
await test('A parameter change invalidates prior preview and prevents old generation from repainting',()=>{
 reset();lowSettings();const w=start(),frame=previewFrame();w.emit({type:'preview',...frame});change('pltYaw',20);assertEmpty();const count=images.length;w.emit({type:'preview',...frame});assert.equal(images.length,count);assert.equal($('pltDownload').disabled,true);
});
await test('Visible native canvas pixels equal the computed camera arrays, with no geometry drawing proxy',async()=>{
 deliver();for(const r of reference.results){const canvas=$(r.wave?'pltWaveCanvas':'pltRayCanvas'),p=livePixels.get(canvas);assert(p);assert.equal(canvas.width,r.width);assert.equal(canvas.height,r.height);assert.equal(p.data.length,r.image.length*4);for(let i=0;i<r.image.length;i++){const s=Math.round(255*(1-Math.exp(-r.image[i]))**(1/2.2));assert.deepEqual(Array.from(p.data.slice(i*4,i*4+4)),[s,s,s,255]);}}assert.equal(calls.filter(c=>!['clearRect'].includes(c.method)).length,0,'Renderer must use camera arrays rather than geometric proxy drawing');
});
await test('Exposure changes only fixed tone mapping; computed arrays, measurements and request count remain unchanged',async()=>{
 deliver();const before=JSON.stringify(report()),count=workers.length,initial=livePixels.get($('pltWaveCanvas')).data.slice();change('pltExposure',2);assert.equal(JSON.stringify(report()),before);assert.equal(workers.length,count);assert.equal($('pltDownload').disabled,false);assert.equal($('pltExposureValue').textContent,'2 EV');const data=livePixels.get($('pltWaveCanvas')).data;assert.notDeepEqual(data,initial);reference.results[0].image.forEach((v,i)=>assert.equal(data[i*4],Math.round(255*(1-Math.exp(-4*v))**(1/2.2))));
});
await test('Every transport control clears prior images, measurements and downloads and feeds the next request',async()=>{
 const cases=[['pltPreset','prism',r=>r.scene.preset,'prism'],['pltModelYaw',25,r=>r.scene.modelRotation[1],25],['pltScale',.6,r=>r.scene.modelScale,.6],['pltMaterial','2',r=>r.scene.modelMaterial,2],['pltYaw',25,r=>r.camera.yaw,25],['pltPitch',10,r=>r.camera.pitch,10],['pltWavelength',.05,r=>r.options.wavelengthMm,.05],['pltDepth',3,r=>r.options.maxDepth,3],['pltResolution',64,r=>r.options.width,64],['pltSamples',32,r=>r.options.samplesPerPixel,32],['pltCompute','cpu',r=>r.backend,'cpu']];
 for(const [id,value,read,want]of cases){reset();deliver();change(id,value);assertEmpty();assertIdle();const w=start();assert.equal(read(w.request),want,id);$('pltCancel').click();}
});
await test('Active parameter changes terminate work and stale result, progress and errors cannot repaint',async()=>{
 const old=start();change('pltYaw',20);assert.equal(old.terminated,true);assertEmpty();assertIdle();const status=$('pltStatus').textContent,count=images.length;old.emit({type:'progress',text:'stale progress',fraction:.9});old.emit({type:'result',result:reference});old.emit({type:'error',error:'stale error'});old.error();assert.equal($('pltStatus').textContent,status);assert.equal(images.length,count);assertEmpty();
});
await test('Cancel then run accepts only the new generation; repeated run clicks do not spawn extra Workers',async()=>{
 const old=start();const n=workers.length;$('pltRun').click();assert.equal(workers.length,n);old.emit({type:'progress',text:'Working 40%',fraction:.4});assert.equal($('pltProgress').value,.4);assert.equal($('pltStatus').textContent,'Working 40%');$('pltCancel').click();assertIdle();assertEmpty();assert.equal(old.terminated,true);const next=start();const current=$('pltStatus').textContent;old.emit({type:'result',result:reference});old.error();assert.equal($('pltStatus').textContent,current);assert.equal($('pltRun').disabled,true);next.emit({type:'result',result:structuredClone(reference)});assertIdle();assert.equal($('pltDownload').disabled,false);
});
await test('Worker result errors, startup errors and postMessage failures leave no fake or stale image',async()=>{
 for(const kind of ['result-error','worker-error','factory-error','post-error']){reset();deliver();if(kind==='factory-error'||kind==='post-error'){plan=kind;$('pltRun').click();}else{const w=start();if(kind==='result-error')w.emit({type:'error',error:'actual invalid input'});else w.error();}assertIdle();assertEmpty();assert.match($('pltStatus').textContent,/未生成|出错|无法启动/);}
});
await test('Worker message deserialization errors release the busy UI and clear output',async()=>{
 const w=start();w.messageError();assertIdle();assertEmpty();assert.match($('pltStatus').textContent,/出错|消息|无法|失败|错误/);
});
await test('Invalid UI physics input is rejected by real CPU runtime and leaves download disabled',async()=>{
 lowSettings();change('pltWavelength',0);plan='execute';$('pltRun').click();await until(()=>$('pltRun').disabled===false);assertIdle();assertEmpty();assert.match($('pltStatus').textContent,/未生成图像.*Wavelength/);
});
await test('Nonfinite result pixels reject, clear partial rendering and cannot leave completed evidence',async()=>{
 const bad=structuredClone(reference);bad.results[1].image[2]=NaN;deliver(bad);assertIdle();assertEmpty();assert.match($('pltStatus').textContent,/结果被拒绝/);
});
await test('An incomplete progressive frame cannot be labelled as a final result',()=>{
 for(const mutate of [r=>r.progressive.completedSamplesPerPixel=0,r=>r.results[0].sampleCounts[0]=0,r=>r.results[1].partial=true,r=>delete r.results[0].sampleCounts,r=>r.results[0].sampleCounts=new Uint16Array(),r=>r.results[0].sampleCounts=r.results[0].sampleCounts.slice(1),r=>r.results[0].minSamplesPerPixel=0,r=>r.results[0].maxSamplesPerPixel=99]){reset();const bad=structuredClone(reference);mutate(bad);deliver(bad);assertIdle();assertEmpty();assert.match($('pltStatus').textContent,/结果被拒绝/);}
});
await test('Truncated or missing comparison result arrays reject rather than publish an incomplete camera image',async()=>{
 for(const variant of ['short','missing-ray','empty']){reset();const bad=structuredClone(reference);if(variant==='short')bad.results[0].image=bad.results[0].image.slice(0,-1);if(variant==='missing-ray')bad.results=bad.results.slice(0,1);if(variant==='empty')bad.results=[];deliver(bad);assertIdle();assertEmpty();assert.match($('pltStatus').textContent,/结果被拒绝/);}
});
await test('Injected GPU success and rejection badges match the retained camera result',async()=>{
 for(const passed of [true,false]){reset();const r=structuredClone(reference);for(const mode of r.results){mode.backend=passed?'mock verified GPU':'CPU 双精度';mode.validation={passed};}r.warnings=passed?[]:['GPU rejected; retained CPU reference'];deliver(r);assert.equal($('pltParity').textContent,passed?'全部通过':'CPU 回退');assert.equal($('pltBackend').textContent,passed?'mock verified GPU':'CPU 双精度');assert.equal($('pltDownload').disabled,false);}
});
await test('A failed ray-mode GPU gate cannot be presented as all comparison gates passed',async()=>{
 const r=structuredClone(reference);r.results[0].backend='mock verified GPU';r.results[0].validation={passed:true};r.results[1].backend='CPU 双精度';r.results[1].validation={passed:false};r.warnings=['Ray mode GPU rejected; retained CPU reference'];deliver(r);assert.notEqual($('pltParity').textContent,'全部通过');assert.match($('pltWarnings').textContent,/Ray mode GPU rejected/);assert.equal(report().results[1].backend,'CPU 双精度');
});
await test('A valid local OBJ is read without uploads and included unchanged in the next Worker request',async()=>{
 let reads=0;upload({name:'local-triangle.obj',size:OBJ.length,text:async()=>{reads++;return OBJ;}});await until(()=>$('pltClearModel').disabled===false);assert.equal(reads,1);assert.match($('pltFileName').textContent,/local-triangle\.obj.*本地/);const w=start();assert.equal(w.request.scene.objText,OBJ);assert.equal(networkRequests,0);$('pltCancel').click();$('pltClearModel').click();assert.equal($('pltClearModel').disabled,true);assertEmpty();assert.equal(start().request.scene.objText,null);
});
await test('Invalid or oversized OBJ never replaces the model and never uploads',async()=>{
 upload({name:'broken.obj',size:30,text:async()=> 'not OBJ geometry'});await until(()=>/OBJ 未载入/.test($('pltStatus').textContent));assert.equal($('pltClearModel').disabled,true);assert.equal(start().request.scene.objText,null);$('pltCancel').click();let reads=0;upload({name:'big.obj',size:2*1024*1024+1,text:async()=>{reads++;return OBJ;}});assert.match($('pltStatus').textContent,/超过 2 MiB/);assert.equal(reads,0);assert.equal(networkRequests,0);assertEmpty();
});
await test('Parameter change while file.text is pending invalidates the older model load',async()=>{
 const d=deferred();upload({name:'stale.obj',size:OBJ.length,text:()=>d.promise});change('pltPreset','prism');const status=$('pltStatus').textContent;d.resolve(OBJ);await pause();await pause();assert.equal($('pltStatus').textContent,status);assert.equal($('pltClearModel').disabled,true);const w=start();assert.equal(w.request.scene.objText,null);assert.equal(w.request.scene.preset,'prism');assert.equal(networkRequests,0);
});
await test('Reset while OBJ read is pending drops the file and restores all initial parameters',async()=>{
 change('pltYaw',35);change('pltExposure',2);const d=deferred();upload({name:'before-reset.obj',size:OBJ.length,text:()=>d.promise});reset();d.resolve(OBJ);await pause();await pause();assert.equal($('pltClearModel').disabled,true);assert.equal($('pltExposure').value,'0');assert.equal($('pltExposureValue').textContent,'0 EV');assert.equal($('pltYawValue').textContent,'0°');assertEmpty();const w=start();assert.equal(w.request.scene.objText,null);assert.equal(w.request.camera.yaw,0);assert.equal(w.request.scene.preset,'gallery');assert.equal(w.request.options.width,48);assert.equal(w.request.options.samplesPerPixel,16);
});
await test('Second OBJ selection wins even when first file read resolves last',async()=>{
 const first=deferred(),second=deferred();upload({name:'first.obj',size:OBJ.length,text:()=>first.promise});upload({name:'second.obj',size:OBJ.length,text:()=>second.promise});second.resolve(OBJ+'# second\n');await until(()=>$('pltClearModel').disabled===false);first.resolve(OBJ+'# first\n');await pause();assert.match($('pltFileName').textContent,/second\.obj/);assert.equal(start().request.scene.objText,OBJ+'# second\n');
});
await test('Run during pending OBJ read uses only the current accepted state and ignores the stale read',async()=>{
 const d=deferred();upload({name:'pending.obj',size:OBJ.length,text:()=>d.promise});const w=start();assert.equal(w.request.scene.objText,null);d.resolve(OBJ);await pause();await pause();assert.equal($('pltClearModel').disabled,true);w.emit({type:'result',result:structuredClone(reference)});assert.equal($('pltDownload').disabled,false);
});
await test('Stale rejected file read cannot overwrite a newer parameter status',async()=>{
 const d=deferred();upload({name:'rejected-old.obj',size:OBJ.length,text:()=>d.promise});change('pltYaw',30);const status=$('pltStatus').textContent;d.reject(new Error('stale read failed'));await pause();assert.equal($('pltStatus').textContent,status);assert.equal($('pltClearModel').disabled,true);assertEmpty();
});
await test('Reset active work ignores late success and resets exposure and all evidence',async()=>{
 const w=start();change('pltExposure',3);reset();assert.equal(w.terminated,true);w.emit({type:'result',result:reference});w.emit({type:'progress',text:'stale after reset',fraction:1});assertIdle();assertEmpty();assert.equal($('pltExposure').value,'0');assert.match($('pltStatus').textContent,/已恢复默认/);
});
await test('Download is inert before valid results and exports only the current local JSON report',async()=>{
 const count=downloads.length;$('pltDownload').click();assert.equal(downloads.length,count);deliver();const data=report();$('pltDownload').click();assert.equal(downloads.length,count+1);const file=downloads.at(-1);assert.equal(file.download,'wave-tracing-local-run.json');assert.equal(file.blob.type,'application/json');assert.deepEqual(JSON.parse(await file.blob.text()),data);assert(data.results.every(r=>r.image===undefined));change('pltDepth',3);$('pltDownload').click();assert.equal(downloads.length,count+1);
});
await test('Pagehide terminates active Worker and late messages stay ignored',async()=>{
 const w=start();window.dispatchEvent(new win.Event('pagehide'));assert(w.terminated);assertIdle();const count=images.length;w.emit({type:'result',result:reference});assert.equal(images.length,count);assertEmpty();
});
const reportPath=new URL('../research/plt-browser-flow-validation.json',import.meta.url);let validation={};try{validation=JSON.parse(fs.readFileSync(reportPath,'utf8'));}catch{}
validation.generatedAt=new Date().toISOString();validation.limitations=['Happy-dom and Canvas spies only: no browser visual QA, accessibility-tree QA, real browser Worker scheduling, or device rendering.','Injected GPU contracts do not establish WGSL compilation, physical-device execution, or WebGPU browser support.'];validation.dom={passed:tests.length,failed:failures.length,tests,failures,actualCpuFixturesComputed:cache.size,actualCpuTaskInvocations:cpuExecutions,mockWorkersCreated:workers.length,networkRequests,canvasImageWrites:images.length,realCpuFixtureResolution:reference?{width:reference.results[0].width,height:reference.results[0].height,samplesPerPixel:reference.options.samplesPerPixel}:null,nativeComputedImageArraysVerified:tests.includes('Visible native canvas pixels equal the computed camera arrays, with no geometry drawing proxy'),browserVisualQA:false};fs.writeFileSync(reportPath,JSON.stringify(validation,null,2)+'\n');console.log(JSON.stringify({domPassed:tests.length,domFailed:failures.length,actualCpuFixturesComputed:cache.size,actualCpuTaskInvocations:cpuExecutions,networkRequests}));await win.happyDOM.close();if(failures.length)process.exitCode=1;
