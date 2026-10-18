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
const {displaySpectralXYZ}=await import('../dist/plt-spectrum.js');
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

const visibleSettings=()=>{change('pltSpectrum','visible');change('pltPreset','micro-gallery');lowSettings();change('pltSource','blackbody3000');};
await test('Spectral controls preserve explicit infrared monochrome default and label units',()=>{
 assert.equal($('pltSpectrum').value,'monochrome');assert.equal($('pltSource').disabled,true);assert.equal($('pltSpectralMaterial').disabled,true);assert.equal($('pltWavelength').disabled,false);assert.match($('pltSpectralNote').textContent,/25 µm.*红外/);assert(document.querySelector('a[href="spectral-method.html"]'));assert($('pltSource').querySelector('option[value="blackbody3000"]'));
});
await test('Visible micro preset is explicit, bounded and does not silently auto-run',()=>{
 const n=workers.length;$('pltVisiblePreset').click();assert.equal(workers.length,n);assert.equal($('pltSpectrum').value,'visible');assert.equal($('pltPreset').value,'micro-gallery');assert.equal($('pltSamples').value,'64');assert.equal($('pltSource').disabled,false);assert.equal($('pltWavelength').disabled,true);assert.match($('pltSpectralNote').textContent,/240 µm.*52 µm/);assertEmpty();const w=start();assert.equal(w.request.options.spectralMode,'visible');assert.equal(w.request.scene.preset,'micro-gallery');
});
await test('Real visible CPU runtime produces colored native canvas pixels exactly from XYZ',async()=>{
 visibleSettings();plan='execute';$('pltRun').click();const worker=lastWorker();await until(()=>$('pltRun').disabled===false);assert.match($('pltStatus').textContent,/计算完成/);reference=await fixture(worker.request);assert(reference.results.every(r=>r.xyz.length===r.width*r.height*3));assert.equal(reference.options.spectralSource,'blackbody3000');let colored=0;
 for(const r of reference.results){const painted=livePixels.get($(r.wave?'pltWaveCanvas':'pltRayCanvas'));for(let i=0;i<r.image.length;i++){const expected=displaySpectralXYZ(r.xyz.subarray(3*i,3*i+3),0).rgb,actual=Array.from(painted.data.slice(4*i,4*i+4));assert.deepEqual(actual,[...expected,255]);colored+=expected[0]!==expected[1]||expected[1]!==expected[2];}}
 assert(colored>100);assert.equal(networkRequests,0);assert.match($('pltWarnings').textContent,/负的线性 sRGB.*显示/);assert(report().results.every(r=>r.spectralNoise&&r.displayGamut&&!r.xyz));
});
await test('Exposure changes only visible display, leaving XYZ, noise and report untouched',()=>{
 visibleSettings();deliver();const before=JSON.stringify(report()),requests=workers.length,initial=livePixels.get($('pltWaveCanvas')).data.slice();change('pltExposure',1.5);assert.equal(JSON.stringify(report()),before);assert.equal(workers.length,requests);assert.notDeepEqual(livePixels.get($('pltWaveCanvas')).data,initial);const r=reference.results[0];r.image.forEach((v,i)=>assert.deepEqual(Array.from(livePixels.get($('pltWaveCanvas')).data.slice(4*i,4*i+3)),displaySpectralXYZ(r.xyz.subarray(3*i,3*i+3),1.5).rgb));
});
await test('All spectral controls invalidate result and update next request',()=>{
 for(const [id,value,read,want]of [['pltSource','E',r=>r.options.spectralSource,'E'],['pltSpectralMaterial','blue',r=>r.options.spectralMaterial,'blue'],['pltSourcePower',2,r=>r.scene.emission,2],['pltSpectrum','monochrome',r=>r.options.spectralMode,'monochrome']]){reset();visibleSettings();deliver();change(id,value);assertEmpty();const w=start();assert.equal(read(w.request),want);}
});
await test('Invalid X or Z and inconsistent Y never paint a completed spectral result',()=>{
 for(const mutate of [r=>r.results[0].xyz[0]=NaN,r=>r.results[0].xyz[2]=-1,r=>r.results[1].xyz=r.results[1].xyz.slice(1),r=>delete r.results[1].xyz,r=>r.results[0].xyz[1]+=1,r=>r.results[0].spectralMode='monochrome']){reset();visibleSettings();const bad=structuredClone(reference);mutate(bad);deliver(bad);assertEmpty();assertIdle();assert.match($('pltStatus').textContent,/结果被拒绝/);}
});
function colorPreview(){return {partial:true,width:reference.options.width,height:reference.options.height,completedSamplesPerPixel:1,results:reference.results.map(r=>({...r,partial:true,sampleCounts:new Uint16Array(r.image.length).fill(1),completedSamplesPerPixel:1,minSamplesPerPixel:1,maxSamplesPerPixel:1}))};}
await test('Spectral preview respects cancellation, sample counts and stale generations',()=>{
 visibleSettings();const w=start(),frame=colorPreview();w.emit({type:'preview',...frame});assert.equal($('pltWaveEmpty').hidden,true,$('pltStatus').textContent);assert.equal($('pltDownload').disabled,true);$('pltCancel').click();assertEmpty();const n=images.length;w.emit({type:'preview',...frame});assert.equal(images.length,n);visibleSettings();const next=start();const bad=colorPreview();bad.results[1].xyz[2]=NaN;next.emit({type:'preview',...bad});assertEmpty();assertIdle();assert.match($('pltStatus').textContent,/渐进帧被拒绝/);
});
await test('Reset returns to scalar research and rejects a mismatched visible completion',()=>{
 visibleSettings();const old=start();$('pltReset').click();assert.equal(old.terminated,true);assert.equal($('pltSpectrum').value,'monochrome');assert.equal($('pltWavelength').disabled,false);assert.equal($('pltSource').disabled,true);assertEmpty();const w=start();w.emit({type:'result',result:structuredClone(reference)});assertEmpty();assert.match($('pltStatus').textContent,/结果被拒绝.*光谱模式/);
});
await test('Negative source input is rejected by actual visible runtime and never finalized',async()=>{
 visibleSettings();change('pltSourcePower',-1);plan='execute';$('pltRun').click();await until(()=>$('pltRun').disabled===false);assertEmpty();assertIdle();assert.match($('pltStatus').textContent,/未生成图像/);
});
const reportData={generatedAt:new Date().toISOString(),scope:'Mocked DOM/Canvas control flow with actual CPU spectral runtime; not browser visual QA',passed:tests.length,failed:failures.length,tests,failures,actualCpuFixturesComputed:cpuExecutions,networkRequests};
fs.writeFileSync(new URL('../research/plt-spectral-dom-validation.json',import.meta.url),JSON.stringify(reportData,null,2));console.log(JSON.stringify({passed:tests.length,failed:failures.length,networkRequests}));win.happyDOM.abort();if(failures.length)process.exitCode=1;
