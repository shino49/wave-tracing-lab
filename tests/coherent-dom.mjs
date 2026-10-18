/** Control-flow QA using happy-dom, a Canvas spy and a module-Worker test double.
 * CPU fixtures execute the real runtime. This does not validate browser pixels,
 * browser Worker scheduling, WebGPU compilation or physical-device execution. */
import fs from 'node:fs';
import assert from 'node:assert/strict';
import {executeCoherentTask} from '../dist/coherent-runtime.js';
const {Window}=await import(process.env.WAVE_TEST_DOM||'happy-dom');
const win=new Window({url:'https://test.local/coherent.html',settings:{disableJavaScriptEvaluation:true,disableCSSFileLoading:true,disableJavaScriptFileLoading:true}});
win.document.write(fs.readFileSync(new URL('../dist/coherent.html',import.meta.url),'utf8'));
const calls=[],images=[],contexts=new WeakMap();
win.HTMLCanvasElement.prototype.getContext=function(){
 if(!contexts.has(this)){const canvas=this;contexts.set(this,new Proxy({createImageData:(width,height)=>({width,height,data:new Uint8ClampedArray(width*height*4)}),putImageData:image=>images.push({canvas,image}),measureText:s=>({width:String(s).length*7})},{get:(t,k)=>k in t?t[k]:(...args)=>calls.push({canvas,method:k,args}),set:(t,k,v)=>(t[k]=v,true)}));}
 return contexts.get(this);
};
Object.assign(globalThis,{window:win,document:win.document,__COHERENT_TEST__:true});
Object.defineProperty(globalThis,'navigator',{value:{},configurable:true});
let fetches=0;globalThis.fetch=async()=>{fetches++;throw new Error('No network is permitted in CPU DOM flow');};
const $=id=>document.getElementById(id),tests=[],failures=[],workers=[],cpuCache=new Map();
let plan='execute',cpuExecutions=0;
const pause=()=>new Promise(resolve=>setTimeout(resolve,0));
const until=async fn=>{for(let i=0;i<1000;i++){if(fn())return;await new Promise(r=>setTimeout(r,5));}throw new Error('Timed out waiting for DOM flow');};
const deferred=()=>{let resolve,reject;const promise=new Promise((a,b)=>{resolve=a;reject=b;});return {promise,resolve,reject};};
const change=(id,value)=>{$(id).value=String(value);$(id).dispatchEvent(new win.Event($(id).tagName==='SELECT'?'change':'input',{bubbles:true}));};
async function fixture(request){const key=JSON.stringify({action:request.action,options:request.options,backend:request.backend});if(!cpuCache.has(key)){cpuExecutions++;cpuCache.set(key,await executeCoherentTask(request));}return structuredClone(cpuCache.get(key));}
class MockWorker{
 constructor(mode){this.mode=mode;this.terminated=false;this.terminateCalls=0;this.messages=[];workers.push(this);}
 terminate(){this.terminated=true;this.terminateCalls++;}
 postMessage(request){this.request=request;this.messages.push(request);if(this.mode==='post-error')throw new Error('mock postMessage failed');if(this.mode==='manual')return;queueMicrotask(async()=>{
  try{this.emit({type:'progress',text:'mock Worker executing actual CPU reference'});const result=await fixture(request);if(!this.terminated)this.emit({type:'result',result});}catch(error){if(!this.terminated)this.emit({type:'error',error:error.message});}
 });}
 // Deliberately allows delivery after termination to test stale queued events.
 emit(data){this.onmessage?.({data:{id:this.request.id,...data}});}
 error(message='mock Worker startup error'){this.onerror?.({message});}
 messageError(){this.onmessageerror?.({});}
}
globalThis.__COHERENT_WORKER_FACTORY__=()=>{if(plan==='factory-error')throw new Error('mock Worker construction failed');return new MockWorker(plan);};
await import('../dist/coherent-app.js');
const lastWorker=()=>workers.at(-1);
const assertIdle=()=>{assert.equal(coherentLab.getState().running,false);assert.equal($('coherentRun').disabled,false);assert.equal($('coherentCancel').disabled,true);assert.equal($('coherentField').getAttribute('aria-busy'),'false');};
const metrics=['detectorSignal','incoherentSignal','outgoingRatio','solveResidual','referenceError','solveTime','cpuTime','gpuTime','totalTime'];
const assertClearMetrics=()=>{for(const id of metrics)assert.equal($(id).textContent,'—',id);assert.doesNotMatch($('gpuParity').textContent,/^通过/);assert.equal($('refinementError').textContent,'尚未运行');};
const assertNoResult=()=>{assert(coherentLab.getState().result===null,'No field result should remain');assert.equal(coherentLab.getState().backend,undefined);assert.equal($('coherentPending').hidden,false);assert.equal($('refineCheck').disabled,true);assertClearMetrics();};
const assertClearDiagnostic=()=>{assert.equal(coherentLab.getDiagnostic(),null);assert.equal(globalThis.coherentLabDiagnostic??null,null);assert.equal($('diagnosticText').value,'');};
const actual=async()=>{plan='execute';const result=await coherentLab.run();assert(result);assertIdle();return result;};
const startManual=()=>{plan='manual';const promise=coherentLab.run();return {promise,worker:lastWorker()};};
const hybrid=result=>({...structuredClone(result),backend:'CPU + WebGPU',parity:{passed:true,points:result.grid.mask.length},timings:{...result.timings,gpuMs:14.5},warnings:[]});
const test=async(name,fn)=>{console.log('TEST:',name);try{await fn();tests.push(name);console.log('PASS');}catch(error){console.log('FAIL:',error.message.slice(0,400));failures.push({name,error:error.stack?.slice(0,12000)});}finally{coherentLab.reset();plan='execute';delete navigator.clipboard;}};

await test('Initial markup is labelled and empty without auto-start under the test flag',async()=>{
 assert.equal(workers.length,0);assertNoResult();assert.equal($('coherentError').hidden,true);assert.equal($('coherentBackend').textContent,'待运行');assert.equal($('coherentCancel').disabled,true);assert.equal($('refineCheck').disabled,true);
 const ids=Array.from(document.querySelectorAll('[id]'),e=>e.id);assert.equal(new Set(ids).size,ids.length);for(const e of $('coherentForm').querySelectorAll('input,select'))assert(document.querySelector(`label[for="${e.id}"]`),e.id);
 for(const c of document.querySelectorAll('canvas'))assert(c.getAttribute('aria-label'));assert.equal($('coherentError').getAttribute('role'),'alert');assert.equal($('coherentStatus').getAttribute('role'),'status');assert.equal(fetches,0);
 assert.equal(coherentLab.getState().options.detectorHalfAngle,.55);assert.equal($('detectorHalfAngle').min,'.1');assert.equal($('detectorHalfAngle').max,'.6');assert.equal($('showPhase').disabled,false);
});
await test('Calculate form submits a Worker request and renders actual CPU fields and diagnostics',async()=>{
 const before=workers.length;$('coherentForm').dispatchEvent(new win.Event('submit',{bubbles:true,cancelable:true}));assert.equal(workers.length,before+1);assert.equal(coherentLab.getState().running,true);assert.equal($('coherentRun').disabled,true);assert.equal($('coherentCancel').disabled,false);assert.equal($('coherentField').getAttribute('aria-busy'),'true');
 await until(()=>!coherentLab.getState().running);assertIdle();const r=coherentLab.getState().result;assert(r);assert.equal(r.backend,'CPU 双精度');assert.equal($('coherentBackend').textContent,'CPU 双精度');assert.equal($('coherentPending').hidden,true);assert.equal($('coherentStale').hidden,true);assert.equal($('detectorSignal').textContent,r.measurement.detectorFlux.toPrecision(4));assert.equal($('gpuTime').textContent,'—');assert.match($('gpuParity').textContent,/CPU 参考/);assert.match($('coherentWarnings').textContent,/WebGPU/);
 assert(lastWorker().terminated);assert.equal(lastWorker().messages.length,1);assert.equal(lastWorker().request.action,'run');assert.equal(lastWorker().request.backend,'auto');assert.equal($('refineCheck').disabled,false);assert.equal(coherentLabDiagnostic.backend,'CPU 双精度');assert.deepEqual(JSON.parse($('diagnosticText').value),coherentLab.getDiagnostic());assert.equal(fetches,0);
});
await test('Canvas spy sees actual mask pixels, alpha, geometry and angular plot calls',async()=>{
 const r=await actual(),image=images.at(-1).image;assert.equal(image.data.length,144*108*4);let masks=0;
 for(let i=0;i<r.grid.mask.length;i++){assert.equal(image.data[i*4+3],255);if(!r.grid.mask[i]){masks++;assert([[51,56,54],[42,47,45]].some(rgb=>rgb.every((v,j)=>image.data[i*4+j]===v)));}}
 assert.equal(masks,r.grid.diagnostics.maskedPixels);assert(calls.some(c=>c.canvas.id==='coherentField'&&c.method==='drawImage'));assert(calls.some(c=>c.canvas.id==='coherentField'&&c.method==='fillText'&&c.args[0]==='B'));assert(calls.some(c=>c.canvas.id==='coherentAngular'&&c.method==='setLineDash'&&c.args[0].length));assert.match($('fieldNotice').textContent,new RegExp(String(masks)));assert.match($('fieldScale').textContent,/dB/);
});
await test('Intensity and pure phase view switches never change numerical outputs or start work',async()=>{
 const r=await actual(),before=workers.length,signal=$('detectorSignal').textContent;$('showPhase').click();assert.equal(coherentLab.getState().view,'phase');assert.equal($('showPhase').getAttribute('aria-pressed'),'true');assert.match($('fieldScale').textContent,/相位|arg/);assert.equal($('detectorSignal').textContent,signal);assert(coherentLab.getState().result===r,'Current result identity changed');
 $('showIntensity').click();assert.equal(coherentLab.getState().view,'intensity');assert.equal($('showIntensity').getAttribute('aria-pressed'),'true');assert.equal(workers.length,before);assert.equal($('detectorSignal').textContent,signal);
});
await test('Forward and full-angle views redraw only the chart and preserve full-angle numerical evidence',async()=>{
 const r=await actual(),count=workers.length,diagnostic=JSON.stringify(coherentLab.getDiagnostic()),signal=$('detectorSignal').textContent;assert.equal(coherentLab.getState().angleView,'forward');
 for(const [id,view]of [['showFull','full'],['showForward','forward']]){$(id).click();assert.equal(coherentLab.getState().angleView,view);assert.equal($(id).getAttribute('aria-pressed'),'true');assert(coherentLab.getState().result===r,'Angular zoom changed calculated data');assert.equal($('detectorSignal').textContent,signal);assert.equal(JSON.stringify(coherentLab.getDiagnostic()),diagnostic);assert.equal(workers.length,count);}
});
await test('Partial and zero coherence disable phase display and return an active phase view to intensity',async()=>{
 await actual();$('showPhase').click();for(const value of [.5,0]){change('gamma',value);assert.equal($('showPhase').disabled,true);assert.equal(coherentLab.getState().view,'intensity');$('showPhase').click();assert.equal(coherentLab.getState().view,'intensity');}
 change('gamma',1);assert.equal($('showPhase').disabled,true);await actual();assert.equal($('showPhase').disabled,false);
});
await test('A stale mixed-state field cannot gain a phase display merely by editing gamma to one',async()=>{
 change('gamma',.5);await actual();assert.equal($('showPhase').disabled,true);change('gamma',1);assert.equal($('showPhase').disabled,true);$('showPhase').click();assert.equal(coherentLab.getState().view,'intensity');await actual();assert.equal($('showPhase').disabled,false);$('showPhase').click();assert.equal(coherentLab.getState().view,'phase');
});
await test('All parameter inputs feed correctly converted Worker options and mark old metrics stale',async()=>{
 for(const [id,value,key,want]of [['screenCount',1,'screenCount',1],['separation',2.5,'separation',2.5],['offset',.6,'offset',.6],['gamma',.4,'gamma',.4],['phase',90,'phase',Math.PI/2],['inputPower',1.4,'inputPower',1.4],['detectorCenter',30,'detectorCenter',Math.PI/6],['detectorHalfAngle',.4,'detectorHalfAngle',.4],['resolution',96,'n',96],['backend','cpu',null,null]]){
  coherentLab.reset();await actual();change(id,value);assert.equal(coherentLab.getState().stale,true,id);assert.equal($('coherentStale').hidden,false,id);assertClearMetrics();assertClearDiagnostic();assert.equal($('refineCheck').disabled,true);assert.equal($('coherentBackend').textContent,'待重新计算');
  const {promise,worker}=startManual();if(key)assert(Math.abs(worker.request.options[key]-want)<1e-12,id);else assert.equal(worker.request.backend,'cpu');coherentLab.cancel();await promise;
 }
});
await test('Cancel terminates active work and late result, progress or error cannot repaint it',async()=>{
 const {promise,worker}=startManual();$('coherentCancel').click();assert.equal(await promise,null);assert(worker.terminated);assertIdle();assertNoResult();assertClearDiagnostic();const status=$('coherentStatus').textContent;
 worker.emit({type:'progress',text:'stale progress'});worker.emit({type:'result',result:await fixture(worker.request)});worker.error('stale error');worker.messageError();assertNoResult();assert.equal($('coherentStatus').textContent,status);assert.equal($('coherentError').hidden,true);
});
await test('Parameter change cancels a pending solve while preserving only the explicitly stale previous image',async()=>{
 const prior=await actual(),{promise,worker}=startManual();change('phase',30);assert.equal(await promise,null);assert(worker.terminated);assertIdle();assert(coherentLab.getState().result===prior,'Previous result identity changed');assert.equal($('coherentStale').hidden,false);assertClearMetrics();assertClearDiagnostic();worker.emit({type:'result',result:await fixture(worker.request)});assert(coherentLab.getState().result===prior,'Previous result identity changed');assertClearMetrics();
});
await test('Reset during a pending calculation restores exact defaults and blocks obsolete output',async()=>{
 change('gamma',.4);change('phase',50);change('separation',2.8);change('backend','cpu');const {promise,worker}=startManual();$('coherentReset').click();assert.equal(await promise,null);assertIdle();assertNoResult();assertClearDiagnostic();assert.equal(coherentLab.getState().options.gamma,1);assert.equal(coherentLab.getState().options.phase,0);assert.equal(coherentLab.getState().options.detectorHalfAngle,.55);assert.equal(coherentLab.getState().options.separation,2.3);assert.equal($('backend').value,'auto');assert.equal(coherentLab.getState().view,'intensity');worker.emit({type:'result',result:await fixture(worker.request)});assertNoResult();
});
await test('Repeated calculate calls replace the worker and out-of-order completion only publishes latest',async()=>{
 const first=startManual(),second=startManual();assert.equal(await first.promise,null);assert(first.worker.terminated);const latest=await fixture(second.worker.request);second.worker.emit({type:'result',result:latest});assert(await second.promise===latest,'Latest promise returned a different result');assertIdle();const state=coherentLab.getState().result;
 first.worker.emit({type:'result',result:hybrid(latest)});first.worker.emit({type:'progress',text:'wrong old job'});first.worker.error();assert(coherentLab.getState().result===state,'Latest result identity changed');assert.equal($('coherentBackend').textContent,'CPU 双精度');assert.equal($('coherentError').hidden,true);
});
await test('Disabled calculate button cannot create a second worker on repeated ordinary clicks',async()=>{
 // happy-dom 20 flags valid decimal range steps due to modulo roundoff.
 // Bypass only that mocked HTML constraint path; actual limits are runtime-tested.
 $('coherentForm').noValidate=true;try{plan='manual';$('coherentRun').click();const count=workers.length;assert.equal(coherentLab.getState().running,true);$('coherentRun').click();$('coherentRun').click();assert.equal(workers.length,count);coherentLab.cancel();assertIdle();}finally{$('coherentForm').noValidate=false;}
});
await test('Wrong response ID is ignored until the current Worker completes',async()=>{
 const {promise,worker}=startManual(),r=await fixture(worker.request);worker.emit({id:worker.request.id+999,type:'result',result:r});assert.equal(coherentLab.getState().running,true);assertNoResult();worker.emit({type:'progress',text:'current progress'});assert.equal($('coherentStatus').textContent,'current progress');worker.emit({type:'result',result:r});assert(await promise===r,'Promise result identity differs');assertIdle();
});
await test('Completed worker cannot change a published result through duplicate completion or late errors',async()=>{
 const {promise,worker}=startManual(),r=await fixture(worker.request);worker.emit({type:'result',result:r});await promise;const status=$('coherentStatus').textContent;worker.emit({type:'progress',text:'late completed progress'});worker.emit({type:'result',result:hybrid(r)});worker.error('late completed error');worker.messageError();assert(coherentLab.getState().result===r,'Current result identity changed');assert.equal($('coherentStatus').textContent,status);assert.equal($('coherentError').hidden,true);
});
await test('Worker result-error clears failed evidence and exposes a locally inspectable failure',async()=>{
 await actual();const {promise,worker}=startManual();worker.emit({type:'error',error:'mock rejected solve',diagnostic:{stage:'solve'}});assert.equal(await promise,null);assertIdle();assertNoResult();assert.match($('coherentError').textContent,/mock rejected solve/);assert.equal($('coherentError').hidden,false);assert.equal($('coherentBackend').textContent,'停止');assert.equal(coherentLab.getDiagnostic().detail.stage,'solve');
});
await test('Worker construction, postMessage, startup and message decoding errors leave reusable controls',async()=>{
 for(const mode of ['factory-error','post-error','startup','decode']){coherentLab.reset();plan=['startup','decode'].includes(mode)?'manual':mode;const running=coherentLab.run();if(mode==='startup')lastWorker().error();if(mode==='decode')lastWorker().messageError();assert.equal(await running,null);assertIdle();assertNoResult();assert.equal($('coherentError').hidden,false);assert.equal($('coherentBackend').textContent,'停止');}
 await actual();assert.equal($('coherentError').hidden,true);
});
await test('Unavailable module Worker reports a visible blocker with no main-thread compute fallback',async()=>{
 const factory=globalThis.__COHERENT_WORKER_FACTORY__,Worker=globalThis.Worker;delete globalThis.__COHERENT_WORKER_FACTORY__;delete globalThis.Worker;
 try{const count=workers.length;assert.equal(await coherentLab.run(),null);assertIdle();assertNoResult();assert.match($('coherentError').textContent,/模块 Worker/);assert.equal(workers.length,count);}finally{globalThis.__COHERENT_WORKER_FACTORY__=factory;if(Worker!==undefined)globalThis.Worker=Worker;}
});
await test('Unknown Worker message fails closed instead of leaving controls stuck',async()=>{
 const {promise,worker}=startManual();worker.emit({type:'unknown'});assert.equal(await promise,null);assertIdle();assertNoResult();assert.match($('coherentError').textContent,/未知消息/);
});
await test('Passing hybrid and rejected GPU CPU fallback receive distinct badges, times and parity text',async()=>{
 let task=startManual(),r=hybrid(await fixture(task.worker.request));task.worker.emit({type:'result',result:r});await task.promise;assert.equal($('coherentBackend').textContent,'CPU + WebGPU');assert.match($('gpuParity').textContent,/^通过.*15552/);assert.equal($('gpuTime').textContent,'14.5 ms');assert.equal(coherentLabDiagnostic.parity.passed,true);
 task=startManual();r={...await fixture(task.worker.request),parity:{passed:false},warnings:['mock GPU rejected; CPU fallback']};task.worker.emit({type:'result',result:r});await task.promise;assert.equal($('coherentBackend').textContent,'CPU 双精度');assert.match($('gpuParity').textContent,/未通过.*CPU/);assert.equal($('gpuTime').textContent,'—');assert.match($('coherentWarnings').textContent,/mock GPU rejected/);assert.equal(coherentLabDiagnostic.parity.passed,false);
});
await test('Changing parameters, cancel and reset remove prior hybrid parity and diagnostic evidence',async()=>{
 for(const operation of ['change','cancel','reset']){coherentLab.reset();const task=startManual(),r=hybrid(await fixture(task.worker.request));task.worker.emit({type:'result',result:r});await task.promise;assert.equal(coherentLabDiagnostic.parity.passed,true);
  if(operation==='change')change('offset',.5);else coherentLab[operation]();assertClearMetrics();assertClearDiagnostic();assert.equal($('coherentWarnings').textContent,'');
 }
});
await test('Clearing or invalidating a run also clears numeric fixed-reference evidence in the notice',async()=>{
 for(const operation of ['change','cancel','reset']){await actual();assert.match($('validationNotice').textContent,/复场.*e-/);if(operation==='change')change('offset',.5);else coherentLab[operation]();assert.doesNotMatch($('validationNotice').textContent,/复场\s+[\d.]+e[-+]/,operation);coherentLab.reset();}
});
await test('Clipboard is touched only on click and receives the exact current local report',async()=>{
 let copies=0,text='';navigator.clipboard={writeText:async value=>{copies++;text=value;}};await actual();assert.equal(copies,0);$('copyDiagnostics').click();await pause();assert.equal(copies,1);assert.deepEqual(JSON.parse(text),coherentLab.getDiagnostic());assert.match($('copyStatus').textContent,/已复制/);assert.equal(fetches,0);
});
await test('Missing clipboard or a denied copy exposes selectable JSON with no upload',async()=>{
 await actual();for(const clipboard of [undefined,{writeText:async()=>{throw new Error('denied');}}]){navigator.clipboard=clipboard;$('coherentDiagnostics').open=false;$('copyDiagnostics').click();await pause();assert.equal($('coherentDiagnostics').open,true);assert.match($('copyStatus').textContent,/请复制/);assert.equal(document.activeElement,$('diagnosticText'));assert.deepEqual(JSON.parse($('diagnosticText').value),coherentLab.getDiagnostic());}assert.equal(fetches,0);
});
await test('No diagnostic means copy reports an empty state without touching clipboard',async()=>{
 let copies=0;navigator.clipboard={writeText:async()=>copies++};$('copyDiagnostics').click();await pause();assert.equal(copies,0);assert.match($('copyStatus').textContent,/尚无/);
});
await test('Delayed clipboard success and failure cannot revive status after input, cancel, reset or a new run',async()=>{
 for(const operation of ['change','cancel','reset','run'])for(const rejects of [false,true]){coherentLab.reset();await actual();const gate=deferred();navigator.clipboard={writeText:()=>gate.promise};$('copyDiagnostics').click();if(operation==='change')change('phase',20);else if(operation==='run'){const pending=startManual();coherentLab.cancel();await pending.promise;}else coherentLab[operation]();if(rejects)gate.reject(new Error('delayed denied'));else gate.resolve();await pause();assert.equal($('copyStatus').textContent,'',`${operation}: ${rejects}`);assertClearDiagnostic();}
});
await test('Preset single/two screen buttons update actual geometry and disable irrelevant controls',async()=>{
 for(const [id,count]of [['presetOne',1],['presetTwo',2]]){plan='execute';$(id).click();await until(()=>!coherentLab.getState().running);const r=coherentLab.getState().result;assert(r);assert.equal(r.options.screenCount,count);assert.equal(r.screens.length,count);assert.equal($('separation').disabled,count===1);assert.equal($('offset').disabled,count===1);assert($(id).classList.contains('active'));assert.equal(lastWorker().request.options.screenCount,count);}
});
await test('Repeated preset clicks terminate obsolete work and keep only the latest selected screen count',async()=>{
 plan='manual';$('presetOne').click();const first=lastWorker();$('presetTwo').click();const second=lastWorker();assert(first.terminated);second.emit({type:'result',result:await fixture(second.request)});first.emit({type:'result',result:await fixture(first.request)});assert.equal(coherentLab.getState().result.options.screenCount,2);assert.equal($('screenCount').value,'2');assertIdle();
});
await test('Refinement retains current display while reporting real 64-to-96 convergence',async()=>{
 const original=await actual(),imageCount=images.length;$('refineCheck').click();assert.equal(lastWorker().request.action,'refine');assert.equal($('refineCheck').disabled,true);await until(()=>!coherentLab.getState().running);assertIdle();assert(coherentLab.getState().result===original,'Refinement replaced the original field');assert.equal(images.length,imageCount);assert.equal(coherentLab.getState().refinement.coarseN,64);assert.equal(coherentLab.getState().refinement.fineN,96);assert.match($('refinementError').textContent,/64→96/);assert.equal(coherentLabDiagnostic.refinement.fineN,96);assert.equal($('coherentBackend').textContent,'CPU 双精度');
});
await test('A failed refinement is visible without replacing the original calculated field',async()=>{
 const original=await actual();plan='manual';const promise=coherentLab.run('refine'),worker=lastWorker();worker.emit({type:'result',result:{action:'refine',refinement:{coarseN:64,fineN:96,fieldError:.01,detectorOperatorError:.02,threshold:2e-4,pass:false}}});await promise;assertIdle();assert(coherentLab.getState().result===original,'Refinement replaced the original field');assert.equal($('coherentError').hidden,false);assert.match($('coherentStatus').textContent,/未通过/);assert.equal(coherentLabDiagnostic.refinement.pass,false);
});
await test('Refinement Worker error clears the previous field and disables further refinement',async()=>{
 await actual();plan='manual';const promise=coherentLab.run('refine');lastWorker().emit({type:'error',error:'mock refinement solve failed'});assert.equal(await promise,null);assertIdle();assertNoResult();assert.equal(coherentLab.getState().refinement,null);assert.match($('coherentError').textContent,/mock refinement solve failed/);
});
await test('Changing parameters or reset during refinement blocks obsolete convergence evidence',async()=>{
 for(const operation of ['change','reset']){await actual();plan='manual';const promise=coherentLab.run('refine'),worker=lastWorker();if(operation==='change')change('phase',20);else coherentLab.reset();assert.equal(await promise,null);worker.emit({type:'result',result:{action:'refine',refinement:{coarseN:64,fineN:96,fieldError:0,pass:true}}});assertClearMetrics();assertClearDiagnostic();assert.equal(coherentLab.getState().refinement,null);coherentLab.reset();}
});
await test('Cancel and reset clear a completed refinement and leave refinement unavailable',async()=>{
 for(const operation of ['cancel','reset']){await actual();plan='manual';const promise=coherentLab.run('refine');lastWorker().emit({type:'result',result:{action:'refine',refinement:{coarseN:64,fineN:96,fieldError:0,detectorOperatorError:0,pass:true}}});await promise;assert(coherentLab.getState().refinement);coherentLab[operation]();assertNoResult();assertClearDiagnostic();assert.equal(coherentLab.getState().refinement,null);}
});
await test('Highest resolution disables further refinement after actual 144-node run',async()=>{
 change('resolution',144);await actual();assert.equal($('refineCheck').disabled,true);assert.match($('refinementError').textContent,/144/);const count=workers.length;$('refineCheck').click();assert.equal(workers.length,count);
});
await test('Experiment navigation uses real local links and pagehide blocks a pending result',async()=>{
 const links=Array.from(document.querySelectorAll('.experiment-nav a'));assert.deepEqual(links.map(a=>a.getAttribute('href')),['tracer.html','index.html','mesh.html','coherent.html']);assert.equal(links[3].getAttribute('aria-current'),'page');for(const a of links)assert(fs.existsSync(new URL('../dist/'+a.getAttribute('href'),import.meta.url)));
 const {promise,worker}=startManual();win.dispatchEvent(new win.Event('pagehide'));assert.equal(await promise,null);assert(worker.terminated);worker.emit({type:'result',result:await fixture(worker.request)});assertIdle();assertNoResult();assert.equal(fetches,0);
});

const reportPath=new URL('../research/coherent-browser-flow-validation.json',import.meta.url);
let report={};try{report=JSON.parse(fs.readFileSync(reportPath,'utf8'));}catch{}
report.generatedAt=new Date().toISOString();report.limitations=['Happy-dom and Canvas spy only; no browser visual or accessibility-tree verification.','GPU contract paths are injected mocks; these tests do not establish GPU shader execution or physical-device support.','Page links and pagehide are tested; actual browser navigation and module-Worker scheduling are not exercised.','Repeated-click test bypasses happy-dom decimal-step constraint-validation defects only; real task input bounds are independently exercised.'];
report.dom={passed:tests.length,failed:failures.length,tests,failures,actualCpuFixturesComputed:cpuExecutions,mockWorkersCreated:workers.length,networkRequests:fetches,canvasImageWrites:images.length};fs.writeFileSync(reportPath,JSON.stringify(report,null,2)+'\n');
console.log(JSON.stringify(report.dom,null,2));await win.happyDOM.close();if(failures.length)process.exitCode=1;
