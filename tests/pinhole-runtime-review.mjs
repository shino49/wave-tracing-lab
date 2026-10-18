/** Independent runtime/lifecycle review. Real CPU optics at 16/32; labelled
 * import-seam mocks for progressive orchestration. No native/browser GPU claim. */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {createHash} from 'node:crypto';
import {executePinholeTask,validatePinholeRequest,PINHOLE_DEFAULTS} from '../dist/pinhole-runtime.js';
import {createPinholeCPURunner,pinholeCPUFFT} from '../dist/pinhole-cpu.js';
import {createPinholeGPURunner,pinholeGPUCapacity} from '../dist/pinhole-gpu.js';
import {installPinholeWorker} from '../dist/pinhole-worker.js';
import {createPinholeScene,buildPinholeSourceLayers} from '../dist/pinhole-scene.js';
import {generateKernels,buildPaddedKernel} from '../dist/pinhole-optics.js';
const sourceFiles=['pinhole-progress.js','pinhole-runtime.js','pinhole-cpu.js','pinhole-gpu.js','pinhole-worker.js','pinhole-app.js'];
const sourceHashes=()=>Object.fromEntries(sourceFiles.map(f=>[f,createHash('sha256').update(fs.readFileSync(new URL(`../dist/${f}`,import.meta.url))).digest('hex')]));
const initialHashes=sourceHashes();
const tests=[],failures=[],evidence={nativeGPUExecuted:false,browserExecuted:false,startedAt:new Date().toISOString(),realCPU:[]};
async function test(name,fn){try{await fn();tests.push(name);console.log('PASS',name);}catch(error){failures.push({name,error:String(error.stack).replace(/data:text\/javascript,[^\s)]+/g,'<mock-runtime-module>')});console.error('FAIL',name,error.message);}}
const pause=(ms=0)=>new Promise(r=>setTimeout(r,ms));
const defer=()=>{let resolve,reject;const promise=new Promise((r,j)=>{resolve=r;reject=j;});return {promise,resolve,reject};};
function relative(a,b,epsilon=1e-11){assert.equal(a.length,b.length);let err=0,peak=0;for(let i=0;i<a.length;i++){assert(Number.isFinite(a[i]));err=Math.max(err,Math.abs(a[i]-b[i]));peak=Math.max(peak,Math.abs(b[i]));}assert(err<=epsilon*Math.max(peak,1e-30),`relative peak error ${err/peak}`);return err/peak;}
function direct(source,n,kernel,m){const out=new Float64Array(n*n);for(let y=0;y<n;y++)for(let x=0;x<n;x++)for(let sy=0;sy<n;sy++)for(let sx=0;sx<n;sx++)out[y*n+x]+=source[sy*n+sx]*kernel[((y-sy+m)%m)*m+(x-sx+m)%m];return out;}
const realResults=new Map();
await test('real defaults and strict fixed physical-grid request contract',()=>{
 assert.equal(PINHOLE_DEFAULTS.resolution,512);assert.equal(PINHOLE_DEFAULTS.sensorWidthMm,8);assert.equal(PINHOLE_DEFAULTS.detectorDistanceMm,50);
 for(const request of [{resolution:1024},{resolution:511},{resolution:0},{depthLayers:3},{backend:'webgpu'},{diameterMm:NaN},{diameterMm:.039},{maxRuntimeMs:Infinity},{sensorWidthMm:9},{pixelQuadrature:8},{spectralStepNm:10}])assert.throws(()=>validatePinholeRequest(request));
 assert.equal(validatePinholeRequest({resolution:16,id:17,unbounded:new Float64Array(99)}).id,undefined);
});
await test('real CPU FFT preserves exact inverse normalization and displacement crop',async()=>{
 for(const n of [16,32]){const m=2*n,source=Float64Array.from({length:n*n},(_,i)=>(i*19%43)/43),kernel=new Float64Array(m*m);kernel[0]=.7;kernel[2*m+1]=.2;kernel[(m-3)*m+(m-2)]=.1;const r=createPinholeCPURunner({width:n});try{await r.convolveAccumulate({source,kernels:[{values:kernel,channel:0},{values:kernel,channel:1,weight:2}]});const result=await r.readResults(),expected=direct(source,n,kernel,m);relative(result.channels[0],expected);relative(result.channels[1],Float64Array.from(expected,x=>2*x));assert.equal(result.stats.transforms,9);assert.equal(result.stats.estimatedPeakComplexBytes,8*2*m*m*8);await assert.rejects(r.readResults(),/closed/);}finally{r.destroy();}}
 const size=16,input=Float64Array.from({length:2*size*size},(_,i)=>Math.sin(i*.23)),copy=input.slice();await pinholeCPUFFT(copy,size);await pinholeCPUFFT(copy,size,{inverse:true});relative(copy,input);
 for(const bad of [{width:1024},{width:512,fftSize:2048},{width:32,outputChannels:5}])assert.throws(()=>createPinholeCPURunner(bad));
 assert(pinholeGPUCapacity().residentBytes<45*1024**2);
});
await test('real CPU 16/32 scene reports finite native XYZ and single pixel-area normalization',async()=>{
 for(const n of [16,32]){const result=await executePinholeTask({resolution:n,spectralMode:'narrowband',depthLayers:2,backend:'cpu'});realResults.set(n,result);assert.equal(result.width,n);assert.equal(result.waveXYZ.length,3*n*n);assert.equal(result.complete,true);assert.equal(result.backend.kind,'cpu');assert.equal(result.diagnostics.cpuShadow,false);assert.equal(result.diagnostics.cpuReferenceImages,0);assert.equal(result.diagnostics.sensor.pixelAreaMm2,(8/n)**2);assert.equal(result.diagnostics.spectral.samples,1);assert.equal(result.stages.length,1);for(const [key,name]of [['waveXYZ','wave'],['geometricXYZ','geometric']]){assert(result[key].every(v=>Number.isFinite(v)&&v>=0));const power=[0,0,0];for(let i=0;i<result[key].length;i++)power[i%3]+=result[key][i]*(8/n)**2;relative(power,result.diagnostics.measurements[name].powerXYZ);}assert(result.diagnostics.collectedFractions.geometric.every(v=>Math.abs(v-1)<1e-6));assert(result.diagnostics.cpuRoundoff.geometric.clampedRoundoff>=0);evidence.realCPU.push({resolution:n,totalMs:result.totalMs,geometricCollectedFractions:result.diagnostics.collectedFractions.geometric,roundoff:result.diagnostics.cpuRoundoff,fftStats:result.diagnostics.convolutionStats});}
});
await test('real 16-grid runtime matches direct spatial sums of exact generated source/kernel grids',async()=>{
 const n=16,m=32,source=await buildPinholeSourceLayers(createPinholeScene(),{resolution:n,sensorWidthMm:8,detectorDistanceMm:50,depthCount:2,diameterMm:.2});const k=await generateKernels({resolution:n,sensorWidthMm:8,detectorDistanceMm:50,diameterMm:.2,depths:Array.from(source.depthsMm),basisNames:source.basisNames,mode:'narrowband',wavelengthNm:550});const out=[new Float64Array(3*n*n),new Float64Array(3*n*n)];
 for(let d=0;d<source.layers.length;d++)for(let b=0;b<source.basisNames.length;b++){const values=source.layers[d].basis[b];if(!values.some(v=>v>0))continue;for(const [index,kind]of ['wave','geometric'].entries()){const kernel=await buildPaddedKernel(k,d,b,{kind});for(let c=0;c<3;c++){const expected=direct(values,n,kernel.channels[c],m);for(let i=0;i<n*n;i++)out[index][3*i+c]+=expected[i];}}}
 evidence.directRuntimeMaxRelativeErrors={wave:relative(realResults.get(n).waveXYZ,out[0]),geometric:relative(realResults.get(n).geometricXYZ,out[1])};
});
await test('real CPU stage restart after injected partial GPU failure exactly matches a clean CPU run',async()=>{
 let calls=0,destroyed=0;const result=await executePinholeTask({resolution:16,spectralMode:'narrowband',depthLayers:2},{gpuFactory:()=>({async convolveAccumulate(){if(++calls===2)throw new Error('MOCK GPU failure after one submitted source task');},async readResults(){throw new Error('failed GPU must never be read');},destroy(){destroyed++;}})});assert.equal(calls,2);assert.equal(destroyed,1);assert.equal(result.backend.kind,'cpu');assert.equal(result.diagnostics.fallback.attemptedGPU,true);relative(result.waveXYZ,realResults.get(16).waveXYZ,0);relative(result.geometricXYZ,realResults.get(16).geometricXYZ,0);assert.equal(result.diagnostics.convolutionStats.convolutions,48);
});
await test('real point mode is labelled CPU pixel integral and never constructs a GPU runner',async()=>{
 let creations=0;const result=await executePinholeTask({mode:'point',resolution:32,spectralMode:'narrowband'},{gpuFactory:()=>{creations++;throw new Error('point must not use GPU');}});assert.equal(creations,0);assert.equal(result.backend.kind,'cpu-point');assert.equal(result.diagnostics.fallback,null);assert.equal(result.diagnostics.convolutionStats,null);assert(result.diagnostics.warnings.some(x=>x.includes('未进行 GPU')));assert(result.waveXYZ.every(v=>Number.isFinite(v)&&v>=0));assert(result.diagnostics.collectedFractions.geometric.every(v=>Math.abs(v-1)<1e-6));
});
await test('CPU runner fails closed after a later malformed kernel; partial state cannot publish',async()=>{
 const n=16,r=createPinholeCPURunner({width:n}),source=new Float32Array(n*n).fill(1),valid=new Float32Array(4*n*n);valid[0]=1;const bad=valid.slice();bad[0]=NaN;
 try{await assert.rejects(r.convolveAccumulate({source,kernels:[{values:valid,channel:0},{values:bad,channel:1}]}),/kernel/);await assert.rejects(r.readResults(),/closed|failed|invalid|poison/i);}finally{r.destroy();}
});
await test('CPU sequential-call guard prevents overlapping transforms without corrupting the active call',async()=>{
 const n=16,r=createPinholeCPURunner({width:n}),source=new Float32Array(n*n).fill(1),kernel=new Float32Array(4*n*n);kernel[0]=1;
 try{const pending=r.convolveAccumulate({source,kernels:[{values:kernel,channel:0}]});await assert.rejects(r.readResults(),/sequential/);await pending;const result=await r.readResults();assert(result.channels[0].every(v=>Math.abs(v-1)<1e-12));}finally{r.destroy();}
});
await test('real scene and optics deadlines remain TimeoutError, and worker publishes the terminal failure',async()=>{
 for(const mode of ['scene','point'])await assert.rejects(executePinholeTask({mode,resolution:32,maxRuntimeMs:1}),e=>e.name==='TimeoutError'&&e.diagnostics.incomplete===true);
 const messages=[],scope={postMessage:m=>messages.push(m)},dispose=installPinholeWorker(scope);try{await scope.onmessage({data:{id:1,mode:'scene',resolution:32,maxRuntimeMs:1}});const errors=messages.filter(m=>m.type==='error');assert.equal(errors.length,1);assert.equal(errors[0].name,'TimeoutError');assert(!messages.some(m=>m.type==='result'));}finally{dispose();}
});
await test('worker reports unexpected internal AbortError while its own generation remains active',async()=>{
 const messages=[],scope={postMessage:m=>messages.push(m)},dispose=installPinholeWorker(scope,async()=>{throw Object.assign(new Error('unexpected internal cancellation'),{name:'AbortError'});});try{await scope.onmessage({data:{id:1}});assert.equal(messages.length,1);assert.equal(messages[0].type,'error');}finally{dispose();}
});
await test('real CPU cancellation publishes no final image',async()=>{
 const c=new AbortController();let previews=0;await assert.rejects(executePinholeTask({resolution:32,spectralMode:'narrowband',backend:'cpu'},{signal:c.signal,progress:p=>{if(p.phase==='scene')c.abort();},preview:()=>previews++}),e=>e.name==='AbortError');assert.equal(previews,0);
});

// Import-seam fixtures: production runtime body is unchanged; imported physics
// and CPU runner are intentionally replaced. These verify control flow ONLY.
let fixtureId=0;globalThis.__pinholeReviewFixtures=new Map();
async function orchestrationFixture({tasks=2,cpuValue=7,depths=1}={}){
 const id=++fixtureId,log={cpuCreated:[],cpuConvolved:[],cpuDestroyed:0,generated:[]};
 const fixture={
 createPinholeScene:()=>({}),
 async buildPinholeSourceLayers(scene,o){return {layers:Array.from({length:depths},(_,d)=>({basis:Array.from({length:tasks},(_,i)=>new Float32Array(o.resolution**2).fill((i+1)*(d+1)))})),depthsMm:Array.from({length:depths},(_,d)=>220+20*d),basisNames:Array.from({length:tasks},(_,i)=>`mock-${i}`),metadata:{mock:true},overview:{objects:[]}};},
 async generateKernels(o){log.generated.push(o.resolution);return {...o,layers:o.depths.map(()=>({basis:o.basisNames.map(()=>({colorXYZ:[1,1,1]}))})),throughput:1,pixelAreaMm2:(8/o.resolution)**2,pixelWidthMm:8/o.resolution,areaMm2:1,spectralSamples:1,spectralStepNm:5,radialStepMm:1,radialCount:2,normalization:'MOCK ONLY'};},
 async buildPaddedKernel(k,d,b){const values=new Float32Array(4*k.resolution**2);values[0]=1;return {channels:[values,values,values]};},
 async renderPointDetector(){throw new Error('MOCK point path unavailable');},
 createPinholeCPURunner({width}){log.cpuCreated.push(width);let count=0;const read=async()=>({channels:Array.from({length:6},()=>new Float64Array(width**2).fill(cpuValue+count)),backend:'MOCK CPU control-flow runner',validation:{finite:true},stats:{mock:true,count}});return {async convolveAccumulate(){count++;log.cpuConvolved.push(width);},readResults:read,snapshotResults:read,destroy(){log.cpuDestroyed++;}};}
 };
 globalThis.__pinholeReviewFixtures.set(id,fixture);
 const url=names=>'data:text/javascript,'+encodeURIComponent(names.map(n=>n==='PINHOLE_OPTICS_SCOPE'?`export const ${n}='MOCK OPTICS, NOT PHYSICAL EVIDENCE';`:`export const ${n}=(...args)=>globalThis.__pinholeReviewFixtures.get(${id}).${n}(...args);`).join('\n'));
 let text=fs.readFileSync(new URL('../dist/pinhole-runtime.js',import.meta.url),'utf8');for(const [file,names]of [['scene',['createPinholeScene','buildPinholeSourceLayers']],['optics',['generateKernels','buildPaddedKernel','renderPointDetector','PINHOLE_OPTICS_SCOPE']],['cpu',['createPinholeCPURunner']]])text=text.replace(`'./pinhole-${file}.js'`,JSON.stringify(url(names)));
 text=text.replace("'./pinhole-progress.js'",JSON.stringify(new URL('../dist/pinhole-progress.js',import.meta.url).href));
 const module=await import('data:text/javascript,'+encodeURIComponent(text)+`#fixture-${id}`);return {execute:module.executePinholeTask,log};
}
function mockRunner(width,{value=2,failAt,readMutation,hold}={}){let calls=0,destroyed=0;const runner={async convolveAccumulate(){calls++;if(calls===failAt)throw new Error('MOCK GPU failure after partial work');if(hold)await hold;},async readResults(){const result={channels:Array.from({length:6},()=>new Float32Array(width**2).fill(value)),backend:{kind:'webgpu',label:'MOCK GPU CONTROL FLOW ONLY',physicalGPUValidated:false,browserValidated:false},validation:{passed:true,cpuParityChecked:false},stats:{mock:true,calls}};readMutation?.(result);return result;},destroy(){destroyed++;},get calls(){return calls;},get destroyed(){return destroyed;}};return runner;}
await test('MOCK progressive native512 orchestration reuses one grid with no shadow or preview mutation leak',async()=>{
 const {execute,log}=await orchestrationFixture({depths:2}),runners=[],previews=[];const result=await execute({resolution:512},{gpuFactory:({width})=>{const r=mockRunner(width,{value:width});runners.push(r);return r;},preview:frame=>{previews.push([frame.width,frame.progressive.completedContributions]);frame.waveXYZ.fill(NaN);frame.geometricXYZ.fill(-1);frame.configuration.diameterMm=99;frame.diagnostics.backendValidation.passed=false;frame.backend.kind='corrupted';frame.timings.totalMs=-99;frame.progressive.sourcePowerFraction=99;}});
 assert.deepEqual(log.generated,[512]);assert.deepEqual(previews,[[512,2],[512,3]]);assert.deepEqual(log.cpuCreated,[]);assert.deepEqual(result.stages.map(s=>s.resolution),[512]);assert(result.waveXYZ.every(v=>v===512));assert.equal(result.configuration.diameterMm,.2);assert(result.stages.every(s=>s.backend.kind==='webgpu'&&s.timings.totalMs>=0));assert.equal(result.diagnostics.backendValidation.passed,true);assert.equal(result.progressive.sourcePowerFraction,1);assert.equal(runners.length,1);assert(runners.every(r=>r.destroyed===1&&r.calls===4));
});
await test('MOCK later GPU failure resets partial output and restarts all native contributions on CPU',async()=>{
 const {execute,log}=await orchestrationFixture({depths:2}),requested=[],runners=[],publications=[];const result=await execute({resolution:512},{gpuFactory:({width})=>{requested.push(width);const r=mockRunner(width,{failAt:3});runners.push(r);return r;},preview:frame=>publications.push(['preview',frame.width,frame.backend.kind,frame.progressive.attempt]),reset:value=>publications.push(['reset',value.attempt])});assert.deepEqual(requested,[512]);assert.deepEqual(log.cpuCreated,[512]);assert.deepEqual(log.cpuConvolved,[512,512,512,512]);assert.deepEqual(publications,[['preview',512,'webgpu',0],['reset',1],['preview',512,'cpu',1],['preview',512,'cpu',1]]);assert(result.waveXYZ.every(v=>v===11));assert.equal(result.diagnostics.fallback.stageResolution,512);assert.equal(result.diagnostics.fallback.attemptedGPU,true);assert.equal(log.cpuDestroyed,1);assert.equal(result.progressive.attempt,1);assert(runners.every(r=>r.destroyed===1));
});
await test('MOCK malformed/nonfinite/negative/unvalidated GPU arrays trigger complete CPU-stage replacement',async()=>{
 for(const mutate of [r=>r.channels.pop(),r=>r.channels[0]=new Float32Array(1),r=>r.channels[0][0]=NaN,r=>r.channels[4][0]=Infinity,r=>r.channels[0][0]=-1,r=>r.channels[0][0]=-1e-30,r=>r.validation.passed=false,r=>delete r.validation]){const {execute,log}=await orchestrationFixture(),gpu=mockRunner(16,{readMutation:mutate});const result=await execute({resolution:16},{gpuFactory:()=>gpu});assert.equal(result.backend.kind,'cpu');assert(result.waveXYZ.every(v=>v===9));assert.equal(log.cpuConvolved.length,2);assert.equal(gpu.destroyed,1);assert.equal(result.diagnostics.fallback.attemptedGPU,true);}
});
await test('MOCK GPU-local AbortError/TimeoutError restart CPU when the task signal is still live',async()=>{
 for(const name of ['AbortError','TimeoutError']){const {execute,log}=await orchestrationFixture(),r=mockRunner(16);r.convolveAccumulate=async()=>{throw Object.assign(new Error('MOCK backend-local failure'),{name});};const result=await execute({resolution:16},{gpuFactory:()=>r});assert.equal(result.backend.kind,'cpu');assert.equal(log.cpuConvolved.length,2);assert.equal(result.diagnostics.fallback.attemptedGPU,true);assert.equal(r.destroyed,1);}
});
await test('MOCK cancellation promptly rejects an uncooperative GPU creation and destroys late runner',async()=>{
 const {execute,log}=await orchestrationFixture(),d=defer(),c=new AbortController();let created=false,previews=0;const p=execute({resolution:16},{signal:c.signal,gpuFactory:()=>{created=true;return d.promise;},preview:()=>previews++});while(!created)await pause();c.abort();await assert.rejects(p,e=>e.name==='AbortError');const r=mockRunner(16);d.resolve(r);await pause();assert.equal(r.destroyed,1);assert.equal(previews,0);assert.equal(log.cpuCreated.length,0);
});
await test('MOCK cancellation rejects uncooperative convolution and ignores its late success',async()=>{
 const {execute,log}=await orchestrationFixture(),d=defer(),c=new AbortController(),r=mockRunner(16,{hold:d.promise});let previews=0;const p=execute({resolution:16},{signal:c.signal,gpuFactory:()=>r,preview:()=>previews++});while(!r.calls)await pause();c.abort();await assert.rejects(p,e=>e.name==='AbortError');d.resolve();await pause();assert.equal(r.destroyed,1);assert.equal(previews,0);assert.equal(log.cpuCreated.length,0);
});
await test('MOCK total runtime cap aborts pending GPU setup without silently CPU-restarting',async()=>{
 const {execute,log}=await orchestrationFixture(),d=defer();await assert.rejects(execute({resolution:16,maxRuntimeMs:10},{gpuFactory:()=>d.promise}),e=>e.name==='TimeoutError'&&e.diagnostics.incomplete===true);assert.equal(log.cpuCreated.length,0);const r=mockRunner(16);d.resolve(r);await pause();assert.equal(r.destroyed,1);
});
await test('worker generations withhold every late event after supersession, cancel, and disposal',async()=>{
 const pending=[],messages=[],scope={postMessage:m=>messages.push(m)},dispose=installPinholeWorker(scope,(request,hooks)=>{const d=defer();pending.push({...d,request,hooks});return d.promise;});
 const first=scope.onmessage({data:{id:1}}),second=scope.onmessage({data:{id:2}});assert(pending[0].hooks.signal.aborted);pending[0].hooks.preview({width:128});pending[0].hooks.progress({fraction:1});pending[0].resolve({stale:true});await first;assert.equal(messages.length,0);pending[1].hooks.preview({width:128});assert.equal(messages[0].id,2);await scope.onmessage({data:{type:'cancel',id:1}});assert(!pending[1].hooks.signal.aborted);await scope.onmessage({data:{type:'cancel',id:2}});pending[1].resolve({cancelled:true});await second;assert.equal(messages.length,1);
 const third=scope.onmessage({data:{id:3}});dispose();pending[2].hooks.progress({fraction:1});pending[2].resolve({disposed:true});await third;assert.equal(messages.length,1);assert.equal(scope.onmessage,null);
});
await test('GPU factory own timeout covers uncooperative shader error-scope completion',async()=>{
 const d=defer();let destroyed=0;const device={lost:new Promise(()=>{}),addEventListener(){},removeEventListener(){},destroy(){destroyed++;},pushErrorScope(){},popErrorScope:()=>d.promise,createShaderModule:()=>({getCompilationInfo:async()=>({messages:[]})})};const gpu={requestAdapter:async()=>({requestDevice:async()=>device})};
 const p=createPinholeGPURunner({width:16,gpu,shaderSource:'MOCK',timeoutMs:10});const observed=await Promise.race([p.then(()=>({status:'resolved'}),error=>({status:'rejected',error})),pause(70).then(()=>({status:'hung'}))]);d.resolve(null);await p.catch(()=>{});assert.equal(observed.status,'rejected','factory remained stuck beyond timeout in popErrorScope');assert.match(observed.error.message,/timed out/);assert.equal(destroyed,1);
});

// DOM is an explicitly simulated UI, not a browser screenshot/layout test.
await test('simulated UI fixes native512 output, rejects malformed contributions, and ignores stale worker messages',async()=>{
 const {Window}=await import('happy-dom'),win=new Window({url:'https://test.local/pinhole.html',settings:{disableJavaScriptEvaluation:true,disableCSSFileLoading:true,disableJavaScriptFileLoading:true}});win.document.write(fs.readFileSync(new URL('../dist/pinhole.html',import.meta.url),'utf8'));
 const workers=[],pixels=new WeakMap();win.HTMLCanvasElement.prototype.getContext=function(){const canvas=this;return new Proxy({createImageData:(width,height)=>({width,height,data:new Uint8ClampedArray(width*height*4)}),putImageData:image=>pixels.set(canvas,image),clearRect:()=>pixels.delete(canvas)},{get:(t,k)=>k in t?t[k]:()=>{}});};
 Object.assign(globalThis,{window:win,document:win.document,Worker:class{constructor(){workers.push(this);}postMessage(m){if(m.type!=='cancel')this.request=m;}terminate(){this.terminated=true;}emit(m){this.onmessage?.({data:{id:this.request.id,...m}});}}});const app=await import('../dist/pinhole-app.js'),$=id=>win.document.getElementById(id);
 assert.equal(workers.length,0);$('phRun').click();const old=workers.at(-1);assert.equal(old.request.resolution,512);assert.equal(old.request.backend,PINHOLE_DEFAULTS.backend);assert.equal(old.request.depthLayers,PINHOLE_DEFAULTS.depthLayers);assert.equal(win.document.querySelector('#phResolution'),null);$('phCancel').click();const status=$('phStatus').textContent;old.emit({type:'error',error:'stale error'});old.emit({type:'result',result:{complete:true}});assert.equal($('phStatus').textContent,status);assert.equal($('phDownload').disabled,true);assert(old.terminated);
 const request=old.request,frame={width:512,height:512,stageResolution:512,requestedResolution:512,complete:false,waveXYZ:new Float32Array(512**2*3),geometricXYZ:new Float32Array(512**2*3),configuration:{...request},progressive:{kind:'contribution-accumulation',attempt:0,completedContributions:5,totalContributions:13,sourcePowerFraction:.774,representedSourceFraction:1}};app.validatePinholeFrame(frame,request,{partial:true});for(const mutate of [f=>f.waveXYZ[0]=NaN,f=>f.geometricXYZ[0]=-1,f=>f.width=128,f=>f.requestedResolution=256,f=>f.configuration.diameterMm=.4,f=>f.progressive.completedContributions=14,f=>f.progressive.sourcePowerFraction=NaN,f=>f.progressive.attempt=1]){const f=structuredClone(frame);mutate(f);assert.throws(()=>app.validatePinholeFrame(f,request,{partial:true}));}assert.throws(()=>app.validatePinholeFrame(frame,request,{partial:true,previousProgressive:{...frame.progressive,completedContributions:9,sourcePowerFraction:.9}}));assert.throws(()=>app.validatePinholeFrame({...frame,complete:true,totalMs:1},request));
 const report=app.boundedPinholeReport({waveXYZ:frame.waveXYZ,diagnostics:{raw:new Float64Array(100),value:Infinity},huge:'x'.repeat(5000)});assert.equal(report.waveXYZ,undefined);assert.equal(report.diagnostics.raw,undefined);assert.equal(report.diagnostics.value,'Infinity');assert.equal(report.huge.length,1024);win.dispatchEvent(new win.Event('pagehide'));await win.happyDOM.close();
});
await test('reviewed production sources remain unchanged during this test run',()=>assert.deepEqual(sourceHashes(),initialHashes));
evidence.testCount=tests.length;evidence.failures=failures;evidence.passed=failures.length===0;evidence.hashes=sourceHashes();console.log('REVIEW_EVIDENCE',JSON.stringify(evidence));console.log(`${tests.length} passed, ${failures.length} failed`);if(failures.length)process.exitCode=1;
