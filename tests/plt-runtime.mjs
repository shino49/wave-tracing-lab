/** Real CPU PLT orchestration plus injected GPU contract tests.
 * No browser, physical GPU, WGSL compilation, or browser Worker executes here. */
import fs from 'node:fs';
import assert from 'node:assert/strict';
import {executePLTTask} from '../dist/plt-runtime.js';
const tests=[],failures=[],originalNavigator=Object.getOwnPropertyDescriptor(globalThis,'navigator'),originalFetch=globalThis.fetch;
Object.defineProperty(globalThis,'navigator',{value:{},configurable:true});
let networkRequests=0,cpuRuns=0;
globalThis.fetch=async()=>{networkRequests++;throw new Error('Network is forbidden in local PLT runtime tests');};
const options={width:8,height:8,samplesPerPixel:1,maxDepth:2,seed:7391};
const request={options,backend:'cpu',scene:{preset:'gallery'}};
const OBJ='v -1 -1 0\nv 1 -1 0\nv 0 1 0\nf 1 2 3\n';
const run=async(r={},hooks={})=>{cpuRuns++;return executePLTTask({...request,...r,options:{...options,...r.options}},hooks);};
const sameImages=(a,b)=>{assert.equal(a.results.length,b.results.length);a.results.forEach((r,i)=>assert.deepEqual(r.image,b.results[i].image));};
const test=async(name,fn)=>{try{await fn();tests.push(name);console.log('PASS:',name);}catch(e){failures.push({name,error:e.stack});console.error('FAIL:',name,e.message);}};
let reference;
await test('CPU smoke executes both physical modes with finite nonnegative arrays, measurements and progress',async()=>{
 const progress=[];reference=await run({}, {progress:p=>progress.push(p),gpuRunner:()=>assert.fail('Explicit CPU must bypass GPU')});
 assert.deepEqual(reference.results.map(r=>r.wave),[true,false]);
 for(const r of reference.results){assert.equal(r.width,8);assert.equal(r.height,8);assert(r.image instanceof Float64Array);assert.equal(r.image.length,64);assert(r.image.every(v=>Number.isFinite(v)&&v>=0));assert(r.image.some(v=>v>0));assert.equal(r.backend,'CPU 双精度');assert.equal(r.validation,null);assert.equal(r.timings.gpu,null);assert.equal(r.measurement.mean,r.image.reduce((a,b)=>a+b,0)/64);assert.equal(r.measurement.unit,'relative native-PLT camera response');assert(r.events>=0);assert(r.terms>0);assert(r.timings.cpuTraversalAndReferenceMs>=0);}
 assert.equal(reference.results[1].events,0);assert(reference.results[0].events>0);assert(reference.scene.triangles>0);assert(reference.scene.emitters>0);assert(reference.totalMs>=0);assert(reference.comparison.absoluteMeanDifference>0);assert.match(reference.scope,/3D backward PLT/);assert.match(reference.privacy,/no model upload.*no server computation/);assert.deepEqual(reference.warnings,[]);assert(progress.length>=4);assert(progress.every(p=>typeof p.text==='string'&&Number.isFinite(p.fraction)&&p.fraction>=0&&p.fraction<=1));assert.equal(progress.at(-1).fraction,1);
});
await test('No WebGPU uses explicit CPU fallback without network and preserves exact reference arrays',async()=>{
 const r=await run({backend:'auto'});sameImages(r,reference);assert.equal(r.warnings.length,2);assert(r.warnings.every(w=>/WebGPU.*CPU/.test(w)));assert(r.results.every(r=>r.validation===null&&r.backend==='CPU 双精度'));assert.equal(networkRequests,0);
});
await test('Fixed seed gives reproducible computed samples; camera changes alter actual transport',async()=>{
 sameImages(await run(),reference);const moved=await run({camera:{yaw:25,pitch:10}});assert.notDeepEqual(moved.camera,reference.camera);assert.notDeepEqual(moved.results[0].image,reference.results[0].image);assert.notDeepEqual(moved.results[1].image,reference.results[1].image);
});
await test('Non-comparison wave and ray modes compute just the selected mode',async()=>{
 for(const wave of [true,false]){const r=await run({compare:false,options:{wave}});assert.equal(r.results.length,1);assert.equal(r.results[0].wave,wave);assert.equal(r.comparison,null);assert.deepEqual(r.results[0].image,reference.results.find(x=>x.wave===wave).image);}
});
await test('All four built-in scenes execute actual CPU paths',async()=>{
 for(const preset of ['gallery','prism','baffles','double-edge']){const r=await run({compare:false,scene:{preset}});assert.equal(r.scene.preset,preset);assert(r.scene.triangles>0);assert(r.results[0].image.every(v=>Number.isFinite(v)&&v>=0));}
});
await test('OBJ text is parsed locally, changes scene geometry and never uploads',async()=>{
 const before=networkRequests;for(const preset of ['gallery','baffles','double-edge']){const r=await run({compare:false,scene:{preset,objText:OBJ,modelRotation:[0,35,0],modelScale:.7,modelMaterial:1}});assert(r.scene.triangles>0);assert.match(r.privacy,/browser memory.*no model upload/);if(preset==='gallery')assert.notEqual(r.scene.triangles,reference.scene.triangles);assert(r.results[0].image.every(Number.isFinite));}assert.equal(networkRequests,before);
});
await test('Invalid numeric sampling inputs reject before GPU admission',async()=>{
 const bad={width:[7,257,8.5,NaN,'8'],height:[7,193,8.5,Infinity],samplesPerPixel:[0,513,1.5,NaN],maxDepth:[0,7,1.5,NaN],wavelengthMm:[0,.5001,NaN,Infinity,'0.025'],boundary:['invalid']};
 for(const [key,values]of Object.entries(bad))for(const value of values)await assert.rejects(run({backend:'auto',options:{[key]:value}},{gpuRunner:()=>assert.fail('Bad input reached GPU')}));
});
await test('Invalid backend, preset, OBJ, model scale and rotation reject',async()=>{
 for(const bad of [{backend:'gpu'},{backend:'server'},{scene:{preset:'bad'}},{scene:{objText:{}}},{scene:{objText:'not OBJ geometry'}},{scene:{modelScale:0}},{scene:{modelScale:Infinity}},{scene:{modelRotation:[0,NaN,0]}}])await assert.rejects(run(bad));
});
await test('Invalid camera parameters reject rather than silently clamping nonfinite values',async()=>{
 for(const camera of [{yaw:NaN},{yaw:Infinity},{pitch:Infinity},{pitch:-Infinity},{pitch:NaN},{distanceScale:0},{distanceScale:-1},{distanceScale:Infinity}])await assert.rejects(run({camera}),JSON.stringify(camera));
});
await test('Passing injected GPU contract receives complete CPU jobs and returns accepted images',async()=>{
 let calls=0;const r=await run({backend:'auto'},{gpuRunner:async(job,hook)=>{calls++;assert.equal(job.reference.length,64);assert(Array.isArray(job.events));assert(Array.isArray(job.terms));assert(hook.signal instanceof AbortSignal);assert.equal(hook.signal.aborted,false);return {image:Float32Array.from(job.reference),validation:{passed:true,coverage:'every-event-and-every-pixel'},label:'mock verified GPU',timings:{gpuMs:1.5}};}});
 assert.equal(calls,2);assert.deepEqual(r.warnings,[]);assert(r.results.every(r=>r.backend==='mock verified GPU'&&r.validation.passed===true&&r.image instanceof Float64Array));
});
await test('Thrown GPU rejection retains CPU images and diagnostic without claiming GPU success',async()=>{
 const diagnostic={stage:'every-event',reason:'bad sample'};const r=await run({backend:'auto'},{gpuRunner:async()=>{throw Object.assign(new Error('injected rejected device'),{diagnostic});}});sameImages(r,reference);assert.equal(r.warnings.length,1);assert(r.warnings.every(x=>/injected rejected device/.test(x)));assert(r.results.every(r=>r.backend==='CPU 双精度'&&r.validation.passed===false&&r.timings.gpu===null));assert.deepEqual(r.results[0].validation.diagnostic,diagnostic);
});
for(const [name,payload] of [
 ['failing parity gate',job=>({image:new Float32Array(job.reference.length).fill(99),validation:{passed:false}})],
 ['missing parity gate',job=>({image:new Float32Array(job.reference.length).fill(99)})],
 ['short camera array',job=>({image:new Float32Array(job.reference.length-1),validation:{passed:true}})],
 ['nonfinite camera sample',job=>({image:Float32Array.from(job.reference,(_,i)=>i?0:NaN),validation:{passed:true}})],
 ['negative camera sample',job=>({image:Float32Array.from(job.reference,(_,i)=>i?0:-1),validation:{passed:true}})]
])await test(`GPU ${name} fails closed to exact CPU reference`,async()=>{
 const r=await run({backend:'auto'},{gpuRunner:async job=>payload(job)});sameImages(r,reference);assert(r.results.every(r=>r.backend==='CPU 双精度'&&r.validation?.passed===false));assert(r.warnings.length>=1);
});
await test('Pre-aborted CPU task and injected GPU AbortError reject as cancellation',async()=>{
 const ac=new AbortController();ac.abort();await assert.rejects(run({}, {signal:ac.signal}),{name:'AbortError'});await assert.rejects(run({backend:'auto'},{gpuRunner:async()=>{throw Object.assign(new Error('cancelled GPU'),{name:'AbortError'});}}),{name:'AbortError'});
});
await test('Explicit CPU still bypasses GPU when navigator.gpu is present',async()=>{
 navigator.gpu={};try{sameImages(await run({}, {gpuRunner:()=>assert.fail('Explicit CPU invoked GPU')}),reference);}finally{delete navigator.gpu;}
});
await test('Cancellation during CPU progress stops actual traversal before publishing a result',async()=>{
 const ac=new AbortController();let seen=0;await assert.rejects(run({}, {signal:ac.signal,progress:()=>{seen++;ac.abort();}}),{name:'AbortError'});assert.equal(seen,1);
});
await test('Actual Worker module preserves request IDs and maps CPU success and errors into messages',async()=>{
 const oldSelf=globalThis.self,messages=[];globalThis.self={postMessage:message=>messages.push(message)};
 try{await import('../dist/plt-worker.js');await self.onmessage({data:{...request,id:317}});assert(messages.some(m=>m.type==='progress'));assert(messages.every(m=>m.id===317));assert.equal(messages.at(-1).type,'result');sameImages(messages.at(-1).result,reference);messages.length=0;await self.onmessage({data:{...request,id:318,options:{...options,width:1}}});assert.equal(messages.length,1);assert.equal(messages[0].id,318);assert.equal(messages[0].type,'error');assert.match(messages[0].error,/width/);}finally{if(oldSelf===undefined)delete globalThis.self;else globalThis.self=oldSelf;}
});
if(originalNavigator)Object.defineProperty(globalThis,'navigator',originalNavigator);else delete globalThis.navigator;globalThis.fetch=originalFetch;
const reportPath=new URL('../research/plt-browser-flow-validation.json',import.meta.url);let report={};try{report=JSON.parse(fs.readFileSync(reportPath,'utf8'));}catch{}
report.generatedAt=new Date().toISOString();report.limitations=['Happy-dom and Canvas spies only: no browser visual QA, accessibility-tree QA, real browser Worker scheduling, or device rendering.','Injected GPU contracts do not establish WGSL compilation, physical-device execution, or WebGPU browser support.'];
report.runtime={passed:tests.length,failed:failures.length,tests,failures,actualCpuSolverExecuted:true,cpuTaskInvocations:cpuRuns,networkRequests,smokeOptions:options,reference:reference?{scene:reference.scene,comparison:reference.comparison,results:reference.results.map(({wave,measurement,events,terms})=>({wave,measurement,events,terms}))}:null};fs.writeFileSync(reportPath,JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify({runtimePassed:tests.length,runtimeFailed:failures.length,networkRequests}));if(failures.length)process.exitCode=1;
