/** Adversarial runtime admission tests. The injected runner is intentionally a
 * CPU-backed canonical fixture generator, not evidence of GPU execution. It is
 * never given the runtime's CPU job, event tape, or computed reference pixels.
 */
import assert from 'node:assert/strict';
import {executePLTTask} from '../dist/plt-runtime.js';
import {buildResidentReferenceBatch} from '../dist/plt-resident-reference.js';
import {decodeResidentAudit,PLT_RESIDENT_LIMITS} from '../dist/plt-resident-gpu.js';
const began=performance.now(),request={scene:{preset:'gallery'},backend:'resident-shadow',compare:true,options:{width:8,height:8,samplesPerPixel:2,maxDepth:4,wave:true},progressive:{tileSize:8,batchSamples:1,maxRuntimeMs:60000,previewIntervalMs:0}};
const cpu=await executePLTTask({...request,backend:'cpu'});let checks=0;
const sameCPU=result=>{assert.equal(result.results.length,cpu.results.length);for(let i=0;i<cpu.results.length;i++){assert.deepEqual(result.results[i].image,cpu.results[i].image);assert.equal(result.results[i].backend,'CPU 双精度');assert.equal(result.results[i].partial,false);assert.equal(result.results[i].completedSamplesPerPixel,request.options.samplesPerPixel);}};
async function candidate(input,{signal}={}){
 assert.deepEqual(Object.keys(input).sort(),['camera','options','rngStates','sampleStart','samplesPerPixel','scene','tile'].sort(),'Resident runner must receive camera/scene/prefix inputs only');
 assert.ok(!('job' in input)&&!('terms' in input)&&!('events' in input)&&!('reference' in input));
 const {scene,camera,options,tile,sampleStart,samplesPerPixel,rngStates}=input,initial=rngStates.slice();
 const job=await buildResidentReferenceBatch(scene,camera,options,{tile,sampleStart,samplesPerPixel,rngStates:rngStates.slice(),signal});assert.deepEqual(rngStates,initial);
 return {candidate:true,admitted:false,audit:job.audit,image:job.audit.reference,timings:{mockOnly:1},validation:{passed:false}};
}
async function reject(name,mutate,{failAt=1}={}){
 let calls=0;const previews=[];
 const result=await executePLTTask(request,{preview:p=>previews.push(p),gpuRunner:async(input,options)=>{calls++;const result=await candidate(input,options);if(calls===failAt)mutate(result);return result;}});
 sameCPU(result);assert.equal(calls,failAt,`${name}: GPU attempts must stop after rejection`);assert.ok(result.warnings.some(w=>w.includes('CPU 双精度参考')));assert.ok(result.results.every(r=>r.validation?.passed===false),`${name}: stale success badge survived global fallback`);
 assert.ok(previews.length>0);assert.ok(previews.at(-1).results.every(r=>r.backend==='CPU 双精度'));checks++;console.log('PASS',name);
}
let acceptedCalls=0;
const accepted=await executePLTTask(request,{gpuRunner:async(input,options)=>{acceptedCalls++;return candidate(input,options);}});
assert.equal(acceptedCalls,4);assert.ok(accepted.results.every(r=>r.validation?.passed&&r.backend.includes('GPU 驻留')));
for(const r of accepted.results){assert.equal(r.validation.fullCPUShadow,true);assert.equal(r.validation.auditCounts.paths,128);assert.equal(r.validation.auditCounts.pixels,128);assert.ok(r.validation.auditCounts.randomWords>=256);assert.match(r.validation.coverage,/every-path-rng-scatter-edge-event-and-pixel/);}
for(let mode=0;mode<2;mode++)for(let i=0;i<64;i++)assert.ok(Math.abs(accepted.results[mode].image[i]-cpu.results[mode].image[i])<=1e-12*Math.max(1,cpu.results[mode].image[i]));checks++;console.log('PASS complete canonical independent fixture admission');
await reject('self-asserted validation cannot replace audit',r=>{delete r.audit;r.validation={passed:true};});
await reject('last path cannot be dropped',r=>r.audit.paths.pop());
await reject('wrong global path key',r=>r.audit.paths.at(-1).key.pixel+=100);
await reject('wrong sample prefix key',r=>r.audit.paths.at(-1).key.sample+=1);
await reject('wrong wave key',r=>r.audit.paths.at(-1).key.wave=false);
await reject('last raw RNG word altered with original count/hash',r=>{const p=r.audit.paths.at(-1);p.rng.words[p.rng.words.length-1]^=1;});
await reject('raw RNG words missing despite matching summaries',r=>delete r.audit.paths.at(-1).rng.words);
await reject('final RNG state mismatch',r=>r.audit.paths.at(-1).rng.final^=1);
await reject('last local event altered even with unchanged pixels',r=>{const e=r.audit.paths.flatMap(p=>p.localEvents).at(-1);assert.ok(e);e.F+=1;});
await reject('unused local event cannot be omitted',r=>{const p=r.audit.paths.find(p=>p.localEvents.some(e=>!e.used));assert.ok(p);p.localEvents.splice(p.localEvents.findIndex(e=>!e.used),1);});
await reject('ordered edge fingerprint mismatch',r=>{const e=r.audit.paths.flatMap(p=>p.localEvents).find(e=>e.edgeCount>0);assert.ok(e);e.edgeHash^=1;});
await reject('exact accepted edge ID cannot hide behind original hash',r=>r.audit.paths.flatMap(p=>p.localEvents).find(e=>e.records.length).records[0].edgeId^=1);
await reject('accepted edge geometry cannot hide behind matching F',r=>r.audit.paths.flatMap(p=>p.localEvents).find(e=>e.records.length).records[0].ro+=1);
await reject('local event source mismatch',r=>r.audit.paths.flatMap(p=>p.localEvents)[0].source[0]+=1);
await reject('omitted scatter rejected even if image unchanged',r=>r.audit.paths.find(p=>p.scatters.length).scatters.pop());
await reject('scatter auxiliary branch mismatch',r=>{const s=r.audit.paths.flatMap(p=>p.scatters).find(s=>s.type==='fsd');assert.ok(s);s.sample.isDirect=!s.sample.isDirect;});

await reject('last vertex topology mismatch',r=>r.audit.paths.at(-1).vertices.at(-1).triangleId=123456);
await reject('nonzero batch diagnostic flags',r=>r.audit.flags=2);
await reject('nonzero path diagnostic flags',r=>r.audit.paths.at(-1).flags=16);
await reject('missing diagnostic flags',r=>delete r.audit.flags);
await reject('last pixel exceeds unchanged admission tolerance',r=>r.audit.reference[r.audit.reference.length-1]+=10);
await reject('nonfinite audited pixel',r=>r.audit.reference[0]=NaN);
await reject('incomplete pixel array',r=>{r.audit.reference=r.audit.reference.slice(0,-1);r.image=r.audit.reference;});
await reject('unaudited replacement image buffer',r=>r.image=r.image.slice());
await reject('audited pixel reorder cannot be accumulated at wrong coordinates',r=>{r.audit.pixelIndices.reverse();r.audit.reference.reverse();});
await reject('late ray failure withdraws previously admitted wave image',r=>r.audit.reference[0]+=100,{failAt:2});
const controller=new AbortController();let cancelCalls=0,cancelPreviews=0,lateResolved=false;
await assert.rejects(()=>executePLTTask(request,{signal:controller.signal,preview:()=>cancelPreviews++,gpuRunner:async(input,options)=>{cancelCalls++;const r=await candidate(input,options);setTimeout(()=>controller.abort(),0);await new Promise(resolve=>setTimeout(resolve,20));lateResolved=true;return r;}}),{name:'AbortError'});
await new Promise(resolve=>setTimeout(resolve,30));assert.equal(cancelCalls,1);assert.equal(cancelPreviews,0);assert.equal(lateResolved,true);checks++;console.log('PASS cancelled candidate resolving late never publishes');
let timeoutPreviews=0;
await assert.rejects(()=>executePLTTask({...request,progressive:{...request.progressive,maxRuntimeMs:40}},{preview:()=>timeoutPreviews++,gpuRunner:()=>new Promise(()=>{})}),{name:'TimeoutError'});assert.equal(timeoutPreviews,0);checks++;console.log('PASS stalled candidate respects runtime deadline');
const stopped=new AbortController();stopped.abort();await assert.rejects(()=>executePLTTask(request,{signal:stopped.signal,gpuRunner:()=>{throw new Error('Must not start');}}),{name:'AbortError'});checks++;console.log('PASS pre-cancel prevents all candidate work');
await assert.rejects(()=>executePLTTask({...request,progressive:false},{gpuRunner:candidate}),/requires bounded progressive/);checks++;console.log('PASS resident backend cannot enter legacy unbounded path');
const resumed=await executePLTTask(request,{gpuRunner:candidate});assert.ok(resumed.results.every(r=>r.validation?.passed));checks++;console.log('PASS a later independent task can succeed after cancellation');

// Synthetic packed bytes exercise defensive decoding without invoking WebGPU.
function packedFixture(){const buffer=new ArrayBuffer(PLT_RESIDENT_LIMITS.auditVectors*16),u=new Uint32Array(buffer),f=new Float32Array(buffer),laneData=new ArrayBuffer(PLT_RESIDENT_LIMITS.laneBytes),lane=new Uint32Array(laneData);u[8]=1;u[11]=1;u[81]=0xffffffff;f[90]=-1;lane[2]=1;lane[3]=2;return {buffer,u,f,config:{tile:{x:0,y:0,width:1,height:1},sampleStart:0,samplesPerPixel:1,options:{wave:false},packedScene:{toWorld:p=>p},laneData,recordData:new ArrayBuffer(0)}};}
const packed=packedFixture(),decoded=decodeResidentAudit(packed.buffer,packed.config);assert.equal(decoded.paths.length,1);assert.equal(decoded.paths[0].vertices[0].type,'escape');assert.equal(decoded.paths[0].vertices[0].center,null);checks++;console.log('PASS zero-edge packed candidate decodes without a zero-size GPU mapping');
const badPacked=(name,mutate)=>{const p=packedFixture();mutate(p);assert.throws(()=>decodeResidentAudit(p.buffer,p.config),RangeError,name);checks++;console.log('PASS packed decoder rejects',name);};
badPacked('wave enum coercion',p=>p.u[2]=2);
badPacked('vertex kind coercion',p=>p.u[80]=3);
badPacked('sampled enum coercion',p=>{p.u[9]=1;p.u[1235]=2;});
badPacked('contribution kind coercion',p=>{p.u[10]=1;p.u[1352]=3;});
badPacked('negative scatter count',p=>p.f[15]=-1);
badPacked('fractional scatter count',p=>p.f[15]=.5);
badPacked('scatter kind coercion',p=>{p.f[15]=1;p.u[1448]=4;});
badPacked('scatter validity coercion',p=>{p.f[15]=1;p.u[1448]=3;p.u[1450]=2;});
badPacked('scatter direct flag coercion',p=>{p.f[15]=1;p.u[1448]=3;p.f[1457]=2;});
badPacked('edge record range overflow',p=>{p.u[9]=1;p.u[1232]=1;});
badPacked('unfinished sample prefix',p=>new Uint32Array(p.config.laneData)[2]=0);
badPacked('path RNG capacity overflow',p=>p.u[6]=65);
badPacked('vertex capacity overflow',p=>p.u[8]=73);
badPacked('event capacity overflow',p=>p.u[9]=7);
badPacked('contribution capacity overflow',p=>p.u[10]=13);

function linkedFixture(count=2){const p=packedFixture();p.u[9]=1;p.u[1232]=count;p.u[1243]=count?0:0xffffffff;p.config.recordData=new ArrayBuffer(count*64);p.ru=new Uint32Array(p.config.recordData);for(let i=0;i<count;i++)p.ru.set([0,0,100+i,i+1<count?i+1:0xffffffff],i*16);return p;}
for(const count of [0,1,2]){const p=linkedFixture(count),a=decodeResidentAudit(p.buffer,p.config);assert.deepEqual(a.paths[0].localEvents[0].records.map(e=>e.edgeId),Array.from({length:count},(_,i)=>100+i));checks++;console.log('PASS linked decoder accepts complete record chain',count);}
const reorderedChain=linkedFixture(2);reorderedChain.u[1243]=1;reorderedChain.ru[19]=0;reorderedChain.ru[3]=0xffffffff;assert.deepEqual(decodeResidentAudit(reorderedChain.buffer,reorderedChain.config).paths[0].localEvents[0].records.map(e=>e.edgeId),[101,100]);checks++;console.log('PASS linked decoding follows links rather than allocation order');
const badLinked=(name,mutate)=>{const p=linkedFixture(2);mutate(p);assert.throws(()=>decodeResidentAudit(p.buffer,p.config),RangeError,name);checks++;console.log('PASS linked decoder rejects',name);};
badLinked('self cycle',p=>p.ru[3]=0);
badLinked('two-record cycle',p=>{p.ru[19]=0;p.u[1232]=3;});
badLinked('wrong path ownership',p=>p.ru[0]=1);
badLinked('wrong event ownership',p=>p.ru[1]=1);
badLinked('out-of-range head',p=>p.u[1243]=2);
badLinked('out-of-range next',p=>p.ru[3]=2);
badLinked('short chain',p=>p.u[1232]=3);
badLinked('long chain',p=>p.u[1232]=1);
badLinked('zero-count nonsentinel head',p=>{p.u[1232]=0;p.config.recordData=new ArrayBuffer(0);});
badLinked('record count exceeds capacity',p=>p.u[1232]=131073);
badLinked('cross-event record reuse',p=>{p.u[9]=2;p.u[1252]=2;p.u[1263]=0;});
badLinked('orphan record',p=>{p.u[1232]=1;p.ru[3]=0xffffffff;});
badLinked('partial record bytes',p=>p.config.recordData=new ArrayBuffer(68));
badLinked('missing record buffer',p=>delete p.config.recordData);
badLinked('record buffer exceeds capacity',p=>p.config.recordData=new ArrayBuffer(131073*64));
badLinked('oversized audit buffer',p=>p.buffer=new ArrayBuffer(p.buffer.byteLength+16));
badLinked('truncated lane buffer',p=>p.config.laneData=p.config.laneData.slice(0,-16));
console.log(JSON.stringify({allPass:true,checks,mockGPU:true,claimsActualGPUExecution:false,elapsedMs:performance.now()-began}));
