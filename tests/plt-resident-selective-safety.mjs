/** Host-control-flow and synthetic packed-byte tests ONLY.
 * No shader compilation, Dawn device, real/software GPU, or GPU-math claim.
 * Canonical CPU replay is real; all GPU readbacks below are synthetic fixtures.
 */
import assert from 'node:assert/strict';
import {createPLTScene,createPLTRandomStates,randomGenerator} from '../dist/plt-core.js';
import {buildResidentReferenceBatch,compareResidentAudit} from '../dist/plt-resident-reference.js';
import {createPLTResidentRunner,decodeResidentAudit,isResidentReplayable,PLT_RESIDENT_LIMITS,RESIDENT_REPLAY_MASK} from '../dist/plt-resident-gpu.js';

globalThis.GPUBufferUsage={MAP_READ:1,COPY_SRC:4,COPY_DST:8,UNIFORM:64,STORAGE:128};
globalThis.GPUMapMode={READ:1};
const WORDS=PLT_RESIDENT_LIMITS.auditVectors*4,LANE_WORDS=PLT_RESIDENT_LIMITS.laneBytes/4;
const AMBIGUOUS=2|2048,SENTINEL=0xffffffff;
let passed=0;const failures=[];
async function test(name,fn){try{await fn();passed++;console.log('PASS',name);}catch(error){failures.push({name,error:error.stack});console.error('FAIL',name,error.stack);}}
function randomPrefix(seed,count=2){const rng=randomGenerator(seed),words=[];let hash=2166136261;for(let i=0;i<count;i++){const word=Math.floor(rng()*4294967296);words.push(word);hash=Math.imul(hash^word,16777619)>>>0;}return {start:seed,final:rng.getState(),count,hash,words};}
function writePath(u,f,slot,{pixel,sample,wave=false,flags=0,value=0,rng}){
 const at=slot*WORDS;u.set([pixel,sample,Number(wave),flags,rng.start,rng.final,rng.count,rng.hash,1,0,1,1],at);
 f.set([value,0,0,0],at+12);u.set(rng.words,at+16);u[at+81]=SENTINEL;f[at+90]=-1;
 u.set([1,0,0,0],at+1352);f.set([value,1,1,0],at+1356);
}
function packedFixture(){
 const tile={x:0,y:0,width:2,height:1},samplesPerPixel=2,sampleStart=1;
 const buffer=new ArrayBuffer(4*WORDS*4),u=new Uint32Array(buffer),f=new Float32Array(buffer);
 const laneData=new ArrayBuffer(2*PLT_RESIDENT_LIMITS.laneBytes),lu=new Uint32Array(laneData),lf=new Float32Array(laneData);
 for(let slot=0;slot<4;slot++)writePath(u,f,slot,{pixel:Math.floor(slot/2),sample:sampleStart+slot%2,flags:slot===0?AMBIGUOUS:0,value:[6000,3,5,7][slot],rng:randomPrefix(123+slot)});
 for(let lane=0;lane<2;lane++){lu.set([lane,999,sampleStart+samplesPerPixel,2],lane*LANE_WORDS);lf[lane*LANE_WORDS+49]=lane===0?3:12;}
 // Failed sample owns two partially constructed records. A retained sample
 // owns a separate event/record; no discarded record may enter that event.
 const recordData=new ArrayBuffer(3*64),ru=new Uint32Array(recordData),rf=new Float32Array(recordData);
 ru.set([0,0,101,1],0);ru.set([0,0,102,SENTINEL],16);ru.set([1,0,201,SENTINEL],32);
 rf[4]=9000;rf[20]=9001;rf[36]=42;
 u[9]=1;u[1232]=2;u[1243]=0;
 u[WORDS+9]=1;u[WORDS+1232]=1;u[WORDS+1243]=2;
 const path={key:{wave:false,pixel:0,sample:sampleStart},flags:0,value:2,rng:randomPrefix(123,5),vertices:[],localEvents:[],scatters:[],contributions:[{type:'emission',depth:0,value:2,weight:1}],termination:{reason:'escaped',depth:0}};
 const recoveredPaths=new Map([[0,{path,flags:AMBIGUOUS}]]);
 const config={tile,sampleStart,samplesPerPixel,options:{width:8,height:8,wave:false},packedScene:{toWorld:p=>p},laneData,recordData,recoveredPaths};
 return {buffer,u,f,lu,lf,ru,rf,path,config};
}
const decode=p=>decodeResidentAudit(p.buffer,p.config);
async function badPacked(name,mutate,pattern){await test(`packed decoder rejects ${name}`,()=>{const p=packedFixture();mutate(p);assert.throws(()=>decode(p),pattern??RangeError);});}

await test('replay mask permits only bounded ambiguity reasons',()=>{
 assert.equal(isResidentReplayable(AMBIGUOUS),true);assert.equal(isResidentReplayable(RESIDENT_REPLAY_MASK),true);
 for(const flags of [0,1,16,32,64,128,2048,AMBIGUOUS|1,AMBIGUOUS|16,AMBIGUOUS|32,AMBIGUOUS|64,AMBIGUOUS|128,AMBIGUOUS|0x80000000,NaN,1.5])assert.equal(isResidentReplayable(flags),false,`${flags}`);
});
await test('quarantines failed partial contributions and records; each whole sample enters once',()=>{
 const p=packedFixture(),result=decode(p);
 assert.equal(result.flags,0);assert.equal(result.paths.length,4);assert.strictEqual(result.paths[0],p.path);
 assert.deepEqual([...result.reference],[2.5,6]);assert.equal(result.paths[0].contributions.length,1);
 assert.equal(result.paths[0].localEvents.length,0);assert.deepEqual(result.paths[1].localEvents[0].records.map(r=>r.edgeId),[201]);
 assert.equal(result.paths[1].localEvents[0].records[0].point[0],42);
 assert.ok(!result.paths.flatMap(path=>path.contributions).some(c=>c.value===6000));
 assert.ok(!result.paths.flatMap(path=>path.localEvents).flatMap(event=>event.records).some(r=>r.edgeId===101||r.edgeId===102));
 assert.deepEqual(new Set(result.paths.map(path=>`${path.key.pixel}:${path.key.sample}`)).size,4);
});
await test('incomplete discarded edge chains may be quarantined without becoming retained events',()=>{
 const p=packedFixture();p.ru[3]=SENTINEL;p.u[9]=0; // Two disconnected allocations before an event completes.
 const result=decode(p);assert.deepEqual([...result.reference],[2.5,6]);assert.equal(result.paths[1].localEvents[0].records[0].edgeId,201);
});
await badPacked('failed partial radiance in the retained subtotal',p=>{p.lf[49]=6003;},/retained GPU subtotal/);
await badPacked('nonfinite retained subtotal',p=>{p.lf[49]=NaN;});
await badPacked('nonfinite retained sample',p=>{p.f[WORDS+12]=NaN;});
await badPacked('negative retained sample',p=>{p.f[WORDS+12]=-3;p.lf[49]=-3;});
await badPacked('unconsumed extra recovery',p=>p.config.recoveredPaths.set(999,{path:p.path,flags:AMBIGUOUS}));
await badPacked('duplicate recovery ownership in another slot',p=>p.config.recoveredPaths.set(1,{path:p.path,flags:AMBIGUOUS}));
await badPacked('recovery for wrong sample',p=>p.path.key.sample++);
await badPacked('recovery for wrong pixel',p=>p.path.key.pixel++);
await badPacked('recovery for wrong wave',p=>p.path.key.wave=true);
await badPacked('nonrecoverable diagnostic hidden by replacement',p=>{p.u[3]|=64;p.config.recoveredPaths.get(0).flags|=64;});
await badPacked('mismatched diagnostic certificate',p=>p.u[3]^=2048);
await badPacked('flagged CPU replacement',p=>p.path.flags=2);
await badPacked('nonfinite CPU replacement',p=>p.path.value=Infinity);
await badPacked('negative CPU replacement',p=>p.path.value=-1);
await badPacked('discarded random word corruption',p=>p.u[16]^=1);
await badPacked('discarded RNG hash corruption',p=>p.u[7]^=1);
await badPacked('discarded RNG endpoint corruption',p=>p.u[5]^=1);
await badPacked('discarded RNG original seed corruption',p=>p.u[4]^=1);
await badPacked('discarded RNG capacity overflow',p=>p.u[6]=65);
await badPacked('incomplete sample prefix after replay',p=>p.lu[2]--);
await badPacked('still-paused lane after replay',p=>p.lu[3]=3);
await badPacked('discarded record claims an unrecovered owner',p=>p.ru[0]=2);
await badPacked('discarded record event outside capacity',p=>p.ru[1]=6);
await badPacked('discarded record points into retained chain',p=>p.ru[3]=2);
await badPacked('discarded record crosses event ownership',p=>p.ru[17]=1);
await badPacked('discarded record links to a different recovered owner',p=>{
 const path=structuredClone(p.path);path.key={wave:false,pixel:1,sample:2};path.rng=randomPrefix(126,5);
 p.config.recoveredPaths.set(3,{path,flags:AMBIGUOUS});p.u[3*WORDS+3]=AMBIGUOUS;p.lf[LANE_WORDS+49]=5;p.ru[16]=3;
});
await badPacked('discarded record out-of-range link',p=>p.ru[3]=3);
await badPacked('discarded self-cycle',p=>p.ru[3]=0);
await badPacked('discarded multi-record cycle',p=>p.ru[19]=0);
await badPacked('discarded chain with duplicate incoming links',p=>{p.config.recordData=new ArrayBuffer(4*64);const ru=new Uint32Array(p.config.recordData);ru.set(p.ru);ru.set([0,0,103,1],48);});
await badPacked('retained event tries to consume a quarantined record',p=>p.u[WORDS+1243]=0);
await badPacked('retained record owned by recovered path',p=>p.ru[32]=0);
await badPacked('partial edge-record bytes',p=>p.config.recordData=p.config.recordData.slice(0,-1));

// This state machine emulates only lifecycle bytes and canonical RNG updates.
// It does not evaluate geometry, operators, traversal or any shader instructions.
function mockGPU(config={}){
 const buffers=[],resumes=[],starts=[],reads=[],writes=[],uniform=new Uint32Array(40);
 let initial,lanes,audit,round=0,status=new Uint32Array(4),scopes=0,destroyed=0;
 const deferred=()=>{let resolve;const promise=new Promise(r=>resolve=r);return {promise,resolve};};
 const lost=deferred(),listeners=new Map();
 const fixture=(label,bytes)=>{
  let value;if(label==='Resident status readback')value=status.buffer;
  if(label==='Resident lane readback')value=lanes.buffer;
  if(label==='Resident audit readback')value=audit;
  if(label==='Resident edge readback')value=new ArrayBuffer(bytes);
  assert.equal(value.byteLength,bytes,`Unexpected mock readback ${label}`);
  const result=value.slice(0);config.corruptRead?.(label,result,{round,uniform,lanes,resumes});return result;
 };
 const simulate=stage=>{
  if(stage==='rr_init'){round=0;resumes.length=0;starts.length=0;lanes=new Uint32Array(LANE_WORDS);lanes.set([initial[0],initial[1],uniform[8],0]);audit=new ArrayBuffer(uniform[9]*WORDS*4);return;}
  if(stage!=='rr_scatter_step')return;
  round++;status=new Uint32Array(4);
  if(lanes[3]===0){
   const sample=lanes[2],slot=sample-uniform[8],flags=config.plan?.[slot]??(slot===0?AMBIGUOUS:0),prefix=randomPrefix(lanes[1]);
   starts.push({sample,seed:lanes[1],partialFinal:prefix.final});
   const value=flags?6000:(config.retainedValue??3),u=new Uint32Array(audit),f=new Float32Array(audit);
   writePath(u,f,slot,{pixel:lanes[0],sample,wave:!!uniform[11],flags,value,rng:prefix});
   lanes[17]=lanes[1];lanes[1]=prefix.final;lanes[2]++;lanes[11]=flags;
   new Float32Array(lanes.buffer)[48]=value;
   if(!flags)new Float32Array(lanes.buffer)[49]+=value;
   lanes[3]=flags&&uniform[37]?3:lanes[2]===uniform[8]+uniform[9]?2:0;
   config.corruptPaused?.(lanes,{round,uniform,starts});
  }
  status[0]=lanes[3]===2?0:1;status[1]=lanes[11];
  if(config.statusFlags!==undefined&&round===1)status[1]=config.statusFlags;
  if(config.recordCount!==undefined)status[2]=config.recordCount;
 };
 const device={limits:{maxBufferSize:1e9,maxStorageBufferBindingSize:1e9},lost:lost.promise,
  addEventListener:(name,fn)=>listeners.set(name,fn),removeEventListener:name=>listeners.delete(name),
  pushErrorScope(){scopes++;},async popErrorScope(){scopes--;return null;},
  createShaderModule:()=>({getCompilationInfo:async()=>({messages:[]})}),
  createComputePipelineAsync:async({compute:{entryPoint}})=>({entryPoint,getBindGroupLayout:()=>({})}),
  createBindGroup:spec=>spec,
  createBuffer(spec){const b={...spec,destroyed:0,mapped:false,destroy(){this.destroyed++;this.mapped=false;},async mapAsync(){config.onMap?.(this.label,{round});if(config.mapWait&&this.label===config.mapLabel)await config.mapWait;this.mapped=true;},getMappedRange(offset,bytes){assert.ok(this.mapped&&!this.destroyed);reads.push(this.label);return fixture(this.label,bytes);},unmap(){this.mapped=false;}};buffers.push(b);return b;},
  createCommandEncoder(){const commands=[];return {beginComputePass(){let pipeline;return {setPipeline:p=>pipeline=p,setBindGroup(){},dispatchWorkgroups:()=>commands.push(pipeline.entryPoint),end(){}};},copyBufferToBuffer(){},finish:()=>({commands})};},
  queue:{writeBuffer(buffer,offset,data){
   const bytes=new Uint8Array(data.buffer??data,data.byteOffset??0,data.byteLength),words=new Uint32Array(bytes.slice().buffer);writes.push({label:buffer.label,offset,bytes:bytes.length});
   if(buffer.label==='Resident parameters')uniform.set(words,offset/4);
   if(buffer.label==='Resident initial prefixes')initial=words;
   if(buffer.label==='Resident persistent lanes'){
    assert.equal(offset,0);assert.equal(words.length,LANE_WORDS);resumes.push(words.slice());lanes=words.slice();config.onResume?.(words,{round});
   }
  },submit:commands=>{for(const stage of commands.flatMap(command=>command.commands))simulate(stage);},onSubmittedWorkDone:async()=>{}},
  destroy(){destroyed++;},
 };
 return {gpu:{requestAdapter:async()=>({info:{description:'HOST SELECTIVE REPLAY MOCK; NO GPU EXECUTION'},requestDevice:async()=>device})},device,resumes,starts,reads,writes,
  get round(){return round;},lose:()=>lost.resolve({message:'mock replay device loss'}),
  check(){assert.equal(scopes,0);assert.equal(destroyed,1);assert.equal(listeners.size,0);for(const buffer of buffers){assert.equal(buffer.destroyed,1,buffer.label);assert.equal(buffer.mapped,false,buffer.label);}},
 };
}
const scene=createPLTScene(),options={width:8,height:8,samplesPerPixel:4,maxDepth:2,wave:false},tile={x:3,y:0,width:1,height:1};
const fresh=()=>({tile,sampleStart:0,samplesPerPixel:2,rngStates:createPLTRandomStates(options)});
async function withRunner(config,fn,{mode='selective',sceneInput=scene,cameraInput=scene.camera}={}){
 const mock=mockGPU(config);let runner;
 try{runner=await createPLTResidentRunner(sceneInput,cameraInput,options,{gpu:mock.gpu,shaderSource:'HOST MOCK ONLY; NOT WGSL',timeoutMs:1000,recoveryMode:mode});await fn(runner,mock);}
 finally{runner?.destroy();await new Promise(resolve=>setTimeout(resolve,0));mock.check();}
}
async function expectRejected(config,pattern,extra){await withRunner(config,async(runner,mock)=>{const input=fresh(),original=input.rngStates.slice();await assert.rejects(runner.run(input),pattern);assert.deepEqual(input.rngStates,original);await assert.rejects(runner.run(fresh()),{name:'AbortError'});},extra);}

await test('host replay starts from saved original RNG, not failed endpoint; later sample uses canonical final RNG',async()=>{
 const input=fresh(),before=input.rngStates.slice(),expected=await buildResidentReferenceBatch(scene,scene.camera,options,{...input,samplesPerPixel:1,rngStates:before.slice()});
 const cpuPath=expected.audit.paths[0];assert.ok(cpuPath.rng.count>2,'Fixture must distinguish saved seed from failed endpoint');
 await withRunner({},async(runner,mock)=>{
  const result=await runner.run(input);assert.equal(result.admitted,false);assert.equal(result.candidate,true);assert.equal(result.recovery.paths,1);
  assert.deepEqual(result.audit.paths[0],cpuPath);assert.equal(mock.resumes.length,1);
  const resumed=mock.resumes[0];assert.equal(resumed[1],cpuPath.rng.final);assert.equal(resumed[17],cpuPath.rng.start);assert.equal(resumed[2],1);assert.equal(resumed[3],0);assert.equal(resumed[11],0);
  assert.notEqual(mock.starts[0].partialFinal,cpuPath.rng.final);assert.equal(mock.starts[1].seed,cpuPath.rng.final);
  assert.equal(result.finalRngStates[0],randomPrefix(cpuPath.rng.final).final);assert.deepEqual(input.rngStates,before);
  assert.equal(new Float32Array(resumed.buffer)[49],0,'Failed partial power must never be added');
  assert.equal(result.image[0],(cpuPath.value+3)/2);assert.equal(result.audit.paths[1].sampleValue,3);
  assert.equal(mock.writes.filter(write=>write.label==='Resident persistent lanes').length,1);
  assert.ok(!mock.writes.some(write=>write.label==='Resident bounded audit'||write.label==='Resident bounded edge records'),'No CPU event/geometry tape is uploaded');
  assert.equal(result.timings.cpuReferenceProvided,false);
 });
});
await test('multiple ambiguous samples are replayed uniquely and next batch continues canonical prefix',async()=>{
 await withRunner({plan:[AMBIGUOUS,AMBIGUOUS]},async(runner,mock)=>{
  const input=fresh(),expectedRng=input.rngStates.slice(),expected=await buildResidentReferenceBatch(scene,scene.camera,options,{...input,rngStates:expectedRng});
  const result=await runner.run(input);assert.equal(result.recovery.paths,2);assert.equal(mock.resumes.length,2);
  assert.deepEqual(result.audit.paths.map(path=>({key:path.key,rng:path.rng,value:path.value})),expected.audit.paths.map(path=>({key:path.key,rng:path.rng,value:path.value})));
  assert.equal(result.finalRngStates[0],expectedRng[3]);assert.equal(result.image[0],expected.audit.reference[0]);assert.equal(compareResidentAudit(expected,result.audit).ok,true);
  assert.equal(mock.resumes[1][3],2,'Replay of the last sample must finish the lane');
  const next={...fresh(),sampleStart:2,rngStates:input.rngStates.slice()};next.rngStates[3]=result.finalRngStates[0];
  const expectedNext=await buildResidentReferenceBatch(scene,scene.camera,options,{...next,rngStates:next.rngStates.slice()});
  const resultNext=await runner.run(next);assert.equal(mock.starts[0].seed,result.finalRngStates[0]);assert.equal(compareResidentAudit(expectedNext,resultNext.audit).ok,true);
 });
});
await test('retained samples on both sides of a replay preserve subtotal and canonical continuation',async()=>{
 const input={...fresh(),samplesPerPixel:3},prefix=input.rngStates.slice();prefix[3]=randomPrefix(prefix[3]).final;
 const expected=await buildResidentReferenceBatch(scene,scene.camera,options,{...input,sampleStart:1,samplesPerPixel:1,rngStates:prefix});
 await withRunner({plan:[0,AMBIGUOUS,0]},async(runner,mock)=>{
  const result=await runner.run(input),path=expected.audit.paths[0];assert.equal(result.recovery.paths,1);
  assert.equal(mock.resumes.length,1);assert.equal(new Float32Array(mock.resumes[0].buffer)[49],3);
  assert.equal(result.audit.paths[0].sampleValue,3);assert.deepEqual(result.audit.paths[1],path);assert.equal(result.audit.paths[2].sampleValue,3);
  assert.equal(mock.starts[2].seed,path.rng.final);assert.equal(result.finalRngStates[0],randomPrefix(path.rng.final).final);
  assert.equal(result.image[0],(3+path.value+3)/3);
 });
});
await test('strict mode remains rejecting on the same ambiguous readback',async()=>{await expectRejected({}, {name:'PLTResidentRejected'}, {mode:'strict'});});
for(const bit of [1,16,32,64,128,0x80000000])await test(`host withholds entire candidate for nonrecoverable flag ${bit}`,()=>expectRejected({plan:[AMBIGUOUS|bit,0]},{name:'PLTResidentRejected'}));
await test('host withholds unexplained global diagnostic bits',()=>expectRejected({statusFlags:AMBIGUOUS|256},/unowned diagnostic flags/));
await test('host withholds ambiguity without an owning failed path',()=>expectRejected({plan:[0,0],statusFlags:AMBIGUOUS},{name:'PLTResidentRejected'}));
await test('host withholds non-paused flagged lane',()=>expectRejected({corruptPaused:lane=>{if(lane[11])lane[3]=0;}},/unpaused or nonrecoverable/));
await test('host withholds recovery sample outside the requested prefix',()=>expectRejected({corruptPaused:lane=>{if(lane[11])lane[2]=0;}},/invalid recovery sample ownership/));
await test('host withholds recovery owned by another pixel',()=>expectRejected({corruptPaused:lane=>{if(lane[11])lane[0]=4;}},/invalid recovery sample ownership/));
await test('host withholds edge pool overflow before replay',()=>expectRejected({recordCount:131073},/edge record count overflow/));
await test('host withholds corrupted discarded RNG even after successful CPU replay',()=>expectRejected({corruptRead:(label,bytes)=>{if(label==='Resident audit readback')new Uint32Array(bytes)[16]^=1;}},/discarded path RNG corruption/));
await test('host withholds leaked failed subtotal after CPU replay',()=>expectRejected({corruptRead:(label,bytes,{round})=>{if(label==='Resident lane readback'&&round>1)new Float32Array(bytes)[49]+=6000;}},/retained GPU subtotal/));
await test('cancellation after CPU replay withholds every candidate and prevents the next sample',async()=>{
 const controller=new AbortController();await withRunner({onResume:()=>controller.abort()},async(runner,mock)=>{
  const input={...fresh(),signal:controller.signal},initial=input.rngStates.slice();await assert.rejects(runner.run(input),{name:'AbortError'});
  assert.equal(mock.resumes.length,1);assert.equal(mock.starts.length,1);assert.ok(!mock.reads.includes('Resident audit readback'));assert.deepEqual(input.rngStates,initial);
 });
});
await test('cancellation at final recovered readback withholds reconstructed image',async()=>{
 const controller=new AbortController();await withRunner({onMap:label=>{if(label==='Resident audit readback')controller.abort();}},async(runner,mock)=>{
  await assert.rejects(runner.run({...fresh(),signal:controller.signal}),{name:'AbortError'});assert.equal(mock.resumes.length,1);assert.ok(!mock.reads.includes('Resident audit readback'));
 });
});
await test('cancelled progress callback after replay cannot publish or resume',async()=>{
 const controller=new AbortController();await withRunner({},async(runner,mock)=>{await assert.rejects(runner.run({...fresh(),signal:controller.signal,onProgress:()=>controller.abort()}),{name:'AbortError'});assert.equal(mock.starts.length,1);});
});
await test('device loss after a CPU replay still withholds the entire candidate',async()=>{
 let lose;await withRunner({onResume:()=>lose()},async(runner,mock)=>{lose=mock.lose;await assert.rejects(runner.run(fresh()),/mock replay device loss/);assert.equal(mock.resumes.length,1);assert.ok(!mock.reads.includes('Resident audit readback'));});
});
await test('scene and camera used for CPU replay are snapshotted before later caller mutation',async()=>{
 const sceneInput=structuredClone(scene),cameraInput=structuredClone(scene.camera),input=fresh();
 const expected=await buildResidentReferenceBatch(scene,scene.camera,options,{...input,samplesPerPixel:1,rngStates:input.rngStates.slice()});
 await withRunner({},async runner=>{
  sceneInput.materials.forEach(material=>{material.emission=999;material.albedo=0;});cameraInput.position.fill(NaN);cameraInput.target.fill(NaN);
  const result=await runner.run(input);assert.deepEqual(result.audit.paths[0],expected.audit.paths[0]);
 },{sceneInput,cameraInput});
});
console.log(JSON.stringify({allPass:failures.length===0,passed,failed:failures.length,hostMockOnly:true,syntheticGPUReadbacks:true,actualGPUExecuted:false,failures},null,2));
if(failures.length)process.exitCode=1;
