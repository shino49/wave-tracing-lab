/** v21 tight-anchor transport against the immutable v20 baseline.
 * CPU corpus: node tests/plt-resident-tight-transport.mjs --cpu-only
 * Compile: node tests/plt-resident-tight-transport.mjs
 * Native: WEBGPU_EXECUTION_BACKEND=vulkan VK_ICD_FILENAMES=/usr/lib/chromium/vk_swiftshader_icd.json node tests/plt-resident-tight-transport.mjs --execute
 * Mandatory independent complete CPU shadow. Software execution is not hardware validation.
 */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import crypto from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {create,globals} from 'webgpu';
import {preparePLTScene} from '../dist/plt-geometry.js';
import {createPLTRandomStates,randomGenerator,sampleCamera} from '../dist/plt-core.js';
import {packResidentScene} from '../dist/plt-resident-geometry.js';
import {createPLTResidentRunner,PLT_RESIDENT_BUILD_ID} from '../dist/plt-resident-gpu.js';
import {buildResidentReferenceBatch,compareResidentAudit,residentPathKey} from '../dist/plt-resident-reference.js';
import {admitResidentWithRecovery} from '../dist/plt-resident-recovery.js';
Object.assign(globalThis,globals);
export const options={width:8,height:8,samplesPerPixel:3,maxDepth:6,wave:false,nee:false,seed:7391};
export const fullTile={x:0,y:0,width:8,height:8};
const materials=[{type:'mirror',albedo:.5,emission:0},{type:'mirror',albedo:.75,emission:0},{type:'diffuse',albedo:0,emission:8,twoSidedEmission:true},{type:'mirror',albedo:.25,emission:0}];
function quad(axis,position,lo,hi,materialIndex){
 const a=(axis+1)%3,b=(axis+2)%3,point=(u,v)=>{const p=[0,0,0];p[axis]=position;p[a]=u;p[b]=v;return p;};
 return {mesh:{vertices:[point(lo,lo),point(hi,lo),point(hi,hi),point(lo,hi)],faces:[[0,1,2],[0,2,3]]},materialIndex};
}
function fixture(name,{mirrors=1,emitter=true,scale=1,shift=[0,0,0],perm=[0,1,2],camera,edit,expected,eligible=true,mustRetain=false,mustFallback=false,requireReduction=false}={}){
 const entries=mirrors>=2?[quad(0,2,-8,8,0),quad(2,3,-8,8,1)]:[quad(2,2,-8,8,0)];
 if(mirrors===3)entries.push(quad(1,2.5,-8,8,3));
 if(emitter)entries.push(quad(2,-2,-16,16,2));
 let c=camera??(mirrors===3?{position:[0,0,0],target:[4,3,4],up:[0,1,0],fov:5}:mirrors===2?{position:[0,0,0],target:[4,1,4],up:[0,1,0],fov:10}:{position:[0,0,0],target:[0,0,2],up:[0,1,0],fov:20});
 edit?.(entries,c);
 const point=p=>perm.map((axis,k)=>scale*p[axis]+shift[k]),direction=p=>perm.map(axis=>p[axis]);
 for(const entry of entries)entry.mesh.vertices=entry.mesh.vertices.map(point);
 c={...c,position:point(c.position),target:point(c.target),up:direction(c.up)};
 const scene=preparePLTScene(entries,{materials:structuredClone(materials)});
 scene.emitters=scene.triangles.filter(t=>scene.materials[t.materialIndex].emission>0);scene.emitterArea=scene.emitters.reduce((a,t)=>a+t.area,0);scene.camera=c;
 return {name,scene,camera:c,expected,eligible,mustRetain,mustFallback,requireReduction,mirrors,scale,shift,perm};
}
export const fixtures=[
 fixture('one-mirror-emission',{expected:4,mustRetain:true}),
 fixture('eighth-scale-one-mirror-emission',{scale:.125,expected:4,mustRetain:true,requireReduction:true}),
 fixture('two-mirrors-emission',{mirrors:2,expected:3,mustRetain:true}),
 fixture('quarter-scale-shifted-yzx',{mirrors:2,scale:.25,shift:[16,-8,32],perm:[1,2,0],expected:3,mustRetain:true}),
 fixture('four-scale-shifted-zxy',{mirrors:2,scale:4,shift:[-32,16,-8],perm:[2,0,1],expected:3,mustRetain:true}),
 fixture('one-mirror-escape',{emitter:false,expected:0,mustRetain:true,edit(entries){entries[0].mesh.faces=entries[0].mesh.faces.map(face=>[...face].reverse());}}),
 fixture('seam-after-mirror',{edit(entries,c){
  // Tune pixel 27/sample 0 to the emitter diagonal after its mirror bounce.
  // Geometry remains exact f32; this camera constant receives an outward error radius.
  const pixel=27,rng=randomGenerator((options.seed^Math.imul(pixel+1,0x9E3779B1))>>>0);
  const b=sampleCamera(c,pixel%8,Math.floor(pixel/8),8,8,rng,.025,false);
  const offset=6*(b.direction[1]-b.direction[0])/b.direction[2];c.position[0]=offset;c.target[0]=offset;
 },expected:4}),
 fixture('grazing-mirror-escape',{emitter:false,camera:{position:[0,0,1.984375],target:[64,0,2],up:[0,1,0],fov:5},expected:0}),
 fixture('near-threshold-mirror',{emitter:false,camera:{position:[0,0,-1e-7],target:[0,0,1],up:[0,1,0],fov:5},edit(entries){for(const p of entries[0].mesh.vertices)p[2]=0;},expected:0}),
 fixture('below-near-mirror',{emitter:false,camera:{position:[0,0,-(2**-24)],target:[0,0,1],up:[0,1,0],fov:5},edit(entries){for(const p of entries[0].mesh.vertices)p[2]=0;},expected:0}),
 fixture('long-distance-mirror',{emitter:false,mustFallback:true,camera:{position:[0,0,-256],target:[0,0,2],up:[0,1,0],fov:5},expected:0}),
 fixture('unsupported-tilted-mirror',{eligible:false,edit(entries){entries[0].mesh.vertices[2][2]+=.0625;entries[0].mesh.vertices[3][2]+=.0625;}}),
 fixture('unsupported-decimal-world',{eligible:false,shift:[.1,.1,.1],expected:4}),
 fixture('half-scale-three-mirrors-negative-permutation',{mirrors:3,scale:.5,shift:[-16,8,-32],perm:[2,1,0],expected:.75,mustRetain:true}),
 fixture('small-grid-two-mirrors',{mirrors:2,scale:1/64,shift:[1,-2,3],perm:[0,2,1],expected:3,mustRetain:true}),
 fixture('eight-scale-two-mirrors',{mirrors:2,scale:8,shift:[16,-16,0],perm:[1,0,2],expected:3,mustRetain:true}),
];
export const modes=[
 {id:'anchor-batched',guard:'axis-mirror-anchor',axisSubmission:'batched'},
 {id:'anchor-round',guard:'axis-mirror-anchor',axisSubmission:'round'},
 {id:'tight-batched',guard:'axis-mirror-tight',axisSubmission:'batched'},
 {id:'tight-round',guard:'axis-mirror-tight',axisSubmission:'round'},
];
export const files=['utd-functions.wgsl','plt-resident-geometry.wgsl','plt-resident-interval.wgsl','plt-resident-axis-guard.wgsl','plt-resident-rng.wgsl','plt-resident-fsd.wgsl','plt-resident-walker.wgsl'];
export const hash=x=>crypto.createHash('sha256').update(x).digest('hex');
export const pinnedBaseline={file:'plt-resident-anchor-validation.json',commit:'72fe8da5df129ab22aaa2940586d9811cb6face3',buildId:'plt-resident-anchor-interval-2026-10-13.4',sha256:'e8410d3a98149bb53eed8d57e17d266fb9285abdabc52fa663cbb05c77a0adff',mode:'anchor-batched',fixtureDefinitionSHA256:'e95b5afeb6615d15b813e271898e50bf2ca8bbd5a8eeac4fee3c774929af40f4',paths:4608,replayPaths:1596,retainedMirrorPaths:2913};
export function readBaseline(){
 const source=fs.readFileSync(new URL(import.meta.url),'utf8');
 const declarations=source.slice(source.indexOf('export const options='),source.indexOf('export const modes=')).replace('export const options=','const options=').replace('export const fullTile=','const fullTile=').replace('export const fixtures=','const fixtures=');
 assert.equal(hash(declarations),'e95b5afeb6615d15b813e271898e50bf2ca8bbd5a8eeac4fee3c774929af40f4','Exact fixture declarations must match v20');
 const bytes=fs.readFileSync(new URL('../research/'+pinnedBaseline.file,import.meta.url));
 assert.equal(hash(bytes),pinnedBaseline.sha256,'Pinned v20 report must remain byte-for-byte unchanged');
 const baseline=JSON.parse(bytes);assert(baseline.allPass&&baseline.actualGPUExecuted&&baseline.fullFixtureSuite,'v20 baseline must be a complete executed validation');
 assert.equal(baseline.buildId,pinnedBaseline.buildId);assert.equal(baseline.totals.paths,pinnedBaseline.paths);
 assert.equal(baseline.totals.byMode['anchor-batched'].replayPaths,pinnedBaseline.replayPaths);
 assert.equal(baseline.totals.byMode['anchor-batched'].retainedMirrorPaths,pinnedBaseline.retainedMirrorPaths);
 return baseline;
}
export function sourceProvenance(){
 const sources=files.map(f=>fs.readFileSync(new URL('../dist/'+f,import.meta.url),'utf8'));
 return {sources,sourceSHA256:Object.fromEntries(files.map((f,i)=>[f,hash(sources[i])])),hostSHA256:Object.fromEntries(['plt-resident-gpu.js','plt-resident-geometry.js','plt-resident-recovery.js','plt-resident-reference.js','plt-core.js'].map(f=>[f,hash(fs.readFileSync(new URL('../dist/'+f,import.meta.url)))]))};
}
export async function buildCorpus(selected=fixtures){
 const baseline=readBaseline(),references=new Map(),cpuFixtures=[];
 const additionalSeedCases=new Set(['one-mirror-emission','two-mirrors-emission','quarter-scale-shifted-yzx','half-scale-three-mirrors-negative-permutation']);
 for(const f of selected){
  const packed=packResidentScene(f.scene);assert.equal(packed.metadata.axisGuard.eligible,f.eligible,`${f.name}: source-domain classification`);
  const batches=[],seeds=additionalSeedCases.has(f.name)?[options.seed,1,0xdeadbeef]:[options.seed];
  let mirrorPaths=0,emissions=0,escapes=0,paths=0;
  for(const seed of seeds){
   const o={...options,seed},rng=createPLTRandomStates(o);
   for(const [sampleStart,samplesPerPixel] of [[0,2],[2,1]]){
    const start=rng.slice(),began=performance.now();
    const reference=await buildResidentReferenceBatch(f.scene,f.camera,o,{tile:fullTile,sampleStart,samplesPerPixel,rngStates:rng});
    for(const path of reference.audit.paths){
     paths++;mirrorPaths+=path.scatters.some(s=>s.type==='mirror');emissions+=path.contributions.length;escapes+=path.termination.reason==='escaped';
     if(f.expected!==undefined){assert.equal(path.value,f.expected,`${f.name}: analytic radiance`);assert.equal(path.contributions.length,f.expected>0?1:0,'Analytic contribution multiplicity');}
     assert.equal(path.rng.count,2,'Mirror-only scenes consume exactly two camera draws');
     if(f.mustRetain)assert.equal(path.scatters.filter(s=>s.type==='mirror').length,f.mirrors,`${f.name}: physical mirror count`);
    }
    batches.push({seed,options:o,sampleStart,samplesPerPixel,start,end:rng.slice(),reference,cpuFullShadowConstructionMs:performance.now()-began});
   }
  }
  if(f.mustRetain)assert.equal(mirrorPaths,paths,`${f.name}: all designed paths reflect`);
  references.set(f.name,batches);
  const fixture={name:f.name,seeds,paths,mirrorPaths,contributions:emissions,escapes,expectedRadiance:f.expected,mirrors:f.mirrors,eligibility:packed.metadata.axisGuard,sourceNormals:f.scene.triangles.map(t=>t.normal),scale:f.scale,translation:f.shift,permutation:f.perm};
  assert.deepEqual(JSON.parse(JSON.stringify(fixture)),baseline.cpuFixtures.find(h=>h.name===f.name),'Fixture geometry, seeds, exact source normals and analytic powers must match v20');
  cpuFixtures.push(fixture);
 }
 return {references,cpuFixtures,baseline};
}
export function assertEquivalent(a,b,label='within-guard scheduling parity'){
 for(const field of ['audit','image','pixelIndices','finalRngStates'])assert.deepEqual(b[field],a[field],`${label}: ${field}`);
 assert.deepEqual(b.recovery.keys,a.recovery.keys,`${label}: full CPU replay ownership/reason/depth`);
 for(const field of ['paths'])assert.equal(b.recovery[field],a.recovery[field],`${label}: recovery ${field}`);
 assert.equal(b.rounds,a.rounds,`${label}: round count`);assert.equal(b.dispatches,a.dispatches,`${label}: pass count`);
}
const histogram=(items,value)=>items.reduce((out,item)=>{const key=value(item);out[key]=(out[key]??0)+1;return out;},{});
export function validateCandidate(f,b,candidate,mode,{pixelCount=64,expectedEnd=b.end}={}){
 assert.equal(candidate.candidate,true);assert.equal(candidate.admitted,false);assert.strictEqual(candidate.image,candidate.audit.reference);
 assert.equal(candidate.audit.complete,true);assert.equal(candidate.audit.flags,0);
 assert.equal(candidate.recovery.continuationGuard,mode.guard);assert.equal(candidate.recovery.axisSubmission,mode.axisSubmission);
 const auditStart=performance.now(),rawAudit=compareResidentAudit(b.reference,candidate.audit,{maxMismatches:4096}),rawShadowAuditMs=performance.now()-auditStart;
 assert(rawAudit.ok,`${f.name}/${mode.id}/${b.seed}/${b.sampleStart}: raw independent shadow mismatch ${JSON.stringify(rawAudit.failures.slice(0,8))}`);
 const expectedTolerance=readBaseline().cases[0].batches[0]['anchor-batched'].rawAudit.tolerance;
 assert.deepEqual(rawAudit.tolerance,expectedTolerance,'Strict geometry/power admission tolerances must remain unchanged');
 const admissionStart=performance.now(),admitted=admitResidentWithRecovery(b.reference,candidate),admissionAuditWallMs=performance.now()-admissionStart;
 assert(admitted.validation.passed);assert.equal(admitted.recovery.numericGeometry.count,0,'No audit-discovered numerical replacement');assert.strictEqual(admitted.image,admitted.audit.reference);
 const replayKeys=new Set(candidate.recovery.keys.map(k=>residentPathKey(k.key))),cpuMap=new Map(b.reference.audit.paths.map(p=>[residentPathKey(p.key),p]));
 let retainedMirrorPaths=0,retainedMirrorEmissionPaths=0,retainedMirrorEscapePaths=0,retainedIntervalQueries=0,contributions=0;
 const retainedByMirrorCount={};
 for(const path of candidate.audit.paths){
  const key=residentPathKey(path.key),cpu=cpuMap.get(key);assert(cpu,'No invented path');assert.deepEqual(path.rng,cpu.rng,'Exact sequential RNG words/state');
  assert.equal(path.contributions.length,cpu.contributions.length,'No dropped or duplicated contribution');contributions+=path.contributions.length;
  if(f.expected!==undefined)assert.equal(path.sampleValue??path.value,f.expected,'Dyadic analytic path power must be exact');
  if(!replayKeys.has(key)){
   assert.equal(path.intervalGuard?.unitProvenance,true,'Retained paths require private camera-unit provenance');
   assert.equal(path.intervalGuard?.anchorProvenance,true,'Retained paths require private anchor provenance');
   assert.equal(path.intervalGuard?.tightProvenance??false,mode.guard==='axis-mirror-tight','Private tight provenance must match guard');
   const mirrors=cpu.scatters.filter(s=>s.type==='mirror').length;
   if(mirrors){retainedMirrorPaths++;retainedMirrorEmissionPaths+=cpu.contributions.length>0;retainedMirrorEscapePaths+=cpu.termination.reason==='escaped';retainedByMirrorCount[mirrors]=(retainedByMirrorCount[mirrors]??0)+1;assert(path.intervalGuard.queries>=mirrors+1,'Every retained continuation needs an interval-checked query');retainedIntervalQueries+=path.intervalGuard.queries;}
  }
 }
 for(let i=0;i<candidate.pixelIndices.length;i++)assert.equal(candidate.finalRngStates[i],expectedEnd[candidate.pixelIndices[i]],'Canonical persistent RNG prefix');
 assert.equal(candidate.pixelIndices.length,pixelCount);assert.equal(candidate.image.length,pixelCount);assert.deepEqual(candidate.pixelIndices,b.reference.audit.pixelIndices,'Complete pixel ownership');
 if(f.expected!==undefined)for(const value of candidate.image)assert.equal(value,f.expected,'Complete analytic image must be exact');
 assert.equal(candidate.audit.paths.length,pixelCount*b.samplesPerPixel);assert.equal(new Set(candidate.audit.paths.map(p=>residentPathKey(p.key))).size,candidate.audit.paths.length);
 if(!f.eligible||f.mustFallback)assert.equal(retainedMirrorPaths,0,'Unsupported source/camera domain must fall back');
 return {paths:candidate.audit.paths.length,replayPaths:candidate.recovery.paths,reasons:histogram(candidate.recovery.keys.flatMap(k=>k.reasons),x=>x),triggerDepths:histogram(candidate.recovery.keys,k=>k.triggerDepth??'unknown'),intervalReasons:histogram(candidate.recovery.keys,k=>k.intervalReason??0),rawAudit,rawShadowAuditMs,retainedMirrorPaths,retainedMirrorEmissionPaths,retainedMirrorEscapePaths,retainedByMirrorCount,retainedIntervalQueries,contributions,numericGeometryReplacements:0,rounds:candidate.rounds,dispatches:candidate.dispatches,timings:admitted.timings,admissionAuditWallMs,gate:admitted.validation,auditSHA256:hash(JSON.stringify(candidate.audit)),replayOwnershipSHA256:hash(JSON.stringify(candidate.recovery.keys))};
}
function checkDecodedHistoryMutations(reference,candidate){
 const recovered=new Set(candidate.recovery.keys.map(k=>residentPathKey(k.key))),index=candidate.audit.paths.findIndex(p=>!recovered.has(residentPathKey(p.key))&&p.scatters.filter(s=>s.type==='mirror').length>=2);
 if(index<0)return {exercised:false,reason:'No complete retained two-mirror path',checks:[]};
 const mutations=[['alter-reflected-origin',c=>{c.audit.paths[index].vertices[1].origin[0]+=.125;}],['reverse-first-mirror-direction',c=>{const d=c.audit.paths[index].scatters[0].direction;for(let k=0;k<3;k++)d[k]=-d[k];}],['drop-second-mirror',c=>{c.audit.paths[index].scatters.splice(1,1);}],['reuse-stale-sample-ownership',c=>{c.audit.paths[index].key.sample+=1;}]];
 return {exercised:true,scope:'Synthetic decoded history at independent full-shadow admission boundary; not private GPU proof-state validation',pathKey:candidate.audit.paths[index].key,checks:mutations.map(([name,mutate])=>{const corrupt=structuredClone(candidate);mutate(corrupt);const raw=compareResidentAudit(reference,corrupt.audit,{maxMismatches:4096});assert.equal(raw.ok,false,`${name}: shadow must reject`);assert.throws(()=>admitResidentWithRecovery(reference,corrupt),undefined,`${name}: admission must reject`);return {name,rawShadowRejected:true,admissionRejected:true,mismatches:raw.mismatchCount,firstFailure:raw.failures[0]};})};
}
/** Test-only real-device lease/cache. Numerical execution and resources stay native.
 * Each production runner still owns/frees its buffers and listeners. Only the
 * device destruction boundary is moved to this harness's finalizer. Existing
 * standalone lifecycle tests retain genuine per-runner device ownership.
 */
export function reuseNativeDevice(nativeGPU){
 let adapterPromise=null,devicePromise=null,realDevice=null,activeLease=false,closed=false;
 const shaderCache=new Map(),pipelineCache=new Map(),moduleKeys=new WeakMap(),disposedDevices=new WeakSet();
 const stats={enabled:true,sequentialLeases:true,realExecutionPreserved:true,scope:'Test harness only; shared shader/pipeline/device setup, genuine per-scene buffers/bind groups/dispatch/readback/error scopes',adapterRequests:0,realAdapterRequests:0,deviceLeases:0,realDeviceRequests:0,releasedLeases:0,realDeviceDestroyCalls:0,shaderHits:0,shaderMisses:0,pipelineHits:0,pipelineMisses:0,shaderEntries:[],pipelineEntries:[],shaderCreationWallMs:0,pipelineCreationWallMs:0};
 const dispose=target=>{if(target&&!disposedDevices.has(target)){disposedDevices.add(target);stats.realDeviceDestroyCalls++;target.destroy();}};
 const canonical=value=>JSON.stringify(value,(_,v)=>v&&typeof v==='object'&&!Array.isArray(v)?Object.fromEntries(Object.keys(v).sort().map(k=>[k,v[k]])):v);
 const forward=(target,property)=>{const value=Reflect.get(target,property,target);return typeof value==='function'?value.bind(target):value;};
 const shader=descriptor=>{
  assert.equal(typeof descriptor.code,'string','Cache supports WGSL source text only');
  assert(Object.keys(descriptor).every(k=>['code','label'].includes(k)),'Unsupported shader descriptor must not be silently cached');
  const key=canonical(descriptor);if(shaderCache.has(key)){stats.shaderHits++;return shaderCache.get(key);}
  const began=performance.now(),module=realDevice.createShaderModule(descriptor);stats.shaderCreationWallMs+=performance.now()-began;
  const entry={descriptorSHA256:hash(key),sourceSHA256:hash(descriptor.code),label:descriptor.label??null,sourceBytes:Buffer.byteLength(descriptor.code)};
  moduleKeys.set(module,{key,entry});shaderCache.set(key,module);stats.shaderMisses++;stats.shaderEntries.push(entry);return module;
 };
 const pipeline=async descriptor=>{
  assert.equal(descriptor.layout,'auto','Only exact automatic pipeline layout is eligible for this cache');
  assert(Object.keys(descriptor).every(k=>['label','layout','compute'].includes(k)),'Unsupported pipeline descriptor');
  assert(Object.keys(descriptor.compute).every(k=>['module','entryPoint','constants'].includes(k)),'Unsupported compute descriptor');
  const source=moduleKeys.get(descriptor.compute.module);assert(source,'Pipeline module must come from this exact native module cache');
  const keyed={shaderDescriptor:source.key,label:descriptor.label??null,layout:'auto',entryPoint:descriptor.compute.entryPoint,constants:descriptor.compute.constants??{}},key=canonical(keyed);
  if(pipelineCache.has(key)){stats.pipelineHits++;return pipelineCache.get(key);}
  const entry={keySHA256:hash(key),sourceSHA256:source.entry.sourceSHA256,shaderDescriptorSHA256:source.entry.descriptorSHA256,label:keyed.label,layout:'auto',entryPoint:keyed.entryPoint,constants:keyed.constants};
  stats.pipelineMisses++;stats.pipelineEntries.push(entry);
  const began=performance.now(),pending=realDevice.createComputePipelineAsync(descriptor);pipelineCache.set(key,pending);
  try{return await pending;}catch(error){pipelineCache.delete(key);throw error;}finally{stats.pipelineCreationWallMs+=performance.now()-began;}
 };
 const lease=async descriptor=>{
  assert(!closed&&!activeLease,'Native device leases must remain sequential');
  // No cross-descriptor device reuse: the production runner requests no extras.
  assert(descriptor===undefined,'Unexpected device requirements must not be silently reused');
  activeLease=true;stats.deviceLeases++;
  if(!devicePromise){stats.realDeviceRequests++;devicePromise=adapterPromise.then(adapter=>adapter.requestDevice());}
  try{const acquired=await devicePromise;if(closed){dispose(acquired);throw new Error('Native device acquired after cache closure');}realDevice=acquired;}catch(error){activeLease=false;throw error;}
  let released=false;
  return new Proxy(realDevice,{get(target,property){
   if(property==='destroy')return ()=>{if(!released){released=true;activeLease=false;stats.releasedLeases++;}};
   if(property==='createShaderModule')return shader;
   if(property==='createComputePipelineAsync')return pipeline;
   return forward(target,property);
  }});
 };
 let adapterOptionsKey=null;
 const gpu={async requestAdapter(options){
  assert(!closed,'Native cache is closed');stats.adapterRequests++;
  const key=canonical(options??{});if(adapterOptionsKey!==null)assert.equal(key,adapterOptionsKey,'Adapter options cannot change inside a cache lease');adapterOptionsKey=key;
  if(!adapterPromise){stats.realAdapterRequests++;adapterPromise=nativeGPU.requestAdapter(options);}
  const adapter=await adapterPromise;if(!adapter)return null;
  return new Proxy(adapter,{get(target,property){return property==='requestDevice'?lease:forward(target,property);}});
 }};
 return {gpu,stats,destroy(){
  if(closed)return;closed=true;const unreleased=activeLease,pending=devicePromise;
  // Dispose now or on late acquisition even when the lease assertion fails.
  // The WeakSet ensures the finalizer and awaiting lease cannot double-destroy.
  try{dispose(realDevice);}finally{pending?.then(dispose,()=>{});shaderCache.clear();pipelineCache.clear();realDevice=null;devicePromise=null;adapterPromise=null;nativeGPU=null;}
  assert(!unreleased,'A production runner has not released its native device lease');
 }};
}
export async function main(){
 const execute=process.argv.includes('--execute'),cpuOnly=process.argv.includes('--cpu-only'),reuseDevice=process.argv.includes('--reuse-native-device');assert(!(execute&&cpuOnly),'Choose native or CPU-only execution');
 const backend=execute?process.env.WEBGPU_EXECUTION_BACKEND:'null';if(execute)assert(['vulkan','metal','d3d11','d3d12','opengl','opengles'].includes(backend),'--execute requires an explicit executable backend');
 const caseNames=process.env.PLT_TIGHT_CASES?.split(',')??(process.env.PLT_TIGHT_CASE?[process.env.PLT_TIGHT_CASE]:null),selected=caseNames?fixtures.filter(f=>caseNames.includes(f.name)):fixtures;assert(selected.length&&(!caseNames||selected.length===new Set(caseNames).size),'Unknown PLT_TIGHT_CASE(S)');const caseLabel=caseNames?caseNames.join('--')+'-':'';
 const {references,cpuFixtures,baseline}=await buildCorpus(selected),provenance=sourceProvenance();
 const report={buildId:PLT_RESIDENT_BUILD_ID,executionRequested:execute,actualGPUExecuted:false,allPass:false,status:'preparing',backend,nodeVersion:process.version,vulkanICD:process.env.VK_ICD_FILENAMES??null,fullCPUShadowRequired:true,physicalGPUValidated:false,browserValidated:false,hardwarePerformanceClaim:false,modes,baseline:pinnedBaseline,testSHA256:hash(fs.readFileSync(new URL(import.meta.url))),sourceSHA256:provenance.sourceSHA256,hostSHA256:provenance.hostSHA256,cpuFixtures,cpuFixtureCorpusMatchesV20:true,selectedFixtures:selected.map(f=>f.name),fullFixtureSuite:selected.length===fixtures.length,cases:[],limitations:['Mandatory complete independent CPU shadow; no numerical geometry replacement','Four modes use one shared tight-compiled runner per fixture; anchor mode skips tight proof arithmetic, not a separately compiled historical v20 runner','Batched/round pairing changes only submission grouping and retains per-lane-round status/replay checkpoints','Compile/setup, full CPU shadow, renderer wall time, replay, readbacks and admission are reported separately','Descriptive software-adapter execution does not establish browser or physical-GPU performance']};
 const save=()=>fs.writeFileSync(new URL(`../research/plt-resident-tight-${caseLabel}${cpuOnly?'cpu-validation':execute?'validation':'compile-validation'}.json`,import.meta.url),JSON.stringify(report,null,2)+'\n');
 if(cpuOnly){report.allPass=true;report.status='cpu-fixtures-passed';report.totals={paths:cpuFixtures.reduce((n,f)=>n+f.paths,0),fixtures:cpuFixtures.length};if(report.fullFixtureSuite)assert.equal(report.totals.paths,4608);save();console.log(JSON.stringify({allPass:true,cpuOnly:true,totals:report.totals,cpuFixtureCorpusMatchesV20:true}));return;}
 report.status='compiling';let gpu=create([`backend=${backend}`]);const nativeCache=reuseDevice?reuseNativeDevice(gpu):null;if(nativeCache)gpu=nativeCache.gpu;report.nativeDeviceReuse=nativeCache?.stats??{enabled:false,scope:'Default genuine per-runner device ownership; no module or pipeline cache'};save();
 try{
  for(const f of execute?selected:selected.slice(0,reuseDevice?2:1)){
   const setup=performance.now(),runner=await createPLTResidentRunner(f.scene,f.camera,options,{gpu,shaderSource:provenance.sources.join('\n'),timeoutMs:120000,recoveryMode:'selective',continuationGuard:'axis-mirror-tight',onSetupProgress:step=>{if(execute)console.log(JSON.stringify({setup:f.name,...step}));}});
   const record={name:f.name,setupWallMs:performance.now()-setup,adapter:runner.diagnostic.adapter,eligibility:runner.packedScene.metadata.axisGuard,batches:[]};report.cases.push(record);
   try{if(execute){
    for(const b of references.get(f.name)){
     const results=new Map(),comparison={seed:b.seed,sampleStart:b.sampleStart,samplesPerPixel:b.samplesPerPixel,cpuFullShadowConstructionMs:b.cpuFullShadowConstructionMs};record.batches.push(comparison);
     for(const mode of modes){
      const input=b.start.slice(),candidate=await runner.run({tile:fullTile,sampleStart:b.sampleStart,samplesPerPixel:b.samplesPerPixel,wave:false,rngStates:input,guard:mode.guard,axisSubmission:mode.axisSubmission});
      report.actualGPUExecuted=true;report.status='validating';assert.deepEqual(input,b.start,'Transactional input RNG mutated');
      comparison[mode.id]=validateCandidate(f,b,candidate,mode);results.set(mode.id,candidate);save();
      if(f.name==='two-mirrors-emission'&&b.seed===options.seed&&b.sampleStart===0&&mode.id==='tight-batched')report.decodedHistoryNegativeControls=checkDecodedHistoryMutations(b.reference,candidate);
     }
     const parityStart=performance.now();for(const guard of ['anchor','tight'])assertEquivalent(results.get(guard+'-batched'),results.get(guard+'-round'),`${f.name}/${b.seed}/${b.sampleStart}/${guard}`);
     comparison.exactWithinGuardAuditRngOwnershipParity=true;comparison.equivalenceCheckMs=performance.now()-parityStart;
     const historical=baseline.cases.find(c=>c.name===f.name)?.batches.find(h=>h.seed===b.seed&&h.sampleStart===b.sampleStart&&h.samplesPerPixel===b.samplesPerPixel)?.['anchor-batched'];assert(historical,'Missing exact v20 baseline batch');
     const checkpointFields=['paths','replayPaths','retainedMirrorPaths','retainedMirrorEmissionPaths','retainedMirrorEscapePaths','retainedByMirrorCount','retainedIntervalQueries','contributions','reasons','triggerDepths','intervalReasons','rounds','dispatches'];
     comparison.anchorBaselineCheckpoint=Object.fromEntries(checkpointFields.map(key=>[key,historical[key]]));
     for(const key of checkpointFields)assert.deepEqual(comparison['anchor-batched'][key],historical[key],`Unchanged v20 anchor baseline: ${f.name}/${b.seed}/${b.sampleStart}/${key}`);
     comparison.anchorBaselineCheckpointIdentical=true;comparison.tightReplayReduction=comparison['anchor-batched'].replayPaths-comparison['tight-batched'].replayPaths;
     save();console.log(JSON.stringify({name:f.name,seed:b.seed,sampleStart:b.sampleStart,anchorReplay:comparison['anchor-batched'].replayPaths,tightReplay:comparison['tight-batched'].replayPaths,tightRetainedMirrorPaths:comparison['tight-batched'].retainedMirrorPaths,exactWithinGuardParity:true,numericReplacements:0}));
    }
    if(f.name==='one-mirror-emission'){
     const tile={x:3,y:3,width:2,height:2},initial=createPLTRandomStates(options),cpuRng=initial.slice(),reference=await buildResidentReferenceBatch(f.scene,f.camera,{...options,wave:true},{tile,sampleStart:0,samplesPerPixel:1,rngStates:cpuRng}),results=new Map();
     for(const mode of modes){
      const rng=initial.slice(),candidate=await runner.run({tile,sampleStart:0,samplesPerPixel:1,wave:true,rngStates:rng,guard:mode.guard,axisSubmission:mode.axisSubmission});assert.deepEqual(rng,initial);assert.equal(candidate.recovery.paths,4);assert(candidate.recovery.keys.every(k=>k.reasons.includes('unmodelledContinuation')),'Wave must reject bounded anchor/tight provenance');
      const rawAudit=compareResidentAudit(reference,candidate.audit,{maxMismatches:4096}),admitted=admitResidentWithRecovery(reference,candidate);assert(rawAudit.ok);assert(admitted.validation.passed);assert.equal(admitted.recovery.numericGeometry.count,0);for(let i=0;i<candidate.pixelIndices.length;i++)assert.equal(candidate.finalRngStates[i],cpuRng[candidate.pixelIndices[i]]);results.set(mode.id,candidate);
     }
     for(const guard of ['anchor','tight'])assertEquivalent(results.get(guard+'-batched'),results.get(guard+'-round'),'unsupported wave domain');
     record.waveDomainRejection={paths:4,allModesReplayPaths:4,rawShadowAllPass:true,numericGeometryReplacements:0,exactWithinGuardAuditRngOwnershipParity:true};save();
    }
   }}finally{runner.destroy();}
  }
  if(execute){
   const batches=report.cases.flatMap(c=>c.batches);report.rawShadowAllPass=true;report.exactWithinGuardAuditRngOwnershipParity=true;report.anchorBaselineCountsPreserved=true;
   const sum=(id,field)=>batches.reduce((n,b)=>n+b[id][field],0),timing=(id,field)=>batches.reduce((n,b)=>n+(b[id].timings[field]??0),0);
   report.totals={paths:sum('anchor-batched','paths'),waveDomainPaths:report.cases.reduce((n,c)=>n+(c.waveDomainRejection?.paths??0),0),cpuFullShadowConstructionMs:batches.reduce((n,b)=>n+b.cpuFullShadowConstructionMs,0),sharedSetupWallMs:report.cases.reduce((n,c)=>n+c.setupWallMs,0),byMode:Object.fromEntries(modes.map(({id})=>[id,{...Object.fromEntries(['replayPaths','retainedMirrorPaths','retainedMirrorEmissionPaths','retainedMirrorEscapePaths','numericGeometryReplacements','rawShadowAuditMs','admissionAuditWallMs'].map(field=>[field,sum(id,field)])),...Object.fromEntries(['gpuResidentTotalWallMs','cpuSelectiveReplayMs','gpuStatusReadbackWallMs','replayReadbackWallMs','gpuFinalReadbackDecodeMs','gpuRoundSubmitAndWaitMs'].map(field=>[field,timing(id,field)]))}]))};
   report.fixtureRetention=report.cases.map(c=>{const count=(id,key)=>c.batches.reduce((n,b)=>n+b[id][key],0),anchorReplay=count('anchor-batched','replayPaths'),tightReplay=count('tight-batched','replayPaths');return {name:c.name,paths:count('anchor-batched','paths'),anchorReplay,tightReplay,replayReduction:anchorReplay-tightReplay,anchorRetainedMirrorPaths:count('anchor-batched','retainedMirrorPaths'),tightRetainedMirrorPaths:count('tight-batched','retainedMirrorPaths'),tightRetainedByMirrorCount:c.batches.reduce((out,b)=>{for(const [count,paths] of Object.entries(b['tight-batched'].retainedByMirrorCount))out[count]=(out[count]??0)+paths;return out;},{})};});
   report.multibounceGoals=['two-mirrors-emission','four-scale-shifted-zxy','half-scale-three-mirrors-negative-permutation','eight-scale-two-mirrors'].map(name=>{const gain=report.fixtureRetention.find(c=>c.name===name);return gain?{...gain,executed:true,met:gain.replayReduction>0&&gain.tightRetainedMirrorPaths>0}:{name,executed:false,met:false};});
   report.allMultibounceGoalsMet=report.multibounceGoals.every(g=>g.met);report.goalMisses=report.multibounceGoals.filter(g=>g.executed&&!g.met).map(g=>g.name);report.totalReplayReduction=report.totals.byMode['anchor-batched'].replayPaths-report.totals.byMode['tight-batched'].replayPaths;
   report.retentionRegressions=report.fixtureRetention.filter(g=>g.replayReduction<0).map(g=>g.name);
   // Retention improvements and misses are outcomes, never substitutes for correctness.
   if(report.fullFixtureSuite){assert.equal(report.totals.paths,4608);assert.equal(report.totals.waveDomainPaths,4);assert.equal(report.totals.byMode['anchor-batched'].replayPaths,1596);assert.equal(report.totals.byMode['anchor-batched'].retainedMirrorPaths,2913);}
  }
  readBaseline();report.allPass=true;report.status='passed';
 }catch(error){report.allPass=false;report.status='failed';report.failure={name:error.name,message:error.message,stack:error.stack};throw error;}
 finally{try{nativeCache?.destroy();}catch(error){report.allPass=false;report.status='failed';report.failure??={name:error.name,message:error.message,stack:error.stack};throw error;}finally{gpu=null;save();}}
 console.log(JSON.stringify({allPass:report.allPass,actualGPUExecuted:report.actualGPUExecuted,fixtures:report.cases.length,totals:report.totals,goalMisses:report.goalMisses}));
}
if(process.argv[1]&&pathToFileURL(process.argv[1]).href===import.meta.url)await main();
