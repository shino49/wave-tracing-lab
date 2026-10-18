/** Anchored mirror transport against the preserved v19 camera-unit baseline.
 * Mandatory independent complete CPU shadow; fixture corpus is unchanged.
 * Compile: node tests/plt-resident-anchor-transport.mjs
 * Execute: WEBGPU_EXECUTION_BACKEND=vulkan \
 * VK_ICD_FILENAMES=/usr/lib/chromium/vk_swiftshader_icd.json \
 * node tests/plt-resident-anchor-transport.mjs --execute
 * Software Vulkan execution is not physical-GPU or browser validation.
 */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import crypto from 'node:crypto';
import {create,globals} from 'webgpu';
import {preparePLTScene} from '../dist/plt-geometry.js';
import {createPLTRandomStates,randomGenerator,sampleCamera} from '../dist/plt-core.js';
import {packResidentScene} from '../dist/plt-resident-geometry.js';
import {createPLTResidentRunner,PLT_RESIDENT_BUILD_ID} from '../dist/plt-resident-gpu.js';
import {buildResidentReferenceBatch,compareResidentAudit,residentPathKey} from '../dist/plt-resident-reference.js';
import {admitResidentWithRecovery} from '../dist/plt-resident-recovery.js';
Object.assign(globalThis,globals);
const execute=process.argv.includes('--execute'),cpuOnly=process.argv.includes('--cpu-only');
assert(!(execute&&cpuOnly),'Choose native execution or CPU fixture inspection');
const backend=execute?process.env.WEBGPU_EXECUTION_BACKEND:'null';
if(execute)assert(['vulkan','metal','d3d11','d3d12','opengl','opengles'].includes(backend),'--execute needs an explicit executable backend; null is compile-only');
const files=['utd-functions.wgsl','plt-resident-geometry.wgsl','plt-resident-interval.wgsl','plt-resident-axis-guard.wgsl','plt-resident-rng.wgsl','plt-resident-fsd.wgsl','plt-resident-walker.wgsl'];
const options={width:8,height:8,samplesPerPixel:3,maxDepth:6,wave:false,nee:false,seed:7391};
const fullTile={x:0,y:0,width:8,height:8};
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
const fixtures=[
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
const modes=[
 {id:'unit-staged',guard:'axis-mirror-unit',axisSubmission:'staged'},
 {id:'anchor-staged',guard:'axis-mirror-anchor',axisSubmission:'staged'},
 {id:'anchor-batched',guard:'axis-mirror-anchor',axisSubmission:'batched'},
];
const report={buildId:PLT_RESIDENT_BUILD_ID,executionRequested:execute,actualGPUExecuted:false,allPass:false,status:'preparing',backend,fullCPUShadowRequired:true,physicalGPUValidated:false,browserValidated:false,hardwarePerformanceClaim:false,modes,cases:[],cpuFixtures:[],limitations:['Conditional anchored camera/mirror provenance; mandatory complete independent CPU shadow','Staged/batched comparisons change only guard submission grouping','Shared setup and CPU shadow construction are reported separately from run/readback/replay/audit costs','Native software execution does not establish browser or physical-GPU performance']};
const selected=process.env.PLT_ANCHOR_CASE?fixtures.filter(f=>f.name===process.env.PLT_ANCHOR_CASE):fixtures;
assert(selected.length,'Unknown PLT_ANCHOR_CASE');
report.selectedFixtures=selected.map(f=>f.name);report.fullFixtureSuite=selected.length===fixtures.length;
report.nodeVersion=process.version;report.vulkanICD=process.env.VK_ICD_FILENAMES??null;
const baselineURL=new URL('../research/plt-resident-unit-validation.json',import.meta.url),baselineBytes=fs.readFileSync(baselineURL),baseline=JSON.parse(baselineBytes);
assert(baseline.allPass&&baseline.actualGPUExecuted&&baseline.fullFixtureSuite,'Preserved v19 baseline must be a complete executed validation');
report.baseline={file:'plt-resident-unit-validation.json',buildId:baseline.buildId,sha256:crypto.createHash('sha256').update(baselineBytes).digest('hex'),paths:baseline.totals.paths,mode:'unit-staged'};
report.testSHA256=crypto.createHash('sha256').update(fs.readFileSync(new URL(import.meta.url))).digest('hex');
const additionalSeedCases=new Set(['one-mirror-emission','two-mirrors-emission','quarter-scale-shifted-yzx','half-scale-three-mirrors-negative-permutation']);
const references=new Map();
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
    if(f.expected!==undefined){assert.equal(path.value,f.expected,`${f.name}: independent analytic radiance`);assert.equal(path.contributions.length,f.expected>0?1:0,'Analytic contribution multiplicity');}
    assert.equal(path.rng.count,2,'Mirror-only scenes consume exactly two camera draws');
    if(f.mustRetain)assert.equal(path.scatters.filter(s=>s.type==='mirror').length,f.mirrors,`${f.name}: physical mirror count`);
   }
   batches.push({seed,options:o,sampleStart,samplesPerPixel,start,end:rng.slice(),reference,cpuFullShadowConstructionMs:performance.now()-began});
  }
 }
 if(f.mustRetain)assert.equal(mirrorPaths,paths,`${f.name}: all designed paths reflect`);
 references.set(f.name,batches);
 report.cpuFixtures.push({name:f.name,seeds,paths,mirrorPaths,contributions:emissions,escapes,expectedRadiance:f.expected,mirrors:f.mirrors,eligibility:packed.metadata.axisGuard,sourceNormals:f.scene.triangles.map(t=>t.normal),scale:f.scale,translation:f.shift,permutation:f.perm});
}
for(const fixture of report.cpuFixtures){const historical=baseline.cpuFixtures.find(f=>f.name===fixture.name);assert(historical,'Missing v19 CPU fixture');assert.deepEqual(JSON.parse(JSON.stringify(fixture)),historical,'CPU fixture definitions, seeds and exact source normals must match v19');}
report.cpuFixtureCorpusMatchesV19=true;
if(cpuOnly){console.log(JSON.stringify({cpuFixtures:report.cpuFixtures,allPass:true,cpuFixtureCorpusMatchesV19:true}));process.exit(0);}
const sources=files.map(f=>fs.readFileSync(new URL('../dist/'+f,import.meta.url),'utf8'));
report.sourceSHA256=Object.fromEntries(files.map((f,i)=>[f,crypto.createHash('sha256').update(sources[i]).digest('hex')]));
report.hostSHA256=Object.fromEntries(['plt-resident-gpu.js','plt-resident-geometry.js','plt-resident-recovery.js','plt-resident-reference.js','plt-core.js'].map(f=>[f,crypto.createHash('sha256').update(fs.readFileSync(new URL('../dist/'+f,import.meta.url))).digest('hex')]));
const saveReport=()=>fs.writeFileSync(new URL(`../research/plt-resident-anchor-${execute?'validation':'compile-validation'}.json`,import.meta.url),JSON.stringify(report,null,2)+'\n');
const histogram=(items,value)=>items.reduce((out,item)=>{const key=value(item);out[key]=(out[key]??0)+1;return out;},{});
const replaySummary=candidate=>({paths:candidate.audit.paths.length,replayPaths:candidate.recovery.paths,reasons:histogram(candidate.recovery.keys.flatMap(k=>k.reasons),x=>x),triggerDepths:histogram(candidate.recovery.keys,k=>k.triggerDepth??'unknown'),intervalReasons:histogram(candidate.recovery.keys,k=>k.intervalReason??0)});
function assertEquivalent(staged,batched,label){
 assert.deepEqual(batched.audit,staged.audit,`${label}: submission grouping changed the complete audit`);
 assert.deepEqual(batched.image,staged.image,`${label}: pixel sums differ`);
 assert.deepEqual(batched.pixelIndices,staged.pixelIndices,`${label}: pixel ownership differs`);
 assert.deepEqual(batched.finalRngStates,staged.finalRngStates,`${label}: final RNG differs`);
 assert.deepEqual(batched.recovery.keys,staged.recovery.keys,`${label}: CPU replay ownership/reason/depth differs`);
 assert.equal(batched.recovery.paths,staged.recovery.paths,`${label}: replay count differs`);
 assert.equal(batched.rounds,staged.rounds,`${label}: round count differs`);
 assert.equal(batched.dispatches,staged.dispatches,`${label}: pass count differs`);
}
// These mutations exercise independent shadow/admission at the decoded boundary.
// They are not claimed as proof of private GPU anchor-state validation.
function checkDecodedHistoryMutations(reference,candidate){
 const recovered=new Set(candidate.recovery.keys.map(k=>residentPathKey(k.key))),index=candidate.audit.paths.findIndex(p=>!recovered.has(residentPathKey(p.key))&&p.scatters.filter(s=>s.type==='mirror').length>=2);
 if(index<0)return {exercised:false,reason:'No complete retained two-mirror path for decoded-history controls',checks:[]};
 const mutations=[
  ['alter-reflected-origin',c=>{c.audit.paths[index].vertices[1].origin[0]+=.125;}],
  ['reverse-first-mirror-direction',c=>{const d=c.audit.paths[index].scatters[0].direction;for(let k=0;k<3;k++)d[k]=-d[k];}],
  ['drop-second-mirror',c=>{c.audit.paths[index].scatters.splice(1,1);}],
  ['reuse-stale-sample-ownership',c=>{c.audit.paths[index].key.sample+=1;}],
 ];
 const checks=mutations.map(([name,mutate])=>{const corrupt=structuredClone(candidate);mutate(corrupt);const raw=compareResidentAudit(reference,corrupt.audit,{maxMismatches:4096});assert.equal(raw.ok,false,`${name}: independent shadow must reject history corruption`);assert.throws(()=>admitResidentWithRecovery(reference,corrupt),undefined,`${name}: admission must reject history corruption`);return {name,rawShadowRejected:true,admissionRejected:true,mismatches:raw.mismatchCount,firstFailure:raw.failures[0]};});
 return {exercised:true,scope:'synthetic decoded history, independent full-shadow and admission boundary',pathKey:candidate.audit.paths[index].key,checks};
}
report.status='compiling';saveReport();
let gpu=create([`backend=${backend}`]);
try{
 for(const f of execute?selected:selected.slice(0,1)){
  const setup=performance.now();
  const runner=await createPLTResidentRunner(f.scene,f.camera,options,{gpu,shaderSource:sources.join('\n'),timeoutMs:120000,recoveryMode:'selective',continuationGuard:'axis-mirror-anchor',onSetupProgress:step=>{if(execute)console.log(JSON.stringify({setup:f.name,...step}));}});
  const record={name:f.name,setupWallMs:performance.now()-setup,adapter:runner.diagnostic.adapter,eligibility:runner.packedScene.metadata.axisGuard,batches:[]};report.cases.push(record);
  try{if(execute){
   for(const b of references.get(f.name)){
    const results=new Map(),comparison={seed:b.seed,sampleStart:b.sampleStart,samplesPerPixel:b.samplesPerPixel,cpuFullShadowConstructionMs:b.cpuFullShadowConstructionMs};record.batches.push(comparison);
    for(const mode of modes){
     const input=b.start.slice(),candidate=await runner.run({tile:fullTile,sampleStart:b.sampleStart,samplesPerPixel:b.samplesPerPixel,wave:false,rngStates:input,guard:mode.guard,axisSubmission:mode.axisSubmission});
     report.actualGPUExecuted=true;report.status='validating';
     assert.deepEqual(input,b.start,'Transactional input RNG was mutated');
     assert.equal(candidate.candidate,true);assert.equal(candidate.admitted,false);assert.strictEqual(candidate.image,candidate.audit.reference);
     assert.equal(candidate.audit.complete,true);assert.equal(candidate.audit.flags,0);
     assert.equal(candidate.recovery.continuationGuard,mode.guard);assert.equal(candidate.recovery.axisSubmission,mode.axisSubmission);
     const auditStart=performance.now(),rawAudit=compareResidentAudit(b.reference,candidate.audit,{maxMismatches:4096}),rawShadowAuditMs=performance.now()-auditStart;
     comparison[mode.id]={...replaySummary(candidate),rawAudit,rawShadowAuditMs};saveReport();
     assert(rawAudit.ok,`${f.name}/${mode.id}/${b.seed}/${b.sampleStart}: raw shadow mismatch ${JSON.stringify(rawAudit.failures.slice(0,8))}`);
     const admissionStart=performance.now(),admitted=admitResidentWithRecovery(b.reference,candidate),admissionAuditWallMs=performance.now()-admissionStart;
     assert(admitted.validation.passed);assert.equal(admitted.recovery.numericGeometry.count,0,'No audit-discovered numerical replacement permitted');
     assert.strictEqual(admitted.image,admitted.audit.reference);
     const replayKeys=new Set(candidate.recovery.keys.map(k=>residentPathKey(k.key))),cpuMap=new Map(b.reference.audit.paths.map(p=>[residentPathKey(p.key),p]));
     let retainedMirrorPaths=0,retainedMirrorEmissionPaths=0,retainedMirrorEscapePaths=0,retainedIntervalQueries=0,contributions=0;
     const retainedByMirrorCount={};
     for(const path of candidate.audit.paths){
      const key=residentPathKey(path.key),cpu=cpuMap.get(key);assert(cpu,'No invented path');assert.deepEqual(path.rng,cpu.rng,'Exact sequential RNG words/state required');
      assert.equal(path.contributions.length,cpu.contributions.length,'No dropped or duplicated contribution');contributions+=path.contributions.length;
      if(f.expected!==undefined)assert.equal(path.sampleValue??path.value,f.expected,'Dyadic analytic power must be exact');
      if(!replayKeys.has(key)){
       assert.equal(path.intervalGuard?.unitProvenance,true,'Every retained unit or anchor path requires private camera-unit provenance');
       assert.equal(path.intervalGuard?.anchorProvenance??false,mode.guard==='axis-mirror-anchor','Private anchor provenance must match selected guard');
       const mirrors=cpu.scatters.filter(s=>s.type==='mirror').length;
       if(mirrors){retainedMirrorPaths++;retainedMirrorEmissionPaths+=cpu.contributions.length>0;retainedMirrorEscapePaths+=cpu.termination.reason==='escaped';retainedByMirrorCount[mirrors]=(retainedByMirrorCount[mirrors]??0)+1;assert(path.intervalGuard.queries>=mirrors+1,'Every retained continuation requires an interval-checked query');retainedIntervalQueries+=path.intervalGuard.queries;}
      }
     }
     for(let i=0;i<candidate.pixelIndices.length;i++)assert.equal(candidate.finalRngStates[i],b.end[candidate.pixelIndices[i]],'Canonical persistent RNG prefix');
     assert.equal(candidate.pixelIndices.length,64);assert.equal(candidate.image.length,64);
     assert.deepEqual(candidate.pixelIndices,b.reference.audit.pixelIndices,'Complete pixel ownership matches CPU shadow');
     if(f.expected!==undefined)for(const value of candidate.image)assert.equal(value,f.expected,'Complete analytic image must be exact');
     assert.equal(candidate.audit.paths.length,64*b.samplesPerPixel);assert.equal(new Set(candidate.audit.paths.map(p=>residentPathKey(p.key))).size,candidate.audit.paths.length);
     if(!f.eligible||f.mustFallback)assert.equal(retainedMirrorPaths,0,'Unsupported source/camera domain must fall back');
     comparison[mode.id]={...comparison[mode.id],retainedMirrorPaths,retainedMirrorEmissionPaths,retainedMirrorEscapePaths,retainedByMirrorCount,retainedIntervalQueries,contributions,numericGeometryReplacements:0,rounds:candidate.rounds,dispatches:candidate.dispatches,timings:admitted.timings,admissionAuditWallMs,gate:admitted.validation};
     if(f.name==='two-mirrors-emission'&&b.seed===options.seed&&b.sampleStart===0&&mode.id==='anchor-staged'){report.decodedHistoryNegativeControls=checkDecodedHistoryMutations(b.reference,candidate);saveReport();}
     results.set(mode.id,candidate);
    }
    const parityStart=performance.now();assertEquivalent(results.get('anchor-staged'),results.get('anchor-batched'),`${f.name}/${b.seed}/${b.sampleStart}`);
    comparison.anchorSubmissionIdentical=true;comparison.anchorEquivalenceCheckMs=performance.now()-parityStart;
    const historical=baseline.cases.find(c=>c.name===f.name)?.batches.find(h=>h.seed===b.seed&&h.sampleStart===b.sampleStart&&h.samplesPerPixel===b.samplesPerPixel);
    assert(historical,'Missing exact v19 baseline batch');
    comparison.unitBaselineCheckpoint={replayPaths:historical['unit-staged'].replayPaths,retainedMirrorPaths:historical['unit-staged'].retainedMirrorPaths};
    assert.equal(comparison['unit-staged'].replayPaths,comparison.unitBaselineCheckpoint.replayPaths,'Unchanged unit baseline replay ownership count');
    assert.equal(comparison['unit-staged'].retainedMirrorPaths,comparison.unitBaselineCheckpoint.retainedMirrorPaths,'Unchanged unit baseline retained mirrors');
    comparison.unitBaselineCheckpointIdentical=true;
    comparison.anchorReplayReduction=comparison['unit-staged'].replayPaths-comparison['anchor-staged'].replayPaths;
    saveReport();console.log(JSON.stringify({name:f.name,seed:b.seed,sampleStart:b.sampleStart,unitReplay:comparison['unit-staged'].replayPaths,anchorReplay:comparison['anchor-staged'].replayPaths,retainedMirrorPaths:comparison['anchor-staged'].retainedMirrorPaths,stagedBatchedIdentical:true,numericReplacements:0}));
   }
   if(f.name==='one-mirror-emission'){
    const tile={x:3,y:3,width:2,height:2},initial=createPLTRandomStates(options),cpuRng=initial.slice();
    const reference=await buildResidentReferenceBatch(f.scene,f.camera,{...options,wave:true},{tile,sampleStart:0,samplesPerPixel:1,rngStates:cpuRng}),results=new Map();
    for(const mode of modes){
     const rng=initial.slice(),candidate=await runner.run({tile,sampleStart:0,samplesPerPixel:1,wave:true,rngStates:rng,guard:mode.guard,axisSubmission:mode.axisSubmission});
     assert.deepEqual(rng,initial);assert.equal(candidate.recovery.paths,4);assert(candidate.recovery.keys.every(k=>k.reasons.includes('unmodelledContinuation')),'Wave must explicitly reject bounded unit/anchor provenance');
     const rawAudit=compareResidentAudit(reference,candidate.audit,{maxMismatches:4096}),admitted=admitResidentWithRecovery(reference,candidate);assert(rawAudit.ok);assert(admitted.validation.passed);assert.equal(admitted.recovery.numericGeometry.count,0);
     for(let i=0;i<candidate.pixelIndices.length;i++)assert.equal(candidate.finalRngStates[i],cpuRng[candidate.pixelIndices[i]]);
     if(f.name==='two-mirrors-emission'&&b.seed===options.seed&&b.sampleStart===0&&mode.id==='anchor-staged'){report.decodedHistoryNegativeControls=checkDecodedHistoryMutations(b.reference,candidate);saveReport();}
     results.set(mode.id,candidate);
    }
    assertEquivalent(results.get('anchor-staged'),results.get('anchor-batched'),'unsupported wave domain');
    record.waveDomainRejection={paths:4,allModesReplayPaths:4,rawShadowAllPass:true,numericGeometryReplacements:0,anchorSubmissionIdentical:true};saveReport();
   }
  }}finally{runner.destroy();}
 }
 if(execute){
  const batches=report.cases.flatMap(c=>c.batches);report.rawShadowAllPass=true;report.anchorSubmissionIdentical=true;
  report.totals={paths:batches.reduce((n,b)=>n+b['unit-staged'].paths,0),waveDomainPaths:report.cases.reduce((n,c)=>n+(c.waveDomainRejection?.paths??0),0),cpuFullShadowConstructionMs:batches.reduce((n,b)=>n+b.cpuFullShadowConstructionMs,0),sharedSetupWallMs:report.cases.reduce((n,c)=>n+c.setupWallMs,0),byMode:Object.fromEntries(modes.map(m=>[m.id,{replayPaths:batches.reduce((n,b)=>n+b[m.id].replayPaths,0),retainedMirrorPaths:batches.reduce((n,b)=>n+b[m.id].retainedMirrorPaths,0),retainedMirrorEmissionPaths:batches.reduce((n,b)=>n+b[m.id].retainedMirrorEmissionPaths,0),retainedMirrorEscapePaths:batches.reduce((n,b)=>n+b[m.id].retainedMirrorEscapePaths,0),numericGeometryReplacements:0,runWallMs:batches.reduce((n,b)=>n+b[m.id].timings.gpuResidentTotalWallMs,0),cpuSelectiveReplayMs:batches.reduce((n,b)=>n+b[m.id].timings.cpuSelectiveReplayMs,0),statusReadbackWallMs:batches.reduce((n,b)=>n+b[m.id].timings.gpuStatusReadbackWallMs,0),replayReadbackWallMs:batches.reduce((n,b)=>n+b[m.id].timings.replayReadbackWallMs,0),finalReadbackDecodeMs:batches.reduce((n,b)=>n+b[m.id].timings.gpuFinalReadbackDecodeMs,0),rawShadowAuditMs:batches.reduce((n,b)=>n+b[m.id].rawShadowAuditMs,0),admissionAuditWallMs:batches.reduce((n,b)=>n+b[m.id].admissionAuditWallMs,0)}]))};
  report.exploratoryAnchorGains=report.cases.map(c=>({name:c.name,paths:c.batches.reduce((n,b)=>n+b['unit-staged'].paths,0),unitReplay:c.batches.reduce((n,b)=>n+b['unit-staged'].replayPaths,0),anchorReplay:c.batches.reduce((n,b)=>n+b['anchor-staged'].replayPaths,0),anchorRetainedMirrorPaths:c.batches.reduce((n,b)=>n+b['anchor-staged'].retainedMirrorPaths,0),anchorRetainedByMirrorCount:c.batches.reduce((out,b)=>{for(const [count,paths] of Object.entries(b['anchor-staged'].retainedByMirrorCount))out[count]=(out[count]??0)+paths;return out;},{})}));
  const goals=['two-mirrors-emission','four-scale-shifted-zxy','half-scale-three-mirrors-negative-permutation','eight-scale-two-mirrors'];
  report.multibounceGoals=goals.map(name=>{const g=report.exploratoryAnchorGains.find(c=>c.name===name);return g?{...g,executed:true,replayReduction:g.unitReplay-g.anchorReplay,met:g.anchorReplay<g.unitReplay&&g.anchorRetainedMirrorPaths>0}:{name,executed:false,met:false};});
  report.allMultibounceGoalsMet=report.multibounceGoals.every(g=>g.met);
  report.goalMisses=report.multibounceGoals.filter(g=>g.executed&&!g.met).map(g=>g.name);
  report.totalReplayReduction=report.totals.byMode['unit-staged'].replayPaths-report.totals.byMode['anchor-staged'].replayPaths;
  // Gains are an experimental outcome, not a correctness gate: keep every miss.
  if(report.fullFixtureSuite){assert.equal(report.totals.paths,4608,'Unchanged v19 paired path corpus');assert.equal(report.totals.waveDomainPaths,4,'Unchanged unsupported-wave corpus');}

 }
 report.allPass=true;report.status='passed';
}catch(error){report.allPass=false;report.status='failed';report.failure={name:error.name,message:error.message,stack:error.stack};throw error;}
finally{gpu=null;saveReport();}
console.log(JSON.stringify({allPass:report.allPass,actualGPUExecuted:report.actualGPUExecuted,fixtures:report.cases.length,totals:report.totals}));
