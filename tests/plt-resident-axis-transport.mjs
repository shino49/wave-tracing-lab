/** End-to-end pure-ray mirror transport, with an independent complete CPU shadow.
 * Compile: node tests/plt-resident-axis-transport.mjs
 * Execute: WEBGPU_EXECUTION_BACKEND=vulkan \
 * VK_ICD_FILENAMES=/usr/lib/chromium/vk_swiftshader_icd.json \
 * node tests/plt-resident-axis-transport.mjs --execute
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
const materials=[{type:'mirror',albedo:.5,emission:0},{type:'mirror',albedo:.75,emission:0},{type:'diffuse',albedo:0,emission:8,twoSidedEmission:true}];
function quad(axis,position,lo,hi,materialIndex){
 const a=(axis+1)%3,b=(axis+2)%3,point=(u,v)=>{const p=[0,0,0];p[axis]=position;p[a]=u;p[b]=v;return p;};
 return {mesh:{vertices:[point(lo,lo),point(hi,lo),point(hi,hi),point(lo,hi)],faces:[[0,1,2],[0,2,3]]},materialIndex};
}
function fixture(name,{mirrors=1,emitter=true,scale=1,shift=[0,0,0],perm=[0,1,2],camera,edit,expected,eligible=true,mustRetain=false,mustFallback=false,requireReduction=false}={}){
 const entries=mirrors===2?[quad(0,2,-8,8,0),quad(2,3,-8,8,1)]:[quad(2,2,-8,8,0)];
 if(emitter)entries.push(quad(2,-2,-16,16,2));
 let c=camera??(mirrors===2?{position:[0,0,0],target:[4,1,4],up:[0,1,0],fov:10}:{position:[0,0,0],target:[0,0,2],up:[0,1,0],fov:20});
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
];
const report={buildId:PLT_RESIDENT_BUILD_ID,executionRequested:execute,actualGPUExecuted:false,allPass:false,status:'preparing',backend,fullCPUShadowRequired:true,physicalGPUValidated:false,browserValidated:false,hardwarePerformanceClaim:false,cases:[],cpuFixtures:[]};
const selected=process.env.PLT_AXIS_CASE?fixtures.filter(f=>f.name===process.env.PLT_AXIS_CASE):fixtures;
assert(selected.length,'Unknown PLT_AXIS_CASE');
report.selectedFixtures=selected.map(f=>f.name);report.fullFixtureSuite=selected.length===fixtures.length;report.nodeVersion=process.version;report.vulkanICD=process.env.VK_ICD_FILENAMES??null;
report.testSHA256=crypto.createHash('sha256').update(fs.readFileSync(new URL(import.meta.url))).digest('hex');
const references=new Map();
for(const f of selected){
 const packed=packResidentScene(f.scene);assert.equal(packed.metadata.axisGuard.eligible,f.eligible,`${f.name}: source eligibility`);
 const rng=createPLTRandomStates(options),batches=[];let mirrorPaths=0,emissions=0,escapes=0;
 for(const [sampleStart,samplesPerPixel] of [[0,2],[2,1]]){
  const start=rng.slice(),began=performance.now();
  const reference=await buildResidentReferenceBatch(f.scene,f.camera,options,{tile:fullTile,sampleStart,samplesPerPixel,rngStates:rng});
  for(const path of reference.audit.paths){
   mirrorPaths+=path.scatters.some(s=>s.type==='mirror');emissions+=path.contributions.length;escapes+=path.termination.reason==='escaped';
   if(f.expected!==undefined){assert.equal(path.value,f.expected,`${f.name}: independently derived radiance`);assert.equal(path.contributions.length,f.expected>0?1:0,`${f.name}: contribution multiplicity`);}
   assert.equal(path.rng.count,2,`${f.name}: deterministic mirrors consume only two camera draws`);
  }
  batches.push({sampleStart,samplesPerPixel,start,end:rng.slice(),reference,cpuFullShadowMs:performance.now()-began});
 }
 if(f.mustRetain)assert.equal(mirrorPaths,192,`${f.name}: all test paths must actually reflect`);
 references.set(f.name,batches);
 report.cpuFixtures.push({name:f.name,paths:192,mirrorPaths,contributions:emissions,escapes,expectedRadiance:f.expected,eligibility:packed.metadata.axisGuard,sourceNormals:f.scene.triangles.map(t=>t.normal),scale:f.scale,translation:f.shift,permutation:f.perm});
}
if(cpuOnly){console.log(JSON.stringify({cpuFixtures:report.cpuFixtures,allPass:true}));process.exit(0);}
const sources=files.map(f=>fs.readFileSync(new URL('../dist/'+f,import.meta.url),'utf8'));
report.sourceSHA256=Object.fromEntries(files.map((f,i)=>[f,crypto.createHash('sha256').update(sources[i]).digest('hex')]));
report.hostSHA256=Object.fromEntries(['plt-resident-gpu.js','plt-resident-geometry.js','plt-resident-recovery.js','plt-resident-reference.js','plt-core.js'].map(f=>[f,crypto.createHash('sha256').update(fs.readFileSync(new URL('../dist/'+f,import.meta.url))).digest('hex')]));
const saveReport=()=>fs.writeFileSync(new URL(`../research/plt-resident-axis-${execute?'validation':'compile-validation'}.json`,import.meta.url),JSON.stringify(report,null,2)+'\n');
report.status='compiling';saveReport();
let gpu=create([`backend=${backend}`]);
try{
 for(const f of execute?selected:selected.slice(0,1)){
  const setup=performance.now(),runner=await createPLTResidentRunner(f.scene,f.camera,options,{gpu,shaderSource:sources.join('\n'),timeoutMs:120000,recoveryMode:'selective',continuationGuard:'axis-mirror-interval',onSetupProgress:step=>{if(execute)console.log(JSON.stringify({setup:f.name,...step}));}});
  const record={name:f.name,setupWallMs:performance.now()-setup,adapter:runner.diagnostic.adapter,eligibility:runner.packedScene.metadata.axisGuard,batches:[]};report.cases.push(record);
  try{if(execute){
   for(const b of references.get(f.name)){
    const pair={sampleStart:b.sampleStart,samplesPerPixel:b.samplesPerPixel,cpuFullShadowMs:b.cpuFullShadowMs};record.batches.push(pair);
    for(const guard of ['replay-unmodelled','axis-mirror-interval']){
     const original=b.start.slice(),candidate=await runner.run({tile:fullTile,sampleStart:b.sampleStart,samplesPerPixel:b.samplesPerPixel,wave:false,rngStates:original,guard});
     report.actualGPUExecuted=true;report.status='validating';
     assert.deepEqual(original,b.start,`${f.name}: transactional input RNG`);assert.equal(candidate.recovery.continuationGuard,guard,'Per-run guard must select the requested algorithm');
     const directAudit=compareResidentAudit(b.reference,candidate.audit,{maxMismatches:4096});
     pair[guard]={paths:candidate.audit.paths.length,replayPaths:candidate.recovery.paths,reasons:candidate.recovery.keys.reduce((a,k)=>{for(const reason of k.reasons)a[reason]=(a[reason]??0)+1;return a;},{}),triggerDepths:candidate.recovery.keys.reduce((a,k)=>{const depth=k.triggerDepth??'unknown';a[depth]=(a[depth]??0)+1;return a;},{}),intervalReasons:candidate.recovery.keys.reduce((a,k)=>{a[k.intervalReason??0]=(a[k.intervalReason??0]??0)+1;return a;},{}),rawAudit:directAudit};saveReport();
     assert(directAudit.ok,`${f.name}/${guard}: raw candidate requires shadow numerical replacement: ${JSON.stringify(directAudit.failures.slice(0,5))}`);
     const admitted=admitResidentWithRecovery(b.reference,candidate);assert(admitted.validation.passed);assert.equal(admitted.recovery.numericGeometry.count,0,'No audit-discovered numerical replacement is permitted');
     assert.strictEqual(admitted.image,admitted.audit.reference,'Only audited pixel data can be admitted');
     const replayKeys=new Set(candidate.recovery.keys.map(item=>residentPathKey(item.key))),cpuMap=new Map(b.reference.audit.paths.map(p=>[residentPathKey(p.key),p]));
     let retainedMirrorPaths=0,retainedMirrorEmissionPaths=0,retainedMirrorEscapePaths=0,retainedIntervalQueries=0,contributions=0;
     for(const path of candidate.audit.paths){
      const key=residentPathKey(path.key),cpu=cpuMap.get(key);assert(cpu,'No invented path');assert.deepEqual(path.rng,cpu.rng,`${f.name}: exact sequential RNG words/state`);
      assert.equal(path.contributions.length,cpu.contributions.length,'No dropped or duplicated contribution');contributions+=path.contributions.length;
      if(f.expected!==undefined)assert.equal(path.sampleValue??path.value,f.expected,`${f.name}: dyadic analytic radiance must be exact`);
      if(!replayKeys.has(key)&&cpu.scatters.some(s=>s.type==='mirror')){
       retainedMirrorPaths++;retainedMirrorEmissionPaths+=cpu.contributions.length>0;retainedMirrorEscapePaths+=cpu.termination.reason==='escaped';
       if(guard==='axis-mirror-interval'){assert(path.intervalGuard?.queries>=2,'Retained mirror must include interval-checked continued query');retainedIntervalQueries+=path.intervalGuard.queries;}
      }
     }
     for(let i=0;i<candidate.pixelIndices.length;i++)assert.equal(candidate.finalRngStates[i],b.end[candidate.pixelIndices[i]],'Canonical persistent RNG for later samples');
     assert.equal(candidate.audit.paths.length,fullTile.width*fullTile.height*b.samplesPerPixel);assert.equal(new Set(candidate.audit.paths.map(p=>residentPathKey(p.key))).size,candidate.audit.paths.length,'No duplicate paths');
     if(guard==='replay-unmodelled')assert.equal(retainedMirrorPaths,0,'Baseline must replay every continued mirror path');
     if(guard==='axis-mirror-interval'&&(!f.eligible||f.mustFallback))assert.equal(retainedMirrorPaths,0,'Unsupported source or camera domain must fall back');
     pair[guard]={paths:candidate.audit.paths.length,replayPaths:candidate.recovery.paths,retainedMirrorPaths,retainedMirrorEmissionPaths,retainedMirrorEscapePaths,retainedIntervalQueries,contributions,retentionTargetApplicable:f.mustRetain,retentionTargetMet:!f.mustRetain||retainedMirrorPaths>=candidate.audit.paths.length/2,reasons:candidate.recovery.keys.reduce((a,k)=>{for(const reason of k.reasons)a[reason]=(a[reason]??0)+1;return a;},{}),triggerDepths:candidate.recovery.keys.reduce((a,k)=>{const depth=k.triggerDepth??'unknown';a[depth]=(a[depth]??0)+1;return a;},{}),intervalReasons:candidate.recovery.keys.reduce((a,k)=>{a[k.intervalReason??0]=(a[k.intervalReason??0]??0)+1;return a;},{}),numericGeometryReplacements:admitted.recovery.numericGeometry.count,timings:admitted.timings,gate:admitted.validation,rawAudit:directAudit};
    }
    pair.replayReduction=pair['replay-unmodelled'].replayPaths-pair['axis-mirror-interval'].replayPaths;
    saveReport();console.log(JSON.stringify({name:f.name,sampleStart:b.sampleStart,baselineReplay:pair['replay-unmodelled'].replayPaths,intervalReplay:pair['axis-mirror-interval'].replayPaths,retainedMirrorPaths:pair['axis-mirror-interval'].retainedMirrorPaths,numericReplacements:pair['axis-mirror-interval'].numericGeometryReplacements}));
   }
   if(f.name==='one-mirror-emission'){
    const tile={x:3,y:3,width:2,height:2},rng=createPLTRandomStates(options),initial=rng.slice(),cpuRng=rng.slice();
    const reference=await buildResidentReferenceBatch(f.scene,f.camera,{...options,wave:true},{tile,sampleStart:0,samplesPerPixel:1,rngStates:cpuRng});
    const candidate=await runner.run({tile,sampleStart:0,samplesPerPixel:1,wave:true,rngStates:rng,guard:'axis-mirror-interval'});
    assert.deepEqual(rng,initial);assert.equal(candidate.recovery.paths,4,'Unsupported wave domain must replay every path');
    assert(candidate.recovery.keys.every(k=>k.reasons.includes('unmodelledContinuation')),'Wave fallback must be explicit');
    const rawAudit=compareResidentAudit(reference,candidate.audit,{maxMismatches:4096}),admitted=admitResidentWithRecovery(reference,candidate);
    assert(rawAudit.ok);assert(admitted.validation.passed);assert.equal(admitted.recovery.numericGeometry.count,0);
    for(let i=0;i<candidate.pixelIndices.length;i++)assert.equal(candidate.finalRngStates[i],cpuRng[candidate.pixelIndices[i]]);
    record.waveDomainRejection={paths:4,replayPaths:candidate.recovery.paths,numericGeometryReplacements:0,rawAudit,gate:admitted.validation};saveReport();
    console.log(JSON.stringify({name:f.name,waveDomainRejection:record.waveDomainRejection.paths,passed:true}));
   }
  }}finally{runner.destroy();}
 }
 if(execute){
  const batches=report.cases.flatMap(c=>c.batches);report.rawShadowAllPass=true;
  report.totals={paths:batches.reduce((n,b)=>n+b['axis-mirror-interval'].paths,0),baselineReplayPaths:batches.reduce((n,b)=>n+b['replay-unmodelled'].replayPaths,0),intervalReplayPaths:batches.reduce((n,b)=>n+b['axis-mirror-interval'].replayPaths,0),retainedMirrorPaths:batches.reduce((n,b)=>n+b['axis-mirror-interval'].retainedMirrorPaths,0),retainedMirrorEmissionPaths:batches.reduce((n,b)=>n+b['axis-mirror-interval'].retainedMirrorEmissionPaths,0),retainedMirrorEscapePaths:batches.reduce((n,b)=>n+b['axis-mirror-interval'].retainedMirrorEscapePaths,0),numericGeometryReplacements:batches.reduce((n,b)=>n+b['axis-mirror-interval'].numericGeometryReplacements,0),waveDomainPaths:report.cases.reduce((n,c)=>n+(c.waveDomainRejection?.paths??0),0)};
  report.retentionTargetFailures=report.cases.flatMap(c=>c.batches.filter(b=>!b['axis-mirror-interval'].retentionTargetMet).map(b=>({name:c.name,sampleStart:b.sampleStart,retained:b['axis-mirror-interval'].retainedMirrorPaths,paths:b['axis-mirror-interval'].paths,replayReduction:b.replayReduction,reasons:b['axis-mirror-interval'].reasons})));
  report.exploratoryRetentionGoalsMet=report.retentionTargetFailures.length===0;
  report.requiredReplayReductionFailures=report.cases.filter(c=>selected.find(f=>f.name===c.name).requireReduction).flatMap(c=>c.batches.filter(b=>b.replayReduction<=0||b['axis-mirror-interval'].retainedMirrorPaths===0).map(b=>({name:c.name,sampleStart:b.sampleStart,replayReduction:b.replayReduction,retainedMirrorPaths:b['axis-mirror-interval'].retainedMirrorPaths})));
  report.requiredReplayReductionPass=report.requiredReplayReductionFailures.length===0;
  assert(report.requiredReplayReductionPass,`Demonstrated narrow mirror class lost actual replay reduction: ${JSON.stringify(report.requiredReplayReductionFailures)}`);
 }
 report.allPass=true;report.status='passed';
}catch(error){report.allPass=false;report.status='failed';report.failure={name:error.name,message:error.message,stack:error.stack};throw error;}
finally{gpu=null;saveReport();}
console.log(JSON.stringify({allPass:report.allPass,actualGPUExecuted:report.actualGPUExecuted,fixtures:report.cases.length,paths:report.cases.reduce((n,c)=>n+c.batches.reduce((m,b)=>m+b['axis-mirror-interval'].paths,0),0),retainedMirrorPaths:report.cases.reduce((n,c)=>n+c.batches.reduce((m,b)=>m+b['axis-mirror-interval'].retainedMirrorPaths,0),0)}));
