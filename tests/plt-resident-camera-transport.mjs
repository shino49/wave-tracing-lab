/** v22 joint-camera opt-in transport, with immutable v20/v21 checkpoints.
 * CPU: node tests/plt-resident-camera-transport.mjs --cpu-only
 * Native: WEBGPU_EXECUTION_BACKEND=vulkan VK_ICD_FILENAMES=/usr/lib/chromium/vk_swiftshader_icd.json node tests/plt-resident-camera-transport.mjs --execute --reuse-native-device
 * Every native path requires an independent complete CPU shadow. Software only.
 */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {pathToFileURL} from 'node:url';
import {create,globals} from 'webgpu';
import {cameraFrame,createPLTRandomStates,randomGenerator} from '../dist/plt-core.js';
import {packResidentScene} from '../dist/plt-resident-geometry.js';
import {createPLTResidentRunner,PLT_RESIDENT_BUILD_ID} from '../dist/plt-resident-gpu.js';
import {buildResidentReferenceBatch,compareResidentAudit,residentPathKey} from '../dist/plt-resident-reference.js';
import {admitResidentWithRecovery} from '../dist/plt-resident-recovery.js';
import {fixtures,options,fullTile,hash,readBaseline,buildCorpus,reuseNativeDevice} from './plt-resident-tight-transport.mjs';
export {fixtures,options,fullTile,hash,reuseNativeDevice};
Object.assign(globalThis,globals);

export const modes=[
 {id:'anchor-round',guard:'axis-mirror-anchor',axisSubmission:'round'},
 {id:'camera-round',guard:'axis-mirror-camera',axisSubmission:'round'},
];
export const shaderFiles=['utd-functions.wgsl','plt-resident-geometry.wgsl','plt-resident-interval.wgsl','plt-resident-axis-guard.wgsl','plt-resident-rng.wgsl','plt-resident-fsd.wgsl','plt-resident-walker.wgsl'];
export const pinnedV21={commit:'916bf9952bcd9f6da8203de3cede5675d347b18c',manifest:'plt-resident-tight-validation-manifest.json',manifestSHA256:'66d28e3c1b503090e05e273c4336db4be9201ada74106c163fbde0056ce31367',transport:'plt-resident-tight-validation.json',transportSHA256:'5accec3c0619d476fbc0c93e6d84a9adbd3be5b4a305b3ccfa64436b543eef6f',timing:'plt-resident-tight-submit-timing-validation.json',timingSHA256:'c04e1940014c2abcb2545ad7db28b5a3503e052f90f996f3c801c759124ef86c',paths:4608,anchorReplay:1596,anchorRetainedMirror:2913,tightReplay:1591,tightRetainedMirror:2918};
export function readBaselines(){
 const v20=readBaseline(),manifestBytes=fs.readFileSync(new URL('../research/'+pinnedV21.manifest,import.meta.url));
 assert.equal(hash(manifestBytes),pinnedV21.manifestSHA256,'v21 validation manifest is immutable');
 const manifest=JSON.parse(manifestBytes);
 for(const [file,sha256] of Object.entries(manifest.files))assert.equal(hash(fs.readFileSync(new URL('../'+file,import.meta.url))),sha256,`Immutable v21 artifact changed: ${file}`);
 const bytes=fs.readFileSync(new URL('../research/'+pinnedV21.transport,import.meta.url));assert.equal(hash(bytes),pinnedV21.transportSHA256);
 const v21=JSON.parse(bytes);assert(v21.allPass&&v21.actualGPUExecuted&&v21.fullFixtureSuite);assert.equal(v21.totals.paths,4608);
 for(const [mode,replays,retained] of [['anchor-round',1596,2913],['tight-round',1591,2918]]){assert.equal(v21.totals.byMode[mode].replayPaths,replays);assert.equal(v21.totals.byMode[mode].retainedMirrorPaths,retained);}
 return {v20,v21};
}
export function sourceProvenance(){
 const sources=shaderFiles.map(file=>fs.readFileSync(new URL('../dist/'+file,import.meta.url),'utf8'));
 const hostFiles=['plt-resident-gpu.js','plt-resident-geometry.js','plt-resident-recovery.js','plt-resident-reference.js','plt-core.js',...fs.readdirSync(new URL('../dist/',import.meta.url)).filter(f=>/^plt-resident-camera.*\.js$/.test(f))];
 return {sources,sourceSHA256:Object.fromEntries(shaderFiles.map((f,i)=>[f,hash(sources[i])])),hostSHA256:Object.fromEntries(hostFiles.map(f=>[f,hash(fs.readFileSync(new URL('../dist/'+f,import.meta.url)))]))};
}
const histogram=(items,value)=>items.reduce((out,item)=>{const key=value(item);out[key]=(out[key]??0)+1;return out;},{});
const camera={position:[-.125,.0625,0],target:[.375,.25,2],up:[.375,1,.25],fov:20};
// The high word rounds to 1.0f before rr_float's [0,1) clamp; the CPU still
// consumes its exact word / 2^32 value. Both camera draws exercise this edge.
const extremePrefixes=[14150146,52078625,2477551629,2515480108];
function cameraFixture(name,width,height,{fov=20,tile,jitterExtremes=false,expected=4,cameraDomain=true}={}){
 const original=fixtures.find(f=>f.name==='one-mirror-emission'),scene=structuredClone(original.scene),c={...structuredClone(camera),fov};scene.camera=c;
 const frame=cameraFrame(c),components=Object.values(frame).flat();assert(components.every(v=>v!==0),'Adversarial camera must have no axis-aligned frame components');assert.equal(new Set(components).size,9,'All nine camera frame components must be distinct');
 return {...original,name,scene,camera:c,expected,expectedValues:expected===undefined?[0,4,8]:[expected],mustRetain:expected===4,requireReduction:false,width,height,tile:tile??{x:0,y:0,width,height},jitterExtremes,cameraDomain,frame};
}
export const adversarialFixtures=[
 cameraFixture('distinct-frame-square',8,8),
 cameraFixture('distinct-frame-wide-17x9',17,9),
 cameraFixture('distinct-frame-tall-9x17',9,17),
 cameraFixture('minimum-fov-wide-31x8',31,8,{fov:5}),
 cameraFixture('maximum-fov-tall-8x31',8,31,{fov:120}),
 cameraFixture('extreme-jitter-13x11',13,11,{jitterExtremes:true}),
 cameraFixture('outside-camera-wide-aspect',41,8,{fov:5,tile:{x:0,y:0,width:32,height:8},cameraDomain:false}),
 cameraFixture('outside-camera-tall-aspect',8,41,{fov:5,tile:{x:0,y:0,width:8,height:32},cameraDomain:false}),
];
export async function buildAdversarialCorpus(selected=adversarialFixtures){
 const references=new Map(),cpuFixtures=[];
 for(const f of selected){
  const o={...options,width:f.width,height:f.height},rng=createPLTRandomStates(o);if(f.jitterExtremes)for(let i=0;i<rng.length;i++)rng[i]=extremePrefixes[i%extremePrefixes.length];
  const initial=rng.slice(),draws=extremePrefixes.map(seed=>{const r=randomGenerator(seed),x=r(),y=r();return {state:seed,x,y,xWord:x*2**32,yWord:y*2**32,xGPU:Math.min(Math.fround(x),1-2**-24),yGPU:Math.min(Math.fround(y),1-2**-24)};});
  if(f.jitterExtremes)for(const axis of ['x','y']){assert(Math.min(...draws.map(d=>d[axis]))<2**-25);assert(Math.max(...draws.map(d=>d[axis]))>1-2**-25);assert(draws.some(d=>Math.fround(d[axis])===1&&d[axis+'GPU']<1),'Exercise f32 camera jitter upper clamp');assert(draws.some(d=>d[axis+'Word']===62));assert(draws.some(d=>d[axis+'Word']===4294967275));}
  const batches=[],powers=[];
  for(const [sampleStart,samplesPerPixel] of [[0,2],[2,1]]){
   const start=rng.slice(),began=performance.now(),reference=await buildResidentReferenceBatch(f.scene,f.camera,o,{tile:f.tile,sampleStart,samplesPerPixel,rngStates:rng});
   for(const p of reference.audit.paths){assert(f.expectedValues.includes(p.value),`${f.name}: analytic one-mirror power ${p.value}`);assert.equal(p.rng.count,2);assert(p.scatters.every(s=>s.type==='mirror'));assert(p.scatters.length<=1);if(f.mustRetain)assert.equal(p.scatters.length,1);powers.push(p.value);}
   batches.push({seed:o.seed,options:o,tile:f.tile,sampleStart,samplesPerPixel,start,end:rng.slice(),reference,cpuFullShadowConstructionMs:performance.now()-began});
  }
  references.set(f.name,batches);cpuFixtures.push({name:f.name,width:f.width,height:f.height,aspect:f.width/f.height,tile:f.tile,paths:powers.length,frame:f.frame,nonzeroDistinctFrameComponents:true,halfFov:Math.tan((f.camera.fov*Math.PI/180)/2),expectedCameraDomain:f.cameraDomain,analyticPowerHistogram:histogram(powers,x=>x),initialRngSHA256:hash(Buffer.from(initial.buffer)),jitterExtremes:f.jitterExtremes?draws:null,sourceEligibility:packResidentScene(f.scene).metadata.axisGuard});
 }
 return {references,cpuFixtures};
}
const checkpointFields=['paths','replayPaths','retainedMirrorPaths','retainedMirrorEmissionPaths','retainedMirrorEscapePaths','retainedByMirrorCount','retainedIntervalQueries','contributions','reasons','triggerDepths','intervalReasons','rounds','dispatches'];
export function assertLegacyCheckpoint(baselines,f,b,result){
 for(const [version,mode] of [['v20','anchor-batched'],['v21','anchor-round']]){
  const expected=baselines[version].cases.find(c=>c.name===f.name)?.batches.find(h=>h.seed===b.seed&&h.sampleStart===b.sampleStart&&h.samplesPerPixel===b.samplesPerPixel)?.[mode];assert(expected,`${version} exact fixture/batch exists`);
  for(const field of checkpointFields)assert.deepEqual(result[field],expected[field],`${version} ${f.name}/${b.seed}/${b.sampleStart}: ${field}`);
 }
 return {v20AndV21CheckpointIdentical:true};
}
export function assertEquivalent(a,b,label){
 for(const field of ['audit','image','pixelIndices','finalRngStates'])assert.deepEqual(b[field],a[field],`${label}: ${field}`);
 assert.deepEqual(b.recovery.keys,a.recovery.keys,`${label}: complete replay ownership`);assert.equal(b.recovery.paths,a.recovery.paths);assert.equal(b.rounds,a.rounds);assert.equal(b.dispatches,a.dispatches);
}
export function validateCandidate(f,b,candidate,mode,{pixelCount=(b.tile??fullTile).width*(b.tile??fullTile).height}={}){
 assert.equal(candidate.candidate,true);assert.equal(candidate.admitted,false);assert.strictEqual(candidate.image,candidate.audit.reference);assert.equal(candidate.audit.complete,true);assert.equal(candidate.audit.flags,0);
 assert.equal(candidate.recovery.continuationGuard,mode.guard);assert.equal(candidate.recovery.axisSubmission,mode.axisSubmission);assert.equal(candidate.recovery.completeCPUPathReplay,true);
 const auditStart=performance.now(),rawAudit=compareResidentAudit(b.reference,candidate.audit,{maxMismatches:4096}),rawShadowAuditMs=performance.now()-auditStart;
 assert(rawAudit.ok,`${f.name}/${mode.id}/${b.seed}/${b.sampleStart}: independent raw shadow ${JSON.stringify(rawAudit.failures.slice(0,8))}`);
 assert.deepEqual(rawAudit.tolerance,{absolute:2e-5,relative:.003,geometryAbsolute:2e-5,geometryRelative:3e-6},'No relaxed power/geometry tolerances');
 const admissionStart=performance.now(),admitted=admitResidentWithRecovery(b.reference,candidate),admissionAuditWallMs=performance.now()-admissionStart;
 assert(admitted.validation.passed);assert.equal(admitted.recovery.numericGeometry.count,0,'No audit-discovered numerical replacement');assert.strictEqual(admitted.image,admitted.audit.reference);
 const replayKeys=new Set(candidate.recovery.keys.map(k=>residentPathKey(k.key))),cpuMap=new Map(b.reference.audit.paths.map(p=>[residentPathKey(p.key),p]));
 assert.equal(replayKeys.size,candidate.recovery.paths,'One replay certificate per recovered path');assert.equal(candidate.recovery.keys.length,candidate.recovery.paths);
 for(const recovery of candidate.recovery.keys){assert(cpuMap.has(residentPathKey(recovery.key)),'No foreign/stale replay owner');assert(recovery.reasons.length>0);}
 let retainedPaths=0,retainedMirrorPaths=0,retainedMirrorEmissionPaths=0,retainedMirrorEscapePaths=0,retainedIntervalQueries=0,contributions=0;const retainedByMirrorCount={};
 for(const path of candidate.audit.paths){
  const key=residentPathKey(path.key),cpu=cpuMap.get(key);assert(cpu,'No invented path');assert.deepEqual(path.rng,cpu.rng,'Exact camera draws, sequential RNG words and final state');assert.equal(path.contributions.length,cpu.contributions.length,'No dropped/duplicated contribution');contributions+=path.contributions.length;
  const value=path.sampleValue??path.value;if(f.expected!==undefined)assert.equal(value,f.expected,'Exact analytic path power');if(f.expectedValues)assert(f.expectedValues.includes(value),'Analytic one-mirror power set');
  if(!replayKeys.has(key)){
   retainedPaths++;assert.equal(path.intervalGuard?.unitProvenance,true);assert.equal(path.intervalGuard?.anchorProvenance,true);assert.equal(path.intervalGuard?.cameraProvenance??false,mode.guard==='axis-mirror-camera','Private camera provenance matches guard');
   const mirrors=cpu.scatters.filter(s=>s.type==='mirror').length;if(mirrors){retainedMirrorPaths++;retainedMirrorEmissionPaths+=cpu.contributions.length>0;retainedMirrorEscapePaths+=cpu.termination.reason==='escaped';retainedByMirrorCount[mirrors]=(retainedByMirrorCount[mirrors]??0)+1;assert(path.intervalGuard.queries>=mirrors+1);retainedIntervalQueries+=path.intervalGuard.queries;}
  }
 }
 for(let i=0;i<candidate.pixelIndices.length;i++)assert.equal(candidate.finalRngStates[i],b.end[candidate.pixelIndices[i]],'Canonical persistent RNG prefix');
 assert.equal(candidate.pixelIndices.length,pixelCount);assert.equal(candidate.image.length,pixelCount);assert.deepEqual(candidate.pixelIndices,b.reference.audit.pixelIndices,'Complete requested pixel ownership');
 if(f.expected!==undefined)for(const value of candidate.image)assert.equal(value,f.expected,'Complete analytic image');
 assert.equal(candidate.audit.paths.length,pixelCount*b.samplesPerPixel);assert.equal(new Set(candidate.audit.paths.map(p=>residentPathKey(p.key))).size,candidate.audit.paths.length);
 if(!f.eligible||f.mustFallback)assert.equal(retainedMirrorPaths,0,'Unsupported geometry/camera source must replay');
 if(f.cameraDomain===false&&mode.guard==='axis-mirror-camera')assert.equal(candidate.recovery.paths,candidate.audit.paths.length,'Out-of-domain camera must fail closed for all paths');
 return {paths:candidate.audit.paths.length,replayPaths:candidate.recovery.paths,reasons:histogram(candidate.recovery.keys.flatMap(k=>k.reasons),x=>x),triggerDepths:histogram(candidate.recovery.keys,k=>k.triggerDepth??'unknown'),intervalReasons:histogram(candidate.recovery.keys,k=>k.intervalReason??0),rawAudit,rawShadowAuditMs,retainedPaths,retainedMirrorPaths,retainedMirrorEmissionPaths,retainedMirrorEscapePaths,retainedByMirrorCount,retainedIntervalQueries,contributions,numericGeometryReplacements:0,rounds:candidate.rounds,dispatches:candidate.dispatches,timings:admitted.timings,admissionAuditWallMs,gate:admitted.validation,auditSHA256:hash(JSON.stringify(candidate.audit)),replayOwnershipSHA256:hash(JSON.stringify(candidate.recovery.keys)),replayOwnershipUnique:true};
}
function decodedHistoryNegativeControls(reference,candidate){
 const replay=new Set(candidate.recovery.keys.map(k=>residentPathKey(k.key))),index=candidate.audit.paths.findIndex(p=>!replay.has(residentPathKey(p.key))&&p.scatters.length>=2);
 if(index<0)return {exercised:false,reason:'No retained two-mirror path'};
 const mutations=[['alter-reflected-origin',c=>{c.audit.paths[index].vertices[1].origin[0]+=.125;}],['reverse-mirror-direction',c=>{c.audit.paths[index].scatters[0].direction=c.audit.paths[index].scatters[0].direction.map(v=>-v);}],['drop-second-mirror',c=>{c.audit.paths[index].scatters.splice(1,1);}],['stale-sample-owner',c=>{c.audit.paths[index].key.sample++;}]];
 return {exercised:true,scope:'Synthetic decoded full-history controls at mandatory shadow/admission boundary, not private GPU proof validation',checks:mutations.map(([name,mutate])=>{const corrupt=structuredClone(candidate);mutate(corrupt);const raw=compareResidentAudit(reference,corrupt.audit,{maxMismatches:4096});assert.equal(raw.ok,false);assert.throws(()=>admitResidentWithRecovery(reference,corrupt));return {name,rawShadowRejected:true,admissionRejected:true,mismatches:raw.mismatchCount};})};
}
export async function main(){
 const execute=process.argv.includes('--execute'),cpuOnly=process.argv.includes('--cpu-only'),reuseDevice=process.argv.includes('--reuse-native-device');assert(!(execute&&cpuOnly));
 const backend=execute?process.env.WEBGPU_EXECUTION_BACKEND:'null';if(execute)assert(['vulkan','metal','d3d11','d3d12','opengl','opengles'].includes(backend),'Native execution requires explicit backend');
 const selectedNames=process.env.PLT_CAMERA_CASES?.split(',')??(process.env.PLT_CAMERA_CASE?[process.env.PLT_CAMERA_CASE]:null),all=[...fixtures,...adversarialFixtures],selected=selectedNames?all.filter(f=>selectedNames.includes(f.name)):all;assert(selected.length&&(!selectedNames||selected.length===new Set(selectedNames).size),'Unknown PLT_CAMERA_CASE(S)');
 const legacySelected=selected.filter(f=>fixtures.includes(f)),adversarialSelected=selected.filter(f=>adversarialFixtures.includes(f)),baselines=readBaselines(),legacy=await buildCorpus(legacySelected),adversarial=await buildAdversarialCorpus(adversarialSelected),references=new Map([...legacy.references,...adversarial.references]);
 const provenance=cpuOnly?null:sourceProvenance(),report={buildId:PLT_RESIDENT_BUILD_ID,executionRequested:execute,actualGPUExecuted:false,allPass:false,status:'preparing',backend,nodeVersion:process.version,vulkanICD:process.env.VK_ICD_FILENAMES??null,fullCPUShadowRequired:true,physicalGPUValidated:false,browserValidated:false,hardwarePerformanceClaim:false,modes,pinnedV21,sourceSHA256:provenance?.sourceSHA256??null,hostSHA256:provenance?.hostSHA256??null,testSHA256:hash(fs.readFileSync(new URL(import.meta.url))),legacyCPUFixtures:legacy.cpuFixtures,adversarialCPUFixtures:adversarial.cpuFixtures,selectedFixtures:selected.map(f=>f.name),fullLegacySuite:legacySelected.length===fixtures.length,fullAdversarialSuite:adversarialSelected.length===adversarialFixtures.length,fullFixtureSuite:selected.length===all.length,cases:[],limitations:['Mandatory complete independent CPU shadow with unchanged raw geometry and power tolerance; zero numerical replacements required.','Both round modes share a camera-compiled runner per fixture. Historical anchor compilation cost is not measured.','Six in-domain and two explicitly out-of-domain camera fixtures are additional evidence; they do not enlarge the declared production domain.','Native software execution is not browser/physical-GPU validation or a hardware performance claim.']};
 const label=selectedNames?selectedNames.join('--')+'-':'',save=()=>fs.writeFileSync(new URL(`../research/plt-resident-camera-${label}${cpuOnly?'cpu-validation':execute?'validation':'compile-validation'}.json`,import.meta.url),JSON.stringify(report,null,2)+'\n');
 if(cpuOnly){report.allPass=true;report.status='cpu-fixtures-passed';report.totals={legacyPaths:legacy.cpuFixtures.reduce((n,f)=>n+f.paths,0),adversarialPaths:adversarial.cpuFixtures.reduce((n,f)=>n+f.paths,0),fixtures:selected.length};if(report.fullLegacySuite)assert.equal(report.totals.legacyPaths,4608);save();console.log(JSON.stringify({allPass:true,cpuOnly:true,totals:report.totals}));return;}
 let gpu=create([`backend=${backend}`]);const cache=reuseDevice?reuseNativeDevice(gpu):null;if(cache)gpu=cache.gpu;report.nativeDeviceReuse=cache?.stats??{enabled:false};report.status='compiling';save();
 try{
  for(const f of execute?selected:selected.slice(0,reuseDevice?2:1)){
   const batches=references.get(f.name),o=batches[0].options,setup=performance.now(),runner=await createPLTResidentRunner(f.scene,f.camera,o,{gpu,shaderSource:provenance.sources.join('\n'),timeoutMs:120000,recoveryMode:'selective',continuationGuard:'axis-mirror-camera',onSetupProgress:step=>console.log(JSON.stringify({setup:f.name,...step}))});
   const record={name:f.name,legacy:fixtures.includes(f),expectedCameraDomain:f.cameraDomain??null,setupWallMs:performance.now()-setup,adapter:runner.diagnostic.adapter,eligibility:runner.packedScene.metadata.axisGuard,batches:[]};report.cases.push(record);
   try{if(execute){
    for(const b of batches){
     const comparison={seed:b.seed,sampleStart:b.sampleStart,samplesPerPixel:b.samplesPerPixel,cpuFullShadowConstructionMs:b.cpuFullShadowConstructionMs};record.batches.push(comparison);
     for(const mode of modes){const input=b.start.slice(),candidate=await runner.run({tile:b.tile??fullTile,sampleStart:b.sampleStart,samplesPerPixel:b.samplesPerPixel,wave:false,rngStates:input,guard:mode.guard,axisSubmission:mode.axisSubmission});report.actualGPUExecuted=true;report.status='validating';assert.deepEqual(input,b.start,'Transactional input RNG must be immutable');comparison[mode.id]=validateCandidate(f,b,candidate,mode);if(record.legacy&&mode.id==='anchor-round')Object.assign(comparison,assertLegacyCheckpoint(baselines,f,b,comparison[mode.id]));if(f.name==='two-mirrors-emission'&&b.seed===options.seed&&b.sampleStart===0&&mode.id==='camera-round')report.decodedHistoryNegativeControls=decodedHistoryNegativeControls(b.reference,candidate);save();}
     comparison.cameraReplayReduction=comparison['anchor-round'].replayPaths-comparison['camera-round'].replayPaths;save();console.log(JSON.stringify({name:f.name,seed:b.seed,sampleStart:b.sampleStart,anchorReplay:comparison['anchor-round'].replayPaths,cameraReplay:comparison['camera-round'].replayPaths,cameraRetainedMirrorPaths:comparison['camera-round'].retainedMirrorPaths,numericReplacements:0}));
    }
    if(f.name==='one-mirror-emission'){
     const tile={x:3,y:3,width:2,height:2},initial=createPLTRandomStates(options),cpuRng=initial.slice(),reference=await buildResidentReferenceBatch(f.scene,f.camera,{...options,wave:true},{tile,sampleStart:0,samplesPerPixel:1,rngStates:cpuRng}),results=[];
     for(const mode of modes){const rng=initial.slice(),candidate=await runner.run({tile,sampleStart:0,samplesPerPixel:1,wave:true,rngStates:rng,guard:mode.guard,axisSubmission:mode.axisSubmission});assert.deepEqual(rng,initial);assert.equal(candidate.recovery.paths,4);assert(candidate.recovery.keys.every(k=>k.reasons.includes('unmodelledContinuation')));const raw=compareResidentAudit(reference,candidate.audit,{maxMismatches:4096}),admitted=admitResidentWithRecovery(reference,candidate);assert(raw.ok&&admitted.validation.passed);assert.equal(admitted.recovery.numericGeometry.count,0);for(let i=0;i<candidate.pixelIndices.length;i++)assert.equal(candidate.finalRngStates[i],cpuRng[candidate.pixelIndices[i]]);results.push({mode:mode.id,replayPaths:4,rawShadowPassed:true,numericGeometryReplacements:0,replayOwnershipSHA256:hash(JSON.stringify(candidate.recovery.keys))});}
     record.waveDomainRejection={paths:4,results};save();
    }
   }}finally{runner.destroy();}
  }
  if(execute){
   const summarize=cases=>{const batches=cases.flatMap(c=>c.batches),sum=(id,field)=>batches.reduce((n,b)=>n+b[id][field],0);return {paths:sum('anchor-round','paths'),byMode:Object.fromEntries(modes.map(({id})=>[id,{...Object.fromEntries(['replayPaths','retainedPaths','retainedMirrorPaths','retainedMirrorEmissionPaths','retainedMirrorEscapePaths','numericGeometryReplacements','rawShadowAuditMs','admissionAuditWallMs'].map(field=>[field,sum(id,field)])),...Object.fromEntries(['gpuResidentTotalWallMs','cpuSelectiveReplayMs','gpuStatusReadbackWallMs','replayReadbackWallMs','gpuFinalReadbackDecodeMs','gpuRoundSubmitAndWaitMs'].map(field=>[field,batches.reduce((n,b)=>n+(b[id].timings[field]??0),0)]))}])),cpuFullShadowConstructionMs:batches.reduce((n,b)=>n+b.cpuFullShadowConstructionMs,0),sharedSetupWallMs:cases.reduce((n,c)=>n+c.setupWallMs,0)};};
   report.totals={legacy:summarize(report.cases.filter(c=>c.legacy)),adversarial:summarize(report.cases.filter(c=>!c.legacy)),waveDomainPaths:report.cases.reduce((n,c)=>n+(c.waveDomainRejection?.paths??0),0)};
   report.fixtureRetention=report.cases.map(c=>{const count=(id,field)=>c.batches.reduce((n,b)=>n+b[id][field],0);return {name:c.name,legacy:c.legacy,paths:count('anchor-round','paths'),anchorReplay:count('anchor-round','replayPaths'),cameraReplay:count('camera-round','replayPaths'),replayReduction:count('anchor-round','replayPaths')-count('camera-round','replayPaths'),anchorRetainedMirrorPaths:count('anchor-round','retainedMirrorPaths'),cameraRetainedMirrorPaths:count('camera-round','retainedMirrorPaths')};});
   report.retentionRegressions=report.fixtureRetention.filter(f=>f.replayReduction<0).map(f=>f.name);report.rawShadowAllPass=true;report.anchorBaselineCountsPreserved=true;
   report.cameraDomainCoverage=report.cases.filter(c=>!c.legacy).map(c=>{const retention=report.fixtureRetention.find(f=>f.name===c.name);return {name:c.name,expectedCameraDomain:c.expectedCameraDomain,paths:retention.paths,cameraReplay:retention.cameraReplay,cameraRetainedMirrorPaths:retention.cameraRetainedMirrorPaths,retainedCameraProofExercised:retention.cameraRetainedMirrorPaths>0,outsideDomainAllReplayed:c.expectedCameraDomain===false?retention.cameraReplay===retention.paths:null};});
   report.allSelectedInDomainCamerasExercised=report.cameraDomainCoverage.filter(c=>c.expectedCameraDomain).every(c=>c.retainedCameraProofExercised);
   assert(report.allSelectedInDomainCamerasExercised,'An in-domain adversarial camera retained no proven GPU paths; general camera coverage has not been exercised');
   report.multibounceGoals=['two-mirrors-emission','four-scale-shifted-zxy','half-scale-three-mirrors-negative-permutation','eight-scale-two-mirrors'].map(name=>{const f=report.fixtureRetention.find(f=>f.name===name);return f?{...f,executed:true,met:f.replayReduction>0&&f.cameraRetainedMirrorPaths>0}:{name,executed:false,met:false};});report.allMultibounceGoalsMet=report.multibounceGoals.every(g=>g.met);
   if(report.fullLegacySuite){assert.equal(report.totals.legacy.paths,4608);assert.equal(report.totals.waveDomainPaths,4);assert.equal(report.totals.legacy.byMode['anchor-round'].replayPaths,1596);assert.equal(report.totals.legacy.byMode['anchor-round'].retainedMirrorPaths,2913);}
  }
  readBaselines();const finishedProvenance=sourceProvenance();assert.deepEqual(finishedProvenance.sourceSHA256,report.sourceSHA256,'Shader files changed during transport validation');assert.deepEqual(finishedProvenance.hostSHA256,report.hostSHA256,'Host files changed during transport validation');assert.equal(hash(fs.readFileSync(new URL(import.meta.url))),report.testSHA256,'Transport harness changed during validation');report.allPass=true;report.status='passed';
 }catch(error){report.allPass=false;report.status='failed';report.failure={name:error.name,message:error.message,stack:error.stack};throw error;}
 finally{try{cache?.destroy();}catch(error){report.allPass=false;report.status='failed';report.failure??={name:error.name,message:error.message,stack:error.stack};throw error;}finally{gpu=null;save();}}
 console.log(JSON.stringify({allPass:report.allPass,actualGPUExecuted:report.actualGPUExecuted,totals:report.totals,retentionRegressions:report.retentionRegressions}));
}
if(process.argv[1]&&pathToFileURL(process.argv[1]).href===import.meta.url)await main();
