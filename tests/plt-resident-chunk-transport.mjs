/** Bounded lane chunks: frozen v22 mathematical/fixture contract, new transport.
 * CPU: node tests/plt-resident-chunk-transport.mjs --cpu-only
 * Native: WEBGPU_EXECUTION_BACKEND=vulkan VK_ICD_FILENAMES=/usr/lib/chromium/vk_swiftshader_icd.json node tests/plt-resident-chunk-transport.mjs --execute --reuse-native-device
 * The reviewed cache is transport-only. Timing uses an ordinary native runner.
 */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {pathToFileURL} from 'node:url';
import {create,globals} from 'webgpu';
import {createPLTRandomStates} from '../dist/plt-core.js';
import {createPLTResidentRunner,PLT_RESIDENT_BUILD_ID,PLT_RESIDENT_LIMITS} from '../dist/plt-resident-gpu.js';
import {buildResidentReferenceBatch,compareResidentAudit,residentPathKey} from '../dist/plt-resident-reference.js';
import {admitResidentWithRecovery} from '../dist/plt-resident-recovery.js';
import {buildCorpus} from './plt-resident-tight-transport.mjs';
import {fixtures,adversarialFixtures,buildAdversarialCorpus,options,fullTile,modes,hash,sourceProvenance,validateCandidate,readBaselines,reuseNativeDevice} from './plt-resident-camera-transport.mjs';
export {fixtures,options,modes,hash,sourceProvenance,validateCandidate};
Object.assign(globalThis,globals);

export const pinnedV22={commit:'0b6afb30408dd757330eaa86cc340e84345aa203',manifest:'plt-resident-camera-validation-manifest.json',manifestSHA256:'ad3f7452592fc9d6619312813dc9ca1585af949cd331f15b9d3401e4bedc5b0e',transport:'plt-resident-camera-validation.json',transportSHA256:'32f0feae30e6a3ab6c0e7ebaecb843133cdbafbe7e4427b7519b8615e6e26979',legacyPaths:4608,adversarialPaths:4563,anchorLegacyReplay:1596,cameraLegacyReplay:1349,anchorLegacyRetainedMirror:2913,cameraLegacyRetainedMirror:3160};
export function readV22(){
 readBaselines();
 const bytes=fs.readFileSync(new URL('../research/'+pinnedV22.manifest,import.meta.url));assert.equal(hash(bytes),pinnedV22.manifestSHA256,'Frozen v22 manifest');
 const manifest=JSON.parse(bytes);for(const [file,sha256] of Object.entries(manifest.files))assert.equal(hash(fs.readFileSync(new URL('../'+file,import.meta.url))),sha256,`Frozen v22 artifact: ${file}`);
 const transportBytes=fs.readFileSync(new URL('../research/'+pinnedV22.transport,import.meta.url));assert.equal(hash(transportBytes),pinnedV22.transportSHA256);const transport=JSON.parse(transportBytes);
 assert(transport.allPass&&transport.actualGPUExecuted&&transport.fullFixtureSuite);assert.equal(transport.totals.legacy.paths,pinnedV22.legacyPaths);assert.equal(transport.totals.adversarial.paths,pinnedV22.adversarialPaths);
 const current=sourceProvenance();assert.deepEqual(current.sourceSHA256,manifest.sourceSHA256,'No mathematical shader changes in this performance checkpoint');
 assert.deepEqual(Object.keys(current.hostSHA256).sort(),Object.keys(manifest.hostSHA256).sort(),'Frozen host module inventory');
 for(const [file,sha256] of Object.entries(manifest.hostSHA256))if(file!=='plt-resident-gpu.js')assert.equal(current.hostSHA256[file],sha256,`No unrelated host change: ${file}`);
 return {manifest,transport};
}
const semanticFields=['paths','replayPaths','retainedPaths','retainedMirrorPaths','retainedMirrorEmissionPaths','retainedMirrorEscapePaths','retainedByMirrorCount','retainedIntervalQueries','contributions','reasons','triggerDepths','intervalReasons','rounds','auditSHA256','replayOwnershipSHA256'];
export function assertV22Checkpoint(v22,f,b,result,mode,{includeDispatches=true}={}){
 const batch=v22.transport.cases.find(c=>c.name===f.name)?.batches.find(x=>x.seed===b.seed&&x.sampleStart===b.sampleStart&&x.samplesPerPixel===b.samplesPerPixel);assert(batch,'Exact frozen v22 fixture/batch');
 const expected=batch[mode.id];for(const field of [...semanticFields,...(includeDispatches?['dispatches']:[])])assert.deepEqual(result[field],expected[field],`${f.name}/${mode.id}/${b.seed}/${b.sampleStart}: frozen ${field}`);
}
export function assertChunkEquivalent(a,b,label='Lane chunk parity'){
 for(const field of ['audit','image','pixelIndices','finalRngStates'])assert.deepEqual(b[field],a[field],`${label}: ${field}`);
 assert.deepEqual(b.recovery.keys,a.recovery.keys,`${label}: complete replay ownership, reasons, depths and flags`);assert.equal(b.recovery.paths,a.recovery.paths);assert.equal(b.rounds,a.rounds,'Chunking must not alter round scheduling');
}
export function assertDispatchShape(f,b,candidate,laneChunkSize){
 assert.equal(candidate.recovery.laneChunkSize,laneChunkSize,'Reported effective run chunk size');
 const tile=b.tile??fullTile,pixels=tile.width*tile.height,triangles=candidate.diagnostic.scene.axisGuard.eligible?f.scene.triangles.length:0,passesPerChunk=8+2*triangles;
 assert.equal(candidate.dispatches,candidate.rounds*Math.ceil(pixels/laneChunkSize)*passesPerChunk,'Same distinct per-stage passes in every bounded lane chunk');
 return {laneChunkSize,chunksPerRound:Math.ceil(pixels/laneChunkSize),finalChunkLanes:(pixels-1)%laneChunkSize+1,passesPerChunk};
}
export async function freshReference(f,o,tile,{sampleStart=0,samplesPerPixel=2,wave=false}={}){
 const began=performance.now(),start=createPLTRandomStates(o),end=start.slice();
 if(sampleStart)await buildResidentReferenceBatch(f.scene,f.camera,{...o,wave},{tile,sampleStart:0,samplesPerPixel:sampleStart,rngStates:end});
 const prefix=end.slice(),reference=await buildResidentReferenceBatch(f.scene,f.camera,{...o,wave},{tile,sampleStart,samplesPerPixel,rngStates:end});
 if(!wave)for(const path of reference.audit.paths){if(f.expected!==undefined)assert.equal(path.value,f.expected,'Independent analytic path power');if(f.expectedValues)assert(f.expectedValues.includes(path.value));assert.equal(path.rng.count,2);}
 return {seed:o.seed,options:o,tile,sampleStart,samplesPerPixel,start:prefix,end,reference,cpuFullShadowConstructionMs:performance.now()-began};
}
function negativeControls(b,candidate){
 const replay=new Set(candidate.recovery.keys.map(k=>residentPathKey(k.key))),index=candidate.audit.paths.findIndex(p=>!replay.has(residentPathKey(p.key))&&p.scatters.length>=2);assert(index>=0,'Negative controls require retained multibounce path');
 const changes=[['alter-reflected-origin',c=>{c.audit.paths[index].vertices[1].origin[0]+=.125;}],['reverse-mirror-direction',c=>{c.audit.paths[index].scatters[0].direction=c.audit.paths[index].scatters[0].direction.map(x=>-x);}],['drop-second-mirror',c=>{c.audit.paths[index].scatters.splice(1,1);}],['stale-owner',c=>{c.audit.paths[index].key.sample++;}],['alter-rng-word',c=>{c.audit.paths[index].rng.words[0]^=1;}],['alter-image',c=>{c.audit.reference[0]+=.125;}]];
 return changes.map(([name,mutate])=>{const corrupt=structuredClone(candidate);mutate(corrupt);const raw=compareResidentAudit(b.reference,corrupt.audit,{maxMismatches:4096});assert.equal(raw.ok,false,name);assert.throws(()=>admitResidentWithRecovery(b.reference,corrupt),undefined,name);return {name,rawShadowRejected:true,admissionRejected:true};});
}
export const boundarySpecifications=[
 {id:'one-lane-17x9',width:17,height:9,tile:{x:3,y:2,width:1,height:1},chunks:[1,32,128]},
 {id:'offset-partial-17x9',width:17,height:9,tile:{x:3,y:2,width:13,height:7},chunks:[17,31,32,33,64,127,128]},
 {id:'143-lanes-13x11',width:13,height:11,tile:{x:0,y:0,width:13,height:11},chunks:[32,128]},
 {id:'maximum-legal-32x32',width:32,height:32,tile:{x:0,y:0,width:32,height:32},chunks:[32,128]},
];

export async function main(){
 const execute=process.argv.includes('--execute'),cpuOnly=process.argv.includes('--cpu-only'),reuse=process.argv.includes('--reuse-native-device');assert(!(execute&&cpuOnly));
 const backend=execute?process.env.WEBGPU_EXECUTION_BACKEND:'null';if(execute)assert(['vulkan','metal','d3d11','d3d12','opengl','opengles'].includes(backend),'Explicit native backend required');
 assert.equal(PLT_RESIDENT_LIMITS.dispatchLanes,32,'No default chunk-size change');assert.equal(PLT_RESIDENT_LIMITS.tilePixels,1024,'No tile-domain expansion');
 const v22=readV22(),legacy=await buildCorpus(),adversarial=await buildAdversarialCorpus(),all=[...fixtures,...adversarialFixtures],references=new Map([...legacy.references,...adversarial.references]),provenance=sourceProvenance();
 const report={buildId:PLT_RESIDENT_BUILD_ID,pinnedV22,allPass:false,status:'preparing',executionRequested:execute,actualGPUExecuted:false,backend,nodeVersion:process.version,vulkanICD:process.env.VK_ICD_FILENAMES??null,fullCPUShadowRequired:true,physicalGPUValidated:false,browserValidated:false,hardwarePerformanceClaim:false,fullFixtureSuite:true,modes,chunkSizes:[32,128],legacyCPUFixtures:legacy.cpuFixtures,adversarialCPUFixtures:adversarial.cpuFixtures,sourceSHA256:provenance.sourceSHA256,hostSHA256:provenance.hostSHA256,testSHA256:hash(fs.readFileSync(new URL(import.meta.url))),cases:[],boundaries:[],limitations:['Exact v22 scenes, shaders, tolerances and shadow admission are retained. Additional tile sizes exercise existing limits without enlarging supported geometry.','Test-only reviewed native device/pipeline cache may be used here. This report is correctness evidence, not a timing comparison.','Each guard is evaluated independently; mathematical guards are not combined.','Default/32 and opt-in/128 must have identical complete audits, RNG, image and replay certificates; physical dispatch counts intentionally differ.']};
 const output=new URL(`../research/plt-resident-chunk-${cpuOnly?'cpu-validation':execute?'validation':'compile-validation'}.json`,import.meta.url),save=()=>fs.writeFileSync(output,JSON.stringify(report,null,2)+'\n');
 const boundaryPrepared=[];for(const spec of boundarySpecifications){const f=fixtures.find(f=>f.name==='one-mirror-emission'),o={...options,width:spec.width,height:spec.height},b=await freshReference(f,o,spec.tile);boundaryPrepared.push({spec,f,o,b});report.boundaries.push({...spec,paths:b.reference.audit.paths.length,cpuFullShadowConstructionMs:b.cpuFullShadowConstructionMs,runs:[]});}
 if(cpuOnly){report.allPass=true;report.status='cpu-fixtures-passed';report.summary={legacyPaths:legacy.cpuFixtures.reduce((n,f)=>n+f.paths,0),adversarialPaths:adversarial.cpuFixtures.reduce((n,f)=>n+f.paths,0),boundaryPaths:boundaryPrepared.reduce((n,p)=>n+p.b.reference.audit.paths.length,0),fixtures:all.length};save();console.log(JSON.stringify({allPass:true,cpuOnly:true,summary:report.summary}));return;}
 let gpu=create([`backend=${backend}`]);const cache=reuse?reuseNativeDevice(gpu):null;if(cache)gpu=cache.gpu;report.nativeDeviceReuse=cache?.stats??{enabled:false};report.status='compiling';save();
 const setup=async(f,o)=>{const began=performance.now(),runner=await createPLTResidentRunner(f.scene,f.camera,o,{gpu,shaderSource:provenance.sources.join('\n'),timeoutMs:120000,recoveryMode:'selective',continuationGuard:'axis-mirror-camera',onSetupProgress:s=>console.log('COMPILE',f.name,s.entryPoint)});assert.equal(runner.diagnostic.dispatchLanes,32);return {runner,setupWallMs:performance.now()-began};};
 try{
  for(const f of execute?all:all.slice(0,1)){
   const batches=references.get(f.name),{runner,setupWallMs}=await setup(f,batches[0].options),record={name:f.name,legacy:fixtures.includes(f),setupWallMs,adapter:runner.diagnostic.adapter,batches:[]};report.cases.push(record);save();
   try{if(execute)for(const b of batches){
    const comparison={seed:b.seed,sampleStart:b.sampleStart,samplesPerPixel:b.samplesPerPixel,paths:b.reference.audit.paths.length,runs:[]};record.batches.push(comparison);
    for(const mode of modes){let canonical;
     for(const laneChunkSize of [32,128]){const rng=b.start.slice(),candidate=await runner.run({tile:b.tile??fullTile,sampleStart:b.sampleStart,samplesPerPixel:b.samplesPerPixel,wave:false,rngStates:rng,guard:mode.guard,axisSubmission:'round',laneChunkSize});report.actualGPUExecuted=true;report.status='validating';assert.deepEqual(rng,b.start,'Immutable caller RNG');const result=validateCandidate(f,b,candidate,mode);assertV22Checkpoint(v22,f,b,result,mode,{includeDispatches:laneChunkSize===32});if(canonical)assertChunkEquivalent(canonical,candidate);else canonical=candidate;comparison.runs.push({modeId:mode.id,...assertDispatchShape(f,b,candidate,laneChunkSize),...result,exactFrozenV22SemanticParity:true});
      if(f.name==='two-mirrors-emission'&&b.seed===options.seed&&b.sampleStart===0&&mode.id==='camera-round'&&laneChunkSize===128)report.decodedNegativeControls=negativeControls(b,candidate);
     }
     if(b===batches[0]){const rng=b.start.slice(),candidate=await runner.run({tile:b.tile??fullTile,sampleStart:b.sampleStart,samplesPerPixel:b.samplesPerPixel,wave:false,rngStates:rng,guard:mode.guard,axisSubmission:'round'});assert.deepEqual(rng,b.start);validateCandidate(f,b,candidate,mode);assertChunkEquivalent(canonical,candidate);assert.equal(candidate.dispatches,canonical.dispatches,'Omitted override must restore constructor/default 32 after 128');comparison.defaultRestoration??=[];comparison.defaultRestoration.push({modeId:mode.id,defaultChunkSize:32,exactParity:true});}
    }save();console.log('PASS',f.name,b.seed,b.sampleStart,'32/128/default');
   }}finally{runner.destroy();}
  }
  if(execute)for(const [index,p] of boundaryPrepared.entries()){
   const {f,o,b,spec}=p,{runner,setupWallMs}=await setup(f,o),record=report.boundaries[index];record.setupWallMs=setupWallMs;
   try{for(const mode of modes){let canonical;for(const laneChunkSize of spec.chunks){const rng=b.start.slice(),candidate=await runner.run({tile:b.tile,sampleStart:b.sampleStart,samplesPerPixel:b.samplesPerPixel,wave:false,rngStates:rng,guard:mode.guard,axisSubmission:'round',laneChunkSize});assert.deepEqual(rng,b.start);const result=validateCandidate(f,b,candidate,mode);if(canonical)assertChunkEquivalent(canonical,candidate);else canonical=candidate;record.runs.push({modeId:mode.id,...assertDispatchShape(f,b,candidate,laneChunkSize),...result,exactAcrossChunkParity:true});save();} }
    if(spec.id==='one-lane-17x9'){const wave=await freshReference(f,o,b.tile,{samplesPerPixel:1,wave:true});record.waveDomainRejection=[];for(const mode of modes)for(const laneChunkSize of [32,128]){const rng=wave.start.slice(),candidate=await runner.run({tile:b.tile,sampleStart:0,samplesPerPixel:1,wave:true,rngStates:rng,guard:mode.guard,axisSubmission:'round',laneChunkSize});assert.deepEqual(rng,wave.start);assert.equal(candidate.recovery.paths,1);assert(candidate.recovery.keys.every(k=>k.reasons.includes('unmodelledContinuation')));assert(compareResidentAudit(wave.reference,candidate.audit).ok);const admitted=admitResidentWithRecovery(wave.reference,candidate);assert(admitted.validation.passed);assert.equal(admitted.recovery.numericGeometry.count,0);record.waveDomainRejection.push({modeId:mode.id,laneChunkSize,paths:1,rawShadowPassed:true,numericGeometryReplacements:0});}}
   }finally{runner.destroy();}console.log('PASS boundary',spec.id);
  }
  if(execute){const sums=legacy=>Object.fromEntries(modes.map(mode=>[mode.id,Object.fromEntries([32,128].map(chunk=>{const runs=report.cases.filter(c=>c.legacy===legacy).flatMap(c=>c.batches.flatMap(b=>b.runs)).filter(r=>r.modeId===mode.id&&r.laneChunkSize===chunk);return [chunk,Object.fromEntries(['paths','replayPaths','retainedMirrorPaths','numericGeometryReplacements'].map(field=>[field,runs.reduce((n,r)=>n+r[field],0)]))];}))]));report.summary={fixtures:report.cases.length,legacy:sums(true),adversarial:sums(false),boundaryRuns:report.boundaries.reduce((n,b)=>n+b.runs.length,0),fullAuditRngImageReplayParity:true,default32Preserved:true};for(const chunk of [32,128]){assert.equal(report.summary.legacy['anchor-round'][chunk].paths,4608);assert.equal(report.summary.legacy['anchor-round'][chunk].replayPaths,1596);assert.equal(report.summary.legacy['camera-round'][chunk].replayPaths,1349);assert.equal(report.summary.adversarial['camera-round'][chunk].paths,4563);}}
  readV22();assert.deepEqual(sourceProvenance().sourceSHA256,report.sourceSHA256);assert.deepEqual(sourceProvenance().hostSHA256,report.hostSHA256,'Host changed during validation');assert.equal(hash(fs.readFileSync(new URL(import.meta.url))),report.testSHA256);report.allPass=true;report.status='passed';
 }catch(error){report.status='failed';report.failure={name:error.name,message:error.message,stack:error.stack};throw error;}
 finally{try{cache?.destroy();}finally{gpu=null;save();}}
 console.log(JSON.stringify({allPass:report.allPass,actualGPUExecuted:report.actualGPUExecuted,summary:report.summary,output:output.pathname}));
}
if(process.argv[1]&&pathToFileURL(process.argv[1]).href===import.meta.url)await main();
