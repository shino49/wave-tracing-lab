import assert from 'node:assert/strict';
import {createPLTScene,buildPLTBatch,createPLTRandomStates,randomGenerator} from '../dist/plt-core.js';
import {buildResidentReferenceBatch,compareResidentAudit,assertResidentAudit,residentHashWords,residentPathKey,RESIDENT_AUDIT_LIMITS} from '../dist/plt-resident-reference.js';
const began=performance.now(),tile={x:1,y:1,width:6,height:6},base={width:8,height:8,samplesPerPixel:4,maxDepth:6,wavelengthMm:.025,seed:7391};
const clone=structuredClone;
let retained,unusedEvents=0,maxDraws=0;
for(const preset of ['gallery','double-edge','baffles'])for(const wave of [false,true]){
 const scene=createPLTScene({preset}),options={...base,wave},referenceStates=createPLTRandomStates(options),ordinaryStates=referenceStates.slice();
 for(const sampleStart of [0,2]){
  const batch={tile,sampleStart,samplesPerPixel:2},cpu=await buildPLTBatch(scene,scene.camera,options,{...batch,rngStates:ordinaryStates}),job=await buildResidentReferenceBatch(scene,scene.camera,options,{...batch,rngStates:referenceStates});
  for(const name of ['events','terms','paths','stats','reference','pixelIndices','multiRegionEvidence'])assert.deepEqual(job[name],cpu[name],`Instrumentation changed ${name}`);
  assert.deepEqual(referenceStates,ordinaryStates,'Instrumentation changed final RNG states');
  assert.equal(job.audit.paths.length,tile.width*tile.height*2);assert.ok(job.audit.paths.length>job.paths.length,'Audit must not inherit 12 diagnostic path cap');
  assert.equal(job.audit.totals.events,job.events.length);assert.equal(job.audit.complete,true);
  assert.equal(new Set(job.audit.paths.map(p=>residentPathKey(p.key))).size,job.audit.paths.length);
  const allEvents=[],allTerms=[];
  for(const path of job.audit.paths){
   const rng=randomGenerator(path.rng.start),words=Array.from({length:path.rng.count},()=>rng()*4294967296);
   assert.deepEqual(path.rng.words,words);assert.equal(path.rng.final,rng.getState());assert.equal(path.rng.hash,residentHashWords(words));maxDraws=Math.max(maxDraws,words.length);
   assert.ok(path.rng.count>=2,'Must include camera draws');assert.ok(path.vertices.length>=1);assert.ok(path.termination);assert.equal(path.camera.origin.length,3);
   for(const event of path.localEvents){allEvents.push(event.id);assert.equal(event.F,event.operator.referenceF);assert.equal(event.edgeHash,residentHashWords(event.edgeIds));assert.deepEqual(event.key,{...path.key,event:event.localId});unusedEvents+=!event.used;}
   for(const c of path.contributions){assert.equal(c.value,c.terms.reduce((sum,t)=>sum+t.eventIds.reduce((v,id)=>v*job.events[id].referenceF,t.weight),0));allTerms.push(...c.terms);}
  }
  assert.deepEqual(allEvents,Array.from({length:job.events.length},(_,i)=>i));assert.deepEqual(allTerms,job.terms);
  const result=assertResidentAudit(job,clone(job.audit));assert.equal(result.checked.events,job.events.length);assert.equal(result.checked.edgeRecords,job.audit.totals.edgeRecords);assert.equal(result.checked.scatters,job.audit.totals.scatters);assert.equal(result.checked.paths,job.audit.paths.length);assert.equal(result.checked.pixels,job.reference.length);assert.equal(result.checked.randomWords,job.audit.totals.randomWords);
  if(wave&&preset==='gallery'&&sampleStart===0)retained=job;
 }
}
assert.ok(unusedEvents>0,'Coverage requires unused FSD events that never reach a contribution');
let adversarialControls=0;
const reject=(label,mutate)=>{adversarialControls++;const bad=clone(retained.audit);mutate(bad);const result=compareResidentAudit(retained,bad);assert.equal(result.ok,false,`${label} was accepted`);assert.ok(result.failures.length>0);assert.throws(()=>assertResidentAudit(retained,bad),{code:'RESIDENT_AUDIT_MISMATCH'});};
reject('missing path',a=>a.paths.pop());
reject('duplicate stable path key',a=>a.paths[1].key={...a.paths[0].key});
reject('wrong wave key',a=>a.paths[0].key.wave=false);
reject('wrong sample key',a=>a.paths[0].key.sample+=10);
reject('raw RNG word mismatch despite matching hash',a=>a.paths[0].rng.words[0]^=1);
reject('RNG final state mismatch',a=>a.paths[0].rng.final^=1);
reject('missing RNG words',a=>delete a.paths[0].rng.words);
reject('triangle topology mismatch',a=>a.paths.find(p=>p.vertices.some(v=>v.triangleId!==null)).vertices.find(v=>v.triangleId!==null).triangleId+=1);
reject('aperture topology mismatch',a=>a.paths[0].vertices[0].apertureCount+=1);
reject('vertex geometry mismatch',a=>a.paths[0].vertices[0].origin[0]+=.01);
reject('local F mismatch',a=>a.paths.find(p=>p.localEvents.length).localEvents[0].F+=1);
reject('ordered edge identity mismatch',a=>a.paths.find(p=>p.localEvents.length).localEvents[0].edgeHash^=1);
reject('unused local F mismatch with unchanged pixels',a=>{const e=a.paths.flatMap(p=>p.localEvents).find(e=>!e.used);assert.ok(e);e.F+=10;});
reject('missing event with unchanged pixels',a=>a.paths.find(p=>p.localEvents.length).localEvents.pop());
reject('wrong deferred sampled branch',a=>{const e=a.paths.find(p=>p.localEvents.length).localEvents[0];e.sampled=!e.sampled;});
reject('missing contribution',a=>a.paths.find(p=>p.contributions.length).contributions.pop());
reject('contribution scalar mismatch',a=>a.paths.find(p=>p.contributions.length).contributions[0].value+=1);
reject('contribution weight mismatch',a=>a.paths.find(p=>p.contributions.length).contributions[0].weight+=1);
reject('wrong termination',a=>a.paths[0].termination.reason='wrong');
reject('pixel mismatch',a=>a.reference[0]+=1);
reject('missing pixel audit',a=>delete a.reference);
reject('incomplete batch',a=>a.complete=false);
reject('batch shader flags',a=>a.flags=2);
reject('missing batch flags',a=>delete a.flags);
reject('path shader flags',a=>a.paths[0].flags=1);
reject('missing path flags',a=>delete a.paths[0].flags);
reject('negative near-zero pixel',a=>{const i=a.reference.findIndex(v=>v===0);assert.ok(i>=0);a.reference[i]=-1e-7;});
reject('nonfinite pixel',a=>a.reference[0]=NaN);
reject('nonfinite local event',a=>a.paths.find(p=>p.localEvents.length).localEvents[0].F=Infinity);
reject('missing local event source',a=>delete a.paths.find(p=>p.localEvents.length).localEvents[0].source);
reject('local event source altered',a=>a.paths.find(p=>p.localEvents.length).localEvents[0].source[0]+=.01);
reject('missing exact accepted-edge records',a=>delete a.paths.find(p=>p.localEvents.length).localEvents[0].records);
reject('accepted edge ID changed without changing fingerprint',a=>a.paths.flatMap(p=>p.localEvents).find(e=>e.records.length).records[0].edgeId^=1);
reject('accepted diffraction point changed',a=>a.paths.flatMap(p=>p.localEvents).find(e=>e.records.length).records[0].point[0]+=.01);
for(const field of ['ri','ro','sinBeta','phiI','phiO','n','phaseExcess','L','fermatResidual'])reject(`accepted edge ${field} changed`,a=>a.paths.flatMap(p=>p.localEvents).find(e=>e.records.length).records[0][field]+=.01);
reject('missing complete scatter trace',a=>delete a.paths.find(p=>p.scatters.length).scatters);
reject('final-depth scatter changed despite no following vertex',a=>{const p=a.paths.find(p=>p.termination.reason==='max-depth');assert.ok(p);const s=p.scatters.at(-1);if(s.type==='fsd')s.sample.wo[0]+=.01;else s.direction[0]+=.01;});
reject('rejected FSD sample direction changed',a=>{const s=a.paths.flatMap(p=>p.scatters).find(s=>s.type==='fsd'&&!s.sample.valid);assert.ok(s);s.sample.wo[0]+=.01;});
reject('FSD selected edge ID changed',a=>{const s=a.paths.flatMap(p=>p.scatters).find(s=>s.type==='fsd');s.sample.edgeId=123456;});
reject('FSD discrete/direct branch changed',a=>{const s=a.paths.flatMap(p=>p.scatters).find(s=>s.type==='fsd');s.sample.isDirect=!s.sample.isDirect;});
reject('FSD inverse PDF weight changed',a=>a.paths.flatMap(p=>p.scatters).find(s=>s.type==='fsd').sample.weight+=1);
reject('FSD PDF changed',a=>a.paths.flatMap(p=>p.scatters).find(s=>s.type==='fsd').sample.pdf.value+=1);
reject('surface sampling PDF changed',a=>a.paths.flatMap(p=>p.scatters).find(s=>s.type==='diffuse').pdf+=1);

const reordered=clone(retained.audit);reordered.paths.reverse();assertResidentAudit(retained,reordered);
const near=clone(retained.audit);near.reference[0]+=1e-6;assertResidentAudit(retained,near);
assert.throws(()=>compareResidentAudit(retained,retained,{geometryAbsolute:1}),RangeError,'Cannot silently widen geometry admission');
// A billion-unit translation must not turn a .01-unit point error into a pass.
const translated=clone(retained.audit),shift=[1e9,-1e9,1e9],translate=p=>{if(p)for(let i=0;i<3;i++)p[i]+=shift[i];};
for(const p of translated.paths){translate(p.camera.origin);for(const v of p.vertices){translate(v.origin);translate(v.center);}for(const e of p.localEvents){translate(e.source);translate(e.destination);for(const r of e.records)translate(r.point);}for(const s of p.scatters)if(s.type==='fsd')translate(s.sample.point);}
assertResidentAudit(translated,clone(translated));
for(const [name,mutate] of [
 ['vertex',a=>a.paths[0].vertices[0].origin[0]+=.01],
 ['event source',a=>a.paths.find(p=>p.localEvents.length).localEvents[0].source[0]+=.01],
 ['accepted-edge point',a=>a.paths.flatMap(p=>p.localEvents).find(e=>e.records.length).records[0].point[0]+=.01],
 ['FSD sampled point',a=>a.paths.flatMap(p=>p.scatters).find(s=>s.type==='fsd').sample.point[0]+=.01],
]){const bad=clone(translated);mutate(bad);assert.equal(compareResidentAudit(translated,bad).ok,false,`Translated ${name} error was hidden by world-coordinate scale`);adversarialControls++;}

const scene=createPLTScene(),options={...base},makeBatch=()=>({tile,samplesPerPixel:2,rngStates:createPLTRandomStates(options)});
await assert.rejects(()=>buildResidentReferenceBatch(scene,scene.camera,options,{...makeBatch(),auditCapacity:{maxPaths:1}}),{code:'RESIDENT_AUDIT_CAPACITY'});
await assert.rejects(()=>buildResidentReferenceBatch(scene,scene.camera,options,{...makeBatch(),auditCapacity:{maxRandomWordsPerPath:1}}),{code:'RESIDENT_AUDIT_CAPACITY'});
await assert.rejects(()=>buildResidentReferenceBatch(scene,scene.camera,options,{...makeBatch(),auditCapacity:{maxVerticesPerPath:1}}),{code:'RESIDENT_AUDIT_CAPACITY'});
await assert.rejects(()=>buildResidentReferenceBatch(scene,scene.camera,options,{...makeBatch(),auditCapacity:{maxPaths:RESIDENT_AUDIT_LIMITS.maxPaths+1}}),RangeError);
const controller=new AbortController();controller.abort();await assert.rejects(()=>buildResidentReferenceBatch(scene,scene.camera,options,{...makeBatch(),signal:controller.signal}),{name:'AbortError'});
// Exercise the full 4096-path cap with cheap escaping rays. The persistent audit
// remains exactly one tile, not the frame, regardless of frame dimensions.
const fullOptions={width:32,height:32,samplesPerPixel:4,wave:false,maxDepth:1},fullScene=createPLTScene(),away={...fullScene.camera,target:[100,100,100]};
const full=await buildResidentReferenceBatch(fullScene,away,fullOptions,{tile:{x:0,y:0,width:32,height:32},samplesPerPixel:4,rngStates:createPLTRandomStates(fullOptions)});assert.equal(full.audit.paths.length,4096);assert.equal(full.audit.totals.randomWords,8192);assertResidentAudit(full,full);
console.log(JSON.stringify({allPass:true,fixtureBatches:12,adversarialControls,unusedEvents,maxDraws,fullCapPaths:full.audit.paths.length,elapsedMs:performance.now()-began}));
