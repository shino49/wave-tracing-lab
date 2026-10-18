/** Host-side recovery proof and adversarial controls. These canonical fixtures
 * are not evidence of native GPU execution or an unaudited performance mode.
 */
import assert from 'node:assert/strict';
import {createPLTScene,createPLTRandomStates} from '../dist/plt-core.js';
import {buildResidentReferenceBatch,compareResidentAudit,assertResidentAudit,residentPathKey,RESIDENT_AUDIT_TOLERANCES} from '../dist/plt-resident-reference.js';
import {admitResidentWithRecovery,RESIDENT_RECOVERY_LIMITS} from '../dist/plt-resident-recovery.js';

const began=performance.now(),clone=structuredClone,scene=createPLTScene({preset:'gallery'}),options={width:8,height:8,samplesPerPixel:3,maxDepth:6,wave:false},tile={x:1,y:0,width:7,height:8},rngStates=createPLTRandomStates(options);
const first=await buildResidentReferenceBatch(scene,scene.camera,options,{tile,sampleStart:0,samplesPerPixel:2,rngStates});
const later=await buildResidentReferenceBatch(scene,scene.camera,options,{tile,sampleStart:2,samplesPerPixel:1,rngStates});
const wave=await buildResidentReferenceBatch(scene,scene.camera,{...options,wave:true},{tile,sampleStart:0,samplesPerPixel:2,rngStates:createPLTRandomStates(options)});
let checks=0;
function check(name,fn){fn();checks++;console.log('PASS',name);}
function sumF32(values){let total=0,correction=0;for(const value of values){const adjusted=Math.fround(Math.fround(value)-correction),next=Math.fround(total+adjusted);correction=Math.fround(Math.fround(next-total)-adjusted);total=next;}return total;}
function assemble(candidate,{f32=false}={}){
 const a=candidate.audit,values=new Map();for(const pixel of a.pixelIndices)values.set(pixel,[]);
 for(const path of a.paths)values.get(path.key.pixel).push(path);
 const image=f32?new Float32Array(a.pixelIndices.length):new Float64Array(a.pixelIndices.length);
 for(let i=0;i<a.pixelIndices.length;i++){
  const paths=values.get(a.pixelIndices[i]).sort((x,y)=>x.key.sample-y.key.sample),samples=paths.map(p=>p.sampleValue??p.value);
  image[i]=(f32?sumF32(samples):samples.reduce((s,v)=>s+v,0))/a.samplesPerPixel;
 }
 a.reference=image;candidate.image=image;candidate.pixelIndices=a.pixelIndices;
 candidate.finalRngStates=new Uint32Array([...a.pixelIndices].map(pixel=>values.get(pixel).at(-1).rng.final));
 return candidate;
}
function fixture(job,{f32=false}={}){
 const audit=clone(job.audit);
 for(const path of audit.paths){
  if(f32){for(const term of path.contributions)term.value=Math.fround(term.value);path.sampleValue=sumF32(path.contributions.map(term=>term.value));delete path.value;}
  else path.sampleValue=path.value;
 }
 return assemble({candidate:true,admitted:false,audit,timings:{fixture:1},validation:{passed:false}},{f32});
}
const eligiblePath=candidate=>candidate.audit.paths.find(path=>path.vertices.some(v=>v.distance!==null)&&path.contributions.some(c=>c.value>0));
function drift(candidate,path=eligiblePath(candidate)){
 const vertex=path.vertices.find(v=>v.distance!==null),bound=RESIDENT_AUDIT_TOLERANCES.geometryAbsolute+RESIDENT_AUDIT_TOLERANCES.geometryRelative*Math.abs(vertex.distance);
 vertex.distance+=bound*2;return residentPathKey(path.key);
}
function reject(name,mutate,{job=first,withDrift=true}={}){
 check(name,()=>{const c=fixture(job);if(withDrift)drift(c);mutate(c);const before=clone(c);assert.throws(()=>admitResidentWithRecovery(job,c),{code:'RESIDENT_AUDIT_MISMATCH'});assert.deepEqual(c,before,'Rejected input was mutated');});
}

check('strictly passing native-shaped candidate stays unchanged and fully audited',()=>{
 const c=fixture(first,{f32:true}),before=clone(c),cpuBefore=clone(first),r=admitResidentWithRecovery(first,c);
 assert.equal(r.validation.ok,true);assert.equal(r.validation.fullCPUShadow,true);assert.equal(r.recovery.numericGeometry.count,0);assert.equal(r.recovery.paths,0);
 assert.equal(r.image,r.audit.reference);assert.deepEqual(r.audit,c.audit);assert.deepEqual(c,before);assert.deepEqual(first,cpuBefore);assert.notEqual(r.audit,c.audit);
 assert.equal(r.validation.checked.paths,112);assert.equal(r.validation.checked.pixels,56);assert.equal(r.timings.fixture,1);
 for(const key of ['residentRecoveryInitialAuditMs','residentRecoveryReplacementMs','residentRecoveryFinalAuditMs','residentRecoveryTotalMs'])assert(r.timings[key]>=0);
});
check('known gallery pixel39/sample2 distance is CPU-replaced without widening strict gate',()=>{
 const c=fixture(later),p=c.audit.paths.find(path=>path.key.pixel===39),before=clone(c),cpuBefore=clone(later);
 assert.equal(p.key.sample,2);assert(Math.abs(p.vertices[4].distance-5.005503176330083)<1e-12);p.vertices[4].distance=5.00554084777832;
 assert.throws(()=>assertResidentAudit(later,c.audit),{code:'RESIDENT_AUDIT_MISMATCH'});
 const r=admitResidentWithRecovery(later,c),metadata=r.recovery.numericGeometry;
 assert.equal(metadata.count,1);assert.deepEqual(metadata.keys,['0:39:2']);assert.equal(metadata.initialAudit.mismatchCount,1);assert.equal(metadata.initialAudit.ok,false);assert.equal(metadata.finalAudit.ok,true);
 assert.equal(metadata.reasons[0].field,'path[0:39:2].vertices[4].distance');assert.equal(metadata.reasons[0].reason,'finite-geometry-drift');
 assert.equal(r.audit.paths.find(path=>path.key.pixel===39).vertices[4].distance,before.audit.paths.find(path=>path.key.pixel===39).vertices[4].distance);
 assert.equal(c.audit.paths.find(path=>path.key.pixel===39).vertices[4].distance,5.00554084777832);assert.equal(c.admitted,false);
 assert.throws(()=>assertResidentAudit(later,c.audit),{code:'RESIDENT_AUDIT_MISMATCH'});assertResidentAudit(later,r.audit);assert.deepEqual(later,cpuBefore);
});
check('replacement preserves retained samples and counts each path once',()=>{
 const c=fixture(first),groups=new Map();for(const p of c.audit.paths){const a=groups.get(p.key.pixel)??[];a.push(p);groups.set(p.key.pixel,a);}
 const paths=[...groups.values()].find(paths=>paths.every(p=>p.value>0&&p.vertices.some(v=>v.distance!==null)));assert(paths);
 const [target,retained]=paths;
 for(const path of paths){for(const contribution of path.contributions)contribution.value*=1.0001;path.value=path.sampleValue=path.contributions.reduce((s,c)=>s+c.value,0);}
 assemble(c);const key=drift(c,target),snapshot=clone(c),r=admitResidentWithRecovery(first,c),pixelIndex=[...c.audit.pixelIndices].indexOf(target.key.pixel),cpu=first.audit.paths.find(p=>residentPathKey(p.key)===key);
 assert.equal(r.image[pixelIndex],(cpu.value+retained.sampleValue)/2);assert.notEqual(r.image[pixelIndex],first.audit.reference[pixelIndex]);
 assert.deepEqual(r.audit.paths.find(p=>residentPathKey(p.key)===residentPathKey(retained.key)),retained);assert.deepEqual(c,snapshot);assert.equal(r.recovery.paths,1);
});
check('multiple geometry errors in one path produce one replacement',()=>{
 const c=fixture(first),path=eligiblePath(c),key=drift(c,path);path.vertices[0].origin[0]+=.00006;
 const r=admitResidentWithRecovery(first,c);assert.equal(r.recovery.numericGeometry.count,1);assert.deepEqual(r.recovery.numericGeometry.keys,[key]);assert.equal(r.recovery.numericGeometry.reasons.length,2);
});
check('path-order changes do not change sample ownership or accumulation',()=>{
 const c=fixture(first,{f32:true});drift(c);c.audit.paths.reverse();const r=admitResidentWithRecovery(first,c);assertResidentAudit(first,r.audit);assert.equal(r.recovery.paths,1);
});
check('prior CPU replay metadata survives post-audit numerical recovery',()=>{
 const c=fixture(first);drift(c);const previous={key:{wave:false,pixel:9,sample:0},flags:4,reasons:['coneTie']};c.recovery={mode:'selective',paths:1,keys:[previous],completeCPUPathReplay:true};
 const r=admitResidentWithRecovery(first,c);assert.equal(r.recovery.mode,'selective');assert.equal(r.recovery.completeCPUPathReplay,true);assert.equal(r.recovery.paths,2);assert.deepEqual(r.recovery.keys[0],previous);assert.equal(r.recovery.numericGeometry.count,1);
});
check('CPU replay value and GPU sampleValue paths assemble together',()=>{
 const c=fixture(first,{f32:true}),target=c.audit.paths.find(p=>p.key.pixel===first.audit.paths[0].key.pixel&&p.key.sample===0),cpu=first.audit.paths.find(p=>residentPathKey(p.key)===residentPathKey(target.key));
 c.audit.paths[c.audit.paths.indexOf(target)]=clone(cpu);assemble(c);drift(c);const r=admitResidentWithRecovery(first,c);assertResidentAudit(first,r.audit);assert.equal(r.recovery.numericGeometry.count,1);
});
check('sub-tolerance drift is strictly accepted without CPU replacement',()=>{
 const c=fixture(first),p=eligiblePath(c);p.vertices[0].origin[0]+=1e-7;const r=admitResidentWithRecovery(first,c);assert.equal(r.recovery.numericGeometry.count,0);assert.deepEqual(r.audit.paths,c.audit.paths);
});

reject('candidate self-asserted validation cannot replace complete audit',c=>{delete c.audit;c.validation={passed:true};});
reject('missing path array',c=>delete c.audit.paths);
reject('missing path',c=>c.audit.paths.pop());
reject('duplicate path key',c=>c.audit.paths[1].key={...c.audit.paths[0].key});
reject('null path entry',c=>c.audit.paths[1]=null);
reject('incomplete audit',c=>c.audit.complete=false);
reject('batch flag',c=>c.audit.flags=1);
reject('unknown batch flag',c=>c.audit.flags=0x40000000);
reject('missing batch flags',c=>delete c.audit.flags);
reject('path flag',c=>eligiblePath(c).flags=1);
reject('unknown path flag',c=>eligiblePath(c).flags=0x40000000);
reject('missing path flags',c=>delete eligiblePath(c).flags);
reject('wrong final RNG state',c=>eligiblePath(c).rng.final^=1);
reject('raw RNG word changed with matching summaries',c=>eligiblePath(c).rng.words[0]^=1);
reject('missing raw RNG words',c=>delete eligiblePath(c).rng.words);
reject('wrong final RNG commit buffer',c=>c.finalRngStates[0]^=1);
reject('missing triangle identity',c=>delete eligiblePath(c).vertices[0].triangleId);
reject('different aperture topology',c=>eligiblePath(c).vertices[0].apertureCount++);
reject('different vertex depth',c=>eligiblePath(c).vertices[0].depth++);
reject('different termination reason',c=>eligiblePath(c).termination.reason='escaped-wrong');
reject('missing geometry coordinate',c=>delete eligiblePath(c).vertices[0].origin);
reject('NaN geometry coordinate',c=>eligiblePath(c).vertices[0].origin[0]=NaN);
reject('infinite geometry scalar',c=>eligiblePath(c).vertices[0].distance=Infinity);
reject('numeric-looking string is not finite geometry evidence',c=>eligiblePath(c).vertices[0].distance='1.00001');
reject('negative radius is not eligible geometry',c=>eligiblePath(c).vertices[0].radius=-.00005);
reject('gross finite geometry error',c=>eligiblePath(c).vertices[0].origin[0]+=1);
reject('missing contribution',c=>eligiblePath(c).contributions.pop());
reject('power disagreement cannot be hidden by geometry replacement',c=>eligiblePath(c).contributions[0].value+=1);
reject('weight disagreement cannot be hidden by geometry replacement',c=>eligiblePath(c).contributions[0].weight+=1);
reject('scatter branch disagreement',c=>c.audit.paths.find(p=>p.scatters.length).scatters[0].type='mirror-wrong');
reject('gross image mismatch',c=>c.audit.reference[0]+=1);
reject('tiny altered image sum still inside power tolerance',c=>c.audit.reference[0]+=1e-10);
reject('image identity substitution',c=>c.image=c.image.slice());
reject('audited pixel order substitution',c=>{c.audit.pixelIndices.reverse();c.audit.reference.reverse();});
reject('unaudited pixel order substitution',c=>{c.pixelIndices=c.pixelIndices.slice();c.pixelIndices.reverse();});
reject('missing sample count',c=>delete c.audit.samplesPerPixel);
reject('wrong sample count',c=>c.audit.samplesPerPixel++);
reject('wrong sample prefix',c=>c.audit.sampleStart++);
reject('missing complete path sum',c=>{const p=eligiblePath(c);delete p.value;delete p.sampleValue;});
reject('contradictory path sum fields',c=>eligiblePath(c).sampleValue+=1e-10);
reject('corrupt path sum and image cannot erase contribution mismatch',c=>{const p=eligiblePath(c);p.sampleValue=p.value+=1e-10;assemble(c);});
reject('sampleValue corruption on a zero-contribution path',c=>{const p=c.audit.paths.find(p=>!p.contributions.length);assert(p);p.sampleValue=p.value=1e-10;assemble(c);});
reject('passing strict gate alone cannot admit an altered sum',c=>{c.audit.reference[0]+=1e-10;assert(compareResidentAudit(first,c.audit).ok);},{withDrift:false});
reject('unused local event power stays mandatory',c=>{const event=c.audit.paths.flatMap(p=>p.localEvents).find(e=>!e.used);assert(event);event.F+=1;},{job:wave});
reject('ordered edge topology stays exact',c=>{const event=c.audit.paths.flatMap(p=>p.localEvents).find(e=>e.records.length);assert(event);event.records[0].edgeId^=1;},{job:wave});
reject('edge-record missing geometry is not recoverable',c=>{const event=c.audit.paths.flatMap(p=>p.localEvents).find(e=>e.records.length);delete event.records[0].ro;},{job:wave});
reject('FSD sampled direction cannot hide missing sample validity',c=>{const s=c.audit.paths.flatMap(p=>p.scatters).find(s=>s.type==='fsd');delete s.sample.valid;},{job:wave});
reject('FSD sampled power cannot hide behind geometry',c=>{const s=c.audit.paths.flatMap(p=>p.scatters).find(s=>s.type==='fsd');s.sample.weight+=1;},{job:wave});

check('bounded event, edge and scatter geometry can recover with unchanged power/topology',()=>{
 const c=fixture(wave),eventPath=c.audit.paths.find(p=>p.localEvents.some(e=>e.records.length)),event=eventPath.localEvents.find(e=>e.records.length),scatterPath=c.audit.paths.find(p=>p.scatters.some(s=>s.type==='fsd'));
 event.source[0]+=.00006;event.records[0].ro+=2*(.00002+.000003*Math.abs(event.records[0].ro));scatterPath.scatters.find(s=>s.type==='fsd').sample.wo[0]+=.00006;
 const r=admitResidentWithRecovery(wave,c);assertResidentAudit(wave,r.audit);assert.equal(r.recovery.numericGeometry.count,new Set([residentPathKey(eventPath.key),residentPathKey(scatterPath.key)]).size);
});
check('translated world coordinates cannot turn gross absolute drift into recovery',()=>{
 const job=clone(first),shift=[1e9,-1e9,1e9],move=value=>{if(value)for(let i=0;i<3;i++)value[i]+=shift[i];};
 for(const p of job.audit.paths){move(p.camera.origin);for(const v of p.vertices){move(v.center);move(v.origin);}for(const e of p.localEvents){move(e.source);move(e.destination);for(const r of e.records)move(r.point);}for(const s of p.scatters)if(s.type==='fsd')move(s.sample.point);}
 const c=fixture(job);eligiblePath(c).vertices[0].origin[0]+=.01;assert.throws(()=>admitResidentWithRecovery(job,c),{code:'RESIDENT_AUDIT_MISMATCH'});
});
check('truncated failure report can never authorize replacements',()=>{
 const job=clone(first),template=job.audit.paths.flatMap(p=>p.vertices).find(v=>v.distance!==null);assert(template);
 // An intentionally overlong synthetic audit isolates the report-cap defense.
 job.audit.paths[0].vertices=Array.from({length:RESIDENT_RECOVERY_LIMITS.maxMismatches+1},()=>clone(template));
 const c=fixture(job);for(const v of c.audit.paths[0].vertices)v.distance+=2*(.00002+.000003*Math.abs(v.distance));
 assert.throws(()=>admitResidentWithRecovery(job,c),error=>error.code==='RESIDENT_AUDIT_MISMATCH'&&/truncated/.test(error.message)&&error.audit.mismatchCount>error.audit.failures.length);
});
check('incomplete CPU shadow never authorizes recovery',()=>{const job=clone(first);job.audit.complete=false;assert.throws(()=>admitResidentWithRecovery(job,fixture(first)),{code:'RESIDENT_AUDIT_MISMATCH'});});
console.log(JSON.stringify({allPass:true,checks,mockGPU:true,actualGPUExecuted:false,fullCPUShadowRequired:true,elapsedMs:performance.now()-began}));
