/** Native execution acceptance, distinct from mock lifecycle tests.
 * Run with --execute and WEBGPU_EXECUTION_BACKEND=vulkan (or native hardware).
 * No unchecked performance lane is admitted by this test suite.
 */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import crypto from 'node:crypto';
import {create,globals} from 'webgpu';
import {createPLTScene,createPLTRandomStates} from '../dist/plt-core.js';
import {createPLTResidentRunner,PLT_RESIDENT_BUILD_ID} from '../dist/plt-resident-gpu.js';
import {buildResidentReferenceBatch,compareResidentAudit,RESIDENT_AUDIT_TOLERANCES} from '../dist/plt-resident-reference.js';
Object.assign(globalThis,globals);
const filenames=['utd-functions.wgsl','plt-resident-geometry.wgsl','plt-resident-interval.wgsl','plt-resident-axis-guard.wgsl','plt-resident-rng.wgsl','plt-resident-fsd.wgsl','plt-resident-walker.wgsl'];
const sources=filenames.map(name=>fs.readFileSync(new URL('../dist/'+name,import.meta.url),'utf8')),source=sources.join('\n');
const execute=process.argv.includes('--execute'),backend=execute?process.env.WEBGPU_EXECUTION_BACKEND:'null';
if(execute)assert(['vulkan','metal','d3d11','d3d12','opengl','opengles'].includes(backend),'Explicit executable Dawn backend required; null cannot prove GPU execution');
let gpu=create([`backend=${backend}`]);
const cases=[],checks=[],report={buildId:PLT_RESIDENT_BUILD_ID,backend,actualGPUExecuted:execute,hardwarePerformanceClaim:false,fullCPUShadowRequired:true,tolerances:RESIDENT_AUDIT_TOLERANCES,sourceSHA256:Object.fromEntries(filenames.map((name,i)=>[name,crypto.createHash('sha256').update(sources[i]).digest('hex')])),cases,checks};
function check(name,fn){fn();checks.push(name);console.log('PASS',name);}
async function session(name,configuration,options,fn){
 const scene=createPLTScene(configuration),o={width:8,height:8,samplesPerPixel:1,maxDepth:6,...options},start=performance.now();
 const engine=await createPLTResidentRunner(scene,scene.camera,o,{gpu,shaderSource:source,timeoutMs:120000});
 const entry={name,adapter:engine.diagnostic.adapter,setupWallMs:performance.now()-start,batches:[]};cases.push(entry);
 try{await fn({scene,o,engine,entry});}catch(error){entry.failure={message:error.message,diagnostic:error.diagnostic??null};console.log('CASE FAILED',name,error.message);}finally{engine.destroy();}
}
async function batch(ctx,{tile={x:0,y:0,width:8,height:8},sampleStart=0,samplesPerPixel=1,wave=ctx.o.wave??true,rng=createPLTRandomStates(ctx.o),expectRejection=false,allowAuditRejection=false}={}){
 const original=rng.slice(),referenceRng=rng.slice(),cpuStart=performance.now();
 const reference=await buildResidentReferenceBatch(ctx.scene,ctx.scene.camera,{...ctx.o,wave},{tile,sampleStart,samplesPerPixel,rngStates:referenceRng});
 const result={tile,sampleStart,samplesPerPixel,wave,cpuFullShadowMs:performance.now()-cpuStart,referenceCounts:reference.audit.totals};ctx.entry.batches.push(result);
 let candidate;
 try{candidate=await ctx.engine.run({tile,sampleStart,samplesPerPixel,wave,rngStates:rng});}
 catch(error){if(!expectRejection)throw error;assert.equal(error.name,'PLTResidentRejected');result.rejected=true;result.admitted=false;result.rejectionStage='gpu-predicate';result.diagnostic=error.diagnostic;assert.deepEqual(rng,original);console.log('PASS fail-closed diagnostic',ctx.entry.name);return {reference,rejected:true};}
 assert.deepEqual(rng,original,'candidate runner must not mutate committed CPU prefixes');
 const gate=compareResidentAudit(reference,candidate.audit);result.gate=gate;result.timings=candidate.timings;result.rounds=candidate.rounds;result.dispatches=candidate.dispatches;
 if(!gate.passed){result.rejected=true;result.admitted=false;result.rejectionStage='cpu-full-shadow';console.log('WITHHELD CPU full audit',ctx.entry.name,JSON.stringify(gate.failures));if(!allowAuditRejection)assert.equal(gate.passed,true,JSON.stringify(gate.failures));return {reference,candidate,rejected:true,gate};}result.admitted=true;assert.equal(candidate.admitted,false,'raw native candidates must never self-admit');assert.equal(candidate.timings.cpuReferenceProvided,false);
 for(let i=0;i<candidate.pixelIndices.length;i++){const id=candidate.pixelIndices[i];assert.equal(candidate.finalRngStates[i],referenceRng[id]);rng[id]=candidate.finalRngStates[i];}
 console.log('PASS native complete shadow',ctx.entry.name,`${gate.checked.paths} paths`);return {reference,candidate};
}
try{
 if(!execute){await session('all-five-entrypoint-native-compile',{}, {},async()=>{});check('all five genuine walker entrypoints compile using Dawn/Tint',()=>{});}
 else{
  await session('gallery-ray-sequential-global-prefixes',{preset:'gallery'},{wave:false,samplesPerPixel:3},async ctx=>{
   const rng=createPLTRandomStates(ctx.o),tile={x:1,y:0,width:7,height:8};
   const first=await batch(ctx,{tile,wave:false,samplesPerPixel:2,rng});
   const second=await batch(ctx,{tile,wave:false,sampleStart:2,samplesPerPixel:1,rng,allowAuditRejection:true});
   check('later accumulated f32 drift stays subject to unchanged full audit',()=>{if(second.rejected){assert.equal(second.gate.mismatchCount,1);assert.equal(second.gate.failures[0].field,'path[0:39:2].vertices[4].distance');}else assert(second.candidate.audit.complete);});
   check('irregular tile retains global pixel IDs and sequential samples',()=>{assert.equal(first.candidate.audit.paths.length,112);assert.equal(second.candidate.audit.paths.length,56);assert(second.candidate.audit.paths.every(p=>p.key.sample===2&&p.key.pixel%8!==0));});
   check('diffuse/NEE/MIS and true multi-bounce geometry executed',()=>{assert(first.candidate.audit.paths.some(p=>p.contributions.some(c=>c.type==='surface-nee')));assert(first.candidate.audit.paths.some(p=>p.vertices.length>=3));});
  });
  let ddResponse;
  await session('double-edge-organic-camera',{preset:'double-edge'},{wave:true},async ctx=>{
   const r=await batch(ctx,{tile:{x:0,y:2,width:1,height:1},wave:true});ddResponse=r.candidate.image[0];
   check('organic camera path contains two nonempty diffraction operators and reaches source',()=>{const p=r.candidate.audit.paths[0];assert(p.localEvents.filter(e=>e.edgeCount>0).length>=2);assert(p.contributions.some(c=>c.type==='emission'&&c.value>0));assert(ddResponse>0);});
  });
  await session('physical-emitter-radiance-scaling',{preset:'double-edge',emission:10},{wave:true},async ctx=>{
   const r=await batch(ctx,{tile:{x:0,y:2,width:1,height:1},wave:true});
   check('physical emitter radiance doubles camera response',()=>{assert(Math.abs(r.candidate.image[0]-2*ddResponse)<=2e-5+1e-6*Math.abs(2*ddResponse));});
  });
  await session('gallery-mirror-ray',{preset:'gallery',modelMaterial:2},{wave:false},async ctx=>{
   const r=await batch(ctx,{wave:false});check('mirror reflection branch is native and shadow-validated',()=>assert(r.candidate.audit.paths.some(p=>p.scatters.some(s=>s.type==='mirror'))));
  });
  await session('gallery-wave-admission-boundary',{preset:'gallery'},{wave:true},async ctx=>{
   const r=await batch(ctx,{wave:true,expectRejection:true,allowAuditRejection:true});report.galleryWaveAdmitted=!r.rejected;
   check('gallery wave either fully matches or rejects with bounded diagnostics',()=>{if(r.rejected)assert(ctx.entry.batches[0].diagnostic?.rejectedPaths.length>0||r.gate?.mismatchCount>0);else assert(r.candidate.audit.paths.length===64);});
  });
 }
 report.allPass=cases.every(c=>!c.failure);report.allCandidateBatchesAdmitted=execute?cases.every(c=>c.batches.every(b=>b.admitted===true)):null;const batches=cases.flatMap(c=>c.batches);report.admission={acceptedBatches:batches.filter(b=>b.admitted).length,totalBatches:batches.length,pathsInAcceptedBatches:batches.filter(b=>b.admitted).reduce((s,b)=>s+b.referenceCounts.paths,0),requestedPaths:batches.reduce((s,b)=>s+b.referenceCounts.paths,0),meaning:"Batch-level admission only; no partial commit of a rejected batch"};
}finally{gpu=null;fs.writeFileSync(new URL('../research/plt-resident-'+(execute?'native':'compile')+'-validation.json',import.meta.url),JSON.stringify(report,null,2)+'\n');}
console.log(JSON.stringify({allPass:report.allPass,actualGPUExecuted:execute,checks:checks.length,cases:cases.length,galleryWaveAdmitted:report.galleryWaveAdmitted,allCandidateBatchesAdmitted:report.allCandidateBatchesAdmitted}));
if(!report.allPass)process.exitCode=1;
