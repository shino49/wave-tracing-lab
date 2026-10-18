/** Complete double-precision shadow for the experimental GPU-resident walker.
 * The ordinary CPU algorithm is observed, never reimplemented or rounded to f32.
 * Every path and local FSD evaluation is retained, even without an image term.
 * Capacity exhaustion rejects the batch; there is no truncated-audit success.
 */
import {buildPLTBatch,PLT_LIMITS,PLT_PROGRESSIVE_LIMITS} from './plt-core.js';

export const RESIDENT_AUDIT_LIMITS=Object.freeze({
 maxPaths:PLT_PROGRESSIVE_LIMITS.maxBatchPaths,
 maxVerticesPerPath:72,
 maxEventsPerPath:PLT_LIMITS.maxDepth,
 maxContributionsPerPath:2*PLT_LIMITS.maxDepth,
 maxRandomWordsPerPath:64,
 maxRandomWords:PLT_PROGRESSIVE_LIMITS.maxBatchPaths*64,
 maxEdgeRecords:PLT_PROGRESSIVE_LIMITS.maxEdgeRecords,
});
export const RESIDENT_AUDIT_TOLERANCES=Object.freeze({absolute:2e-5,relative:.003,geometryAbsolute:2e-5,geometryRelative:3e-6});
export const residentHashWord=(hash,word)=>Math.imul((hash^word)>>>0,16777619)>>>0;
export const residentHashWords=words=>words.reduce((h,w)=>residentHashWord(h,w),2166136261);
export const residentPathKey=key=>`${key.wave?1:0}:${key.pixel}:${key.sample}`;
const copyVector=v=>v==null?null:[...v];
const copyBeam=b=>({...b,origin:copyVector(b.origin),direction:copyVector(b.direction)});
const capacityError=label=>Object.assign(new RangeError(`Resident audit capacity exceeded: ${label}; reduce tile/sample batch`),{code:'RESIDENT_AUDIT_CAPACITY'});
function capacityOptions(input={}){
 const out={...RESIDENT_AUDIT_LIMITS};
 for(const [key,value] of Object.entries(input)){
  if(!(key in out)||!Number.isSafeInteger(value)||value<1||value>out[key])throw new RangeError(`Invalid resident audit capacity: ${key}`);
  out[key]=value;
 }
 return out;
}

/** Returns a standard buildPLTBatch job with an additional mandatory job.audit.
 * rngStates have exactly the ordinary builder's mutation semantics. Callers
 * wanting transactional GPU admission should pass a cloned candidate state.
 */
export async function buildResidentReferenceBatch(scene,camera,options={},batch={}){
 const {auditCapacity,...buildOptions}=batch,capacity=capacityOptions(auditCapacity);
 if(buildOptions.tile&&buildOptions.tile.width*buildOptions.tile.height*(buildOptions.samplesPerPixel??1)>capacity.maxPaths)throw capacityError('paths');
 const paths=[],keys=new Set(),totals={paths:0,vertices:0,events:0,contributions:0,scatters:0,randomWords:0,edgeRecords:0};
 const audit={version:1,complete:false,flags:0,capacity,paths,totals};
 const observer={beginPath({wave,pixel,localPixel,sample,rngState,job}){
  if(paths.length>=capacity.maxPaths)throw capacityError('paths');
  const key={wave,pixel,sample},keyString=residentPathKey(key);
  if(keys.has(keyString))throw new Error(`Duplicate resident path key: ${keyString}`);
  keys.add(keyString);
  const firstEvent=job.events.length,firstTerm=job.terms.length;
  const path={key,keyString,localPixel,flags:0,camera:null,vertices:[],localEvents:[],contributions:[],scatters:[],events:[],termination:null,rng:{start:rngState,final:null,count:0,hash:2166136261,words:[]}};
  paths.push(path);totals.paths++;
  return {
   onRandomWord(word){
    if(path.rng.count>=capacity.maxRandomWordsPerPath||totals.randomWords>=capacity.maxRandomWords)throw capacityError('random words');
    path.rng.words.push(word);path.rng.hash=residentHashWord(path.rng.hash,word);path.rng.count++;totals.randomWords++;
   },
   onCamera(beam){path.camera=copyBeam(beam);},
   onTraversal({vertex,depth,nullSteps,beam}){
    if(path.vertices.length>=capacity.maxVerticesPerPath)throw capacityError('traversal vertices');
    path.vertices.push({type:vertex?(vertex.hit?'surface':'region'):'escape',depth,nullSteps,triangleId:vertex?.hit?.triangleId??null,
     center:copyVector(vertex?.center),distance:vertex?.distance??null,radius:vertex?.region?.radius??0,apertureCount:0,
     ballistic:!!vertex?.ballistic,origin:copyVector(beam.origin),direction:copyVector(beam.direction),radius0:beam.radius0,tanAngle:beam.tanAngle});
    totals.vertices++;
   },
   onVertex({aperture}){
    const vertex=path.vertices.at(-1);vertex.apertureCount=aperture?.edges.length??0;
   },
   onLocalEvent({id,depth,sampled,event,soft,hard,diagnostics}){
    if(path.localEvents.length>=capacity.maxEventsPerPath)throw capacityError('local events');
    if(totals.edgeRecords+event.edges.length>capacity.maxEdgeRecords)throw capacityError('edge records');
    const edgeIds=event.edges.map(e=>e.edgeId);
    path.localEvents.push({key:{...key,event:path.localEvents.length},id,localId:path.localEvents.length,depth,sampled,
     F:event.referenceF,directAmplitude:event.directAmplitude,edgeCount:edgeIds.length,edgeIds,edgeHash:residentHashWords(edgeIds),
     soft:copyVector(soft),hard:copyVector(hard),source:copyVector(event.source),destination:copyVector(event.destination),records:event.edges,operator:event,diagnostics:{warnings:[...(diagnostics?.warnings??[])],endpointExclusions:diagnostics?.endpointExclusions??0,invalidEdges:diagnostics?.invalidEdges??0},used:false});
    totals.events++;totals.edgeRecords+=event.edges.length;
   },
   onContribution({type,depth,weight,firstTerm:from,termEnd:to}){
    if(path.contributions.length>=capacity.maxContributionsPerPath)throw capacityError('contributions');
    // Same multiplication order as replayPLT, before any sample normalization.
    const terms=job.terms.slice(from,to).map(t=>({...t,eventIds:[...t.eventIds]}));
    const value=terms.reduce((sum,t)=>sum+t.eventIds.reduce((v,id)=>v*job.events[id].referenceF,t.weight),0);
    path.contributions.push({type,depth,weight,value,terms,firstTerm:from,termEnd:to});totals.contributions++;
   },
   onScatter(record){
    if(path.scatters.length>=PLT_LIMITS.maxDepth)throw capacityError('scatters');
    path.scatters.push(structuredClone(record));totals.scatters++;
   },
   onTermination({pathEvents,...termination}){
    path.termination={...termination};path.events=structuredClone(pathEvents);
   },
   onEnd({rngState:final}){
    if(!path.termination)throw new Error('Resident audit path ended without termination');
    path.rng.final=final;
    const used=new Set(job.terms.slice(firstTerm).flatMap(t=>t.eventIds));
    for(const event of path.localEvents)event.used=used.has(event.id);
    path.eventRange=[firstEvent,job.events.length];path.termRange=[firstTerm,job.terms.length];
    path.value=path.contributions.reduce((sum,c)=>sum+c.value,0);
   },
  };
 }};
 const job=await buildPLTBatch(scene,camera,options,{...buildOptions,audit:observer});
 if(paths.length!==job.width*job.height*job.samplesPerPixel||totals.events!==job.events.length)throw new Error('Resident audit did not capture every path and local event');
 audit.complete=true;audit.pixelIndices=new Uint32Array(job.pixelIndices);audit.reference=new Float64Array(job.reference);
 audit.sampleStart=job.sampleStart;audit.samplesPerPixel=job.samplesPerPixel;audit.tile={...job.tile};job.audit=audit;
 return job;
}

/** Strict, independent admission audit. The GPU decoder supplies canonical
 * objects, not a CPU event tape. Topology/RNG are exact; numerical power checks
 * retain the existing 2e-5 + .003*|reference| bound. Position tolerance is a
 * separate f32 geometry check and cannot excuse any topology disagreement.
 */
export function compareResidentAudit(reference,candidate,{geometryAbsolute=RESIDENT_AUDIT_TOLERANCES.geometryAbsolute,geometryRelative=RESIDENT_AUDIT_TOLERANCES.geometryRelative,maxMismatches=32}={}){
 const expected=reference?.audit??reference,actual=candidate?.audit??candidate;
 if(!expected?.complete||!Array.isArray(expected.paths))throw new TypeError('A complete CPU resident reference audit is required');
 if(!(Number.isFinite(geometryAbsolute)&&geometryAbsolute>=0&&geometryAbsolute<=RESIDENT_AUDIT_TOLERANCES.geometryAbsolute&&Number.isFinite(geometryRelative)&&geometryRelative>=0&&geometryRelative<=RESIDENT_AUDIT_TOLERANCES.geometryRelative))throw new RangeError('Invalid resident geometry tolerance');
 if(!Number.isSafeInteger(maxMismatches)||maxMismatches<1||maxMismatches>4096)throw new RangeError('Invalid resident mismatch report capacity');
 const failures=[],checked={paths:0,vertices:0,events:0,edgeRecords:0,contributions:0,scatters:0,randomWords:0,pixels:0},maxError={power:0,geometry:0};let mismatchCount=0;
 const fail=(field,cpu,gpu)=>{mismatchCount++;if(failures.length<maxMismatches)failures.push({field,cpu,gpu});};
 const exact=(a,b,label)=>{if(a!==b)fail(label,a,b);};
 const numeric=(a,b,label,geometry=false)=>{
  if(!Number.isFinite(a)||!Number.isFinite(b)){fail(label,a,b);return;}
  const delta=Math.abs(a-b),bound=geometry?geometryAbsolute+geometryRelative*Math.abs(a):RESIDENT_AUDIT_TOLERANCES.absolute+RESIDENT_AUDIT_TOLERANCES.relative*Math.abs(a);
  maxError[geometry?'geometry':'power']=Math.max(maxError[geometry?'geometry':'power'],delta);
  if(delta>bound)fail(label,a,b);
 };
 const vector=(a,b,label,geometry=true)=>{
  if(a===null){exact(a,b,label);return;}
  if(!b||typeof b.length!=='number'){fail(label,a,b);return;}
  exact(a.length,b.length,`${label}.length`);for(let i=0;i<a.length;i++)numeric(a[i],b[i],`${label}[${i}]`,geometry);
 };
 // World-space translations must not enlarge the admission tolerance. Anchor
 // positions to the exact CPU camera origin; directions and lengths use their
 // natural, untranslated scale. Candidate camera metadata is never trusted.
 const point=(a,b,label,origin)=>{
  if(a===null){exact(a,b,label);return;}
  if(!b||typeof b.length!=='number'){fail(label,a,b);return;}
  exact(a.length,b.length,`${label}.length`);for(let i=0;i<a.length;i++)numeric(a[i]-origin[i],b[i]-origin[i],`${label}[${i}]`,true);
 };
 const list=(a,b,label)=>{if(!Array.isArray(b)){fail(label,`array(${a.length})`,b);return [];}exact(a.length,b.length,`${label}.length`);return b.length>a.length?b.slice(0,a.length):b;};
 exact(true,actual?.complete,'complete');exact(0,actual?.flags,'flags');
 const gpuPaths=list(expected.paths,actual?.paths,'paths'),pathMap=new Map();
 for(const path of gpuPaths){
  if(!path?.key||typeof path.key.wave!=='boolean'||!Number.isSafeInteger(path.key.pixel)||!Number.isSafeInteger(path.key.sample)){fail('path.key','stable wave/pixel/sample',path?.key);continue;}
  const key=residentPathKey(path.key);if(pathMap.has(key))fail(`path[${key}]`,'one record','duplicate');else pathMap.set(key,path);
 }
 for(const cpu of expected.paths){
  const label=`path[${residentPathKey(cpu.key)}]`,gpu=pathMap.get(residentPathKey(cpu.key));checked.paths++;
  if(!gpu){fail(label,'present','missing');continue;}
  exact(0,gpu.flags,`${label}.flags`);const origin=cpu.camera.origin;
  for(const k of ['wave','pixel','sample'])exact(cpu.key[k],gpu.key[k],`${label}.key.${k}`);
  for(const k of ['start','final','count','hash'])exact(cpu.rng[k],gpu.rng?.[k],`${label}.rng.${k}`);
  const words=gpu.rng?.words;
  if(!words||typeof words.length!=='number')fail(`${label}.rng.words`,cpu.rng.words,words);
  else{exact(cpu.rng.words.length,words.length,`${label}.rng.words.length`);for(let i=0;i<cpu.rng.words.length;i++){checked.randomWords++;exact(cpu.rng.words[i],words[i],`${label}.rng.words[${i}]`);}}
  const vertices=list(cpu.vertices,gpu.vertices,`${label}.vertices`);
  for(const [i,a] of cpu.vertices.entries()){
   checked.vertices++;const b=vertices[i],v=`${label}.vertices[${i}]`;if(!b){fail(v,'present','missing');continue;}
   for(const k of ['type','depth','triangleId','apertureCount'])exact(a[k],b[k],`${v}.${k}`);
   for(const k of ['center','origin'])point(a[k],b[k],`${v}.${k}`,origin);vector(a.direction,b.direction,`${v}.direction`);
   if(a.distance===null)exact(a.distance,b.distance,`${v}.distance`);else numeric(a.distance,b.distance,`${v}.distance`,true);
   for(const k of ['radius','tanAngle'])numeric(a[k],b[k],`${v}.${k}`,true);
  }
  const events=list(cpu.localEvents,gpu.localEvents,`${label}.localEvents`);
  for(const [i,a] of cpu.localEvents.entries()){
   checked.events++;const b=events[i],e=`${label}.localEvents[${i}]`;if(!b){fail(e,'present','missing');continue;}
   for(const k of ['depth','sampled','directAmplitude','edgeCount','edgeHash'])exact(a[k],b[k],`${e}.${k}`);
   if(!(b.F>=0))fail(`${e}.F.nonnegative`,true,false);numeric(a.F,b.F,`${e}.F`);for(const k of ['soft','hard'])vector(a[k],b[k],`${e}.${k}`,false);
   point(a.destination,b.destination,`${e}.destination`,origin);point(a.source,b.source,`${e}.source`,origin);
   const records=list(a.records,b.records,`${e}.records`);
   for(const [j,edge] of a.records.entries()){
    checked.edgeRecords++;const actualEdge=records[j],r=`${e}.records[${j}]`;if(!actualEdge){fail(r,'present','missing');continue;}
    exact(edge.edgeId,actualEdge.edgeId,`${r}.edgeId`);point(edge.point,actualEdge.point,`${r}.point`,origin);
    for(const k of ['ri','ro','sinBeta','phiI','phiO','n','phaseExcess','L','fermatResidual'])numeric(edge[k],actualEdge[k],`${r}.${k}`,true);
   }
   // Records above require exact ordered IDs. The redundant compact ID array,
   // when supplied, must also agree; its hash never replaces those full IDs.
   if(b.edgeIds!==undefined){exact(a.edgeIds.length,b.edgeIds.length,`${e}.edgeIds.length`);for(let k=0;k<a.edgeIds.length;k++)exact(a.edgeIds[k],b.edgeIds[k],`${e}.edgeIds[${k}]`);}
  }
  const scatters=list(cpu.scatters,gpu.scatters,`${label}.scatters`);
  for(const [i,a] of cpu.scatters.entries()){
   checked.scatters++;const b=scatters[i],s=`${label}.scatters[${i}]`;if(!b){fail(s,'present','missing');continue;}
   for(const k of ['type','depth'])exact(a[k],b[k],`${s}.${k}`);
   if(a.type==='fsd'){
    if(!b.sample){fail(`${s}.sample`,'present','missing');continue;}
    for(const k of ['valid','edgeId','isDirect'])exact(a.sample[k],b.sample[k],`${s}.sample.${k}`);
    vector(a.sample.wo,b.sample.wo,`${s}.sample.wo`);point(a.sample.point,b.sample.point,`${s}.sample.point`,origin);
    numeric(a.sample.weight,b.sample.weight,`${s}.sample.weight`);numeric(a.sample.pdf.value,b.sample.pdf?.value,`${s}.sample.pdf.value`);
    if(!(b.sample.weight>=0&&b.sample.pdf?.value>=0))fail(`${s}.sample.nonnegative`,true,false);
   }else{
    vector(a.direction,b.direction,`${s}.direction`);for(const k of ['pdf','weight'])numeric(a[k],b[k],`${s}.${k}`);
    if(!(b.pdf>=0&&b.weight>=0))fail(`${s}.nonnegative`,true,false);
   }
  }
  const contributions=list(cpu.contributions,gpu.contributions,`${label}.contributions`);
  for(const [i,a] of cpu.contributions.entries()){
   checked.contributions++;const b=contributions[i],c=`${label}.contributions[${i}]`;if(!b){fail(c,'present','missing');continue;}
   for(const k of ['type','depth'])exact(a[k],b[k],`${c}.${k}`);if(!(b.value>=0&&b.weight>=0))fail(`${c}.nonnegative`,true,false);numeric(a.value,b.value,`${c}.value`);numeric(a.weight,b.weight,`${c}.weight`);
  }
  for(const k of ['reason','depth'])exact(cpu.termination[k],gpu.termination?.[k],`${label}.termination.${k}`);
 }
 const cpuPixels=expected.pixelIndices,gpuPixels=actual?.pixelIndices,cpuImage=expected.reference,gpuImage=actual?.reference;
 if(!gpuPixels||!gpuImage||typeof gpuPixels.length!=='number'||typeof gpuImage.length!=='number')fail('pixels','all global pixel IDs and batch mean values',null);
 else{
  exact(cpuPixels.length,gpuPixels.length,'pixelIndices.length');exact(cpuImage.length,gpuImage.length,'reference.length');
  const pixelMap=new Map();for(let i=0;i<Math.min(gpuPixels.length,cpuPixels.length);i++){const id=gpuPixels[i];if(pixelMap.has(id))fail(`pixel[${id}]`,'one record','duplicate');else pixelMap.set(id,gpuImage[i]);}
  for(let i=0;i<cpuPixels.length;i++){checked.pixels++;const value=pixelMap.get(cpuPixels[i]);if(!(value>=0))fail(`pixel[${cpuPixels[i]}].nonnegative`,true,false);numeric(cpuImage[i],value,`pixel[${cpuPixels[i]}]`);}
 }
 return {ok:mismatchCount===0,passed:mismatchCount===0,mismatchCount,failures,checked,maxError,tolerance:{...RESIDENT_AUDIT_TOLERANCES,geometryAbsolute,geometryRelative},fullCPUShadow:true};
}

export function assertResidentAudit(reference,candidate,options){
 const result=compareResidentAudit(reference,candidate,options);
 if(!result.ok)throw Object.assign(new Error(`Resident GPU audit rejected: ${result.mismatchCount} mismatch(es); ${result.failures[0]?.field??'unknown'}`),{code:'RESIDENT_AUDIT_MISMATCH',audit:result});
 return result;
}
