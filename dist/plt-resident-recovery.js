/** Experimental, full-shadow-only recovery of bounded numerical geometry drift.
 * This is deliberately separate from assertResidentAudit: strict admission is
 * unchanged. An original complete audit must prove exact topology/RNG and pass
 * every power/image check before any CPU path can replace a GPU path.
 */
import {compareResidentAudit,RESIDENT_AUDIT_TOLERANCES,residentPathKey} from './plt-resident-reference.js';

// This is an eligibility ceiling for CPU replacement, not an admission bound.
// Every returned path still passes the original, unchanged strict audit.
export const RESIDENT_RECOVERY_LIMITS=Object.freeze({maxMismatches:4096,maxGeometryBoundMultiple:16});
const now=()=>globalThis.performance?.now?.()??Date.now();
const own=(value,key)=>Object.prototype.hasOwnProperty.call(value,key);
const nonnegative=value=>Number.isFinite(value)&&value>=0;
const sum=values=>values.reduce((total,value)=>total+value,0);
function sumF32(values){
 let total=0,correction=0;
 for(const value of values){const adjusted=Math.fround(Math.fround(value)-correction),next=Math.fround(total+adjusted);correction=Math.fround(Math.fround(next-total)-adjusted);total=next;}
 return total;
}
const powerClose=(cpu,gpu)=>nonnegative(cpu)&&nonnegative(gpu)&&Math.abs(cpu-gpu)<=RESIDENT_AUDIT_TOLERANCES.absolute+RESIDENT_AUDIT_TOLERANCES.relative*Math.abs(cpu);
function rejected(message,audit,field){
 const error=new Error(`Resident recovery rejected: ${message}`);error.code='RESIDENT_AUDIT_MISMATCH';error.audit=audit;
 if(field)error.recovery={eligible:false,field,reason:message};
 return error;
}

/** Validate sums BEFORE replacement. Otherwise a corrupt sampleValue or image
 * could disappear when the host reconstructs a recovered pixel. Exact decoded
 * f32 Kahan and ordinary double sums are supported; the broad power tolerance
 * is never used to excuse disagreement with the candidate's own audit totals.
 */
function checkAssembly(expected,candidate,audit){
 const actual=candidate.audit,fail=(message,field)=>{throw rejected(message,audit,field);};
 if(candidate.image!==actual.reference)fail('image is not the audited pixel buffer','image');
 const spp=expected.samplesPerPixel;
 if(!Number.isSafeInteger(spp)||spp<1||actual.samplesPerPixel!==spp)fail('missing or inconsistent sample count','samplesPerPixel');
 if(!Number.isSafeInteger(expected.sampleStart)||expected.sampleStart<0)fail('missing CPU sample prefix','sampleStart');
 if(actual.sampleStart!==undefined&&actual.sampleStart!==expected.sampleStart)fail('wrong candidate sample prefix','sampleStart');
 const cpuMap=new Map(expected.paths.map(path=>[residentPathKey(path.key),path])),byPixel=new Map();
 if(actual.pixelIndices.length!==expected.pixelIndices.length)fail('incomplete pixel indices','pixelIndices');
 for(let i=0;i<expected.pixelIndices.length;i++){
  const pixel=expected.pixelIndices[i];
  if(actual.pixelIndices[i]!==pixel)fail('audited pixel order differs from the committed tile','pixelIndices');
  if(candidate.pixelIndices!==undefined&&candidate.pixelIndices[i]!==pixel)fail('candidate pixel order differs from its audit','pixelIndices');
  byPixel.set(pixel,[]);
 }
 if(candidate.pixelIndices!==undefined&&candidate.pixelIndices.length!==expected.pixelIndices.length)fail('incomplete candidate pixel indices','pixelIndices');
 for(const path of actual.paths){
  const key=residentPathKey(path.key),cpu=cpuMap.get(key),field=`path[${key}]`,hasSample=own(path,'sampleValue'),hasValue=own(path,'value');
  if(!cpu||!byPixel.has(path.key.pixel)||path.key.sample<expected.sampleStart||path.key.sample>=expected.sampleStart+spp)fail('invalid path ownership',field);
  if(!hasSample&&!hasValue)fail('missing complete path value',`${field}.sampleValue`);
  const value=hasSample?path.sampleValue:path.value,terms=path.contributions.map(item=>item.value);
  if(!nonnegative(value)||terms.some(v=>!nonnegative(v)))fail('nonfinite or negative path value',`${field}.sampleValue`);
  if(hasSample&&hasValue&&path.sampleValue!==path.value)fail('conflicting path values',`${field}.sampleValue`);
  if(value!==sum(terms)&&value!==sumF32(terms))fail('path value is not its contribution sum',`${field}.sampleValue`);
  if(!powerClose(cpu.value,value))fail('path power mismatch',`${field}.sampleValue`);
  byPixel.get(path.key.pixel).push({sample:path.key.sample,value,path});
 }
 for(let i=0;i<expected.pixelIndices.length;i++){
  const pixel=expected.pixelIndices[i],paths=byPixel.get(pixel).sort((a,b)=>a.sample-b.sample),field=`pixel[${pixel}]`;
  if(paths.length!==spp||paths.some((path,j)=>path.sample!==expected.sampleStart+j))fail('missing or duplicate pixel sample',field);
  const values=paths.map(path=>path.value),value=actual.reference[i];
  if(value!==sum(values)/spp&&value!==Math.fround(sumF32(values)/spp))fail('image is not its audited sample sum',field);
  if(candidate.finalRngStates!==undefined&&(candidate.finalRngStates.length!==expected.pixelIndices.length||candidate.finalRngStates[i]!==paths.at(-1).path.rng.final))fail('final RNG buffer differs from audited path','finalRngStates');
 }
 return byPixel;
}

/** Discover recoverable failures through typed geometry traversal, not regexes
 * over failure labels. The evidence contains the exact anchored operands used
 * by compareResidentAudit. Missing/nonfinite values and shape mismatches never
 * produce eligible evidence and therefore cannot be replaced away.
 */
function geometryEvidence(expected,actual,normalized){
 const validPaths=value=>Array.isArray(value?.paths)?value.paths.filter(path=>path?.key&&typeof path.key.wave==='boolean'&&Number.isSafeInteger(path.key.pixel)&&Number.isSafeInteger(path.key.sample)):[];
 const actualMap=new Map(validPaths(actual).map(path=>[residentPathKey(path.key),path])),normalizedMap=new Map(validPaths(normalized).map(path=>[residentPathKey(path.key),path])),evidence=new Map();
 const scalar=(cpu,gpu,field,key,set,{origin=0,positive=false}={})=>{
  if(!Number.isFinite(cpu)||!Number.isFinite(gpu)||positive&&gpu<0)return;
  const a=cpu-origin,b=gpu-origin,delta=Math.abs(a-b),bound=RESIDENT_AUDIT_TOLERANCES.geometryAbsolute+RESIDENT_AUDIT_TOLERANCES.geometryRelative*Math.abs(a);
  if(!Number.isFinite(delta)||delta<=bound||delta>bound*RESIDENT_RECOVERY_LIMITS.maxGeometryBoundMultiple)return;
  evidence.set(field,{field,cpu:a,gpu:b,key,delta,bound,reason:'finite-geometry-drift'});set(cpu);
 };
 const vector=(a,b,out,field,key,origin)=>{
  if(a===null||!a||!b||!out||a.length!==3||b.length!==3||out.length!==3)return;
  for(let i=0;i<3;i++)scalar(a[i],b[i],`${field}[${i}]`,key,value=>{out[i]=value;},{origin:origin?.[i]??0});
 };
 for(const cpu of expected.paths){
  const key=residentPathKey(cpu.key),gpu=actualMap.get(key),out=normalizedMap.get(key),label=`path[${key}]`,origin=cpu.camera.origin;
  if(!gpu||!out)continue;
  for(let i=0;i<cpu.vertices.length;i++){
   const a=cpu.vertices[i],b=gpu.vertices?.[i],c=out.vertices?.[i],field=`${label}.vertices[${i}]`;if(!b||!c)continue;
   for(const name of ['center','origin'])vector(a[name],b[name],c[name],`${field}.${name}`,key,origin);
   vector(a.direction,b.direction,c.direction,`${field}.direction`,key);
   for(const name of ['distance','radius','tanAngle'])scalar(a[name],b[name],`${field}.${name}`,key,value=>{c[name]=value;},{positive:true});
  }
  for(let i=0;i<cpu.localEvents.length;i++){
   const a=cpu.localEvents[i],b=gpu.localEvents?.[i],c=out.localEvents?.[i],field=`${label}.localEvents[${i}]`;if(!b||!c)continue;
   for(const name of ['source','destination'])vector(a[name],b[name],c[name],`${field}.${name}`,key,origin);
   for(let j=0;j<a.records.length;j++){
    const ar=a.records[j],br=b.records?.[j],cr=c.records?.[j],record=`${field}.records[${j}]`;if(!br||!cr)continue;
    vector(ar.point,br.point,cr.point,`${record}.point`,key,origin);
    for(const name of ['ri','ro','sinBeta','phiI','phiO','n','phaseExcess','L','fermatResidual'])scalar(ar[name],br[name],`${record}.${name}`,key,value=>{cr[name]=value;},{positive:['ri','ro','sinBeta','n','L'].includes(name)});
   }
  }
  for(let i=0;i<cpu.scatters.length;i++){
   const a=cpu.scatters[i],b=gpu.scatters?.[i],c=out.scatters?.[i],field=`${label}.scatters[${i}]`;if(!b||!c)continue;
   if(a.type==='fsd'){
    if(!b.sample||!c.sample)continue;
    vector(a.sample.wo,b.sample.wo,c.sample.wo,`${field}.sample.wo`,key);vector(a.sample.point,b.sample.point,c.sample.point,`${field}.sample.point`,key,origin);
   }else vector(a.direction,b.direction,c.direction,`${field}.direction`,key);
  }
 }
 return evidence;
}

/** Return a new admitted candidate. Recovery metadata distinguishes replacement
 * from original strict admission. A CPU full shadow is compulsory in both cases.
 * Only the paths with proven finite geometry drift are replaced; all remaining
 * decoded samples survive unchanged and each sample enters the image once.
 */
export function admitResidentWithRecovery(referenceJob,candidate){
 const began=now(),expected=referenceJob?.audit??referenceJob;
 let initial;
 try{initial=compareResidentAudit(expected,candidate?.audit,{maxMismatches:RESIDENT_RECOVERY_LIMITS.maxMismatches});}
 catch(error){throw rejected(`full audit could not complete: ${error.message}`,null);}
 const initialAuditMs=now()-began;
 if(initial.mismatchCount!==initial.failures.length)throw rejected('truncated full-audit failure report',initial);
 if(!candidate?.audit)throw rejected('missing complete candidate audit',initial);
 // Every original failure must match numerical evidence from actual geometry
 // operands. No unchecked array shape or failure label can become evidence.
 const prepareStart=now(),normalized=structuredClone(candidate.audit),evidence=geometryEvidence(expected,candidate.audit,normalized);
 for(const failure of initial.failures){
  const proof=evidence.get(failure.field);
  if(!proof||proof.cpu!==failure.cpu||proof.gpu!==failure.gpu)throw rejected('failure is not bounded, finite geometry drift',initial,failure.field);
 }
 checkAssembly(expected,candidate,initial);
 const normalizedAudit=compareResidentAudit(expected,normalized,{maxMismatches:RESIDENT_RECOVERY_LIMITS.maxMismatches});
 if(!normalizedAudit.ok)throw rejected('nongeometry disagreement remains after geometry normalization',normalizedAudit);
 const keys=[...new Set([...evidence.values()].map(item=>item.key))],keySet=new Set(keys),cpuMap=new Map(expected.paths.map(path=>[residentPathKey(path.key),path]));
 const recovered=structuredClone(candidate.audit);
 for(let i=0;i<recovered.paths.length;i++){
  const key=residentPathKey(recovered.paths[i].key);
  if(keySet.has(key)){const path=structuredClone(cpuMap.get(key));path.sampleValue=path.value;recovered.paths[i]=path;}
 }
 if(keys.length){
  const byPixel=new Map();for(const path of recovered.paths){const paths=byPixel.get(path.key.pixel)??[];paths.push(path);byPixel.set(path.key.pixel,paths);}
  recovered.reference=new Float64Array(recovered.pixelIndices.length);
  for(let i=0;i<recovered.pixelIndices.length;i++){const paths=byPixel.get(recovered.pixelIndices[i]).sort((a,b)=>a.key.sample-b.key.sample);recovered.reference[i]=sum(paths.map(path=>own(path,'sampleValue')?path.sampleValue:path.value))/expected.samplesPerPixel;}
 }
 const replacementMs=now()-prepareStart,finalStart=now(),validation=compareResidentAudit(expected,recovered,{maxMismatches:RESIDENT_RECOVERY_LIMITS.maxMismatches});
 if(!validation.ok)throw rejected('reconstructed candidate failed final complete audit',validation);
 const result={...candidate,candidate:true,admitted:true,audit:recovered,image:recovered.reference,pixelIndices:recovered.pixelIndices,validation,
  recovery:{...candidate.recovery,fullCPUShadow:true,mode:candidate.recovery?.mode??'strict-shadow-geometry-replacement',paths:(candidate.recovery?.paths??0)+keys.length,
   keys:[...(candidate.recovery?.keys??[]),...keys.map(key=>({key:{...cpuMap.get(key).key},flags:0,reasons:['finite-geometry-drift']}))],
   numericGeometry:{count:keys.length,keys,reasons:[...evidence.values()],initialMismatchCount:initial.mismatchCount,initialAudit:initial,normalizedAudit,finalAudit:validation}},
  timings:{...candidate.timings,residentRecoveryInitialAuditMs:initialAuditMs,residentRecoveryReplacementMs:replacementMs,residentRecoveryFinalAuditMs:now()-finalStart,residentRecoveryTotalMs:now()-began,fullCPUShadow:true}};
 checkAssembly(expected,result,validation);
 result.timings.residentRecoveryTotalMs=now()-began;
 return result;
}
