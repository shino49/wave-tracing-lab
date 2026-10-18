// Failure-only observational rerun. Never used by image computation/admission.
import {coefficient,stableCotTransition,erfcxScaled} from './utd-math.js';
export const PLT_TRACE_SCHEMA='plt-event-failure-trace-v1';
export const PLT_TRACE_MAX_EVENTS=8,PLT_TRACE_MAX_EDGES=2,PLT_TRACE_EVENT_FLOATS=208;
// Output: 8 reconstructed event lanes, 2 x 96 edge lanes, 8 constant probes.
export const PLT_FAILURE_TRACE_WGSL=`
@group(0) @binding(7) var<storage,read> pltTraceIds:array<u32>;
@group(0) @binding(8) var<storage,read_write> pltTrace:array<vec4f>;
fn plt_trace_cot(base:u32,beta:f32,n:f32,kL:f32,side:f32){
 let nearest=floor((beta+side*UTD_PI)/(2.0*UTD_PI*n)+0.5);
 let delta=beta-(2.0*UTD_PI*n*nearest-side*UTD_PI);
 let t=delta*0.5; let h=delta/(2.0*n);
 let st=utd_sinc(t); let sh=utd_sinc(h); let ch=utd_cos_small(h);
 let rootKL=sqrt(kL); let u=sqrt(2.0)*rootKL*abs(t*st); let inv=1.0/(1.0+u);
 let ratio=sign(delta)*n*st*ch/sh;
 pltTrace[base]=vec4f(nearest,delta,t,h);
 pltTrace[base+1u]=vec4f(st,sh,ch,ratio);
 pltTrace[base+2u]=vec4f(rootKL,u,inv,u*inv);
 pltTrace[base+3u]=vec4f(utd_erfcx_scaled(u*inv),utd_cot_transition(beta,n,kL,side));
}
@compute @workgroup_size(1)
fn plt_failure_trace(@builtin(global_invocation_id) id:vec3u){
 let index=pltTraceIds[id.x]; let event=events[index]; let base=id.x*52u;
 var soft=vec2f(event.field.x,0.0); var hard=soft;
 for(var j=0u;j<event.edgeRange.y;j++){
  let edge=edges[event.edgeRange.x+j]; let a=edge.angles; let b=edge.distances;
  let phase=utd_cis(-a.x*b.z)/sqrt(b.x);
  soft+=utd_cmul(utd_coefficient(a.x,a.y,a.z,a.w,b.w,b.y,-1.0),phase);
  hard+=utd_cmul(utd_coefficient(a.x,a.y,a.z,a.w,b.w,b.y,1.0),phase);
  if(j<2u){
   let at=base+2u+j*24u; let cis=utd_cis(-a.x*b.z);
   let ds=utd_coefficient(a.x,a.y,a.z,a.w,b.w,b.y,-1.0);
   let dh=utd_coefficient(a.x,a.y,a.z,a.w,b.w,b.y,1.0);
   pltTrace[at]=a; pltTrace[at+1u]=b;
   pltTrace[at+2u]=vec4f(cis,phase);
   pltTrace[at+3u]=vec4f(ds,dh);
   pltTrace[at+4u]=vec4f(utd_cmul(ds,phase),utd_cmul(dh,phase));
   pltTrace[at+5u]=vec4f(dot(cis,cis),sqrt(b.x),1.0/sqrt(b.x),2.0*a.y*sqrt(2.0*UTD_PI*a.x)*b.y);
   plt_trace_cot(at+6u,a.w-a.z,a.y,a.x*b.w,1.0);
   plt_trace_cot(at+10u,a.w-a.z,a.y,a.x*b.w,-1.0);
   plt_trace_cot(at+14u,a.w+a.z,a.y,a.x*b.w,1.0);
   plt_trace_cot(at+18u,a.w+a.z,a.y,a.x*b.w,-1.0);
   let theta=-a.x*b.z;
   pltTrace[at+22u]=vec4f(sin(theta),cos(theta),sqrt(a.x),sqrt(a.x*b.w));
   pltTrace[at+23u]=vec4f(fma(1.0000001192092896,a.x,-a.x),1.0000001192092896*a.x-a.x,exp(-0.125),sqrt(2.0));
  }
 }
 let sp=dot(soft,soft);let hp=dot(hard,hard);var selected=0.5*(sp+hp);
 if(event.edgeRange.z==0u){selected=sp;}if(event.edgeRange.z==1u){selected=hp;}
 pltTrace[base]=vec4f(soft,hard);pltTrace[base+1u]=vec4f(selected,sp,hp,f32(event.edgeRange.y));
 pltTrace[base+50u]=vec4f(sin(0.125),cos(0.125),sqrt(2.0),1.0/sqrt(2.0));
 pltTrace[base+51u]=vec4f(utd_erfcx_scaled(0.5),utd_erfcx_scaled(0.9860957183805704));
}
`;
const safe=v=>Number.isFinite(v)?v:Number.isNaN(v)?'NaN':v>0?'+Infinity':'-Infinity';
const numbers=v=>Array.from(v,safe),words=v=>Array.from(new Uint32Array(v.buffer,v.byteOffset,v.length));
const mul=(a,b)=>[a[0]*b[0]-a[1]*b[1],a[0]*b[1]+a[1]*b[0]];
const norm=a=>a[0]*a[0]+a[1]*a[1];
const sinc=x=>x===0?1:Math.sin(x)/x;
function cotReference(beta,n,kL,side){
 const nearest=Math.floor((beta+side*Math.PI)/(2*Math.PI*n)+.5),delta=beta-(2*Math.PI*n*nearest-side*Math.PI),t=delta/2,h=delta/(2*n),st=sinc(t),sh=sinc(h),ch=Math.cos(h),ratio=Math.sign(delta)*n*st*ch/sh,rootKL=Math.sqrt(kL),u=Math.SQRT2*rootKL*Math.abs(t*st),inv=1/(1+u);
 return [nearest,delta,t,h,st,sh,ch,ratio,rootKL,u,inv,u*inv,...erfcxScaled(u*inv),...stableCotTransition(beta,n,kL,side,2e-6)];
}
function edgeReference(f){
 const [k,n,phiI,phiO,ro,sinBeta,phaseExcess,L]=f,theta=-k*phaseExcess,cis=[Math.cos(theta),Math.sin(theta)],phase=cis.map(x=>x/Math.sqrt(ro)),ds=coefficient(k,n,phiI,phiO,L,sinBeta,'soft',2e-6),dh=coefficient(k,n,phiI,phiO,L,sinBeta,'hard',2e-6);
 return {cis,phase,ds,dh,values:[...f,...cis,...phase,...ds,...dh,...mul(ds,phase),...mul(dh,phase),norm(cis),Math.sqrt(ro),1/Math.sqrt(ro),2*n*Math.sqrt(2*Math.PI*k)*sinBeta,
 ...cotReference(phiO-phiI,n,k*L,1),...cotReference(phiO-phiI,n,k*L,-1),...cotReference(phiO+phiI,n,k*L,1),...cotReference(phiO+phiI,n,k*L,-1),
 Math.sin(theta),Math.cos(theta),Math.sqrt(k),Math.sqrt(k*L),(1.0000001192092896*k-k),(1.0000001192092896*k-k),Math.exp(-.125),Math.SQRT2]};
}
export function selectPLTTraceEvents(gate,eventCount){const selected=new Set();for(const failure of gate.failures??[]){const i=failure.index;if(Number.isInteger(i)&&i>=0&&i<eventCount)selected.add(i);if(selected.size===PLT_TRACE_MAX_EVENTS)break;}return [...selected];}
export function describePLTFailureTrace(packed,original,ids,output){
 if(output!==null&&(!(output instanceof Float32Array)||output.length!==ids.length*PLT_TRACE_EVENT_FLOATS))throw new Error('Incorrect PLT trace readback length');
 const eu=new Uint32Array(packed.eventData),ef=new Float32Array(packed.eventData);
 const events=ids.map((index,i)=>{
  const header=eu.subarray(index*8,index*8+4),[start,total,boundary]=header,direct=ef[index*8+4],raw=original.subarray(index*8,index*8+8),reconstructed=output?.subarray(i*208,i*208+8),edges=[];let soft=[direct,0],hard=[direct,0];
  for(let j=0;j<total;j++){
   const f=packed.edgeData.subarray((start+j)*8,(start+j+1)*8),reference=edgeReference(f),s=mul(reference.ds,reference.phase),h=mul(reference.dh,reference.phase);soft=soft.map((v,c)=>v+s[c]);hard=hard.map((v,c)=>v+h[c]);
   if(j<PLT_TRACE_MAX_EDGES){const trace=output?.subarray(i*208+8+j*96,i*208+8+(j+1)*96);edges.push({localIndex:j,packedInput:numbers(f),packedInputWords:words(f),gpu:trace?numbers(trace):null,gpuWords:trace?words(trace):null,cpuAtPackedInputs:numbers(reference.values)});}
  }
  const sp=norm(soft),hp=norm(hard),cpu=[...soft,...hard,boundary===0?sp:boundary===1?hp:(sp+hp)/2,sp,hp,total];
  const rawWords=words(raw),traceWords=reconstructed?words(reconstructed):null,laneErrors=reconstructed?Array.from(raw,(v,c)=>Math.abs(v-reconstructed[c])):null;
  return {index,referenceF:packed.eventReference[index],packedEventHeader:Array.from(header),packedDirect:direct,originalGPU:numbers(raw),originalGPUWords:rawWords,
   reconstructedGPU:reconstructed?numbers(reconstructed):null,reconstructedGPUWords:traceWords,cpuAtPackedInputs:numbers(cpu),reconstruction:reconstructed?{bitExactAllLanes:rawWords.every((v,c)=>v===traceWords[c]),bitExactSelectedPower:rawWords[4]===traceWords[4],absoluteErrors:numbers(laneErrors),selectedPowerWithinOriginalGate:Number.isFinite(laneErrors[4])&&laneErrors[4]<=2e-5+.003*Math.abs(raw[4])}:null,
   totalEdges:total,capturedEdges:edges.length,omittedEdges:total-edges.length,edges,intrinsicConstants:output?numbers(output.subarray(i*208+200,i*208+208)):null};
 });
 return {schema:PLT_TRACE_SCHEMA,status:output?'captured':'unavailable',scope:'Separate failure-only rerun with unchanged source prefix, helpers and original input buffers. Compiler context may differ. Original GPU readback is authoritative; probe never changes image admission.',
  limits:{events:8,edgesPerEvent:2},layout:{event:['softRe','softIm','hardRe','hardIm','selectedPower','softPower','hardPower','edgeCount'],packedEdge:['k','n','phiI','phiO','ro','sinBeta','phaseExcess','L'],edgeBlocks:'96 lanes: packedInput[8], cis[2], scaledPhase[2], coefficientSoft[2], coefficientHard[2], contributionSoft[2], contributionHard[2], cisNorm/sqrtRo/invSqrtRo/denominator[4], four cot blocks[16 each], intrinsic probes[8]',cot:'nearest,delta,t,h,sincT,sincH,cosH,ratio,rootKL,u,inv,uInv,erfcx[2],cotTransition[2]',intrinsicConstants:'sin(.125),cos(.125),sqrt(2),inverseSqrt(2),erfcxScaled(.5)[2],erfcxScaled(.9860957183805704)[2]',cpu:'f64 evaluation of actual packed f32 operands; L is packed lane 7. Separate from original f64 referenceF. CPU FMA probes shown in f64, not an f32 fused/unfused oracle.'},events};
}
