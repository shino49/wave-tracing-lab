import {slabAbsorptionAt} from './coherent-mesh-slab.js';
/** Frequency-independent geometry for nondispersive scalar point-source paths.
 * Every coefficient and propagation phase is evaluated again at each lambda.
 * No final mono D amplitude is cached as if it were a broadband envelope. */
import {compileCoherentSource,reducedPhase} from './coherent-mesh-core.js';
import {evaluateMeshTransport} from './coherent-mesh-transport.js';
import {coefficient} from './utd-math.js';
export function compileCoherentSpectralSource(scene,source){
 // Longest admitted wavelength gives conservative endpoint / low-kL warnings.
 // Geometry-only exclusions do not depend on wavelength in this bounded model.
 return compileCoherentSource(scene,{...source,wavelengthMm:.00083});
}
export function prepareCoherentSpectralSample(scene,compiled,receiver,options={}){
 const result=evaluateMeshTransport(scene,compiled,receiver,options),templates=result.routes.map(r=>{
  if(r.kind==='transmission')return {kind:3,amplitudeScale:r.unabsorbedAmplitude,interactionPhaseRad:r.interactionPhaseRad,opticalExcessMm:r.excess,glassLength:r.glassLength,slab:r.slab,routeKind:r.kind,route:r};
  if(r.kind==='direct'||r.kind==='reflection')return {kind:1,amplitudeScale:r.amplitude,interactionPhaseRad:r.interactionPhaseRad??compiled.source.phaseRad,opticalExcessMm:r.excess??0,routeKind:r.kind,route:r};
  const material=r.familyData?.mirrorMaterial,mirrorScale=material?Math.sqrt(material.reflectance):1;
  return {kind:2,amplitudeScale:compiled.amplitudeScale*mirrorScale/(Math.sqrt(r.ri)*Math.sqrt(r.ro)*Math.sqrt(r.ri+r.ro)),interactionPhaseRad:compiled.source.phaseRad+(material?.phaseRad??0),opticalExcessMm:r.opticalExcessMm,
   n:r.n,phiI:r.phiI,phiO:r.phiO,L:r.L,sinBeta:r.sinBeta,boundary:r.boundary,routeKind:r.kind,route:r};
 });
 for(const t of templates)if(!(Number.isFinite(t.amplitudeScale)&&t.amplitudeScale>=0&&Number.isFinite(t.opticalExcessMm)&&Math.abs(t.opticalExcessMm)<=1e5))throw new Error('Spectral route template exceeds finite/excess capacity');
 return {...result,templates};
}
export function evaluateCoherentSpectralTemplate(t,wavelengthNm){
 if(!Number.isFinite(wavelengthNm)||wavelengthNm<360||wavelengthNm>830)throw new RangeError('Visible wavelength must be360–830nm');
 const k=2*Math.PI/(wavelengthNm/1e6),d=t.kind===2?coefficient(k,t.n,t.phiI,t.phiO,t.L,t.sinBeta,t.boundary,1e-12):[1,0];
 const amplitude=t.amplitudeScale*Math.hypot(...d)*(t.kind===3?Math.exp(-.5*slabAbsorptionAt(t.slab,wavelengthNm)*t.glassLength):1),phaseRadReduced=reducedPhase(t.interactionPhaseRad+Math.atan2(d[1],d[0])-k*t.opticalExcessMm);
 if(!Number.isFinite(amplitude)||!Number.isFinite(phaseRadReduced))throw new Error('Nonfinite spectral coefficient or phase');
 return {amplitude,phaseRadReduced,real:amplitude*Math.cos(phaseRadReduced),imag:amplitude*Math.sin(phaseRadReduced),kind:t.kind};
}
export function evaluateCoherentSpectralSampleCPU(templates,nodes){
 let coherent=0,incoherent=0,geometric=0;const coherentXYZ=[0,0,0],incoherentXYZ=[0,0,0],geometricXYZ=[0,0,0];
 for(const node of nodes){let re=0,im=0,inc=0,geo=0;for(const t of templates){const v=evaluateCoherentSpectralTemplate(t,node.wavelengthNm);re+=v.real;im+=v.imag;const p=v.amplitude*v.amplitude;inc+=p;if(t.kind===1||t.kind===3)geo+=p;}
  const c=re*re+im*im,w=node.weight;coherent+=w*c;incoherent+=w*inc;geometric+=w*geo;for(let j=0;j<3;j++){const wc=w*node.xyz[j];coherentXYZ[j]+=wc*c;incoherentXYZ[j]+=wc*inc;geometricXYZ[j]+=wc*geo;}
 }
 return {coherent,incoherent,geometric,coherentXYZ,incoherentXYZ,geometricXYZ};
}
/** Compact immutable geometry records consumed by the spectral WGSL runner. */
export function packCoherentSpectralTemplates(templates,target,offset,cotDeltas){
 for(let j=0;j<templates.length;j++){const t=templates[j],at=offset+16*j,hi=Math.fround(t.opticalExcessMm);target[at]=t.amplitudeScale;target[at+1]=reducedPhase(t.interactionPhaseRad);target[at+2]=hi;target[at+3]=t.opticalExcessMm-hi;target[at+4]=t.kind;
  if(t.kind===3)target[at+5]=t.glassLength;
  if(t.kind===2){target[at+5]=t.n;target[at+6]=t.phiI;target[at+7]=t.phiO;target[at+8]=t.L;target[at+9]=t.sinBeta;target[at+10]=t.boundary==='soft'?-1:1;target.set(cotDeltas(t.n,t.phiI,t.phiO),at+12);}
 }
}
