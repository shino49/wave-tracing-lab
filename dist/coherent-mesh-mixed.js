/** One external planar reflection and one finite-source UTD interaction.
 * Unfold only the reflected endpoint, retain the physical edge frame, and
 * validate every actual broken leg. No R-D-R, repeated D or mirror-rim model. */
import {reflectCoherentPoint,coherentMirrorBounce,coherentSegmentVisible,reducedPhase} from './coherent-mesh-core.js';
import {prepareMeshDiffraction,evaluateDiffractionFamily} from './coherent-mesh-diffraction.js';
import {distance,dot,sub,add,scale} from './mesh-geometry.js';
import {stableBrokenPathExcess} from './plt-fsd.js';
const cache=new WeakMap();
export function prepareMixedDiffraction(scene){
 const base=prepareMeshDiffraction(scene),old=cache.get(scene);if(old?.base===base)return old;
 const families=[];
 for(const plane of scene.mirrorPlanes){
  const eligible=base.eligible.filter(({wedge:w})=>!(Math.abs(dot(sub(w.a,plane.point),plane.normal))<=plane.tolerance&&Math.abs(dot(sub(w.b,plane.point),plane.normal))<=plane.tolerance));
  if(eligible.length)for(const order of ['RD','DR'])families.push({plane,order,prepared:{...base,eligible,excluded:[],materialExcludedEdges:0,unsupportedTopology:[]}});
 }
 const result={base,families,count:families.reduce((n,f)=>n+f.prepared.eligible.length,0)};cache.set(scene,result);return result;
}
function excess(vertices){let d=0;for(let i=1;i<vertices.length-1;i++)d+=stableBrokenPathExcess(vertices[0],vertices[i],vertices[i+1]);return d;}
export function evaluateMixedDiffraction(scene,compiled,receiver){
 if(scene.maxReflections!==2)throw new Error('Mixed R-D / D-R requires GO reflection order 2 for its boundary partners');
 const prepared=prepareMixedDiffraction(scene),source=compiled.source.position,routes=[],goSeams=[];
 const warnings=new Set(),invalidReasons=new Set(),excluded=[];
 const diagnostics={model:'one-external-planar-reflection-plus-one-spherical-UTD',eligibleFamilies:prepared.count,edgeEvaluations:0,visibilityQueries:0,excluded,warnings:[],invalidReasons:[],valid:true,
  omittedFamilies:['R-D-R','two-or-more-diffractions','external-mirror-rim-transitions','edge-endpoint-and-corner-diffraction']};
 for(const family of prepared.families){
  const {plane,order}=family,isRD=order==='RD',image=reflectCoherentPoint(isRD?source:receiver,plane);
  const src=isRD?image:source,dst=isRD?receiver:image;
  if(distance(src,dst)<1e-12){excluded.push({interactionOrder:order,mirrorPlaneId:plane.id,reason:'coincident-virtual-endpoints-no-physical-reflection'});continue;}
  const admitted=(w,hit)=>{
   const q=hit.point,bounce=coherentMirrorBounce(scene,plane,isRD?source:q,isRD?q:receiver);if(!bounce)return null;
   if(bounce.boundary){warnings.add('mixed-external-mirror-rim-not-modeled');return null;}
   if(hit.sinBeta>0&&hit.ri>0&&hit.ro>0){const width=3/Math.sqrt(compiled.k*hit.sinBeta*hit.sinBeta*(1/hit.ri+1/hit.ro));for(const sign of [-1,1]){const near=add(q,scale(w.e,sign*width)),patch=coherentMirrorBounce(scene,plane,isRD?source:near,isRD?near:receiver);if(!patch||patch.boundary||patch.material.reflectance!==bounce.material.reflectance||Math.abs(reducedPhase(patch.material.phaseRad-bounce.material.phaseRad))>1e-12)warnings.add('mixed-external-mirror-illumination-boundary-nearby');}}
   const vertices=isRD?[source,bounce.point,q,receiver]:[source,q,bounce.point,receiver];
   for(let j=0;j<3;j++){diagnostics.visibilityQueries++;if(!coherentSegmentVisible(scene,vertices[j],vertices[j+1]))return null;}
   return {mirrorPoint:bounce.point,mirrorMaterial:bounce.material,vertices,opticalExcessMm:excess(vertices)};
  };
  const result=evaluateDiffractionFamily(scene,src,dst,{wavelengthMm:compiled.source.wavelengthMm,power:compiled.source.power,referenceDistance:distance(source,receiver)},{prepared:family.prepared,admit:admitted});
  for(const r of result.routes){const m=r.familyData.mirrorMaterial;routes.push({...r,kind:'mixed-diffraction',interactionOrder:order,mirrorPlaneId:plane.id,vertices:r.familyData.vertices,
   amplitude:r.amplitude*Math.sqrt(m.reflectance),phaseRadReduced:reducedPhase(r.phaseRadReduced+compiled.source.phaseRad+m.phaseRad)});}
  function seam(record,wedgePlane=null){
   const f=record.familyData;if(!f)return;
   const m=f.mirrorMaterial,sequence=wedgePlane?(isRD?[plane.id,wedgePlane.id]:[wedgePlane.id,plane.id]):[plane.id];
   const vertices=wedgePlane?f.vertices:[source,f.mirrorPoint,receiver];
   const pathLength=vertices.slice(1).reduce((sum,p,j)=>sum+distance(vertices[j],p),0),e=excess(vertices);
   goSeams.push({kind:'reflection',sequence,vertices,pathLength,excess:e,amplitude:.5*compiled.amplitudeScale*Math.sqrt(m.reflectance)/pathLength,
    interactionPhaseRad:compiled.source.phaseRad+m.phaseRad+(wedgePlane?(record.boundary==='soft'?Math.PI:0):0),phaseRadReduced:reducedPhase(compiled.source.phaseRad+m.phaseRad+(wedgePlane?(record.boundary==='soft'?Math.PI:0):0)-compiled.k*e),boundary:true,mixedBoundaryPartner:order,mixedSeamEvent:{edgeId:record.edgeId,point:record.point,order,type:wedgePlane?'reflection':'direct'}});
  }
  for(const s of result.directBoundaries)seam(s);
  for(const s of result.reflectionBoundaries){const wedgePlane=scene.mirrorPlanes.find(p=>p.triangles.some(t=>s.triangleIds.includes(t.id)));if(wedgePlane&&wedgePlane.id!==plane.id)seam(s,wedgePlane);}
  diagnostics.edgeEvaluations+=result.diagnostics.edgeEvaluations;
  for(const warning of result.diagnostics.warnings)warnings.add(warning);
  for(const reason of result.diagnostics.invalidReasons)invalidReasons.add(reason);
  for(const x of result.diagnostics.excluded)excluded.push({...x,interactionOrder:order,mirrorPlaneId:plane.id});
 }
 diagnostics.warnings=[...warnings];diagnostics.invalidReasons=[...invalidReasons];diagnostics.valid=!invalidReasons.size;
 return {routes,goSeams,diagnostics};
}
