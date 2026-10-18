/** First-order spherical-source scalar UTD on a prepared 3-D triangle scene.
 * exp(+i omega t), outgoing exp(-ikR). The caller owns all direct/reflected
 * geometrical fields; this module returns ONLY edge-diffracted amplitudes.
 * NASA CR-2401 (19740020597), Eqs. 46–47, 50 and 57.
 * Straight convex soft/hard wedges, stationary interior points, finite source.
 * No impedance wedges, vertices, multiple diffraction, or reflection+D paths.
 */
import {add,sub,scale,dot,cross,length,distance,normalize,diffractionPoint,mergeCollinearWedges} from './mesh-geometry.js';
import {coherentSegmentVisible} from './coherent-mesh-core.js';
import {coefficient} from './utd-math.js';
import {stableBrokenPathExcess} from './plt-fsd.js';

const PI=Math.PI,TAU=2*PI;
export const COHERENT_DIFFRACTION_LIMITS=Object.freeze({
  maxEligibleEdges:64,boundaryEpsilon:1e-12,directionTangentEpsilon:1e-12,
  minSinBeta:1e-3,minKL:10,endpointFresnelGuard:3,maxPhaseArgument:2**45,
  maxTopologyPairs:2000000,
});
const cache=new WeakMap();
const wrap=x=>((x%TAU)+TAU)%TAU;
const reduce=x=>{const y=x%TAU;return y>PI?y-TAU:y< -PI?y+TAU:y;};
const point=(p,name)=>{
  if(!Array.isArray(p)||p.length!==3||p.some(x=>!Number.isFinite(x)||Math.abs(x)>1e9))
    throw new RangeError(`${name} must be a finite 3-vector within ±1e9 mm`);
  return p;
};
const positive=(x,name)=>{
  if(!(Number.isFinite(x)&&x>0))throw new RangeError(`${name} must be positive and finite`);
  return x;
};
function materialBoundary(material){
  if(material?.type!=='mirror'||material.reflectance!==1)return null;
  const phase=material.phaseRad%TAU;
  if(phase===0)return 'hard';
  if(phase===PI||phase===-PI)return 'soft';
  return null;
}

// Exact local halfspace classification avoids epsilon-sized sample offsets.
// p lies on the triangle plane; direction is tangent to that plane.
function triangleCoversDirection(t,p,direction,eps){
  const points=[t.a,t.b,t.c];
  for(let i=0;i<3;i++){
    const a=points[i],edge=sub(points[(i+1)%3],a),size=length(edge);
    const h=dot(cross(edge,sub(p,a)),t.normal);
    if(h < -eps*size)return false;
    if(h<=eps*size&&dot(cross(edge,direction),t.normal)<-1e-12*size)return false;
  }
  return true;
}
function triangleLineInterval(t,a,delta,eps){
  let lo=0,hi=1;
  const points=[t.a,t.b,t.c];
  for(let i=0;i<3;i++){
    const p=points[i],edge=sub(points[(i+1)%3],p),size=length(edge);
    const h=dot(cross(edge,sub(a,p)),t.normal),dh=dot(cross(edge,delta),t.normal);
    if(Math.abs(dh)<=1e-14*size*length(delta)){if(h < -eps*size)return null;continue;}
    const crossing=-h/dh;if(dh>0)lo=Math.max(lo,crossing);else hi=Math.min(hi,crossing);
    if(lo>hi+eps/length(delta))return null;
  }
  return lo<=1&&hi>=0?[Math.max(0,lo),Math.min(1,hi)]:null;
}

/** Reconstruct thin-plate boundaries from the scene-global coplanar UNION.
 * Mesh entries, triangle diagonals, overlaps and T-junctions are authoring
 * choices, not separate diffractors. Split before classifying partial overlap.
 * Solid creases retain the existing topology contract, separately below.
 */
function unionPlateWedges(scene,raw){
  const groups=[],fragments=[],excluded=[],all=[];let pairs=0;
  if(!raw.some(x=>x.wedge.alpha===0))return {all:raw,excluded,topologyCap:false};
  const eps=128*Number.EPSILON*Math.max(1,...scene.bounds.min.map(Math.abs),...scene.bounds.max.map(Math.abs));
  const tolerance=Math.max(1e-10,eps);
  for(const t of scene.triangles){
    const n=[...t.normal],axis=n.map(Math.abs).indexOf(Math.max(...n.map(Math.abs)));if(n[axis]<0)for(let j=0;j<3;j++)n[j]=-n[j];
    let group=null;for(const g of groups){
      if(++pairs>COHERENT_DIFFRACTION_LIMITS.maxTopologyPairs)return {all:[],excluded,topologyCap:true};
      if(distance(g.normal,n)<1e-11&&Math.abs(dot(sub(t.a,g.point),g.normal))<=tolerance){group=g;break;}
    }
    if(!group){group={id:groups.length,normal:n,point:t.a,triangles:[]};groups.push(group);}
    group.triangles.push(t);
  }
  for(const item of raw){
    const w=item.wedge;if(w.alpha!==0){all.push(item);continue;}
    const group=groups.find(g=>Math.abs(dot(g.normal,w.n1))>1-1e-11&&Math.abs(dot(sub(w.a,g.point),g.normal))<=tolerance);
    if(!group){excluded.push({edgeId:w.edgeId,reason:'missing-coplanar-union-plane'});continue;}
    pairs+=group.triangles.length;if(pairs>COHERENT_DIFFRACTION_LIMITS.maxTopologyPairs)return {all:[],excluded,topologyCap:true};
    const delta=sub(w.b,w.a),cuts=[0,1];
    for(const t of group.triangles){const interval=triangleLineInterval(t,w.a,delta,tolerance);if(interval)cuts.push(...interval);}
    cuts.sort((a,b)=>a-b);const breaks=cuts.filter((x,i)=>i===0||x-cuts[i-1]>tolerance/w.l);
    for(let i=0;i<breaks.length-1;i++){
      const lo=breaks[i],hi=breaks[i+1];if((hi-lo)*w.l<=tolerance)continue;
      const a=add(w.a,scale(delta,lo)),b=add(w.a,scale(delta,hi)),v=scale(add(a,b),.5),e=normalize(sub(b,a));
      const transverse=normalize(cross(group.normal,e));
      pairs+=2*group.triangles.length;if(pairs>COHERENT_DIFFRACTION_LIMITS.maxTopologyPairs)return {all:[],excluded,topologyCap:true};
      const plus=group.triangles.filter(t=>triangleCoversDirection(t,v,transverse,tolerance));
      const minus=group.triangles.filter(t=>triangleCoversDirection(t,v,scale(transverse,-1),tolerance));
      if(plus.length&&minus.length){excluded.push({edgeId:w.edgeId,reason:'internal-coplanar-authoring-seam'});continue;}
      const occupied=plus.length?plus:minus;
      if(!occupied.length){excluded.push({edgeId:w.edgeId,reason:'unresolved-coplanar-union-boundary'});continue;}
      const boundaries=occupied.map(t=>materialBoundary(scene.materials[t.materialIndex]));
      if(boundaries.some(x=>x===null||x!==item.boundary)){
        excluded.push({edgeId:w.edgeId,reason:'overlapping-or-changing-boundary-material'});continue;
      }
      const tff=plus.length?transverse:scale(transverse,-1),ids=occupied.map(t=>t.id);
      const duplicate=fragments.find(f=>f.groupId===group.id&&f.boundary===item.boundary&&
        ((distance(f.wedge.a,a)<=4*tolerance&&distance(f.wedge.b,b)<=4*tolerance)||
         (distance(f.wedge.a,b)<=4*tolerance&&distance(f.wedge.b,a)<=4*tolerance)));
      if(duplicate){duplicate.originalEdgeIds.push(w.edgeId);continue;}
      const id=fragments.length;
      fragments.push({groupId:group.id,boundary:item.boundary,originalEdgeIds:[w.edgeId],wedge:{
        ...w,id,edgeId:id,a,b,v,l:distance(a,b),e,n1:group.normal,n2:scale(group.normal,-1),t1:tff,t2:tff,
        componentId:-1-(2*group.id+(item.boundary==='hard'?1:0)),vertexIds:[],triangleIds:ids,faceTriangleIds:[ids,ids],segmentIds:[id],
      }});
    }
  }
  // Canonical geometric endpoint identities allow cross-entry fragment merging.
  const vertices=[];
  for(const f of fragments)f.wedge.vertexIds=[f.wedge.a,f.wedge.b].map(p=>{
    let id=vertices.findIndex(v=>distance(v,p)<=4*tolerance);if(id<0){id=vertices.length;vertices.push(p);}return `union:${id}`;
  });
  const merged=mergeCollinearWedges(fragments.map(f=>f.wedge));
  for(const w of merged){
    const records=w.segmentIds.map(id=>fragments[id]),boundary=records[0].boundary;
    // Do not merge across a material step along a collinear rim.
    if(records.some(r=>r.boundary!==boundary)){
      for(const r of records)all.push({wedge:{...r.wedge,id:scene.edges.length+r.wedge.id,edgeId:scene.edges.length+r.wedge.id,originalEdgeIds:r.originalEdgeIds},boundary:r.boundary});
    }else all.push({wedge:{...w,id:scene.edges.length+w.id,edgeId:scene.edges.length+w.id,
      originalEdgeIds:[...new Set(records.flatMap(r=>r.originalEdgeIds))]},boundary});
  }
  return {all,excluded,topologyCap:false};
}

/** The geometry is immutable, like preparePLTScene. Rebuild after geometry
 * changes. A material signature also avoids stale admission after UI changes.
 */
export function prepareMeshDiffraction(scene){
  if(!Array.isArray(scene?.triangles)||!Array.isArray(scene?.wedges)||!Array.isArray(scene?.materials)||!scene.bvh)
    throw new TypeError('Expected a material-indexed preparePLTScene result');
  const signature=JSON.stringify(scene.materials.map(m=>[m?.type,m?.reflectance,m?.phaseRad]));
  const previous=cache.get(scene);
  if(previous?.signature===signature)return previous;
  const raw=[],excluded=[];
  for(const wedge of scene.wedges){
    const boundaries=wedge.triangleIds.map(id=>materialBoundary(scene.materials[scene.triangles[id]?.materialIndex]));
    if(!boundaries.length||boundaries.some(b=>b===null)){
      excluded.push({edgeId:wedge.edgeId,reason:'nonideal-or-nonreflecting-edge-material'});continue;
    }
    if(boundaries.some(b=>b!==boundaries[0])){
      excluded.push({edgeId:wedge.edgeId,reason:'inconsistent-scalar-boundary-across-edge'});continue;
    }
    raw.push({wedge,boundary:boundaries[0]});
  }
  const union=unionPlateWedges(scene,raw),eligible=union.all;
  const materialExcludedEdges=excluded.length;
  excluded.push(...union.excluded);
  const unsupportedTopology=(scene.unsupportedEdges??[]).map(e=>({edgeId:e.edgeId,reason:e.code}));
  const result={signature,eligible,excluded,unsupportedTopology,materialExcludedEdges,topologyCap:union.topologyCap,
    supported:!union.topologyCap&&eligible.length<=COHERENT_DIFFRACTION_LIMITS.maxEligibleEdges};
  cache.set(scene,result);return result;
}

function unboundedStationary(w,src,dst){
  const e=w.e??normalize(sub(w.b,w.a)),v=w.v??scale(add(w.a,w.b),.5);
  const s=sub(src,v),d=sub(dst,v),sa=dot(s,e),da=dot(d,e);
  const rs=length(sub(s,scale(e,sa))),rd=length(sub(d,scale(e,da)));
  if(!(rs+rd>0))return null;
  const axial=sa+(da-sa)*(rs/(rs+rd)),q=add(v,scale(e,axial));
  const ri=distance(src,q),ro=distance(dst,q);
  if(!(ri>0&&ro>0))return null;
  const sinBeta=rs/ri;
  return {point:q,axial,ri,ro,sinBeta,endpointDistance:w.l/2-Math.abs(axial)};
}

function orientedFrame(w,wi){
  const eps=COHERENT_DIFFRACTION_LIMITS.directionTangentEpsilon;
  const first=dot(wi,w.n1)>eps;
  if(!first&&dot(wi,w.n2)<=eps)return null;
  return {nff:first?w.n1:w.n2,tff:first?w.t1:w.t2,nbf:first?w.n2:w.n1,
    frontTriangleIds:w.faceTriangleIds?.[first?0:1]??w.triangleIds,
    backTriangleIds:w.faceTriangleIds?.[first?1:0]??w.triangleIds};
}
function angles(frame,wi,wo){
  return {phiI:wrap(Math.atan2(dot(wi,frame.nff),dot(wi,frame.tff))),
    phiO:wrap(Math.atan2(dot(wo,frame.nff),dot(wo,frame.tff)))};
}

/**
 * Routes use the one receiver-common carrier exp(-ik*referenceDistance).
 * Default referenceDistance=|receiver-source|; never square an edge in here.
 * Material-excluded edges remain opaque to both leg visibility queries.
 * diagnostics.valid only reports arithmetic/safety admission. Local model
 * warnings never claim full-wave accuracy and do not mask the whole pixel.
 */
export function evaluateMeshDiffraction(scene,sourcePosition,receiverPosition,options={}){
  return evaluateDiffractionFamily(scene,sourcePosition,receiverPosition,options);
}
/** Internal reusable image-endpoint operator. A family must admit the ACTUAL
 * broken physical legs; virtual straight segments must never test visibility.
 * Kept separate from user-facing scene options. */
export function evaluateDiffractionFamily(scene,sourcePosition,receiverPosition,options={},family=null){
  const src=point(sourcePosition,'Source'),dst=point(receiverPosition,'Receiver');
  const wavelengthMm=positive(options.wavelengthMm,'wavelengthMm'),k=TAU/wavelengthMm;
  const power=options.power??1;
  if(!(Number.isFinite(power)&&power>=0))throw new RangeError('power must be finite and nonnegative');
  const directDistance=positive(distance(src,dst),'Source-receiver distance');
  const referenceDistance=options.referenceDistance??directDistance;
  if(!(Number.isFinite(referenceDistance)&&referenceDistance>=0))throw new RangeError('referenceDistance must be finite and nonnegative');
  const prepared=family?.prepared??prepareMeshDiffraction(scene),routes=[],reflectionBoundaries=[],directBoundaryEdges=[],directBoundaries=[];
  const excluded=prepared.excluded.map(x=>({...x})),warnings=new Set(),invalidReasons=new Set();
  if(prepared.materialExcludedEdges)warnings.add('edge-material-diffraction-not-modeled');
  if(prepared.excluded.some(x=>x.reason==='overlapping-or-changing-boundary-material'))warnings.add('overlapping-or-changing-boundary-material');
  if(prepared.unsupportedTopology.length)warnings.add('unsupported-wedge-topology');
  const diagnostics={model:'first-order-spherical-source-scalar-UTD',eligibleEdges:prepared.eligible.length,
    materialExcludedEdges:prepared.materialExcludedEdges,unsupportedTopology:prepared.unsupportedTopology,
    excluded,warnings:[],invalidReasons:[],valid:true,visibilityQueries:0,edgeEvaluations:0,
    timeConvention:'exp(+i omega t); outgoing exp(-ikR)',
    omittedFamilies:['edge-endpoints-and-vertices','two-or-more-diffractions','reflection-before-or-after-diffraction','impedance-and-absorber-edge-diffraction']};
  const finish=()=>{
    diagnostics.warnings=[...warnings];diagnostics.invalidReasons=[...invalidReasons];diagnostics.valid=invalidReasons.size===0;
    return {routes,directBoundary:directBoundaryEdges.length>0,directBoundaryEdges,directBoundaries,reflectionBoundaries,diagnostics};
  };
  if(!prepared.supported){invalidReasons.add(prepared.topologyCap?'coplanar-union-safety-cap':'eligible-edge-safety-cap');return finish();}
  if(!Number.isFinite(k)){invalidReasons.add('nonfinite-wavenumber');return finish();}
  if(power===0)return finish();
  const visible=(a,b)=>{diagnostics.visibilityQueries++;return coherentSegmentVisible(scene,a,b);};
  const eps=COHERENT_DIFFRACTION_LIMITS.boundaryEpsilon;
  const sourceAmplitude=Math.sqrt(power/(4*PI));
  for(const {wedge:w,boundary} of prepared.eligible){
    const hit=diffractionPoint(w,src,dst);
    if(!hit){
      excluded.push({edgeId:w.edgeId,reason:'no-interior-finite-edge-saddle'});
      const h=unboundedStationary(w,src,dst);
      if(h&&h.sinBeta>=COHERENT_DIFFRACTION_LIMITS.minSinBeta){
        const width=1/Math.sqrt(k*h.sinBeta*h.sinBeta*(1/h.ri+1/h.ro));
        if(h.endpointDistance>=-COHERENT_DIFFRACTION_LIMITS.endpointFresnelGuard*width){
          const endpoint=distance(h.point,w.a)<distance(h.point,w.b)?w.a:w.b;
          if(family?family.admit(w,{point:endpoint}):visible(src,endpoint)&&visible(endpoint,dst))warnings.add('endpoint-dominated-diffraction-not-modeled');
        }
      }
      continue;
    }
    const frame=orientedFrame(w,hit.wi);
    if(!frame){excluded.push({edgeId:w.edgeId,reason:'incident-inside-or-tangent-to-wedge'});continue;}
    const {phiI,phiO}=angles(frame,hit.wi,hit.wo),n=2-w.alpha/PI;
    if(phiI>n*PI+eps||phiO>n*PI+eps){excluded.push({edgeId:w.edgeId,reason:'inside-solid-angular-sector'});continue;}
    const familyData=family?family.admit(w,hit):null;
    if(family?!familyData:(!visible(src,hit.point)||!visible(hit.point,dst))){
      excluded.push({edgeId:w.edgeId,reason:'occluded-edge-leg'});continue;
    }
    if(hit.atEndpoint){excluded.push({edgeId:w.edgeId,reason:'endpoint-stationary-point-not-modeled'});warnings.add('endpoint-dominated-diffraction-not-modeled');continue;}
    if(hit.sinBeta<COHERENT_DIFFRACTION_LIMITS.minSinBeta){
      excluded.push({edgeId:w.edgeId,reason:'edge-grazing-not-modeled'});warnings.add('edge-grazing-not-modeled');continue;
    }
    if(Math.min(phiI,phiO,n*PI-phiI,n*PI-phiO)<COHERENT_DIFFRACTION_LIMITS.minSinBeta){
      excluded.push({edgeId:w.edgeId,reason:'face-grazing-not-modeled'});warnings.add('face-grazing-not-modeled');continue;
    }
    const L=(hit.ri/(hit.ri+hit.ro))*hit.ro*hit.sinBeta*hit.sinBeta;
    if(!(L>0&&Number.isFinite(k*L))){excluded.push({edgeId:w.edgeId,reason:'nonfinite-reduced-distance'});invalidReasons.add('nonfinite-reduced-distance');continue;}
    if(k*L<COHERENT_DIFFRACTION_LIMITS.minKL)warnings.add('low-kL-asymptotic-regime');
    const width=1/Math.sqrt(k*hit.sinBeta*hit.sinBeta*(1/hit.ri+1/hit.ro));
    const endpointDistance=Math.min(distance(hit.point,w.a),distance(hit.point,w.b));
    if(endpointDistance<COHERENT_DIFFRACTION_LIMITS.endpointFresnelGuard*width)
      warnings.add('endpoint-dominated-diffraction-not-modeled');
    const excess=familyData?.opticalExcessMm??(stableBrokenPathExcess(src,hit.point,dst)+(directDistance-referenceDistance));
    const phaseArgument=k*excess;
    if(!Number.isFinite(phaseArgument)||Math.abs(phaseArgument)>COHERENT_DIFFRACTION_LIMITS.maxPhaseArgument){
      excluded.push({edgeId:w.edgeId,reason:'phase-resolution-limit'});invalidReasons.add('phase-resolution-limit');continue;
    }
    const d=coefficient(k,n,phiI,phiO,L,hit.sinBeta,boundary,eps);
    const amplitude=sourceAmplitude*Math.hypot(...d)/(Math.sqrt(hit.ri)*Math.sqrt(hit.ro)*Math.sqrt(hit.ri+hit.ro));
    const phaseRadReduced=reduce(Math.atan2(d[1],d[0])-reduce(phaseArgument));
    if(!Number.isFinite(amplitude)||!Number.isFinite(phaseRadReduced)){
      excluded.push({edgeId:w.edgeId,reason:'nonfinite-diffracted-field'});invalidReasons.add('nonfinite-diffracted-field');continue;
    }
    diagnostics.edgeEvaluations++;
    routes.push({kind:'diffraction',edgeId:w.edgeId,point:[...hit.point],boundary,...(familyData?{familyData}:{}),
      originalEdgeIds:w.originalEdgeIds??[w.edgeId],triangleIds:[...w.triangleIds],amplitude,phaseRadReduced,opticalLengthMm:hit.ri+hit.ro,
      opticalExcessMm:excess,ri:hit.ri,ro:hit.ro,L,n,sinBeta:hit.sinBeta,phiI,phiO,
      coefficient:[...d],endpointFresnelWidthMm:width,endpointDistanceMm:endpointDistance});
    if(Math.abs(Math.abs(phiO-phiI)-PI)<=eps){directBoundaryEdges.push(w.edgeId);directBoundaries.push({edgeId:w.edgeId,point:[...hit.point],familyData});}
    const sum=phiO+phiI;
    if(Math.abs(sum-PI)<=eps)
      reflectionBoundaries.push({edgeId:w.edgeId,triangleIds:[...frame.frontTriangleIds],point:[...hit.point],boundary,familyData});
    if(Math.abs(sum-(2*n*PI-PI))<=eps)
      reflectionBoundaries.push({edgeId:w.edgeId,triangleIds:[...frame.backTriangleIds],point:[...hit.point],boundary,familyData});
  }
  if(directBoundaryEdges.length>1)warnings.add('multiple-direct-shadow-boundaries');
  return finish();
}
