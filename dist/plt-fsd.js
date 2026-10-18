/** Local free-space-diffraction operator for the backward 3-D region walker.
 * Independently implemented from the Wave Tracing paper and audited native
 * control flow. Coherence is LOCAL to one region; the result is an intensity
 * transfer, not a complex amplitude to multiply across arbitrary paths.
 * Units are consistent scene lengths and inverse lengths. See architecture.
 */
import {
  add,sub,scale,dot,cross,length,distance,normalize,diffractionPoint,
} from './mesh-geometry.js';
import {segmentVisible,coneContains} from './plt-geometry.js';
import {coefficient,add as cadd,mul as cmul,scale as cscale,cis,abs2} from './utd-math.js';

const TAU=2*Math.PI;
export const FSD_LIMITS=Object.freeze({minSinBeta:1e-3,boundaryEpsilon:2e-6,importanceSigmaScale:45,directionTangentEpsilon:1e-12});
const wrap=x=>((x%TAU)+TAU)%TAU;
const wrapSigned=x=>{const y=wrap(x);return y>Math.PI?y-TAU:y;};
const vector=(v,name)=>{if(!Array.isArray(v)||v.length!==3||!v.every(Number.isFinite))throw new RangeError(`${name} must be a finite 3-vector`);return [...v];};
const positive=(v,name)=>{if(!(Number.isFinite(v)&&v>0))throw new RangeError(`${name} must be positive and finite`);return v;};
const unit=(v,name)=>normalize(vector(v,name));
const tolerance=(...v)=>128*Number.EPSILON*Math.max(1,...v.flat().map(Math.abs));
const directions=(edge,wi,wo)=>({phiI:wrap(Math.atan2(dot(wi,edge.nff),dot(wi,edge.tff))),phiO:wrap(Math.atan2(dot(wo,edge.nff),dot(wo,edge.tff)))});
// Normal/direction dot products are dimensionless. Treat both sides of the
// numerical tangent band identically so a rigid rotation cannot activate an
// exactly coplanar path through a +1e-16 round-off perturbation.
const outside=(edge,d)=>dot(d,edge.nff)>FSD_LIMITS.directionTangentEpsilon||dot(d,edge.nbf)>FSD_LIMITS.directionTangentEpsilon;
function orientForFSD(wedge,wi) {
  const first=dot(wi,wedge.n1)>FSD_LIMITS.directionTangentEpsilon;
  if(!first&&dot(wi,wedge.n2)<=FSD_LIMITS.directionTangentEpsilon)return null;
  const nff=first?wedge.n1:wedge.n2,tff=first?wedge.t1:wedge.t2,nbf=first?wedge.n2:wedge.n1;
  return {...wedge,nff:[...nff],tff:[...tff],nbf:[...nbf],e:normalize(cross(nff,tff)),
    frontTriangleId:wedge.faceTriangleIds?.[first?0:1]?.[0]??wedge.triangleIds?.[first?0:1]??wedge.triangleIds?.[0]};
}
const random=rng=>{
  const r=typeof rng==='function'?rng():rng?.r?.();
  if(!(r>=0&&r<1&&Number.isFinite(r)))throw new RangeError('rng must return a finite sample in [0,1)');
  return r;
};

/** Normalized wrapped normal on one angular period, including ALL images.
 * Narrow distributions use positive Gaussian images, broad ones a rapidly
 * convergent Fourier series. Truncation is below double precision relevance.
 */
export function wrappedNormalPDF(angle,sigma) {
  positive(sigma,'sigma');
  if(!Number.isFinite(angle))throw new RangeError('angle must be finite');
  const x=wrapSigned(angle);
  if(sigma<1) {
    const limit=Math.ceil(9*sigma/TAU)+1;
    let sum=0;
    for(let j=-limit;j<=limit;j++){const z=(x+j*TAU)/sigma;sum+=Math.exp(-.5*z*z);}
    return sum/(Math.sqrt(TAU)*sigma);
  }
  let sum=1;
  for(let j=1;j<100;j++) {
    const amplitude=Math.exp(-.5*(j*sigma)**2);
    if(amplitude<1e-17)break;
    sum+=2*amplitude*Math.cos(j*x);
  }
  return Math.max(0,sum/TAU);
}

/** The directional saddle on a finite edge, for a source and outgoing wo.
 * The point is NOT clamped to a physical or region-created endpoint.
 */
export function diffractionPointDirection(edge,source,wo) {
  const src=vector(source,'source'),out=unit(wo,'outgoing direction');
  const v=edge.v??scale(add(edge.a,edge.b),.5),e=edge.e??normalize(sub(edge.b,edge.a));
  const l=edge.l??distance(edge.a,edge.b),s=sub(src,v),sa=dot(s,e);
  const radial=length(sub(s,scale(e,sa))),cosBeta=dot(out,e);
  const sinBeta=Math.sqrt(Math.max(0,1-cosBeta*cosBeta));
  if(sinBeta<FSD_LIMITS.minSinBeta||radial<=tolerance(src,v))return null;
  const axial=sa+radial*cosBeta/sinBeta;
  if(Math.abs(axial)>l/2)return null;
  const point=add(v,scale(e,axial)),ri=distance(src,point);
  if(!(ri>0))return null;
  const wi=scale(sub(src,point),1/ri);
  return {point,axial,ri,wi,wo:out,sinBeta,fermatResidual:dot(wi,e)+cosBeta};
}

/** All geometry-worker wedges must already be clipped to this region's
 * footprint. No selected-edge pair, extruded 2-D fixture, or global path list.
 */
export function makeAperture(region,incomingDirection,k,boundary='unpolarized') {
  positive(k,'wavenumber');
  if(!['soft','hard','unpolarized'].includes(boundary))throw new RangeError('Unknown FSD boundary');
  const incoming=unit(incomingDirection,'incoming direction'),wi=scale(incoming,-1);
  const center=vector(region?.center??region?.region?.center,'region center');
  const boundaryEpsilon=region.boundaryEpsilon??region.region?.boundaryEpsilon??FSD_LIMITS.boundaryEpsilon;
  if(!(Number.isFinite(boundaryEpsilon)&&boundaryEpsilon>=0&&boundaryEpsilon<=.001))throw new RangeError('FSD boundary epsilon must be finite in [0, .001]');
  if(!Array.isArray(region.wedges))throw new TypeError('region.wedges must contain the collected, clipped wedges');
  const edges=[],skipped=[];
  for(const wedge of region.wedges) {
    const oriented=orientForFSD(wedge,wi);
    if(!oriented){skipped.push({edgeId:wedge.edgeId??wedge.id,reason:'incident-inside-or-tangent-to-wedge'});continue;}
    if(!(oriented.l>0&&Number.isFinite(oriented.l)))throw new RangeError('Aperture edge length must be positive and finite');
    if(!(oriented.alpha>=0&&oriented.alpha<=Math.PI))throw new RangeError('Aperture needs convex PEC wedges or open plates');
    edges.push(oriented);
  }
  return {center,edges,k,boundary,incomingDirection:incoming,region,
    boundaryEpsilon,
    diagnostics:{skipped,model:'local-UTD-region-intensity',finiteSourceCurvature:'native-local-plane-approximation'}};
}

/** Native-style density along the Keller cone, summed over the aperture.
 * This is an angular proposal density (rad^-1), NOT a solid-angle PDF.
 * The sampled edge coordinate is an auxiliary variable as in the upstream
 * implementation; its length/Jacobian is NOT claimed to cancel rigorously.
 */
export function pdfFSD(aperture,source,wo) {
  if(!aperture.edges.length)return 0;
  let density=0;
  for(const edge of aperture.edges) {
    const p=diffractionPointDirection(edge,source,wo);
    if(!p||!outside(edge,p.wi)||!outside(edge,p.wo))continue;
    const {phiI,phiO}=directions(edge,p.wi,p.wo);
    const sigma=Math.sqrt(FSD_LIMITS.importanceSigmaScale/(aperture.k*p.ri));
    density+=(wrappedNormalPDF(phiO-(Math.PI+phiI),sigma)+wrappedNormalPDF(phiO-(Math.PI-phiI),sigma))/2;
  }
  return density/(aperture.edges.length+1);
}

export function sampleFSD(aperture,source,rng=Math.random) {
  const src=vector(source,'source'),count=aperture.edges.length;
  const selected=Math.min(count,Math.floor(random(rng)*(count+1)));
  const base={valid:false,wo:[0,0,1],point:[...aperture.center],isDirect:false,weight:0,
    pdf:{measure:'angle',value:0},edgeId:null,
    auxiliary:{edgeProbability:1/(count+1),positionDensity:null},
    diagnostics:{measure:'native-Keller-angle-with-auxiliary-edge-position',solidAngleUnbiased:false}};
  if(selected===count) {
    if(distance(src,aperture.center)<=tolerance(src,aperture.center))return {...base,diagnostics:{...base.diagnostics,reason:'source-at-region-center'}};
    return {...base,valid:true,wo:normalize(sub(aperture.center,src)),isDirect:true,weight:count+1,
      pdf:{measure:'discrete',value:1/(count+1)}};
  }
  const edge=aperture.edges[selected],coordinate=random(rng)-.5;
  const point=add(edge.v,scale(edge.e,coordinate*edge.l));
  const ri=distance(src,point),wi=ri>0?scale(sub(src,point),1/ri):[0,0,0];
  const edgeId=edge.edgeId??edge.id;
  const auxiliary={edgeProbability:1/(count+1),positionDensity:1/edge.l};
  if(!(ri>0)||!outside(edge,wi))return {...base,point,edgeId,auxiliary,diagnostics:{...base.diagnostics,reason:'invalid-incident-edge-side'}};
  const cosBeta=dot(wi,edge.e),sinBeta=Math.sqrt(Math.max(0,1-cosBeta*cosBeta));
  if(sinBeta<FSD_LIMITS.minSinBeta)return {...base,point,edgeId,auxiliary,diagnostics:{...base.diagnostics,reason:'edge-grazing'}};
  const phiI=Math.atan2(dot(wi,edge.nff),dot(wi,edge.tff));
  const sigma=Math.sqrt(FSD_LIMITS.importanceSigmaScale/(aperture.k*ri));
  const gaussian=Math.sqrt(-2*Math.log(Math.max(Number.MIN_VALUE,random(rng))))*Math.cos(TAU*random(rng));
  const phiO=(random(rng)<.5?Math.PI+phiI:Math.PI-phiI)+sigma*gaussian;
  const wo=normalize(add(scale(add(scale(edge.tff,Math.cos(phiO)),scale(edge.nff,Math.sin(phiO))),sinBeta),scale(edge.e,-cosBeta)));
  if(!outside(edge,wo))return {...base,wo,point,edgeId,auxiliary,diagnostics:{...base.diagnostics,reason:'outgoing-inside-wedge'}};
  const pdf=pdfFSD(aperture,src,wo);
  if(!(pdf>0&&Number.isFinite(1/pdf)))return {...base,wo,point,edgeId,auxiliary,diagnostics:{...base.diagnostics,reason:'zero-or-nonfinite-angle-density'}};
  return {...base,valid:true,wo,point,edgeId,auxiliary,weight:1/pdf,pdf:{measure:'angle',value:pdf}};
}

/** (|p-s|+|d-p|)-|d-s| without subtracting long, nearly equal paths. */
export function stableBrokenPathExcess(source,point,destination) {
  const a=sub(point,source),b=sub(destination,point),ra=length(a),rb=length(b),r=distance(source,destination);
  if(!(ra>0&&rb>0))return 0;
  const du=sub(scale(a,1/ra),scale(b,1/rb));
  return (ra/(ra+rb+r))*rb*dot(du,du);
}

/** Replay exactly the local operator's CPU-double arithmetic. GPU receives
 * these same geometric records and applies the shared hardened UTD kernel.
 */
export function evaluateOperator(operator) {
  let soft=[operator.directAmplitude,0],hard=[operator.directAmplitude,0];
  for(const record of operator.edges) {
    const {k,n,phiI,phiO,ro,sinBeta,phaseExcess}=record;
    const L=ro*sinBeta*sinBeta,phase=cscale(cis(-(k*phaseExcess)%TAU),1/Math.sqrt(ro));
    soft=cadd(soft,cmul(coefficient(k,n,phiI,phiO,L,sinBeta,'soft',operator.boundaryEpsilon),phase));
    hard=cadd(hard,cmul(coefficient(k,n,phiI,phiO,L,sinBeta,'hard',operator.boundaryEpsilon),phase));
  }
  const F=operator.boundary==='soft'?abs2(soft):operator.boundary==='hard'?abs2(hard):(abs2(soft)+abs2(hard))/2;
  return {F,f:F,soft,hard};
}

/** Construct geometric UTD records without evaluating any diffraction coefficients.
 * Native angular proposals do not depend on F; this tape can be evaluated on
 * either processor after the complete path has been constructed. */
export function buildFSDOperator(scene,aperture,source,destination,previousCone) {
  const src=vector(source,'source'),dst=vector(destination,'destination');
  const warnings=new Set(),excluded=[],records=[];
  const contains=previousCone?coneContains(previousCone,dst):false;
  if(!previousCone)warnings.add('missing-previous-cone-disables-direct-term');
  let directAmplitude=contains&&segmentVisible(scene,src,dst)?1:0;
  let directBoundary=false;
  for(const edge of aperture.edges) {
    const edgeId=edge.edgeId??edge.id;
    const hit=diffractionPoint(edge,src,dst);
    if(!hit){excluded.push({edgeId,reason:'no-finite-edge-stationary-point'});continue;}
    if(hit.atEndpoint){excluded.push({edgeId,reason:'endpoint-stationary-point-not-modeled'});warnings.add('endpoint-corner-diffraction-not-modeled');continue;}
    if(!outside(edge,hit.wi)||!outside(edge,hit.wo))continue;
    // Exclude only the leg endpoints, never an entire incident face.
    if(!segmentVisible(scene,src,hit.point)||!segmentVisible(scene,hit.point,dst))continue;
    const {phiI,phiO}=directions(edge,hit.wi,hit.wo),n=2-edge.alpha/Math.PI;
    if(phiI>n*Math.PI||phiO>n*Math.PI)continue;
    if(hit.sinBeta<FSD_LIMITS.minSinBeta){excluded.push({edgeId,reason:'edge-grazing-not-modeled'});warnings.add('grazing-asymptotic-exclusion');continue;}
    const L=hit.ro*hit.sinBeta*hit.sinBeta;
    if(!(L>0&&Number.isFinite(aperture.k*L))){excluded.push({edgeId,reason:'nonfinite-coefficient-scale'});warnings.add('nonfinite-coefficient-scale');continue;}
    if(aperture.k*L<10)warnings.add('low-frequency-asymptotic-approximation');
    if(Math.min(phiI,phiO,n*Math.PI-phiI,n*Math.PI-phiO)<FSD_LIMITS.minSinBeta)warnings.add('surface-grazing-asymptotic-approximation');
    const width=Math.sqrt(hit.ro/aperture.k)/hit.sinBeta;
    const physicalMargin=Math.min(distance(hit.point,edge.originalA??edge.a),distance(hit.point,edge.originalB??edge.b));
    const clippedMargin=Math.min(distance(hit.point,edge.a),distance(hit.point,edge.b));
    if(physicalMargin<2*width)warnings.add('finite-endpoint-neighborhood-not-modeled');
    if((edge.clippedStart||edge.clippedEnd)&&clippedMargin<2*width)warnings.add('region-truncated-stationary-neighborhood');
    const angularSeam=Math.abs(Math.abs(phiO-phiI)-Math.PI)<=aperture.boundaryEpsilon;
    // On the direct shadow seam the two visible legs form the straight path.
    // Endpoint-only leg queries distinguish this edge contact from blockers.
    if(contains&&angularSeam)directBoundary=true;
    if(Math.abs(phiO+phiI-Math.PI)<=aperture.boundaryEpsilon||Math.abs(phiO+phiI-(2*n-1)*Math.PI)<=aperture.boundaryEpsilon)warnings.add('reflected-shadow-seam-local-approximation');
    const phaseExcess=stableBrokenPathExcess(src,hit.point,dst);
    if(!Number.isFinite(aperture.k*phaseExcess)){excluded.push({edgeId,reason:'nonfinite-phase-scale'});warnings.add('nonfinite-phase-scale');continue;}
    records.push({k:aperture.k,n,phiI,phiO,ro:hit.ro,sinBeta:hit.sinBeta,phaseExcess,edgeId,
      point:hit.point,ri:hit.ri,L,fermatResidual:hit.fermatResidual,endpointFresnelWidth:width});
  }
  if(directBoundary)directAmplitude=.5;
  const operator={k:aperture.k,boundary:aperture.boundary,boundaryEpsilon:aperture.boundaryEpsilon,directAmplitude,edges:records};
  return {operator,records,directWeight:directAmplitude,edgeCount:records.length,
    diagnostics:{warnings:[...warnings],excluded,directBoundary,
      endpointExclusions:excluded.filter(e=>e.reason==='no-finite-edge-stationary-point'||e.reason==='endpoint-stationary-point-not-modeled').length,
      invalidEdges:excluded.filter(e=>e.reason!=='no-finite-edge-stationary-point'&&e.reason!=='endpoint-stationary-point-not-modeled').length,
      fullWaveValidated:false,finiteSourceCurvature:'native-local-plane-approximation',
      coherence:'one-interaction-region-only',samplingMeasure:'Keller-angle-with-auxiliary-edge-position'}};
}

/** Backward-compatible strict CPU evaluation. Geometry is exactly the same as
 * the deferred demo tape; only this wrapper evaluates the UTD coefficients. */
export function evaluateFSD(scene,aperture,source,destination,previousCone) {
  const built=buildFSDOperator(scene,aperture,source,destination,previousCone);
  const values=evaluateOperator(built.operator),valid=Number.isFinite(values.F)&&values.soft.every(Number.isFinite)&&values.hard.every(Number.isFinite);
  return {...values,valid,...built,diagnostics:{...built.diagnostics,numericallyFinite:valid}};
}
