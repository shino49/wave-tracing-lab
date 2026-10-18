/** Isolated CPU two-region, centre-connected conditional transfer.
 * Units: mm and radians. Two downward half-planes with parallel straight edges
 * are represented by bounded triangle proxies. A rigid transform of this 2-D
 * fixture is supported; arbitrary 3-D geometry/skew edges are not.
 *
 * A transported pure Gaussian is a UNIT-NORMALIZED GEOMETRIC PROPOSAL. The
 * scattering fit is not the covariance of an exact scattered wave. Conditional
 * weights are dimensionless, not watts/radiance or a calibrated detector result.
 * One edge contribution is retained in each local shadow; the two local powers
 * multiply. No global complex DD phase or cross-path coherence is supplied.
 */
import {prepareMesh,transformPoint,add,sub,scale,dot,length,normalize,
  gatherConeTriangles,rayTriangleIntersect} from './mesh-geometry.js';
import {firstConeTriangleContact,clipEdgeToRegion,UTD_LIMITS} from './utd-scene.js';
import {coefficient,abs2,stableCotTransition} from './utd-math.js';
import {createGaussianPacket,propagatePacket,packetPurityResidual,
  packetCurvature,packetParaxialDiagnostics} from './wave-packet.js';

const PI=Math.PI,TAU=2*PI;
export const DD_SCOPE='Experimental native-style centre-connected local-plane conditional power and normalized geometric proposal chain; no calibrated radiance, exact scattering covariance, or global coherent DD field';
export const DD_LIMITS=Object.freeze({maxTriangles:UTD_LIMITS.maxTriangles,
  minKr:10,maxCoefficientKr:100000,minTransitionArgument:10,minFaceGrazingAngle:1e-3,
  endpointWidthMultiples:2,maxBallisticSegments:16,maxBallisticWavelengths:65536});
const KEYS=new Set(['source','E1','E2','receiver','k','sigma','aim1','aim2','boundary',
  'inputWeight','rotationDeg','translation','proxyBottom','proxyHalfLength','proxySubdivisions']);
const vector=(v,name)=>{
  if(!Array.isArray(v)||v.length!==3||!v.every(Number.isFinite))throw new RangeError(`${name} requires three finite coordinates`);
  return [...v];
};
const number=(x,name,min,max)=>{
  if(!Number.isFinite(x)||x<min||x>max)throw new RangeError(`${name} must be finite in [${min}, ${max}]`);
  return x;
};
function settings(input){
  if(!input||Array.isArray(input)||typeof input!=='object')throw new RangeError('config must be an object');
  for(const key of Object.keys(input))if(!KEYS.has(key))throw new RangeError(`Unsupported config key: ${key}`);
  const k=number(input.k??256,'k',1e-6,1e6),q=Math.sqrt(256/k);
  const c={k,wavelengthMm:TAU/k,boundary:input.boundary??'soft',
    sigma:number(input.sigma??.07*q,'sigma',1e-6,1e4),
    aim1:number(input.aim1??.1*q,'aim1',-1e4,1e4),aim2:number(input.aim2??.175*q,'aim2',-1e4,1e4),
    inputWeight:number(input.inputWeight??1,'inputWeight',0,1e100),
    proxyBottom:number(input.proxyBottom??-20,'proxyBottom',-1e5,1e5),
    proxyHalfLength:number(input.proxyHalfLength??20,'proxyHalfLength',1e-5,1e5),
    proxySubdivisions:number(input.proxySubdivisions??1,'proxySubdivisions',1,1e4),
    rotationDeg:vector(input.rotationDeg??[0,0,0],'rotationDeg'),
    translation:vector(input.translation??[0,0,0],'translation')};
  if(!['soft','hard'].includes(c.boundary))throw new RangeError('boundary must be soft or hard');
  if(!Number.isInteger(c.proxySubdivisions))throw new RangeError('proxySubdivisions must be an integer');
  for(const [key,fallback] of Object.entries({source:[-2,0,0],E1:[0,1,0],E2:[8,2,0],receiver:[16,-2,0]})){
    c[key]=vector(input[key]??fallback,key);
    if(c[key][2]!==0)throw new RangeError('Only a canonical z=0 fixture plus a common rigid transform is supported');
    if(c[key].some(x=>Math.abs(x)>1e5))throw new RangeError('Canonical coordinates exceed the numeric range');
  }
  if(!(c.source[0]<c.E1[0]&&c.E1[0]<c.E2[0]&&c.E2[0]<c.receiver[0]&&c.E1[1]<c.E2[1]))
    throw new RangeError('Require ordered source, screen 1, screen 2, receiver in x, with screen 2 tip above screen 1');
  if(c.proxyBottom>=Math.min(c.E1[1],c.E2[1]))throw new RangeError('Proxy bottom must lie below both tips');
  // Validate the common rigid transform through the existing geometry library.
  transformPoint([0,0,0],c);
  return c;
}
function makeFixture(c){
  const point=p=>transformPoint(p,c),zero=point([0,0,0]);
  const axes=[[1,0,0],[0,1,0],[0,0,1]].map(p=>normalize(sub(point(p),zero)));
  const vertices=[],faces=[],screenByTriangle=[];
  for(const [screen,edge] of [c.E1,c.E2].entries())for(let j=0;j<c.proxySubdivisions;j++){
    const za=c.proxyHalfLength*(2*j/c.proxySubdivisions-1),zb=c.proxyHalfLength*(2*(j+1)/c.proxySubdivisions-1),n=vertices.length;
    vertices.push(...[[edge[0],c.proxyBottom,za],[edge[0],edge[1],za],[edge[0],edge[1],zb],[edge[0],c.proxyBottom,zb]].map(point));
    faces.push([n,n+1,n+2],[n,n+2,n+3]);screenByTriangle.push(screen,screen);
  }
  return {mesh:prepareMesh({vertices,faces}),point,axes,zero,screenByTriangle,
    source:point(c.source),E1:point(c.E1),E2:point(c.E2),receiver:point(c.receiver)};
}
function supportBounds(cone,near,far,axes=[[1,0,0],[0,1,0],[0,0,1]],origin=[0,0,0]){
  const min=[Infinity,Infinity,Infinity],max=[-Infinity,-Infinity,-Infinity];
  for(const t of [near,far]){
    const center=sub(add(cone.origin,scale(cone.direction,t)),origin),r=cone.radius0+cone.tanAngle*t;
    axes.forEach((axis,j)=>{
      const h=r*Math.sqrt(Math.max(0,1-dot(axis,cone.direction)**2)),p=dot(center,axis);
      min[j]=Math.min(min[j],p-h);max[j]=Math.max(max[j],p+h);
    });
  }
  return {min,max};
}
function frustumCoordinates(region,point){
  const p=sub(point,region.cone.origin),t=dot(p,region.direction),r=region.cone.radius0+region.cone.tanAngle*t;
  const radial=length(sub(p,scale(region.direction,t)));
  return {axial:t,radial,radius:r,inside:t>=region.near-1e-9&&t<=region.far+1e-9&&radial<=r+1e-9};
}
function ellipsoidValue(region,point){
  const p=sub(point,region.center),z=dot(p,region.direction);
  return (dot(p,p)-.75*z*z)/(region.radius**2);
}
function discover(mesh,cone){
  const contacts=gatherConeTriangles(mesh,cone).map(t=>firstConeTriangleContact(t,cone)).filter(Boolean).sort((a,b)=>a.distance-b.distance);
  if(!contacts.length)return null;
  const first=contacts[0],near=first.distance,radius=cone.radius0+cone.tanAngle*near,far=near+2*radius;
  const frustum={...cone,near,far};
  const triangleIds=gatherConeTriangles(mesh,frustum).filter(t=>firstConeTriangleContact(t,frustum)).map(t=>t.id);
  return {cone:{...cone},direction:[...cone.direction],center:add(cone.origin,scale(cone.direction,near)),
    near,far,depth:2*radius,radius,axes:[radius,radius,2*radius],firstContact:first,
    triangleIds,bounds:supportBounds(cone,near,far)};
}
function nearestRayHit(mesh,origin,direction,minDistance,maxDistance){
  let hit=null;
  for(const triangle of mesh.triangles){
    const h=rayTriangleIntersect(origin,direction,triangle,{minDistance,maxDistance});
    if(h&&(!hit||h.distance<hit.distance))hit={...h,triangleId:triangle.id};
  }
  return hit;
}
function traverse(mesh,cone,wavelengthMm){
  let dist=0;const steps=[];
  // The centred proposal ray starts inside its envelope: native additional
  // self-intersection/min-ballistic-distance term is zero. No old faces skipped.
  for(let j=0;j<=DD_LIMITS.maxBallisticSegments;j++){
    const wavelengths=j===DD_LIMITS.maxBallisticSegments?Infinity:Math.min(16*4**j,DD_LIMITS.maxBallisticWavelengths);
    const B=wavelengths*wavelengthMm,hi=dist+B*1.001;
    const hit=nearestRayHit(mesh,cone.origin,cone.direction,Math.max(1e-9,dist),hi);
    const step={index:j,from:dist,to:dist+B,ballisticLengthWavelengths:wavelengths,
      rayInterval:[Math.max(1e-9,dist),hi],rayTriangleTests:mesh.triangles.length,
      excludedTriangleIds:[],overlapFactor:1.001};
    steps.push(step);
    if(hit){step.decision='opaque-ray-hit';step.hit=hit;return {valid:false,reason:'opaque-ray-hit',steps};}
    dist+=B;
    if(!Number.isFinite(dist)){step.decision='exhausted';return {valid:false,reason:'no-region',steps};}
    const region=discover(mesh,{...cone,near:dist});
    step.contactDistance=region?.near??null;
    step.minimumProgress=(cone.radius0+cone.tanAngle*dist)/2;
    if(!region){step.decision='no-cone-contact';return {valid:false,reason:'no-region',steps};}
    step.diffusiveProgress=region.near-dist;
    if(step.diffusiveProgress>=step.minimumProgress){step.decision='region';return {valid:true,region,steps};}
    step.decision='short-progress-continue-ballistic';
  }
  throw new Error('Unreachable traversal state');
}
function fitOutgoing(region,direction){
  const tau=region.cone.tanAngle,c=dot(region.direction,direction),h=Math.sqrt(Math.max(0,1-c*c));
  const caps=[region.near,region.far].map(t=>{
    const s=t-region.near,r=region.cone.radius0+tau*t;
    return {s,r,bound:r+s*h-tau*s*c+tau*r*h};
  });
  const radius0=Math.max(...caps.map(x=>x.bound));
  return {origin:[...region.center],direction:[...direction],radius0,tanAngle:tau,near:0,far:Infinity,
    fit:{caps,cosTurn:c,sinTurn:h,contains:'full prior frustum in radial linear envelope; prior points are not clipped by the new near plane'}};
}
function proposalState(packet,label,weight){
  return {...packet,label,coherentPhaseDefined:packet.coherentPhaseDefined!==false,
    normalization:'unit integral over the perpendicular plane; geometric proposal only',
    physicalDistanceMm:packet.opticalPathMm,conditionalWeight:weight,
    proposalVirtualAgeMm:packet.crossCovariance[0]/packet.angularCovariance[0],
    curvature:packetCurvature(packet),purityResidual:packetPurityResidual(packet),
    paraxialDiagnostics:packetParaxialDiagnostics(packet)};
}
function freeProposal(packet,s){
  if(packet.coherentPhaseDefined!==false)return propagatePacket(packet,s);
  // Only the covariance shear and physical distance are retained. An arbitrary
  // internal arithmetic gauge must never become a claimed scattered phase.
  return {...propagatePacket({...packet,phaseRad:0,gouyPhaseRad:0},s),phaseRad:null,gouyPhaseRad:null,coherentPhaseDefined:false};
}
function scatteredProposal(old,cone,edgeAxis){
  const X=(cone.radius0/3)**2,T=(cone.tanAngle/3)**2,radicand=X*T-1/(4*old.k*old.k);
  if(!(Number.isFinite(radicand)&&radicand>=-1e-14*X*T))throw new RangeError('Fitted proposal violates pure-state uncertainty');
  const C=Math.sqrt(Math.max(0,radicand)),B=C/X;
  const packet=createGaussianPacket({position:cone.origin,direction:cone.direction,u:edgeAxis,
    wavelengthMm:old.wavelengthMm,positionCovariance:[X,0,0,X],curvature:[B,0,0,B],powerW:1});
  return {...packet,opticalPathMm:old.opticalPathMm,phaseRad:null,gouyPhaseRad:null,
    coherentPhaseDefined:false,role:'geometry-driven outgoing proposal, not exact scattered covariance'};
}
function coneFromProposal(packet,fit){
  const radius0=3*Math.sqrt(packet.positionCovariance[0]),tanAngle=3*Math.sqrt(packet.angularCovariance[0]);
  if(Math.abs(radius0-fit.radius0)>1e-11*Math.max(1,radius0)||Math.abs(tanAngle-fit.tanAngle)>1e-11*Math.max(1,tanAngle))
    throw new RangeError('Proposal covariance and fitted discovery envelope disagree');
  // These covariance-derived values, rather than a second independent cone,
  // are the actual input to the next native-style ballistic/diffusive query.
  return {...fit,origin:[...packet.position],direction:[...packet.direction],radius0,tanAngle,
    construction:'derived from transported proposal X and T after conservative full-frustum fit'};
}
function localDiagnostics(k,phi_i,phi_o,ro,boundary){
  const transitionTerms=[];
  for(const [family,beta] of [['difference',phi_o-phi_i],['sum',phi_o+phi_i]])for(const sign of [1,-1]){
    const nearest=Math.floor((beta+sign*PI)/(4*PI)+.5),delta=beta-(4*PI*nearest-sign*PI);
    const term=stableCotTransition(beta,2,k*ro,sign);
    transitionTerms.push({family,sign,delta,argument:2*k*ro*Math.sin(delta/2)**2,termMagnitude:Math.hypot(...term)});
  }
  const D=coefficient(k,2,phi_i,phi_o,ro,1,boundary),sum=transitionTerms.reduce((s,t)=>s+t.termMagnitude,0);
  // The common coefficient prefactor cancels out of this conditioning ratio.
  const unscaled=Math.hypot(...D)*4*Math.sqrt(2*PI*k);
  return {transitionTerms,minTransitionArgument:Math.min(...transitionTerms.map(t=>t.argument)),
    faceGrazingAngle:Math.min(phi_i,TAU-phi_i,phi_o,TAU-phi_o),sinBeta:1,kr:k*ro,
    angularCancellationRatio:sum===0?0:unscaled/sum,localPlaneDistance:ro,
    localCoefficient:D,localPower:abs2(D)/ro,localCoherentContributionCount:1,
    directContributionCount:0,finiteSourceReducedDistanceUsed:false};
}

/** Returns a total, explicitly masked result for unsupported configurations.
 * conditionalPower is f1*f2; conditionalWeight includes inputWeight. An invalid
 * result has null power/weight, never a valid zero. coefficientRecord.valid is
 * suitable for the isolated DD GPU operator (invalid GPU lanes are zero-masked).
 */
export function traceDoubleDiffraction(config={}){
  const result={valid:false,admissionReasons:[],checks:{},geometryTrace:[],packetStates:[],
    coefficientRecord:{angles:[0,0,0,0],ro1:0,ro2:0,ri2:0,k:0,boundary:'soft',valid:false},
    conditionalPower:null,conditionalWeight:null,scope:DD_SCOPE,
    admissionPolicy:'native-centre-conditional',physicallyCalibrated:false,fullPacketScatteringValidated:false,
    terminalConnection:'centre-connected-next-event',outgoingBundleClearanceProven:false,
    limitations:['Local-plane native centre approximation; finite proposal curvature is not used to change UTD L.',
      'No full-wave accuracy certification, endpoint diffraction, arbitrary 3-D/skew edges, or normalized scattering-mode decomposition.',
      'The pre-existing single-region plane-curvature admission test is inapplicable to this explicitly approximated native recursion.',
      'The terminal connection checks its centre ray and selected edge leg; it is not an organic cone traversal or proof of full outgoing-bundle clearance.',
      '3-sigma geometric clipping and the existing 2 sqrt(ro/k) endpoint rule are heuristic admission guards, not an error bound.']};
  const reject=(code,message)=>result.admissionReasons.push({code,message});
  const check=(name,passed,message)=>{result.checks[name]=!!passed;if(!passed)reject(name,message);};
  try{
    const c=settings(config);result.parameters=c;
    result.coefficientRecord.k=c.k;result.coefficientRecord.boundary=c.boundary;
    if(4*c.proxySubdivisions>DD_LIMITS.maxTriangles){reject('triangle-safety-cap',`CPU/GPU cap is ${DD_LIMITS.maxTriangles} triangles`);return result;}
    const f=makeFixture(c),{mesh,source,E1,E2,receiver}=f;
    result.mesh={triangleCount:mesh.triangles.length,screenByTriangle:f.screenByTriangle,
      representation:'finite proxies for two downward infinite screens; artificial boundaries must be outside both regions'};
    const aim1=f.point(add(c.E1,[0,c.aim1,0])),direction=normalize(sub(aim1,source));
    const packet=createGaussianPacket({position:source,direction,u:f.axes[2],wavelengthMm:c.wavelengthMm,
      positionCovariance:[c.sigma**2,0,0,c.sigma**2],powerW:1});
    const cone0={origin:source,direction,radius0:3*Math.sqrt(packet.positionCovariance[0]),
      tanAngle:3*Math.sqrt(packet.angularCovariance[0]),near:0,far:Infinity};
    result.packetStates.push(proposalState(packet,'source',c.inputWeight));
    const t1=traverse(mesh,cone0,c.wavelengthMm);result.geometryTrace.push({stage:1,...t1});
    if(!t1.valid){reject('initial-traversal',t1.reason);return result;}
    const first=t1.region;
    const p1=freeProposal(packet,first.near);
    result.packetStates.push(proposalState(p1,'region1-incoming',c.inputWeight));
    const fit1=fitOutgoing(first,normalize(sub(f.point(add(c.E2,[0,c.aim2,0])),first.center)));
    const out1=scatteredProposal(p1,fit1,f.axes[2]),cone1=coneFromProposal(out1,fit1);
    result.packetStates.push(proposalState(out1,'region1-outgoing',null));
    const t2=traverse(mesh,cone1,c.wavelengthMm);result.geometryTrace.push({stage:2,...t2});
    if(!t2.valid){reject('second-traversal',t2.reason);return result;}
    const second=t2.region,p2=freeProposal(out1,second.near);
    result.packetStates.push(proposalState(p2,'region2-incoming',null));
    const fit2=fitOutgoing(second,normalize(sub(receiver,second.center))),out2=scatteredProposal(p2,fit2,f.axes[2]),cone2=coneFromProposal(out2,fit2);
    const receiverDistance=length(sub(receiver,second.center)),terminal=freeProposal(out2,receiverDistance);
    result.packetStates.push(proposalState(out2,'region2-outgoing',null),proposalState(terminal,'receiver',null));
    result.cones=[cone0,cone1,cone2];result.regions=[first,second];
    result.originalCollimationGuard={applicable:false,thresholdRad:.25,
      interpretation:'Diagnostic only: the old single-region plane-curvature gate is superseded by the explicitly approximated native-centre-conditional policy',
      stages:[first,second].map((region,j)=>{
        const proposal=[p1,p2][j],B=packetCurvature(proposal);
        const scaleRad=.5*c.k*region.radius**2*Math.max(Math.abs(B[0]),Math.abs(B[3]));
        return j===0?{kind:'source-Gaussian phase-curvature diagnostic',knownPhysicalPhase:true,
          phaseCurvatureRad:scaleRad,passed:scaleRad<=.25}:
          {kind:'fitted-proposal quadratic phase scale',knownPhysicalPhase:false,
            phaseCurvatureRad:null,proposalQuadraticPhaseScaleRad:scaleRad,passed:null,
            exceedsDiagnosticThreshold:scaleRad>.25,
            interpretation:'Proposal curvature only; coherent scattering phase is undefined, so this is not a known physical scattered-wavefront phase error'};
      })};
    result.originalCollimationGuard.passed=result.originalCollimationGuard.stages[0].passed;
    check('aabb-disjoint',first.bounds.max.some((v,j)=>v<second.bounds.min[j]||second.bounds.max[j]<first.bounds.min[j]),'The actual frustum AABBs overlap');
    for(const [j,region] of [first,second].entries()){
      check(`region${j+1}-intended-screen`,region.triangleIds.length>0&&region.triangleIds.every(id=>f.screenByTriangle[id]===j),'Region contains the wrong screen or no triangles');
      const edge=[E1,E2][j],coordinates=frustumCoordinates(region,edge),ellipsoid=ellipsoidValue(region,edge);
      region.stationaryPoint={point:edge,frustum:coordinates,ellipsoidValue:ellipsoid,fermatResidual:0};
      check(`edge${j+1}-in-frustum`,coordinates.inside,'Stationary edge point lies outside actual frustum');
      check(`edge${j+1}-in-ellipsoid`,ellipsoid<=1+1e-9,'Stationary edge point lies outside the native footprint ellipsoid');
      for(const [name,p] of [['source',source],['receiver',receiver]])
        check(`${name}-outside-region${j+1}`,!frustumCoordinates(region,p).inside&&ellipsoidValue(region,p)>1+1e-9,`${name} lies in an interaction region`);
      const local=supportBounds(region.cone,region.near,region.far,f.axes,f.zero);
      region.proxyBoundaryMargins={bottom:local.min[1]-c.proxyBottom,negativeZ:local.min[2]+c.proxyHalfLength,positiveZ:c.proxyHalfLength-local.max[2]};
      check(`region${j+1}-proxy-boundaries`,Object.values(region.proxyBoundaryMargins).every(x=>x>1e-8),'Artificial rectangle boundaries are not demonstrably outside the whole frustum');
    }
    const segment=(a,b)=>{const v=sub(b,a),s=length(v);return nearestRayHit(mesh,a,scale(v,1/s),1e-8,s-1e-8);};
    result.visibility={};
    for(const [name,a,b] of [['source-to-centre1',source,first.center],['centre1-to-centre2',first.center,second.center],['centre2-to-receiver',second.center,receiver],
      ['source-to-edge1',source,E1],['edge1-to-centre2',E1,second.center],['centre1-to-edge2',first.center,E2],['edge2-to-receiver',E2,receiver]]){
      const hit=segment(a,b);result.visibility[name]={clear:!hit,hit};check(`${name}-clear`,!hit,'An opaque triangle blocks the selected centre/edge connection');
    }
    for(const [name,a,b] of [['local1-shadow',source,second.center],['local2-shadow',first.center,receiver]]){
      const hit=segment(a,b);result.visibility[name]={blocked:!!hit,hit};check(name,!!hit,'The local direct contribution is visible, outside this one-contribution shadow model');
    }
    const angle=v=>{const a=Math.atan2(dot(v,f.axes[0]),-dot(v,f.axes[1]));return a<0?a+TAU:a;};
    const angles=[angle(sub(source,E1)),angle(sub(second.center,E1)),angle(sub(first.center,E2)),angle(sub(receiver,E2))];
    const ro1=length(sub(second.center,E1)),ri2=length(sub(E2,first.center)),ro2=length(sub(receiver,E2));
    result.coefficientRecord={angles,ro1,ro2,ri2,k:c.k,boundary:c.boundary,valid:false};
    check('coefficient-numeric-domain',[ro1,ro2].every(ro=>ro>=1e-6&&ro<=1e6&&c.k*ro<=DD_LIMITS.maxCoefficientKr),
      'Coefficient record exceeds the bounded isolated CPU/GPU numeric domain');
    result.localStages=[];
    for(const [j,region] of [first,second].entries()){
      const ro=[ro1,ro2][j],edge=[c.E1,c.E2][j];
      const clipped=clipEdgeToRegion({a:f.point([edge[0],edge[1],-c.proxyHalfLength]),b:f.point([edge[0],edge[1],c.proxyHalfLength]),l:2*c.proxyHalfLength},region);
      const halfLength=clipped?Math.min(length(sub(clipped.a,[E1,E2][j])),length(sub(clipped.b,[E1,E2][j]))):0;
      const width=Math.sqrt(ro/c.k),stage=localDiagnostics(c.k,angles[2*j],angles[2*j+1],ro,c.boundary);
      Object.assign(stage,{clippedEdge:clipped,clippedHalfLength:halfLength,stationaryWidth:width,
        stationaryWidthDefinition:'sqrt(ro/(k sinBeta^2)); sinBeta=1, NOT sqrt(lambda*ro)',
        requiredHalfLength:DD_LIMITS.endpointWidthMultiples*width,endpointGuardRatio:halfLength/(DD_LIMITS.endpointWidthMultiples*width)});
      result.localStages.push(stage);
      check(`region${j+1}-endpoint-guard`,halfLength>=DD_LIMITS.endpointWidthMultiples*width,'Ellipsoid-clipped edge has less than two existing stationary widths');
      check(`region${j+1}-high-frequency`,stage.kr>=DD_LIMITS.minKr,'Local k ro is below the supported high-frequency threshold');
      check(`region${j+1}-away-from-transition`,stage.minTransitionArgument>=DD_LIMITS.minTransitionArgument,'A local transition argument is below the declared separated-transition threshold');
      check(`region${j+1}-away-from-grazing`,stage.faceGrazingAngle>=DD_LIMITS.minFaceGrazingAngle,'Surface grazing is unsupported');
    }
    const factors=result.localStages.map(s=>s.localPower),power=factors[0]*factors[1],weight=c.inputWeight*power;
    check('finite-conditional-power',[...factors,power,weight].every(x=>Number.isFinite(x)&&x>=0),'Conditional transfer is outside finite range');
    check('proposal-invariants',result.packetStates.every(p=>p.purityResidual<1e-8&&p.powerW===1&&p.positionCovariance.every(Number.isFinite)),'Proposal lost pure-state normalization or finite covariance');
    result.physicalPath={centreAxisDistanceMm:terminal.opticalPathMm,
      edgeBrokenPathDistanceMm:length(sub(E1,source))+length(sub(E2,E1))+length(sub(receiver,E2)),
      coherentPhaseDefined:false,interpretation:'Geometric travel history only; covariance virtual age is not physical travel or scattered phase'};
    result.valid=result.admissionReasons.length===0;result.coefficientRecord.valid=result.valid;
    if(result.valid){
      result.conditionalPower=power;result.conditionalWeight=weight;
      result.packetStates[2].conditionalWeight=result.packetStates[3].conditionalWeight=c.inputWeight*factors[0];
      result.packetStates[4].conditionalWeight=result.packetStates[5].conditionalWeight=weight;
    }
    return result;
  }catch(error){reject('unsupported-config-or-numerics',error.message);return result;}
}
